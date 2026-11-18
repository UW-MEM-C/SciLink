"""AI-proposed fix for a failed job (scilink.hpc.ai_fix) and whole-file
replacements in the approval-gated retry (scilink.hpc.retry).

The critic is a scripted stand-in for ``RunCritic`` — the seam is duck-typed,
so no LLM (or openai SDK) is needed.
"""
from __future__ import annotations

import json

import pytest

from scilink.hpc.ai_fix import request_ai_fix
from scilink.hpc.diagnosis import CAT_VASP
from scilink.hpc.retry import build_retry_proposal, prepare_and_submit_retry

from test_job_diagnosis import (
    BZINTS_BOX,
    KPOINTS,
    POSCAR,
    POTCAR_OK,
    SACCT_FAILED,
    WD,
    FakeConn,
    FakeSched,
    _diag,
)

# The live case: ISMEAR=-5 with a Gamma-only mesh. The subroutine name is
# blanked so no rule matches and the job falls to the generic box fallback.
GENERIC_BOX = BZINTS_BOX.replace("subroutine BZINTS: Tetrahedron",
                                 "subroutine XXXXXX: Tetrahedron").replace(
    "method fails", "meth0d fails")
INCAR_TET = "SYSTEM = TiO2\nENCUT = 520\nISMEAR = -5\nNSW = 0\n"
KPOINTS_GAMMA = "Automatic\n0\nGamma\n1 1 1\n0 0 0\n"
KPOINTS_FIXED = "Automatic\n0\nGamma\n4 4 3\n0 0 0\n"
SCRIPT = f"#!/bin/bash\n#SBATCH --time=00:30:00\ncd '{WD}'\nsrun vasp_std\n"


class ScriptedCritic:
    def __init__(self, report=None, exc=None):
        self.report, self.exc, self.calls = report, exc, []

    def assess(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return self.report


def _conn(**extra):
    files = {"submit.sh": SCRIPT, "INCAR": INCAR_TET, "KPOINTS": KPOINTS_GAMMA,
             "POSCAR": POSCAR, "POTCAR": POTCAR_OK, "vasp.out": GENERIC_BOX,
             "slurm-7001.out": "", "slurm-7001.err": ""}
    files.update(extra)
    return FakeConn(files=files, sacct=SACCT_FAILED)


def _generic(conn):
    d = _diag(conn)
    assert (d.category, d.cause) == (CAT_VASP, "vasp_fatal_error")
    return d


def test_ai_fix_keeps_only_real_changes_to_replaceable_files(tmp_path):
    conn = _conn()
    diag = _generic(conn)
    critic = ScriptedCritic({
        "status": "success", "run_status": "failed",
        "reasoning": "ISMEAR=-5 needs at least 4 k-points; the mesh is Gamma only.",
        "suggested_fixes": {"KPOINTS": KPOINTS_FIXED,
                            "INCAR": INCAR_TET,          # unchanged → not a change
                            "POTCAR": "garbage",         # never replaceable
                            "submit.sh": "rm -rf /"},    # never replaceable
    })
    n_cmds = len(conn.commands)
    fix = request_ai_fix(critic, conn, diag, session_dir=tmp_path,
                         research_goal="Relax rutile TiO2", model="claude-test")

    assert fix.ok and not fix.error and "4 k-points" in fix.reasoning
    assert fix.suggested_files == {"KPOINTS": KPOINTS_FIXED}
    assert any("POTCAR" in d for d in fix.dropped) and any("submit.sh" in d for d in fix.dropped)
    # The critic saw the real inputs, the VASP skill and the rule-based finding.
    kw = critic.calls[0]
    assert kw["skill"] == "vasp" and kw["domain"] == "periodic_dft"
    assert kw["input_files"]["KPOINTS"].strip() == KPOINTS_GAMMA.strip()
    assert "Tetrahedron" in kw["research_goal"] and "Relax rutile TiO2" in kw["research_goal"]
    staged = tmp_path / "diagnostics" / "job_7001" / "ai_snapshot"
    assert "Tetrahedron" in (staged / "vasp.out").read_text()
    saved = json.loads((tmp_path / "diagnostics" / "job_7001" / "ai_fix.json").read_text())
    assert saved["suggested_files"] == {"KPOINTS": KPOINTS_FIXED}
    # Bounded remote reads only.
    new = conn.commands[n_cmds:]
    assert new and all(("head -n" in c or "tail -n" in c) and "cat " not in c for c in new)


def test_truncated_poscar_is_never_replaced(tmp_path):
    big = POSCAR + "".join(f"0.0 0.0 {i/3000:.4f}\n" for i in range(2500))
    conn = _conn(POSCAR=big)
    diag = _generic(conn)
    critic = ScriptedCritic({"status": "success", "reasoning": "x",
                             "suggested_fixes": {"POSCAR": "shortened\n"}})
    fix = request_ai_fix(critic, conn, diag, session_dir=tmp_path)
    assert fix.suggested_files == {}
    assert any("POSCAR" in d and "too long" in d for d in fix.dropped)


@pytest.mark.parametrize("critic", [
    ScriptedCritic(exc=RuntimeError("proxy down")),
    ScriptedCritic({"status": "error", "error": "Could not parse LLM response as JSON."}),
])
def test_ai_failure_is_reported_not_raised(tmp_path, critic):
    conn = _conn()
    fix = request_ai_fix(critic, conn, _generic(conn), session_dir=tmp_path)
    assert not fix.ok and fix.error and fix.suggested_files == {}


def test_deterministic_failure_without_a_fix_cannot_be_resubmitted_unchanged():
    conn = _conn()
    p = build_retry_proposal(_generic(conn))
    assert not p.allowed and "AI" in p.blocked_reason
    assert p.describe() == "resubmit unchanged"


def test_ai_fix_retry_uploads_the_new_kpoints_and_leaves_the_failed_run_alone():
    conn = _conn()
    before = dict(conn.files)
    diag = _generic(conn)
    p = build_retry_proposal(diag, file_replacements={"KPOINTS": KPOINTS_FIXED,
                                                      "POTCAR": "no"},
                             use_rule_actions=False, source="ai")
    assert p.allowed and p.file_replacements == {"KPOINTS": KPOINTS_FIXED}
    assert "KPOINTS replaced" in p.describe()
    assert any("POTCAR" in n for n in p.notes)

    sched = FakeSched()
    res = prepare_and_submit_retry(conn, sched, p, approved=True, script_name="submit.sh")
    new = f"{WD}/retry_1"
    assert res["files_replaced"] == ["KPOINTS"] and res["source"] == "ai"
    assert conn.files[f"{new}/KPOINTS"] == KPOINTS_FIXED
    assert conn.files[f"{new}/INCAR"].strip() == INCAR_TET.strip()
    assert conn.files[f"{new}/POSCAR"] == POSCAR and conn.files[f"{new}/POTCAR"] == POTCAR_OK
    assert sched.submitted == [(f"{new}/submit.sh", new)]
    for k, v in before.items():
        assert conn.files[k] == v


def test_replaced_poscar_wins_over_contcar_restart():
    conn = _conn(CONTCAR=POSCAR.replace("TiO2", "CONT"))
    diag = _generic(conn)
    p = build_retry_proposal(diag, file_replacements={"POSCAR": "NEW POSCAR\n"},
                             use_rule_actions=False, source="ai")
    p.restart_from_contcar = True
    res = prepare_and_submit_retry(conn, FakeSched(), p, approved=True, script_name="submit.sh")
    assert conn.files[f"{WD}/retry_1/POSCAR"] == "NEW POSCAR\n"
    assert conn.files[f"{WD}/retry_1/KPOINTS"] == KPOINTS_GAMMA  # copied, not replaced
    assert res["files_replaced"] == ["POSCAR"]
    assert any("CONTCAR" in n for n in p.notes)
