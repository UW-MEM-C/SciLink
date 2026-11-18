"""LLM-proposed fix for a failed VASP job.

The rule-based cascade in :mod:`scilink.hpc.diagnosis` names the failure;
this module asks SciLink's post-run critic (``RunCritic``, the live
runtime-fix agent) to interpret it and propose corrected input files. The
result is only a *proposal*: it feeds :func:`scilink.hpc.retry.build_retry_proposal`
and is resubmitted solely after the user's explicit approval.

``scilink.hpc`` stays free of agent imports: the critic is injected and only
needs an ``assess(**kwargs) -> dict`` method with ``RunCritic.assess``'s
signature. Remote reads are bounded ``head`` / ``tail`` calls — no file is
downloaded whole. The log excerpts are staged locally under the job's
diagnostics folder with the names the VASP skill's ``snapshot_run`` parser
looks for, so the critic sees them as a normal (partial) run directory.
"""
from __future__ import annotations

import json
import posixpath
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .diagnosis import JobDiagnosis, diagnosis_dir
from .retry import REPLACEABLE_FILES

_MISSING = "__SCILINK_NO_FILE__"

# Bounded reads: (remote name, local staged name, lines). Outputs are tails.
_OUTPUT_TAILS: Tuple[Tuple[str, str, int], ...] = (
    ("vasp.out", "vasp.out", 200),
    ("OUTCAR", "OUTCAR", 200),
    ("OSZICAR", "OSZICAR", 60),
)
_LOG_TAIL = 120
# Inputs are heads. A POSCAR longer than this is not shown whole, so the
# critic may not replace it (it would come back truncated).
_INPUT_HEADS: Tuple[Tuple[str, int], ...] = (("INCAR", 300), ("KPOINTS", 60),
                                             ("POSCAR", 2000))


class CriticLike(Protocol):
    def assess(self, **kwargs: Any) -> Dict[str, Any]: ...


@dataclass
class AIFix:
    job_id: str
    ok: bool = False
    error: str = ""
    reasoning: str = ""
    diagnostic_notes: str = ""
    failure_class: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)
    # Whole corrected files, filtered to REPLACEABLE_FILES and to real changes.
    suggested_files: Dict[str, str] = field(default_factory=dict)
    # Current contents of the inputs shown to the critic (for diffs).
    current_files: Dict[str, str] = field(default_factory=dict)
    dropped: List[str] = field(default_factory=list)
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AIFix":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# ── remote reads ──────────────────────────────────────────────

def _q(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _read(conn, path: str, verb: str, n: int) -> Optional[str]:
    p = _q(path)
    try:
        out, _, rc = conn.run(
            f"if [ -f {p} ]; then {verb} -n {int(n)} {p}; else echo {_MISSING}; fi",
            timeout=60)
    except Exception:  # noqa: BLE001 — a dead link means "not available"
        return None
    if out.strip() == _MISSING or (rc != 0 and not out):
        return None
    return out


def collect_context(
    conn, diag: JobDiagnosis, *, stdout_file: str = "", stderr_file: str = "",
) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    """Return ``(outputs, inputs, truncated)``: log tails keyed by the local
    name to stage them under, input heads keyed by filename, and the inputs
    that were longer than the read limit."""
    wd = diag.work_dir.rstrip("/")

    def path(name: str) -> str:
        return name if name.startswith("/") else posixpath.join(wd, name)

    outputs: Dict[str, str] = {}
    for remote, local, n in _OUTPUT_TAILS:
        text = _read(conn, path(remote), "tail", n)
        if text and text.strip():
            outputs[local] = text
    for remote, local in ((stdout_file, "stdout"), (stderr_file, "stderr")):
        if remote:
            text = _read(conn, path(remote), "tail", _LOG_TAIL)
            if text and text.strip():
                outputs[local] = text

    inputs: Dict[str, str] = {}
    truncated: List[str] = []
    for name, n in _INPUT_HEADS:
        text = _read(conn, path(name), "head", n + 1)
        if text is None:
            continue
        lines = text.splitlines()
        if len(lines) > n:
            truncated.append(name)
            text = "\n".join(lines[:n]) + "\n"
        inputs[name] = text
    return outputs, inputs, truncated


# ── the request ───────────────────────────────────────────────

def _goal_text(diag: JobDiagnosis, research_goal: str) -> str:
    evidence = "\n".join(
        f"[{e.file}]\n" + "\n".join(e.lines) for e in diag.evidence[:3])
    return (
        f"{research_goal.strip() or 'Run this VASP calculation successfully.'}\n\n"
        "This job FAILED on the cluster. SciLink's log scan reported: "
        f"{diag.summary}\n"
        f"Evidence:\n{evidence}\n\n"
        "Explain the root cause in plain language and return the corrected "
        "input files (complete files, same filenames) that fix it while keeping "
        "the calculation's intent. Only INCAR, KPOINTS and POSCAR can be changed."
    )


def _normalise(text: str) -> str:
    return "\n".join(ln.rstrip() for ln in text.replace("\r\n", "\n").strip().splitlines())


def request_ai_fix(
    critic: CriticLike,
    conn,
    diag: JobDiagnosis,
    *,
    session_dir: str | Path,
    research_goal: str = "",
    stdout_file: str = "",
    stderr_file: str = "",
    model: str = "",
) -> AIFix:
    """Ask the critic for a fix and save it as ``ai_fix.json`` next to the
    diagnosis. Never raises for LLM trouble: the error is in ``AIFix.error``."""
    fix = AIFix(job_id=diag.job_id, model=model)
    outputs, inputs, truncated = collect_context(
        conn, diag, stdout_file=stdout_file, stderr_file=stderr_file)
    fix.current_files = dict(inputs)
    if not inputs:
        fix.error = "Could not read the job's input files on the cluster."
        return _save(fix, session_dir)

    staging = diagnosis_dir(session_dir, diag.job_id) / "ai_snapshot"
    staging.mkdir(parents=True, exist_ok=True)
    for old in staging.iterdir():
        if old.is_file():
            old.unlink()
    for name, text in outputs.items():
        (staging / name).write_text(text, encoding="utf-8", newline="\n")

    try:
        report = critic.assess(
            output_dir=str(staging),
            research_goal=_goal_text(diag, research_goal),
            skill="vasp",
            domain="periodic_dft",
            fixes_mode="always",
            input_files=inputs,
        )
    except Exception as exc:  # noqa: BLE001 — surface, don't crash the monitor
        fix.error = f"The AI request failed: {type(exc).__name__}: {exc}"
        return _save(fix, session_dir)

    if not isinstance(report, dict) or report.get("status") == "error":
        err = (report or {}).get("error") if isinstance(report, dict) else None
        fix.error = f"The AI returned no usable answer ({err or 'unknown error'})."
        return _save(fix, session_dir)

    fix.reasoning = str(report.get("reasoning") or "")
    fix.diagnostic_notes = str(report.get("diagnostic_notes") or "")
    fix.failure_class = report.get("failure_class")
    fix.recommendations = [str(r) for r in (report.get("recommendations") or [])]

    for name, text in (report.get("suggested_fixes") or {}).items():
        base = posixpath.basename(str(name)).strip()
        if base not in REPLACEABLE_FILES or not isinstance(text, str):
            fix.dropped.append(f"{name}: only {', '.join(REPLACEABLE_FILES)} can be changed")
            continue
        if base in truncated:
            fix.dropped.append(f"{base}: the file is too long to be shown whole, "
                               "so a rewrite of it is not trusted")
            continue
        if not text.strip():
            fix.dropped.append(f"{base}: empty")
            continue
        if base in inputs and _normalise(text) == _normalise(inputs[base]):
            continue
        fix.suggested_files[base] = text if text.endswith("\n") else text + "\n"
    fix.ok = True
    return _save(fix, session_dir)


def _save(fix: AIFix, session_dir: str | Path) -> AIFix:
    try:
        d = diagnosis_dir(session_dir, fix.job_id)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "ai_fix.json", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(fix.to_dict(), fh, indent=2, ensure_ascii=False)
    except OSError:
        pass
    return fix
