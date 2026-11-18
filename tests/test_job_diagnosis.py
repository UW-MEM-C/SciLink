"""Tests for the post-run failure cascade (`scilink/hpc/diagnosis.py`) and the
approval-gated retry helpers (`scilink/hpc/retry.py`).

A fake connection emulates the remote shell (sacct, ls, tail, head, grep)
over an in-memory directory and logs every command, so each test can assert
both *what* was concluded and *where the cascade stopped* — i.e. that files
belonging to later steps were never read.

Log/output fixtures reproduce the text real tools print (slurmstepd kill
lines, VASP 6 error boxes, OSZICAR DAV/F= lines, sacct -P rows).
"""
from __future__ import annotations

import re
import shlex

import pytest

from scilink.hpc.diagnosis import (
    CAT_CONVERGENCE,
    CAT_INPUTS,
    CAT_RESOURCES,
    CAT_SCRIPT,
    CAT_UNKNOWN,
    CAT_VASP,
    diagnose_job,
    load_diagnosis,
    parse_oszicar,
    parse_slurm_time,
    save_diagnosis,
)
from scilink.hpc.retry import (
    RetryPolicy,
    apply_incar_changes,
    apply_resource_changes,
    build_retry_proposal,
    prepare_and_submit_retry,
    rewrite_workdir,
)

WD = "/gpfs/scrubbed/lperalt/TiO2_rutile"


# ── fake remote shell ─────────────────────────────────────────

class FakeConn:
    """Tiny interpreter for exactly the command shapes the diagnosis and
    retry modules emit. Unknown commands fail loudly."""

    def __init__(self, files=None, sacct="", sacct_rc=0, work_dir=WD):
        self.wd = work_dir
        self.files = {f"{work_dir}/{k}": v for k, v in (files or {}).items()}
        self.sacct = sacct
        self.sacct_rc = sacct_rc
        self.commands: list[str] = []
        self.uploaded: dict[str, str] = {}
        self.dirs = {work_dir}

    # -- helpers used by assertions --
    def read_paths(self) -> list[str]:
        """Basenames of files whose content was requested."""
        out = []
        for c in self.commands:
            m = re.match(r"if \[ -f '([^']+)' \]", c)
            if m:
                out.append(m.group(1).rsplit("/", 1)[-1])
        return out

    # -- interface --
    def run(self, cmd, timeout=300):
        self.commands.append(cmd)
        if cmd.startswith("sacct "):
            return (self.sacct, "", 0) if self.sacct_rc == 0 else \
                ("", "sacct: error: Slurm accounting storage is disabled", self.sacct_rc)
        if cmd.startswith("ls -1A "):
            d = shlex.split(cmd)[2].rstrip("/")
            names = sorted({p[len(d) + 1:].split("/")[0]
                            for p in self.files if p.startswith(d + "/")})
            return "\n".join(names) + "\n", "", 0
        if cmd.startswith("ls -1d "):
            root = re.search(r"ls -1d '([^']+)'/retry_\*", cmd).group(1)
            hits = sorted(d for d in self.dirs if re.fullmatch(re.escape(root) + r"/retry_\d+", d))
            return "\n".join(hits) + ("\n" if hits else ""), "", 0 if hits else 2
        if cmd.startswith("mkdir "):
            d = shlex.split(cmd)[1]
            if d in self.dirs:
                return "", f"mkdir: cannot create directory '{d}': File exists", 1
            self.dirs.add(d)
            return "", "", 0
        if cmd.startswith("test -s "):
            p = shlex.split(cmd)[2]
            return "", "", 0 if self.files.get(p) else 1
        if cmd.startswith("cp ") or "; then cp " in cmd:
            for part in cmd.split(" && "):
                self._cp(part)
            return "", "", 0
        m = re.match(r"if \[ -f '([^']+)' \]; then (.*?); (?:else echo (\S+); )?fi$", cmd, re.S)
        if m:
            return self._guarded(m.group(1), m.group(2), m.group(3))
        if " && " in cmd and cmd.startswith("cp "):
            for part in cmd.split(" && "):
                self._cp(part)
            return "", "", 0
        raise AssertionError(f"FakeConn: unexpected command {cmd!r}")

    def _cp(self, part):
        m = re.match(r"if \[ -f '([^']+)' \]; then cp '([^']+)' '([^']+)'; fi$", part)
        if m:
            if m.group(1) in self.files:
                dst = m.group(3) + m.group(2).rsplit("/", 1)[-1]
                self.files[dst] = self.files[m.group(2)]
            return
        src, dst = shlex.split(part)[1:3]
        self.files[dst] = self.files[src]

    def _guarded(self, path, body, missing):
        if path not in self.files:
            return (f"{missing}\n" if missing else ""), "", 0
        text = self.files[path]
        lines = text.splitlines()
        if body.startswith("tail -n ") and "__SCILINK_SPLIT__" in body:
            n = int(body.split()[2])
            kws = re.findall(r"-e '([^']*)'", body)
            k = int(re.search(r"\| tail -n (\d+)", body).group(1))
            hits = [ln for ln in lines if any(w.lower() in ln.lower() for w in kws)][-k:]
            return "\n".join(lines[-n:]) + "\n__SCILINK_SPLIT__\n" + "\n".join(hits), "", 0
        if body.startswith("tail -n "):
            return "\n".join(lines[-int(body.split()[2]):]) + "\n", "", 0
        if body.startswith("head -n "):
            return "\n".join(lines[: int(body.split()[2])]) + "\n", "", 0
        if body.startswith("grep -F TITEL"):
            hits = [ln for ln in lines if "TITEL" in ln]
            return "\n".join(hits) + "\n", "", 0 if hits else 1
        if body.startswith("cat "):
            return text, "", 0
        raise AssertionError(f"FakeConn: unexpected guarded body {body!r}")

    def upload_text(self, content, remote):
        self.uploaded[remote] = content
        self.files[remote] = content


class FakeSched:
    def __init__(self):
        self.submitted = []

    def submit(self, script, work_dir=""):
        self.submitted.append((script, work_dir))
        return "999001"


# ── realistic fixtures ────────────────────────────────────────

POSCAR = """TiO2 rutile 2x2x2
1.0
   9.187 0.000 0.000
   0.000 9.187 0.000
   0.000 0.000 5.917
Ti O
16 32
Direct
0.0 0.0 0.0
"""

POTCAR_OK = """  PAW_PBE Ti_pv 07Sep2000
   TITEL  = PAW_PBE Ti_pv 07Sep2000
  PAW_PBE O 08Apr2002
   TITEL  = PAW_PBE O 08Apr2002
"""

POTCAR_SWAPPED = """   TITEL  = PAW_PBE O 08Apr2002
   TITEL  = PAW_PBE Ti_pv 07Sep2000
"""

INCAR_RELAX = """SYSTEM = TiO2 rutile
PREC = Accurate
ENCUT = 520
EDIFF = 1E-6
EDIFFG = -0.01
ISMEAR = 0
SIGMA = 0.05
IBRION = 2
ISIF = 3
NSW = 100
NELM = 60
ALGO = Normal
"""

INCAR_STATIC = """SYSTEM = TiO2 static
ENCUT = 520
EDIFF = 1E-6
ISMEAR = 0
SIGMA = 0.05
NSW = 0
IBRION = -1
NELM = 60
"""

KPOINTS = """Automatic
0
Gamma
4 4 6
0 0 0
"""

OUTCAR_NORMAL_END = "\n".join(
    [" POSITION   TOTAL-FORCE (eV/Angst)"] * 5 + [
        " General timing and accounting informations for this job:",
        " ========================================================",
        "                   Total CPU time used (sec):     3605.112",
        "                           Elapsed time (sec):     3621.004",
        "                     Voluntary context switches:     12044",
    ])

OUTCAR_TRUNCATED = "\n".join([
    " running on   64 total cores",
    " POTCAR:    PAW_PBE Ti_pv 07Sep2000",
    "--------------------------------------- Iteration      3(  12)  ---------------------------------------",
    "    POTLOK:  cpu time    0.4121: real time    0.4133",
])

CONTAINER_BANNER = """INFO:    Using cached SIF image
Running VASP 6.4.3 in container /gpfs/projects/ntml/containers/vasp_6.4.3.sif
 running on   64 total cores
 distrk:  each k-point on   64 cores,    1 groups
"""


def _oszicar_converged(n_ionic=3, n_elec=18):
    out = []
    for i in range(1, n_ionic + 1):
        for j in range(1, n_elec + 1):
            out.append(f"DAV: {j:3d}    -0.{900 - i:03d}12345E+03   -0.1E-0{min(j, 9)}   "
                       f"-0.3E-0{min(j, 9)}   512   0.1E-0{min(j, 9)}")
        out.append(f"   {i} F= -.{900 - i:03d}12345E+03 E0= -.{900 - i:03d}10000E+03  "
                   f"d E =-.{i}0000E-02")
    return "\n".join(out) + "\n"


SACCT_FAILED = "7001|FAILED|1:0|00:41:12||04:00:00|64G\n" \
               "7001.batch|FAILED|1:0|00:41:12|20123456K||\n" \
               "7001.extern|COMPLETED|0:0|00:41:12|0||\n"


def _diag(conn, job_id="7001", **kw):
    return diagnose_job(conn, job_id, WD, **kw)


# ── step 1: sacct ─────────────────────────────────────────────

def test_timeout_is_classified_from_sacct_and_stops_at_step_1():
    conn = FakeConn(
        files={"slurm-7001.err": "slurmstepd: error: *** JOB 7001 ... DUE TO TIME LIMIT ***",
               "OUTCAR": OUTCAR_TRUNCATED, "INCAR": INCAR_RELAX},
        sacct="7001|TIMEOUT|0:0|04:00:12||04:00:00|64G\n"
              "7001.batch|CANCELLED|0:15|04:00:14|30123456K||\n"
              "7001.extern|COMPLETED|0:0|04:00:12|0||\n",
    )
    d = _diag(conn)
    assert d.category == CAT_RESOURCES
    assert d.cause == "timeout"
    assert d.stopped_at_step == 1
    assert d.evidence[0].file == "sacct"
    assert any("TIMEOUT" in ln for ln in d.evidence[0].lines)
    # Proposed --time doubles the limit.
    assert d.suggested_actions[0].resource_changes == {"time": "08:00:00"}
    # Relaxation → restart from CONTCAR (reads INCAR, nothing else).
    assert d.suggested_actions[0].restart_from_contcar is True
    assert "OUTCAR" not in conn.read_paths()
    assert "slurm-7001.err" not in conn.read_paths()
    assert not any(c.startswith("ls ") for c in conn.commands)


def test_oom_on_batch_step_row_is_caught_even_when_main_row_says_failed():
    """Older Slurm reports FAILED on the allocation row and OUT_OF_MEMORY
    only on the .batch step — the distinction HPCJob._MAP collapses."""
    conn = FakeConn(
        files={"OUTCAR": OUTCAR_TRUNCATED},
        sacct="7002|FAILED|0:125|00:12:03||04:00:00|16G\n"
              "7002.batch|OUT_OF_MEMORY|0:125|00:12:03|16777216K||\n"
              "7002.extern|COMPLETED|0:0|00:12:03|0||\n",
    )
    d = _diag(conn, job_id="7002")
    assert (d.category, d.cause, d.stopped_at_step) == (CAT_RESOURCES, "out_of_memory", 1)
    assert d.slurm["max_rss"] == "16777216K"
    assert d.suggested_actions[0].resource_changes == {"mem": "32768M"}
    assert conn.read_paths() == []


def test_cancelled_by_user_is_resources_not_a_vasp_error():
    conn = FakeConn(sacct="7003|CANCELLED by 51234|0:0|00:02:00||04:00:00|64G\n")
    d = _diag(conn, job_id="7003")
    assert (d.category, d.cause, d.stopped_at_step) == (CAT_RESOURCES, "cancelled", 1)
    assert "51234" in d.summary


def test_running_job_is_not_diagnosed_as_failed():
    conn = FakeConn(sacct="7004|RUNNING|0:0|00:10:00||04:00:00|64G\n")
    d = _diag(conn, job_id="7004")
    assert d.failed is False and d.category is None and d.stopped_at_step == 1


def test_timeout_without_sacct_is_found_in_slurmstepd_line_at_step_2():
    conn = FakeConn(
        files={"slurm-7005.out": CONTAINER_BANNER,
               "slurm-7005.err": "slurmstepd: error: *** JOB 7005 ON g3012 CANCELLED AT "
                                 "2026-09-29T21:00:03 DUE TO TIME LIMIT ***\n",
               "OUTCAR": OUTCAR_TRUNCATED},
        sacct_rc=1,
    )
    d = _diag(conn, job_id="7005", fallback_state="FAILED")
    assert (d.category, d.cause, d.stopped_at_step) == (CAT_RESOURCES, "slurm_time_limit", 2)
    assert d.evidence[-1].file == "slurm-7005.err"
    assert "OUTCAR" not in conn.read_paths()


# ── step 2: script / container ───────────────────────────────

def test_missing_potcar_abort_stops_at_err_before_touching_vasp_files():
    conn = FakeConn(
        files={"TiO2_7001.err": "Missing POTCAR for element Ti at "
                                "/gpfs/projects/ntml/potpaw_PBE/Ti/POTCAR\n",
               "TiO2_7001.out": "",
               "INCAR": INCAR_RELAX, "POSCAR": POSCAR, "OSZICAR": _oszicar_converged()},
        sacct=SACCT_FAILED,
    )
    d = _diag(conn)
    assert (d.category, d.cause, d.stopped_at_step) == (CAT_SCRIPT, "missing_potcar", 2)
    assert d.evidence[-1].file == "TiO2_7001.err"
    assert "Missing POTCAR for element Ti" in d.evidence[-1].lines[0]
    assert conn.read_paths() == ["TiO2_7001.err"]  # stopped at first log
    assert "Ti_pv" in d.suggested_actions[0].description


def test_elements_guard_abort_is_a_script_failure():
    conn = FakeConn(
        files={"slurm-7001.err": "No elements parsed from POSCAR; aborting.\n"},
        sacct=SACCT_FAILED,
    )
    d = _diag(conn)
    assert (d.category, d.cause, d.stopped_at_step) == (CAT_SCRIPT, "no_elements", 2)


def test_gpu_not_visible_in_container_is_a_script_failure():
    conn = FakeConn(
        files={"slurm-7001.out": "NVIDIA-SMI has failed because it couldn't communicate "
                                 "with the NVIDIA driver.\n",
               "slurm-7001.err": "WARNING: Could not find any nv files on this host!\n"},
        sacct=SACCT_FAILED,
    )
    d = _diag(conn)
    assert (d.category, d.cause, d.stopped_at_step) == (CAT_SCRIPT, "gpu_not_detected", 2)
    assert d.evidence[-1].file == "slurm-7001.err"  # stderr checked first


def test_wizard_log_paths_are_used_when_tracked():
    conn = FakeConn(
        files={"TiO2_7001.err": "FATAL:   could not open image /gpfs/x.sif: "
                                "failed to retrieve path\n"},
        sacct=SACCT_FAILED,
    )
    d = _diag(conn, stderr_file=f"{WD}/TiO2_7001.err", stdout_file=f"{WD}/TiO2_7001.out")
    assert (d.category, d.cause) == (CAT_SCRIPT, "container_error")


def test_step2_does_not_stop_on_vasp_output_in_batch_stdout():
    """run_vasp_container.sh without redirection: VASP's error lands in the
    batch .out. Step 2 must NOT claim it (no generic 'error' pattern); step 3
    must pick it up and name the batch .out as the VASP stdout it used."""
    vasp_err = (CONTAINER_BANNER +
                " -----------------------------------------------------------------------------\n"
                "|     Error EDDDAV: Call to ZHEGV failed. Returncode = 17 1 16                |\n"
                " -----------------------------------------------------------------------------\n"
                "srun: error: g3012: task 0: Exited with exit code 1\n")
    conn = FakeConn(files={"slurm-7001.out": vasp_err, "slurm-7001.err": "",
                           "OUTCAR": OUTCAR_TRUNCATED},
                    sacct=SACCT_FAILED)
    d = _diag(conn)
    assert (d.category, d.cause, d.stopped_at_step) == (CAT_VASP, "edddav", 3)
    assert d.evidence[-1].file == "slurm-7001.out"
    assert d.suggested_actions[0].incar_changes == {"ALGO": "All"}
    assert "OUTCAR" not in conn.read_paths()


def test_chat_submitted_job_logs_vasp_stdout_and_stderr_are_found():
    """submit_simulation_job writes --output=vasp.stdout --error=vasp.stderr;
    sacct does not report those paths, so they must be discovered."""
    conn = FakeConn(files={"job.sh": "#!/bin/bash\n", "vasp.stdout": CONTAINER_BANNER,
                           "vasp.stderr": "Lmod has detected the following error: "
                                          "The following module(s) are unknown: \"vasp/6.4\"\n",
                           "OUTCAR": OUTCAR_TRUNCATED},
                    sacct=SACCT_FAILED)
    d = _diag(conn)
    assert (d.category, d.cause, d.stopped_at_step) == (CAT_SCRIPT, "module_error", 2)
    assert conn.read_paths() == ["vasp.stderr"]


def test_chat_submitted_job_vasp_error_in_vasp_stdout_is_read_once():
    conn = FakeConn(files={"vasp.stdout": CONTAINER_BANNER + " ZBRENT: fatal error in bracketing\n",
                           "vasp.stderr": "", "OUTCAR": OUTCAR_TRUNCATED},
                    sacct=SACCT_FAILED)
    d = _diag(conn)
    assert (d.category, d.cause, d.stopped_at_step) == (CAT_VASP, "zbrent", 3)
    assert d.evidence[-1].file == "vasp.stdout"
    assert conn.read_paths() == ["vasp.stderr", "vasp.stdout"]


# ── step 3: VASP stdout ───────────────────────────────────────

def test_zbrent_in_vasp_out_stops_at_step_3():
    vasp_out = "\n".join([
        "      11 F= -.87912345E+03 E0= -.87910000E+03  d E =-.1234E-03",
        "  ZBRENT: fatal error in bracketing",
        "     please rerun with smaller EDIFF, or copy CONTCAR",
        "     to POSCAR and continue",
    ])
    conn = FakeConn(files={"slurm-7001.out": CONTAINER_BANNER, "slurm-7001.err": "",
                           "vasp.out": vasp_out, "OUTCAR": OUTCAR_TRUNCATED,
                           "OSZICAR": _oszicar_converged(), "INCAR": INCAR_RELAX},
                    sacct=SACCT_FAILED)
    d = _diag(conn)
    assert (d.category, d.cause, d.stopped_at_step) == (CAT_VASP, "zbrent", 3)
    assert d.evidence[-1].file == "vasp.out"
    assert any("ZBRENT" in ln for ln in d.evidence[-1].lines)
    assert d.suggested_actions[0].incar_changes == {"IBRION": "1"}
    assert d.suggested_actions[0].restart_from_contcar is True
    reads = conn.read_paths()
    assert reads[:3] == ["slurm-7001.err", "slurm-7001.out", "vasp.out"]
    assert "OUTCAR" not in reads and "OSZICAR" not in reads


def test_subspace_matrix_error_in_vasp_out():
    conn = FakeConn(files={"vasp.out": "WARNING: Sub-Space-Matrix is not hermitian in DAV   4\n"
                                       "  -1.2345E-02\n"},
                    sacct=SACCT_FAILED)
    d = _diag(conn)
    assert (d.category, d.cause) == (CAT_VASP, "subspace_not_hermitian")


def test_killed_by_signal_9_is_suspected_oom():
    conn = FakeConn(files={"vasp.out": "===================================================\n"
                                       "=   BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES\n"
                                       "=   KILLED BY SIGNAL: 9 (Killed)\n"},
                    sacct=SACCT_FAILED)
    d = _diag(conn)
    assert (d.category, d.cause, d.stopped_at_step) == (CAT_RESOURCES, "killed_signal_9", 3)


# ── step 4: OUTCAR ────────────────────────────────────────────

def test_symmetry_error_only_in_outcar_is_found_by_remote_grep():
    outcar = "\n".join(
        ["  VRHFIN =Ti: 3p3d4s"]
        + [" line of setup"] * 500  # error is far above the tail window
        + [" internal error in subroutine SGRCON:",
           " Found some non-integer element in rotation matrix"]
        + [" padding"] * 200)
    conn = FakeConn(files={"slurm-7001.out": CONTAINER_BANNER, "OUTCAR": outcar},
                    sacct=SACCT_FAILED)
    d = _diag(conn)
    assert (d.category, d.cause, d.stopped_at_step) == (CAT_VASP, "symmetry", 4)
    assert d.evidence[-1].file == "OUTCAR"
    assert d.suggested_actions[0].incar_changes == {"ISYM": "0"}
    # Never a full download: OUTCAR only via tail + grep.
    outcar_cmds = [c for c in conn.commands if "OUTCAR'" in c]
    assert outcar_cmds and all("tail -n 60" in c and "grep -F" in c and "cat " not in c
                               for c in outcar_cmds)


# ── fatal VASP error boxes (any message, rule or not) ─────────

ERROR_BOX = """\
 -----------------------------------------------------------------------------
|                                                                             |
|     EEEEEEE  RRRRRR   RRRRRR   OOOOOOO  RRRRRR      ###     ###     ###      |
|     E        R     R  R     R  O     O  R     R     ###     ###     ###      |
|     E        R     R  R     R  O     O  R     R     ###     ###     ###      |
|     EEEEE    RRRRRR   RRRRRR   O     O  RRRRRR       #       #       #       |
|     E        R   R    R   R    O     O  R   R                                |
|     E        R    R   R    R   O     O  R    R      ###     ###     ###      |
|     EEEEEEE  R     R  R     R  OOOOOOO  R     R     ###     ###     ###      |
|                                                                             |
|     The number of bands is not sufficient to hold all electrons.            |
|     Please increase NBANDS (currently 96, need at least 128).               |
|                                                                             |
|       ---->  I REFUSE TO CONTINUE WITH THIS SICK JOB ... BYE!!! <----       |
|                                                                             |
 -----------------------------------------------------------------------------
"""

WARNING_BOX = """\
 -----------------------------------------------------------------------------
|                                                                             |
|           W    W    AA    RRRRR   N    N  II  N    N   GGGG   !!!           |
|           W    W   A  A   R    R  NN   N  II  NN   N  G    G  !!!           |
|           W    W  A    A  R    R  N N  N  II  N N  N  G       !!!           |
|           W WW W  AAAAAA  RRRRR   N  N N  II  N  N N  G  GGG   !            |
|           WW  WW  A    A  R   R   N   NN  II  N   NN  G    G                |
|           W    W  A    A  R    R  N    N  II  N    N   GGGG   !!!           |
|                                                                             |
|     The distance between some ions is large; consider a larger cutoff.     |
|                                                                             |
 -----------------------------------------------------------------------------
"""


def _box_evidence(d):
    ev = d.evidence[-1]
    return ev, "\n".join(ev.lines)


def test_unmatched_error_box_in_outcar_is_a_vasp_error_with_the_whole_box():
    """Regression: a fatal box whose message matches no rule used to fall
    through to 'outcar_truncated' showing only the box's last 3 lines."""
    conn = FakeConn(files={"slurm-7001.out": CONTAINER_BANNER, "slurm-7001.err": "",
                           "OUTCAR": OUTCAR_TRUNCATED + "\n" + ERROR_BOX,
                           "OSZICAR": _oszicar_converged(), "INCAR": INCAR_RELAX},
                    sacct=SACCT_FAILED)
    d = _diag(conn)
    assert (d.category, d.cause, d.stopped_at_step) == (CAT_VASP, "vasp_fatal_error", 4)
    ev, text = _box_evidence(d)
    assert ev.file == "OUTCAR" and ev.rule == "vasp_error_box"
    assert "The number of bands is not sufficient to hold all electrons." in text
    assert "Please increase NBANDS" in text
    assert "I REFUSE TO CONTINUE WITH THIS SICK JOB" in text
    assert "EEEEEEE" not in text  # ASCII-art banner stripped
    assert "number of bands is not sufficient" in d.summary
    assert "I REFUSE" not in d.summary
    assert not any(e.rule == "outcar_truncated" for e in d.evidence)
    assert "OSZICAR" not in conn.read_paths()


def test_unmatched_error_box_in_batch_stdout_stops_at_step_3():
    conn = FakeConn(files={"slurm-7001.out": CONTAINER_BANNER + ERROR_BOX,
                           "slurm-7001.err": "", "OUTCAR": OUTCAR_TRUNCATED + "\n" + ERROR_BOX},
                    sacct=SACCT_FAILED)
    d = _diag(conn)
    assert (d.category, d.cause, d.stopped_at_step) == (CAT_VASP, "vasp_fatal_error", 3)
    ev, text = _box_evidence(d)
    assert ev.file == "slurm-7001.out"
    assert "Please increase NBANDS" in text and "I REFUSE TO CONTINUE" in text
    assert "OUTCAR" not in conn.read_paths()


def test_rule_hit_inside_a_box_shows_the_full_box_as_evidence():
    zbrent_box = ERROR_BOX.replace(
        "|     The number of bands is not sufficient to hold all electrons.            |\n"
        "|     Please increase NBANDS (currently 96, need at least 128).               |\n",
        "|     ZBRENT: fatal error in bracketing                                       |\n"
        "|     please rerun with smaller EDIFF, or copy CONTCAR                        |\n"
        "|     to POSCAR and continue                                                  |\n")
    assert "ZBRENT" in zbrent_box
    conn = FakeConn(files={"vasp.out": zbrent_box, "INCAR": INCAR_RELAX}, sacct=SACCT_FAILED)
    d = _diag(conn)
    assert (d.category, d.cause, d.stopped_at_step) == (CAT_VASP, "zbrent", 3)
    ev, text = _box_evidence(d)
    assert ev.rule == "zbrent"
    assert "please rerun with smaller EDIFF" in text and "to POSCAR and continue" in text
    assert "I REFUSE TO CONTINUE" in text


BZINTS_BOX = """\
 -----------------------------------------------------------------------------
|                     _     ____    _    _    _____     _                     |
|                    | |   |  _ \\  | |  | |  / ____|   | |                    |
|                    | |   | |_) | | |  | | | |  __    | |                    |
|                    |_|   |  _ <  | |  | | | | |_ |   |_|                    |
|                     _    | |_) | | |__| | | |__| |    _                     |
|                    (_)   |____/   \\____/   \\_____|   (_)                    |
|                                                                             |
|     VERY BAD NEWS! internal error in subroutine BZINTS: Tetrahedron         |
|     method fails (number of k-points < 4)          2                        |
|                                                                             |
|     If you are not a developer, you should not encounter this problem.      |
|     Please submit a bug report.                                             |
|                                                                             |
 -----------------------------------------------------------------------------
"""


@pytest.mark.parametrize("box", [
    BZINTS_BOX,
    # Same message with the subroutine name gone: only the text wrapped across
    # two box rows ("Tetrahedron" / "method fails") identifies it.
    BZINTS_BOX.replace("subroutine BZINTS: Tetrahedron", "subroutine XXXXXX: Tetrahedron"),
])
def test_tetrahedron_message_wrapped_across_box_rows_is_recognised(box):
    """Regression (live Tillicum run): the message is split over two rows, so a
    per-line match missed it and the job fell to the generic box fallback."""
    conn = FakeConn(files={"vasp.out": box, "INCAR": "ISMEAR = -5\n"}, sacct=SACCT_FAILED)
    d = _diag(conn)
    assert (d.category, d.cause, d.stopped_at_step) == (CAT_VASP, "tetrahedron", 3)
    ev, text = _box_evidence(d)
    assert ev.rule == "tetrahedron"
    assert "method fails (number of k-points < 4)" in text
    assert d.suggested_actions[0].incar_changes == {"ISMEAR": "0", "SIGMA": "0.05"}
    assert "k-point" in d.suggested_actions[0].description


def test_bug_box_is_captured_through_internal_error_marker():
    bug_box = """\
 -----------------------------------------------------------------------------
|                     _     ____    _    _    _____     _                     |
|                    | |   |  _ \\  | |  | |  / ____|   | |                    |
|                    | |   | |_) | | |  | | | |  __    | |                    |
|                    |_|   |  _ <  | |  | | | | |_ |   |_|                    |
|                     _    | |_) | | |__| | | |__| |    _                     |
|                    (_)   |____/   \\____/   \\_____|   (_)                    |
|                                                                             |
|     internal error in: mpi.F  at line: 1825                                 |
|                                                                             |
|     M_divide: can not subdivide 16 nodes by 6                               |
|                                                                             |
|     If you are not a developer, you should not encounter this problem.      |
|     Please submit a bug report.                                             |
|                                                                             |
 -----------------------------------------------------------------------------
"""
    conn = FakeConn(files={"vasp.out": bug_box}, sacct=SACCT_FAILED)
    d = _diag(conn)
    assert (d.category, d.cause) == (CAT_VASP, "npar_divide")
    ev, text = _box_evidence(d)
    assert "internal error in: mpi.F" in text and "M_divide" in text
    assert "____" not in text  # BUG banner stripped


def test_warning_box_alone_is_not_a_fatal_error():
    conn = FakeConn(files={"slurm-7001.out": CONTAINER_BANNER + WARNING_BOX,
                           "slurm-7001.err": "",
                           "OUTCAR": OUTCAR_TRUNCATED + "\n" + WARNING_BOX,
                           "OSZICAR": _oszicar_converged(),
                           "INCAR": INCAR_RELAX, "POSCAR": POSCAR, "POTCAR": POTCAR_OK,
                           "KPOINTS": KPOINTS},
                    sacct=SACCT_FAILED)
    d = _diag(conn)
    assert d.cause != "vasp_fatal_error"
    assert (d.category, d.cause) == (CAT_UNKNOWN, "undetermined")
    # The non-box fallback now shows enough of the OUTCAR tail to read.
    trunc = next(e for e in d.evidence if e.rule == "outcar_truncated")
    assert len(trunc.lines) == 10


# ── step 5: OSZICAR ───────────────────────────────────────────

def test_scf_hitting_nelm_is_convergence_even_when_slurm_says_completed():
    """Static run: VASP exits cleanly after NELM with an unconverged SCF —
    Slurm reports COMPLETED 0:0, OUTCAR ends normally."""
    osz = "\n".join(f"DAV: {j:3d}    -0.87912345E+03   0.4{j % 10}E-02   -0.1E-02  1024   0.3E+00"
                    for j in range(1, 61))
    osz += "\n   1 F= -.87912345E+03 E0= -.87910000E+03  d E =-.8791E+03\n"
    conn = FakeConn(files={"slurm-7001.out": CONTAINER_BANNER, "slurm-7001.err": "",
                           "OUTCAR": OUTCAR_NORMAL_END, "OSZICAR": osz,
                           "INCAR": INCAR_STATIC, "POSCAR": POSCAR, "POTCAR": POTCAR_OK},
                    sacct="7001|COMPLETED|0:0|01:00:21||04:00:00|64G\n")
    d = _diag(conn)
    assert (d.category, d.cause, d.stopped_at_step) == (CAT_CONVERGENCE, "scf_not_converged", 5)
    assert d.evidence[-1].file == "OSZICAR"
    assert any("DAV:  60" in ln for ln in d.evidence[-1].lines)
    act = d.suggested_actions[0]
    assert act.incar_changes["ALGO"] == "All" and int(act.incar_changes["NELM"]) > 60
    assert "POSCAR" not in conn.read_paths() and "POTCAR" not in conn.read_paths()


def test_run_killed_inside_an_scf_loop_at_nelm():
    osz = _oszicar_converged(n_ionic=4) + "\n".join(
        f"RMM: {j:3d}    -0.87912345E+03   0.4E-02   -0.1E-02  1024   0.3E+00"
        for j in range(1, 81)) + "\n"
    conn = FakeConn(files={"OUTCAR": OUTCAR_TRUNCATED, "OSZICAR": osz,
                           "INCAR": INCAR_RELAX.replace("NELM = 60", "NELM = 80")},
                    sacct=SACCT_FAILED)
    d = _diag(conn)
    assert (d.category, d.cause, d.stopped_at_step) == (CAT_CONVERGENCE, "scf_not_converged", 5)


def test_oscillating_ionic_steps_are_convergence():
    energies = [-879.10, -879.30, -879.12, -879.31, -879.11, -879.32, -879.10, -879.30]
    osz = ""
    for i, e in enumerate(energies, 1):
        osz += "DAV:   1    -0.8E+03   0.1E-06  -0.1E-06  512  0.1E-04\n"
        osz += f"   {i} F= {e:.8E} E0= {e:.8E}  d E =0.1E+00\n"
    conn = FakeConn(files={"OUTCAR": OUTCAR_TRUNCATED, "OSZICAR": osz, "INCAR": INCAR_RELAX},
                    sacct=SACCT_FAILED)
    d = _diag(conn)
    assert (d.category, d.cause) == (CAT_CONVERGENCE, "ionic_oscillation")
    assert d.suggested_actions[0].restart_from_contcar


def test_parse_oszicar_counts_electronic_steps_per_ionic_step():
    steps, trailing = parse_oszicar(_oszicar_converged(n_ionic=2, n_elec=7))
    assert [s["n"] for s in steps] == [1, 2]
    assert [s["n_elec"] for s in steps] == [7, 7]
    assert trailing == []


# ── step 6: inputs / undetermined ────────────────────────────

def test_poscar_potcar_order_mismatch_is_inputs():
    conn = FakeConn(files={"slurm-7001.out": CONTAINER_BANNER, "OUTCAR": OUTCAR_TRUNCATED,
                           "OSZICAR": _oszicar_converged(n_ionic=1),
                           "INCAR": INCAR_RELAX, "POSCAR": POSCAR,
                           "POTCAR": POTCAR_SWAPPED, "KPOINTS": KPOINTS},
                    sacct=SACCT_FAILED)
    d = _diag(conn)
    assert (d.category, d.cause, d.stopped_at_step) == (CAT_INPUTS, "input_inconsistency", 6)
    assert any(ev.file == "POTCAR" for ev in d.evidence)
    assert "['O', 'Ti']" in d.summary


def test_magmom_length_mismatch_is_inputs():
    incar = INCAR_RELAX + "ISPIN = 2\nMAGMOM = 16*1.0 16*0.0\n"
    conn = FakeConn(files={"INCAR": incar, "POSCAR": POSCAR, "POTCAR": POTCAR_OK,
                           "KPOINTS": KPOINTS},
                    sacct=SACCT_FAILED)
    d = _diag(conn)
    assert d.category == CAT_INPUTS
    assert "MAGMOM has 32 values for 48 atoms" in d.summary


def test_clean_failure_is_reported_as_undetermined():
    conn = FakeConn(files={"slurm-7001.out": CONTAINER_BANNER, "slurm-7001.err": "",
                           "OUTCAR": OUTCAR_TRUNCATED, "OSZICAR": _oszicar_converged(),
                           "INCAR": INCAR_RELAX, "POSCAR": POSCAR, "POTCAR": POTCAR_OK,
                           "KPOINTS": KPOINTS},
                    sacct=SACCT_FAILED)
    d = _diag(conn)
    assert (d.category, d.cause, d.stopped_at_step) == (CAT_UNKNOWN, "undetermined", 6)
    assert d.summary.startswith("Cause undetermined")
    assert any(ev.rule == "outcar_truncated" for ev in d.evidence)
    checked = {fc.file for fc in d.files_checked}
    assert {"sacct -j 7001", "slurm-7001.err", "slurm-7001.out", "OUTCAR",
            "OSZICAR", "POSCAR", "POTCAR", "INCAR", "KPOINTS"} <= checked


def test_clean_completed_run_is_not_a_failure():
    conn = FakeConn(files={"slurm-7001.out": CONTAINER_BANNER, "OUTCAR": OUTCAR_NORMAL_END,
                           "OSZICAR": _oszicar_converged(n_ionic=1),
                           "INCAR": INCAR_STATIC, "POSCAR": POSCAR, "POTCAR": POTCAR_OK,
                           "KPOINTS": KPOINTS},
                    sacct="7001|COMPLETED|0:0|01:00:21||04:00:00|64G\n")
    d = _diag(conn)
    assert d.failed is False and d.category is None


def test_connection_errors_do_not_crash_the_cascade():
    class Dead:
        def run(self, cmd, timeout=0):
            raise OSError("Socket is closed")

    d = diagnose_job(Dead(), "7001", WD, fallback_state="FAILED")
    # Unreachable ≠ "POSCAR missing": must not be misreported as an input error.
    assert (d.category, d.cause) == (CAT_UNKNOWN, "workdir_unreachable")
    assert "Socket is closed" in d.summary
    assert d.failed


def test_save_and_load_roundtrip(tmp_path):
    conn = FakeConn(sacct="7001|TIMEOUT|0:0|04:00:12||04:00:00|64G\n")
    d = _diag(conn)
    jp = save_diagnosis(d, tmp_path)
    assert jp == tmp_path / "diagnostics" / "job_7001" / "diagnosis.json"
    md = (jp.parent / "diagnosis.md").read_text(encoding="utf-8")
    assert "Resources" in md and "TIMEOUT" in md
    back = load_diagnosis(tmp_path, "7001")
    assert back.to_dict() == d.to_dict()


def test_diagnoses_saved_with_old_category_values_still_load():
    from scilink.hpc.diagnosis import JobDiagnosis
    old = JobDiagnosis.from_dict({"job_id": "1", "work_dir": "/x", "category": "recursos"})
    assert old.category == CAT_RESOURCES and old.category_label == "Resources"
    assert JobDiagnosis.from_dict(
        {"job_id": "1", "work_dir": "/x", "category": "error_vasp"}).category == CAT_VASP


# ── retry helpers ─────────────────────────────────────────────

def _wizard_script(work_dir=WD, time="04:00:00"):
    from scilink.ui.components.vasp_workflow import _build_slurm_script
    return _build_slurm_script(
        job_name="TiO2", account="ntml", partition="gpu-h200", time_limit=time,
        nodes=1, ntasks_per_node=4, extra_sbatch="--gpus-per-node=4\n--mem=64G",
        work_dir=work_dir, module_commands="module load apptainer",
        pseudo_dir="/gpfs/projects/ntml/potpaw_PBE", elements=["Ti", "O"],
        vasp_command="/gpfs/projects/ntml/scripts/run_vasp_container.sh",
    )


def test_rewrite_workdir_moves_cd_to_the_retry_folder():
    new = f"{WD}/retry_1"
    out = rewrite_workdir(_wizard_script(), WD, new)
    assert f"cd '{new}'" in out
    assert f"cd '{WD}'\n" not in out
    assert "/gpfs/projects/ntml/potpaw_PBE" in out  # other paths untouched


def test_rewrite_workdir_refuses_foreign_cd_and_leftover_references():
    with pytest.raises(ValueError):
        rewrite_workdir(f"#!/bin/bash\ncd /somewhere/else\nvasp_std\n", WD, WD + "/retry_1")
    with pytest.raises(ValueError):
        rewrite_workdir(f"#!/bin/bash\ncd '{WD}'\ncp {WD}/WAVECAR .\n", WD, WD + "/retry_1")


def test_apply_changes_to_incar_and_sbatch():
    inc = apply_incar_changes(INCAR_RELAX, {"ALGO": "All", "NELM": "120", "ISYM": "0"})
    assert "ALGO = All" in inc and "NELM = 120" in inc and inc.rstrip().endswith("ISYM = 0")
    assert inc.count("ALGO") == 1
    scr = apply_resource_changes(_wizard_script(), {"time": "08:00:00", "mem": "131072M"})
    assert "#SBATCH --time=08:00:00" in scr and "04:00:00" not in scr
    assert "#SBATCH --mem=131072M" in scr and "--mem=64G" not in scr


def test_retry_proposal_caps_resources_and_enforces_retry_limit():
    conn = FakeConn(sacct="7001|TIMEOUT|0:0|30:00:12||30:00:00|64G\n")
    d = _diag(conn)
    policy = RetryPolicy(max_time_limit="48:00:00")
    p = build_retry_proposal(d, policy=policy, current_script=_wizard_script(time="30:00:00"))
    # 60 h wanted (2 × 30 h), capped at the policy's 48 h.
    assert parse_slurm_time(p.resource_changes["time"]) == 48 * 3600
    assert p.allowed and p.requires_approval
    assert any("capped" in n for n in p.notes)
    blocked = build_retry_proposal(d, attempts_so_far=1, policy=policy)
    assert not blocked.allowed and "Retry limit" in blocked.blocked_reason
    assert policy.can_auto_resubmit() is False
    assert RetryPolicy(auto_resubmit=True).can_auto_resubmit() is False


def test_retry_requires_approval_and_never_touches_the_failed_run():
    files = {"submit.sh": _wizard_script(), "INCAR": INCAR_RELAX, "POSCAR": POSCAR,
             "CONTCAR": POSCAR.replace("TiO2 rutile 2x2x2", "relaxed"),
             "KPOINTS": KPOINTS, "POTCAR": POTCAR_OK, "OUTCAR": OUTCAR_TRUNCATED}
    conn = FakeConn(files=files, sacct="7001|TIMEOUT|0:0|04:00:12||04:00:00|64G\n")
    before = dict(conn.files)
    d = _diag(conn)
    p = build_retry_proposal(d, current_script=files["submit.sh"])
    sched = FakeSched()

    n_cmds = len(conn.commands)
    with pytest.raises(PermissionError):
        prepare_and_submit_retry(conn, sched, p, approved=False)
    with pytest.raises(PermissionError):
        prepare_and_submit_retry(conn, sched, p, approved="yes")  # truthy is not approval
    assert len(conn.commands) == n_cmds and sched.submitted == []

    res = prepare_and_submit_retry(conn, sched, p, approved=True)
    new = f"{WD}/retry_1"
    assert res["work_dir"] == new and res["job_id"] == "999001"
    assert sched.submitted == [(f"{new}/submit.sh", new)]
    assert res["structure_from"] == "CONTCAR"
    assert conn.files[f"{new}/POSCAR"].startswith("relaxed")
    assert "#SBATCH --time=08:00:00" in conn.files[f"{new}/submit.sh"]
    assert f"cd '{new}'" in conn.files[f"{new}/submit.sh"]
    # Original files byte-identical, nothing written outside retry_1.
    for k, v in before.items():
        assert conn.files[k] == v
    assert all(k.startswith(new + "/") for k in conn.uploaded)

    # A second retry never reuses retry_1.
    res2 = prepare_and_submit_retry(conn, sched, p, approved=True)
    assert res2["work_dir"] == f"{WD}/retry_2"
