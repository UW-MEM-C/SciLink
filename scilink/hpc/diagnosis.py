"""Cascading failure diagnosis for VASP jobs run on a SLURM cluster.

When a job ends badly, :func:`diagnose_job` walks the run's evidence in a
fixed order and stops at the first step that explains the failure:

1. ``sacct`` — TIMEOUT / OUT_OF_MEMORY / NODE_FAIL / CANCELLED are
   conclusive on their own; FAILED / COMPLETED only record the exit code.
2. The batch script's stderr, then stdout — script-level aborts (missing
   POTCAR, the ELEMENTS guard, container runtime errors, GPU not visible,
   ``command not found``) plus slurmstepd time/memory kills when ``sacct``
   was unavailable.
3. VASP's own stdout (``vasp.out`` & co.; when the run was wrapped in a
   container script that did not redirect it, the batch stdout *is* VASP's
   stdout, so it is re-scanned with the VASP rules) — ZBRENT, EDDDAV,
   Sub-Space-Matrix, symmetry errors, … and any other fatal VASP error box.
4. The end of ``OUTCAR`` — normal termination vs cut off mid-run, plus the
   error boxes VASP 6 also writes there.
5. ``OSZICAR`` — SCF that hit NELM, oscillating ionic steps, NSW exhausted.
6. Input consistency — POSCAR vs POTCAR elements/order, INCAR tag
   combinations. If nothing is found the cause is reported as undetermined.

Everything is read through ``conn.run`` with remote ``tail`` / ``grep`` /
``head`` so no output file is ever downloaded whole. The only thing required
of ``conn`` is ``run(cmd, timeout=...) -> (stdout, stderr, rc)``, which keeps
this module free of paramiko and of the simulation-agent stack (and makes it
testable with a fake connection).

The diagnosis never changes anything on the cluster. Proposed corrections
live in :mod:`scilink.hpc.retry` and are applied only after explicit human
approval.
"""
from __future__ import annotations

import json
import posixpath
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── categories ────────────────────────────────────────────────

CAT_RESOURCES = "resources"
CAT_SCRIPT = "script"
CAT_CONVERGENCE = "convergence"
CAT_VASP = "vasp_error"
CAT_INPUTS = "inputs"
CAT_UNKNOWN = "unknown"

CATEGORY_LABELS: Dict[str, str] = {
    CAT_RESOURCES: "Resources",
    CAT_SCRIPT: "Script",
    CAT_CONVERGENCE: "Convergence",
    CAT_VASP: "VASP error",
    CAT_INPUTS: "Inputs",
    CAT_UNKNOWN: "Unknown",
}

# Category values written by earlier (Spanish) versions of this module, so
# diagnoses already saved in a session still load with the right label.
_LEGACY_CATEGORIES = {
    "recursos": CAT_RESOURCES,
    "convergencia": CAT_CONVERGENCE,
    "error_vasp": CAT_VASP,
    "entradas": CAT_INPUTS,
    "desconocida": CAT_UNKNOWN,
}

STEP_NAMES: Dict[int, str] = {
    1: "Slurm state (sacct)",
    2: "Script logs (batch stderr / stdout)",
    3: "VASP standard output",
    4: "End of OUTCAR",
    5: "OSZICAR",
    6: "Input validation",
}

_MISSING = "__SCILINK_NO_FILE__"
_SPLIT = "__SCILINK_SPLIT__"

# Tail sizes: enough to reach the error block without pulling whole files.
_LOG_TAIL = 400
_OUTCAR_TAIL = 60
_OSZICAR_TAIL = 400
_INCAR_HEAD = 300


# ── data model ────────────────────────────────────────────────

@dataclass
class Evidence:
    step: int
    file: str
    lines: List[str]
    rule: str = ""


@dataclass
class FileCheck:
    step: int
    file: str
    method: str
    exists: Optional[bool] = None
    note: str = ""


@dataclass
class SuggestedAction:
    description: str
    incar_changes: Dict[str, str] = field(default_factory=dict)
    resource_changes: Dict[str, str] = field(default_factory=dict)
    restart_from_contcar: bool = False


@dataclass
class JobDiagnosis:
    job_id: str
    work_dir: str
    engine: str = "vasp"
    failed: bool = True
    category: Optional[str] = None
    cause: str = ""
    summary: str = ""
    stopped_at_step: Optional[int] = None
    slurm: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Evidence] = field(default_factory=list)
    files_checked: List[FileCheck] = field(default_factory=list)
    suggested_actions: List[SuggestedAction] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    @property
    def category_label(self) -> str:
        if self.category is None:
            return "No failure detected"
        return CATEGORY_LABELS.get(self.category, self.category)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category_label"] = self.category_label
        d["stopped_at_step_name"] = STEP_NAMES.get(self.stopped_at_step or 0, "")
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JobDiagnosis":
        d = dict(d)
        d.pop("category_label", None)
        d.pop("stopped_at_step_name", None)
        d["category"] = _LEGACY_CATEGORIES.get(d.get("category"), d.get("category"))
        d["evidence"] = [Evidence(**e) for e in d.get("evidence", [])]
        d["files_checked"] = [FileCheck(**f) for f in d.get("files_checked", [])]
        d["suggested_actions"] = [
            SuggestedAction(**a) for a in d.get("suggested_actions", [])
        ]
        return cls(**d)

    def to_markdown(self) -> str:
        out = [
            f"# Diagnosis of job {self.job_id}",
            "",
            f"- **Remote directory:** `{self.work_dir}`",
            f"- **Category:** {self.category_label}",
            f"- **Cause:** `{self.cause or '—'}`",
            f"- **Stopped at:** step {self.stopped_at_step} — "
            f"{STEP_NAMES.get(self.stopped_at_step or 0, '—')}",
            f"- **Generated:** {self.created_at}",
            "",
            "## Summary",
            "",
            self.summary or "—",
        ]
        if self.slurm:
            out += ["", "## Slurm", ""]
            for k, v in self.slurm.items():
                if k != "rows":
                    out.append(f"- {k}: `{v}`")
        if self.evidence:
            out += ["", "## Evidence", ""]
            for ev in self.evidence:
                out.append(f"**Step {ev.step} — `{ev.file}`**"
                           + (f" (rule `{ev.rule}`)" if ev.rule else ""))
                out += ["", "```text", *ev.lines, "```", ""]
        if self.suggested_actions:
            out += ["## Suggested action", ""]
            for a in self.suggested_actions:
                out.append(f"- {a.description}")
                for k, v in a.incar_changes.items():
                    out.append(f"  - INCAR `{k} = {v}`")
                for k, v in a.resource_changes.items():
                    out.append(f"  - Resource `{k}` → `{v}`")
                if a.restart_from_contcar:
                    out.append("  - Restart from CONTCAR")
        if self.notes:
            out += ["", "## Notes", ""]
            out += [f"- {n}" for n in self.notes]
        out += ["", "## Files checked", ""]
        for fc in self.files_checked:
            state = {True: "present", False: "missing", None: "?"}[fc.exists]
            extra = f" — {fc.note}" if fc.note else ""
            out.append(f"- Step {fc.step}: `{fc.file}` ({fc.method}; {state}){extra}")
        return "\n".join(out) + "\n"


# ── pattern rules ─────────────────────────────────────────────

ActionFn = Callable[["_Ctx"], SuggestedAction]


@dataclass(frozen=True)
class _Rule:
    rule_id: str
    pattern: str
    category: str
    summary: str
    action: Callable[["_Ctx"], SuggestedAction]
    flags: int = re.IGNORECASE | re.MULTILINE

    def search(self, text: str) -> List[str]:
        rx = re.compile(self.pattern, self.flags)
        return [ln.rstrip() for ln in text.splitlines() if rx.search(ln)]


def _act(description: str, **kw) -> ActionFn:
    return lambda ctx: SuggestedAction(description=description, **kw)


def _time_action(ctx: "_Ctx") -> SuggestedAction:
    cur = parse_slurm_time(ctx.slurm.get("timelimit", ""))
    changes = {"time": format_slurm_time(cur * 2)} if cur else {}
    return SuggestedAction(
        description=(
            "Increase the time limit (--time). For a relaxation, restart from "
            "CONTCAR so the completed ionic steps are not repeated."
        ),
        resource_changes=changes,
        restart_from_contcar=ctx.is_relaxation(),
    )


def _mem_action(ctx: "_Ctx") -> SuggestedAction:
    req = parse_mem_mb(ctx.slurm.get("req_mem", ""))
    used = parse_mem_mb(ctx.slurm.get("max_rss", ""))
    base = max(v for v in (req, used, 0.0) if v is not None)
    changes = {"mem": f"{int(base * 2)}M"} if base else {}
    return SuggestedAction(
        description=(
            "Request more memory (--mem) or spread the run over more nodes; "
            "check whether NCORE/KPAR or a high ENCUT/NBANDS drive the usage."
        ),
        resource_changes=changes,
    )


def _scf_action(ctx: "_Ctx") -> SuggestedAction:
    algo = ctx.incar.get("ALGO", "Normal").strip().lower()
    nelm = _to_int(ctx.incar.get("NELM"), 60)
    new_algo = "All" if algo in ("normal", "all") else "Normal"
    if algo == "all":
        new_algo = "Damped"
    return SuggestedAction(
        description=(
            f"The SCF did not converge within NELM={nelm} steps: change ALGO "
            f"({ctx.incar.get('ALGO', 'Normal')} → {new_algo}) and raise NELM. "
            "If it persists, lower AMIX/BMIX (metallic or magnetic systems) "
            "or check MAGMOM/ISPIN."
        ),
        incar_changes={"ALGO": new_algo, "NELM": str(min(max(nelm * 2, 120), 300))},
    )


def _ionic_action(ctx: "_Ctx") -> SuggestedAction:
    ibrion = _to_int(ctx.incar.get("IBRION"), 2)
    changes = {"IBRION": "1", "POTIM": "0.2"} if ibrion != 1 else {"IBRION": "2", "POTIM": "0.1"}
    return SuggestedAction(
        description=(
            "The ionic steps oscillate: switch the optimizer and reduce POTIM, "
            "restarting from CONTCAR. If it continues, check the starting "
            "structure or adjust EDIFFG."
        ),
        incar_changes=changes,
        restart_from_contcar=True,
    )


def _nsw_action(ctx: "_Ctx") -> SuggestedAction:
    nsw = _to_int(ctx.incar.get("NSW"), 0)
    return SuggestedAction(
        description=(
            f"All NSW={nsw} ionic steps were used without reaching EDIFFG: "
            "restart from CONTCAR with a larger NSW."
        ),
        incar_changes={"NSW": str(max(nsw * 2, 100))} if nsw else {},
        restart_from_contcar=True,
    )


# Step 1/2: scheduler kills (slurmstepd messages) — used when sacct is
# unavailable or did not record the state. Order matters: first match wins.
_SLURM_LOG_RULES: List[_Rule] = [
    _Rule("slurm_time_limit", r"DUE TO TIME LIMIT", CAT_RESOURCES,
          "Slurm killed the job for reaching its time limit.", _time_action),
    _Rule("gpu_out_of_memory",
          r"CUDA error.*out of memory|cudaErrorMemoryAllocation|CUDA_ERROR_OUT_OF_MEMORY",
          CAT_RESOURCES, "The GPU ran out of memory.",
          _act("Reduce the load per GPU (more GPUs/nodes, lower NBANDS/ENCUT) "
               "or use a GPU with more memory.")),
    _Rule("slurm_oom",
          r"oom[-_ ]kill|Out Of Memory|out-of-memory|Exceeded job memory limit",
          CAT_RESOURCES, "The kernel/Slurm killed the job for running out of memory.",
          _mem_action),
    _Rule("slurm_node_fail", r"DUE TO NODE FAILURE|NODE_FAIL", CAT_RESOURCES,
          "A cluster node failed (not a problem with the inputs).",
          _act("Resubmit unchanged; if it happens again on the same node, "
               "exclude it with --exclude.")),
    _Rule("slurm_cancelled", r"\*\*\* JOB \d+ ON \S+ CANCELLED AT", CAT_RESOURCES,
          "The job was cancelled (scancel by the user/administrator, or preemption).",
          _act("No input fix needed; resubmit only if the cancellation was "
               "accidental.")),
]

# Step 2: failures of the batch script / container wrapper itself.
_SCRIPT_RULES: List[_Rule] = [
    _Rule("dos_line_breaks", r"DOS line breaks", CAT_SCRIPT,
          "sbatch rejected the script because of Windows line endings (CRLF).",
          _act("Rewrite submit.sh with Unix line endings (LF).")),
    _Rule("no_elements", r"No elements parsed from POSCAR", CAT_SCRIPT,
          "The script's ELEMENTS check aborted: no elements were read from POSCAR.",
          _act("Regenerate submit.sh from a native VASP POSCAR "
               "(line 6 holding the element symbols).")),
    _Rule("missing_potcar", r"Missing POTCAR for element", CAT_SCRIPT,
          "The script could not find an element's POTCAR under PSEUDO_DIR.",
          _act("Check PSEUDO_DIR and the pseudopotential folder name "
               "(e.g. Ti_pv instead of Ti).")),
    _Rule("gpu_not_detected",
          r"NVIDIA-SMI has failed|No devices were found|no CUDA-capable device"
          r"|nvidia-smi: command not found|Could not find any nv files on this host"
          r"|CUDA driver version is insufficient",
          CAT_SCRIPT, "The GPU is not visible from the job/container.",
          _act("Request the GPU in the script (--gres=gpu:N or --gpus) and use "
               "a GPU partition; run the container with GPU support (--nv).")),
    _Rule("container_error",
          r"^\s*FATAL:|(apptainer|singularity|podman): command not found"
          r"|could not open image|failed to create container|image .*not found",
          CAT_SCRIPT, "The container runtime failed before VASP started.",
          _act("Check the container image path and bind mounts in the run "
               "command (or the site's container wrapper), and that the "
               "container runtime module is loaded.")),
    _Rule("module_error",
          r"Lmod has detected the following error|module: command not found",
          CAT_SCRIPT, "Loading modules in the script failed.",
          _act("Fix the 'module load' commands in the script.")),
    _Rule("launcher_error",
          r"srun: error: Unable to|mpirun was unable to launch|ORTE was unable"
          r"|PMIx.*(error|failed)",
          CAT_SCRIPT, "The MPI launcher could not start the processes.",
          _act("Check the requested ntasks/nodes against the script's MPI command.")),
    _Rule("command_not_found", r"command not found|: Permission denied"
          r"|^\S*(bash|sh|srun)\S*: .*No such file or directory",
          CAT_SCRIPT, "The script tried to run a command or file that does not "
          "exist or is not executable.",
          _act("Check the path of the executable or wrapper script in the run "
               "command and its execute permissions.")),
]

# Steps 3/4: VASP's own error messages. Specific rules before generic ones.
_VASP_RULES: List[_Rule] = [
    _Rule("potcar_mismatch",
          r"number of ions in POSCAR and POTCAR|ERROR: the type of the POTCAR"
          r"|POTCAR.*(not|doesn't) match|Error reading POTCAR|No POTCAR found",
          CAT_INPUTS, "POSCAR and POTCAR are not consistent.",
          _act("Reassemble POTCAR in the same order as the POSCAR element line.")),
    _Rule("incar_read_error", r"Error reading item .* from file INCAR", CAT_INPUTS,
          "VASP could not read a tag in INCAR.",
          _act("Fix the format/value of the reported INCAR tag.")),
    _Rule("ions_too_close", r"distance between some ions is very small", CAT_INPUTS,
          "Some atoms in the structure are too close together.",
          _act("Check/regenerate the structure (POSCAR).")),
    _Rule("zbrent", r"ZBRENT: fatal error", CAT_VASP,
          "ZBRENT: the ionic optimizer's line search failed.",
          _act("Switch to IBRION=1 and restart from CONTCAR; if it persists, "
               "lower EDIFF.", incar_changes={"IBRION": "1"},
               restart_from_contcar=True)),
    _Rule("subspace_not_hermitian", r"Sub-Space-Matrix is not hermitian", CAT_VASP,
          "The sub-space matrix is not hermitian (electronic instability).",
          _act("Set ALGO to Normal (or All) and check LREAL.",
               incar_changes={"ALGO": "Normal"})),
    _Rule("edddav",
          r"Error EDDDAV|EDDDAV: Call to ZHEGV failed|EDDDAV.*not converge",
          CAT_VASP, "The Davidson algorithm (EDDDAV) failed.",
          _act("Switch to ALGO=All.", incar_changes={"ALGO": "All"})),
    _Rule("eddrmm", r"EDDRMM: call to ZHEGV failed", CAT_VASP,
          "RMM-DIIS (EDDRMM) failed.",
          _act("Switch to ALGO=Normal.", incar_changes={"ALGO": "Normal"})),
    _Rule("brmix", r"BRMIX: very serious problems", CAT_VASP,
          "BRMIX: serious problems in the charge-density mixing.",
          _act("Reduce the mixing (AMIX/BMIX) and raise NELM.",
               incar_changes={"AMIX": "0.1", "BMIX": "0.01", "NELM": "200"})),
    _Rule("zpotrf", r"ZPOTRF", CAT_VASP,
          "LAPACK ZPOTRF failed (basis could not be orthonormalized; geometry "
          "or ionic step).",
          _act("Reduce POTIM, check the geometry and try ISYM=0.",
               incar_changes={"POTIM": "0.1"})),
    _Rule("symmetry",
          r"internal error in subroutine (SGRCON|IBZKPT|PRICEL|POSMAP|RHOSYG)"
          r"|inverse of rotation matrix was not found|non-integer element in rotation matrix",
          CAT_VASP, "Symmetry error in VASP.",
          _act("Turn symmetry off (ISYM=0) or adjust SYMPREC; use a "
               "Gamma-centred mesh.", incar_changes={"ISYM": "0"})),
    _Rule("rspher", r"ERROR RSPHER|Therefore set LREAL=\.FALSE\.", CAT_VASP,
          "The real-space projection is unstable.",
          _act("Use LREAL=.FALSE.", incar_changes={"LREAL": ".FALSE."})),
    _Rule("nbands", r"highest band is occupied|no unoccupied bands", CAT_VASP,
          "Not enough empty bands.",
          _act("Increase NBANDS (≈ +50 %).")),
    _Rule("tetrahedron", r"Tetrahedron\s+method\s+fails|DENTET|BZINTS", CAT_VASP,
          "The tetrahedron method failed (too few k-points).",
          _act("ISMEAR=-5 (tetrahedron) needs at least 4 irreducible k-points and "
               "this KPOINTS mesh gives fewer. Use Gaussian smearing (ISMEAR=0, "
               "SIGMA=0.05), or keep ISMEAR=-5 with a denser k-point mesh.",
               incar_changes={"ISMEAR": "0", "SIGMA": "0.05"})),
    _Rule("npar_divide", r"M_divide: can not subdivide", CAT_VASP,
          "NPAR/NCORE is not compatible with the number of MPI processes.",
          _act("Remove NPAR/NCORE or change ntasks so it divides evenly.")),
    _Rule("killed_signal_9", r"KILLED BY SIGNAL: 9", CAT_RESOURCES,
          "A VASP process was killed with SIGKILL (usually out of memory).",
          _mem_action),
    _Rule("segfault", r"forrtl: severe \(174\)|SIGSEGV|segmentation fault", CAT_VASP,
          "VASP ended with a segmentation fault.",
          _act("Usually memory/stack: add 'ulimit -s unlimited' to the script, "
               "request more memory or more nodes.")),
    _Rule("mpi_abort", r"MPI_ABORT was invoked|Fatal error in PMPI"
          r"|BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES",
          CAT_VASP, "VASP aborted through MPI without a more specific message.",
          _act("Read VASP's full error block printed before the abort.")),
]

# Keywords used to pre-filter OUTCAR remotely with ``grep -F`` so we never
# pull the whole file; the Python regexes above then classify the hits.
_OUTCAR_KEYWORDS = [
    "ZBRENT", "EDDDAV", "EDDRMM", "Sub-Space-Matrix", "BRMIX", "ZPOTRF",
    "internal error in subroutine", "rotation matrix", "RSPHER", "LREAL",
    "highest band", "unoccupied bands", "Tetrahedron", "DENTET", "M_divide",
    "number of ions", "distance between some ions", "Error reading item",
    "VERY BAD NEWS", "I REFUSE TO CONTINUE", "internal error in:",
]

_NORMAL_END_MARKERS = (
    "General timing and accounting informations for this job",
    "Voluntary context switches",
)


# ── VASP fatal error boxes ────────────────────────────────────
#
# VASP 6 prints fatal errors inside an ASCII box that ends with
# "I REFUSE TO CONTINUE WITH THIS SICK JOB" (ERROR box) or starts with
# "internal error in:" (BUG box). WARNING boxes share the same border but
# the run continues, so boxes are only picked up through these markers.

_FATAL_MARKERS = ("I REFUSE TO CONTINUE WITH THIS SICK JOB", "internal error in:",
                  "VERY BAD NEWS")
_BOX_BORDER = re.compile(r"^\s*-{10,}\s*$")
_BANNER_CHARS = set("ERO#_|()/\\<> ")
_BOX_NOISE = ("I REFUSE TO CONTINUE", "If you are not a developer",
              "Please submit a bug report", "internal error in:")


def _box_inner(line: str) -> str:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return s.strip()


def _is_banner(inner: str) -> bool:
    """The big ASCII-art "ERROR" / "BUG" lettering inside a box."""
    return len(inner) >= 15 and set(inner) <= _BANNER_CHARS


def extract_error_boxes(text: str) -> List[List[str]]:
    """Return every fatal VASP error block in ``text``, one list of lines per
    block: the box borders and message lines verbatim, without the ASCII-art
    banner and empty ``|   |`` rows. Unboxed VASP 5 messages ("VERY BAD
    NEWS! …") yield the marker line with a few lines of context."""
    lines = text.splitlines()
    boxes: List[List[str]] = []
    seen: set = set()
    for i, ln in enumerate(lines):
        if not any(m.lower() in ln.lower() for m in _FATAL_MARKERS):
            continue
        top = next((j for j in range(i - 1, max(-1, i - 80), -1)
                    if _BOX_BORDER.match(lines[j])), None)
        bottom = next((j for j in range(i + 1, min(len(lines), i + 40))
                       if _BOX_BORDER.match(lines[j])), None)
        boxed = top is not None and bottom is not None \
            and all(_box_inner(x) != x.strip() or not x.strip()
                    for x in lines[top + 1:bottom])
        start, end = (top, bottom) if boxed else (max(0, i - 2), min(len(lines), i + 4) - 1)
        if start in seen:
            continue
        seen.add(start)
        block = []
        for raw in lines[start:end + 1]:
            if _BOX_BORDER.match(raw):
                block.append(raw.rstrip())
                continue
            inner = _box_inner(raw)
            if inner and not _is_banner(inner):
                block.append(raw.rstrip())
        boxes.append(block)
    return boxes


def _box_text(block: List[str]) -> str:
    """All inner text of an error block on one line (whitespace collapsed)."""
    return " ".join(" ".join(_box_inner(ln) for ln in block
                             if not _BOX_BORDER.match(ln)).split())


def _box_message(block: List[str]) -> str:
    """The informative text of an error block, for the one-line summary."""
    msg = [_box_inner(ln) for ln in block
           if not _BOX_BORDER.match(ln) and _box_inner(ln)
           and not any(n.lower() in ln.lower() for n in _BOX_NOISE)]
    text = " ".join(msg) or " ".join(_box_inner(ln) for ln in block if _box_inner(ln))
    return (text[:240] + "…") if len(text) > 240 else text


# ── context / remote access ───────────────────────────────────

def _q(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


class _Ctx:
    """Mutable state for one cascade run: the connection, caches of what has
    already been fetched, and the diagnosis being built."""

    def __init__(self, conn, diag: JobDiagnosis, timeout: float):
        self.conn = conn
        self.diag = diag
        self.timeout = timeout
        self.commands: List[str] = []
        self.slurm: Dict[str, Any] = diag.slurm
        self._incar: Optional[Dict[str, str]] = None
        self._incar_text: Optional[str] = None
        self.listing: Optional[List[str]] = None
        self.texts: Dict[str, str] = {}

    # remote primitives ------------------------------------------------
    def run(self, cmd: str) -> Tuple[str, str, int]:
        self.commands.append(cmd)
        try:
            return self.conn.run(cmd, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001 — a dead link is a finding, not a crash
            return "", f"{type(exc).__name__}: {exc}", 255

    def path(self, name: str) -> str:
        if name.startswith("/"):
            return name
        return posixpath.join(self.diag.work_dir, name)

    def exists(self, name: str) -> Optional[bool]:
        if self.listing is None:
            return None
        return posixpath.basename(name) in self.listing if not name.startswith("/") \
            or posixpath.dirname(name) == self.diag.work_dir.rstrip("/") else None

    def tail(self, name: str, n: int) -> Optional[str]:
        p = _q(self.path(name))
        out, _, rc = self.run(
            f"if [ -f {p} ]; then tail -n {int(n)} {p}; else echo {_MISSING}; fi"
        )
        if out.strip() == _MISSING or (rc != 0 and not out):
            return None
        return out

    def head(self, name: str, n: int) -> Optional[str]:
        p = _q(self.path(name))
        out, _, rc = self.run(
            f"if [ -f {p} ]; then head -n {int(n)} {p}; else echo {_MISSING}; fi"
        )
        if out.strip() == _MISSING or (rc != 0 and not out):
            return None
        return out

    def tail_and_grep(self, name: str, n: int, keywords: List[str],
                      max_hits: int = 30) -> Optional[Tuple[str, str]]:
        """One round trip: last ``n`` lines plus the last ``max_hits`` lines
        of the whole file matching any fixed-string keyword."""
        p = _q(self.path(name))
        kw = " ".join(f"-e {_q(k)}" for k in keywords)
        out, _, rc = self.run(
            f"if [ -f {p} ]; then tail -n {int(n)} {p}; echo {_SPLIT}; "
            f"grep -F -i {kw} {p} | tail -n {int(max_hits)}; "
            f"else echo {_MISSING}; fi"
        )
        if out.strip() == _MISSING or (rc != 0 and not out):
            return None
        tail_part, _, grep_part = out.partition(_SPLIT)
        return tail_part, grep_part

    # INCAR, fetched once and shared by steps 4-6 --------------------------
    @property
    def incar(self) -> Dict[str, str]:
        if self._incar is None:
            self._incar_text = self.head("INCAR", _INCAR_HEAD) or ""
            self._incar = parse_incar(self._incar_text)
        return self._incar

    @property
    def incar_text(self) -> str:
        _ = self.incar
        return self._incar_text or ""

    def is_relaxation(self) -> bool:
        return _to_int(self.incar.get("NSW"), 0) > 0 and \
            _to_int(self.incar.get("IBRION"), -1) in (1, 2, 3)

    # diagnosis helpers ------------------------------------------------
    def check(self, step: int, name: str, method: str,
              exists: Optional[bool], note: str = "") -> None:
        self.diag.files_checked.append(FileCheck(step, name, method, exists, note))

    def conclude(self, step: int, category: str, cause: str, summary: str,
                 actions: List[SuggestedAction]) -> JobDiagnosis:
        self.diag.category = category
        self.diag.cause = cause
        self.diag.summary = summary
        self.diag.stopped_at_step = step
        self.diag.suggested_actions = actions
        self.diag.failed = True
        return self.diag


def _apply_rules(ctx: _Ctx, step: int, fname: str, text: str,
                 rules: List[_Rule]) -> Optional[_Rule]:
    """Scan ``text`` with ``rules``. Records evidence for every matching rule
    and returns the highest-priority match (or None). When a hit sits inside
    a VASP error box, the whole box is the evidence, not just the hit line."""
    first: Optional[_Rule] = None
    boxes: Optional[List[List[str]]] = None
    for rule in rules:
        hits = rule.search(text)
        if boxes is None and (hits or any(m.lower() in text.lower()
                                          for m in _FATAL_MARKERS)):
            boxes = extract_error_boxes(text)
        if hits:
            hit_keys = {h.strip() for h in hits}
            box = next((b for b in boxes or [] if hit_keys & {ln.strip() for ln in b}),
                       None)
        else:
            # VASP wraps long messages across box rows ("Tetrahedron" /
            # "method fails"), so also match against each box's joined text.
            box = next((b for b in boxes or [] if rule.search(_box_text(b))), None)
            if box is None:
                continue
        lines = box if box is not None else _dedupe(hits)[:8]
        ctx.diag.evidence.append(Evidence(step, fname, lines, rule.rule_id))
        if first is None:
            first = rule
    return first


def _dedupe(lines: List[str]) -> List[str]:
    seen, out = set(), []
    for ln in lines:
        key = ln.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(ln)
    return out


def _conclude_rule(ctx: _Ctx, step: int, rule: _Rule, where: str) -> JobDiagnosis:
    return ctx.conclude(step, rule.category, rule.rule_id,
                        f"{rule.summary} (evidence in {where})",
                        [rule.action(ctx)])


def _conclude_box(ctx: _Ctx, step: int, fname: str, text: str) -> Optional[JobDiagnosis]:
    """Fallback when no specific rule matched: a fatal VASP error block is
    still conclusive — report it verbatim as a VASP error."""
    boxes = extract_error_boxes(text)
    if not boxes:
        return None
    block = boxes[-1]
    ctx.diag.evidence.append(Evidence(step, fname, block, "vasp_error_box"))
    return ctx.conclude(
        step, CAT_VASP, "vasp_fatal_error",
        f"VASP stopped with a fatal error: {_box_message(block)} (evidence in {fname})",
        [SuggestedAction(
            "No built-in rule covers this message. SciLink's AI can interpret it "
            "and propose corrected inputs for your approval (automatic in the "
            "wizard's Monitor panel); the full VASP error usually names the tag "
            "or file involved.")])


# ── steps ─────────────────────────────────────────────────────

_SACCT_FIELDS = "JobID,State,ExitCode,Elapsed,MaxRSS,Timelimit,ReqMem"


def _step1_sacct(ctx: _Ctx, fallback_state: str,
                 fallback_exit: Optional[int]) -> Optional[JobDiagnosis]:
    jid = ctx.diag.job_id
    cmd = f"sacct -j {_q(jid)} -P --noheader --format={_SACCT_FIELDS}"
    out, err, rc = ctx.run(cmd)
    rows = parse_sacct(out) if rc == 0 else []
    ctx.check(1, f"sacct -j {jid}", "sacct", bool(rows),
              "" if rows else f"sacct returned no data ({(err or 'empty').strip()[:120]})")
    if rows:
        main = next((r for r in rows if r["job_id"] == jid), rows[0])
        states = [r["state"] for r in rows]
        max_rss = max((r["max_rss"] for r in rows if r["max_rss"]),
                      key=lambda v: parse_mem_mb(v) or 0.0, default="")
        ctx.slurm.update({
            "state": main["state"], "raw_state": main["raw_state"],
            "exit_code": main["exit_code"], "elapsed": main["elapsed"],
            "timelimit": main["timelimit"],
            "req_mem": main["req_mem"] or next((r["req_mem"] for r in rows if r["req_mem"]), ""),
            "max_rss": max_rss, "step_states": states,
        })
        evidence_lines = [r["raw"] for r in rows]
    else:
        ctx.slurm.update({"state": (fallback_state or "UNKNOWN").upper(),
                          "exit_code": "" if fallback_exit is None else str(fallback_exit),
                          "source": "scheduler.status (sacct unavailable)"})
        states = [ctx.slurm["state"]]
        evidence_lines = [f"State={ctx.slurm['state']} ExitCode={ctx.slurm['exit_code']}"]

    def _ev():
        ctx.diag.evidence.append(Evidence(1, "sacct", evidence_lines, "sacct_state"))

    if "TIMEOUT" in states:
        _ev()
        return ctx.conclude(
            1, CAT_RESOURCES, "timeout",
            f"The job reached its time limit ({ctx.slurm.get('timelimit') or '?'}) "
            f"after running for {ctx.slurm.get('elapsed') or '?'}.",
            [_time_action(ctx)])
    if "OUT_OF_MEMORY" in states:
        _ev()
        return ctx.conclude(
            1, CAT_RESOURCES, "out_of_memory",
            f"Slurm terminated the job for running out of memory (MaxRSS "
            f"{ctx.slurm.get('max_rss') or '?'}, requested "
            f"{ctx.slurm.get('req_mem') or '?'}).",
            [_mem_action(ctx)])
    if "NODE_FAIL" in states:
        _ev()
        return ctx.conclude(
            1, CAT_RESOURCES, "node_fail",
            "A cluster node failed; this is not a problem with the inputs.",
            [SuggestedAction("Resubmit unchanged; if it happens again, exclude "
                             "the node with --exclude.")])
    main_state = ctx.slurm.get("state", "")
    if main_state == "CANCELLED":
        _ev()
        who = ctx.slurm.get("raw_state", "")
        return ctx.conclude(
            1, CAT_RESOURCES, "cancelled",
            f"The job was cancelled ({who or 'CANCELLED'}): scancel by the user/"
            "administrator, or preemption. This is not a VASP failure.",
            [SuggestedAction("No input fix needed; resubmit only if the "
                             "cancellation was accidental.")])
    if main_state in ("PENDING", "RUNNING", "REQUEUED", "SUSPENDED", "CONFIGURING",
                      "COMPLETING"):
        ctx.diag.failed = False
        ctx.diag.summary = f"The job has not finished yet (state {main_state})."
        ctx.diag.stopped_at_step = 1
        return ctx.diag
    if main_state == "FAILED" or (main_state and main_state != "COMPLETED"):
        ctx.diag.notes.append(
            f"Slurm: {main_state} with ExitCode {ctx.slurm.get('exit_code') or '?'} — "
            "does not explain the cause on its own; checking the logs.")
    return None


def _resolve_logs(ctx: _Ctx, stdout_file: str, stderr_file: str) -> Tuple[str, str]:
    """Pick the batch stdout/stderr paths: the ones the scheduler/wizard
    reported, else ``slurm-<id>.*`` / ``*_<id>.*`` found in the run dir."""
    jid = re.escape(ctx.diag.job_id)
    listing = ctx.listing or []

    def _pick(ext: str) -> str:
        pat = re.compile(rf"(^slurm-{jid}|[-_.]{jid})\.{ext}$")
        hits = sorted(n for n in listing if pat.search(n))
        return hits[0] if hits else ""

    def _norm(p: str) -> str:
        if not p:
            return ""
        return posixpath.basename(p) if posixpath.dirname(p).rstrip("/") == \
            ctx.diag.work_dir.rstrip("/") else p

    err = _norm(stderr_file) or _pick("err")
    out = _norm(stdout_file) or _pick("out")
    # Jobs submitted via the chat tools use --output=vasp.stdout /
    # --error=vasp.stderr, which sacct does not report back.
    if not err and "vasp.stderr" in listing:
        err = "vasp.stderr"
    if not out and "vasp.stdout" in listing:
        out = "vasp.stdout"
    return out, err


_VASP_STDOUT_CANDIDATES = ("vasp.out", "vasp.stdout", "vasp_std.out", "vasp.log", "stdout")


def _step2_script(ctx: _Ctx, out_name: str, err_name: str) -> Optional[JobDiagnosis]:
    for name, label in ((err_name, "batch stderr"), (out_name, "batch stdout")):
        if not name:
            ctx.check(2, f"({label})", "tail", False,
                      "no slurm-<id>/<name>_<id> log found in the directory")
            continue
        text = ctx.tail(name, _LOG_TAIL)
        ctx.check(2, name, f"tail -n {_LOG_TAIL}", text is not None)
        if text is None:
            continue
        ctx.texts[name] = text
        rule = _apply_rules(ctx, 2, name, text, _SLURM_LOG_RULES + _SCRIPT_RULES)
        if rule is not None:
            return _conclude_rule(ctx, 2, rule, name)
    return None


def _step3_vasp_stdout(ctx: _Ctx, out_name: str, err_name: str) -> Optional[JobDiagnosis]:
    listing = ctx.listing or []
    vasp_out = next((c for c in _VASP_STDOUT_CANDIDATES if c in listing), "")
    sources: List[Tuple[str, str]] = []
    if vasp_out and vasp_out == out_name and out_name in ctx.texts:
        # The batch stdout *is* VASP's stdout (e.g. vasp.stdout): reuse it.
        sources.append((out_name, ctx.texts[out_name]))
        ctx.check(3, out_name, "re-scanned with VASP rules", True,
                  "batch stdout = VASP stdout")
    elif vasp_out:
        text = ctx.tail(vasp_out, _LOG_TAIL)
        ctx.check(3, vasp_out, f"tail -n {_LOG_TAIL}", text is not None)
        if text is not None:
            ctx.texts[vasp_out] = text
            sources.append((vasp_out, text))
    else:
        ctx.check(3, "vasp.out", "ls", False,
                  "no separate VASP stdout; when the run command does not "
                  "redirect it (e.g. a container wrapper), VASP writes to the "
                  "batch stdout")
        if out_name in ctx.texts:
            sources.append((out_name, ctx.texts[out_name]))
            ctx.check(3, out_name, "re-scanned with VASP rules", True,
                      "used as VASP stdout")
    # VASP/Fortran crashes (forrtl, SIGSEGV) land on stderr.
    if err_name in ctx.texts:
        sources.append((err_name, ctx.texts[err_name]))
    for name, text in sources:
        rule = _apply_rules(ctx, 3, name, text, _VASP_RULES)
        if rule is not None:
            return _conclude_rule(ctx, 3, rule, name)
    for name, text in sources:
        if _conclude_box(ctx, 3, name, text) is not None:
            return ctx.diag
    return None


def _step4_outcar(ctx: _Ctx) -> Optional[JobDiagnosis]:
    res = ctx.tail_and_grep("OUTCAR", _OUTCAR_TAIL, _OUTCAR_KEYWORDS)
    ctx.check(4, "OUTCAR", f"tail -n {_OUTCAR_TAIL} + grep -F", res is not None)
    if res is None:
        ctx.diag.notes.append(
            "No OUTCAR: VASP never started (or writes to another directory).")
        ctx.slurm.setdefault("outcar", "missing")
        return None
    tail_text, grep_text = res
    rule = _apply_rules(ctx, 4, "OUTCAR", tail_text + "\n" + grep_text, _VASP_RULES)
    normal = any(m in tail_text for m in _NORMAL_END_MARKERS)
    ctx.slurm["outcar"] = "normal_end" if normal else "truncated"
    if rule is not None:
        return _conclude_rule(ctx, 4, rule, "OUTCAR")
    # The fatal box sits at the very end of OUTCAR, so it is in the tail;
    # the grep hits are the fallback when it is not.
    if _conclude_box(ctx, 4, "OUTCAR", tail_text) is not None \
            or _conclude_box(ctx, 4, "OUTCAR", grep_text) is not None:
        return ctx.diag
    if normal:
        ctx.diag.notes.append("OUTCAR ended normally (final timing block present).")
    else:
        last = [ln for ln in tail_text.splitlines() if ln.strip()][-10:]
        ctx.diag.evidence.append(Evidence(4, "OUTCAR", last, "outcar_truncated"))
        ctx.diag.notes.append(
            "OUTCAR stops mid-run (no final timing block).")
    return None


_OSZ_ELEC = re.compile(r"^\s*(DAV|RMM|CG|SDA|ELF|DMP|DIA|RMS)\s*:\s*(\d+)", re.I)
_OSZ_IONIC = re.compile(r"^\s*(\d+)\s+F=\s*(\S+)\s+E0=\s*(\S+)(?:\s+d\s*E\s*=\s*(\S+))?")


def parse_oszicar(text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Split OSZICAR text into ionic steps.

    Returns ``(steps, trailing)`` where each step is ``{"n", "F", "E0", "dE",
    "n_elec", "last_elec_line", "line"}`` and ``trailing`` holds electronic
    lines after the last ionic line (the step the run died in).
    """
    steps: List[Dict[str, Any]] = []
    block: List[str] = []
    for line in text.splitlines():
        if _OSZ_ELEC.match(line):
            block.append(line.rstrip())
            continue
        m = _OSZ_IONIC.match(line)
        if m:
            n_elec = max((int(_OSZ_ELEC.match(b).group(2)) for b in block), default=0)
            steps.append({
                "n": int(m.group(1)), "F": _to_float(m.group(2)),
                "E0": _to_float(m.group(3)), "dE": _to_float(m.group(4)),
                "n_elec": n_elec, "last_elec_line": block[-1] if block else "",
                "line": line.rstrip(),
            })
            block = []
    return steps, block


def _step5_oszicar(ctx: _Ctx) -> Optional[JobDiagnosis]:
    text = ctx.tail("OSZICAR", _OSZICAR_TAIL)
    ctx.check(5, "OSZICAR", f"tail -n {_OSZICAR_TAIL}", text is not None)
    if text is None:
        return None
    nelm = _to_int(ctx.incar.get("NELM"), 60)
    nsw = _to_int(ctx.incar.get("NSW"), 0)
    steps, trailing = parse_oszicar(text)

    # (a) died inside an SCF loop that already hit NELM
    if trailing:
        n_trail = int(_OSZ_ELEC.match(trailing[-1]).group(2))
        if n_trail >= nelm:
            ctx.diag.evidence.append(Evidence(5, "OSZICAR", trailing[-2:], "scf_nelm"))
            return ctx.conclude(
                5, CAT_CONVERGENCE, "scf_not_converged",
                f"The SCF of the last ionic step reached NELM={nelm} without "
                "converging and the run ended there.", [_scf_action(ctx)])
    # (b) completed ionic steps whose SCF stopped at NELM (unconverged)
    capped = [s for s in steps if s["n_elec"] >= nelm]
    if capped:
        lines = []
        for s in capped[-3:]:
            lines += [s["last_elec_line"], s["line"]]
        ctx.diag.evidence.append(Evidence(5, "OSZICAR", lines, "scf_nelm"))
        return ctx.conclude(
            5, CAT_CONVERGENCE, "scf_not_converged",
            f"{len(capped)} ionic step(s) ended their SCF at NELM={nelm} "
            "(not converged).", [_scf_action(ctx)])
    # (c) oscillating ionic energies
    if len(steps) >= 6:
        e0 = [s["E0"] for s in steps[-7:] if s["E0"] is not None]
        diffs = [b - a for a, b in zip(e0, e0[1:])]
        signs = [d > 0 for d in diffs if abs(d) > 1e-6]
        flips = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
        if len(diffs) >= 5 and flips >= 3:
            early = max(abs(d) for d in diffs[: len(diffs) // 2])
            late = max(abs(d) for d in diffs[len(diffs) // 2:])
            if late >= 0.5 * early:
                ctx.diag.evidence.append(
                    Evidence(5, "OSZICAR", [s["line"] for s in steps[-6:]],
                             "ionic_oscillation"))
                return ctx.conclude(
                    5, CAT_CONVERGENCE, "ionic_oscillation",
                    "The ionic-step energy oscillates instead of decreasing: "
                    "the relaxation is not converging.", [_ionic_action(ctx)])
    # (d) NSW exhausted without "reached required accuracy"
    if nsw > 1 and steps and steps[-1]["n"] >= nsw:
        reached = any("reached required accuracy" in t for t in ctx.texts.values())
        if not reached:
            ctx.diag.evidence.append(Evidence(5, "OSZICAR", [steps[-1]["line"]], "nsw_reached"))
            return ctx.conclude(
                5, CAT_CONVERGENCE, "nsw_reached",
                f"All NSW={nsw} ionic steps completed without reaching the "
                "required accuracy.", [_nsw_action(ctx)])
    if steps:
        ctx.diag.notes.append(
            f"OSZICAR: {len(steps)} ionic step(s) in the checked tail; "
            f"max SCF steps {max(s['n_elec'] for s in steps)} < NELM={nelm}.")
    return None


def _step6_inputs(ctx: _Ctx) -> JobDiagnosis:
    issues: List[Tuple[str, List[str], str]] = []  # (file, lines, message)

    poscar = ctx.head("POSCAR", 9)
    ctx.check(6, "POSCAR", "head -n 9", poscar is not None)
    potcar_titles = None
    out, _, rc = ctx.run(
        f"if [ -f {_q(ctx.path('POTCAR'))} ]; then grep -F TITEL {_q(ctx.path('POTCAR'))}; "
        f"else echo {_MISSING}; fi")
    if out.strip() != _MISSING and rc in (0, 1):
        potcar_titles = [ln.strip() for ln in out.splitlines() if ln.strip()]
    ctx.check(6, "POTCAR", "grep -F TITEL", potcar_titles is not None)
    incar_text = ctx.incar_text
    ctx.check(6, "INCAR", f"head -n {_INCAR_HEAD}", bool(incar_text),
              "" if incar_text else "INCAR is missing or empty")
    kpoints = ctx.head("KPOINTS", 6)
    ctx.check(6, "KPOINTS", "head -n 6", kpoints is not None)

    species: List[str] = []
    counts: List[int] = []
    if poscar is None:
        issues.append(("POSCAR", [], "No POSCAR in the job directory."))
    else:
        pl = poscar.splitlines()
        if len(pl) >= 7 and all(t.isalpha() for t in pl[5].split()):
            species = pl[5].split()
            try:
                counts = [int(t) for t in pl[6].split()]
            except ValueError:
                issues.append(("POSCAR", pl[5:7], "The atom-count line is not numeric."))
            if counts and len(counts) != len(species):
                issues.append(("POSCAR", pl[5:7],
                               f"{len(species)} elements but {len(counts)} counts."))
        else:
            issues.append(("POSCAR", pl[4:7],
                           "POSCAR line 6 does not hold element symbols "
                           "(VASP 4 format, or not a POSCAR file)."))

    if potcar_titles is None:
        issues.append(("POTCAR", [], "No POTCAR in the job directory "
                                     "(the script assembles it; if missing, it "
                                     "aborted before that)."))
    elif species:
        pot_el = [potcar_element(t) for t in potcar_titles]
        if pot_el != species:
            issues.append(("POTCAR", potcar_titles,
                           f"POTCAR elements {pot_el} do not match POSCAR "
                           f"{species} (same order required)."))

    inc = ctx.incar
    natoms = sum(counts) if counts else 0
    if inc and natoms:
        if _truthy(inc.get("ISPIN") == "2") and "MAGMOM" in inc:
            n_mag = len(expand_vasp_list(inc["MAGMOM"]))
            if n_mag not in (natoms, 3 * natoms):
                issues.append(("INCAR", [f"MAGMOM = {inc['MAGMOM']}"],
                               f"MAGMOM has {n_mag} values for {natoms} atoms."))
    if inc and species and _truthy(inc.get("LDAU")):
        for tag in ("LDAUL", "LDAUU", "LDAUJ"):
            if tag in inc and len(expand_vasp_list(inc[tag])) != len(species):
                issues.append(("INCAR", [f"{tag} = {inc[tag]}"],
                               f"{tag} needs {len(species)} values (one per species)."))
    if inc and _truthy(inc.get("LHFCALC")) and \
            inc.get("ALGO", "").strip().lower() in ("fast", "veryfast", "very_fast", "f", "v"):
        issues.append(("INCAR", [f"LHFCALC = {inc['LHFCALC']}", f"ALGO = {inc['ALGO']}"],
                       "Hybrid functionals (LHFCALC) do not support ALGO=Fast/VeryFast "
                       "(RMM-DIIS)."))
    if inc and _to_int(inc.get("ICHARG"), 0) >= 10 and _to_int(inc.get("NSW"), 0) > 0 \
            and _to_int(inc.get("IBRION"), -1) != -1:
        issues.append(("INCAR", [f"ICHARG = {inc['ICHARG']}", f"NSW = {inc['NSW']}"],
                       "ICHARG≥10 (fixed, non-self-consistent density) is "
                       "incompatible with ionic relaxation."))
    if inc and _to_int(inc.get("ISMEAR"), 1) == -5 and kpoints:
        mesh = kpoint_mesh(kpoints)
        if mesh and mesh[0] * mesh[1] * mesh[2] < 4:
            issues.append(("INCAR", [f"ISMEAR = {inc['ISMEAR']}", f"KPOINTS mesh {mesh}"],
                           "ISMEAR=-5 (tetrahedron) needs at least 4 k-points."))
    for dup in duplicate_incar_tags(incar_text):
        issues.append(("INCAR", [dup], f"Duplicate INCAR tag: {dup}."))
    for issue in _unknown_incar_tags(incar_text):
        issues.append(("INCAR", [issue], issue))

    if issues:
        for fname, lines, msg in issues:
            ctx.diag.evidence.append(Evidence(6, fname, lines or [msg], "input_check"))
        msgs = "; ".join(m for _, _, m in issues[:4])
        return ctx.conclude(
            6, CAT_INPUTS, "input_inconsistency",
            f"No explicit error in the logs, but the inputs have problems: {msgs}",
            [SuggestedAction("Fix the flagged inputs and resubmit.")])

    state = ctx.slurm.get("state", "")
    exit_code = str(ctx.slurm.get("exit_code", ""))
    if state == "COMPLETED" and exit_code in ("", "0", "0:0") \
            and ctx.slurm.get("outcar") == "normal_end":
        ctx.diag.failed = False
        ctx.diag.category = None
        ctx.diag.cause = "no_failure"
        ctx.diag.stopped_at_step = 6
        ctx.diag.summary = ("No failure detected: Slurm reports COMPLETED, OUTCAR "
                            "ended normally and the inputs are consistent.")
        return ctx.diag
    return ctx.conclude(
        6, CAT_UNKNOWN, "undetermined",
        "Cause undetermined: none of the checked files explains the failure and "
        "the inputs are consistent. " + " ".join(ctx.diag.notes),
        [SuggestedAction(
            "Inspect the full stderr/stdout manually and, if needed, run a short "
            "interactive test (salloc) with the same script.")])


# ── public API ────────────────────────────────────────────────

def diagnose_job(
    conn,
    job_id: str,
    work_dir: str,
    *,
    stdout_file: str = "",
    stderr_file: str = "",
    fallback_state: str = "",
    fallback_exit_code: Optional[int] = None,
    timeout: float = 30,
) -> JobDiagnosis:
    """Run the diagnosis cascade for one job and return the result.

    Args:
        conn: Anything with ``run(cmd, timeout=...) -> (out, err, rc)``
            (an ``HPCConnection`` in production).
        job_id: Scheduler job id.
        work_dir: Remote run directory holding INCAR/OUTCAR/… .
        stdout_file / stderr_file: Batch log paths if known (the wizard and
            ``scontrol`` report them); otherwise they are discovered.
        fallback_state / fallback_exit_code: Scheduler status to use when
            ``sacct`` is unavailable (accounting disabled, no permission).
    """
    diag = JobDiagnosis(job_id=str(job_id), work_dir=work_dir.rstrip("/") or "/")
    ctx = _Ctx(conn, diag, timeout)

    result = _step1_sacct(ctx, fallback_state, fallback_exit_code)
    if result is not None:
        return _finish(ctx)

    out, err, rc = ctx.run(f"ls -1A {_q(diag.work_dir)}")
    if rc != 0:
        # Without the run dir every later step would report "file missing",
        # which reads like an input problem. Say what actually happened.
        ctx.check(2, diag.work_dir, "ls -1A", False, (err or "").strip()[:200])
        ctx.conclude(
            2, CAT_UNKNOWN, "workdir_unreachable",
            f"Cause undetermined: the job directory could not be accessed "
            f"({(err or 'no response').strip()[:200]}).",
            [SuggestedAction("Check the SSH connection and that the remote "
                             "directory still exists (purged scratch?), then "
                             "diagnose again.")])
        return _finish(ctx)
    ctx.listing = [ln.strip() for ln in out.splitlines() if ln.strip()]
    out_name, err_name = _resolve_logs(ctx, stdout_file, stderr_file)

    for step in (lambda: _step2_script(ctx, out_name, err_name),
                 lambda: _step3_vasp_stdout(ctx, out_name, err_name),
                 lambda: _step4_outcar(ctx),
                 lambda: _step5_oszicar(ctx)):
        if step() is not None:
            return _finish(ctx)
    _step6_inputs(ctx)
    return _finish(ctx)


def _finish(ctx: _Ctx) -> JobDiagnosis:
    ctx.diag.slurm.pop("rows", None)
    return ctx.diag


def diagnosis_dir(session_dir: str | Path, job_id: str) -> Path:
    return Path(session_dir) / "diagnostics" / f"job_{job_id}"


def save_diagnosis(diag: JobDiagnosis, session_dir: str | Path) -> Path:
    """Write ``diagnosis.json`` + ``diagnosis.md`` under
    ``<session_dir>/diagnostics/job_<id>/`` and return the JSON path."""
    d = diagnosis_dir(session_dir, diag.job_id)
    d.mkdir(parents=True, exist_ok=True)
    jp = d / "diagnosis.json"
    with open(jp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(diag.to_dict(), fh, indent=2, ensure_ascii=False)
    with open(d / "diagnosis.md", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(diag.to_markdown())
    return jp


def load_diagnosis(session_dir: str | Path, job_id: str) -> Optional[JobDiagnosis]:
    jp = diagnosis_dir(session_dir, job_id) / "diagnosis.json"
    if not jp.exists():
        return None
    return JobDiagnosis.from_dict(json.loads(jp.read_text(encoding="utf-8")))


# ── parsing helpers (public for reuse and tests) ──────────────

def parse_sacct(raw: str) -> List[Dict[str, str]]:
    rows = []
    for line in raw.strip().splitlines():
        p = line.split("|")
        if len(p) < 5:
            continue
        p += [""] * (7 - len(p))
        raw_state = p[1].strip()
        rows.append({
            "job_id": p[0].strip(), "raw_state": raw_state,
            "state": (raw_state.split() or ["UNKNOWN"])[0].rstrip("+"),
            "exit_code": p[2].strip(), "elapsed": p[3].strip(),
            "max_rss": p[4].strip(), "timelimit": p[5].strip(),
            "req_mem": p[6].strip(), "raw": line.strip(),
        })
    return rows


def parse_slurm_time(s: str) -> Optional[int]:
    """``[D-]HH:MM:SS`` / ``MM:SS`` / ``MM`` → seconds; None if unlimited."""
    s = (s or "").strip()
    if not s or not s[0].isdigit():
        return None
    days = 0
    if "-" in s:
        d, _, s = s.partition("-")
        days = int(d)
    parts = [int(x) for x in s.split(":") if x != ""]
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = 0, parts[0], parts[1]
    elif len(parts) == 1:
        h, m, sec = 0, parts[0], 0
    else:
        return None
    return ((days * 24 + h) * 60 + m) * 60 + sec


def format_slurm_time(seconds: int) -> str:
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"


def parse_mem_mb(s: str) -> Optional[float]:
    """``16000M`` / ``16G`` / ``15623400K`` / ``4000Mc`` → MB (per-cpu/node
    suffixes ``c``/``n`` are ignored)."""
    m = re.match(r"^\s*([\d.]+)\s*([KMGT]?)[cn]?\s*$", s or "", re.I)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).upper() or "M"
    return val * {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[unit]


def parse_incar(text: str) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    for raw in (text or "").splitlines():
        line = re.split(r"[#!]", raw, maxsplit=1)[0]
        for stmt in line.split(";"):
            if "=" in stmt:
                k, _, v = stmt.partition("=")
                if k.strip():
                    tags[k.strip().upper()] = v.strip()
    return tags


def duplicate_incar_tags(text: str) -> List[str]:
    seen, dups = set(), []
    for raw in (text or "").splitlines():
        line = re.split(r"[#!]", raw, maxsplit=1)[0]
        for stmt in line.split(";"):
            if "=" in stmt:
                k = stmt.partition("=")[0].strip().upper()
                if k in seen and k not in dups:
                    dups.append(k)
                seen.add(k)
    return dups


def expand_vasp_list(value: str) -> List[str]:
    """Expand VASP's ``N*x`` shorthand: ``"2*5.0 3*0"`` → 5 items."""
    out: List[str] = []
    for tok in value.split():
        if "*" in tok:
            n, _, v = tok.partition("*")
            try:
                out += [v] * int(n)
                continue
            except ValueError:
                pass
        out.append(tok)
    return out


def potcar_element(titel_line: str) -> str:
    """``TITEL  = PAW_PBE Ti_pv 07Sep2000`` → ``Ti``."""
    rhs = titel_line.split("=", 1)[-1].split()
    token = rhs[1] if len(rhs) > 1 else (rhs[0] if rhs else "")
    return re.split(r"[_.]", token)[0]


def kpoint_mesh(kpoints_text: str) -> Optional[Tuple[int, int, int]]:
    lines = kpoints_text.splitlines()
    if len(lines) < 4:
        return None
    try:
        if int(lines[1].split()[0]) != 0:
            return None  # explicit list, not an automatic mesh
        vals = [int(x) for x in lines[3].split()[:3]]
    except (ValueError, IndexError):
        return None
    return (vals[0], vals[1], vals[2]) if len(vals) == 3 else None


def _unknown_incar_tags(incar_text: str) -> List[str]:
    if not incar_text:
        return []
    try:
        from scilink.skills.periodic_dft.vasp.vasp_syntax import check_incar_syntax
        found = check_incar_syntax(incar_text)
    except Exception:  # noqa: BLE001 — optional (needs pymatgen)
        return []
    msgs = []
    for iss in found:
        tag = iss.get("tag")
        if not tag:
            continue
        sug = f" (did you mean {iss['suggested']}?)" if iss.get("suggested") else ""
        msgs.append(f"Unknown INCAR tag: {tag}{sug}")
    return msgs


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v or "").strip().strip(".").upper() in ("TRUE", "T", "1", "YES")


def _to_int(v, default: int) -> int:
    try:
        return int(float(str(v).split()[0]))
    except (TypeError, ValueError, IndexError):
        return default


def _to_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
