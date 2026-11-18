"""Human-approved resubmission of a failed job.

A :class:`RetryProposal` is built from a :class:`~scilink.hpc.diagnosis.JobDiagnosis`
and *shown* to the user; nothing is sent to the cluster until
:func:`prepare_and_submit_retry` is called with ``approved=True``.

Guarantees:

* The failed run's directory is never modified: every retry goes into a new
  ``retry_N`` subfolder, created with a plain ``mkdir`` (no ``-p``) so an
  existing folder is never reused or overwritten.
* Resource increases are capped by :class:`RetryPolicy`.
* ``RetryPolicy.auto_resubmit`` exists for a future automatic mode but
  :meth:`RetryPolicy.can_auto_resubmit` returns False for now — every retry
  needs explicit approval regardless of the agent's autonomy level.
"""
from __future__ import annotations

import posixpath
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .diagnosis import (
    CAT_CONVERGENCE,
    CAT_INPUTS,
    CAT_SCRIPT,
    CAT_VASP,
    JobDiagnosis,
    format_slurm_time,
    parse_incar,
    parse_mem_mb,
    parse_slurm_time,
)


@dataclass
class RetryPolicy:
    max_retries_per_job: int = 1
    max_time_factor: float = 2.0
    max_time_limit: str = "48:00:00"
    max_mem_factor: float = 2.0
    max_mem_mb: Optional[int] = None
    max_nodes: Optional[int] = None
    # Reserved for a future opt-in automatic mode. Not honoured yet.
    auto_resubmit: bool = False

    def can_auto_resubmit(self) -> bool:
        return False


@dataclass
class RetryProposal:
    job_id: str
    source_dir: str
    retry_root: str
    category: Optional[str]
    cause: str
    incar_changes: Dict[str, str] = field(default_factory=dict)
    resource_changes: Dict[str, str] = field(default_factory=dict)
    restart_from_contcar: bool = False
    # Whole-file replacements (e.g. from the AI fix), limited to
    # ``REPLACEABLE_FILES``. INCAR tag edits still apply on top.
    file_replacements: Dict[str, str] = field(default_factory=dict)
    source: str = "rules"
    attempt: int = 1
    allowed: bool = True
    blocked_reason: str = ""
    notes: List[str] = field(default_factory=list)
    requires_approval: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        parts = []
        for k, v in self.incar_changes.items():
            parts.append(f"INCAR {k} = {v}")
        for k, v in self.resource_changes.items():
            parts.append(f"--{k}={v}")
        for name in self.file_replacements:
            parts.append(f"{name} replaced")
        if self.restart_from_contcar and "POSCAR" not in self.file_replacements:
            parts.append("POSCAR ← CONTCAR")
        return ", ".join(parts) or "resubmit unchanged"

    def has_changes(self) -> bool:
        return bool(self.incar_changes or self.resource_changes
                    or self.file_replacements or self.restart_from_contcar)


# Input files a retry may replace wholesale. Never POTCAR (licensed, and not
# something to regenerate) and never the submit script (only its #SBATCH
# resources change, through the capped path).
REPLACEABLE_FILES = ("INCAR", "KPOINTS", "POSCAR")

# Failures that repeat identically when the same inputs are resubmitted.
_DETERMINISTIC = (CAT_VASP, CAT_INPUTS, CAT_SCRIPT, CAT_CONVERGENCE)


# ── proposal ──────────────────────────────────────────────────

def original_run_dir(work_dir: str) -> str:
    """``…/run/retry_2`` → ``…/run``: retries of a retry stay siblings
    (``retry_3``) under the original run instead of nesting."""
    return re.sub(r"(/retry_\d+)+/?$", "", work_dir.rstrip("/")) or work_dir

def build_retry_proposal(
    diag: JobDiagnosis,
    *,
    attempts_so_far: int = 0,
    policy: Optional[RetryPolicy] = None,
    current_script: str = "",
    retry_root: Optional[str] = None,
    file_replacements: Optional[Dict[str, str]] = None,
    use_rule_actions: bool = True,
    source: str = "rules",
) -> RetryProposal:
    """Merge the diagnosis' suggested actions into one capped proposal.

    ``current_script`` (the failed run's ``submit.sh``) supplies the current
    ``--time`` / ``--mem`` so the cap can be computed against them.
    ``file_replacements`` (whole files, e.g. from the AI fix) are added to
    the proposal; with ``use_rule_actions=False`` the rule-based actions are
    left out so the two fixes are never mixed.
    """
    policy = policy or RetryPolicy()
    prop = RetryProposal(
        job_id=diag.job_id,
        source_dir=diag.work_dir,
        retry_root=retry_root or original_run_dir(diag.work_dir),
        category=diag.category,
        cause=diag.cause,
        attempt=attempts_so_far + 1,
        source=source,
    )
    if not diag.failed:
        prop.allowed = False
        prop.blocked_reason = "The diagnosis did not detect a failure."
        return prop
    if attempts_so_far >= policy.max_retries_per_job:
        prop.allowed = False
        prop.blocked_reason = (
            f"Retry limit reached ({attempts_so_far}/"
            f"{policy.max_retries_per_job}).")

    for act in (diag.suggested_actions if use_rule_actions else []):
        prop.incar_changes.update(act.incar_changes)
        prop.resource_changes.update(act.resource_changes)
        prop.restart_from_contcar |= act.restart_from_contcar
    for name, text in (file_replacements or {}).items():
        if name not in REPLACEABLE_FILES:
            prop.notes.append(f"Ignored a proposed change to {name}: only "
                              f"{', '.join(REPLACEABLE_FILES)} can be replaced.")
            continue
        prop.file_replacements[name] = text if text.endswith("\n") else text + "\n"

    cur_time, cur_mem = current_resources(current_script)
    prop.resource_changes, cap_notes = cap_resources(
        prop.resource_changes, cur_time=cur_time, cur_mem=cur_mem, policy=policy,
        slurm_timelimit=diag.slurm.get("timelimit", ""),
        slurm_req_mem=diag.slurm.get("req_mem", ""))
    prop.notes += cap_notes

    if not prop.has_changes():
        if diag.category in _DETERMINISTIC:
            # Resubmitting identical inputs would fail the same way.
            if prop.allowed:
                prop.allowed = False
                prop.blocked_reason = (
                    "There is no change to apply yet, and resubmitting the same "
                    "inputs would fail the same way. Get a proposed fix from "
                    "SciLink's AI (automatic in the wizard's Monitor panel), or "
                    "correct the inputs yourself.")
        elif diag.cause not in ("node_fail", "cancelled"):
            prop.notes.append(
                "No automatic fix exists for this cause: the retry would resubmit "
                "the same inputs.")
    return prop


def cap_resources(
    changes: Dict[str, str],
    *,
    cur_time: str = "",
    cur_mem: str = "",
    policy: RetryPolicy,
    slurm_timelimit: str = "",
    slurm_req_mem: str = "",
) -> Tuple[Dict[str, str], List[str]]:
    out = dict(changes)
    notes: List[str] = []
    if "time" in out:
        base = parse_slurm_time(cur_time) or parse_slurm_time(slurm_timelimit)
        want = parse_slurm_time(out["time"]) or 0
        limits = [parse_slurm_time(policy.max_time_limit) or want]
        if base:
            limits.append(int(base * policy.max_time_factor))
        capped = min([want] + limits)
        if capped < want:
            notes.append(f"Time capped by the retry policy: {out['time']} → "
                         f"{format_slurm_time(capped)}.")
        out["time"] = format_slurm_time(capped)
    if "mem" in out:
        base = parse_mem_mb(cur_mem) or parse_mem_mb(slurm_req_mem)
        want = parse_mem_mb(out["mem"]) or 0.0
        limits = [want]
        if base:
            limits.append(base * policy.max_mem_factor)
        if policy.max_mem_mb:
            limits.append(float(policy.max_mem_mb))
        capped = min(limits)
        if capped < want:
            notes.append(f"Memory capped by the retry policy: {out['mem']} → "
                         f"{int(capped)}M.")
        out["mem"] = f"{int(capped)}M"
    if "nodes" in out and policy.max_nodes:
        try:
            if int(out["nodes"]) > policy.max_nodes:
                notes.append(f"Nodes capped by the retry policy at {policy.max_nodes}.")
                out["nodes"] = str(policy.max_nodes)
        except ValueError:
            out.pop("nodes")
    return out, notes


# ── text transforms (pure, testable) ──────────────────────────

_SBATCH_RX = r"^(\s*#SBATCH\s+(?:--{opt}[=\s]|-{short}\s+))(\S+)(.*)$"


def current_resources(script: str) -> Tuple[str, str]:
    t = re.search(_SBATCH_RX.format(opt="time", short="t"), script or "", re.M)
    m = re.search(_SBATCH_RX.format(opt="mem", short="__none__"), script or "", re.M)
    return (t.group(2) if t else "", m.group(2) if m else "")


def apply_resource_changes(script: str, changes: Dict[str, str]) -> str:
    """Rewrite ``#SBATCH --time/--mem/--nodes`` lines (adding them after the
    last ``#SBATCH`` line when missing)."""
    lines = script.split("\n")
    short = {"time": "t", "nodes": "N", "mem": "__none__"}
    for opt, val in changes.items():
        rx = re.compile(_SBATCH_RX.format(opt=re.escape(opt), short=short.get(opt, "__none__")))
        done = False
        for i, ln in enumerate(lines):
            m = rx.match(ln)
            if m:
                lines[i] = f"#SBATCH --{opt}={val}"
                done = True
        if not done:
            idx = max((i for i, ln in enumerate(lines) if ln.lstrip().startswith("#SBATCH")),
                      default=0)
            lines.insert(idx + 1, f"#SBATCH --{opt}={val}")
    return "\n".join(lines)


def apply_incar_changes(incar: str, changes: Dict[str, str]) -> str:
    """Set tags in INCAR text: replace the existing assignment, else append.
    Comments and the order of other tags are preserved."""
    lines = incar.rstrip("\n").split("\n") if incar else []
    for tag, val in changes.items():
        rx = re.compile(rf"^(\s*){re.escape(tag)}\s*=\s*[^#!;]*", re.I)
        hit = False
        for i, ln in enumerate(lines):
            if rx.match(ln):
                lines[i] = rx.sub(lambda m: f"{m.group(1)}{tag} = {val}", ln, count=1)
                hit = True
        if not hit:
            lines.append(f"{tag} = {val}")
    return "\n".join(lines) + "\n"


def rewrite_workdir(script: str, old_dir: str, new_dir: str) -> str:
    """Point a submit script at ``new_dir``.

    Rewrites ``cd <old>`` and ``#SBATCH --chdir=<old>``. Refuses (ValueError)
    when the script ``cd``s somewhere else or still references ``old_dir``
    anywhere else afterwards — in that case running it from the new folder
    could write into the failed run's directory.
    """
    old = old_dir.rstrip("/")
    variants = {old, f"'{old}'", f'"{old}"'}

    def _is_old(target: str) -> bool:
        return target.rstrip("/") in variants or target.strip("'\"").rstrip("/") == old

    out_lines = []
    for ln in script.split("\n"):
        s = ln.strip()
        m = re.match(r"^(\s*cd\s+)(\S+)(.*)$", ln)
        if m and not s.startswith("#"):
            if _is_old(m.group(2)):
                ln = f"{m.group(1)}'{new_dir}'{m.group(3)}"
            elif not re.fullmatch(
                    r"[\"']?\$\{?(SLURM_SUBMIT_DIR|PBS_O_WORKDIR|LS_SUBCWD)\}?[\"']?",
                    m.group(2)):
                raise ValueError(f"The script changes to another directory: {s!r}")
        m = re.match(r"^(\s*#SBATCH\s+(?:--chdir[=\s]|-D\s+))(\S+)(.*)$", ln)
        if m and _is_old(m.group(2)):
            ln = f"#SBATCH --chdir={new_dir}{m.group(3)}"
        out_lines.append(ln)
    result = "\n".join(out_lines)
    leftover = result.replace(new_dir, "")
    # ``old`` followed by "/" (a file inside it) counts; ``old`` as the prefix
    # of a sibling directory name (``old2``, ``old.bak``) does not.
    if re.search(rf"{re.escape(old)}(?![\w.-])", leftover):
        raise ValueError(
            "The script still references the original directory; it cannot be "
            "relocated safely.")
    return result


# ── remote side ───────────────────────────────────────────────

def _q(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def next_retry_dir(conn, root: str) -> str:
    """Create and return ``<root>/retry_N`` with N one past the highest
    existing retry. Uses ``mkdir`` without ``-p``: fails if it exists."""
    root = root.rstrip("/")
    out, _, _ = conn.run(f"ls -1d {_q(root)}/retry_* 2>/dev/null", timeout=30)
    nums = [int(m.group(1)) for m in re.finditer(r"retry_(\d+)\s*$", out, re.M)]
    n = max(nums, default=0) + 1
    path = f"{root}/retry_{n}"
    _, err, rc = conn.run(f"mkdir {_q(path)}", timeout=30)
    if rc != 0:
        raise RuntimeError(f"Could not create {path}: {err.strip()}")
    return path


_COPY_FILES = ("KPOINTS", "POTCAR")


def prepare_and_submit_retry(
    conn,
    scheduler,
    proposal: RetryProposal,
    *,
    approved: bool,
    script_name: str = "submit.sh",
) -> Dict[str, Any]:
    """Create ``retry_N`` next to the failed run, write the corrected inputs
    there and submit. The failed run's files are only read.

    Raises:
        PermissionError: when ``approved`` is not True (no remote action is
            taken), or when the proposal is blocked by the retry policy.
    """
    if approved is not True:
        raise PermissionError(
            "A retry requires the user's explicit approval.")
    if not proposal.allowed:
        raise PermissionError(proposal.blocked_reason or "Retry not allowed.")

    src = proposal.source_dir.rstrip("/")

    def _read(name: str) -> Optional[str]:
        p = _q(posixpath.join(src, name))
        out, _, rc = conn.run(f"if [ -f {p} ]; then cat {p}; else echo __NOFILE__; fi",
                              timeout=60)
        if rc != 0 or out.strip() == "__NOFILE__":
            return None
        return out

    script = _read(script_name)
    incar = _read("INCAR")
    if script is None or incar is None:
        raise FileNotFoundError(
            f"{script_name} or INCAR is missing in {src}; cannot prepare the retry.")
    structure_src = "POSCAR"
    replacements = {k: v for k, v in proposal.file_replacements.items()
                    if k in REPLACEABLE_FILES}
    if "POSCAR" in replacements:
        structure_src = "POSCAR (replaced)"
        if proposal.restart_from_contcar:
            proposal.notes.append("The replaced POSCAR is used instead of CONTCAR.")
    elif proposal.restart_from_contcar:
        _, _, rc = conn.run(f"test -s {_q(posixpath.join(src, 'CONTCAR'))}", timeout=30)
        if rc == 0:
            structure_src = "CONTCAR"
        else:
            proposal.notes.append("CONTCAR is empty or missing: reusing POSCAR.")

    resource_changes = dict(proposal.resource_changes)
    if resource_changes and getattr(scheduler, "name", "SLURM").upper() != "SLURM":
        proposal.notes.append(
            "Resource changes skipped: only #SBATCH directives are rewritten.")
        resource_changes = {}
    # Validate the script before creating anything on the cluster.
    rewrite_workdir(script, src, posixpath.join(proposal.retry_root, "retry_0"))

    new_dir = next_retry_dir(conn, proposal.retry_root)
    new_script = rewrite_workdir(script, src, new_dir)
    new_script = apply_resource_changes(new_script, resource_changes)
    new_incar = apply_incar_changes(replacements.get("INCAR", incar),
                                    proposal.incar_changes)

    conn.upload_text(new_incar, f"{new_dir}/INCAR")
    conn.upload_text(new_script.replace("\r\n", "\n"), f"{new_dir}/{script_name}")
    for name in ("KPOINTS", "POSCAR"):
        if name in replacements:
            conn.upload_text(replacements[name].replace("\r\n", "\n"),
                             f"{new_dir}/{name}")
    cmds = []
    if "POSCAR" not in replacements:
        cmds.append(f"cp {_q(posixpath.join(src, structure_src))} {_q(new_dir + '/POSCAR')}")
    for f in _COPY_FILES:
        if f in replacements:
            continue
        s = _q(posixpath.join(src, f))
        cmds.append(f"if [ -f {s} ]; then cp {s} {_q(new_dir + '/')}; fi")
    _, err, rc = conn.run(" && ".join(cmds), timeout=120)
    if rc != 0:
        raise RuntimeError(f"Copying the inputs to {new_dir} failed: {err.strip()}")

    job_id = scheduler.submit(f"{new_dir}/{script_name}", work_dir=new_dir)
    return {
        "job_id": job_id,
        "work_dir": new_dir,
        "structure_from": structure_src,
        "incar_changes": dict(proposal.incar_changes),
        "resource_changes": resource_changes,
        "files_replaced": sorted(replacements),
        "source": proposal.source,
        "attempt": proposal.attempt,
        "parent_job_id": proposal.job_id,
    }


def file_diff(old: str, new: str, name: str) -> str:
    """Unified diff of one replaced input file, for display."""
    import difflib
    return "".join(difflib.unified_diff(
        (old or "").replace("\r\n", "\n").splitlines(keepends=True),
        new.replace("\r\n", "\n").splitlines(keepends=True),
        fromfile=f"{name} (failed run)", tofile=f"{name} (retry)", n=2))


def incar_diff(old: str, changes: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """``[(tag, old_value_or_'—', new_value)]`` for display."""
    cur = parse_incar(old)
    return [(k, cur.get(k.upper(), "—"), v) for k, v in changes.items()]
