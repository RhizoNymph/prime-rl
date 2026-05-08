import json
import os
import queue
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from prime_rl.sweep.materialize import TrialArtifacts, write_json
from prime_rl.utils.monitor import SWEEP_METRICS_JSONL_ENV

TrialCompleteCallback = Callable[[TrialArtifacts, int], bool]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_status(artifacts: TrialArtifacts) -> dict:
    return json.loads(artifacts.status_path.read_text())


def _write_status(artifacts: TrialArtifacts, **updates) -> None:
    status = _read_status(artifacts)
    status.update(updates)
    write_json(artifacts.status_path, status)


def _metrics_jsonl_path(artifact: TrialArtifacts) -> str:
    return (artifact.run_dir / "metrics.jsonl").as_posix()


def _reset_metrics_jsonl(artifact: TrialArtifacts) -> None:
    """Truncate the sidecar metrics file before a fresh attempt.

    FileMonitor opens in append mode, so without truncation a failed
    attempt's later steps would survive into the retry. read_final_summary
    selects the largest reported step, which would then return the failed
    attempt's value instead of the successful retry's value. The pruning
    loop has the same hazard: a stale row from a previous attempt can fire
    should_prune() before the new attempt has reported anything.
    """
    path = Path(_metrics_jsonl_path(artifact))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def _build_env(artifact: TrialArtifacts, gpu_group: list[int] | None) -> dict[str, str]:
    """Inherit the parent env, pin CUDA_VISIBLE_DEVICES, and route the trial's
    step-indexed metrics to the canonical sidecar file the sweep controller
    reads (final objective + intermediate pruning).
    """
    env = os.environ.copy()
    if gpu_group is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in gpu_group)
    env[SWEEP_METRICS_JSONL_ENV] = _metrics_jsonl_path(artifact)
    return env


def _run_with_retries(artifact: TrialArtifacts, gpu_group: list[int] | None, retry_budget: int) -> int:
    """Run a single trial, retrying transient failures up to ``retry_budget`` times.

    Returns the final returncode. Each attempt is recorded as a fresh
    ``running`` transition with the cumulative attempt count and the assigned
    device group in status.json.
    """
    env = _build_env(artifact, gpu_group)
    attempts = 0
    while True:
        attempts += 1
        _reset_metrics_jsonl(artifact)
        _write_status(
            artifact,
            state="running",
            started_at=utc_now(),
            attempts=attempts,
            gpu_group=list(gpu_group) if gpu_group is not None else None,
        )
        result = subprocess.run(artifact.command, env=env)
        if result.returncode == 0:
            _write_status(artifact, state="completed", finished_at=utc_now(), returncode=0)
            return 0
        if attempts > retry_budget:
            _write_status(artifact, state="failed", finished_at=utc_now(), returncode=result.returncode)
            return result.returncode


def _is_completed(artifact: TrialArtifacts) -> bool:
    return _read_status(artifact).get("state") == "completed"


def _is_submitted_or_completed(artifact: TrialArtifacts) -> bool:
    return _read_status(artifact).get("state") in {"completed", "submitted"}


def _run_sequential(
    artifacts: list[TrialArtifacts],
    gpu_group: list[int] | None,
    continue_on_failure: bool,
    retry_budget: int,
    on_trial_complete: TrialCompleteCallback | None,
) -> int:
    failures = 0
    for artifact in artifacts:
        returncode = _run_with_retries(artifact, gpu_group, retry_budget)
        if returncode != 0:
            failures += 1
            if not continue_on_failure:
                raise SystemExit(returncode)
        if on_trial_complete is not None and on_trial_complete(artifact, returncode):
            break
    return failures


def _run_parallel(
    artifacts: list[TrialArtifacts],
    max_parallel: int,
    gpu_groups: list[list[int]],
    continue_on_failure: bool,
    retry_budget: int,
    on_trial_complete: TrialCompleteCallback | None,
) -> int:
    """Run trials concurrently, pinning each to a disjoint GPU group.

    The pool of GPU groups acts as a semaphore: a worker pulls a group before
    launching its subprocess and returns it on completion. This guarantees no
    two parallel trials share a device.
    """
    group_pool: queue.Queue[list[int]] = queue.Queue()
    for group in gpu_groups:
        group_pool.put(group)

    halt = threading.Event()
    failure_lock = threading.Lock()
    failure_count = 0

    def task(artifact: TrialArtifacts) -> None:
        nonlocal failure_count
        if halt.is_set():
            return
        group = group_pool.get()
        try:
            returncode = _run_with_retries(artifact, group, retry_budget)
        finally:
            group_pool.put(group)
        if returncode != 0:
            with failure_lock:
                failure_count += 1
            if not continue_on_failure:
                halt.set()
        if on_trial_complete is not None and on_trial_complete(artifact, returncode):
            halt.set()

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        list(executor.map(task, artifacts))

    return failure_count


def run_trials_locally(
    artifacts: list[TrialArtifacts],
    max_parallel: int = 1,
    gpu_groups: list[list[int]] | None = None,
    continue_on_failure: bool = True,
    retry_budget: int = 1,
    on_trial_complete: TrialCompleteCallback | None = None,
) -> int:
    """Run trials sequentially or in parallel. Returns the failed-trial count.

    Trials whose status.json already records ``state == "completed"`` are
    skipped so ``--resume`` only re-runs unfinished work. For parallel runs
    the caller must pass ``gpu_groups`` with at least ``max_parallel`` disjoint
    device groups; this is validated upstream by ``LocalSweepSchedulerConfig``.
    The optional ``on_trial_complete`` callback runs after each completed
    trial; returning True from it halts new submissions while in-flight
    trials finish.
    """
    pending = [artifact for artifact in artifacts if not _is_completed(artifact)]

    if max_parallel == 1:
        single_group = gpu_groups[0] if gpu_groups else None
        return _run_sequential(pending, single_group, continue_on_failure, retry_budget, on_trial_complete)

    if gpu_groups is None or len(gpu_groups) < max_parallel:
        raise ValueError(
            f"Parallel local scheduler requires gpu_groups with at least max_parallel={max_parallel} "
            f"entries (got {0 if gpu_groups is None else len(gpu_groups)})."
        )

    return _run_parallel(
        pending,
        max_parallel,
        gpu_groups[:max_parallel],
        continue_on_failure,
        retry_budget,
        on_trial_complete,
    )


def compress_array_indices(indices: list[int]) -> str:
    """Compress a sorted list of array indices into a SLURM ``--array`` spec.

    SLURM accepts both individual indices and ``a-b`` ranges, comma-separated:
    ``--array=0-2,5,7-10``. Compressing contiguous runs keeps the scheduler
    spec compact and avoids hitting argv length limits on large sweeps.

    >>> compress_array_indices([0, 1, 2, 5, 7, 8, 9])
    '0-2,5,7-9'
    >>> compress_array_indices([3])
    '3'
    >>> compress_array_indices([])
    ''
    """
    if not indices:
        return ""
    sorted_indices = sorted(set(indices))
    runs: list[tuple[int, int]] = []
    start = prev = sorted_indices[0]
    for idx in sorted_indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


def query_running_array_tasks(array_job_id: str | None) -> set[int]:
    """Return the array task indices still pending or running in SLURM.

    Phase 8 resume: when the prior controller's ``sbatch`` is still
    scheduling/running tasks, those tasks must NOT be re-submitted into a
    fresh array job. We query ``squeue`` once at resume time to learn which
    indices the cluster still owns and skip them.

    Empty array job ID, ``squeue`` failure, or a missing binary all return
    an empty set — the caller falls back to status-only filtering, which
    still treats ``submitted`` tasks as owned by the prior array job to avoid
    duplicate cluster submissions when queue visibility is unavailable.
    Surfacing the squeue error inline would be noisier without making the
    answer better.
    """
    if not array_job_id:
        return set()
    try:
        result = subprocess.run(
            ["squeue", "-j", array_job_id, "-h", "-t", "pending,running", "-o", "%a"],
            capture_output=True,
            text=True,
        )
    except (OSError, FileNotFoundError):
        return set()
    if result.returncode != 0:
        return set()
    indices: set[int] = set()
    for line in (result.stdout or "").splitlines():
        token = line.strip()
        if not token:
            continue
        # %a is usually a single int per row, but SLURM can show ranges
        # (e.g. "3-5") or comma-lists ("3,5,7") for batched array states.
        for piece in token.split(","):
            piece = piece.strip()
            if "-" in piece:
                start_str, _, end_str = piece.partition("-")
                try:
                    start, end = int(start_str), int(end_str)
                except ValueError:
                    continue
                if start <= end:
                    indices.update(range(start, end + 1))
            else:
                try:
                    indices.add(int(piece))
                except ValueError:
                    continue
    return indices


def _read_slurm_block_from_resolved(resolved_path: Path) -> dict:
    """Pull the [slurm] block out of a resolved trial config.

    Phase 8 array submission shares one resource block across the whole
    array; we read it from the first trial since static sweeps can't vary
    slurm.* fields (the validator enforces that).
    """
    import tomli

    with open(resolved_path, "rb") as f:
        resolved = tomli.load(f)
    return resolved.get("slurm", {}) or {}


def _render_array_sbatch(
    *,
    study_dir: Path,
    work_dir: Path,
    array_spec: str,
    slurm_block: dict,
    log_dir: Path,
) -> str:
    """Compose the sbatch script content for the array submission.

    Resource directives are pulled from ``slurm_block`` (the [slurm] block
    of any trial's resolved config). Anything we don't explicitly handle is
    passed through as-is via ``extra_directives`` so users keep their
    cluster-specific knobs.
    """
    directives: list[str] = [f"#SBATCH --array={array_spec}"]
    directives.append(f"#SBATCH --output={log_dir.as_posix()}/array-%A_%a.out")
    directives.append(f"#SBATCH --error={log_dir.as_posix()}/array-%A_%a.err")

    # Common SBATCH directives that map to typical [slurm] config fields.
    name_to_directive = {
        "partition": "--partition",
        "nodes": "--nodes",
        "ntasks": "--ntasks",
        "ntasks_per_node": "--ntasks-per-node",
        "cpus_per_task": "--cpus-per-task",
        "gpus_per_node": "--gpus-per-node",
        "gres": "--gres",
        "mem": "--mem",
        "time": "--time",
        "account": "--account",
        "qos": "--qos",
        "constraint": "--constraint",
        "exclusive": "--exclusive",  # boolean
        "job_name": "--job-name",
    }
    for key, flag in name_to_directive.items():
        if key not in slurm_block:
            continue
        value = slurm_block[key]
        if isinstance(value, bool):
            if value:
                directives.append(f"#SBATCH {flag}")
        else:
            directives.append(f"#SBATCH {flag}={value}")

    extra = slurm_block.get("extra_directives", []) or []
    for line in extra:
        line = str(line).strip()
        if line.startswith("#SBATCH"):
            directives.append(line)
        else:
            directives.append(f"#SBATCH {line}")

    body = (
        "set -euo pipefail\n"
        f'cd "{work_dir.as_posix()}"\n'
        f'exec uv run sweep-array-task "{study_dir.as_posix()}"\n'
    )
    return "#!/bin/bash\n" + "\n".join(directives) + "\n\n" + body


def submit_trials_to_slurm_array(
    artifacts: list[TrialArtifacts],
    *,
    study_dir: Path,
    array_indices: list[int] | None = None,
) -> tuple[str | None, list[int]]:
    """Submit one ``sbatch --array`` job covering the given trial indices.

    ``array_indices`` selects which trial indices to include; defaults to
    every trial in ``artifacts``. Returns ``(array_job_id, submitted_indices)``
    so the caller can record the SLURM job ID at study level.

    On submission failure, returns ``(None, [])`` and writes ``status="failed"``
    to every targeted artifact. The aggregate sweep failure handling lives
    upstream in the controller; this function focuses on the SLURM contract.
    """
    if not artifacts:
        return None, []

    indices = sorted(set(array_indices)) if array_indices is not None else list(range(len(artifacts)))
    if not indices:
        return None, []

    selected = [a for i, a in enumerate(artifacts) if i in set(indices)]
    if not selected:
        return None, []

    array_spec = compress_array_indices(indices)
    log_dir = study_dir / "slurm"
    log_dir.mkdir(parents=True, exist_ok=True)

    slurm_block = _read_slurm_block_from_resolved(selected[0].resolved_path)
    sbatch_text = _render_array_sbatch(
        study_dir=study_dir,
        work_dir=Path.cwd(),
        array_spec=array_spec,
        slurm_block=slurm_block,
        log_dir=log_dir,
    )
    sbatch_path = study_dir / "array.sbatch"
    sbatch_path.write_text(sbatch_text)

    started = utc_now()
    for artifact in selected:
        _write_status(artifact, state="submitting", started_at=started, attempts=1)

    result = subprocess.run(
        ["sbatch", "--parsable", sbatch_path.as_posix()],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        finished = utc_now()
        for artifact in selected:
            _write_status(artifact, state="failed", finished_at=finished, returncode=result.returncode)
        return None, []

    array_job_id = (result.stdout or "").strip().split(";")[0]
    if not array_job_id:
        # sbatch returned 0 but no job id — treat as a malformed submission.
        finished = utc_now()
        for artifact in selected:
            _write_status(artifact, state="failed", finished_at=finished, returncode=-1)
        return None, []

    submitted_at = utc_now()
    for artifact in selected:
        _write_status(
            artifact,
            state="submitted",
            finished_at=submitted_at,
            slurm_job_id=array_job_id,
            returncode=0,
        )
    return array_job_id, indices


def submit_trials_to_slurm(
    artifacts: list[TrialArtifacts],
    continue_on_failure: bool = True,
    retry_budget: int = 1,
) -> int:
    """Submit trials through the target entrypoint's SLURM support.

    The target entrypoint owns SLURM rendering/submission. Throughput is
    governed by the cluster's own scheduling, not this controller, so there
    is no in-flight cap here. Submission failures (not job failures) are
    retried up to ``retry_budget``.
    """
    failures = 0
    for artifact in artifacts:
        if _is_submitted_or_completed(artifact):
            continue
        attempts = 0
        while True:
            attempts += 1
            _write_status(artifact, state="submitting", started_at=utc_now(), attempts=attempts)
            result = subprocess.run(artifact.command)
            if result.returncode == 0:
                _write_status(artifact, state="submitted", finished_at=utc_now(), returncode=0)
                break
            if attempts > retry_budget:
                _write_status(artifact, state="failed", finished_at=utc_now(), returncode=result.returncode)
                failures += 1
                if not continue_on_failure:
                    raise SystemExit(result.returncode)
                break
    return failures


EXIT_CODE_FILENAME = "exit_code"
"""Per-orchestrator returncode written by ``rl-multi-run``; the source of truth
for per-trial failure attribution in multi_run_lora sweeps."""


def _read_orchestrator_exit_code(artifact: TrialArtifacts) -> int | None:
    """Read ``<run_dir>/control/exit_code`` written by the launcher.

    Returns ``None`` when the file is missing — typically the launcher died
    before this orchestrator started, or the file was lost. Callers treat
    ``None`` as an infrastructure failure distinct from a recorded non-zero
    returncode.
    """
    path = artifact.run_dir / "control" / EXIT_CODE_FILENAME
    if not path.exists():
        return None
    raw = path.read_text().strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def build_multi_run_command(
    artifacts: list[TrialArtifacts],
    shared_paths: list[Path],
    shared_dir: Path,
) -> list[str]:
    """Compose the ``rl-multi-run`` invocation for a wave of trials.

    Pulled out so the Optuna wave driver can spawn the same command via
    ``Popen`` (for mid-flight pruning) instead of ``subprocess.run``.
    """
    shared_dir.mkdir(parents=True, exist_ok=True)
    output_override_path = shared_dir / "_output_override.toml"
    output_override_path.write_text(f'output_dir = "{shared_dir.as_posix()}"\n')

    command: list[str] = ["rl-multi-run"]
    for path in shared_paths:
        command.extend(["@", path.as_posix()])
    command.extend(["@", output_override_path.as_posix()])
    command.extend(
        ["--runs-dir", ":".join(artifact.run_dir.as_posix() for artifact in artifacts)]
    )
    return command


def reconcile_multi_run_artifact(
    artifact: TrialArtifacts,
    *,
    aggregate_returncode: int,
    finished_at: str,
) -> str:
    """Reconcile one artifact's status from the launcher's per-run signals.

    Returns the final ``state`` written ("completed", "failed", or "pruned").
    Pre-existing ``state="pruned"`` is preserved verbatim — the controller
    sets it before writing ``evicted.txt`` for that run, and the orchestrator's
    non-zero exit must not flip it to ``failed``.

    When the per-run ``exit_code`` is missing we treat it as an infrastructure
    failure: prefer the aggregate launcher returncode for diagnostics, but
    failing back to ``-1`` if even that is zero (a paradox: the launcher
    exited cleanly but produced no exit_code for this orchestrator).
    """
    status = _read_status(artifact)
    if status.get("state") == "pruned":
        return "pruned"

    per_run_code = _read_orchestrator_exit_code(artifact)
    if per_run_code == 0:
        _write_status(artifact, state="completed", finished_at=finished_at, returncode=0)
        return "completed"

    if per_run_code is None:
        # Launcher died before recording this run's exit code. Pick the
        # aggregate returncode if it carries useful info; -1 as a fallback
        # so the field is never zero on a failure path.
        effective = aggregate_returncode if aggregate_returncode != 0 else -1
    else:
        effective = per_run_code
    _write_status(artifact, state="failed", finished_at=finished_at, returncode=effective)
    return "failed"


def submit_trials_to_multi_run_lora(
    artifacts: list[TrialArtifacts],
    shared_paths: list[Path],
    shared_dir: Path,
    continue_on_failure: bool = True,
) -> int:
    """Launch one ``rl-multi-run`` invocation that drives every artifact in parallel.

    The trainer's ``MultiRunManager`` discovers the per-trial ``run_*``
    directories under ``shared_dir``; an override TOML pins the trainer's
    ``output_dir`` to ``shared_dir`` so it doesn't fall back to whatever
    directory the user's base TOML carried. Trials run concurrently inside
    one trainer process; the launcher writes ``<run_dir>/control/exit_code``
    per orchestrator, and we reconcile per-trial state from those files
    (instead of marking every trial failed on a non-zero aggregate).

    No retry loop: re-running a single failed orchestrator without
    restarting the trainer needs dynamic slot replacement (Phase 7c).
    Phase 5b's ``FileMonitor`` sidecar metrics still work because
    ``rl-multi-run`` injects ``PRIME_RL_SWEEP_METRICS_JSONL`` per
    orchestrator; the controller reads each trial's
    ``<run_dir>/metrics.jsonl`` via ``read_final_summary`` after the
    invocation exits.
    """
    command = build_multi_run_command(artifacts, shared_paths, shared_dir)

    started = utc_now()
    for artifact in artifacts:
        _write_status(artifact, state="running", started_at=started, attempts=1, gpu_group=None)

    result = subprocess.run(command)

    finished = utc_now()
    failures = 0
    for artifact in artifacts:
        state = reconcile_multi_run_artifact(
            artifact, aggregate_returncode=result.returncode, finished_at=finished
        )
        if state == "failed":
            failures += 1

    if failures > 0 and not continue_on_failure:
        raise SystemExit(result.returncode if result.returncode != 0 else 1)
    return failures
