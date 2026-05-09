import os
import queue
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from prime_rl.sweep.materialize import (
    TrialArtifacts,
    read_status_json,
    write_json,
    write_multi_run_output_override,
)
from prime_rl.utils.monitor import SWEEP_METRICS_JSONL_ENV

TrialCompleteCallback = Callable[[TrialArtifacts, int], bool]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_status(artifacts: TrialArtifacts) -> dict:
    return read_status_json(artifacts.status_path)


def _write_status(artifacts: TrialArtifacts, **updates) -> None:
    status = _read_status(artifacts)
    status.update(updates)
    write_json(artifacts.status_path, status)


def _launch_error(exc: OSError) -> str:
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _write_launch_failure_status(
    artifact: TrialArtifacts,
    exc: OSError,
    *,
    finished_at: str | None = None,
) -> None:
    _write_status(
        artifact,
        state="failed",
        finished_at=finished_at or utc_now(),
        returncode=-1,
        objective=None,
        failure_stage="launch",
        error=_launch_error(exc),
    )


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

    Legacy ``final_summary.json`` fallback files are attempt-scoped too. If
    the new attempt never writes metrics, stale summaries from an older run
    must not be mistaken for a fresh objective.
    """
    path = Path(_metrics_jsonl_path(artifact))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    for summary_path in artifact.run_dir.glob("run-*/final_summary.json"):
        summary_path.unlink()


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
        try:
            result = subprocess.run(artifact.command, env=env)
        except OSError as exc:
            if attempts > retry_budget:
                _write_launch_failure_status(artifact, exc)
                return -1
            continue
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
        stop_for_failure = returncode != 0 and not continue_on_failure
        if returncode != 0:
            failures += 1
        if on_trial_complete is not None and on_trial_complete(artifact, returncode):
            break
        if stop_for_failure:
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
            try:
                result = subprocess.run(artifact.command)
            except OSError as exc:
                if attempts > retry_budget:
                    _write_launch_failure_status(artifact, exc)
                    failures += 1
                    if not continue_on_failure:
                        raise SystemExit(1) from exc
                    break
                continue
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

EVICTED_FILENAME = "evicted.txt"


def _mark_inactive_multi_run_dirs_evicted(
    shared_dir: Path,
    active_run_dirs: list[Path],
    *,
    reason: str,
) -> None:
    """Make the trainer ignore stale ``run_*`` dirs outside the current launch.

    ``rl-multi-run`` starts orchestrators only for the explicit ``--runs-dir``
    list, but the trainer's ``MultiRunManager`` discovers every
    ``<shared_dir>/run_*`` directory. Completed runs from earlier Optuna waves
    or stale dirs from an old study must therefore be hidden before a fresh
    launcher invocation, otherwise the trainer can allocate slots for runs
    with no matching orchestrator process.
    """
    if not shared_dir.exists():
        return

    active = {run_dir.resolve() for run_dir in active_run_dirs}
    for run_dir in shared_dir.glob("run_*"):
        if not run_dir.is_dir() or run_dir.resolve() in active:
            continue
        control_dir = run_dir / "control"
        control_dir.mkdir(parents=True, exist_ok=True)
        evicted_path = control_dir / EVICTED_FILENAME
        if not evicted_path.exists():
            evicted_path.write_text(reason + "\n")


def _reset_multi_run_artifact_runtime(artifact: TrialArtifacts) -> None:
    """Clear per-attempt sidecars before launching a multi-run orchestrator."""
    _reset_metrics_jsonl(artifact)
    control_dir = artifact.run_dir / "control"
    (control_dir / EXIT_CODE_FILENAME).unlink(missing_ok=True)
    (control_dir / EVICTED_FILENAME).unlink(missing_ok=True)


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
    output_override_path = write_multi_run_output_override(shared_dir)

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
        per_run_code = _read_orchestrator_exit_code(artifact)
        effective = per_run_code if per_run_code is not None else aggregate_returncode
        if effective == 0:
            effective = -1
        _write_status(artifact, state="pruned", finished_at=finished_at, returncode=effective)
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
    retry_budget: int = 1,
) -> int:
    """Launch one ``rl-multi-run`` invocation that drives every artifact in parallel.

    The trainer's ``MultiRunManager`` discovers the per-trial ``run_*``
    directories under ``shared_dir``; an override TOML pins the trainer's
    ``output_dir`` to ``shared_dir`` so it doesn't fall back to whatever
    directory the user's base TOML carried. Trials run concurrently inside
    one trainer process; the launcher writes ``<run_dir>/control/exit_code``
    per orchestrator, and we reconcile per-trial state from those files
    (instead of marking every trial failed on a non-zero aggregate).

    Runtime failures are not retried: re-running a single failed
    orchestrator without restarting the trainer needs dynamic slot
    replacement (Phase 7c). Launcher spawn failures are different: no
    shared trainer has started yet, so they are retried up to
    ``retry_budget`` before the whole wave is marked failed.
    Phase 5b's ``FileMonitor`` sidecar metrics still work because
    ``rl-multi-run`` injects ``PRIME_RL_SWEEP_METRICS_JSONL`` per
    orchestrator; the controller reads each trial's
    ``<run_dir>/metrics.jsonl`` via ``read_final_summary`` after the
    invocation exits.
    """
    command = build_multi_run_command(artifacts, shared_paths, shared_dir)

    attempts = 0
    while True:
        attempts += 1
        started = utc_now()
        for artifact in artifacts:
            _reset_multi_run_artifact_runtime(artifact)
            _write_status(artifact, state="running", started_at=started, attempts=attempts, gpu_group=None)
        _mark_inactive_multi_run_dirs_evicted(
            shared_dir,
            [artifact.run_dir for artifact in artifacts],
            reason="Inactive run directory is not part of the current sweep wave.",
        )

        try:
            result = subprocess.run(command)
        except OSError as exc:
            if attempts > retry_budget:
                finished = utc_now()
                for artifact in artifacts:
                    _write_launch_failure_status(artifact, exc, finished_at=finished)
                return len(artifacts)
            continue
        break

    finished = utc_now()
    failures = 0
    for artifact in artifacts:
        state = reconcile_multi_run_artifact(
            artifact, aggregate_returncode=result.returncode, finished_at=finished
        )
        if state == "failed":
            failures += 1

    return failures
