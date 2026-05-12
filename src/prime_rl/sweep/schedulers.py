import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from prime_rl.sweep.materialize import (
    TrialArtifacts,
    read_status_json,
    record_trial_pruned,
    write_json,
    write_multi_run_output_override,
)
from prime_rl.sweep.metrics import read_final_summary, read_intermediate_metric
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
    synchronous: bool = False,
    on_trial_complete: TrialCompleteCallback | None = None,
) -> int:
    """Submit trials through the target entrypoint's SLURM support.

    The target entrypoint owns SLURM rendering/submission. Throughput is
    governed by the cluster's own scheduling, not this controller, so there
    is no in-flight cap here. Submission failures (not job failures) are
    retried up to ``retry_budget``.

    When ``synchronous=True``, each trial is submitted via ``sbatch --wait``
    and the controller blocks until that job exits. The trial state moves
    pending -> running -> completed/failed, matching the local scheduler's
    contract, so Optuna and early stopping can observe per-trial outcomes.
    The ``on_trial_complete`` callback fires after each trial finishes and
    can halt new submissions (used for trial-level early stopping).
    """
    if synchronous:
        return _submit_trials_to_slurm_sync(
            artifacts,
            continue_on_failure=continue_on_failure,
            retry_budget=retry_budget,
            on_trial_complete=on_trial_complete,
        )

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


def _slurm_script_path(artifact: TrialArtifacts) -> Path:
    """Return the sbatch script the entrypoint's ``--dry-run`` materializes.

    Each entrypoint writes ``<run_dir>/<entrypoint>.sbatch`` (``rl.sbatch``,
    ``sft.sbatch``), not a fixed filename — so the synchronous SLURM path
    must derive the script name from the trial's command rather than
    hard-coding ``rl.sbatch``. ``artifact.command`` is shaped as
    ``["uv", "run", "<entrypoint>", ...]``.
    """
    entrypoint = artifact.command[2]
    return artifact.run_dir / f"{entrypoint}.sbatch"


def _run_with_retries_slurm_sync(artifact: TrialArtifacts, retry_budget: int) -> int:
    """Run one trial under SLURM, blocking until the job exits.

    Two-step: (1) ``uv run rl ... --dry-run`` renders the sbatch script,
    (2) ``sbatch --wait <script>`` submits and blocks. The retry budget
    covers both submission and job failures; transient cluster hiccups
    (sbatch returning non-zero, queue backpressure) get one more shot
    before the trial is marked failed.
    """
    env = _build_env(artifact, gpu_group=None)
    attempts = 0
    while True:
        attempts += 1
        _reset_metrics_jsonl(artifact)
        _write_status(
            artifact,
            state="running",
            started_at=utc_now(),
            attempts=attempts,
            gpu_group=None,
        )
        try:
            dryrun = subprocess.run(artifact.command + ["--dry-run"], env=env)
        except OSError as exc:
            if attempts > retry_budget:
                _write_launch_failure_status(artifact, exc)
                return -1
            continue
        if dryrun.returncode != 0:
            if attempts > retry_budget:
                _write_status(
                    artifact, state="failed", finished_at=utc_now(), returncode=dryrun.returncode
                )
                return dryrun.returncode
            continue

        script_path = _slurm_script_path(artifact)
        if not script_path.exists():
            # Should be unreachable when --dry-run returns 0, but defend so
            # we surface a clear error rather than crash on FileNotFoundError.
            _write_status(
                artifact,
                state="failed",
                finished_at=utc_now(),
                returncode=-1,
                failure_stage="materialization",
                error=f"sbatch script missing after --dry-run: {script_path}",
            )
            return -1

        # The dry-run prints its own "Dry run complete" message, which reads
        # like the sweep is done — make it obvious we're now blocking on
        # sbatch and the controller hasn't exited.
        print(
            f"[sweep] Submitting trial {artifact.trial.id} via 'sbatch --wait' "
            f"({script_path}); controller will block until the job exits."
        )
        try:
            result = subprocess.run(["sbatch", "--wait", str(script_path)], env=env)
        except OSError as exc:
            if attempts > retry_budget:
                _write_launch_failure_status(artifact, exc)
                return -1
            continue
        if result.returncode == 0:
            _write_status(artifact, state="completed", finished_at=utc_now(), returncode=0)
            return 0
        if attempts > retry_budget:
            _write_status(
                artifact, state="failed", finished_at=utc_now(), returncode=result.returncode
            )
            return result.returncode


def _submit_trials_to_slurm_sync(
    artifacts: list[TrialArtifacts],
    *,
    continue_on_failure: bool,
    retry_budget: int,
    on_trial_complete: TrialCompleteCallback | None = None,
) -> int:
    failures = 0
    for artifact in artifacts:
        if _is_completed(artifact):
            continue
        returncode = _run_with_retries_slurm_sync(artifact, retry_budget)
        if returncode != 0:
            failures += 1
        if on_trial_complete is not None and on_trial_complete(artifact, returncode):
            break
        if returncode != 0 and not continue_on_failure:
            break
    return failures


_SLURM_TERMINAL_STATES_OK = {"COMPLETED"}
_SLURM_TERMINAL_STATES_BAD = {
    "FAILED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
    "PREEMPTED",
    "REVOKED",
}
_SLURM_TERMINAL_STATES_CANCELLED = {"CANCELLED"}


def _render_sbatch_script(artifact: TrialArtifacts, env: dict[str, str]) -> int:
    """Run ``<entrypoint> --dry-run`` to materialize the sbatch script.

    Returns the dry-run returncode. The caller is responsible for writing the
    status row on failure; we keep this helper pure so it can be reused by the
    non-pruning and pruning SLURM-sync paths.
    """
    try:
        result = subprocess.run(artifact.command + ["--dry-run"], env=env)
    except OSError as exc:
        raise exc
    return result.returncode


def _submit_sbatch_parsable(script_path: Path, env: dict[str, str]) -> tuple[int, str | None]:
    """Submit a script with ``sbatch --parsable``; return ``(returncode, jobid)``.

    ``--parsable`` writes only the job id (optionally ``jobid;cluster``) to
    stdout, which lets us track the job without scraping the human-readable
    ``Submitted batch job <id>`` line. Returns ``jobid=None`` when stdout is
    empty or sbatch exits non-zero — the caller decides whether to retry.
    """
    try:
        result = subprocess.run(
            ["sbatch", "--parsable", str(script_path)],
            env=env,
            capture_output=True,
            text=True,
        )
    except OSError:
        raise
    if result.returncode != 0:
        return result.returncode, None
    raw = (result.stdout or "").strip()
    if not raw:
        return result.returncode, None
    jobid = raw.split(";", 1)[0].strip() or None
    return result.returncode, jobid


class SqueueQueryError(RuntimeError):
    """Raised when ``squeue`` cannot be reached or returns an unrecognized error.

    Distinguished from "job not in queue" (which the caller treats as
    terminal): a transient query failure must NOT be silently mapped to
    "job is gone", or the controller would break out of its polling loop
    while the SLURM job is still running.
    """


_SQUEUE_NOT_FOUND_PATTERNS = (
    "invalid job id",
    "invalid jobid",
    "no such job",
    "job not found",
)


def _query_squeue_state(jobid: str) -> str | None:
    """Return the SLURM state of ``jobid`` via ``squeue``, or ``None`` if the
    job is not in the active queue.

    Raises ``SqueueQueryError`` when the query itself fails — missing binary,
    munge/auth error, slurmctld unreachable, or any non-"invalid job id"
    stderr. Distinguishing query failure from "job is gone" is essential:
    silently collapsing them would let a transient ``squeue`` outage break
    the controller's polling loop while the job is still running, after
    which the controller would advance to the next Optuna trial and contend
    for the still-active allocation.
    """
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", jobid, "-o", "%T"],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise SqueueQueryError(f"squeue invocation failed: {exc}") from exc
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode == 0:
        return stdout or None
    # Non-zero: older SLURM versions return non-zero for "job not found" while
    # newer versions return 0 with empty stdout. Treat the "job not found"
    # error string as terminal; anything else is a query failure to retry.
    stderr_lower = stderr.lower()
    if any(pattern in stderr_lower for pattern in _SQUEUE_NOT_FOUND_PATTERNS):
        return None
    raise SqueueQueryError(
        f"squeue returned rc={result.returncode}, stderr={stderr!r}"
    )


def _query_sacct_state(jobid: str) -> str | None:
    """Return the terminal state of ``jobid`` via ``sacct``, or ``None``.

    The first line of ``sacct -j <id> -n -P -o State`` is the batch step's
    state. Returns ``None`` when sacct produces no output or is unavailable.
    """
    try:
        result = subprocess.run(
            ["sacct", "-j", jobid, "-n", "-P", "-o", "State"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    for line in (result.stdout or "").splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        if token:
            return token
    return None


def _scancel_job(jobid: str, *, grace_seconds: float = 30.0, poll_interval: float = 1.0) -> bool:
    """Cancel a SLURM job and confirm it has left the queue.

    Returns ``True`` only when ``squeue`` reports the job is no longer in
    the queue within ``grace_seconds``. Returns ``False`` otherwise — the
    scancel binary was unavailable, scancel returned non-zero, squeue
    queries kept failing, or the job stayed in the queue past the deadline.

    Callers MUST treat ``False`` as "cancellation NOT confirmed": the SLURM
    job may still be running, so advancing the sweep would race the next
    trial against the still-active allocation. Record the trial as failed
    in that case, not pruned.
    """
    try:
        subprocess.run(["scancel", jobid], capture_output=True)
    except OSError:
        # Fall through to the wait loop: maybe an earlier scancel by another
        # tenant is in flight and the job will still leave the queue.
        pass

    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            state = _query_squeue_state(jobid)
        except SqueueQueryError:
            # Can't tell whether the job left — keep polling until the
            # deadline rather than declaring success.
            time.sleep(poll_interval)
            continue
        if state is None:
            return True
        time.sleep(poll_interval)

    return False


_SLURM_NON_TERMINAL_STATES = {
    "PENDING",
    "RUNNING",
    "REQUEUED",
    "RESIZING",
    "SUSPENDED",
    "CONFIGURING",
    "COMPLETING",
}


def _wait_for_sacct_terminal_state(
    jobid: str,
    *,
    timeout_seconds: float = 60.0,
    poll_interval: float = 2.0,
) -> str | None:
    """Poll sacct until it reports a terminal state (or the budget is spent).

    The job already left ``squeue`` so we know it has finished. However sacct
    can lag by several seconds while the accounting daemon catches up on most
    real clusters (and longer when the DB is congested) — without a backoff
    we would record a COMPLETED job as failed simply because we asked too
    soon. ``None`` results are also retried since an empty sacct response is
    indistinguishable from "not committed yet."

    Returns the observed terminal state, or ``None`` if the timeout elapses
    before any terminal state shows up. ``CANCELLED`` is treated as terminal
    here: it is the state we expect after a prune-driven ``scancel``.
    """
    deadline = time.monotonic() + timeout_seconds
    last_state: str | None = None
    while True:
        state = _query_sacct_state(jobid)
        if state is not None:
            head = state.split()[0]
            if head not in _SLURM_NON_TERMINAL_STATES:
                return state
            last_state = state
        if time.monotonic() >= deadline:
            return last_state
        time.sleep(poll_interval)


def _slurm_job_terminal_returncode(jobid: str, *, timeout_seconds: float = 60.0) -> int:
    """Map the sacct terminal state to a POSIX-ish returncode.

    SLURM does not expose the underlying exit code in a single canonical way
    across configurations, so we collapse the state into 0 for COMPLETED and
    -1 for everything else. The sweep's status.json carries enough context
    (failure_stage, error) to distinguish causes.

    We poll sacct for up to ``timeout_seconds`` to avoid the common case
    where a successful job has left the queue but its accounting record has
    not yet been committed — a naive single-shot read returns ``None`` and
    would mark the trial failed.
    """
    state = _wait_for_sacct_terminal_state(jobid, timeout_seconds=timeout_seconds)
    if state is None:
        return -1
    head = state.split()[0]
    if head in _SLURM_TERMINAL_STATES_OK:
        return 0
    return -1


def _run_trial_with_pruning_slurm_sync(
    artifact: TrialArtifacts,
    optuna_trial,  # type: ignore[no-untyped-def]
    metric: str,
    poll_interval: float,
    attempt: int = 1,
):
    """Submit a SLURM job for one trial and poll metrics for Optuna pruning.

    Mirrors the local ``_run_trial_with_pruning`` contract:

    - Each new ``(step, value)`` in the shared-FS ``metrics.jsonl`` is forwarded
      to ``optuna_trial.report``; ``should_prune()`` is checked after every new
      report.
    - On a prune signal we ``scancel`` the underlying SLURM job (which kills
      every step process on the compute node, the equivalent of SIGTERM-ing
      the local process group).
    - On natural completion the final objective is read from the same sidecar
      so the sampler sees the value the rest of the sweep records.

    Requires a shared filesystem between the controller and the compute node
    so ``metrics.jsonl`` is visible to both — SLURM-sync sweeps already assume
    this for status/manifest reads.
    """
    # Lazy import to keep this file importable without optuna installed.
    from prime_rl.sweep.optuna_loop import _PollingOutcome

    env = _build_env(artifact, gpu_group=None)
    _reset_metrics_jsonl(artifact)
    _write_status(
        artifact,
        state="running",
        started_at=utc_now(),
        attempts=attempt,
        gpu_group=None,
    )

    try:
        dryrun_rc = _render_sbatch_script(artifact, env)
    except OSError as exc:
        return _PollingOutcome(
            state="failed",
            returncode=-1,
            objective=None,
            launch_error=True,
            launch_exception=exc,
        )
    if dryrun_rc != 0:
        _write_status(artifact, state="failed", finished_at=utc_now(), returncode=dryrun_rc)
        return _PollingOutcome(state="failed", returncode=dryrun_rc, objective=None)

    script_path = _slurm_script_path(artifact)
    if not script_path.exists():
        _write_status(
            artifact,
            state="failed",
            finished_at=utc_now(),
            returncode=-1,
            failure_stage="materialization",
            error=f"sbatch script missing after --dry-run: {script_path}",
        )
        return _PollingOutcome(state="failed", returncode=-1, objective=None)

    try:
        submit_rc, jobid = _submit_sbatch_parsable(script_path, env)
    except OSError as exc:
        return _PollingOutcome(
            state="failed",
            returncode=-1,
            objective=None,
            launch_error=True,
            launch_exception=exc,
        )
    if submit_rc != 0 or jobid is None:
        _write_status(
            artifact,
            state="failed",
            finished_at=utc_now(),
            returncode=submit_rc if submit_rc != 0 else -1,
            failure_stage="submission",
            error=f"sbatch --parsable failed: rc={submit_rc}, jobid={jobid!r}",
        )
        return _PollingOutcome(
            state="failed",
            returncode=submit_rc if submit_rc != 0 else -1,
            objective=None,
        )

    _write_status(artifact, slurm_job_id=jobid)
    # Make the submission visible — the dry-run output above looks like the
    # sweep is done, so without this the controller appears to hang silently.
    print(
        f"[sweep] Submitted trial {artifact.trial.id} as SLURM job {jobid}; "
        f"polling metrics.jsonl every {poll_interval:.1f}s for Optuna pruning."
    )

    last_reported_step: int | None = None
    reports_sent = 0
    pruned = False
    prune_step: int | None = None
    prune_value: float | None = None
    consecutive_squeue_failures = 0
    # Three consecutive squeue failures over poll_interval cadence give a
    # short tolerance for transient cluster hiccups (controller restart,
    # brief slurmctld unavailability) without letting a persistently broken
    # queue silently terminate the polling loop.
    max_squeue_failures = 3

    while True:
        sample = read_intermediate_metric(artifact.run_dir, metric)
        report_just_sent = False
        if sample is not None:
            step, value = sample
            if last_reported_step is None or step > last_reported_step:
                optuna_trial.report(value, step)
                last_reported_step = step
                reports_sent += 1
                report_just_sent = True
                prune_step = step
                prune_value = value

        try:
            state = _query_squeue_state(jobid)
        except SqueueQueryError as exc:
            consecutive_squeue_failures += 1
            if consecutive_squeue_failures >= max_squeue_failures:
                # We can't see the queue, so we don't know whether the job
                # is still alive. Mark the trial unsafe-to-continue: the
                # outer loop must halt the sweep regardless of
                # continue_on_failure, because submitting the next Optuna
                # trial would race the possibly-still-running allocation.
                # Resume can pick the job back up via the recorded
                # slurm_job_id once the cluster is healthy.
                _write_status(
                    artifact,
                    state="failed",
                    finished_at=utc_now(),
                    returncode=-1,
                    failure_stage="squeue",
                    error=f"squeue unavailable: {exc}",
                )
                return _PollingOutcome(
                    state="failed",
                    returncode=-1,
                    objective=None,
                    reports_sent=reports_sent,
                    unsafe_to_continue=True,
                )
            time.sleep(poll_interval)
            continue
        consecutive_squeue_failures = 0

        if state is None:
            # Job left the queue (terminal). Stop polling.
            break

        # Only prune while the job is still in the queue, and only after
        # forwarding a fresh report. A stale should_prune() call could
        # otherwise fire repeatedly on the same data and waste cluster time.
        if report_just_sent and optuna_trial.should_prune():
            pruned = True
            break

        time.sleep(poll_interval)

    if pruned:
        cancelled = _scancel_job(jobid)
        if not cancelled:
            # scancel could not be confirmed. The SLURM job may still be
            # running, so we mark the trial unsafe-to-continue: the outer
            # loop must halt the sweep regardless of continue_on_failure,
            # because submitting the next Optuna trial would race the
            # still-active allocation. Resume retains the job id so the
            # operator can investigate.
            _write_status(
                artifact,
                state="failed",
                finished_at=utc_now(),
                returncode=-1,
                failure_stage="scancel",
                error=f"scancel did not confirm SLURM job {jobid} left the queue",
            )
            return _PollingOutcome(
                state="failed",
                returncode=-1,
                objective=None,
                reports_sent=reports_sent,
                unsafe_to_continue=True,
            )
        terminal_rc = _slurm_job_terminal_returncode(jobid)
        record_trial_pruned(
            artifact.status_path,
            prune_step,
            prune_value,
            returncode=terminal_rc if terminal_rc != 0 else -1,
            finished_at=utc_now(),
        )
        return _PollingOutcome(
            state="pruned",
            returncode=terminal_rc if terminal_rc != 0 else -1,
            objective=None,
            pruned_at_step=prune_step,
            pruned_value=prune_value,
            reports_sent=reports_sent,
        )

    returncode = _slurm_job_terminal_returncode(jobid)
    if returncode == 0:
        objective = read_final_summary(artifact.run_dir, metric)
        _write_status(artifact, state="completed", finished_at=utc_now(), returncode=0)
        return _PollingOutcome(
            state="completed", returncode=0, objective=objective, reports_sent=reports_sent
        )

    _write_status(artifact, state="failed", finished_at=utc_now(), returncode=returncode)
    return _PollingOutcome(
        state="failed", returncode=returncode, objective=None, reports_sent=reports_sent
    )


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
