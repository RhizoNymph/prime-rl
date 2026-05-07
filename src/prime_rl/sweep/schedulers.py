import json
import os
import queue
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from prime_rl.sweep.materialize import TrialArtifacts, write_json

TrialCompleteCallback = Callable[[TrialArtifacts, int], bool]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_status(artifacts: TrialArtifacts) -> dict:
    return json.loads(artifacts.status_path.read_text())


def _write_status(artifacts: TrialArtifacts, **updates) -> None:
    status = _read_status(artifacts)
    status.update(updates)
    write_json(artifacts.status_path, status)


def _build_env(gpu_group: list[int] | None) -> dict[str, str] | None:
    """Inherit the parent env but pin CUDA_VISIBLE_DEVICES for the trial."""
    if gpu_group is None:
        return None
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in gpu_group)
    return env


def _run_with_retries(artifact: TrialArtifacts, gpu_group: list[int] | None, retry_budget: int) -> int:
    """Run a single trial, retrying transient failures up to ``retry_budget`` times.

    Returns the final returncode. Each attempt is recorded as a fresh
    ``running`` transition with the cumulative attempt count and the assigned
    device group in status.json.
    """
    env = _build_env(gpu_group)
    attempts = 0
    while True:
        attempts += 1
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
