import json
import subprocess
from datetime import datetime, timezone

from prime_rl.sweep.materialize import TrialArtifacts, write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_status(artifacts: TrialArtifacts) -> dict:
    return json.loads(artifacts.status_path.read_text())


def _write_status(artifacts: TrialArtifacts, **updates) -> None:
    status = _read_status(artifacts)
    status.update(updates)
    write_json(artifacts.status_path, status)


def _run_with_retries(artifact: TrialArtifacts, retry_budget: int) -> int:
    """Run a single trial, retrying transient failures up to ``retry_budget`` times.

    Returns the final returncode. Each attempt is recorded as a fresh
    ``running`` transition with the cumulative attempt count in status.json.
    """
    attempts = 0
    while True:
        attempts += 1
        _write_status(artifact, state="running", started_at=utc_now(), attempts=attempts)
        result = subprocess.run(artifact.command)
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


def run_trials_locally(
    artifacts: list[TrialArtifacts],
    max_parallel: int = 1,
    continue_on_failure: bool = True,
    retry_budget: int = 1,
) -> int:
    """Run trials sequentially. Returns the count of failed trials.

    Trials whose status.json already records ``state == "completed"`` are
    skipped so ``--resume`` only re-runs the work that did not finish.
    """
    if max_parallel != 1:
        raise ValueError(
            f"Local sweep scheduler only supports max_parallel=1 (got {max_parallel}). "
            "Parallel execution lands in Phase 3."
        )

    failures = 0
    for artifact in artifacts:
        if _is_completed(artifact):
            continue
        returncode = _run_with_retries(artifact, retry_budget)
        if returncode != 0:
            failures += 1
            if not continue_on_failure:
                raise SystemExit(returncode)
    return failures


def submit_trials_to_slurm(
    artifacts: list[TrialArtifacts],
    max_parallel: int = 1,
    continue_on_failure: bool = True,
    retry_budget: int = 1,
) -> int:
    """Submit trials through the target entrypoint's SLURM support.

    The target entrypoint owns SLURM rendering/submission. max_parallel is kept
    in the config for forward compatibility with controller-managed queues.
    Submission failures (not job failures) are retried up to retry_budget.
    """
    _ = max_parallel
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
