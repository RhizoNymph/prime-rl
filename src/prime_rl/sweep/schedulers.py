import json
import subprocess
import time
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


def run_trials_locally(artifacts: list[TrialArtifacts], max_parallel: int = 1) -> None:
    if max_parallel == 1:
        for artifact in artifacts:
            _write_status(artifact, state="running", started_at=utc_now())
            result = subprocess.run(artifact.command)
            state = "completed" if result.returncode == 0 else "failed"
            _write_status(artifact, state=state, finished_at=utc_now(), returncode=result.returncode)
            if result.returncode != 0:
                raise SystemExit(result.returncode)
        return

    running: list[tuple[TrialArtifacts, subprocess.Popen]] = []
    pending = list(artifacts)

    while pending or running:
        while pending and len(running) < max_parallel:
            artifact = pending.pop(0)
            process = subprocess.Popen(artifact.command)
            _write_status(artifact, state="running", started_at=utc_now(), pid=process.pid)
            running.append((artifact, process))

        next_running: list[tuple[TrialArtifacts, subprocess.Popen]] = []
        for artifact, process in running:
            returncode = process.poll()
            if returncode is None:
                next_running.append((artifact, process))
                continue
            state = "completed" if returncode == 0 else "failed"
            _write_status(artifact, state=state, finished_at=utc_now(), returncode=returncode)
            if returncode != 0:
                for _, live_process in next_running:
                    live_process.terminate()
                raise SystemExit(returncode)
        running = next_running
        if running:
            time.sleep(0.2)


def submit_trials_to_slurm(artifacts: list[TrialArtifacts], max_parallel: int = 1) -> None:
    # The target entrypoint owns SLURM rendering/submission. max_parallel is kept
    # in the config for forward compatibility with controller-managed queues.
    _ = max_parallel
    for artifact in artifacts:
        _write_status(artifact, state="submitting", started_at=utc_now())
        result = subprocess.run(artifact.command)
        state = "submitted" if result.returncode == 0 else "failed"
        _write_status(artifact, state=state, finished_at=utc_now(), returncode=result.returncode)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
