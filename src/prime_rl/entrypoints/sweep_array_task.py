"""SLURM array task wrapper for sweep submissions.

Phase 8: a static sweep with ``scheduler.use_array=True`` submits a single
``sbatch --array=0-N-1`` job. Each array task runs this entrypoint; we use
``$SLURM_ARRAY_TASK_ID`` to look up the matching variant in the study's
``manifest.json`` and exec the trial command.

Usage (from the array sbatch script):

    uv run sweep-array-task /path/to/study

Status flow per task:

1. Find the variant whose ``array_task_index == SLURM_ARRAY_TASK_ID``.
2. Write ``status.json`` ``state="running"`` with ``slurm_job_id`` set
   from the SLURM env vars so the controller and ``sacct`` can be
   correlated post-hoc.
3. Run the trial's recorded ``command`` (already a fully-resolved
   ``uv run rl @ base.toml @ overrides.toml``).
4. Write ``state="completed"`` on returncode 0, else ``state="failed"``.

Phase 8 deliberately keeps this wrapper simple: no retries, no metric
reading, no early stopping. The cluster handles failure visibility and
the post-hoc summary regenerates from the per-task ``status.json`` files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_variant(manifest: dict, array_index: int) -> dict:
    for variant in manifest.get("variants", []):
        if variant.get("array_task_index") == array_index:
            return variant
    raise SystemExit(
        f"No variant found with array_task_index={array_index} in manifest. "
        "Did the study get re-materialized between submission and task start?"
    )


def _slurm_job_id() -> str | None:
    """Compose the canonical ``<job>_<task>`` identifier for this task."""
    job = os.environ.get("SLURM_ARRAY_JOB_ID")
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if job is None or task is None:
        return None
    return f"{job}_{task}"


def _write_status(status_path: Path, **updates) -> None:
    status = json.loads(status_path.read_text())
    status.update(updates)
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single SLURM array task for a sweep trial.")
    parser.add_argument("study_dir", type=Path, help="Path to the sweep study output directory.")
    args = parser.parse_args()

    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw is None:
        raise SystemExit(
            "SLURM_ARRAY_TASK_ID is not set; sweep-array-task is only meant to run "
            "inside a SLURM array job."
        )
    array_index = int(raw)

    manifest_path: Path = args.study_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    variant = _find_variant(manifest, array_index)
    status_path = Path(variant["status_path"])
    command = variant["command"]

    _write_status(
        status_path,
        state="running",
        started_at=_utc_now(),
        slurm_job_id=_slurm_job_id(),
        attempts=1,
    )

    result = subprocess.run(command)
    finished_at = _utc_now()

    if result.returncode == 0:
        _write_status(status_path, state="completed", finished_at=finished_at, returncode=0)
        sys.exit(0)
    else:
        _write_status(status_path, state="failed", finished_at=finished_at, returncode=result.returncode)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
