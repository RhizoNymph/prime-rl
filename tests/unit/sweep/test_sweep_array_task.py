"""Phase 8: SLURM array task wrapper.

Each ``sbatch --array=0-N-1`` job invokes ``sweep-array-task`` with
``$SLURM_ARRAY_TASK_ID`` set; the wrapper looks up the variant in
``manifest.json`` by ``array_task_index`` and execs the trial command.
Test covers the lookup, env handling, and status transitions.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from prime_rl.entrypoints import sweep_array_task


def _write_study(tmp_path: Path, variants: list[dict]) -> Path:
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    for variant in variants:
        status_path = Path(variant["status_path"])
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps(
                {
                    "id": variant["id"],
                    "label": variant["id"],
                    "state": "submitted",
                    "objective": None,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    (study_dir / "manifest.json").write_text(
        json.dumps({"variants": variants}, indent=2, sort_keys=True) + "\n"
    )
    return study_dir


def test_sweep_array_task_runs_correct_variant(tmp_path: Path, monkeypatch) -> None:
    variants = [
        {
            "id": "0000-aa",
            "command": ["echo", "trial-0"],
            "status_path": (tmp_path / "study" / "trials" / "0000-aa" / "status.json").as_posix(),
            "array_task_index": 0,
        },
        {
            "id": "0001-bb",
            "command": ["echo", "trial-1"],
            "status_path": (tmp_path / "study" / "trials" / "0001-bb" / "status.json").as_posix(),
            "array_task_index": 1,
        },
    ]
    study_dir = _write_study(tmp_path, variants)

    captured: dict = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "42")
    monkeypatch.setattr(sweep_array_task.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["sweep-array-task", study_dir.as_posix()])

    with pytest.raises(SystemExit) as excinfo:
        sweep_array_task.main()
    assert excinfo.value.code == 0

    assert captured["command"] == ["echo", "trial-1"]

    status = json.loads(Path(variants[1]["status_path"]).read_text())
    assert status["state"] == "completed"
    assert status["returncode"] == 0
    assert status["slurm_job_id"] == "42_1"


def test_sweep_array_task_records_failure(tmp_path: Path, monkeypatch) -> None:
    variants = [
        {
            "id": "0000-aa",
            "command": ["false"],
            "status_path": (tmp_path / "study" / "trials" / "0000-aa" / "status.json").as_posix(),
            "array_task_index": 0,
        },
    ]
    study_dir = _write_study(tmp_path, variants)

    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "100")
    monkeypatch.setattr(
        sweep_array_task.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=2),
    )
    monkeypatch.setattr(sys, "argv", ["sweep-array-task", study_dir.as_posix()])

    with pytest.raises(SystemExit) as excinfo:
        sweep_array_task.main()
    assert excinfo.value.code == 2

    status = json.loads(Path(variants[0]["status_path"]).read_text())
    assert status["state"] == "failed"
    assert status["returncode"] == 2


def test_sweep_array_task_errors_on_missing_index(tmp_path: Path, monkeypatch) -> None:
    study_dir = _write_study(tmp_path, [])

    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "5")
    monkeypatch.setattr(sys, "argv", ["sweep-array-task", study_dir.as_posix()])

    with pytest.raises(SystemExit) as excinfo:
        sweep_array_task.main()
    assert "No variant" in str(excinfo.value)
