import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomli_w

from prime_rl.configs.sweep import SweepConfig
from prime_rl.sweep.materialize import Trial, materialize_trial
from prime_rl.sweep.schedulers import (
    compress_array_indices,
    query_running_array_tasks,
    run_trials_locally,
    submit_trials_to_slurm_array,
)


def _materialize(tmp_path: Path, count: int) -> tuple[SweepConfig, list]:
    base_path = tmp_path / "base.toml"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    with open(base_path, "wb") as f:
        tomli_w.dump({"data": {"type": "fake"}, "max_steps": 1}, f)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )

    artifacts = []
    for idx in range(count):
        trial = Trial(id=f"{idx:04d}-deadbeef", label=f"t{idx}", parameters={"optim.lr": 1e-5})
        artifacts.append(materialize_trial(config, trial))
    return config, artifacts


def test_sequential_run_pins_cuda_visible_devices(tmp_path: Path, monkeypatch) -> None:
    _, artifacts = _materialize(tmp_path, count=2)

    captured_envs: list[str | None] = []

    def fake_run(_command, env=None):
        captured_envs.append(env.get("CUDA_VISIBLE_DEVICES") if env is not None else None)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    failures = run_trials_locally(artifacts, max_parallel=1, gpu_groups=[[2, 3]])

    assert failures == 0
    assert captured_envs == ["2,3", "2,3"]
    for artifact in artifacts:
        status = json.loads(artifact.status_path.read_text())
        assert status["state"] == "completed"
        assert status["gpu_group"] == [2, 3]


def test_parallel_run_assigns_disjoint_groups_per_worker(tmp_path: Path, monkeypatch) -> None:
    _, artifacts = _materialize(tmp_path, count=6)

    inflight_lock = threading.Lock()
    inflight: dict[str, str] = {}
    seen_overlap = False

    def fake_run(_command, env=None):
        nonlocal seen_overlap
        devices = env.get("CUDA_VISIBLE_DEVICES") if env is not None else None
        with inflight_lock:
            if devices in inflight.values():
                seen_overlap = True
            inflight[id(_command)] = devices
        time.sleep(0.01)
        with inflight_lock:
            inflight.pop(id(_command), None)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    failures = run_trials_locally(
        artifacts,
        max_parallel=3,
        gpu_groups=[[0], [1], [2]],
    )

    assert failures == 0
    assert not seen_overlap
    devices_seen = {json.loads(a.status_path.read_text())["gpu_group"][0] for a in artifacts}
    assert devices_seen == {0, 1, 2}


def test_parallel_run_requires_enough_groups(tmp_path: Path) -> None:
    _, artifacts = _materialize(tmp_path, count=2)

    try:
        run_trials_locally(artifacts, max_parallel=4, gpu_groups=[[0]])
    except ValueError as exc:
        assert "max_parallel=4" in str(exc)
    else:
        raise AssertionError("Expected ValueError when gpu_groups is too short")


def test_parallel_run_records_failures_and_continues(tmp_path: Path, monkeypatch) -> None:
    _, artifacts = _materialize(tmp_path, count=4)

    def fake_run(_command, env=None):
        devices = env.get("CUDA_VISIBLE_DEVICES") if env is not None else None
        returncode = 0 if devices == "0" else 1
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    failures = run_trials_locally(
        artifacts,
        max_parallel=2,
        gpu_groups=[[0], [1]],
        continue_on_failure=True,
        retry_budget=0,
    )

    assert failures > 0


@pytest.mark.parametrize(
    "indices,expected",
    [
        ([], ""),
        ([3], "3"),
        ([0, 1, 2], "0-2"),
        ([0, 1, 2, 5, 7, 8, 9], "0-2,5,7-9"),
        ([5, 0, 2, 1], "0-2,5"),  # unsorted input
        ([0, 0, 1, 1, 2], "0-2"),  # duplicates
    ],
)
def test_compress_array_indices(indices, expected):
    assert compress_array_indices(indices) == expected


def _materialize_with_slurm(tmp_path: Path, count: int, slurm_block: dict) -> list:
    """Materialize artifacts with a [slurm] block written into each resolved.toml.

    submit_trials_to_slurm_array reads the block from the first artifact's
    resolved.toml; tests inject a representative one so the rendered sbatch
    script has expected SBATCH directives.
    """
    _, artifacts = _materialize(tmp_path, count=count)
    for artifact in artifacts:
        with open(artifact.resolved_path, "rb") as f:
            import tomli

            resolved = tomli.load(f)
        resolved["slurm"] = slurm_block
        with open(artifact.resolved_path, "wb") as f:
            tomli_w.dump(resolved, f)
    return artifacts


def test_submit_trials_to_slurm_array_writes_sbatch_and_submits(
    tmp_path: Path, monkeypatch
) -> None:
    artifacts = _materialize_with_slurm(
        tmp_path,
        count=4,
        slurm_block={
            "partition": "gpu",
            "gpus_per_node": 4,
            "time": "01:00:00",
            "cpus_per_task": 8,
            "exclusive": True,
        },
    )
    study_dir = tmp_path / "study"

    captured: dict = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        return SimpleNamespace(returncode=0, stdout="123456;cluster\n", stderr="")

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    array_job_id, indices = submit_trials_to_slurm_array(artifacts, study_dir=study_dir)

    assert array_job_id == "123456"
    assert indices == [0, 1, 2, 3]
    assert captured["command"][:2] == ["sbatch", "--parsable"]

    sbatch_path = study_dir / "array.sbatch"
    assert sbatch_path.exists()
    text = sbatch_path.read_text()
    assert "#SBATCH --array=0-3" in text
    assert "#SBATCH --partition=gpu" in text
    assert "#SBATCH --gpus-per-node=4" in text
    assert "#SBATCH --time=01:00:00" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --exclusive" in text
    assert "uv run sweep-array-task" in text
    assert str(study_dir) in text

    # Per-trial status flipped to "submitted" with the SLURM job id.
    for artifact in artifacts:
        status = json.loads(artifact.status_path.read_text())
        assert status["state"] == "submitted"
        assert status["slurm_job_id"] == "123456"


def test_submit_trials_to_slurm_array_filters_to_resume_indices(
    tmp_path: Path, monkeypatch
) -> None:
    artifacts = _materialize_with_slurm(tmp_path, count=5, slurm_block={"partition": "gpu"})
    study_dir = tmp_path / "study"

    captured: dict = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        return SimpleNamespace(returncode=0, stdout="999\n", stderr="")

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    array_job_id, indices = submit_trials_to_slurm_array(
        artifacts, study_dir=study_dir, array_indices=[0, 2, 3]
    )

    assert array_job_id == "999"
    assert indices == [0, 2, 3]
    text = (study_dir / "array.sbatch").read_text()
    assert "#SBATCH --array=0,2-3" in text


def test_submit_trials_to_slurm_array_marks_failed_on_sbatch_error(
    tmp_path: Path, monkeypatch
) -> None:
    artifacts = _materialize_with_slurm(tmp_path, count=2, slurm_block={"partition": "gpu"})
    study_dir = tmp_path / "study"

    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="sbatch: error\n")

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    array_job_id, indices = submit_trials_to_slurm_array(artifacts, study_dir=study_dir)
    assert array_job_id is None
    assert indices == []
    for artifact in artifacts:
        status = json.loads(artifact.status_path.read_text())
        assert status["state"] == "failed"


def test_query_running_array_tasks_parses_squeue_output(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        assert command[:3] == ["squeue", "-j", "999"]
        return SimpleNamespace(returncode=0, stdout="0\n3\n7\n", stderr="")

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    assert query_running_array_tasks("999") == {0, 3, 7}


def test_query_running_array_tasks_handles_ranges_and_lists(monkeypatch) -> None:
    """SLURM may collapse pending tasks into ranges or comma-lists."""
    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="0-2\n5,7\n", stderr="")

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    assert query_running_array_tasks("999") == {0, 1, 2, 5, 7}


def test_query_running_array_tasks_returns_empty_on_squeue_failure(monkeypatch) -> None:
    """squeue not installed / failing → empty set, caller falls back to status."""
    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="squeue: error\n")

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)
    assert query_running_array_tasks("999") == set()


def test_query_running_array_tasks_returns_empty_when_binary_missing(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        raise FileNotFoundError("squeue not found")

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)
    assert query_running_array_tasks("999") == set()


def test_query_running_array_tasks_returns_empty_when_no_job_id() -> None:
    """No prior job → no live tasks, period; don't even shell out."""
    assert query_running_array_tasks(None) == set()
    assert query_running_array_tasks("") == set()
