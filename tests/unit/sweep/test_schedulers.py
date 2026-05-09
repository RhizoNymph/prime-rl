import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import tomli_w

from prime_rl.configs.sweep import SweepConfig
from prime_rl.sweep.materialize import Trial, materialize_trial
from prime_rl.sweep.schedulers import (
    run_trials_locally,
    submit_trials_to_multi_run_lora,
    submit_trials_to_slurm,
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


def test_local_run_marks_launch_oserror_failed(tmp_path: Path, monkeypatch) -> None:
    _, artifacts = _materialize(tmp_path, count=1)

    def fake_run(_command, env=None):
        raise FileNotFoundError("missing launcher")

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    failures = run_trials_locally(artifacts, retry_budget=0)

    assert failures == 1
    status = json.loads(artifacts[0].status_path.read_text())
    assert status["state"] == "failed"
    assert status["returncode"] == -1
    assert status["failure_stage"] == "launch"
    assert "FileNotFoundError" in status["error"]


def test_local_run_retries_launch_oserror(tmp_path: Path, monkeypatch) -> None:
    _, artifacts = _materialize(tmp_path, count=1)
    attempts = 0

    def fake_run(_command, env=None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FileNotFoundError("temporary launcher miss")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    failures = run_trials_locally(artifacts, retry_budget=1)

    assert failures == 0
    assert attempts == 2
    status = json.loads(artifacts[0].status_path.read_text())
    assert status["state"] == "completed"
    assert status["returncode"] == 0
    assert status["attempts"] == 2
    assert "failure_stage" not in status


def test_slurm_submission_marks_launch_oserror_failed(tmp_path: Path, monkeypatch) -> None:
    _, artifacts = _materialize(tmp_path, count=1)

    def fake_run(_command):
        raise FileNotFoundError("missing sbatch wrapper")

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    failures = submit_trials_to_slurm(artifacts, retry_budget=0)

    assert failures == 1
    status = json.loads(artifacts[0].status_path.read_text())
    assert status["state"] == "failed"
    assert status["returncode"] == -1
    assert status["failure_stage"] == "launch"


def test_slurm_submission_retries_launch_oserror(tmp_path: Path, monkeypatch) -> None:
    _, artifacts = _materialize(tmp_path, count=1)
    attempts = 0

    def fake_run(_command):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FileNotFoundError("temporary sbatch wrapper miss")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    failures = submit_trials_to_slurm(artifacts, retry_budget=1)

    assert failures == 0
    assert attempts == 2
    status = json.loads(artifacts[0].status_path.read_text())
    assert status["state"] == "submitted"
    assert status["returncode"] == 0
    assert status["attempts"] == 2
    assert "failure_stage" not in status


def test_multi_run_lora_marks_all_artifacts_failed_on_launcher_oserror(
    tmp_path: Path, monkeypatch
) -> None:
    _, artifacts = _materialize(tmp_path, count=2)

    def fake_run(_command):
        raise FileNotFoundError("missing rl-multi-run")

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    failures = submit_trials_to_multi_run_lora(
        artifacts,
        shared_paths=[tmp_path / "shared.toml"],
        shared_dir=tmp_path / "shared",
        retry_budget=0,
    )

    assert failures == 2
    for artifact in artifacts:
        status = json.loads(artifact.status_path.read_text())
        assert status["state"] == "failed"
        assert status["returncode"] == -1
        assert status["failure_stage"] == "launch"


def test_multi_run_lora_retries_launcher_oserror(
    tmp_path: Path, monkeypatch
) -> None:
    _, artifacts = _materialize(tmp_path, count=2)
    attempts = 0

    def fake_run(command):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise FileNotFoundError("temporary rl-multi-run miss")
        runs_idx = command.index("--runs-dir")
        for raw_run_dir in command[runs_idx + 1].split(":"):
            control_dir = Path(raw_run_dir) / "control"
            control_dir.mkdir(parents=True, exist_ok=True)
            (control_dir / "exit_code").write_text("0\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    failures = submit_trials_to_multi_run_lora(
        artifacts,
        shared_paths=[tmp_path / "shared.toml"],
        shared_dir=tmp_path / "shared",
        retry_budget=1,
    )

    assert failures == 0
    assert attempts == 2
    for artifact in artifacts:
        status = json.loads(artifact.status_path.read_text())
        assert status["state"] == "completed"
        assert status["returncode"] == 0
        assert status["attempts"] == 2
        assert "failure_stage" not in status
