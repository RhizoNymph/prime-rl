import json
from pathlib import Path

import tomli_w

from prime_rl.configs.sweep import SweepConfig
from prime_rl.sweep.controller import run_sweep


def write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def test_run_sweep_dry_run_materializes_without_launching(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    launched = False

    def fake_local(*args, **kwargs):
        nonlocal launched
        launched = True

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    config = SweepConfig(
        name="unit",
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        dry_run=True,
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
    )

    run_sweep(config)

    assert not launched
    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    variant_ids = [variant["id"] for variant in manifest["variants"]]
    assert [vid[:4] for vid in variant_ids] == ["0000", "0001"]
    assert all(len(vid) == 13 and vid[4] == "-" for vid in variant_ids)
    assert (tmp_path / "study" / "trials" / variant_ids[0] / "resolved.toml").exists()
    assert "git" in manifest
    assert set(manifest["git"]) == {"sha", "dirty"}
    for variant in manifest["variants"]:
        assert len(variant["resolved_checksum"]) == 64
        assert variant["base_checksums"][base_path.as_posix()]


def test_run_sweep_dispatches_local_scheduler(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    called = {}

    def fake_local(artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None):
        called["count"] = len(artifacts)
        called["max_parallel"] = max_parallel
        called["gpu_groups"] = gpu_groups
        called["continue_on_failure"] = continue_on_failure
        called["retry_budget"] = retry_budget
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
    )

    run_sweep(config)

    assert called == {
        "count": 1,
        "max_parallel": 1,
        "gpu_groups": None,
        "continue_on_failure": True,
        "retry_budget": 1,
    }


def test_run_sweep_exits_nonzero_when_trials_fail(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    monkeypatch.setattr(
        "prime_rl.sweep.controller.run_trials_locally",
        lambda *args, **kwargs: 2,
    )

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5, 1e-4]}},
    )

    try:
        run_sweep(config)
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("Expected SystemExit when trials failed")


def test_run_sweep_random_strategy_dispatches_through_local_scheduler(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    captured = {}

    def fake_local(artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None):
        captured["count"] = len(artifacts)
        captured["parameters"] = [artifact.trial.parameters for artifact in artifacts]
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={"type": "random", "num_trials": 5, "seed": 13},
        parameters={
            "optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4},
            "data.temperature": {"distribution": "uniform", "min": 0.6, "max": 1.2},
        },
    )

    run_sweep(config)

    assert captured["count"] == 5
    for params in captured["parameters"]:
        assert 1e-6 <= params["optim.lr"] <= 1e-4
        assert 0.6 <= params["data.temperature"] <= 1.2


def test_run_sweep_resume_skips_completed_trials(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    base_config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
    )

    runs: list[list[str]] = []

    def fake_local(artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None):
        runs.append([artifact.trial.id for artifact in artifacts])
        first_status = json.loads(artifacts[0].status_path.read_text())
        first_status.update({"state": "completed", "returncode": 0})
        artifacts[0].status_path.write_text(json.dumps(first_status, indent=2, sort_keys=True) + "\n")
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    run_sweep(SweepConfig(**base_config_kwargs))

    completed_id = runs[0][0]
    pending_id = runs[0][1]

    completed_status_path = tmp_path / "study" / "trials" / completed_id / "status.json"
    pending_status_path = tmp_path / "study" / "trials" / pending_id / "status.json"
    assert json.loads(completed_status_path.read_text())["state"] == "completed"
    pending_status = json.loads(pending_status_path.read_text())
    pending_status.update({"state": "failed", "returncode": 1})
    pending_status_path.write_text(json.dumps(pending_status, indent=2, sort_keys=True) + "\n")

    resume_runs: list[list[str]] = []

    def fake_local_resume(
        artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None
    ):
        resume_runs.append(
            [(artifact.trial.id, json.loads(artifact.status_path.read_text())["state"]) for artifact in artifacts]
        )
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)

    run_sweep(SweepConfig(**base_config_kwargs, resume=True))

    assert resume_runs[0] == [(completed_id, "completed"), (pending_id, "pending")]
    assert json.loads(completed_status_path.read_text())["state"] == "completed"


def test_run_sweep_records_objective_and_halts_on_threshold(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    import subprocess as real_subprocess
    from types import SimpleNamespace

    seq = iter([0.9, 0.8, 0.2])
    real_run = real_subprocess.run

    def fake_run(command, env=None, **kwargs):
        if command[:2] == ["git", "rev-parse"] or command[:2] == ["git", "status"]:
            return real_run(command, **kwargs)
        overrides = [part for part in command if part.endswith("overrides.toml")]
        if not overrides:
            return real_run(command, **kwargs)
        run_dir = Path(overrides[0]).parent / "run"
        summary_dir = run_dir / "run-fake"
        summary_dir.mkdir(parents=True, exist_ok=True)
        (summary_dir / "final_summary.json").write_text(json.dumps({"reward": next(seq)}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5, 1e-4, 1e-3]}},
        objective={"metric": "reward", "direction": "maximize"},
        early_stopping={"type": "threshold", "threshold": 0.5},
        wandb=None,
    )

    run_sweep(config)

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    summary = manifest["summary"]
    assert summary["completed"] == 3
    assert summary["best_value"] == 0.9
    assert summary["halted_by_early_stopping"] is True
    assert summary["halt_reason"] == "threshold"


def test_run_sweep_passes_gpu_groups_to_local_scheduler(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    captured = {}

    def fake_local(artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None):
        captured["max_parallel"] = max_parallel
        captured["gpu_groups"] = gpu_groups
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "local",
            "max_parallel": 2,
            "gpu_assignment": {"visible_devices": [[0, 1], [2, 3]]},
        },
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
    )

    run_sweep(config)

    assert captured == {"max_parallel": 2, "gpu_groups": [[0, 1], [2, 3]]}
