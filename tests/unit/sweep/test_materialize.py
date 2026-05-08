import json
from pathlib import Path

import tomli
import tomli_w
from pydantic import ValidationError
from pydantic_config import ConfigFileError

from prime_rl.configs.sweep import SweepConfig
from prime_rl.sweep.materialize import Trial, materialize_trial


def write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def read_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomli.load(f)


def test_materialize_trial_writes_artifacts(tmp_path: Path) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    config = SweepConfig(
        name="unit-sweep",
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
    )
    trial = Trial(id="0000", label="lr_1e-5", parameters={"optim.lr": 1e-5})

    artifact = materialize_trial(config, trial)

    overrides = read_toml(artifact.overrides_path)
    assert overrides["output_dir"] == (tmp_path / "study" / "trials" / "0000" / "run").as_posix()
    assert overrides["optim"]["lr"] == 1e-5
    assert overrides["wandb"]["group"] == "unit-sweep"
    assert overrides["wandb"]["name"] == "lr_1e-5"

    resolved = read_toml(artifact.resolved_path)
    assert resolved["output_dir"] == overrides["output_dir"]
    assert resolved["optim"]["lr"] == 1e-5

    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "pending"
    assert artifact.command == [
        "uv",
        "run",
        "sft",
        "@",
        base_path.as_posix(),
        "@",
        artifact.overrides_path.as_posix(),
    ]
    assert artifact.command_path.read_text().strip() == " ".join(artifact.command)


def test_materialize_trial_rejects_bad_target_path(tmp_path: Path) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"does.not.exist": {"values": [1]}},
        wandb=None,
    )
    trial = Trial(id="0000", label="bad", parameters={"does.not.exist": 1})

    try:
        materialize_trial(config, trial)
    except (ConfigFileError, ValidationError, SystemExit):
        pass
    else:
        raise AssertionError("Expected target config validation to fail")


def test_materialize_trial_preserves_completed_status_on_resume(tmp_path: Path) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    trial = Trial(id="0000-deadbeef", label="lr_1e-5", parameters={"optim.lr": 1e-5})

    artifact = materialize_trial(config, trial)
    completed = json.loads(artifact.status_path.read_text())
    completed.update({"state": "completed", "returncode": 0, "objective": 0.42})
    artifact.status_path.write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n")

    materialize_trial(
        config,
        trial,
        resume=True,
        expected_checksums={
            "resolved_checksum": artifact.resolved_checksum,
            "base_checksums": artifact.base_checksums,
        },
    )
    after = json.loads(artifact.status_path.read_text())
    assert after["state"] == "completed"
    assert after["objective"] == 0.42

    materialize_trial(config, trial, resume=False)
    reset = json.loads(artifact.status_path.read_text())
    assert reset["state"] == "pending"


def test_materialize_trial_detects_base_drift_on_resume(tmp_path: Path) -> None:
    from prime_rl.sweep.materialize import SweepDriftError

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    trial = Trial(id="0000-deadbeef", label="lr_1e-5", parameters={"optim.lr": 1e-5})

    artifact = materialize_trial(config, trial)
    completed = json.loads(artifact.status_path.read_text())
    completed.update({"state": "completed", "returncode": 0})
    artifact.status_path.write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n")

    expected = {
        "resolved_checksum": artifact.resolved_checksum,
        "base_checksums": artifact.base_checksums,
    }

    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 99})

    try:
        materialize_trial(config, trial, resume=True, expected_checksums=expected)
    except SweepDriftError as exc:
        assert "base" in str(exc).lower()
    else:
        raise AssertionError("Expected SweepDriftError when base file changed under a completed trial")


# ---------------------------------------------------------------------------
# Phase 7a — multi_run_lora trial layout
# ---------------------------------------------------------------------------


def _stub_resolved_rl_config(monkeypatch, captured: dict) -> None:
    """Replace validate_target_config so we don't need a fully valid RLConfig."""
    from types import SimpleNamespace

    from prime_rl.sweep import materialize as mat_mod

    def fake_validate(entrypoint, args):
        captured["entrypoint"] = entrypoint
        captured["args"] = list(args)
        # Read the overrides toml the materializer just wrote so the test can
        # assert on the values that flowed through.
        overrides_path = Path(args[-1])
        captured["overrides"] = tomli.loads(overrides_path.read_text())
        orch = captured["overrides"].get("orchestrator", {})

        class FakeOrch:
            def model_dump(self, *, exclude_none=True, mode="json"):
                return orch

        return SimpleNamespace(orchestrator=FakeOrch())

    monkeypatch.setattr(mat_mod, "validate_target_config", fake_validate)


def test_materialize_multi_run_trial_writes_run_layout(tmp_path: Path, monkeypatch) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    captured: dict = {}
    _stub_resolved_rl_config(monkeypatch, captured)

    config = SweepConfig(
        name="multi-run-test",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 2,
            "shared": [shared_path],
        },
        parameters={"orchestrator.optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(id="0000-deadbeef", label="lr_1e-5", parameters={"orchestrator.optim.lr": 1e-5})

    artifact = materialize_multi_run_trial(config, trial, scheduler)

    expected_run_dir = tmp_path / "study" / "shared" / "run_0000-deadbeef"
    assert artifact.run_dir == expected_run_dir
    assert artifact.trial_dir == expected_run_dir
    assert (expected_run_dir / "control" / "orch.toml").exists()
    assert (expected_run_dir / "status.json").exists()

    # The output_dir injected into the resolved orch.toml must match the
    # run dir the trainer will discover, otherwise the FileMonitor sidecar
    # would land somewhere the controller never reads.
    assert captured["overrides"]["orchestrator"]["output_dir"] == expected_run_dir.as_posix()
    assert captured["overrides"]["orchestrator"]["optim"]["lr"] == 1e-5

    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "pending"
    assert status["id"] == "0000-deadbeef"


def test_materialize_multi_run_trial_injects_wandb_overrides(tmp_path: Path, monkeypatch) -> None:
    from prime_rl.configs.sweep import MultiRunLoRASchedulerConfig
    from prime_rl.sweep.materialize import materialize_multi_run_trial

    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    captured: dict = {}
    _stub_resolved_rl_config(monkeypatch, captured)

    config = SweepConfig(
        name="multi-run-wandb",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        parameters={"orchestrator.optim.lr": {"values": [1e-5]}},
    )
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    trial = Trial(id="0001-cafebabe", label="lr_1e-5", parameters={"orchestrator.optim.lr": 1e-5})

    materialize_multi_run_trial(config, trial, scheduler)

    wandb = captured["overrides"]["orchestrator"]["wandb"]
    # group defaults to the sweep name when wandb.group is unset
    assert wandb["group"] == "multi-run-wandb"
    assert wandb["name"] == "lr_1e-5"
    # tags include the canonical sweep markers
    assert "sweep" in wandb["tags"]
    assert "trial:0001-cafebabe" in wandb["tags"]
    assert "study:multi-run-wandb" in wandb["tags"]
