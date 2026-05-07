import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomli_w

pytest.importorskip("optuna")

from prime_rl.configs.sweep import SweepConfig  # noqa: E402
from prime_rl.sweep.controller import run_sweep  # noqa: E402


def write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def _install_fake_run(monkeypatch, sequence):
    import subprocess as real_subprocess

    seq = iter(sequence)
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


def test_optuna_random_sweep_records_best_trial(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    rewards = [0.4, 0.7, 0.3, 0.9, 0.2]
    _install_fake_run(monkeypatch, rewards)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={"type": "optuna", "num_trials": 5, "sampler": "random", "seed": 7},
        parameters={
            "optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4},
            "optim.warmup": {"distribution": "int_uniform", "min": 0, "max": 10, "step": 2},
        },
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(config)

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    summary = manifest["summary"]
    assert summary["completed"] == 5
    assert summary["best_value"] == max(rewards)
    assert len(manifest["variants"]) == 5
    for variant in manifest["variants"]:
        assert "optim.lr" in variant["overrides"]
        assert "optim.warmup" in variant["overrides"]


def test_optuna_sweep_halts_on_threshold(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    rewards = [0.9, 0.8, 0.2, 0.7, 0.6]
    _install_fake_run(monkeypatch, rewards)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={"type": "optuna", "num_trials": 5, "sampler": "random", "seed": 7},
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        early_stopping={"type": "threshold", "threshold": 0.5},
        wandb=None,
    )

    run_sweep(config)

    summary = json.loads((tmp_path / "study" / "manifest.json").read_text())["summary"]
    assert summary["completed"] == 3
    assert summary["halted_by_early_stopping"] is True
    assert summary["halt_reason"] == "threshold"
    assert summary["best_value"] == 0.9
