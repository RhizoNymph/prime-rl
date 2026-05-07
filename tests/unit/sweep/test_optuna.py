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


def test_optuna_resume_runs_only_remaining_budget(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    rewards = [0.4, 0.5, 0.6, 0.7, 0.8]
    _install_fake_run(monkeypatch, rewards)

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 5,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
            "study_name": "resume-study",
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    config = SweepConfig(**base_kwargs)
    config.strategy.num_trials = 3
    run_sweep(config)

    first_manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert len(first_manifest["variants"]) == 3
    first_ids = [v["id"] for v in first_manifest["variants"]]

    resume_config = SweepConfig(**base_kwargs, resume=True)
    run_sweep(resume_config)

    final_manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert len(final_manifest["variants"]) == 5
    assert [v["id"] for v in final_manifest["variants"][:3]] == first_ids
    assert final_manifest["summary"]["completed"] == 5
    assert final_manifest["summary"]["best_value"] == max(rewards)


def test_optuna_marks_failed_materialization_in_storage(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.5, 0.6])

    import prime_rl.sweep.optuna_loop as loop

    original_materialize = loop.materialize_trial
    call_count = {"n": 0}

    def flaky_materialize(config, trial, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise ValueError("fake config validation failure")
        return original_materialize(config, trial, **kwargs)

    monkeypatch.setattr(loop, "materialize_trial", flaky_materialize)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 3,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    try:
        run_sweep(config)
    except SystemExit as exc:
        assert exc.code == 1

    import optuna

    study = optuna.load_study(study_name="sweep", storage=storage_url)
    states = [t.state for t in study.trials]
    assert optuna.trial.TrialState.FAIL in states
    assert not any(state == optuna.trial.TrialState.RUNNING for state in states)


def test_optuna_resume_reconciles_running_trial_with_recorded_objective(
    tmp_path: Path, monkeypatch
) -> None:
    """Crash after subprocess finished but before study.tell(): replay the value."""
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.7, 0.6])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 2,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    config = SweepConfig(**base_kwargs)
    config.strategy.num_trials = 1
    run_sweep(config)

    import optuna

    study = optuna.load_study(study_name="sweep", storage=storage_url)
    completed_trial = study.trials[0]
    completed_value = completed_trial.value
    completed_params = dict(completed_trial.params)

    # Simulate the post-completion-but-pre-tell crash: ask another trial,
    # leave it RUNNING, and have its sweep status.json record an objective.
    pending = study.ask()
    pending_index = pending.number
    pending_id = f"{pending_index:04d}-fake"
    trial_dir = tmp_path / "study" / "trials" / pending_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    status_path = trial_dir / "status.json"
    status_path.write_text(json.dumps({"state": "completed", "returncode": 0, "objective": 0.95}))

    manifest_path = tmp_path / "study" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["variants"].append(
        {
            "id": pending_id,
            "label": pending_id,
            "status_path": status_path.as_posix(),
            "output_dir": (trial_dir / "run").as_posix(),
        }
    )
    manifest_path.write_text(json.dumps(manifest))

    resume_config = SweepConfig(**base_kwargs, resume=True)
    run_sweep(resume_config)

    study = optuna.load_study(study_name="sweep", storage=storage_url)
    states = [t.state for t in study.trials]
    assert all(state == optuna.trial.TrialState.COMPLETE for state in states)
    values = sorted(t.value for t in study.trials)
    assert values == sorted([completed_value, 0.95])
    assert dict(study.trials[0].params) == completed_params


def test_optuna_resume_reconciles_running_trial_with_no_recorded_objective(
    tmp_path: Path, monkeypatch
) -> None:
    """Crash before subprocess finished: mark the orphaned trial FAIL."""
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.4, 0.5])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 2,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    initial = SweepConfig(**base_kwargs)
    initial.strategy.num_trials = 1
    run_sweep(initial)

    import optuna

    study = optuna.load_study(study_name="sweep", storage=storage_url)
    study.ask()  # leak a RUNNING trial

    run_sweep(SweepConfig(**base_kwargs, resume=True))

    study = optuna.load_study(study_name="sweep", storage=storage_url)
    states = [t.state for t in study.trials]
    # Reconciliation marks the orphan FAIL; the failed slot still counts
    # toward the num_trials budget, matching Optuna's own optimize() semantics.
    assert optuna.trial.TrialState.RUNNING not in states
    assert optuna.trial.TrialState.FAIL in states
    assert sum(1 for s in states if s == optuna.trial.TrialState.COMPLETE) == 1


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
