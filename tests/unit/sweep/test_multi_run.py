"""End-to-end-ish coverage for the multi_run_lora sweep path.

The full sweep -> rl-multi-run -> trainer/inference/orchestrator stack
requires GPUs and a model; here we monkeypatch the parts that need real
infra (``validate_target_config`` for the resolved orchestrator config,
``subprocess.run`` for the rl-multi-run invocation) and assert on the
sweep-side contract: layout on disk, the command shape, and per-trial
objective recording from each run's metrics.jsonl sidecar.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import tomli
import tomli_w

from prime_rl.configs.sweep import SweepConfig
from prime_rl.sweep.controller import run_sweep


def write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def _stub_validate_target_config(monkeypatch) -> None:
    """Replace pydantic-config validation with a passthrough that mirrors
    the override TOML into a fake resolved orchestrator config."""
    from prime_rl.sweep import materialize as mat_mod

    def fake_validate(entrypoint, args):
        overrides_path = Path(args[-1])
        overrides = tomli.loads(overrides_path.read_text())
        orch = overrides.get("orchestrator", {})

        class FakeOrch:
            def model_dump(self, *, exclude_none=True, mode="json"):
                return orch

        return SimpleNamespace(orchestrator=FakeOrch())

    monkeypatch.setattr(mat_mod, "validate_target_config", fake_validate)


def test_multi_run_lora_sweep_end_to_end(tmp_path: Path, monkeypatch) -> None:
    """A grid over orchestrator.optim.lr produces N run dirs, invokes
    rl-multi-run once, and records per-trial objectives from each run's
    metrics.jsonl."""
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    config = SweepConfig(
        name="lora-sweep",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 3,
            "shared": [shared_path],
        },
        parameters={"orchestrator.optim.lr": {"values": [1e-5, 3e-5, 1e-4]}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    # The trial id is index-prefixed and contains a hash, so we look up
    # rewards by the index parsed out of run_<NNNN>-<hash> at call time.
    captured: dict = {"commands": []}
    rewards_by_index = {0: 0.4, 1: 0.7, 2: 0.3}

    import subprocess as real_subprocess

    real_run = real_subprocess.run

    def fake_run(command, env=None, **kwargs):
        if command[:2] == ["git", "rev-parse"] or command[:2] == ["git", "status"]:
            return real_run(command, **kwargs)
        captured["commands"].append(list(command))
        idx = command.index("--runs-dir")
        run_dirs = [Path(p) for p in command[idx + 1].split(":") if p]
        for run_dir in run_dirs:
            # Trial IDs are <NNNN>-<hash>; the leading int is the index.
            trial_index = int(run_dir.name.removeprefix("run_").split("-", 1)[0])
            reward = rewards_by_index[trial_index]
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "metrics.jsonl").write_text(
                json.dumps({"step": 1, "reward": reward}) + "\n"
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    run_sweep(config)

    # Exactly one rl-multi-run command was issued, against the three run dirs.
    assert len(captured["commands"]) == 1
    cmd = captured["commands"][0]
    assert cmd[0] == "rl-multi-run"
    assert "--runs-dir" in cmd
    runs_dir_arg = cmd[cmd.index("--runs-dir") + 1]
    run_dirs = runs_dir_arg.split(":")
    assert len(run_dirs) == 3
    for piece in run_dirs:
        path = Path(piece)
        assert path.exists()
        assert (path / "control" / "orch.toml").exists()
        assert (path / "metrics.jsonl").exists()

    # Manifest summary records the best objective.
    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    summary = manifest["summary"]
    assert summary["completed"] == 3
    assert summary["best_value"] == 0.7

    # Per-trial status.json got the right objectives.
    objectives = []
    for variant in manifest["variants"]:
        status = json.loads(Path(variant["status_path"]).read_text())
        objectives.append(status["objective"])
    assert sorted(objectives) == [0.3, 0.4, 0.7]


def test_multi_run_lora_sweep_marks_all_failed_when_invocation_returns_nonzero(
    tmp_path: Path, monkeypatch
) -> None:
    """Phase 7a's coarse failure attribution: a non-zero rl-multi-run exit
    is recorded as a failure for every trial because we don't yet have
    per-orchestrator status reconciliation."""
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    captured: dict = {"commands": []}

    import subprocess as real_subprocess

    real_run = real_subprocess.run

    def fake_run(command, env=None, **kwargs):
        if command[:2] == ["git", "rev-parse"] or command[:2] == ["git", "status"]:
            return real_run(command, **kwargs)
        captured["commands"].append(list(command))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    config = SweepConfig(
        name="lora-sweep-fail",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 2,
            "shared": [shared_path],
        },
        parameters={"orchestrator.optim.lr": {"values": [1e-5, 3e-5]}},
        wandb=None,
    )

    try:
        run_sweep(config)
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("Expected SystemExit when rl-multi-run returncode != 0")

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    for variant in manifest["variants"]:
        status = json.loads(Path(variant["status_path"]).read_text())
        assert status["state"] == "failed"
        assert status["returncode"] == 1


def test_multi_run_lora_dry_run_lists_run_dirs(tmp_path: Path, monkeypatch, capsys) -> None:
    """dry_run materializes the layout but does not invoke rl-multi-run."""
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    called: list = []

    import subprocess as real_subprocess

    real_run = real_subprocess.run

    def fake_run(command, env=None, **kwargs):
        if command[:2] == ["git", "rev-parse"] or command[:2] == ["git", "status"]:
            return real_run(command, **kwargs)
        called.append(list(command))

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    config = SweepConfig(
        name="lora-dry",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 2,
            "shared": [shared_path],
        },
        parameters={"orchestrator.optim.lr": {"values": [1e-5, 3e-5]}},
        wandb=None,
        dry_run=True,
    )

    run_sweep(config)

    rl_multi_run_calls = [c for c in called if c and c[0] == "rl-multi-run"]
    assert rl_multi_run_calls == []

    out = capsys.readouterr().out
    assert "Materialized 2 run dir(s)" in out
    # Run dirs exist on disk so the user can inspect orch.toml etc.
    assert (tmp_path / "study" / "shared").exists()
    assert sum(1 for p in (tmp_path / "study" / "shared").glob("run_*")) == 2
