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
            # 7b contract: launcher writes per-run exit_code; reconcile reads it.
            (run_dir / "control").mkdir(parents=True, exist_ok=True)
            (run_dir / "control" / "exit_code").write_text("0\n")
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


def test_multi_run_lora_sweep_attributes_failures_per_orchestrator(
    tmp_path: Path, monkeypatch
) -> None:
    """Phase 7b: per-run ``control/exit_code`` files drive per-trial state.

    Mixed exit codes across the wave should produce mixed states — the failed
    orchestrator's trial is ``failed`` with its own exit code; survivors are
    ``completed`` with the recovered objective. The aggregate launcher
    returncode no longer overrides per-run state.
    """
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    config = SweepConfig(
        name="lora-mixed",
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

    # exit_code 1 for the middle trial; rewards only recorded for survivors.
    exit_codes_by_index = {0: 0, 1: 1, 2: 0}
    rewards_by_index = {0: 0.4, 2: 0.3}

    import subprocess as real_subprocess

    real_run = real_subprocess.run

    def fake_run(command, env=None, **kwargs):
        if command[:2] == ["git", "rev-parse"] or command[:2] == ["git", "status"]:
            return real_run(command, **kwargs)
        idx = command.index("--runs-dir")
        run_dirs = [Path(p) for p in command[idx + 1].split(":") if p]
        for run_dir in run_dirs:
            trial_index = int(run_dir.name.removeprefix("run_").split("-", 1)[0])
            (run_dir / "control").mkdir(parents=True, exist_ok=True)
            code = exit_codes_by_index[trial_index]
            (run_dir / "control" / "exit_code").write_text(f"{code}\n")
            if code == 0 and trial_index in rewards_by_index:
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "metrics.jsonl").write_text(
                    json.dumps({"step": 1, "reward": rewards_by_index[trial_index]}) + "\n"
                )
        # Aggregate non-zero so the controller's failure-counting branch fires.
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    try:
        run_sweep(config)
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("Expected SystemExit when one orchestrator failed")

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    states_by_index: dict[int, dict] = {}
    for variant in manifest["variants"]:
        trial_idx = int(variant["id"].split("-", 1)[0])
        states_by_index[trial_idx] = json.loads(Path(variant["status_path"]).read_text())

    assert states_by_index[0]["state"] == "completed"
    assert states_by_index[0]["returncode"] == 0
    assert states_by_index[0]["objective"] == 0.4
    assert states_by_index[1]["state"] == "failed"
    assert states_by_index[1]["returncode"] == 1
    assert states_by_index[2]["state"] == "completed"
    assert states_by_index[2]["objective"] == 0.3

    # Manifest summary picks the best across only the completed trials.
    summary = manifest["summary"]
    assert summary["completed"] == 2
    assert summary["best_value"] == 0.4


def test_multi_run_lora_sweep_preserves_pre_marked_pruned_state(
    tmp_path: Path, monkeypatch
) -> None:
    """If status.json already shows ``pruned`` (controller pre-marked it
    before writing evicted.txt), the orchestrator's inevitable non-zero exit
    must not flip it back to ``failed``."""
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    config = SweepConfig(
        name="lora-pruned",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 2,
            "shared": [shared_path],
        },
        parameters={"orchestrator.optim.lr": {"values": [1e-5, 3e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    import subprocess as real_subprocess

    real_run = real_subprocess.run

    def fake_run(command, env=None, **kwargs):
        if command[:2] == ["git", "rev-parse"] or command[:2] == ["git", "status"]:
            return real_run(command, **kwargs)
        idx = command.index("--runs-dir")
        run_dirs = [Path(p) for p in command[idx + 1].split(":") if p]
        # The "pruned" trial is index 0 — its status.json was pre-marked
        # by the controller, then the orchestrator exited non-zero.
        for trial_index, run_dir in enumerate(run_dirs):
            (run_dir / "control").mkdir(parents=True, exist_ok=True)
            if trial_index == 0:
                status_path = run_dir / "status.json"
                status = json.loads(status_path.read_text())
                status["state"] = "pruned"
                status["pruned_reason"] = "test prune"
                status["pruned_at_step"] = 1
                status["pruned_value"] = 0.05
                status["objective"] = None
                status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
                (run_dir / "control" / "exit_code").write_text("1\n")
            else:
                (run_dir / "control" / "exit_code").write_text("0\n")
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "metrics.jsonl").write_text(
                    json.dumps({"step": 1, "reward": 0.6}) + "\n"
                )
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    # The pruned trial doesn't count as a failure, so the surviving trial
    # carries the sweep — no SystemExit since failures==0.
    run_sweep(config)

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    states_by_index: dict[int, dict] = {}
    for variant in manifest["variants"]:
        trial_idx = int(variant["id"].split("-", 1)[0])
        states_by_index[trial_idx] = json.loads(Path(variant["status_path"]).read_text())

    assert states_by_index[0]["state"] == "pruned"
    assert states_by_index[0]["pruned_reason"] == "test prune"
    assert states_by_index[1]["state"] == "completed"
    assert states_by_index[1]["objective"] == 0.6


def test_multi_run_lora_sweep_marks_all_failed_when_exit_codes_missing(
    tmp_path: Path, monkeypatch
) -> None:
    """Launcher death (no per-run exit_code files) is treated as an
    infrastructure failure: every trial without a recorded exit code is
    marked failed with the aggregate returncode (or -1 fallback)."""
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
        # No exit_code files written — simulates launcher dying before any
        # orchestrator started.
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    config = SweepConfig(
        name="lora-launcher-died",
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
        raise AssertionError("Expected SystemExit when launcher exited non-zero")

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    for variant in manifest["variants"]:
        status = json.loads(Path(variant["status_path"]).read_text())
        assert status["state"] == "failed"
        assert status["returncode"] == 2


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
