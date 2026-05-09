"""End-to-end-ish coverage for the multi_run_lora sweep path.

The full sweep -> rl-multi-run -> trainer/inference/orchestrator stack
requires GPUs and a model; here we monkeypatch the parts that need real
infra (``validate_target_config`` for the resolved orchestrator config,
``subprocess.Popen`` for the rl-multi-run invocation via the shared
``fake_multi_run_popen`` fixture in conftest.py) and assert on the
sweep-side contract: layout on disk, the command shape, and per-trial
objective recording from each run's metrics.jsonl sidecar.

Phase 7e: static (grid/random) sweeps run through the same continuous-flow
loop as Optuna sweeps. The launcher invocation gains ``--watch-slots`` and
the controller writes ``<shared_dir>/control/done`` once every trial has
been settled.
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


def test_multi_run_lora_sweep_end_to_end(tmp_path: Path, monkeypatch, fake_multi_run_popen) -> None:
    """A grid over orchestrator.optim.lr produces N run dirs, invokes
    rl-multi-run once with --watch-slots, and records per-trial objectives
    from each run's metrics.jsonl."""
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    fake_multi_run_popen.rewards_by_index = {0: 0.4, 1: 0.7, 2: 0.3}

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

    run_sweep(config)

    # Single launcher invocation, three initial run dirs, watch-slots enabled.
    assert len(fake_multi_run_popen.instances) == 1
    cmd = fake_multi_run_popen.instances[0].command
    assert cmd[0] == "rl-multi-run"
    assert "--watch-slots" in cmd
    assert "--runs-dir" in cmd
    runs_dir_arg = cmd[cmd.index("--runs-dir") + 1]
    initial_run_dirs = runs_dir_arg.split(":")
    assert len(initial_run_dirs) == 3
    for piece in initial_run_dirs:
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


def test_multi_run_lora_sweep_streams_when_grid_exceeds_concurrency(
    tmp_path: Path, monkeypatch, fake_multi_run_popen
) -> None:
    """Phase 7e: grid with 4 trials and max_concurrent_runs=2 streams.

    Pre-7e this combination was rejected with ``num_trials > max_concurrent_runs``.
    Continuous-flow lifts that: the controller materializes only 2 initial run
    dirs in the launcher's --runs-dir, then drops the next trial's status into
    place as each slot frees and the launcher's slot-watch loop spawns the
    new orchestrator.
    """
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    fake_multi_run_popen.rewards_by_index = {0: 0.1, 1: 0.4, 2: 0.7, 3: 0.3}

    config = SweepConfig(
        name="lora-streaming",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 2,
            "shared": [shared_path],
        },
        parameters={"orchestrator.optim.lr": {"values": [1e-5, 3e-5, 1e-4, 5e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(config)

    assert len(fake_multi_run_popen.instances) == 1
    cmd = fake_multi_run_popen.instances[0].command
    runs_dir_arg = cmd[cmd.index("--runs-dir") + 1]
    initial_run_dirs = runs_dir_arg.split(":")
    assert len(initial_run_dirs) == 2  # initial cohort sized to max_concurrent_runs
    assert "--watch-slots" in cmd

    # Done marker written so the launcher tears down.
    shared_dir = tmp_path / "study" / "shared"
    assert (shared_dir / "control" / "done").exists()

    # All 4 trials materialized.
    materialized = sorted(p.name for p in shared_dir.glob("run_*"))
    assert len(materialized) == 4

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    summary = manifest["summary"]
    assert summary["completed"] == 4
    assert summary["best_value"] == 0.7


def test_multi_run_lora_sweep_attributes_failures_per_orchestrator(
    tmp_path: Path, monkeypatch, fake_multi_run_popen
) -> None:
    """Phase 7b/7e: per-run ``control/exit_code`` files drive per-trial state.

    Mixed exit codes across the cohort produce mixed states — the failed
    orchestrator's trial is ``failed`` with its own exit code; survivors are
    ``completed`` with the recovered objective.
    """
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    fake_multi_run_popen.exit_codes_by_index = {0: 0, 1: 1, 2: 0}
    fake_multi_run_popen.rewards_by_index = {0: 0.4, 2: 0.3}

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
        retry_budget=0,  # disable auto-retry so the failure surfaces directly
        wandb=None,
    )

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

    summary = manifest["summary"]
    assert summary["completed"] == 2
    assert summary["best_value"] == 0.4


def test_multi_run_lora_static_fail_fast_writes_done_before_wait(
    tmp_path: Path, monkeypatch, fake_multi_run_popen
) -> None:
    """Static continuous-flow fail-fast exits still signal the launcher."""
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    fake_multi_run_popen.exit_codes_by_index = {0: 1}
    fake_multi_run_popen.assert_done_on_wait = True

    config = SweepConfig(
        name="lora-static-fail-fast",
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
        retry_budget=0,
        continue_on_failure=False,
        wandb=None,
    )

    try:
        run_sweep(config)
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("Expected SystemExit when static fail-fast trial failed")

    assert (tmp_path / "study" / "shared" / "control" / "done").exists()
    assert len(fake_multi_run_popen.instances) == 1


def test_multi_run_lora_sweep_resume_skips_already_completed_trials(
    tmp_path: Path, monkeypatch, fake_multi_run_popen
) -> None:
    """Phase 7c: ``--resume`` only relaunches trials whose prior status is
    pending. Completed/pruned/failed trials keep their preserved artifacts
    and stay out of the next ``rl-multi-run`` invocation.
    """
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    base_kwargs = dict(
        name="lora-resume",
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

    fake_multi_run_popen.rewards_by_index = {0: 0.4, 1: 0.7, 2: 0.3}
    run_sweep(SweepConfig(**base_kwargs))

    # Confirm run 1 left every trial completed.
    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    for variant in manifest["variants"]:
        status = json.loads(Path(variant["status_path"]).read_text())
        assert status["state"] == "completed"

    # Reset the fake's instance list before resume so we can assert on what
    # the resume run does (not the prior run's invocation).
    fake_multi_run_popen.instances = []

    import os

    from prime_rl.sweep import multi_run as multi_run_mod

    shared_dir = tmp_path / "study" / "shared"
    (shared_dir / ".launcher.pid").write_text(f"{os.getpid()}\n")
    (shared_dir / ".launcher.heartbeat").touch()
    monkeypatch.setattr(multi_run_mod, "_wait_for_pid_exit", lambda *_a, **_kw: None)

    run_sweep(SweepConfig(**base_kwargs, resume=True))

    # No rl-multi-run invocation: every trial was already terminal.
    assert fake_multi_run_popen.instances == []
    assert (shared_dir / "control" / "done").exists()

    manifest_after = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert manifest_after["summary"]["best_value"] == 0.7


def test_multi_run_lora_sweep_live_attach_skips_popen(
    tmp_path: Path, monkeypatch
) -> None:
    """Phase 7e: a fresh ``.launcher.pid`` + heartbeat triggers live-attach.

    The controller skips ``subprocess.Popen`` and instead drops new
    ``run_*/control/orch.toml`` into the shared dir, relying on the existing
    launcher's watch-slots loop. The test simulates that loop via the
    ``time.sleep`` monkeypatch: each tick scans for new run dirs and writes
    their exit_code (mirroring what the real launcher would do).
    """
    import os

    from prime_rl.sweep import multi_run as multi_run_mod

    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    shared_dir = tmp_path / "study" / "shared"
    shared_dir.mkdir(parents=True)

    # Pre-write PID + heartbeat so _detect_running_launcher returns the
    # current test process's PID (which is definitely alive).
    (shared_dir / ".launcher.pid").write_text(f"{os.getpid()}\n")
    (shared_dir / ".launcher.heartbeat").touch()

    seen: set[str] = set()

    def simulate_launcher_tick(*_args, **_kwargs):
        for run_dir in sorted(shared_dir.glob("run_*")):
            if run_dir.name in seen:
                continue
            if not (run_dir / "control" / "orch.toml").exists():
                continue
            (run_dir / "metrics.jsonl").write_text(
                json.dumps({"step": 1, "reward": 0.5}) + "\n"
            )
            (run_dir / "control" / "exit_code").write_text("0\n")
            seen.add(run_dir.name)

    import subprocess as real_subprocess

    real_popen = real_subprocess.Popen
    rl_popen_calls: list = []

    def selective_popen(command, **kwargs):
        if command and command[0] == "rl-multi-run":
            rl_popen_calls.append(list(command))
            raise AssertionError(
                f"Popen for rl-multi-run should not be called on live-attach: {command}"
            )
        # Non-rl-multi-run commands (git, etc.) use the real Popen.
        return real_popen(command, **kwargs)

    monkeypatch.setattr(multi_run_mod.time, "sleep", simulate_launcher_tick)
    monkeypatch.setattr(multi_run_mod.subprocess, "Popen", selective_popen)
    # The "live launcher" is just os.getpid() — without this patch the
    # controller would block in _wait_for_pid_exit waiting for the test
    # process itself to exit.
    monkeypatch.setattr(multi_run_mod, "_wait_for_pid_exit", lambda *_a, **_kw: None)

    config = SweepConfig(
        name="lora-live-attach",
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
        resume=True,
        wandb=None,
    )

    run_sweep(config)

    # Live-attach: no rl-multi-run Popen call, done marker dropped so the
    # attached launcher knows it can drain.
    assert rl_popen_calls == []
    assert (shared_dir / "control" / "done").exists()

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    summary = manifest["summary"]
    assert summary["completed"] == 2


def test_multi_run_lora_dry_run_lists_run_dirs(tmp_path: Path, monkeypatch, capsys) -> None:
    """dry_run materializes the layout but does not invoke rl-multi-run."""
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

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

    out = capsys.readouterr().out
    assert "Materialized 2 run dir(s)" in out
    assert (tmp_path / "study" / "shared").exists()
    assert sum(1 for p in (tmp_path / "study" / "shared").glob("run_*")) == 2
