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


def test_optuna_fresh_run_rejects_existing_study_in_storage(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.5])

    base_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 1,
            "sampler": "random",
            "seed": 7,
            "storage": storage_url,
            "study_name": "shared",
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(SweepConfig(**base_kwargs))

    import optuna

    with pytest.raises(optuna.exceptions.DuplicatedStudyError):
        run_sweep(SweepConfig(**base_kwargs))


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


def test_optuna_resume_reconciles_pruned_trial_as_pruned_state(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression: a controller crash between record_trial_pruned() and
    study.tell(PRUNED) leaves the Optuna trial RUNNING in storage even
    though its sweep status.json reads ``state="pruned"``. Resume must
    reconcile that as TrialState.PRUNED, not the FAIL fallback."""
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    storage_url = f"sqlite:///{tmp_path / 'optuna.db'}"
    _install_fake_run(monkeypatch, [0.6, 0.5])

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

    # Simulate the crash window: ask leaks a RUNNING trial, and the sweep's
    # status.json for that trial records the prune decision the controller
    # never got to tell Optuna.
    pending = study.ask()
    pending_index = pending.number
    pending_id = f"{pending_index:04d}-pruned"
    trial_dir = tmp_path / "study" / "trials" / pending_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    status_path = trial_dir / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "state": "pruned",
                "objective": None,
                "pruned_at_step": 5,
                "pruned_value": 0.01,
            }
        )
    )

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

    run_sweep(SweepConfig(**base_kwargs, resume=True))

    study = optuna.load_study(study_name="sweep", storage=storage_url)
    states = [t.state for t in study.trials]
    assert optuna.trial.TrialState.RUNNING not in states
    # Crucial: the orphan with status="pruned" must come back as PRUNED, not
    # FAIL, so the sampler's history correctly reflects deliberate stops.
    assert optuna.trial.TrialState.PRUNED in states
    assert optuna.trial.TrialState.FAIL not in states


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


# ---------------------------------------------------------------------------
# Phase 5b — pruning
# ---------------------------------------------------------------------------


class _FakePopen:
    """Stand-in for ``subprocess.Popen`` used by the pruning driver tests.

    Drops a metrics.jsonl into the run directory on construction so the
    polling reader has data on the very first iteration, then reports a
    configurable returncode after the first ``wait()`` call.
    """

    def __init__(self, command, env=None, returncode: int = 0, rows=None, **kwargs):
        self._returncode_value = returncode
        self._returncode: int | None = None
        self.terminated = False
        self.killed = False
        self.pid = 12345
        self._waits = 0
        if env is not None and rows is not None:
            metrics_path = Path(env["PRIME_RL_SWEEP_METRICS_JSONL"])
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    @property
    def returncode(self) -> int | None:
        return self._returncode

    def poll(self) -> int | None:
        return self._returncode

    def wait(self, timeout=None) -> int:
        self._waits += 1
        # First call to wait blocks the polling loop long enough for one
        # metrics.jsonl read; second call returns the configured returncode.
        if self._waits >= 2:
            self._returncode = self._returncode_value
            return self._returncode_value
        import subprocess

        raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout or 0)


def _patch_popen_for_trials(monkeypatch, factory) -> None:
    """Intercept Popen calls from the optuna pruning driver only.

    git_metadata() and other infrastructure also use subprocess (and thus
    Popen under the hood). Patching Popen globally breaks them; patching the
    ``subprocess.Popen`` attribute the optuna_loop module imports lets us
    target only the pruning driver's spawn site.
    """
    import subprocess as real_subprocess

    real_popen = real_subprocess.Popen

    def dispatch(command, *args, **kwargs):
        # Real Popen for everything that is not a trial subprocess (uv run rl/sft
        # compositions). The trial command shape is `["uv", "run", "rl"|"sft", "@", base, "@", overrides]`.
        if isinstance(command, (list, tuple)) and command and command[0] == "uv":
            return factory(command, *args, **kwargs)
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr("prime_rl.sweep.optuna_loop.subprocess.Popen", dispatch)


def _patch_terminate(monkeypatch, terminated: list[bool]) -> None:
    def fake_terminate(process, grace_seconds=10.0):
        # Mirror the real function's idempotency: a finished process is a
        # no-op so the finally-clause cleanup does not double-count.
        if process.poll() is not None:
            return
        terminated.append(True)
        process._returncode = -15

    monkeypatch.setattr("prime_rl.sweep.optuna_loop._terminate_process_group", fake_terminate)


def test_run_trial_with_pruning_returns_pruned_when_should_prune_fires(tmp_path: Path, monkeypatch) -> None:
    """The polling driver must terminate the process and return PRUNED when
    optuna_trial.should_prune() returns True after a report."""
    from prime_rl.sweep.materialize import Trial, materialize_trial
    from prime_rl.sweep.optuna_loop import _run_trial_with_pruning

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})
    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trial = Trial(id="0000-pruneme", label="pruneme", parameters={"optim.lr": 1e-5})
    artifact = materialize_trial(config, trial)

    rows = [{"step": 1, "reward": 0.05}]

    def popen_factory(*args, **kwargs):
        return _FakePopen(*args, rows=rows, returncode=0, **kwargs)

    _patch_popen_for_trials(monkeypatch, popen_factory)
    terminated: list[bool] = []
    _patch_terminate(monkeypatch, terminated)

    reports: list[tuple[int, float]] = []

    class FakeOptunaTrial:
        def report(self, value, step):
            reports.append((step, value))

        def should_prune(self):
            return True

    outcome = _run_trial_with_pruning(
        artifact,
        gpu_group=None,
        optuna_trial=FakeOptunaTrial(),
        metric="reward",
        poll_interval=0.01,
    )

    assert outcome.state == "pruned"
    assert outcome.pruned_at_step == 1
    assert outcome.pruned_value == 0.05
    assert reports == [(1, 0.05)]
    assert terminated == [True]
    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "pruned"
    assert status["pruned_at_step"] == 1
    assert status["pruned_value"] == 0.05
    assert status["objective"] is None


def test_run_trial_with_pruning_records_objective_on_clean_completion(tmp_path: Path, monkeypatch) -> None:
    """A trial that runs to completion without should_prune firing returns the
    final value from metrics.jsonl and is recorded as completed."""
    from prime_rl.sweep.materialize import Trial, materialize_trial
    from prime_rl.sweep.optuna_loop import _run_trial_with_pruning

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})
    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trial = Trial(id="0000-good", label="good", parameters={"optim.lr": 1e-5})
    artifact = materialize_trial(config, trial)

    rows = [{"step": 1, "reward": 0.1}, {"step": 2, "reward": 0.4}]

    def popen_factory(*args, **kwargs):
        return _FakePopen(*args, rows=rows, returncode=0, **kwargs)

    _patch_popen_for_trials(monkeypatch, popen_factory)
    _patch_terminate(monkeypatch, [])

    class FakeOptunaTrial:
        def report(self, value, step):
            pass

        def should_prune(self):
            return False

    outcome = _run_trial_with_pruning(
        artifact,
        gpu_group=None,
        optuna_trial=FakeOptunaTrial(),
        metric="reward",
        poll_interval=0.01,
    )

    assert outcome.state == "completed"
    assert outcome.objective == 0.4
    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "completed"
    assert status["returncode"] == 0


def test_run_trial_with_pruning_does_not_prune_after_subprocess_exit(tmp_path: Path, monkeypatch) -> None:
    """Regression: a completed trial must not be re-classified as pruned even
    if its final intermediate point would have triggered should_prune().

    The polling loop reads the metric *after* process.wait() returns the real
    return code; without the gate on returncode, a should_prune-eligible last
    value would discard the valid final objective.
    """
    from prime_rl.sweep.materialize import Trial, materialize_trial
    from prime_rl.sweep.optuna_loop import _run_trial_with_pruning

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})
    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trial = Trial(id="0000-late", label="late", parameters={"optim.lr": 1e-5})
    artifact = materialize_trial(config, trial)

    rows = [{"step": 1, "reward": 0.05}]

    class _FakePopenAlreadyExited(_FakePopen):
        """Returncode-on-first-wait variant: process is already done by the
        time the polling loop gets to read its metric."""

        def wait(self, timeout=None) -> int:
            self._returncode = self._returncode_value
            return self._returncode_value

    def popen_factory(*args, **kwargs):
        return _FakePopenAlreadyExited(*args, rows=rows, returncode=0, **kwargs)

    _patch_popen_for_trials(monkeypatch, popen_factory)
    terminated: list[bool] = []
    _patch_terminate(monkeypatch, terminated)

    class FakeOptunaTrial:
        def report(self, value, step):
            pass

        def should_prune(self):
            return True  # pruner WOULD prune, but we are past the natural exit

    outcome = _run_trial_with_pruning(
        artifact,
        gpu_group=None,
        optuna_trial=FakeOptunaTrial(),
        metric="reward",
        poll_interval=0.01,
    )

    assert outcome.state == "completed"
    assert outcome.objective == 0.05
    assert terminated == []  # never had to terminate
    status = json.loads(artifact.status_path.read_text())
    assert status["state"] == "completed"


def test_run_trial_with_pruning_skips_retry_after_intermediate_reports(tmp_path: Path, monkeypatch) -> None:
    """Regression: a failed attempt that already called optuna_trial.report
    must not be retried within the same Optuna trial. Retries on the same
    trial inherit the failed attempt's intermediate values, biasing pruning
    decisions and silently dropping duplicate-step reports."""
    from prime_rl.sweep.materialize import Trial, materialize_trial
    from prime_rl.sweep.optuna_loop import _run_trial_with_pruning_and_retries

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})
    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trial = Trial(id="0000-noretry", label="noretry", parameters={"optim.lr": 1e-5})
    artifact = materialize_trial(config, trial)

    rows = [{"step": 1, "reward": 0.05}]
    spawned = {"n": 0}

    def popen_factory(*args, **kwargs):
        spawned["n"] += 1
        # Returncode 1 => failed attempt.
        return _FakePopen(*args, rows=rows, returncode=1, **kwargs)

    _patch_popen_for_trials(monkeypatch, popen_factory)
    _patch_terminate(monkeypatch, [])

    reports: list[tuple[int, float]] = []

    class FakeOptunaTrial:
        def report(self, value, step):
            reports.append((step, value))

        def should_prune(self):
            return False

    outcome = _run_trial_with_pruning_and_retries(
        artifact,
        gpu_group=None,
        optuna_trial=FakeOptunaTrial(),
        metric="reward",
        poll_interval=0.01,
        retry_budget=3,  # high budget on purpose; the early-exit must override
    )

    assert outcome.state == "failed"
    assert outcome.reports_sent == 1
    # Crucial assertion: only one Popen, despite retry_budget=3, because the
    # first attempt already sent intermediate reports.
    assert spawned["n"] == 1
    assert reports == [(1, 0.05)]


def test_run_with_retries_truncates_metrics_jsonl_between_attempts(tmp_path: Path, monkeypatch) -> None:
    """Regression: if attempt 1 fails after writing higher steps than the
    successful retry, the sidecar must be truncated between attempts so
    read_final_summary returns the retry's value, not the failed run's."""
    from prime_rl.sweep.materialize import Trial, materialize_trial
    from prime_rl.sweep.metrics import read_final_summary
    from prime_rl.sweep.schedulers import _run_with_retries

    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})
    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
        wandb=None,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    trial = Trial(id="0000-retry", label="retry", parameters={"optim.lr": 1e-5})
    artifact = materialize_trial(config, trial)

    metrics_path = artifact.run_dir / "metrics.jsonl"
    artifact.run_dir.mkdir(parents=True, exist_ok=True)

    attempts = {"n": 0}

    def fake_run(command, env=None, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            # Failed attempt wrote higher-step bad value before crashing.
            metrics_path.write_text(json.dumps({"step": 50, "reward": 0.01}) + "\n")
            return SimpleNamespace(returncode=1)
        # Successful retry writes a lower-step good value. Without truncation,
        # read_final_summary would see step=50 reward=0.01 from the failed run
        # and report that instead of step=10 reward=0.9.
        existing = metrics_path.read_text() if metrics_path.exists() else ""
        metrics_path.write_text(existing + json.dumps({"step": 10, "reward": 0.9}) + "\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    returncode = _run_with_retries(artifact, gpu_group=None, retry_budget=1)

    assert returncode == 0
    assert attempts["n"] == 2
    # Truncation between attempts must wipe the failed run's row before the
    # successful retry writes its own line.
    assert read_final_summary(artifact.run_dir, "reward") == 0.9


def test_optuna_sweep_with_median_pruner_runs_to_completion(tmp_path: Path, monkeypatch) -> None:
    """End-to-end: sweep with median pruner enabled runs through the pruning
    code path. Trials with monotonically-improving metrics should not be
    pruned, so all trials complete successfully."""
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    final_rewards = iter([0.4, 0.7, 0.9])

    def popen_factory(*args, **kwargs):
        reward = next(final_rewards)
        rows = [{"step": 1, "reward": reward}]
        return _FakePopen(*args, rows=rows, returncode=0, **kwargs)

    _patch_popen_for_trials(monkeypatch, popen_factory)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={
            "type": "optuna",
            "num_trials": 3,
            "sampler": "random",
            "seed": 7,
            "pruner": {"type": "median", "n_startup_trials": 1, "n_warmup_steps": 0},
            "poll_interval_seconds": 0.01,
        },
        parameters={"optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(config)

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert manifest["strategy"]["pruner"]["type"] == "median"
    assert len(manifest["variants"]) == 3
    summary = manifest["summary"]
    # All three trials produced a valid objective; pruner only fires when the
    # trajectory is below the median, which is impossible with one prior
    # completion + an improving series.
    assert summary["completed"] == 3
    assert summary["best_value"] == 0.9
