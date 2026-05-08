"""Wave-mode Optuna driver for the multi_run_lora scheduler (Phase 7b).

The full multi-run + Optuna stack exercises trainer + inference + N
orchestrators in one ``rl-multi-run`` invocation per wave; here we stub
``validate_target_config`` for the resolved orchestrator config and
``subprocess.Popen`` for the launcher invocation, plus replace the Optuna
study with a controllable stub. The contract under test is:

- One ``rl-multi-run`` invocation per wave, sized to ``max_concurrent_runs``.
- Mid-wave pruning writes ``<run_dir>/control/evicted.txt`` and records
  ``state="pruned"`` in ``status.json`` *before* the orchestrator exits.
- After each wave, ``study.tell`` is called with the right state per trial:
  PRUNED for pruned trials, the recovered objective for completed trials,
  FAIL for trials whose orchestrator exited non-zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tomli
import tomli_w

pytest.importorskip("optuna")

import optuna  # noqa: E402

from prime_rl.configs.sweep import SweepConfig  # noqa: E402
from prime_rl.sweep.controller import run_sweep  # noqa: E402


def write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def _stub_validate_target_config(monkeypatch) -> None:
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


class _TrialStub:
    """Minimal Optuna ``Trial`` stand-in.

    Captures ``suggest_*`` calls (returns the first option for determinism),
    records ``report`` calls, and lets the test toggle ``should_prune``.
    """

    def __init__(self, number: int) -> None:
        self.number = number
        self.params: dict[str, Any] = {}
        self.reports: list[tuple[float, int]] = []
        self.should_prune_returns = False

    def suggest_categorical(self, name: str, choices):
        v = choices[0]
        self.params[name] = v
        return v

    def suggest_float(self, name: str, low, high, log=False):
        v = low
        self.params[name] = v
        return v

    def suggest_int(self, name: str, low, high, step=1):
        v = low
        self.params[name] = v
        return v

    def report(self, value: float, step: int) -> None:
        self.reports.append((value, step))

    def should_prune(self) -> bool:
        return self.should_prune_returns


class _StudyStub:
    """In-memory Optuna ``Study`` stand-in.

    The wave driver only uses ``ask``, ``tell``, and the
    ``optuna.trial.TrialState`` enum (which is the real one). Tracking trials
    in an instance attribute lets the test inspect what got asked / told.
    """

    def __init__(self) -> None:
        self.asked: list[_TrialStub] = []
        self.tells: list[tuple[_TrialStub, Any, Any]] = []
        self._on_ask = None  # optional callback to mark trials as prune-bound

    def ask(self) -> _TrialStub:
        trial = _TrialStub(number=len(self.asked))
        self.asked.append(trial)
        if self._on_ask is not None:
            self._on_ask(trial)
        return trial

    def tell(self, trial: _TrialStub, value=None, state=None) -> None:
        self.tells.append((trial, value, state))


class _FakePopen:
    """Fake ``subprocess.Popen`` that simulates a wave's ``rl-multi-run`` run.

    On construction it parses ``--runs-dir`` from the command and seeds each
    run dir's ``metrics.jsonl`` with a single ``(step=1, reward=...)`` row so
    the wave driver's poll loop has something to ``report`` on its first
    tick. ``poll()`` returns ``None`` for the first call (the poll loop runs
    once) and ``0`` thereafter; on the transition it writes
    ``<run_dir>/control/exit_code`` for every survivor, mirroring what the
    real launcher does in production.

    Non-``rl-multi-run`` invocations (e.g. ``git rev-parse`` for the manifest's
    git metadata) are delegated to the real ``subprocess.Popen`` — patching
    ``multi_run.subprocess.Popen`` patches the stdlib module attribute, so
    every Popen call in the process flows through here while the test runs.
    """

    instances: list["_FakePopen"] = []
    _real_popen = None  # populated by _install_fake_optuna_runtime

    def __new__(cls, command, **kwargs):
        if not command or command[0] != "rl-multi-run":
            assert cls._real_popen is not None  # set up by the test fixture
            return cls._real_popen(command, **kwargs)
        instance = super().__new__(cls)
        return instance

    def __init__(self, command, **kwargs) -> None:
        if not command or command[0] != "rl-multi-run":
            return  # __new__ delegated to real Popen; nothing to init
        _FakePopen.instances.append(self)
        self.command = list(command)
        idx = command.index("--runs-dir")
        self.run_dirs = [Path(p) for p in command[idx + 1].split(":") if p]
        self._poll_count = 0
        self.returncode: int | None = None

        # Seed metrics so the wave driver has something to report on the
        # first poll tick. Reward 0.5 is arbitrary; the prune decision in
        # tests comes from the trial stub's should_prune flag, not the value.
        for run_dir in self.run_dirs:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "metrics.jsonl").write_text(
                json.dumps({"step": 1, "reward": 0.5}) + "\n"
            )

    def poll(self) -> int | None:
        self._poll_count += 1
        if self._poll_count == 1:
            return None
        # On the second poll, simulate the launcher writing exit_codes and
        # exiting cleanly. Pruned trials had their orch exit with non-zero
        # in production; the controller pre-marked them so reconcile won't
        # read the exit_code anyway, but we still write 1 for diagnostics.
        for run_dir in self.run_dirs:
            (run_dir / "control").mkdir(parents=True, exist_ok=True)
            status_path = run_dir / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            code = "1\n" if status.get("state") == "pruned" else "0\n"
            (run_dir / "control" / "exit_code").write_text(code)
        self.returncode = 0
        return 0

    def wait(self) -> int:
        if self.returncode is None:
            self.poll()
        return self.returncode  # type: ignore[return-value]


def _install_fake_optuna_runtime(monkeypatch, study: _StudyStub) -> None:
    """Replace the create-study helper, Popen, and time.sleep so the wave
    driver runs synchronously against the fake process."""
    import subprocess as real_subprocess

    from prime_rl.sweep import multi_run as multi_run_mod

    _FakePopen._real_popen = real_subprocess.Popen
    monkeypatch.setattr(multi_run_mod, "_create_study", lambda *a, **kw: study)
    monkeypatch.setattr(multi_run_mod.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(multi_run_mod.time, "sleep", lambda *_a, **_kw: None)
    _FakePopen.instances.clear()


def test_multi_run_optuna_wave_prunes_one_trial_and_completes_others(
    tmp_path: Path, monkeypatch
) -> None:
    """One wave of 3 trials: trial #1 is configured to prune; the other
    two run to completion and report their objective."""
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    study = _StudyStub()

    # Mark the second asked trial as prune-bound the moment Optuna asks for it,
    # so when the wave driver calls should_prune() after report(), we prune it.
    def _on_ask(trial: _TrialStub) -> None:
        if trial.number == 1:
            trial.should_prune_returns = True

    study._on_ask = _on_ask

    _install_fake_optuna_runtime(monkeypatch, study)

    config = SweepConfig(
        name="lora-optuna-prune",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 3,
            "shared": [shared_path],
        },
        strategy={"type": "optuna", "num_trials": 3, "sampler": "random"},
        parameters={"orchestrator.optim.lr": {"values": [1e-5, 3e-5, 1e-4]}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(config)

    # One wave → one Popen.
    assert len(_FakePopen.instances) == 1
    invocation = _FakePopen.instances[0]
    assert invocation.command[0] == "rl-multi-run"
    assert "--runs-dir" in invocation.command

    # Three trials asked, three told.
    assert len(study.asked) == 3
    assert len(study.tells) == 3

    # Trial #1 told as PRUNED, others told a numeric value.
    state_by_number = {trial.number: (value, state) for trial, value, state in study.tells}
    pruned_value, pruned_state = state_by_number[1]
    assert pruned_state == optuna.trial.TrialState.PRUNED
    assert pruned_value is None

    for ok_number in (0, 2):
        value, state = state_by_number[ok_number]
        assert state is None  # natural completion: tell(value) with no state kwarg
        assert value == 0.5

    # The pruned trial has evicted.txt + status.json state="pruned".
    pruned_run_dir = invocation.run_dirs[1]
    assert (pruned_run_dir / "control" / "evicted.txt").exists()
    pruned_status = json.loads((pruned_run_dir / "status.json").read_text())
    assert pruned_status["state"] == "pruned"
    assert pruned_status["pruned_at_step"] == 1
    assert pruned_status["pruned_reason"].startswith("optuna prune")

    # Manifest summary's ``completed`` count tracks trials with a recorded
    # objective; pruned trials report None and are not counted.
    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    summary = manifest["summary"]
    assert summary["completed"] == 2

    # And the per-trial status.json files record one pruned + two completed.
    states_by_index: dict[int, str] = {}
    for variant in manifest["variants"]:
        trial_idx = int(variant["id"].split("-", 1)[0])
        states_by_index[trial_idx] = json.loads(Path(variant["status_path"]).read_text())["state"]
    assert states_by_index == {0: "completed", 1: "pruned", 2: "completed"}


def test_multi_run_optuna_runs_in_waves(tmp_path: Path, monkeypatch) -> None:
    """``num_trials=4, max_concurrent_runs=2`` produces two waves of two
    trials each; each wave is one Popen invocation."""
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    study = _StudyStub()
    _install_fake_optuna_runtime(monkeypatch, study)

    config = SweepConfig(
        name="lora-optuna-waves",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 2,
            "shared": [shared_path],
        },
        strategy={"type": "optuna", "num_trials": 4, "sampler": "random"},
        parameters={"orchestrator.optim.lr": {"values": [1e-5, 3e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(config)

    assert len(_FakePopen.instances) == 2
    for invocation in _FakePopen.instances:
        runs_idx = invocation.command.index("--runs-dir")
        run_dirs = invocation.command[runs_idx + 1].split(":")
        assert len(run_dirs) == 2

    assert len(study.asked) == 4
    assert len(study.tells) == 4
    # Every tell carries the wave's objective value (0.5 by fake convention).
    for _trial, value, state in study.tells:
        assert state is None
        assert value == 0.5
