"""Continuous-flow Optuna driver for the multi_run_lora scheduler (Phase 7c).

The full multi-run + Optuna stack exercises trainer + inference + N
orchestrators in one ``rl-multi-run --watch-slots`` invocation that lives
across the whole sweep; here we stub ``validate_target_config`` for the
resolved orchestrator config and ``subprocess.Popen`` for the launcher
invocation, plus replace the Optuna study with a controllable stub. The
contract under test is:

- One ``rl-multi-run`` invocation for the entire sweep (vs one-per-wave
  in 7b).
- The controller maintains target concurrency = ``max_concurrent_runs``,
  asking Optuna for replacements as slots free.
- Mid-flight pruning still writes ``<run_dir>/control/evicted.txt`` and
  records ``state="pruned"`` in ``status.json`` before the orchestrator
  exits.
- ``study.tell`` is called with the right state per trial: PRUNED for
  pruned trials, the recovered objective for completed trials, FAIL for
  trials whose orchestrator exited non-zero.
- When all trials are settled the controller writes
  ``<shared_dir>/control/done`` and the launcher tears down.
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


def _install_fake_optuna_runtime(monkeypatch, study: _StudyStub) -> None:
    """Override Optuna's ``_create_study`` with the test's stub.

    The Popen patch + time.sleep patch + ``FakeMultiRunPopen`` reset all live
    in the shared ``fake_multi_run_popen`` fixture (conftest.py); this helper
    just supplies the study.
    """
    from prime_rl.sweep import multi_run as multi_run_mod

    monkeypatch.setattr(multi_run_mod, "_create_study", lambda *a, **kw: study)


def test_multi_run_optuna_wave_prunes_one_trial_and_completes_others(
    tmp_path: Path, monkeypatch, fake_multi_run_popen
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
    assert len(fake_multi_run_popen.instances) == 1
    invocation = fake_multi_run_popen.instances[0]
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


def test_multi_run_optuna_runs_continuously(tmp_path: Path, monkeypatch, fake_multi_run_popen) -> None:
    """``num_trials=4, max_concurrent_runs=2``: one launcher, four trials.

    Phase 7c: continuous-flow replaces wave-mode. The controller spawns
    ``rl-multi-run`` once with ``max_concurrent_runs`` initial run dirs
    and feeds replacements into the launcher's slot-watch loop as slots
    free, so we expect exactly one Popen invocation regardless of
    ``num_trials``.
    """
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    study = _StudyStub()
    _install_fake_optuna_runtime(monkeypatch, study)

    config = SweepConfig(
        name="lora-optuna-continuous",
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

    # Single launcher for the whole sweep.
    assert len(fake_multi_run_popen.instances) == 1
    invocation = fake_multi_run_popen.instances[0]
    runs_idx = invocation.command.index("--runs-dir")
    initial_run_dirs = invocation.command[runs_idx + 1].split(":")
    # Initial cohort sized to max_concurrent_runs.
    assert len(initial_run_dirs) == 2
    assert "--watch-slots" in invocation.command

    # The controller must drop the done marker so the launcher tears down.
    shared_dir = tmp_path / "study" / "shared"
    assert (shared_dir / "control" / "done").exists()

    # All 4 trials get materialized as run_* dirs under the shared dir.
    materialized = sorted(p.name for p in shared_dir.glob("run_*"))
    assert len(materialized) == 4

    assert len(study.asked) == 4
    assert len(study.tells) == 4
    for _trial, value, state in study.tells:
        assert state is None
        assert value == 0.5


def test_multi_run_optuna_auto_retries_failed_trials(tmp_path: Path, monkeypatch, fake_multi_run_popen) -> None:
    """Phase 7d-A: a failed orchestrator with retry budget is re-materialized.

    The first attempt's run dir gets exit_code=1 (no metrics). The controller
    must materialize ``run_<id>-r1`` with the same params and add it to the
    live set; the launcher's slot-watch loop picks it up. Optuna sees one ask
    and one tell with the success value — retries are transparent to the
    search backend.
    """
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    study = _StudyStub()
    _install_fake_optuna_runtime(monkeypatch, study)

    # First attempt of every trial fails; the retry succeeds.
    fake_multi_run_popen.fail_first_attempt = True

    config = SweepConfig(
        name="lora-optuna-retry",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 1,
            "shared": [shared_path],
        },
        strategy={"type": "optuna", "num_trials": 1, "sampler": "random"},
        parameters={"orchestrator.optim.lr": {"values": [1e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        retry_budget=1,
        wandb=None,
    )

    run_sweep(config)

    # Optuna sees one logical trial: one ask, one tell with the retry's value.
    assert len(study.asked) == 1
    assert len(study.tells) == 1
    _trial, value, state = study.tells[0]
    assert state is None
    assert value == 0.5

    shared_dir = tmp_path / "study" / "shared"
    materialized = sorted(p.name for p in shared_dir.glob("run_*"))
    # One initial dir + one retry dir.
    assert len(materialized) == 2
    assert any(name.endswith("-r1") for name in materialized)

    # The retry dir's status is completed; the original is failed.
    initial = next(p for p in shared_dir.glob("run_*") if "-r" not in p.name.removeprefix("run_"))
    retry = next(p for p in shared_dir.glob("run_*") if "-r1" in p.name)
    assert json.loads((initial / "status.json").read_text())["state"] == "failed"
    retry_status = json.loads((retry / "status.json").read_text())
    assert retry_status["state"] == "completed"
    assert retry_status["attempts"] == 2


def test_multi_run_optuna_fail_fast_writes_done_before_wait(
    tmp_path: Path, monkeypatch, fake_multi_run_popen
) -> None:
    """Fail-fast exits still signal ``--watch-slots`` before waiting.

    Regression coverage: without the done marker in the controller's
    ``finally`` path, ``proc.wait()`` can block forever after a trial failure
    with ``continue_on_failure = false``.
    """
    shared_path = tmp_path / "shared.toml"
    write_toml(shared_path, {})

    _stub_validate_target_config(monkeypatch)

    study = _StudyStub()
    _install_fake_optuna_runtime(monkeypatch, study)
    fake_multi_run_popen.exit_codes_by_index = {0: 1}
    fake_multi_run_popen.assert_done_on_wait = True

    config = SweepConfig(
        name="lora-optuna-fail-fast",
        entrypoint="rl",
        base=[shared_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "multi_run_lora",
            "max_concurrent_runs": 2,
            "shared": [shared_path],
        },
        strategy={"type": "optuna", "num_trials": 3, "sampler": "random"},
        parameters={"orchestrator.optim.lr": {"values": [1e-5, 3e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        continue_on_failure=False,
        wandb=None,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_sweep(config)

    assert exc_info.value.code == 1
    assert (tmp_path / "study" / "shared" / "control" / "done").exists()
    assert len(fake_multi_run_popen.instances) == 1
