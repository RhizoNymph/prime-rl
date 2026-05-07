"""Optuna ask/tell driver for sweep trials.

Phase 5 supports the Optuna TPE and Random samplers. Trials run sequentially:
the controller asks Optuna for the next parameter set, materializes a trial,
runs it through the existing scheduler primitives, reads the final objective,
and tells Optuna the result before proposing the next trial. Pruners
(median / ASHA / Hyperband) require intermediate-metric reporting and land in
a follow-up phase.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prime_rl.configs.sweep import (
    ChoiceParameterConfig,
    IntUniformParameterConfig,
    LocalSweepSchedulerConfig,
    LogUniformParameterConfig,
    OptunaStrategyConfig,
    SweepConfig,
    SweepParameterConfig,
    UniformParameterConfig,
)
from prime_rl.sweep.early_stopping import TrialOutcome, TrialOutcomeTracker
from prime_rl.sweep.materialize import Trial, TrialArtifacts, materialize_trial, record_trial_objective
from prime_rl.sweep.metrics import read_final_summary
from prime_rl.sweep.schedulers import _run_with_retries
from prime_rl.sweep.search import parameters_hash, trial_label

if TYPE_CHECKING:  # pragma: no cover
    import optuna


def _import_optuna() -> Any:
    try:
        import optuna  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "Optuna strategy requires the [hpo] extra. Install with: uv sync --extra hpo"
        ) from exc
    return optuna


def _suggest_parameters(
    optuna_trial: optuna.Trial,
    parameters: dict[str, SweepParameterConfig],
) -> dict[str, Any]:
    suggested: dict[str, Any] = {}
    for path, config in parameters.items():
        if isinstance(config, ChoiceParameterConfig):
            suggested[path] = optuna_trial.suggest_categorical(path, config.values)
        elif isinstance(config, UniformParameterConfig):
            suggested[path] = optuna_trial.suggest_float(path, config.min, config.max)
        elif isinstance(config, LogUniformParameterConfig):
            suggested[path] = optuna_trial.suggest_float(path, config.min, config.max, log=True)
        elif isinstance(config, IntUniformParameterConfig):
            suggested[path] = optuna_trial.suggest_int(path, config.min, config.max, step=config.step)
        else:
            raise ValueError(f"Unsupported parameter type for Optuna: {type(config)!r}")
    return suggested


def _build_sampler(optuna: Any, strategy: OptunaStrategyConfig) -> Any:
    if strategy.sampler == "tpe":
        return optuna.samplers.TPESampler(seed=strategy.seed)
    if strategy.sampler == "random":
        return optuna.samplers.RandomSampler(seed=strategy.seed)
    raise ValueError(f"Unsupported Optuna sampler: {strategy.sampler}")


def _create_study(optuna: Any, config: SweepConfig) -> Any:
    strategy = config.strategy
    assert isinstance(strategy, OptunaStrategyConfig)
    assert config.objective is not None  # validated upstream
    direction = "maximize" if config.objective.direction == "maximize" else "minimize"
    return optuna.create_study(
        study_name=strategy.study_name or config.name or "sweep",
        storage=strategy.storage,
        sampler=_build_sampler(optuna, strategy),
        direction=direction,
        load_if_exists=True,
    )


def _make_trial(index: int, parameters: dict[str, Any]) -> Trial:
    trial_id = f"{index:04d}-{parameters_hash(parameters)}"
    label = trial_label(parameters) or trial_id
    return Trial(id=trial_id, label=label, parameters=parameters)


def _load_previous_variants(config: SweepConfig) -> list[dict[str, Any]]:
    manifest_path = config.output_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    return json.loads(manifest_path.read_text()).get("variants", []) or []


def _seed_tracker_from_previous(tracker: TrialOutcomeTracker, previous_variants: list[dict[str, Any]]) -> None:
    for variant in previous_variants:
        status_path = Path(variant.get("status_path", ""))
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text())
        if status.get("state") != "completed":
            continue
        tracker.observe(
            TrialOutcome(
                trial_id=variant.get("id", ""),
                label=variant.get("label", "") or variant.get("id", ""),
                objective=status.get("objective"),
            )
        )


def _variant_status_for_trial_number(
    previous_variants: list[dict[str, Any]],
    trial_number: int,
) -> dict[str, Any] | None:
    """Match an Optuna trial number to its sweep trial via the ``NNNN-...`` id prefix."""
    prefix = f"{trial_number:04d}-"
    for variant in previous_variants:
        if not variant.get("id", "").startswith(prefix):
            continue
        status_path = Path(variant.get("status_path", ""))
        if not status_path.exists():
            return None
        return json.loads(status_path.read_text())
    return None


def _reconcile_running_trials(optuna: Any, study: Any, previous_variants: list[dict[str, Any]]) -> int:
    """Tell Optuna about any RUNNING trials left over from an interrupted run.

    A controller crash between ``study.ask()`` and ``study.tell()`` leaves a
    trial RUNNING in persistent storage forever. On resume we walk those
    trials and:

    - if the matching sweep status.json shows ``completed`` with a finite
      objective, tell Optuna the value so adaptive sampling can use it;
    - otherwise tell ``TrialState.FAIL`` so the slot stops blocking.

    Returns the number of trials reconciled, mostly for tests / logging.
    """
    reconciled = 0
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.RUNNING:
            continue
        status = _variant_status_for_trial_number(previous_variants, trial.number)
        objective: float | None = None
        if status is not None and status.get("state") == "completed":
            value = status.get("objective")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                objective = float(value)
        # study.tell() accepts a trial number or a Trial; FrozenTrial is not
        # accepted, so pass trial.number.
        if objective is not None:
            study.tell(trial.number, objective)
        else:
            study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        reconciled += 1
    return reconciled


def run_optuna_sweep(
    config: SweepConfig,
    write_manifest_with_variants: Any,
    build_variant: Any,
) -> tuple[int, TrialOutcomeTracker | None, list[TrialArtifacts]]:
    """Drive an Optuna study end-to-end.

    Returns ``(failures, tracker, artifacts)`` so the caller can write the
    final manifest summary and exit code in the same shape as the static
    flow. Resume honors persistent storage: previously consumed slots in
    ``study.trials`` are not re-asked, the manifest preserves earlier
    variants, and the tracker is seeded from prior outcomes.
    """
    optuna = _import_optuna()
    strategy = config.strategy
    assert isinstance(strategy, OptunaStrategyConfig)
    assert isinstance(config.scheduler, LocalSweepSchedulerConfig)  # rejected upstream otherwise

    study = _create_study(optuna, config)

    gpu_groups = (
        config.scheduler.gpu_assignment.visible_devices if config.scheduler.gpu_assignment is not None else None
    )
    gpu_group = gpu_groups[0] if gpu_groups else None

    tracker = TrialOutcomeTracker(config.objective, config.early_stopping) if config.objective else None

    previous_variants = _load_previous_variants(config) if config.resume else []
    if config.resume:
        reconciled = _reconcile_running_trials(optuna, study, previous_variants)
        if reconciled:
            print(f"Reconciled {reconciled} RUNNING Optuna trial(s) from interrupted resume.")
        if tracker is not None:
            _seed_tracker_from_previous(tracker, previous_variants)

    artifacts: list[TrialArtifacts] = []
    failures = 0
    already_consumed = len(study.trials) if config.resume else 0

    for index in range(already_consumed, strategy.num_trials):
        if tracker is not None and tracker.halted:
            break

        optuna_trial = study.ask()
        params = _suggest_parameters(optuna_trial, config.parameters)
        trial = _make_trial(index, params)

        try:
            artifact = materialize_trial(config, trial)
        except Exception as exc:
            # Sampled parameters failed target-config validation. Mark the
            # asked trial failed in Optuna so persistent storage doesn't
            # leak a RUNNING slot, then continue per failure policy.
            study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
            failures += 1
            if not config.continue_on_failure:
                raise SystemExit(1) from exc
            print(f"Optuna trial {index:04d} failed materialization: {exc}")
            continue

        artifacts.append(artifact)
        write_manifest_with_variants(config, previous_variants + [build_variant(a) for a in artifacts])

        returncode = _run_with_retries(artifact, gpu_group, config.retry_budget)
        objective_value = (
            read_final_summary(artifact.run_dir, config.objective.metric)
            if returncode == 0 and config.objective is not None
            else None
        )
        record_trial_objective(artifact.status_path, objective_value)

        if objective_value is None:
            study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
        else:
            study.tell(optuna_trial, objective_value)

        if returncode != 0:
            failures += 1
            if not config.continue_on_failure:
                raise SystemExit(returncode)

        if tracker is not None:
            outcome = TrialOutcome(trial_id=trial.id, label=trial.label, objective=objective_value)
            if tracker.observe(outcome):
                break

    write_manifest_with_variants(config, previous_variants + [build_variant(a) for a in artifacts])

    return failures, tracker, artifacts


# Re-exported for the controller to update the manifest summary.
def tracker_summary(tracker: TrialOutcomeTracker) -> dict[str, Any]:
    return asdict(tracker.summary())
