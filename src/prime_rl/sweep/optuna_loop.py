"""Optuna ask/tell driver for sweep trials.

Phase 5a supports TPE / Random samplers without pruning: trials run to
completion, the controller reads the final objective, and tells Optuna the
result before asking for the next parameter set.

Phase 5b adds pruning. When ``strategy.pruner`` is non-trivial the controller
spawns the trial as a child process group, polls ``metrics.jsonl`` while the
trial runs, calls ``optuna_trial.report(value, step)`` and
``optuna_trial.should_prune()`` between samples, and on a prune signal sends
SIGTERM (escalating to SIGKILL) to the trial's process group. Pruned trials
are recorded with ``state="pruned"`` in ``status.json`` and reported to Optuna
as ``TrialState.PRUNED`` so adaptive sampling can distinguish them from
completed runs and outright failures.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from prime_rl.configs.sweep import (
    AshaPrunerConfig,
    ChoiceParameterConfig,
    HyperbandPrunerConfig,
    IntUniformParameterConfig,
    LocalSweepSchedulerConfig,
    LogUniformParameterConfig,
    MedianPrunerConfig,
    NoPrunerConfig,
    OptunaStrategyConfig,
    PrunerConfig,
    SweepConfig,
    SweepParameterConfig,
    UniformParameterConfig,
)
from prime_rl.sweep.early_stopping import TrialOutcome, TrialOutcomeTracker
from prime_rl.sweep.materialize import (
    Trial,
    TrialArtifacts,
    materialize_trial,
    record_trial_objective,
    record_trial_pruned,
)
from prime_rl.sweep.metrics import read_final_summary, read_intermediate_metric
from prime_rl.sweep.schedulers import _build_env, _run_with_retries, _write_status, utc_now
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


def _build_pruner(optuna: Any, pruner: PrunerConfig) -> Any:
    """Map our discriminated PrunerConfig union to an Optuna pruner instance.

    ``NopPruner`` is Optuna's no-op default; using it explicitly keeps the
    study creation symmetric across pruner choices.
    """
    if isinstance(pruner, NoPrunerConfig):
        return optuna.pruners.NopPruner()
    if isinstance(pruner, MedianPrunerConfig):
        return optuna.pruners.MedianPruner(
            n_startup_trials=pruner.n_startup_trials,
            n_warmup_steps=pruner.n_warmup_steps,
            interval_steps=pruner.interval_steps,
        )
    if isinstance(pruner, AshaPrunerConfig):
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=pruner.min_resource,
            reduction_factor=pruner.reduction_factor,
            min_early_stopping_rate=pruner.min_early_stopping_rate,
        )
    if isinstance(pruner, HyperbandPrunerConfig):
        return optuna.pruners.HyperbandPruner(
            min_resource=pruner.min_resource,
            max_resource=pruner.max_resource,
            reduction_factor=pruner.reduction_factor,
        )
    raise ValueError(f"Unsupported Optuna pruner: {pruner!r}")


def _create_study(optuna: Any, config: SweepConfig) -> Any:
    """Create or reload the Optuna study.

    ``load_if_exists`` is gated on ``config.resume``: a fresh sweep must start
    from an empty optimization history, otherwise old trials would bias the
    sampler and the storage would silently accumulate trials across runs that
    the user thought were independent. With persistent storage and no
    ``resume`` flag, optuna raises ``DuplicatedStudyError`` to surface the
    collision instead of attaching silently.
    """
    strategy = config.strategy
    assert isinstance(strategy, OptunaStrategyConfig)
    assert config.objective is not None  # validated upstream
    direction = "maximize" if config.objective.direction == "maximize" else "minimize"
    return optuna.create_study(
        study_name=strategy.study_name or config.name or "sweep",
        storage=strategy.storage,
        sampler=_build_sampler(optuna, strategy),
        pruner=_build_pruner(optuna, strategy.pruner),
        direction=direction,
        load_if_exists=config.resume,
    )


def _make_trial(index: int, parameters: dict[str, Any]) -> Trial:
    trial_id = f"{index:04d}-{parameters_hash(parameters)}"
    label = trial_label(parameters) or trial_id
    return Trial(id=trial_id, label=label, parameters=parameters)


@dataclass
class _PollingOutcome:
    """Result of running a trial with intermediate-metric polling."""

    state: Literal["completed", "pruned", "failed"]
    returncode: int
    objective: float | None
    pruned_at_step: int | None = None
    pruned_value: float | None = None


def _terminate_process_group(process: subprocess.Popen[bytes], grace_seconds: float = 10.0) -> None:
    """Stop the trial subprocess and any descendants it spawned.

    The sweep launches children with ``start_new_session=True`` so the trial
    and everything it spawns share a process group. We send SIGTERM to the
    whole group, wait briefly for graceful exit, then escalate to SIGKILL.
    Killing only the parent is not enough: the trainer/orchestrator/inference
    children would be reparented to init and keep running, holding GPUs and
    skewing the next trial's measurements.
    """
    if process.poll() is not None:
        return
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass


def _run_trial_with_pruning(
    artifact: TrialArtifacts,
    gpu_group: list[int] | None,
    optuna_trial: optuna.Trial,
    metric: str,
    poll_interval: float,
) -> _PollingOutcome:
    """Spawn a trial and poll its metrics.jsonl for Optuna pruning decisions.

    Each new ``(step, value)`` pair the trial reports is forwarded to
    ``optuna_trial.report``. After every report we ask
    ``optuna_trial.should_prune()``; on True we terminate the trial's process
    group and return a ``pruned`` outcome. On natural exit we read the final
    objective from the same sidecar so the sampler sees the same value the
    rest of the sweep records.

    No retry loop here: the polling driver is meant to be the caller's
    single attempt, with retries handled by the outer loop only when the
    trial actually fails (returncode != 0 and no prune signal).
    """
    env = _build_env(artifact, gpu_group)
    _write_status(
        artifact,
        state="running",
        started_at=utc_now(),
        attempts=1,
        gpu_group=list(gpu_group) if gpu_group is not None else None,
    )
    process = subprocess.Popen(artifact.command, env=env, start_new_session=True)
    last_reported_step: int | None = None

    try:
        while True:
            try:
                returncode = process.wait(timeout=poll_interval)
            except subprocess.TimeoutExpired:
                returncode = None

            sample = read_intermediate_metric(artifact.run_dir, metric)
            if sample is not None:
                step, value = sample
                if last_reported_step is None or step > last_reported_step:
                    optuna_trial.report(value, step)
                    last_reported_step = step
                    if optuna_trial.should_prune():
                        _terminate_process_group(process)
                        record_trial_pruned(artifact.status_path, step, value)
                        return _PollingOutcome(
                            state="pruned",
                            returncode=process.returncode if process.returncode is not None else -1,
                            objective=None,
                            pruned_at_step=step,
                            pruned_value=value,
                        )

            if returncode is not None:
                break
    finally:
        # Belt-and-suspenders: if we exit through an unexpected path the
        # trial process must not be left running.
        _terminate_process_group(process)

    if returncode == 0:
        objective = read_final_summary(artifact.run_dir, metric)
        _write_status(artifact, state="completed", finished_at=utc_now(), returncode=0)
        return _PollingOutcome(state="completed", returncode=0, objective=objective)

    _write_status(artifact, state="failed", finished_at=utc_now(), returncode=returncode)
    return _PollingOutcome(state="failed", returncode=returncode, objective=None)


def _run_trial_with_pruning_and_retries(
    artifact: TrialArtifacts,
    gpu_group: list[int] | None,
    optuna_trial: optuna.Trial,
    metric: str,
    poll_interval: float,
    retry_budget: int,
) -> _PollingOutcome:
    """Wrap the polling driver in the project's retry-on-failure semantics.

    Pruned and completed outcomes return immediately. Only ``failed``
    outcomes (subprocess returncode != 0 with no prune signal) are retried,
    so a deliberately stopped trial is never resurrected.
    """
    attempts = 0
    while True:
        attempts += 1
        outcome = _run_trial_with_pruning(artifact, gpu_group, optuna_trial, metric, poll_interval)
        if outcome.state in ("completed", "pruned"):
            return outcome
        if attempts > retry_budget:
            return outcome


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

        if isinstance(strategy.pruner, NoPrunerConfig):
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
        else:
            outcome = _run_trial_with_pruning_and_retries(
                artifact,
                gpu_group,
                optuna_trial,
                config.objective.metric,
                strategy.poll_interval_seconds,
                config.retry_budget,
            )
            objective_value = outcome.objective
            if outcome.state == "completed":
                record_trial_objective(artifact.status_path, objective_value)
                if objective_value is None:
                    # Completed without a recorded objective (e.g. metric never
                    # logged): treat as a failed observation so adaptive
                    # sampling does not get a phantom value.
                    study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
                else:
                    study.tell(optuna_trial, objective_value)
            elif outcome.state == "pruned":
                # record_trial_pruned already set status.json fields.
                study.tell(optuna_trial, state=optuna.trial.TrialState.PRUNED)
            else:  # failed
                record_trial_objective(artifact.status_path, None)
                study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
                failures += 1
                if not config.continue_on_failure:
                    raise SystemExit(outcome.returncode)

        if tracker is not None:
            tracker_outcome = TrialOutcome(
                trial_id=trial.id,
                label=trial.label,
                objective=objective_value,
            )
            if tracker.observe(tracker_outcome):
                break

    write_manifest_with_variants(config, previous_variants + [build_variant(a) for a in artifacts])

    return failures, tracker, artifacts


# Re-exported for the controller to update the manifest summary.
def tracker_summary(tracker: TrialOutcomeTracker) -> dict[str, Any]:
    return asdict(tracker.summary())
