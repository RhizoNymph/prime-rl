"""Sweep-side runtime helpers for shared-trainer LoRA multi-run sweeps.

The trainer (``prime_rl/trainer/runs.py:MultiRunManager``) writes
``<run_dir>/control/evicted.txt`` to evict a run when it's about to lose its
LoRA slot. The orchestrator (``prime_rl/orchestrator/orchestrator.py``) polls
that same file at the top of each training loop iteration and exits.

Phase 7b adds a third writer: the sweep controller itself, when an Optuna
sampler decides one of the in-flight trials should be pruned. This module is
the bridge plus the wave driver that runs Optuna against ``rl-multi-run``.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prime_rl.configs.sweep import (
    MultiRunLoRASchedulerConfig,
    OptunaStrategyConfig,
    SweepConfig,
)
from prime_rl.sweep.early_stopping import TrialOutcome, TrialOutcomeTracker
from prime_rl.sweep.materialize import (
    Trial,
    TrialArtifacts,
    materialize_multi_run_trial,
    multi_run_shared_dir,
    record_trial_objective,
)
from prime_rl.sweep.metrics import read_final_summary, read_intermediate_metric
from prime_rl.sweep.optuna_loop import (
    _create_study,
    _import_optuna,
    _make_trial,
    _suggest_parameters,
)
from prime_rl.sweep.schedulers import (
    _read_status,
    _write_status,
    build_multi_run_command,
    reconcile_multi_run_artifact,
    utc_now,
)

if TYPE_CHECKING:  # pragma: no cover
    import optuna

EVICTED_FILENAME = "evicted.txt"


def prune_run(
    run_dir: Path,
    reason: str,
    *,
    step: int | None = None,
    value: float | None = None,
) -> None:
    """Pre-mark the trial pruned in ``status.json`` then write ``evicted.txt``.

    Order matters: the orchestrator's eviction handler raises ``RuntimeError``
    and the orchestrator exits non-zero. If we wrote ``evicted.txt`` first and
    crashed before updating ``status.json``, the launcher's exit-code
    reconciliation would misclassify the deliberately-pruned trial as
    ``failed`` (it has no way to know the eviction was a sampler decision
    rather than a slot-pressure eviction from the trainer).

    ``step`` and ``value`` are recorded on the status when the caller has them
    — Optuna prunes know both — but they're optional so callers without that
    context (manual prune, future heuristics) can still mark the trial pruned.
    """
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text())
    status["state"] = "pruned"
    status["pruned_reason"] = reason
    if step is not None:
        status["pruned_at_step"] = int(step)
    if value is not None:
        status["pruned_value"] = float(value)
    # Surfacing pruned trials with a None objective keeps the manifest
    # summary's best-value computation symmetric with single-trial pruning
    # (see materialize.record_trial_pruned).
    status["objective"] = None
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

    control_dir = run_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    (control_dir / EVICTED_FILENAME).write_text(reason + "\n")


def _poll_wave_for_pruning(
    optuna: Any,
    proc: subprocess.Popen[bytes],
    artifacts: list[TrialArtifacts],
    optuna_trials: list[optuna.Trial],
    metric: str,
    poll_interval: float,
) -> None:
    """While ``rl-multi-run`` runs, drive Optuna's report/should_prune for each artifact.

    On each tick:

    1. For every artifact whose status is not already ``pruned``, read the
       latest ``(step, value)`` from its ``metrics.jsonl`` sidecar.
    2. If we've never reported this step (or any step at all) for this trial,
       call ``optuna_trial.report(value, step)`` and check ``should_prune``.
    3. On a prune signal, write ``status.json`` + ``evicted.txt`` so the
       orchestrator winds down. The trainer's MultiRunManager picks the same
       file up on its next ``discover_runs()`` cycle and frees the LoRA slot.

    The trainer-side eviction handles process termination — we never SIGTERM
    the orchestrator ourselves. Survivors keep running until the wave's
    ``rl-multi-run`` exits naturally.
    """
    last_step: dict[Path, int | None] = {a.run_dir: None for a in artifacts}

    while proc.poll() is None:
        for artifact, optuna_trial in zip(artifacts, optuna_trials):
            try:
                status = _read_status(artifact)
            except FileNotFoundError:
                continue
            if status.get("state") == "pruned":
                continue

            sample = read_intermediate_metric(artifact.run_dir, metric)
            if sample is None:
                continue
            step, value = sample
            prev = last_step[artifact.run_dir]
            if prev is not None and step <= prev:
                continue

            optuna_trial.report(value, step)
            last_step[artifact.run_dir] = step

            if optuna_trial.should_prune():
                prune_run(
                    artifact.run_dir,
                    reason=f"optuna prune at step {step}",
                    step=step,
                    value=value,
                )
        time.sleep(poll_interval)


def _tell_wave_results(
    optuna: Any,
    study: optuna.Study,
    artifacts: list[TrialArtifacts],
    optuna_trials: list[optuna.Trial],
    metric: str,
    aggregate_returncode: int,
) -> tuple[int, list[float | None]]:
    """Reconcile per-trial state and tell Optuna each trial's result.

    Returns ``(failures, objectives)`` where ``objectives[i]`` is the recorded
    objective for ``artifacts[i]`` (or ``None`` for pruned/failed trials).
    The caller folds those into the ``TrialOutcomeTracker`` for the manifest
    summary.
    """
    finished_at = utc_now()
    failures = 0
    objectives: list[float | None] = []

    for artifact, optuna_trial in zip(artifacts, optuna_trials):
        state = reconcile_multi_run_artifact(
            artifact, aggregate_returncode=aggregate_returncode, finished_at=finished_at
        )

        objective: float | None = None
        if state == "completed":
            objective = read_final_summary(artifact.run_dir, metric)
            record_trial_objective(artifact.status_path, objective)
            if objective is None:
                # Clean exit but the metric never showed up — Optuna learned
                # nothing from this slot. Tell FAIL and count it as a sweep
                # failure (mirrors the single-trial Optuna driver).
                study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
                failures += 1
            else:
                study.tell(optuna_trial, objective)
        elif state == "pruned":
            study.tell(optuna_trial, state=optuna.trial.TrialState.PRUNED)
        else:  # failed
            record_trial_objective(artifact.status_path, None)
            study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
            failures += 1

        objectives.append(objective)

    return failures, objectives


def run_multi_run_optuna_sweep(
    config: SweepConfig,
    *,
    write_manifest_with_variants: Any,
    build_variant: Any,
) -> tuple[int, TrialOutcomeTracker | None, list[TrialArtifacts]]:
    """Drive an Optuna study against ``rl-multi-run`` in waves.

    Each wave asks Optuna for ``min(max_concurrent_runs, remaining)`` trials,
    materializes them as ``run_*`` dirs under the shared trainer dir, spawns
    one ``rl-multi-run`` invocation, polls each run's ``metrics.jsonl`` for
    Optuna ``report``/``should_prune`` decisions, and finally tells Optuna
    each trial's result.

    Slot replacement is intentionally not supported: a slot freed mid-wave
    by pruning sits idle until the wave finishes. True slot replacement
    needs ``rl-multi-run`` to accept new run dirs over the wire (Phase 7c).
    """
    optuna = _import_optuna()
    strategy = config.strategy
    scheduler = config.scheduler
    assert isinstance(strategy, OptunaStrategyConfig)
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)
    assert config.objective is not None  # validated upstream

    study = _create_study(optuna, config)
    metric = config.objective.metric
    wave_size = scheduler.max_concurrent_runs
    total = strategy.num_trials
    poll_interval = strategy.poll_interval_seconds

    tracker = TrialOutcomeTracker(config.objective, config.early_stopping)
    shared_dir = multi_run_shared_dir(config)
    shared_dir.mkdir(parents=True, exist_ok=True)

    all_artifacts: list[TrialArtifacts] = []
    failures = 0
    submitted = 0

    while submitted < total:
        if tracker.halted:
            break
        this_wave = min(wave_size, total - submitted)

        # 1. Ask Optuna for `this_wave` trials and materialize each.
        # Failed materializations are dropped from the wave (and reported as
        # Optuna FAIL); survivors stay paired so the poll loop and reconcile
        # step have aligned (optuna_trial, artifact) lists.
        wave_pairs: list[tuple[optuna.Trial, TrialArtifacts]] = []
        for offset in range(this_wave):
            optuna_trial = study.ask()
            params = _suggest_parameters(optuna_trial, config.parameters)
            sweep_trial = _make_trial(submitted + offset, params)
            try:
                artifact = materialize_multi_run_trial(config, sweep_trial, scheduler)
            except Exception as exc:
                study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
                failures += 1
                if not config.continue_on_failure:
                    raise SystemExit(1) from exc
                print(f"Optuna trial {sweep_trial.id} failed materialization: {exc}")
                continue
            wave_pairs.append((optuna_trial, artifact))

        # If every trial in the wave failed materialization there's nothing
        # to launch; advance the counter and try the next wave.
        if not wave_pairs:
            submitted += this_wave
            continue

        wave_optuna_trials = [pair[0] for pair in wave_pairs]
        wave_artifacts = [pair[1] for pair in wave_pairs]
        all_artifacts.extend(wave_artifacts)
        write_manifest_with_variants(config, [build_variant(a) for a in all_artifacts])

        started = utc_now()
        for artifact in wave_artifacts:
            _write_status(artifact, state="running", started_at=started, attempts=1, gpu_group=None)

        # 2. Spawn rl-multi-run for this wave.
        command = build_multi_run_command(wave_artifacts, scheduler.shared, shared_dir)
        proc = subprocess.Popen(command)

        try:
            _poll_wave_for_pruning(
                optuna, proc, wave_artifacts, wave_optuna_trials, metric, poll_interval
            )
        finally:
            proc.wait()

        # 4. Reconcile per-trial state and tell Optuna.
        wave_failures, objectives = _tell_wave_results(
            optuna, study, wave_artifacts, wave_optuna_trials, metric, proc.returncode
        )
        failures += wave_failures

        if wave_failures > 0 and not config.continue_on_failure:
            raise SystemExit(proc.returncode if proc.returncode != 0 else 1)

        # 5. Fold objectives into the tracker for early stopping + summary.
        for artifact, objective in zip(wave_artifacts, objectives):
            tracker.observe(
                TrialOutcome(
                    trial_id=artifact.trial.id,
                    label=artifact.trial.label,
                    objective=objective,
                )
            )
            if tracker.halted:
                break

        submitted += this_wave

    write_manifest_with_variants(config, [build_variant(a) for a in all_artifacts])
    return failures, tracker, all_artifacts


# Re-exported for the controller to update the manifest summary.
def tracker_summary(tracker: TrialOutcomeTracker) -> dict[str, Any]:
    return asdict(tracker.summary())
