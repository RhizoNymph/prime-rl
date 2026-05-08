"""Sweep-side runtime helpers for shared-trainer LoRA multi-run sweeps.

The trainer (``prime_rl/trainer/runs.py:MultiRunManager``) writes
``<run_dir>/control/evicted.txt`` to evict a run when it's about to lose its
LoRA slot. The orchestrator (``prime_rl/orchestrator/orchestrator.py``) polls
that same file at the top of each training loop iteration and exits.

Phase 7b added a third writer: the sweep controller itself, when an Optuna
sampler decides one of the in-flight trials should be pruned.

Phase 7c reshapes the Optuna driver from wave-based to continuous-flow:
``rl-multi-run --watch-slots`` runs once for the whole sweep and grows new
orchestrators on demand as the controller drops fresh ``run_*/control/orch.toml``
files into the shared dir. The controller's loop maintains target concurrency
of ``max_concurrent_runs`` live trials, asking Optuna for a replacement the
moment any slot frees instead of waiting for an entire wave to finish.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prime_rl.configs.sweep import (
    MultiRunLoRASchedulerConfig,
    OptunaStrategyConfig,
    SweepConfig,
)
from prime_rl.sweep.early_stopping import TrialOutcome, TrialOutcomeTracker
from prime_rl.sweep.materialize import (
    TrialArtifacts,
    materialize_multi_run_trial,
    multi_run_shared_dir,
    record_trial_objective,
)
from prime_rl.sweep.metrics import read_final_summary, read_intermediate_metric
from prime_rl.sweep.optuna_loop import (
    _create_study,
    _import_optuna,
    _load_previous_variants,
    _make_trial,
    _reconcile_running_trials,
    _seed_tracker_from_previous,
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
DONE_MARKER_NAME = "done"
EXIT_CODE_FILENAME = "exit_code"
LAUNCHER_PID_FILENAME = ".launcher.pid"
LAUNCHER_HEARTBEAT_FILENAME = ".launcher.heartbeat"
DEFAULT_HEARTBEAT_TOLERANCE_SECONDS = 60.0


def _process_alive(pid: int) -> bool:
    """``os.kill(pid, 0)`` raises ``ProcessLookupError`` if the PID is dead.

    Permission errors (e.g. PID belongs to another user) count as "alive"
    here because we still don't want to spawn a duplicate launcher in that
    case — the user almost certainly hit a misconfiguration we should
    surface, not silently overwrite.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _detect_running_launcher(
    shared_dir: Path,
    *,
    max_age_seconds: float = DEFAULT_HEARTBEAT_TOLERANCE_SECONDS,
) -> int | None:
    """Return the PID of a still-running ``rl-multi-run --watch-slots``, or None.

    The launcher writes ``.launcher.pid`` at startup and refreshes
    ``.launcher.heartbeat`` on each watch-slots tick. Resume considers the
    launcher live when both files exist, the PID is alive, and the heartbeat
    is within ``max_age_seconds`` (covers a slow tick + clock skew).

    A live launcher means the controller can attach via the file protocol
    (drop ``run_*/control/orch.toml`` and the launcher's slot-watch loop
    spawns the orchestrator) instead of re-spawning the trainer/inference
    stack. Anything stale or absent → fall back to stop+resume.
    """
    pid_path = shared_dir / LAUNCHER_PID_FILENAME
    heartbeat_path = shared_dir / LAUNCHER_HEARTBEAT_FILENAME
    if not pid_path.exists() or not heartbeat_path.exists():
        return None
    try:
        pid = int(pid_path.read_text().strip())
    except (OSError, ValueError):
        return None
    if not _process_alive(pid):
        return None
    try:
        age = time.time() - heartbeat_path.stat().st_mtime
    except OSError:
        return None
    if age > max_age_seconds:
        return None
    return pid


def _wait_for_pid_exit(pid: int, *, poll_interval: float = 0.5) -> None:
    """Block until ``pid`` is no longer alive."""
    while _process_alive(pid):
        time.sleep(poll_interval)


def _launcher_is_alive(
    proc: subprocess.Popen[bytes] | None, existing_pid: int | None
) -> bool:
    """True iff the launcher (owned or attached) is still running."""
    if proc is not None:
        return proc.poll() is None
    if existing_pid is not None:
        return _process_alive(existing_pid)
    return False


def _wait_for_launcher_exit(
    proc: subprocess.Popen[bytes] | None, existing_pid: int | None
) -> None:
    """Block until the launcher exits, regardless of who owns it."""
    if proc is not None:
        proc.wait()
    elif existing_pid is not None:
        _wait_for_pid_exit(existing_pid)


@dataclass
class _LiveTrial:
    """Continuous-flow tracking for one in-flight trial.

    The wave driver could rely on aligned ``(artifact, optuna_trial)`` lists;
    continuous-flow needs random-access lookups by ``run_dir`` because slots
    free in arbitrary order, so we group every per-trial bookkeeping field
    here. ``last_step`` is the highest metrics.jsonl step we've already
    forwarded to Optuna; ``attempts`` is the 1-based attempt count for
    auto-retry — incremented when a failed trial is re-materialized.
    """

    optuna_trial: Any  # optuna.Trial; typed loosely to avoid an unconditional import
    artifact: TrialArtifacts
    last_step: int | None = None
    attempts: int = 1


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


def _ask_and_materialize(
    optuna: Any,
    study: Any,
    config: SweepConfig,
    scheduler: MultiRunLoRASchedulerConfig,
    trial_index: int,
) -> tuple[Any, TrialArtifacts] | None:
    """Ask Optuna for a fresh trial and materialize its run dir.

    Returns ``None`` when materialization fails — the caller has already
    counted the failure and told Optuna ``FAIL``. Pulled out so the
    continuous-flow loop's "ask one more" branch reads as a single call
    instead of a dozen lines of Optuna boilerplate.
    """
    optuna_trial = study.ask()
    params = _suggest_parameters(optuna_trial, config.parameters)
    sweep_trial = _make_trial(trial_index, params)
    try:
        artifact = materialize_multi_run_trial(config, sweep_trial, scheduler)
    except Exception as exc:
        study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
        print(f"Optuna trial {sweep_trial.id} failed materialization: {exc}")
        return None
    return optuna_trial, artifact


def _exit_code_path(run_dir: Path) -> Path:
    return run_dir / "control" / EXIT_CODE_FILENAME


def _write_done_marker(shared_dir: Path) -> None:
    """Tell ``rl-multi-run --watch-slots`` it can exit once the last orchestrator finishes."""
    control_dir = shared_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    (control_dir / DONE_MARKER_NAME).write_text("done\n")


def _process_pruning_signal(
    optuna: Any,
    live_trial: _LiveTrial,
    metric: str,
) -> None:
    """Forward the latest ``(step, value)`` to Optuna and prune if asked.

    Idempotent: re-reading the same step in a later poll tick is a no-op.
    Already-pruned trials short-circuit so we don't double-write evicted.txt
    while we wait for the orchestrator's process to actually exit.
    """
    try:
        status = _read_status(live_trial.artifact)
    except FileNotFoundError:
        return
    if status.get("state") == "pruned":
        return

    sample = read_intermediate_metric(live_trial.artifact.run_dir, metric)
    if sample is None:
        return
    step, value = sample
    if live_trial.last_step is not None and step <= live_trial.last_step:
        return

    live_trial.optuna_trial.report(value, step)
    live_trial.last_step = step

    if live_trial.optuna_trial.should_prune():
        prune_run(
            live_trial.artifact.run_dir,
            reason=f"optuna prune at step {step}",
            step=step,
            value=value,
        )


def _reconcile_trial_state(
    live_trial: _LiveTrial,
    metric: str,
) -> tuple[str, float | None]:
    """Reconcile one finished trial without telling Optuna.

    Returns ``(state, objective)`` where ``state`` is ``"completed"``,
    ``"pruned"``, or ``"failed"`` and ``objective`` is the recorded value
    (``None`` for non-completed states or for completed runs whose metric
    never showed up). The caller decides whether the outcome is final
    (call ``study.tell``) or whether to retry (re-materialize and skip
    the tell).
    """
    state = reconcile_multi_run_artifact(
        live_trial.artifact, aggregate_returncode=0, finished_at=utc_now()
    )

    objective: float | None = None
    if state == "completed":
        objective = read_final_summary(live_trial.artifact.run_dir, metric)
        record_trial_objective(live_trial.artifact.status_path, objective)
        if objective is None:
            # Clean exit but the metric never showed up — the orchestrator
            # produced no usable result. Treat as a failure so the retry /
            # FAIL branch gets a chance instead of telling Optuna a number
            # the sampler can't actually use.
            state = "failed"
    elif state == "failed":
        record_trial_objective(live_trial.artifact.status_path, None)

    return state, objective


def _tell_final_outcome(
    optuna: Any,
    study: Any,
    live_trial: _LiveTrial,
    state: str,
    objective: float | None,
) -> None:
    """Forward the final outcome to Optuna once the trial is past retry."""
    if state == "completed" and objective is not None:
        study.tell(live_trial.optuna_trial, objective)
    elif state == "pruned":
        study.tell(live_trial.optuna_trial, state=optuna.trial.TrialState.PRUNED)
    else:  # failed (or completed with no objective — collapsed to failed in _reconcile_trial_state)
        study.tell(live_trial.optuna_trial, state=optuna.trial.TrialState.FAIL)


def run_multi_run_optuna_sweep(
    config: SweepConfig,
    *,
    write_manifest_with_variants: Any,
    build_variant: Any,
) -> tuple[int, TrialOutcomeTracker | None, list[TrialArtifacts]]:
    """Drive an Optuna study against ``rl-multi-run`` in continuous-flow mode.

    Spawns ``rl-multi-run --watch-slots`` exactly once per sweep and keeps
    ``max_concurrent_runs`` trials live at all times: as each slot frees the
    controller asks Optuna for a replacement, materializes its ``run_*`` dir,
    and the launcher's slot-watch loop picks it up. When every trial has been
    submitted to the study and the live set is empty, the controller writes
    ``<shared_dir>/control/done`` and the launcher tears down.

    This replaces 7b's wave loop. The wave-mode "slot freed mid-wave sits
    idle" tradeoff is gone, so heavily-pruned sweeps no longer waste GPU
    cycles. Cross-trial pruning also gets fresher signal because pruners see
    each ``study.tell`` as soon as the trial finishes.
    """
    optuna = _import_optuna()
    strategy = config.strategy
    scheduler = config.scheduler
    assert isinstance(strategy, OptunaStrategyConfig)
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)
    assert config.objective is not None  # validated upstream

    study = _create_study(optuna, config)
    metric = config.objective.metric
    target_concurrency = scheduler.max_concurrent_runs
    total = strategy.num_trials
    poll_interval = strategy.poll_interval_seconds

    tracker = TrialOutcomeTracker(config.objective, config.early_stopping)
    shared_dir = multi_run_shared_dir(config)
    shared_dir.mkdir(parents=True, exist_ok=True)

    previous_variants = _load_previous_variants(config) if config.resume else []
    if config.resume:
        reconciled = _reconcile_running_trials(optuna, study, previous_variants)
        if reconciled:
            print(f"Reconciled {reconciled} RUNNING Optuna trial(s) from interrupted resume.")
        _seed_tracker_from_previous(tracker, previous_variants)

    all_artifacts: list[TrialArtifacts] = []
    failures = 0
    # On resume, every trial Optuna has seen counts toward num_trials so we
    # only ask for the remainder. len(study.trials) covers completed +
    # pruned + failed because _reconcile_running_trials already told the
    # study about every prior RUNNING slot.
    submitted = len(study.trials) if config.resume else 0

    # 1. Materialize the initial cohort. The launcher needs at least one
    # run dir at startup (its --runs-dir argument). Any later trials get
    # picked up by the watch-slots loop.
    initial_count = min(target_concurrency, total - submitted)
    live: dict[Path, _LiveTrial] = {}
    for _ in range(initial_count):
        if tracker.halted:
            break
        result = _ask_and_materialize(optuna, study, config, scheduler, submitted)
        submitted += 1
        if result is None:
            failures += 1
            if not config.continue_on_failure:
                raise SystemExit(1)
            continue
        optuna_trial, artifact = result
        all_artifacts.append(artifact)
        live[artifact.run_dir] = _LiveTrial(optuna_trial=optuna_trial, artifact=artifact)

    if not live:
        # Every initial materialization failed (or we were already past
        # `total` on resume); nothing to launch.
        write_manifest_with_variants(
            config, previous_variants + [build_variant(a) for a in all_artifacts]
        )
        return failures, tracker, all_artifacts

    write_manifest_with_variants(
        config, previous_variants + [build_variant(a) for a in all_artifacts]
    )

    started = utc_now()
    for live_trial in live.values():
        _write_status(
            live_trial.artifact, state="running", started_at=started, attempts=1, gpu_group=None
        )

    # 2. Spawn rl-multi-run --watch-slots, OR attach to a live one (Phase 7e).
    initial_artifacts = [lt.artifact for lt in live.values()]
    existing_pid = _detect_running_launcher(shared_dir) if config.resume else None
    proc: subprocess.Popen[bytes] | None
    if existing_pid is not None:
        # Live-attach: the prior launcher is still running and accepting
        # new run dirs through its watch-slots loop. Skip the trainer +
        # inference re-spawn; just drop the new run dirs into shared_dir
        # and the launcher picks them up.
        proc = None
        print(f"Live-attach: resuming against running launcher (pid={existing_pid}).")
    else:
        command = build_multi_run_command(initial_artifacts, scheduler.shared, shared_dir)
        command.append("--watch-slots")
        proc = subprocess.Popen(command)

    try:
        # 3. Continuous-flow loop.
        while live:
            if not _launcher_is_alive(proc, existing_pid):
                # Launcher died unexpectedly. Settle whatever's still alive
                # so Optuna's view doesn't have RUNNING trials hanging. No
                # retries on this path: the trainer is gone, retrying would
                # need us to re-spawn the entire stack.
                for live_trial in list(live.values()):
                    state, objective = _reconcile_trial_state(live_trial, metric)
                    _tell_final_outcome(optuna, study, live_trial, state, objective)
                    if state == "failed":
                        failures += 1
                    tracker.observe(
                        TrialOutcome(
                            trial_id=live_trial.artifact.trial.id,
                            label=live_trial.artifact.trial.label,
                            objective=objective,
                        )
                    )
                live.clear()
                break

            for run_dir in list(live.keys()):
                live_trial = live[run_dir]
                _process_pruning_signal(optuna, live_trial, metric)

                # An exit_code file appears the moment the launcher reaps
                # this orchestrator. That's the signal to settle the trial
                # and free up its slot.
                if not _exit_code_path(run_dir).exists():
                    continue

                state, objective = _reconcile_trial_state(live_trial, metric)

                # Auto-retry: a transient failure with budget left re-uses
                # the same Optuna trial (same params) but materializes a
                # fresh run_<id>-r<N> dir the launcher's slot-watch loop
                # picks up. Optuna sees this as one logical trial; only
                # the final outcome is told.
                if (
                    state == "failed"
                    and live_trial.attempts < config.retry_budget + 1
                ):
                    new_attempt = live_trial.attempts + 1
                    try:
                        retry_artifact = materialize_multi_run_trial(
                            config, live_trial.artifact.trial, scheduler, attempt=new_attempt - 1
                        )
                    except Exception as exc:
                        # Materialization shouldn't fail at retry time
                        # (params already validated), but if it does fall
                        # through to the final-FAIL branch.
                        print(
                            f"Optuna trial {live_trial.artifact.trial.id} retry "
                            f"{new_attempt} failed materialization: {exc}"
                        )
                    else:
                        all_artifacts.append(retry_artifact)
                        _write_status(
                            retry_artifact,
                            state="running",
                            started_at=utc_now(),
                            attempts=new_attempt,
                            gpu_group=None,
                        )
                        live.pop(run_dir)
                        live[retry_artifact.run_dir] = _LiveTrial(
                            optuna_trial=live_trial.optuna_trial,
                            artifact=retry_artifact,
                            attempts=new_attempt,
                        )
                        write_manifest_with_variants(
                            config,
                            previous_variants + [build_variant(a) for a in all_artifacts],
                        )
                        continue

                _tell_final_outcome(optuna, study, live_trial, state, objective)
                if state == "failed":
                    failures += 1
                    if not config.continue_on_failure:
                        # Leave remaining live trials for the launcher to
                        # tear down; we'll exit after writing the done marker.
                        live.pop(run_dir)
                        raise SystemExit(1)

                tracker.observe(
                    TrialOutcome(
                        trial_id=live_trial.artifact.trial.id,
                        label=live_trial.artifact.trial.label,
                        objective=objective,
                    )
                )
                live.pop(run_dir)

                # Replenish the freed slot if there are trials left to ask
                # for and early stopping hasn't fired.
                if submitted < total and not tracker.halted:
                    result = _ask_and_materialize(optuna, study, config, scheduler, submitted)
                    submitted += 1
                    if result is None:
                        failures += 1
                        if not config.continue_on_failure:
                            raise SystemExit(1)
                        continue
                    new_optuna_trial, new_artifact = result
                    all_artifacts.append(new_artifact)
                    _write_status(
                        new_artifact,
                        state="running",
                        started_at=utc_now(),
                        attempts=1,
                        gpu_group=None,
                    )
                    live[new_artifact.run_dir] = _LiveTrial(
                        optuna_trial=new_optuna_trial, artifact=new_artifact
                    )
                    write_manifest_with_variants(
                        config,
                        previous_variants + [build_variant(a) for a in all_artifacts],
                    )

            time.sleep(poll_interval)

        # 4. All trials told. Signal the launcher and wait for it to drain.
        _write_done_marker(shared_dir)
    finally:
        _wait_for_launcher_exit(proc, existing_pid)

    write_manifest_with_variants(
        config, previous_variants + [build_variant(a) for a in all_artifacts]
    )
    return failures, tracker, all_artifacts


def _reconcile_static_trial_state(
    live_trial: _LiveTrial,
    metric: str,
) -> tuple[str, float | None]:
    """Static-mode counterpart to ``_reconcile_trial_state``.

    Identical reconcile logic minus the metric-was-None → "failed" collapse:
    static sweeps don't tell anyone about the trial in a way the search
    backend cares about, so a missing metric just becomes ``objective=None``
    on a completed trial. Auto-retry still triggers on actual subprocess
    failures (state="failed" from a non-zero exit_code).
    """
    state = reconcile_multi_run_artifact(
        live_trial.artifact, aggregate_returncode=0, finished_at=utc_now()
    )

    objective: float | None = None
    if state == "completed":
        objective = read_final_summary(live_trial.artifact.run_dir, metric)
        record_trial_objective(live_trial.artifact.status_path, objective)
    elif state == "failed":
        record_trial_objective(live_trial.artifact.status_path, None)

    return state, objective


def run_multi_run_static_continuous_sweep(
    config: SweepConfig,
    artifacts: list[TrialArtifacts],
    *,
    write_manifest_with_variants: Any,
    build_variant: Any,
) -> tuple[int, TrialOutcomeTracker | None]:
    """Drive a static (grid/random) shared-trainer LoRA sweep continuously.

    Mirrors ``run_multi_run_optuna_sweep``'s slot-replacement loop without the
    Optuna asks/tells/pruning. Pre-materialized artifacts feed a queue; the
    initial cohort sized ``max_concurrent_runs`` launches via
    ``rl-multi-run --watch-slots``. As each slot frees the controller pulls
    the next pending artifact, writes its status to ``running``, and the
    launcher's slot-watch loop spawns the orchestrator.

    Lifts the 7a wave-or-bust limit: large grids stream through the same
    launcher invocation instead of needing ``num_trials <= max_concurrent_runs``.
    """
    scheduler = config.scheduler
    assert isinstance(scheduler, MultiRunLoRASchedulerConfig)

    metric = config.objective.metric if config.objective else None
    target_concurrency = scheduler.max_concurrent_runs
    poll_interval = 2.0  # static has no strategy.poll_interval_seconds; mirror the launcher's tick

    tracker = TrialOutcomeTracker(config.objective, config.early_stopping) if config.objective else None
    shared_dir = multi_run_shared_dir(config)
    shared_dir.mkdir(parents=True, exist_ok=True)

    # Already-terminal trials (resume) keep their preserved artifacts and
    # only contribute to the tracker; only "pending" trials get launched.
    pending_queue: list[TrialArtifacts] = [
        a for a in artifacts if json.loads(a.status_path.read_text()).get("state") == "pending"
    ]

    # Seed tracker from preserved completed trials so the manifest summary
    # reflects history.
    if tracker is not None:
        for artifact in artifacts:
            status = json.loads(artifact.status_path.read_text())
            if status.get("state") != "completed":
                continue
            tracker.observe(
                TrialOutcome(
                    trial_id=artifact.trial.id,
                    label=artifact.trial.label,
                    objective=status.get("objective"),
                )
            )

    failures = 0

    if not pending_queue:
        if config.resume:
            print("Resume: every trial already terminal, no new work to launch.")
        return failures, tracker

    # 1. Initial cohort.
    initial = pending_queue[:target_concurrency]
    remaining: list[TrialArtifacts] = pending_queue[target_concurrency:]
    live: dict[Path, _LiveTrial] = {}
    started = utc_now()
    for artifact in initial:
        _write_status(artifact, state="running", started_at=started, attempts=1, gpu_group=None)
        # optuna_trial=None — static has no search backend to tell.
        live[artifact.run_dir] = _LiveTrial(optuna_trial=None, artifact=artifact)

    # 2. Spawn rl-multi-run --watch-slots, OR attach to a live one (Phase 7e).
    existing_pid = _detect_running_launcher(shared_dir) if config.resume else None
    proc: subprocess.Popen[bytes] | None
    if existing_pid is not None:
        proc = None
        print(f"Live-attach: resuming against running launcher (pid={existing_pid}).")
    else:
        command = build_multi_run_command(initial, scheduler.shared, shared_dir)
        command.append("--watch-slots")
        proc = subprocess.Popen(command)

    try:
        while live:
            if not _launcher_is_alive(proc, existing_pid):
                # Launcher died; settle anything still alive without retry.
                for live_trial in list(live.values()):
                    state, objective = _reconcile_static_trial_state(live_trial, metric or "")
                    if state == "failed":
                        failures += 1
                    if tracker is not None:
                        tracker.observe(
                            TrialOutcome(
                                trial_id=live_trial.artifact.trial.id,
                                label=live_trial.artifact.trial.label,
                                objective=objective,
                            )
                        )
                live.clear()
                break

            for run_dir in list(live.keys()):
                live_trial = live[run_dir]
                if not _exit_code_path(run_dir).exists():
                    continue

                state, objective = _reconcile_static_trial_state(live_trial, metric or "")

                # Auto-retry on transient failure.
                if state == "failed" and live_trial.attempts < config.retry_budget + 1:
                    new_attempt = live_trial.attempts + 1
                    try:
                        retry_artifact = materialize_multi_run_trial(
                            config, live_trial.artifact.trial, scheduler, attempt=new_attempt - 1
                        )
                    except Exception as exc:
                        print(
                            f"Static trial {live_trial.artifact.trial.id} retry "
                            f"{new_attempt} failed materialization: {exc}"
                        )
                    else:
                        artifacts.append(retry_artifact)
                        _write_status(
                            retry_artifact,
                            state="running",
                            started_at=utc_now(),
                            attempts=new_attempt,
                            gpu_group=None,
                        )
                        live.pop(run_dir)
                        live[retry_artifact.run_dir] = _LiveTrial(
                            optuna_trial=None,
                            artifact=retry_artifact,
                            attempts=new_attempt,
                        )
                        write_manifest_with_variants(
                            config, [build_variant(a) for a in artifacts]
                        )
                        continue

                if state == "failed":
                    failures += 1
                    if not config.continue_on_failure:
                        live.pop(run_dir)
                        raise SystemExit(1)

                if tracker is not None:
                    tracker.observe(
                        TrialOutcome(
                            trial_id=live_trial.artifact.trial.id,
                            label=live_trial.artifact.trial.label,
                            objective=objective,
                        )
                    )
                live.pop(run_dir)

                # Pull the next pending trial off the queue and let the
                # launcher's watch-slots loop spawn its orchestrator.
                if remaining and (tracker is None or not tracker.halted):
                    next_artifact = remaining.pop(0)
                    _write_status(
                        next_artifact,
                        state="running",
                        started_at=utc_now(),
                        attempts=1,
                        gpu_group=None,
                    )
                    live[next_artifact.run_dir] = _LiveTrial(
                        optuna_trial=None, artifact=next_artifact
                    )

            time.sleep(poll_interval)

        _write_done_marker(shared_dir)
    finally:
        _wait_for_launcher_exit(proc, existing_pid)

    write_manifest_with_variants(config, [build_variant(a) for a in artifacts])
    return failures, tracker


# Re-exported for the controller to update the manifest summary.
def tracker_summary(tracker: TrialOutcomeTracker) -> dict[str, Any]:
    return asdict(tracker.summary())
