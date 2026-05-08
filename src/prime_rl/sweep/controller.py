import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import tomli_w

from prime_rl.configs.sweep import (
    GridStrategyConfig,
    LocalSweepSchedulerConfig,
    MultiRunLoRASchedulerConfig,
    OptunaStrategyConfig,
    RandomStrategyConfig,
    SlurmSweepSchedulerConfig,
    SweepConfig,
)
from prime_rl.sweep.early_stopping import TrialOutcome, TrialOutcomeTracker
from prime_rl.sweep.materialize import (
    Trial,
    TrialArtifacts,
    materialize_multi_run_trial,
    materialize_trial,
    multi_run_shared_dir,
    record_trial_objective,
)
from prime_rl.sweep.metrics import read_final_summary
from prime_rl.sweep.multi_run import (
    run_multi_run_optuna_sweep,
    run_multi_run_static_continuous_sweep,
)
from prime_rl.sweep.optuna_loop import run_optuna_sweep
from prime_rl.sweep.reproducibility import git_metadata
from prime_rl.sweep.schedulers import (
    run_trials_locally,
    submit_trials_to_slurm,
    submit_trials_to_slurm_array,
)
from prime_rl.sweep.search import expand_grid, sample_random


def _write_toml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def build_variant(
    artifact: TrialArtifacts,
    *,
    array_task_index: int | None = None,
) -> dict[str, Any]:
    return {
        "id": artifact.trial.id,
        "label": artifact.trial.label,
        "output_dir": artifact.run_dir.as_posix(),
        "overrides": artifact.trial.parameters,
        "command": artifact.command,
        "status_path": artifact.status_path.as_posix(),
        "resolved_checksum": artifact.resolved_checksum,
        "base_checksums": artifact.base_checksums,
        "array_task_index": array_task_index,
    }


def write_manifest_with_variants(config: SweepConfig, variants: list[dict[str, Any]]) -> None:
    manifest = {
        "name": config.name,
        "entrypoint": config.entrypoint,
        "strategy": config.strategy.model_dump(mode="json"),
        "scheduler": config.scheduler.model_dump(mode="json"),
        "objective": config.objective.model_dump(mode="json") if config.objective else None,
        "early_stopping": config.early_stopping.model_dump(mode="json") if config.early_stopping else None,
        "git": git_metadata(),
        "variants": variants,
    }
    (config.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _write_manifest(config: SweepConfig, artifacts: list[TrialArtifacts]) -> None:
    write_manifest_with_variants(config, [build_variant(a) for a in artifacts])


def _update_manifest_summary(config: SweepConfig, summary: dict[str, Any] | None) -> None:
    manifest_path = config.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["summary"] = summary
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _expand_trials(config: SweepConfig) -> list[Trial]:
    if isinstance(config.strategy, GridStrategyConfig):
        return expand_grid(config.parameters)
    if isinstance(config.strategy, RandomStrategyConfig):
        return sample_random(
            config.parameters,
            num_trials=config.strategy.num_trials,
            seed=config.strategy.seed,
        )
    raise ValueError(f"Unsupported sweep strategy: {config.strategy!r}")


def _previous_checksums(config: SweepConfig) -> dict[str, dict[str, Any]]:
    """Map trial_id -> {resolved_checksum, base_checksums} from the prior manifest."""
    manifest_path = config.output_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text())
    return {
        variant["id"]: {
            "resolved_checksum": variant.get("resolved_checksum"),
            "base_checksums": variant.get("base_checksums") or {},
        }
        for variant in manifest.get("variants", [])
    }


def _materialize_study(config: SweepConfig) -> list[TrialArtifacts]:
    if config.output_dir.exists() and config.clean_output_dir:
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    expected = _previous_checksums(config) if config.resume else {}

    _write_toml(config.output_dir / "study.toml", config.model_dump(exclude_none=True, mode="json"))

    trials = _expand_trials(config)
    artifacts = [
        materialize_trial(config, trial, resume=config.resume, expected_checksums=expected.get(trial.id))
        for trial in trials
    ]
    _write_manifest(config, artifacts)
    return artifacts


def _materialize_multi_run_study(config: SweepConfig) -> list[TrialArtifacts]:
    """Materialize all trials as ``run_*`` subdirs under a shared trainer dir.

    Multi-run sweeps invoke ``rl-multi-run`` exactly once with all trials
    laid out up front. ``--resume`` (Phase 7c) preserves the per-trial
    artifacts of completed/pruned/failed trials and only re-prepares the
    pending ones; live-attach against a still-running trainer is still 7d+.
    """
    if config.output_dir.exists() and config.clean_output_dir:
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    multi_run_shared_dir(config).mkdir(parents=True, exist_ok=True)

    expected = _previous_checksums(config) if config.resume else {}

    _write_toml(config.output_dir / "study.toml", config.model_dump(exclude_none=True, mode="json"))

    trials = _expand_trials(config)
    assert isinstance(config.scheduler, MultiRunLoRASchedulerConfig)
    artifacts = [
        materialize_multi_run_trial(
            config,
            trial,
            config.scheduler,
            resume=config.resume,
            expected_checksums=expected.get(trial.id),
        )
        for trial in trials
    ]
    _write_manifest(config, artifacts)
    return artifacts


def _build_trial_callback(config: SweepConfig, tracker: TrialOutcomeTracker | None):
    if config.objective is None or tracker is None:
        return None

    metric = config.objective.metric

    def on_trial_complete(artifact: TrialArtifacts, returncode: int) -> bool:
        objective = read_final_summary(artifact.run_dir, metric) if returncode == 0 else None
        record_trial_objective(artifact.status_path, objective)
        outcome = TrialOutcome(trial_id=artifact.trial.id, label=artifact.trial.label, objective=objective)
        return tracker.observe(outcome)

    return on_trial_complete


def _slurm_array_indices_to_submit(artifacts: list[TrialArtifacts], resume: bool) -> list[int]:
    """Pick array indices that need submission.

    Phase 8 resume: completed/submitted trials keep their prior array job
    on the cluster (or already finished); only ``pending`` and ``failed``
    indices come back into a new submission. Without resume, we submit
    every index.
    """
    if not resume:
        return list(range(len(artifacts)))

    indices: list[int] = []
    for index, artifact in enumerate(artifacts):
        try:
            status = json.loads(artifact.status_path.read_text())
        except FileNotFoundError:
            indices.append(index)
            continue
        if status.get("state") in {"completed", "submitted"}:
            continue
        indices.append(index)
    return indices


def _record_array_job_id(config: SweepConfig, array_job_id: str | None) -> None:
    """Stamp the SLURM array job ID at the top of the manifest.

    Lets ``sacct``/``squeue`` correlate post-hoc and, on resume, lets the
    next controller run know which job already covers prior submissions.
    """
    manifest_path = config.output_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest["array_job_id"] = array_job_id
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _seed_tracker_from_resume(tracker: TrialOutcomeTracker, artifacts: list[TrialArtifacts]) -> None:
    """Replay each completed trial's recorded objective into the tracker.

    Without this the resumed scheduler skips already-completed trials, so the
    tracker never sees them — the manifest summary would forget earlier work
    and patience/threshold decisions would not account for completed trials.
    """
    for artifact in artifacts:
        status = json.loads(artifact.status_path.read_text())
        if status.get("state") != "completed":
            continue
        outcome = TrialOutcome(
            trial_id=artifact.trial.id,
            label=artifact.trial.label,
            objective=status.get("objective"),
        )
        tracker.observe(outcome)


def _run_optuna(config: SweepConfig) -> None:
    if config.output_dir.exists() and config.clean_output_dir:
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_toml(config.output_dir / "study.toml", config.model_dump(exclude_none=True, mode="json"))

    if config.dry_run:
        print(
            f"Dry run for Optuna strategy is a no-op: trials are proposed sequentially based on "
            f"prior objectives, so they cannot be materialized up front."
        )
        return

    failures, tracker, artifacts = run_optuna_sweep(
        config,
        write_manifest_with_variants=write_manifest_with_variants,
        build_variant=build_variant,
    )

    if tracker is not None:
        summary = asdict(tracker.summary())
        _update_manifest_summary(config, summary)
        if summary["best_trial_id"] is not None:
            label = tracker.best_label or summary["best_trial_id"]
            print(f"Best trial: {label} ({summary['best_value']})")
        if summary["halted_by_early_stopping"]:
            print(f"Sweep halted by early stopping ({summary['halt_reason']}).")

    if failures > 0:
        print(f"Sweep finished with {failures} failed trial(s) out of {len(artifacts)}.")
        raise SystemExit(1)


def _run_multi_run_static(config: SweepConfig) -> None:
    """Drive a static (grid/random) shared-trainer LoRA sweep through ``rl-multi-run``.

    Phase 7e: continuous-flow replaces the wave path. Trials are
    pre-materialized; the driver maintains ``max_concurrent_runs`` live
    trials and pulls the next pending artifact off the queue as each slot
    frees. On ``--resume``, trials whose prior status was
    completed/pruned/failed are kept verbatim and only contribute to the
    tracker; only ``state == "pending"`` artifacts go into the launcher.
    """
    assert isinstance(config.scheduler, MultiRunLoRASchedulerConfig)
    artifacts = _materialize_multi_run_study(config)

    if config.dry_run:
        print(
            f"Dry run complete. Materialized {len(artifacts)} run dir(s) under "
            f"{multi_run_shared_dir(config)}."
        )
        for artifact in artifacts:
            print(f"  {artifact.run_dir}")
        return

    failures, tracker = run_multi_run_static_continuous_sweep(
        config,
        artifacts,
        write_manifest_with_variants=write_manifest_with_variants,
        build_variant=build_variant,
    )

    if tracker is not None:
        summary = asdict(tracker.summary())
        _update_manifest_summary(config, summary)
        if summary["best_trial_id"] is not None:
            label = tracker.best_label or summary["best_trial_id"]
            print(f"Best trial: {label} ({summary['best_value']})")
        if summary["halted_by_early_stopping"]:
            print(f"Sweep halted by early stopping ({summary['halt_reason']}).")

    if failures > 0:
        print(f"Sweep finished with {failures} failed trial(s) out of {len(artifacts)}.")
        raise SystemExit(1)


def _run_multi_run_optuna(config: SweepConfig) -> None:
    """Drive an Optuna study against ``rl-multi-run`` in waves of size ``max_concurrent_runs``."""
    assert isinstance(config.scheduler, MultiRunLoRASchedulerConfig)
    assert isinstance(config.strategy, OptunaStrategyConfig)
    if config.output_dir.exists() and config.clean_output_dir:
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    multi_run_shared_dir(config).mkdir(parents=True, exist_ok=True)
    _write_toml(config.output_dir / "study.toml", config.model_dump(exclude_none=True, mode="json"))

    if config.dry_run:
        print(
            "Dry run for Optuna + multi_run_lora is a no-op: trials are proposed wave by wave "
            "based on prior objectives, so they cannot be materialized up front."
        )
        return

    failures, tracker, artifacts = run_multi_run_optuna_sweep(
        config,
        write_manifest_with_variants=write_manifest_with_variants,
        build_variant=build_variant,
    )

    if tracker is not None:
        summary = asdict(tracker.summary())
        _update_manifest_summary(config, summary)
        if summary["best_trial_id"] is not None:
            label = tracker.best_label or summary["best_trial_id"]
            print(f"Best trial: {label} ({summary['best_value']})")
        if summary["halted_by_early_stopping"]:
            print(f"Sweep halted by early stopping ({summary['halt_reason']}).")

    if failures > 0:
        print(f"Sweep finished with {failures} failed trial(s) out of {len(artifacts)}.")
        raise SystemExit(1)


def _run_multi_run(config: SweepConfig) -> None:
    """Dispatch a shared-trainer LoRA sweep based on the search strategy."""
    assert isinstance(config.scheduler, MultiRunLoRASchedulerConfig)
    if isinstance(config.strategy, OptunaStrategyConfig):
        _run_multi_run_optuna(config)
        return
    _run_multi_run_static(config)


def run_sweep(config: SweepConfig) -> None:
    # multi_run_lora dispatches first because the Optuna + multi_run_lora
    # combination has its own wave driver — falling through to _run_optuna
    # would launch single-trial mode against the wrong scheduler.
    if isinstance(config.scheduler, MultiRunLoRASchedulerConfig):
        _run_multi_run(config)
        return

    if isinstance(config.strategy, OptunaStrategyConfig):
        _run_optuna(config)
        return

    artifacts = _materialize_study(config)

    if config.dry_run:
        print(f"Dry run complete. Materialized {len(artifacts)} trial(s) under {config.output_dir}.")
        for artifact in artifacts:
            print(" ".join(artifact.command))
        return

    track_objectives = config.objective is not None and isinstance(config.scheduler, LocalSweepSchedulerConfig)
    if config.objective is not None and not track_objectives:
        print(
            "Note: objective tracking is only computed for the local scheduler; "
            "SLURM trials run asynchronously after submission and produce their own status.json."
        )
    tracker = TrialOutcomeTracker(config.objective, config.early_stopping) if track_objectives else None
    on_trial_complete = _build_trial_callback(config, tracker)

    if tracker is not None and config.resume:
        _seed_tracker_from_resume(tracker, artifacts)

    failures = 0
    if tracker is not None and tracker.halted:
        print("Skipping new trials: early stopping already triggered by completed trials in the study.")
    elif isinstance(config.scheduler, LocalSweepSchedulerConfig):
        gpu_groups = (
            config.scheduler.gpu_assignment.visible_devices if config.scheduler.gpu_assignment is not None else None
        )
        failures = run_trials_locally(
            artifacts,
            max_parallel=config.scheduler.max_parallel,
            gpu_groups=gpu_groups,
            continue_on_failure=config.continue_on_failure,
            retry_budget=config.retry_budget,
            on_trial_complete=on_trial_complete,
        )
    elif isinstance(config.scheduler, SlurmSweepSchedulerConfig):
        if config.scheduler.use_array:
            # Phase 8: one sbatch --array=... covers the whole study. The
            # array_task_index lives in the manifest so sweep-array-task can
            # find each variant by index, so re-write the manifest with
            # indices stamped in *before* submitting.
            variants_with_index = [
                build_variant(artifact, array_task_index=index)
                for index, artifact in enumerate(artifacts)
            ]
            write_manifest_with_variants(config, variants_with_index)

            indices_to_submit = _slurm_array_indices_to_submit(artifacts, config.resume)
            if not indices_to_submit:
                if config.resume:
                    print("Resume: every array task already terminal, no new submission.")
                failures = 0
            else:
                array_job_id, submitted = submit_trials_to_slurm_array(
                    artifacts,
                    study_dir=config.output_dir,
                    array_indices=indices_to_submit,
                )
                _record_array_job_id(config, array_job_id)
                failures = 0 if array_job_id is not None else len(submitted) or len(indices_to_submit)
                if failures > 0 and not config.continue_on_failure:
                    raise SystemExit(1)
        else:
            failures = submit_trials_to_slurm(
                artifacts,
                continue_on_failure=config.continue_on_failure,
                retry_budget=config.retry_budget,
            )
    else:
        raise ValueError(f"Unsupported sweep scheduler: {config.scheduler}")

    if tracker is not None:
        summary = asdict(tracker.summary())
        _update_manifest_summary(config, summary)
        if summary["best_trial_id"] is not None:
            label = tracker.best_label or summary["best_trial_id"]
            print(f"Best trial: {label} ({summary['best_value']})")
        if summary["halted_by_early_stopping"]:
            print(f"Sweep halted by early stopping ({summary['halt_reason']}).")

    if failures > 0:
        print(f"Sweep finished with {failures} failed trial(s) out of {len(artifacts)}.")
        raise SystemExit(1)
