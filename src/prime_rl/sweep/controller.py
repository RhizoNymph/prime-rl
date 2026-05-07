import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import tomli_w

from prime_rl.configs.sweep import (
    GridStrategyConfig,
    LocalSweepSchedulerConfig,
    RandomStrategyConfig,
    SlurmSweepSchedulerConfig,
    SweepConfig,
)
from prime_rl.sweep.early_stopping import TrialOutcome, TrialOutcomeTracker
from prime_rl.sweep.materialize import Trial, TrialArtifacts, materialize_trial, write_json
from prime_rl.sweep.metrics import read_final_summary
from prime_rl.sweep.reproducibility import git_metadata
from prime_rl.sweep.schedulers import run_trials_locally, submit_trials_to_slurm
from prime_rl.sweep.search import expand_grid, sample_random


def _write_toml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def _write_manifest(config: SweepConfig, artifacts: list[TrialArtifacts]) -> None:
    variants = []
    for artifact in artifacts:
        variants.append(
            {
                "id": artifact.trial.id,
                "label": artifact.trial.label,
                "output_dir": artifact.run_dir.as_posix(),
                "overrides": artifact.trial.parameters,
                "command": artifact.command,
                "status_path": artifact.status_path.as_posix(),
                "resolved_checksum": artifact.resolved_checksum,
                "base_checksums": artifact.base_checksums,
            }
        )

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


def _record_objective(artifact: TrialArtifacts, value: float | None) -> None:
    status = json.loads(artifact.status_path.read_text())
    status["objective"] = value
    write_json(artifact.status_path, status)


def _build_trial_callback(config: SweepConfig, tracker: TrialOutcomeTracker | None):
    if config.objective is None or tracker is None:
        return None

    metric = config.objective.metric

    def on_trial_complete(artifact: TrialArtifacts, returncode: int) -> bool:
        objective = read_final_summary(artifact.run_dir, metric) if returncode == 0 else None
        _record_objective(artifact, objective)
        outcome = TrialOutcome(trial_id=artifact.trial.id, label=artifact.trial.label, objective=objective)
        return tracker.observe(outcome)

    return on_trial_complete


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


def run_sweep(config: SweepConfig) -> None:
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
