import json
import shutil
from pathlib import Path
from typing import Any

import tomli_w

from prime_rl.configs.sweep import LocalSweepSchedulerConfig, SlurmSweepSchedulerConfig, SweepConfig
from prime_rl.sweep.materialize import TrialArtifacts, materialize_trial
from prime_rl.sweep.reproducibility import git_metadata
from prime_rl.sweep.schedulers import run_trials_locally, submit_trials_to_slurm
from prime_rl.sweep.search import expand_grid


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
        "strategy": config.strategy,
        "scheduler": config.scheduler.model_dump(mode="json"),
        "git": git_metadata(),
        "variants": variants,
    }
    (config.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _materialize_study(config: SweepConfig) -> list[TrialArtifacts]:
    if config.output_dir.exists() and config.clean_output_dir:
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    _write_toml(config.output_dir / "study.toml", config.model_dump(exclude_none=True, mode="json"))

    trials = expand_grid(config.parameters)
    artifacts = [materialize_trial(config, trial) for trial in trials]
    _write_manifest(config, artifacts)
    return artifacts


def run_sweep(config: SweepConfig) -> None:
    artifacts = _materialize_study(config)

    if config.dry_run:
        print(f"Dry run complete. Materialized {len(artifacts)} trial(s) under {config.output_dir}.")
        for artifact in artifacts:
            print(" ".join(artifact.command))
        return

    if isinstance(config.scheduler, LocalSweepSchedulerConfig):
        run_trials_locally(artifacts, max_parallel=config.scheduler.max_parallel)
    elif isinstance(config.scheduler, SlurmSweepSchedulerConfig):
        submit_trials_to_slurm(artifacts, max_parallel=config.scheduler.max_parallel)
    else:
        raise ValueError(f"Unsupported sweep scheduler: {config.scheduler}")
