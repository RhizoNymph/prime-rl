import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tomli_w

from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig
from prime_rl.configs.sweep import SweepConfig
from prime_rl.sweep.reproducibility import file_checksum
from prime_rl.utils.config import BaseConfig, cli


@dataclass(frozen=True)
class Trial:
    id: str
    label: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class TrialArtifacts:
    trial: Trial
    trial_dir: Path
    run_dir: Path
    overrides_path: Path
    resolved_path: Path
    command_path: Path
    status_path: Path
    command: list[str]
    resolved_checksum: str
    base_checksums: dict[str, str]


def set_dotted_path(data: dict[str, Any], path: str, value: Any) -> None:
    if not path:
        raise ValueError("Sweep parameter path cannot be empty")

    parts = path.split(".")
    current = data
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {path}: {part} is already set to a non-table value")
        current = child
    current[parts[-1]] = value


def build_nested_overrides(flat_overrides: dict[str, Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for path, value in flat_overrides.items():
        set_dotted_path(nested, path, value)
    return nested


def sanitize_label_part(value: Any) -> str:
    text = str(value)
    for char in ("/", "\\", " ", ":", ",", "[", "]", "{", "}", "'", '"'):
        text = text.replace(char, "_")
    return text


def trial_label(parameters: dict[str, Any], max_len: int = 96) -> str:
    parts = []
    for path, value in parameters.items():
        name = path.split(".")[-1].replace("_", "-")
        parts.append(f"{name}_{sanitize_label_part(value)}")
    label = "-".join(parts)
    return label if len(label) <= max_len else ""


def command_for_trial(
    entrypoint: Literal["rl", "sft"],
    base_paths: list[Path],
    overrides_path: Path,
) -> list[str]:
    """Compose the launcher command from base files plus the generated overrides.

    This matches the form a user would type by hand and keeps per-trial diffs
    small. The frozen ``resolved.toml`` is written separately as a reproducible
    artifact but is not used as the launch input.
    """
    cmd = ["uv", "run", entrypoint]
    for base in base_paths:
        cmd.extend(["@", base.as_posix()])
    cmd.extend(["@", overrides_path.as_posix()])
    return cmd


def write_toml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def record_trial_objective(status_path: Path, value: float | None) -> None:
    """Persist an objective value into a trial's status.json, preserving other fields."""
    status = json.loads(status_path.read_text())
    status["objective"] = value
    write_json(status_path, status)


def record_trial_pruned(status_path: Path, step: int, value: float) -> None:
    """Mark a trial as pruned by intermediate-metric reporting.

    The Optuna pruning loop terminates the trial subprocess when the sampler
    decides the trajectory is unpromising. We record state="pruned" plus the
    step/value the prune fired on so the manifest can tell pruned trials
    apart from completed and failed runs without re-deriving the cause.
    """
    status = json.loads(status_path.read_text())
    status["state"] = "pruned"
    status["pruned_at_step"] = int(step)
    status["pruned_value"] = float(value)
    status["objective"] = None
    write_json(status_path, status)


def validate_target_config(entrypoint: Literal["rl", "sft"], args: list[str]) -> BaseConfig:
    config_cls = RLConfig if entrypoint == "rl" else SFTConfig
    return cli(config_cls, args=args)


def _target_config_to_toml(config: BaseConfig) -> dict[str, Any]:
    return config.model_dump(exclude_none=True, mode="json")


def _merge_wandb_overrides(config: SweepConfig, flat_overrides: dict[str, Any], trial: Trial) -> None:
    if config.wandb is None or not config.wandb.enabled:
        return

    group = config.wandb.group or config.name
    if group is not None:
        flat_overrides["wandb.group"] = group

    flat_overrides["wandb.name"] = trial.label or trial.id

    tags = list(dict.fromkeys([*config.wandb.tags, "sweep", f"trial:{trial.id}"]))
    if config.name is not None:
        tags.append(f"study:{config.name}")
    flat_overrides["wandb.tags"] = list(dict.fromkeys(tags))


TERMINAL_RESUME_STATES = frozenset({"completed", "submitted"})


class SweepDriftError(RuntimeError):
    """Raised when --resume would skip a trial whose effective config has changed."""


def _existing_terminal_status(status_path: Path) -> dict[str, Any] | None:
    """Return parsed status.json if its state should be preserved on resume."""
    if not status_path.exists():
        return None
    status = json.loads(status_path.read_text())
    if status.get("state") in TERMINAL_RESUME_STATES:
        return status
    return None


def _check_resume_drift(
    trial: Trial,
    preserved_status: dict[str, Any],
    expected: dict[str, Any] | None,
    new_resolved_checksum: str,
    new_base_checksums: dict[str, str],
) -> None:
    """Refuse to skip a terminal trial whose recorded config differs from the live one.

    Trial IDs hash sweep parameters only, so a base TOML edit between runs
    leaves the ID stable while the resolved config changes underneath us.
    Without this check ``--resume`` would silently honor the old ``status.json``
    and skip work that no longer reflects the current configuration.
    """
    if expected is None:
        return

    expected_resolved = expected.get("resolved_checksum")
    expected_bases = expected.get("base_checksums") or {}

    changed_bases = [
        base for base, checksum in new_base_checksums.items() if expected_bases.get(base, checksum) != checksum
    ]
    resolved_drift = expected_resolved is not None and expected_resolved != new_resolved_checksum

    if not (changed_bases or resolved_drift):
        return

    detail = f"changed base files: {changed_bases}" if changed_bases else "the resolved config changed"
    raise SweepDriftError(
        f"Refusing to skip {preserved_status['state']} trial {trial.id} on resume because "
        f"{detail}. Drop --resume to start fresh, revert the change, or remove the trial directory."
    )


def materialize_trial(
    config: SweepConfig,
    trial: Trial,
    resume: bool = False,
    expected_checksums: dict[str, Any] | None = None,
) -> TrialArtifacts:
    trial_dir = config.output_dir / "trials" / trial.id
    run_dir = trial_dir / "run"
    overrides_path = trial_dir / "overrides.toml"
    resolved_path = trial_dir / "resolved.toml"
    command_path = trial_dir / "command.txt"
    status_path = trial_dir / "status.json"

    flat_overrides = dict(trial.parameters)
    flat_overrides["output_dir"] = run_dir.as_posix()
    _merge_wandb_overrides(config, flat_overrides, trial)

    overrides = build_nested_overrides(flat_overrides)
    write_toml(overrides_path, overrides)

    args: list[str] = []
    for base_path in config.base:
        args.extend(["@", base_path.as_posix()])
    args.extend(["@", overrides_path.as_posix()])

    resolved_config = validate_target_config(config.entrypoint, args)
    write_toml(resolved_path, _target_config_to_toml(resolved_config))

    command = command_for_trial(config.entrypoint, config.base, overrides_path)
    command_path.write_text(" ".join(command) + "\n")

    resolved_checksum = file_checksum(resolved_path)
    base_checksums = {base.as_posix(): file_checksum(base) for base in config.base}

    preserved_status = _existing_terminal_status(status_path) if resume else None
    if preserved_status is not None:
        _check_resume_drift(trial, preserved_status, expected_checksums, resolved_checksum, base_checksums)
    else:
        write_json(
            status_path,
            {
                "id": trial.id,
                "label": trial.label,
                "state": "pending",
                "pid": None,
                "slurm_job_id": None,
                "gpu_group": None,
                "returncode": None,
                "objective": None,
            },
        )

    return TrialArtifacts(
        trial=trial,
        trial_dir=trial_dir,
        run_dir=run_dir,
        overrides_path=overrides_path,
        resolved_path=resolved_path,
        command_path=command_path,
        status_path=status_path,
        command=command,
        resolved_checksum=resolved_checksum,
        base_checksums=base_checksums,
    )


def _merge_multi_run_wandb_overrides(
    config: SweepConfig, flat_overrides: dict[str, Any], trial: Trial
) -> None:
    """Tag the per-run orchestrator's W&B run with sweep + trial metadata."""
    if config.wandb is None or not config.wandb.enabled:
        return

    group = config.wandb.group or config.name
    if group is not None:
        flat_overrides["orchestrator.wandb.group"] = group

    flat_overrides["orchestrator.wandb.name"] = trial.label or trial.id

    tags = list(dict.fromkeys([*config.wandb.tags, "sweep", f"trial:{trial.id}"]))
    if config.name is not None:
        tags.append(f"study:{config.name}")
    flat_overrides["orchestrator.wandb.tags"] = list(dict.fromkeys(tags))


def multi_run_shared_dir(config: SweepConfig) -> Path:
    """Directory that hosts the shared trainer's output and per-run subdirs.

    The trainer's ``MultiRunManager`` scans ``<dir>/run_*`` so every trial
    directory must sit directly under this path with a ``run_`` prefix.
    """
    return config.output_dir / "shared"


def multi_run_trial_dir(config: SweepConfig, trial: Trial) -> Path:
    """Per-trial directory the trainer will discover as a ``run_*`` slot."""
    return multi_run_shared_dir(config) / f"run_{trial.id}"


def materialize_multi_run_trial(
    config: SweepConfig,
    trial: Trial,
    scheduler: Any,  # MultiRunLoRASchedulerConfig — typed as Any to avoid an import cycle
) -> TrialArtifacts:
    """Write a per-trial ``run_<id>/control/orch.toml`` for a shared-trainer sweep.

    The shared base TOMLs in ``scheduler.shared`` resolve to a full RLConfig.
    Per-trial parameter overrides (already prefixed with ``orchestrator.``)
    are layered on top, the orchestrator block is extracted, and its TOML is
    written where the trainer's ``MultiRunManager`` will find it. Returned
    ``TrialArtifacts.run_dir`` points at the per-trial directory so the
    sweep's existing metrics readers (``read_final_summary`` /
    ``read_intermediate_metric``) keep working unchanged once the
    orchestrator's ``FileMonitor`` writes ``metrics.jsonl`` there.
    """
    run_dir = multi_run_trial_dir(config, trial)
    control_dir = run_dir / "control"
    overrides_path = run_dir / "overrides.toml"
    resolved_path = run_dir / "resolved.toml"
    command_path = run_dir / "command.txt"
    status_path = run_dir / "status.json"
    orch_config_path = control_dir / "orch.toml"

    # Trial overrides already use orchestrator.* paths thanks to the
    # validator's allowlist. Add the per-run output_dir + W&B identity.
    flat_overrides: dict[str, Any] = dict(trial.parameters)
    flat_overrides["orchestrator.output_dir"] = run_dir.as_posix()
    _merge_multi_run_wandb_overrides(config, flat_overrides, trial)

    overrides = build_nested_overrides(flat_overrides)
    write_toml(overrides_path, overrides)

    # Resolve the full RLConfig with the shared base + trial overrides; the
    # orchestrator block is what we ship to the trainer.
    args: list[str] = []
    for base_path in scheduler.shared:
        args.extend(["@", base_path.as_posix()])
    args.extend(["@", overrides_path.as_posix()])

    resolved_rl_config = validate_target_config("rl", args)
    orchestrator_dict = resolved_rl_config.orchestrator.model_dump(exclude_none=True, mode="json")

    write_toml(resolved_path, orchestrator_dict)
    write_toml(orch_config_path, orchestrator_dict)

    command = ["rl-multi-run", "@", *(p.as_posix() for p in scheduler.shared), f"--run={run_dir.as_posix()}"]
    command_path.write_text(" ".join(command) + "\n")

    resolved_checksum = file_checksum(resolved_path)
    base_checksums = {base.as_posix(): file_checksum(base) for base in scheduler.shared}

    write_json(
        status_path,
        {
            "id": trial.id,
            "label": trial.label,
            "state": "pending",
            "pid": None,
            "slurm_job_id": None,
            "gpu_group": None,
            "returncode": None,
            "objective": None,
        },
    )

    return TrialArtifacts(
        trial=trial,
        trial_dir=run_dir,
        run_dir=run_dir,
        overrides_path=overrides_path,
        resolved_path=resolved_path,
        command_path=command_path,
        status_path=status_path,
        command=command,
        resolved_checksum=resolved_checksum,
        base_checksums=base_checksums,
    )
