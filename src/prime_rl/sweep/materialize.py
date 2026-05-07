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


def materialize_trial(config: SweepConfig, trial: Trial) -> TrialArtifacts:
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

    write_json(
        status_path,
        {
            "id": trial.id,
            "label": trial.label,
            "state": "pending",
            "pid": None,
            "slurm_job_id": None,
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
