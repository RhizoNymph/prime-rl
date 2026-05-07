from pathlib import Path

import pytest
import tomli_w
from pydantic import ValidationError

from prime_rl.configs.sweep import (
    ChoiceParameterConfig,
    IntUniformParameterConfig,
    LogUniformParameterConfig,
    RandomStrategyConfig,
    SweepConfig,
    UniformParameterConfig,
)
from prime_rl.utils.config import cli


def write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def test_sweep_config_defaults(tmp_path: Path) -> None:
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
    )

    assert config.entrypoint == "rl"
    assert config.strategy.type == "grid"
    assert config.scheduler.type == "local"
    assert config.scheduler.max_parallel == 1
    assert isinstance(config.parameters["optim.lr"], ChoiceParameterConfig)


def test_sweep_config_loads_from_cli_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "sweep.toml"
    write_toml(
        config_path,
        {
            "entrypoint": "sft",
            "base": ["base.toml"],
            "output_dir": "outputs/study",
            "scheduler": {"type": "slurm"},
            "parameters": {"optim.lr": {"values": [1e-5]}},
        },
    )

    config = cli(SweepConfig, args=["@", config_path.as_posix()])

    assert config.entrypoint == "sft"
    assert config.scheduler.type == "slurm"
    assert config.parameters["optim.lr"].values == [1e-5]


def test_slurm_scheduler_rejects_max_parallel(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={"type": "slurm", "max_parallel": 4},
            parameters={"optim.lr": {"values": [1e-5]}},
        )


def test_sweep_config_requires_base(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="base"):
        SweepConfig(
            base=[],
            output_dir=tmp_path / "study",
            parameters={"optim.lr": {"values": [1e-5]}},
        )


def test_sweep_config_requires_parameters(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="parameters"):
        SweepConfig(base=[tmp_path / "base.toml"], output_dir=tmp_path / "study", parameters={})


def test_sweep_parameter_requires_values(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="values"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={"optim.lr": {"values": []}},
        )


def test_local_max_parallel_requires_gpu_assignment(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="gpu_assignment"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={"type": "local", "max_parallel": 2},
            parameters={"optim.lr": {"values": [1e-5]}},
        )


def test_local_max_parallel_requires_enough_gpu_groups(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="requires at least"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={
                "type": "local",
                "max_parallel": 4,
                "gpu_assignment": {"visible_devices": [[0], [1]]},
            },
            parameters={"optim.lr": {"values": [1e-5]}},
        )


def test_local_gpu_assignment_rejects_overlapping_groups(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="only appear in one"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={
                "type": "local",
                "max_parallel": 2,
                "gpu_assignment": {"visible_devices": [[0, 1], [1, 2]]},
            },
            parameters={"optim.lr": {"values": [1e-5]}},
        )


def test_local_gpu_assignment_rejects_empty_group(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="at least one device"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            scheduler={
                "type": "local",
                "max_parallel": 2,
                "gpu_assignment": {"visible_devices": [[0], []]},
            },
            parameters={"optim.lr": {"values": [1e-5]}},
        )


def test_local_max_parallel_with_gpu_assignment_validates(tmp_path: Path) -> None:
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "local",
            "max_parallel": 2,
            "gpu_assignment": {"visible_devices": [[0, 1], [2, 3]]},
        },
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
    )
    assert config.scheduler.max_parallel == 2
    assert config.scheduler.gpu_assignment.visible_devices == [[0, 1], [2, 3]]


def test_random_strategy_accepts_distribution_parameters(tmp_path: Path) -> None:
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        strategy={"type": "random", "num_trials": 4, "seed": 7},
        parameters={
            "optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4},
            "optim.warmup": {"distribution": "int_uniform", "min": 0, "max": 10, "step": 2},
            "data.temperature": {"distribution": "uniform", "min": 0.6, "max": 1.2},
        },
    )

    assert isinstance(config.strategy, RandomStrategyConfig)
    assert config.strategy.num_trials == 4
    assert config.strategy.seed == 7
    assert isinstance(config.parameters["optim.lr"], LogUniformParameterConfig)
    assert isinstance(config.parameters["optim.warmup"], IntUniformParameterConfig)
    assert isinstance(config.parameters["data.temperature"], UniformParameterConfig)


def test_grid_strategy_rejects_distribution_parameters(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="Grid strategy"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={"optim.lr": {"distribution": "uniform", "min": 0.0, "max": 1.0}},
        )


def test_log_uniform_requires_positive_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="positive"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            strategy={"type": "random", "num_trials": 1},
            parameters={"optim.lr": {"distribution": "log_uniform", "min": 0.0, "max": 1e-4}},
        )


def test_resume_and_clean_output_dir_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            parameters={"optim.lr": {"values": [1e-5]}},
            resume=True,
            clean_output_dir=True,
        )


def test_resume_rejects_unseeded_random_strategy(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="seed"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            strategy={"type": "random", "num_trials": 4},
            parameters={"optim.lr": {"distribution": "uniform", "min": 0.0, "max": 1.0}},
            resume=True,
        )


def test_resume_accepts_seeded_random_strategy(tmp_path: Path) -> None:
    config = SweepConfig(
        base=[tmp_path / "base.toml"],
        output_dir=tmp_path / "study",
        strategy={"type": "random", "num_trials": 4, "seed": 11},
        parameters={"optim.lr": {"distribution": "uniform", "min": 0.0, "max": 1.0}},
        resume=True,
    )
    assert config.strategy.seed == 11


def test_int_uniform_rejects_non_divisible_step(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="divisible"):
        SweepConfig(
            base=[tmp_path / "base.toml"],
            output_dir=tmp_path / "study",
            strategy={"type": "random", "num_trials": 4},
            parameters={"optim.warmup": {"distribution": "int_uniform", "min": 0, "max": 10, "step": 4}},
        )
