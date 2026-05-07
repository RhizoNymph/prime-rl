from pathlib import Path

import pytest
import tomli_w
from pydantic import ValidationError

from prime_rl.configs.sweep import SweepConfig
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
    assert config.strategy == "grid"
    assert config.scheduler.type == "local"
    assert config.scheduler.max_parallel == 1


def test_sweep_config_loads_from_cli_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "sweep.toml"
    write_toml(
        config_path,
        {
            "entrypoint": "sft",
            "base": ["base.toml"],
            "output_dir": "outputs/study",
            "scheduler": {"type": "slurm", "max_parallel": 4},
            "parameters": {"optim.lr": {"values": [1e-5]}},
        },
    )

    config = cli(SweepConfig, args=["@", config_path.as_posix()])

    assert config.entrypoint == "sft"
    assert config.scheduler.type == "slurm"
    assert config.scheduler.max_parallel == 4
    assert config.parameters["optim.lr"].values == [1e-5]


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
