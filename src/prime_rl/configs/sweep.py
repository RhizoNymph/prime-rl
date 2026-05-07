from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, model_validator

from prime_rl.utils.config import BaseConfig


class SweepParameterConfig(BaseConfig):
    """Choice-valued parameter for Phase 1 grid sweeps."""

    values: Annotated[list[Any], Field(description="Explicit values to sweep over.")]

    @model_validator(mode="after")
    def validate_values(self):
        if not self.values:
            raise ValueError("Sweep parameter values must be non-empty")
        return self


class LocalSweepSchedulerConfig(BaseConfig):
    """Run generated trials as local subprocesses."""

    type: Literal["local"] = "local"

    max_parallel: Annotated[int, Field(ge=1, description="Maximum local trials to run concurrently.")] = 1

    @model_validator(mode="after")
    def reject_parallel_until_phase_3(self):
        if self.max_parallel > 1:
            raise ValueError(
                "Local sweep scheduler does not yet support max_parallel > 1. "
                "Parallel execution requires explicit GPU assignment (Phase 3)."
            )
        return self


class SlurmSweepSchedulerConfig(BaseConfig):
    """Submit generated trials through the target entrypoint's SLURM support."""

    type: Literal["slurm"] = "slurm"

    max_parallel: Annotated[int, Field(ge=1, description="Maximum SLURM submissions to keep in flight.")] = 1


SweepSchedulerConfig: TypeAlias = Annotated[
    LocalSweepSchedulerConfig | SlurmSweepSchedulerConfig,
    Field(discriminator="type"),
]


class SweepWandbConfig(BaseConfig):
    """W&B metadata injected into generated trials."""

    enabled: bool = True
    group: str | None = None
    tags: list[str] = ["sweep"]


class SweepConfig(BaseConfig):
    """Configures a hyperparameter sweep study."""

    name: str | None = None
    entrypoint: Literal["rl", "sft"] = "rl"
    base: list[Path]
    output_dir: Path
    strategy: Literal["grid"] = "grid"
    scheduler: SweepSchedulerConfig = LocalSweepSchedulerConfig()
    parameters: dict[str, SweepParameterConfig]
    wandb: SweepWandbConfig | None = SweepWandbConfig()
    continue_on_failure: Annotated[
        bool,
        Field(description="Schedule remaining trials when one fails. Set false to halt-on-first-fail."),
    ] = True
    retry_budget: Annotated[
        int,
        Field(ge=0, description="Retry a failed trial up to this many times before marking it failed."),
    ] = 1
    dry_run: bool = False
    clean_output_dir: bool = False

    @model_validator(mode="after")
    def validate_sweep(self):
        if not self.base:
            raise ValueError("Sweep base must include at least one target config file")
        if not self.parameters:
            raise ValueError("Sweep parameters must include at least one parameter")
        return self
