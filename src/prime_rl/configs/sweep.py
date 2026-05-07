from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Discriminator, Field, Tag, model_validator

from prime_rl.utils.config import BaseConfig


class ChoiceParameterConfig(BaseConfig):
    """Choice-valued parameter sampled from an explicit list."""

    distribution: Literal["choice"] = "choice"
    values: Annotated[list[Any], Field(description="Explicit values to sweep over.")]

    @model_validator(mode="after")
    def validate_values(self):
        if not self.values:
            raise ValueError("Sweep parameter values must be non-empty")
        return self


class UniformParameterConfig(BaseConfig):
    """Continuous parameter sampled uniformly on [min, max]."""

    distribution: Literal["uniform"]
    min: float
    max: float

    @model_validator(mode="after")
    def validate_range(self):
        if self.min >= self.max:
            raise ValueError("Uniform parameter requires min < max")
        return self


class LogUniformParameterConfig(BaseConfig):
    """Continuous parameter sampled uniformly in log-space on [min, max]."""

    distribution: Literal["log_uniform"]
    min: float
    max: float

    @model_validator(mode="after")
    def validate_range(self):
        if self.min <= 0 or self.max <= 0:
            raise ValueError("Log-uniform parameter requires positive min and max")
        if self.min >= self.max:
            raise ValueError("Log-uniform parameter requires min < max")
        return self


class IntUniformParameterConfig(BaseConfig):
    """Integer parameter sampled uniformly from {min, min+step, ..., max}."""

    distribution: Literal["int_uniform"]
    min: int
    max: int
    step: Annotated[int, Field(ge=1)] = 1

    @model_validator(mode="after")
    def validate_range(self):
        if self.min >= self.max:
            raise ValueError("Int-uniform parameter requires min < max")
        if (self.max - self.min) % self.step != 0:
            raise ValueError(
                f"Int-uniform range [{self.min}, {self.max}] is not divisible by step {self.step}; "
                "non-divisible ranges silently truncate the search space (the inclusive max is never sampled). "
                "Pick a step that divides (max - min) evenly."
            )
        return self


def _parameter_discriminator(value: Any) -> str:
    """Default to ``choice`` so the bare ``{"values": [...]}`` form keeps working."""
    if isinstance(value, dict):
        return value.get("distribution", "choice")
    return getattr(value, "distribution", "choice")


SweepParameterConfig: TypeAlias = Annotated[
    Annotated[ChoiceParameterConfig, Tag("choice")]
    | Annotated[UniformParameterConfig, Tag("uniform")]
    | Annotated[LogUniformParameterConfig, Tag("log_uniform")]
    | Annotated[IntUniformParameterConfig, Tag("int_uniform")],
    Discriminator(_parameter_discriminator),
]


class GridStrategyConfig(BaseConfig):
    """Exhaustive grid over choice-valued parameters."""

    type: Literal["grid"] = "grid"


class RandomStrategyConfig(BaseConfig):
    """Independent random samples from the declared parameter distributions."""

    type: Literal["random"] = "random"
    num_trials: Annotated[int, Field(ge=1, description="Number of trials to draw.")]
    seed: Annotated[int | None, Field(description="Optional seed for reproducibility.")] = None


SearchStrategyConfig: TypeAlias = Annotated[
    GridStrategyConfig | RandomStrategyConfig,
    Field(discriminator="type"),
]


class LocalGpuAssignmentConfig(BaseConfig):
    """Static round-robin assignment of CUDA_VISIBLE_DEVICES to local workers.

    Each entry in ``visible_devices`` is one device group that pins one trial
    subprocess. Groups are disjoint by construction so two parallel workers
    never share a GPU.
    """

    visible_devices: Annotated[
        list[list[int]],
        Field(min_length=1, description="Disjoint device groups assigned to parallel workers."),
    ]

    @model_validator(mode="after")
    def validate_groups(self):
        if any(not group for group in self.visible_devices):
            raise ValueError("Each visible_devices group must contain at least one device index")
        flat = [device for group in self.visible_devices for device in group]
        if any(device < 0 for device in flat):
            raise ValueError("visible_devices indices must be non-negative")
        if len(flat) != len(set(flat)):
            raise ValueError("Each device may only appear in one visible_devices group")
        return self


class LocalSweepSchedulerConfig(BaseConfig):
    """Run generated trials as local subprocesses."""

    type: Literal["local"] = "local"

    max_parallel: Annotated[int, Field(ge=1, description="Maximum local trials to run concurrently.")] = 1
    gpu_assignment: Annotated[
        LocalGpuAssignmentConfig | None,
        Field(description="Required for max_parallel > 1; pins each worker to a disjoint device group."),
    ] = None

    @model_validator(mode="after")
    def validate_parallel(self):
        if self.max_parallel > 1:
            if self.gpu_assignment is None:
                raise ValueError(
                    "max_parallel > 1 requires explicit gpu_assignment so parallel workers do not "
                    "silently colocate trainer/inference stacks on the same GPUs."
                )
            available = len(self.gpu_assignment.visible_devices)
            if available < self.max_parallel:
                raise ValueError(
                    f"max_parallel={self.max_parallel} requires at least {self.max_parallel} "
                    f"visible_devices groups, got {available}."
                )
        return self


class SlurmSweepSchedulerConfig(BaseConfig):
    """Submit generated trials through the target entrypoint's SLURM support.

    Throughput is governed by the SLURM cluster, not this scheduler. A
    controller-managed in-flight cap will land in a later phase; until then
    there is intentionally no ``max_parallel`` knob to avoid promising
    throttling we do not enforce.
    """

    type: Literal["slurm"] = "slurm"


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
    strategy: SearchStrategyConfig = GridStrategyConfig()
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
    resume: Annotated[
        bool,
        Field(description="Reattach to an existing study output dir; preserve completed trial state."),
    ] = False
    dry_run: bool = False
    clean_output_dir: bool = False

    @model_validator(mode="after")
    def validate_sweep(self):
        if not self.base:
            raise ValueError("Sweep base must include at least one target config file")
        if not self.parameters:
            raise ValueError("Sweep parameters must include at least one parameter")
        if self.resume and self.clean_output_dir:
            raise ValueError("resume and clean_output_dir are mutually exclusive")
        if isinstance(self.strategy, GridStrategyConfig):
            non_choice = [
                path for path, parameter in self.parameters.items() if not isinstance(parameter, ChoiceParameterConfig)
            ]
            if non_choice:
                raise ValueError(
                    "Grid strategy only supports choice (values=...) parameters, "
                    f"but these declare distributions instead: {non_choice}"
                )
        if self.resume and isinstance(self.strategy, RandomStrategyConfig) and self.strategy.seed is None:
            raise ValueError(
                "resume requires a deterministic trial set, but the random strategy has no seed. "
                "Set strategy.seed so trial IDs match the previous study, or drop resume."
            )
        return self
