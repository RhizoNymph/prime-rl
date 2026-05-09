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


class NoPrunerConfig(BaseConfig):
    """Disable pruning. Trials run to completion regardless of intermediate values."""

    type: Literal["none"] = "none"


class MedianPrunerConfig(BaseConfig):
    """Optuna's MedianPruner: prune trials whose intermediate value falls below
    the running median of completed trials at the same step."""

    type: Literal["median"] = "median"
    n_startup_trials: Annotated[
        int,
        Field(ge=0, description="Trials that must complete before pruning is enabled."),
    ] = 5
    n_warmup_steps: Annotated[
        int,
        Field(ge=0, description="Steps within a trial that are exempt from pruning."),
    ] = 0
    interval_steps: Annotated[
        int,
        Field(ge=1, description="Pruning is only checked every Nth reported step."),
    ] = 1


class AshaPrunerConfig(BaseConfig):
    """Optuna's SuccessiveHalvingPruner (ASHA). Promotes trials whose intermediate
    value is in the top ``1/reduction_factor`` at each rung."""

    type: Literal["asha"] = "asha"
    min_resource: Annotated[
        int | Literal["auto"],
        Field(description="Minimum resource (steps) before a trial can be pruned."),
    ] = "auto"
    reduction_factor: Annotated[
        int,
        Field(ge=2, description="At each rung, keep the top 1/reduction_factor of trials."),
    ] = 4
    min_early_stopping_rate: Annotated[
        int,
        Field(ge=0, description="Bracket index offset; 0 enables the most aggressive bracket."),
    ] = 0


class HyperbandPrunerConfig(BaseConfig):
    """Optuna's HyperbandPruner: runs successive-halving across multiple brackets."""

    type: Literal["hyperband"] = "hyperband"
    min_resource: Annotated[
        int,
        Field(ge=1, description="Smallest resource budget evaluated in any bracket."),
    ] = 1
    max_resource: Annotated[
        int | Literal["auto"],
        Field(description="Largest resource budget; ``auto`` infers from reported steps."),
    ] = "auto"
    reduction_factor: Annotated[
        int,
        Field(ge=2, description="At each rung, keep the top 1/reduction_factor of trials."),
    ] = 3


PrunerConfig: TypeAlias = Annotated[
    NoPrunerConfig | MedianPrunerConfig | AshaPrunerConfig | HyperbandPrunerConfig,
    Field(discriminator="type"),
]


class OptunaStrategyConfig(BaseConfig):
    """Adaptive sampling backed by Optuna.

    Samplers: ``tpe`` (default) and ``random``. Pruners: ``none`` (default),
    ``median``, ``asha`` (successive-halving), and ``hyperband``. Pruners need
    intermediate metric reporting from the trial; the controller polls a
    sidecar metrics stream while the trial runs and calls
    ``optuna_trial.report``/``should_prune`` between samples.

    Storage defaults to in-memory; pass a SQLAlchemy URL (e.g.
    ``"sqlite:///optuna.db"``) to persist the study across resume.
    """

    type: Literal["optuna"] = "optuna"
    num_trials: Annotated[int, Field(ge=1, description="Number of trials to evaluate.")]
    seed: int | None = None
    sampler: Literal["tpe", "random"] = "tpe"
    pruner: PrunerConfig = NoPrunerConfig()
    storage: Annotated[
        str | None,
        Field(description="SQLAlchemy storage URL for study persistence; in-memory if unset."),
    ] = None
    study_name: Annotated[
        str | None,
        Field(description="Optuna study_name; defaults to the sweep name."),
    ] = None
    poll_interval_seconds: Annotated[
        float,
        Field(
            gt=0,
            description=(
                "How often the controller polls the trial's intermediate metrics "
                "while pruning is enabled. Ignored when pruner.type == 'none'."
            ),
        ),
    ] = 5.0


SearchStrategyConfig: TypeAlias = Annotated[
    GridStrategyConfig | RandomStrategyConfig | OptunaStrategyConfig,
    Field(discriminator="type"),
]


class LocalGpuAssignmentConfig(BaseConfig):
    """Static round-robin assignment of CUDA_VISIBLE_DEVICES to local workers.

    Each entry in ``visible_devices`` is one device group that pins one trial
    subprocess. Groups are disjoint by construction so two parallel workers
    never share a GPU. ``mode`` is currently fixed to ``"static"``; future
    modes (``"exclusive"`` for live GPU discovery, ``"none"`` to leave
    ``CUDA_VISIBLE_DEVICES`` untouched) will land in later phases.
    """

    mode: Literal["static"] = "static"
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


# Parameter paths a multi_run_lora sweep is allowed to vary. Must stay in
# sync with what the trainer's MultiRunManager treats as per-run-safe; see
# src/prime_rl/trainer/runs.py for the runtime validation hook. Anything
# under trainer.*, model.*, deployment.*, or inference.* is shared across
# runs and would silently mismatch between trials, so it is rejected at
# config-load time.
MULTI_RUN_LORA_PARAMETER_PREFIXES: tuple[str, ...] = (
    "orchestrator.optim.",
    "orchestrator.model.lora.",
    "orchestrator.sampling.",
    "orchestrator.environment.",
    "orchestrator.batch.",
    "orchestrator.buffer.",
    "orchestrator.eval.",
)


class MultiRunLoRASchedulerConfig(BaseConfig):
    """Run all trials concurrently against one shared trainer + inference.

    Static sweeps launch a single ``rl-multi-run`` invocation that brings up
    one trainer (with ``trainer.max_concurrent_runs >= num_trials``), one
    inference server, and ``num_trials`` orchestrators — one per trial.
    Optuna sweeps run in continuous-flow mode so newly freed slots can be
    replenished while in-flight trials are pruned between intermediate
    metric reports. Resume can reattach to an existing shared trainer state
    and continue pending trials.
    """

    type: Literal["multi_run_lora"] = "multi_run_lora"
    max_concurrent_runs: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Number of concurrent orchestrator runs against the shared trainer. "
                "Must match (or be <=) trainer.max_concurrent_runs in the shared base config."
            ),
        ),
    ]
    shared: Annotated[
        list[Path],
        Field(
            min_length=1,
            description=(
                "RLConfig base TOML(s) describing the shared trainer + inference. "
                "Trial overrides apply to the orchestrator block only."
            ),
        ),
    ]


SweepSchedulerConfig: TypeAlias = Annotated[
    LocalSweepSchedulerConfig | SlurmSweepSchedulerConfig | MultiRunLoRASchedulerConfig,
    Field(discriminator="type"),
]


class SweepWandbConfig(BaseConfig):
    """W&B metadata injected into generated trials."""

    enabled: bool = True
    group: str | None = None
    tags: list[str] = ["sweep"]


class ObjectiveConfig(BaseConfig):
    """Names the metric the sweep optimizes and where to read it from."""

    metric: Annotated[
        str,
        Field(description="Metric key inside final_summary.json (forward-slash-separated)."),
    ]
    direction: Literal["maximize", "minimize"]
    source: Literal["final_summary"] = "final_summary"


class ThresholdStoppingConfig(BaseConfig):
    """Halt the study after a trial whose objective is on the wrong side of a threshold."""

    type: Literal["threshold"] = "threshold"
    threshold: float
    min_trials: Annotated[
        int,
        Field(ge=1, description="Minimum completed trials before threshold can fire."),
    ] = 1


class PatienceStoppingConfig(BaseConfig):
    """Halt the study after N consecutive completed trials with no improvement."""

    type: Literal["patience"] = "patience"
    patience: Annotated[int, Field(ge=1, description="Consecutive non-improving trials required to halt.")]
    min_trials: Annotated[
        int,
        Field(ge=1, description="Minimum completed trials before patience can fire."),
    ] = 1


EarlyStoppingConfig: TypeAlias = Annotated[
    ThresholdStoppingConfig | PatienceStoppingConfig,
    Field(discriminator="type"),
]


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
    objective: ObjectiveConfig | None = None
    early_stopping: EarlyStoppingConfig | None = None
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
        if self.early_stopping is not None and self.objective is None:
            raise ValueError(
                "early_stopping requires an objective so the controller knows which metric to compare."
            )
        if self.early_stopping is not None and isinstance(self.scheduler, SlurmSweepSchedulerConfig):
            raise ValueError(
                "early_stopping is not supported with the SLURM scheduler: the controller submits "
                "jobs and exits, so it never observes trial completion to decide when to halt."
            )
        if isinstance(self.strategy, OptunaStrategyConfig):
            if self.objective is None:
                raise ValueError("Optuna strategy requires an objective to optimize.")
            if isinstance(self.scheduler, SlurmSweepSchedulerConfig):
                raise ValueError(
                    "Optuna strategy is not supported with the SLURM scheduler: the controller "
                    "must observe each trial's objective before proposing the next one."
                )
            if isinstance(self.scheduler, LocalSweepSchedulerConfig) and self.scheduler.max_parallel > 1:
                raise ValueError(
                    "Optuna strategy runs sequentially (ask/tell needs each trial's objective "
                    "before proposing the next), so scheduler.max_parallel must be 1."
                )
            if self.resume and self.strategy.storage is None:
                raise ValueError(
                    "Resume with the Optuna strategy requires strategy.storage so the study "
                    "can be reloaded; in-memory studies vanish when the controller exits."
                )
        if isinstance(self.scheduler, MultiRunLoRASchedulerConfig):
            if self.entrypoint != "rl":
                raise ValueError(
                    "multi_run_lora scheduler is RL-only; the shared-trainer architecture "
                    "depends on the trainer's MultiRunManager which only the rl entrypoint runs."
                )
            offending = [
                path
                for path in self.parameters
                if not any(path.startswith(prefix) for prefix in MULTI_RUN_LORA_PARAMETER_PREFIXES)
            ]
            if offending:
                allowed = ", ".join(MULTI_RUN_LORA_PARAMETER_PREFIXES)
                raise ValueError(
                    "multi_run_lora sweeps may only vary per-run orchestrator fields. "
                    f"These parameter paths are not in the allowlist ({allowed}): {offending}. "
                    "Trainer/model/deployment/inference settings cannot vary inside one shared trainer."
                )
        return self
