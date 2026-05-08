# Hyperparameter Optimization Design

This document specifies hyperparameter optimization support for PRIME-RL across
multiple implementation phases. The full design covers static sweeps, adaptive
optimization, W&B sweep agents, shared-trainer LoRA sweeps, local GPU assignment,
and controlled mutation of active runs. The roadmap at the end defines the order
in which these capabilities should be implemented.

The core principle is that the first implementation should be a thin launcher
around existing `rl` and `sft` runs, while later phases can add richer
coordination once the artifact format, validation flow, and scheduler boundaries
are stable.

## Goals

- Launch reproducible hyperparameter studies over existing typed config fields.
- Reuse the existing config system, validation, entrypoints, SLURM support, and
  monitoring infrastructure.
- Materialize generated variants and controller decisions to disk before they
  affect a run.
- Keep every trial inspectable as a normal PRIME-RL run whenever possible.
- Support both independent full-run studies and optimized shared-trainer LoRA
  studies.
- Support static strategies such as grid and random search.
- Support adaptive strategies such as Bayesian optimization, Optuna studies,
  Hyperband/ASHA, and early stopping.
- Support W&B tracking and, eventually, W&B sweep agents.
- Support local and cluster schedulers with explicit resource ownership.

## Design Principles

- The sweep controller owns trial generation, scheduling, monitoring, and early
  stopping decisions. The trainer and orchestrator stay focused on training;
  there is no active-run control API.
- Generated configs are ordinary TOML configs. The launcher composes them as
  `uv run <rl|sft> @ <base.toml> @ <overrides.toml>` so a trial command matches
  what a user would type. A frozen `resolved.toml` is also written for each
  trial as a reproducible single-file artifact.
- Validation happens before launch whenever the trial is known up front.
  Adaptive studies validate each newly proposed trial before scheduling it.
- Trial identity is stable and filesystem-backed. Trial IDs combine a
  zero-padded integer index with a short stable hash of the override values
  (`0000-a1b2c3d4`) so trial identity survives deletion and regeneration.
- Reproducibility is recorded in the manifest: resolved-config checksum, base
  file checksums, git commit SHA, git dirty flag.
- Advanced schedulers must be explicit. For example, local GPU allocation and
  shared-trainer LoRA sweeps are enabled through scheduler config rather than
  inferred from the search space.

## Terminology

- **Study**: One hyperparameter optimization run, backed by a study output
  directory and manifest.
- **Trial**: One evaluated configuration.
- **Variant**: A statically generated trial for grid/random sweeps. In this
  document, "trial" is preferred for the general concept.
- **Search strategy**: The component that proposes trial parameter values.
- **Scheduler**: The component that turns a trial into local processes, SLURM
  jobs, W&B agent commands, or shared-trainer orchestrator runs.
- **Objective**: The metric and direction used to compare trials.
- **Controller**: The long-running process behind `uv run sweep` that expands
  trials, schedules work, monitors progress, records results, and makes early
  stopping decisions.

## User Interface

Add a new entrypoint:

```bash
uv run sweep @ sweeps/reverse_text.toml
uv run sweep @ sweeps/reverse_text.toml --dry-run
uv run sweep @ sweeps/reverse_text.toml --resume
```

The sweep entrypoint accepts TOML files through the same `@` syntax used by
other entrypoints. CLI overrides apply to sweep config fields, not target run
fields. Target run fields are overridden through the sweep `parameters` section
or generated trial override files.

### Grid Sweep Example

```toml
name = "reverse-text-lr-temp"
entrypoint = "rl"
base = ["examples/reverse_text/rl.toml"]
output_dir = "outputs/studies/reverse-text-lr-temp"

[strategy]
type = "grid"

[scheduler]
type = "local"
max_parallel = 1

[objective]
metric = "eval/reward_mean"
direction = "maximize"

[wandb]
enabled = true
group = "reverse-text-lr-temp"
tags = ["sweep"]

[parameters."trainer.optim.lr"]
values = [1e-5, 3e-5, 1e-4]

[parameters."orchestrator.train.sampling.temperature"]
values = [0.7, 1.0]
```

### Random Search Example

```toml
name = "reverse-text-random"
entrypoint = "rl"
base = ["examples/reverse_text/rl.toml"]
output_dir = "outputs/studies/reverse-text-random"

[strategy]
type = "random"
num_trials = 32
seed = 7

[scheduler]
type = "slurm"
max_parallel = 8

[parameters."trainer.optim.lr"]
distribution = "log_uniform"
min = 1e-6
max = 1e-4

[parameters."orchestrator.train.sampling.temperature"]
distribution = "uniform"
min = 0.6
max = 1.2
```

### Optuna Example

```toml
name = "reverse-text-optuna"
entrypoint = "rl"
base = ["examples/reverse_text/rl.toml"]
output_dir = "outputs/studies/reverse-text-optuna"

[strategy]
type = "optuna"
num_trials = 64
seed = 7
storage = "sqlite:///outputs/studies/reverse-text-optuna/optuna.db"
sampler = "tpe"
pruner = "asha"

[scheduler]
type = "slurm"
max_parallel = 8

[objective]
metric = "eval/reward_mean"
direction = "maximize"

[parameters."trainer.optim.lr"]
distribution = "log_float"
low = 1e-6
high = 1e-4
```

### W&B Sweep Agent Example

```toml
name = "reverse-text-wandb-agent"
entrypoint = "rl"
base = ["examples/reverse_text/rl.toml"]
output_dir = "outputs/studies/reverse-text-wandb-agent"

[strategy]
type = "wandb_agent"
project = "prime-rl"
entity = "prime"
method = "bayes"
count = 32

[scheduler]
type = "wandb_agent"

[objective]
metric = "eval/reward_mean"
direction = "maximize"

[parameters."trainer.optim.lr"]
distribution = "log_uniform_values"
min = 1e-6
max = 1e-4
```

### Shared-Trainer LoRA Sweep Example

```toml
name = "reverse-text-lora-shared"
entrypoint = "rl"
base = ["examples/reverse_text/rl.toml"]
output_dir = "outputs/studies/reverse-text-lora-shared"

[strategy]
type = "grid"

[scheduler]
type = "multi_run_lora"
max_concurrent_runs = 4

[parameters."orchestrator.optim.lr"]
values = [1e-5, 3e-5]

[parameters."orchestrator.model.lora.alpha"]
values = [16.0, 32.0]
```

## Config Schema

Add `src/prime_rl/configs/sweep.py`.

The schema should use discriminated unions for strategies and schedulers.

### Top-Level Config

```python
class SweepConfig(BaseConfig):
    name: str | None = None
    entrypoint: Literal["rl", "sft"] = "rl"
    base: list[Path]
    output_dir: Path
    strategy: SearchStrategyConfig = GridStrategyConfig()
    scheduler: SchedulerConfig = LocalSchedulerConfig()
    objective: ObjectiveConfig | None = None
    parameters: dict[str, ParameterConfig]
    wandb: SweepWandbConfig | None = SweepWandbConfig()
    early_stopping: EarlyStoppingConfig | None = None
    continue_on_failure: bool = True
    retry_budget: int = 1
    resume: bool = False
    dry_run: bool = False
    clean_output_dir: bool = False
```

Validation rules:

- `base` must be non-empty.
- `parameters` must be non-empty unless `strategy.type = "wandb_agent"` and the
  referenced W&B sweep already exists.
- Every parameter must be valid for the selected strategy.
- `entrypoint = "inference"` is rejected because inference jobs are not training
  trials and do not have comparable objective semantics.
- Adaptive strategies require `objective`.
- Early stopping requires a scheduler that can observe trial metrics and either
  stop jobs or mark them as pruned.
- `multi_run_lora` requires `entrypoint = "rl"` and a LoRA-enabled trainer
  config.

### Parameter Config

Support explicit values and typed distributions:

```python
class ChoiceParameterConfig(BaseConfig):
    values: list[Any]


class UniformParameterConfig(BaseConfig):
    distribution: Literal["uniform"]
    min: float
    max: float


class LogUniformParameterConfig(BaseConfig):
    distribution: Literal["log_uniform"]
    min: float
    max: float


class IntUniformParameterConfig(BaseConfig):
    distribution: Literal["int_uniform"]
    min: int
    max: int
    step: int = 1


ParameterConfig = Annotated[
    ChoiceParameterConfig
    | UniformParameterConfig
    | LogUniformParameterConfig
    | IntUniformParameterConfig,
    Field(discriminator="distribution"),
]
```

Because `values` does not have a `distribution` discriminator, implementation may
use a custom `model_validator` or separate fields instead of the exact union
above. The important behavior is that static choices and numeric distributions
are both represented explicitly.

### Search Strategy Configs

```python
class GridStrategyConfig(BaseConfig):
    type: Literal["grid"] = "grid"


class RandomStrategyConfig(BaseConfig):
    type: Literal["random"] = "random"
    num_trials: int
    seed: int | None = None


class OptunaStrategyConfig(BaseConfig):
    type: Literal["optuna"] = "optuna"
    num_trials: int
    seed: int | None = None
    storage: str | None = None
    sampler: Literal["tpe", "random"] = "tpe"
    pruner: Literal["none", "median", "asha", "hyperband"] = "none"


class WandbAgentStrategyConfig(BaseConfig):
    type: Literal["wandb_agent"] = "wandb_agent"
    project: str
    entity: str | None = None
    method: Literal["grid", "random", "bayes"] = "bayes"
    sweep_id: str | None = None
    count: int | None = None
```

### Scheduler Configs

```python
class LocalSchedulerConfig(BaseConfig):
    type: Literal["local"] = "local"
    max_parallel: int = 1
    gpu_assignment: LocalGpuAssignmentConfig | None = None


class SlurmSchedulerConfig(BaseConfig):
    type: Literal["slurm"] = "slurm"
    max_parallel: int = 1
    use_array: bool = False


class WandbAgentSchedulerConfig(BaseConfig):
    type: Literal["wandb_agent"] = "wandb_agent"


class MultiRunLoRASchedulerConfig(BaseConfig):
    type: Literal["multi_run_lora"] = "multi_run_lora"
    max_concurrent_runs: int
```

### Objective and Early Stopping Configs

```python
class ObjectiveConfig(BaseConfig):
    metric: str
    direction: Literal["maximize", "minimize"]
    source: Literal["wandb", "manifest", "metrics_server", "prime_monitor"] = "wandb"


class EarlyStoppingConfig(BaseConfig):
    type: Literal["threshold", "patience", "asha", "hyperband"]
    metric: str
    direction: Literal["maximize", "minimize"]
    min_steps: int = 0
    check_interval_seconds: int = 300
```

The exact early stopping fields should vary by type, but every variant needs a
minimum training budget and a metric source.

## Search Strategy Semantics

### Grid

Grid search expands all choice-valued parameters in deterministic insertion
order. Tests should pin ordering.

### Random

Random search samples `num_trials` independent assignments from the declared
distributions. Random search must record the seed and sampled values in
`manifest.json` so the study can be reproduced.

### Bayesian / Optuna

Optuna is the preferred in-process adaptive optimizer because it provides
samplers, pruners, persistent storage, and a clear trial API. The controller
asks Optuna for a trial, materializes a config, schedules the trial, reports
intermediate metrics, and marks the trial complete/pruned/failed.

Optuna support should be optional unless the project decides to add it as a hard
dependency. If optional, the config validator should emit a clear error when
`strategy.type = "optuna"` is requested without Optuna installed.

### Hyperband / ASHA

Hyperband/ASHA can be implemented through Optuna pruners or a native controller.
The first version should prefer Optuna pruners to avoid duplicating scheduling
logic. Native Hyperband can be considered only if PRIME-RL needs custom resource
semantics that Optuna cannot express cleanly.

### W&B Sweep Agent

W&B agent mode delegates search to W&B. PRIME-RL still owns config materialization
and target entrypoint execution:

1. Create or attach to a W&B sweep.
2. Run an agent command that receives W&B-proposed parameter values.
3. Convert W&B parameters into a PRIME-RL override TOML.
4. Launch the target `rl` or `sft` run.

This mode should not be the only adaptive strategy because it couples search to
an external service. It is valuable for teams already operating through W&B.

## Trial Materialization

All schedulers use the same trial layout where possible:

```text
outputs/studies/reverse-text-lr-temp/
  study.toml
  manifest.json
  controller.log
  trials/
    0000-a1b2c3d4/
      overrides.toml
      resolved.toml
      command.txt
      status.json
      metrics.jsonl
      run/
    0001-e5f6a7b8/
      overrides.toml
      resolved.toml
      command.txt
      status.json
      metrics.jsonl
      run/
```

Trial IDs are `<index>-<hash8>`, where `<index>` is the zero-padded enumeration
order and `<hash8>` is a stable 8-character hash of the trial's flat override
dict. The integer prefix preserves enumeration order; the hash makes IDs robust
to deletion and regeneration.

`study.toml` is the resolved sweep config.

`manifest.json` contains study metadata, trial metadata, commands, search
strategy state needed for reproducibility, and final outcomes. It also records:

- Resolved-config checksum per trial.
- Base file checksums (the files referenced by `base = [...]`).
- Git commit SHA at study creation time.
- Git dirty flag (uncommitted changes present at study creation).

`overrides.toml` contains generated overrides, including `output_dir`. This is
what the launcher composes into the actual command.

`resolved.toml` contains the fully resolved target config after applying base
files and overrides. It is written as a frozen single-file artifact for
reproducibility — the launcher does not run from it directly.

`command.txt` contains the exact command used for the trial, in the form
`uv run <rl|sft> @ <base.toml> @ ... @ <overrides.toml>`.

`status.json` records the current trial state:

```json
{
  "id": "0000-a1b2c3d4",
  "state": "completed",
  "pid": 12345,
  "slurm_job_id": null,
  "started_at": "2026-05-05T12:00:00Z",
  "finished_at": "2026-05-05T13:00:00Z",
  "objective": 0.73,
  "attempts": 1
}
```

`metrics.jsonl` is an optional normalized metric stream. It should be written by
the controller when it can observe metrics from W&B, the metrics server, Prime
Monitor, or trial output files.

## Dotted Path Overrides

Parameter keys are dotted target config paths. The sweep tool converts each
trial's dotted paths into a nested override TOML:

```toml
[parameters."trainer.optim.lr"]
values = [1e-5]
```

generates:

```toml
[trainer.optim]
lr = 1e-5
```

The target config system remains the source of truth. A bad path must fail by
validating the fully resolved target config before launch. Adaptive strategies
validate each new trial before scheduling it.

## Trial Output Directories

Every independent full-run trial must get a unique run output directory:

```text
{sweep.output_dir}/trials/{trial_id}/run
```

The generated override must set:

```toml
output_dir = "outputs/studies/<name>/trials/0000-a1b2c3d4/run"
```

This prevents collisions with the existing `validate_output_dir` behavior in the
`rl` and `sft` entrypoints.

Shared-trainer LoRA trials use the same study/trial metadata layout, but their
active orchestrator directories map to trainer-discovered `run_*` directories.
The trial manifest must record both the trial ID and the generated `run_*`
directory.

## Validation Flow

For each proposed trial:

1. Convert dotted parameter paths to a nested override dictionary.
2. Add generated fields such as `output_dir` and W&B metadata.
3. Write `overrides.toml`.
4. Resolve the target config using:
   - `RLConfig` for `entrypoint = "rl"`
   - `SFTConfig` for `entrypoint = "sft"`
5. Write `resolved.toml` and record its checksum in the manifest.
6. Write `command.txt` containing
   `uv run <rl|sft> @ <base.toml> @ ... @ <overrides.toml>`.
7. Mark the trial `pending` in `status.json`.

For static strategies, any validation failure should fail the whole study before
launching any trials. For adaptive strategies, a validation failure should mark
that proposed trial invalid and report failure to the search backend. If many
adaptive proposals fail validation, the controller should stop with a clear
configuration error.

Implementation detail: use the existing `prime_rl.utils.config.cli` wrapper with
explicit `args`, or call into `pydantic_config` through the same public wrapper
so config semantics stay identical to normal runs.

## Scheduling

### Local Scheduler

The local scheduler launches target entrypoints as subprocesses, composing the
base file and the generated overrides directly:

```bash
uv run rl @ examples/reverse_text/rl.toml @ outputs/studies/.../trials/0000-a1b2c3d4/overrides.toml
```

This matches the command form a user would write by hand, makes per-trial diffs
small (`overrides.toml` shows only what changed), and lets bug fixes to the
base file flow into resumed trials. Drift is detected by comparing the live
base file checksum against the value recorded in the manifest at materialization
time; if it changed, the launcher errors out unless the user explicitly opts
into the new content.

Behavior:

- `max_parallel = 1` (default) runs trials sequentially with no `gpu_assignment`
  required.
- `max_parallel > 1` is allowed when `[scheduler.gpu_assignment]` declares at
  least that many disjoint `visible_devices` groups (Phase 3). Each parallel
  worker holds one group for the lifetime of its subprocess, so two trials
  never share a GPU. Without `gpu_assignment` the validator rejects
  `max_parallel > 1` to avoid silently colocating trainer/inference stacks on
  the same devices.
- Failure handling is governed by the failure-policy fields documented under
  [Failure Semantics](#failure-semantics): `continue_on_failure` (default
  `true`) and `retry_budget` (default `1`).

### Local GPU Assignment

Local GPU assignment should be explicit:

```toml
[scheduler]
type = "local"
max_parallel = 2

[scheduler.gpu_assignment]
mode = "static"
visible_devices = [[0, 1], [2, 3]]
```

Modes:

- `static` (Phase 3, current default): assign declared `CUDA_VISIBLE_DEVICES`
  groups round-robin to parallel workers.
- `exclusive` (future): inspect available GPUs before launching and reserve a
  group for the subprocess.
- `none` (future): do not set `CUDA_VISIBLE_DEVICES`; fall back to whatever
  the parent environment exposes.

The scheduler writes the assigned devices to `status.json` (`gpu_group`) and
sets `CUDA_VISIBLE_DEVICES` on the subprocess environment so retries and
post-mortems can see which devices ran which trial.

### SLURM Scheduler

The SLURM scheduler submits one job per trial by invoking the normal target
entrypoint on the resolved trial config. If the target config contains `[slurm]`,
the existing `rl` or `sft` entrypoint renders and submits its own `sbatch`
script.

SLURM arrays are a later optimization:

- One array job can reduce scheduler pressure for large static studies.
- Arrays are harder for adaptive strategies because future trials are not known
  up front.
- Array support should be limited to static grid/random studies unless the
  controller has a persistent worker model.

### W&B Agent Scheduler

The W&B agent scheduler starts `wandb agent` or an equivalent SDK-driven agent.
Each W&B-assigned config becomes a PRIME-RL trial. The trial still gets a local
`trials/<id>` directory, generated TOML files, and `status.json`.

### Shared-Trainer LoRA Scheduler

The `multi_run_lora` scheduler launches one trainer/inference stack and creates
multiple orchestrator run directories under that trainer output directory.

Constraints:

- `entrypoint = "rl"` only.
- Trainer config must enable LoRA.
- Trainer `max_concurrent_runs` must be set to the scheduler concurrency.
- Sweepable fields must be restricted to per-run orchestrator-safe fields:
  - `orchestrator.optim.*`
  - `orchestrator.model.lora.*` subject to trainer max rank validation
  - orchestrator sampling fields
  - orchestrator environment fields that do not require trainer changes
  - orchestrator batch/buffer/eval fields
- Trainer/model/deployment/inference fields cannot vary inside one shared
  trainer study.

This mode should have its own validation allowlist. A user should get a clear
error when attempting to sweep unsupported paths.

## Metrics and Objectives

The controller needs normalized access to trial metrics. The codebase already
runs `MultiMonitor` over W&B and Prime Monitor and writes a `final_summary.json`
to every run directory. The sweep tool layers on top of these without inventing
a new transport:

- **Final-objective sweeps** (grid, random, and any study where only the final
  value matters) read `<run_dir>/final_summary.json`. This is canonical because
  it is written unconditionally, requires no auth or network, and contains the
  same numbers W&B saw at the end of the run.
- **Intermediate-metric pruning** (Phase 4 early stopping, Phase 5 ASHA/
  Hyperband) reads W&B step-indexed history, falling back to Prime Monitor when
  W&B is disabled.
- **Trainer-level status** (TPS, MFU, queue depth) is available through the
  existing Prometheus metrics server but is not authoritative for objective
  values — it describes throughput, not training quality.

The objective config names one metric and direction. The controller records:

- latest observed objective value
- best observed objective value
- step associated with best value
- source and timestamp

The design does not prescribe metric names; the user config does. The default
metric names used in examples follow the conventions already published by the
codebase:

- RL eval reward: `eval/<env_name>/avg@<rollouts_per_example>` (see
  `orchestrator/envs.py`).
- SFT validation loss: `val/loss` (see `trainer/sft/train.py`).

## Early Stopping and Pruning

Early stopping can operate in two ways:

- **Independent run stopping**: terminate a local process or cancel a SLURM job.
- **Shared-trainer pruning**: evict or stop a specific `run_*` orchestrator in a
  `multi_run_lora` study.

For independent full runs, the controller can stop work without trainer changes.
For shared-trainer pruning, the implementation should use existing run eviction
where possible. If finer-grained behavior is needed, add an explicit control file
or API rather than relying on process signals.

Early stopping decisions must be written to `status.json` and the manifest with
the metric value and rule that triggered the stop.

## W&B Behavior

For controller-owned W&B tracking:

- Set target run `wandb.group` to the study name unless overridden.
- Set target run `wandb.name` to the trial label or ID.
- Append tags such as `sweep`, `study:<name>`, and `trial:<id>`.

For W&B agent mode:

- W&B owns trial assignment.
- PRIME-RL still writes trial artifacts and launches ordinary target configs.
- The W&B run ID should be recorded in `status.json` and `manifest.json`.

The design should avoid requiring W&B for non-W&B strategies.

## Code Points to Edit

### New Files

- `src/prime_rl/configs/sweep.py`
  - Defines study config, strategy configs, scheduler configs, parameter configs,
    objective configs, W&B config, and early stopping configs.

- `src/prime_rl/entrypoints/sweep.py`
  - CLI entrypoint.
  - Loads `SweepConfig`.
  - Creates/resumes the controller.

- `src/prime_rl/sweep/`
  - New package for implementation code.

- `src/prime_rl/sweep/controller.py`
  - Study controller.
  - Owns trial lifecycle, manifest updates, resume, and failure semantics.

- `src/prime_rl/sweep/search.py`
  - Grid/random strategy implementations.
  - Optuna adapter when available.
  - W&B agent adapter when implemented.

- `src/prime_rl/sweep/materialize.py`
  - Dotted path conversion.
  - Override TOML writing.
  - Resolved config writing and validation.

- `src/prime_rl/sweep/schedulers.py`
  - Local, SLURM, W&B agent, and multi-run LoRA schedulers.

- `src/prime_rl/sweep/metrics.py`
  - Metric readers for W&B, manifest/JSONL, metrics server, and Prime Monitor.

- `src/prime_rl/sweep/early_stopping.py`
  - Early stopping and pruning policies.

- Tests:
  - `tests/unit/sweep/test_config.py`
  - `tests/unit/sweep/test_expansion.py`
  - `tests/unit/sweep/test_materialize.py`
  - `tests/unit/sweep/test_controller.py`
  - `tests/unit/sweep/test_schedulers.py`
  - `tests/unit/sweep/test_metrics.py`
  - `tests/unit/sweep/test_early_stopping.py`

- Optional examples:
  - `configs/debug/sweep/rl_grid.toml`
  - `configs/debug/sweep/rl_random.toml`
  - `examples/reverse_text/sweep.toml`

### Existing Files

- `pyproject.toml`
  - Add:
    ```toml
    sweep = "prime_rl.entrypoints.sweep:main"
    ```
  - Later phases may add optional dependencies for Optuna. If Optuna is optional,
    prefer an extra such as:
    ```toml
    [project.optional-dependencies]
    hpo = ["optuna>=4"]
    ```

- `src/prime_rl/configs/__init__.py`
  - Export `SweepConfig` if this package convention is used for other configs.

- `src/prime_rl/entrypoints/rl.py`
  - No change for independent full-run studies.
  - Later phases may need small hooks for shared-trainer launcher integration if
    the scheduler cannot reuse existing entrypoint behavior cleanly.

- `src/prime_rl/trainer/runs.py`
  - Later shared-trainer LoRA phase may need helper APIs around run discovery,
    run metadata, or controlled pruning. Prefer additive helpers over changing
    existing discovery semantics.

- `src/prime_rl/utils/monitor/wandb.py`
  - W&B agent and richer W&B grouping may need run ID propagation or summary
    metric conventions.

- `docs/entrypoints.md`
  - Add `sweep` once implemented.

- `docs/sweeps.md`
  - User-facing public docs page (created when the feature is ready for users).
  - The build-time spec under `docs/hyperparam-optim/` stays for design
    history; `docs/sweeps.md` is the page agents and users land on.

- `docs/configs.md`
  - Add sweep dotted-path and generated override docs once implemented.

- `docs/mint.json`
  - Add the public sweep docs page once the feature is ready for users.

- `skills/config/SKILL.md`
  - Update after implementation so agents know `uv run sweep` exists and how it
    composes configs.

- `skills/entrypoints/SKILL.md`
  - Update after implementation with the new entrypoint and examples.

## Failure Semantics

Failure handling is governed by two orthogonal knobs on `SweepConfig`:

- `continue_on_failure: bool = True` — when a trial reaches its final failed
  state, schedule remaining trials anyway. A single broken config should not
  halt exploration of the rest of the space. Set `false` to halt-on-first-fail
  for CI / smoke runs.
- `retry_budget: int = 1` — retry a failed trial up to this many times before
  marking it failed. Catches transient failures (CUDA OOM at startup, W&B
  network blips, SLURM submission hiccups) without silently retrying
  configuration bugs. Validation failures are deterministic and are never
  retried.

Trial state machine: `pending → running → (completed | failed | pruned |
stopped)`. A retry transitions back to `running` and increments `attempts` in
`status.json`. Validation failures go directly to `failed` with no retry.

Other rules:

- Sweep config validation errors fail immediately.
- Static trial validation errors fail the whole study before launching any
  trials.
- Adaptive trial validation errors mark the proposed trial invalid and report
  failure to the search backend.
- Early-stopped trials are marked `pruned` or `stopped`, not `failed`.
- Already submitted SLURM jobs are not cancelled automatically unless early
  stopping or failure policy explicitly requests cancellation.

## Resume Semantics

`uv run sweep @ study.toml --resume` should:

- Load the existing `manifest.json`.
- Reconstruct completed, failed, running, and pending trial state.
- Avoid regenerating different trial IDs for existing trials.
- Reattach to observable local/SLURM/W&B jobs where possible.
- For static studies, schedule remaining pending trials.
- For adaptive studies, restore the search backend state before proposing new
  trials.

## Roadmap

### Phase 1: Static Full-Run Sweeps

Implement the foundation:

- `sweep` entrypoint.
- `SweepConfig` with grid strategy.
- Local scheduler with sequential execution. `max_parallel > 1` is rejected
  with a configuration error until Phase 3.
- SLURM scheduler with one job per trial.
- Dotted path override materialization.
- Hybrid launch command: `uv run <rl|sft> @ <base.toml> @ <overrides.toml>`.
- Trial directories with hashed IDs (`<index>-<hash8>`), `study.toml`,
  `manifest.json` (incl. resolved-config + base file checksums and git commit
  metadata), `status.json`, `overrides.toml`, `resolved.toml`, and
  `command.txt`.
- Dry-run mode.
- W&B grouping/name/tag injection (always injected when `wandb.enabled`;
  matches existing W&B library convention).
- Unit tests for config, expansion, materialization, and mocked scheduling.

Phase 1 intentionally does not include random search, adaptive optimization,
W&B agents, local GPU assignment, shared-trainer LoRA sweeps, or early
stopping.

### Phase 2: Failure Policy, Random Search, Resume

Add:

- `continue_on_failure: bool = True` and `retry_budget: int = 1` knobs with
  state-machine support in the controller.
- Random strategy with uniform/log-uniform/int distributions.
- Seeded reproducibility.
- Study resume.

### Phase 3: Local Resource Assignment

Add:

- Explicit local GPU assignment via `[scheduler.gpu_assignment]`.
- Static `CUDA_VISIBLE_DEVICES` groups (`mode = "static"`); `exclusive` mode
  (live GPU discovery) and `none` mode are deferred to a later phase.
- Parallel local scheduler (`ThreadPoolExecutor` over a queue of GPU groups)
  with resource ownership recorded in `status.json` as `gpu_group`.
- Lift the Phase-1 prohibition on `max_parallel > 1`; the validator now
  rejects `max_parallel > 1` only when `gpu_assignment` is missing or has
  fewer groups than `max_parallel`.

### Phase 4: Metrics and Early Stopping

Add:

- Metric reader for `final_summary.json` (final-objective sweeps); W&B
  step-indexed history and Prime Monitor fallback are deferred to a later
  phase (they slot in alongside Optuna/ASHA which need intermediate values).
- Objective tracking: `ObjectiveConfig` records metric + direction, the
  controller writes the per-trial value into `status.json` and a best-trial
  summary into `manifest.json`.
- Threshold and patience early stopping at trial-completion granularity:
  after each trial finishes the tracker decides whether to halt new
  submissions; in-flight trials finish naturally.
- In-flight termination of running trials and SLURM cancellation are still
  open and slot in alongside intermediate-metric pruning when ASHA/Hyperband
  arrive in Phase 5+.

### Phase 5: Optuna and Hyperband/ASHA

Phase 5a (this branch):

- Optional Optuna integration via the `prime-rl[hpo]` extra.
- `OptunaStrategyConfig` with TPE / Random samplers, optional SQLAlchemy
  storage URL, `seed`, and `study_name`.
- Sequential ask/tell driver: the controller asks Optuna for one parameter
  set at a time, materializes a trial, runs it through the existing
  scheduler primitives, and tells Optuna the final objective before the
  next ask. Failures and `None` objectives report `TrialState.FAIL`.
- Resume requires `strategy.storage` (in-memory studies vanish on exit).
  Optuna is rejected with the SLURM scheduler since the controller must
  observe each trial before proposing the next.

Deferred to Phase 5b:

- Median / ASHA / Hyperband pruners through Optuna (currently ``pruner =
  "none"`` is the only accepted value).
- Intermediate metric reporting from the controller (W&B step-indexed
  history / Prime Monitor fallback).

### Phase 6: W&B Sweep Agent Integration

Add:

- W&B sweep creation/attachment.
- W&B agent scheduler.
- Conversion from W&B-assigned parameters to PRIME-RL override TOML.
- Trial artifact writing for W&B-managed trials.

### Phase 7: Shared-Trainer LoRA Sweeps

Initial implementation is LoRA-only because that is the only trainer
architecture that supports multiple per-run weight sets in memory at once. The
scheduler and validator are designed against a generic "per-run parameter set"
interface so future trainer architectures can plug in without rewriting the
scheduler.

Add:

- `multi_run_lora` scheduler.
- Allowlist validation for per-run safe parameters (swappable strategy, not
  hard-coded LoRA fields).
- Generation of `run_* / control/orch.toml` directories.
- Shared trainer/inference launcher behavior.
- Run-level pruning through existing eviction semantics where possible.

### Phase 8: SLURM Arrays

Optimize SLURM submission for large static studies:

- Single array job per static grid/random study.
- Manifest mapping from array task index to trial ID.
- Skipped for adaptive strategies because future trials are not known up front.

## Decisions

The questions originally listed in this section have been resolved. Recording
the resolutions here so the rationale is visible to future work:

- **Launch form** — hybrid. The launcher composes base + overrides on the
  command line so the trial command matches user-authored runs and base-file
  fixes flow into resumed trials. `resolved.toml` is still written as a
  reproducible single-file artifact; the manifest records resolved-config and
  base-file checksums to detect drift.
- **Local `max_parallel > 1`** — supported as of Phase 3 when paired with
  `[scheduler.gpu_assignment] mode = "static"` declaring at least
  `max_parallel` disjoint `visible_devices` groups. The validator still
  rejects `max_parallel > 1` without `gpu_assignment` so two parallel
  workers cannot silently colocate on the same GPUs. `exclusive` and
  `none` modes are still future work.
- **Optuna packaging** — `prime-rl[hpo]` extra. The validator emits a clear
  install hint when `strategy.type = "optuna"` is requested without the extra.
- **Canonical metric source** — `final_summary.json` for final-objective
  sweeps, W&B step-indexed history for intermediate-metric pruning, Prime
  Monitor as fallback when W&B is disabled. The Prometheus metrics server is
  not authoritative for objective values.
- **Example metric names** — `eval/<env_name>/avg@<rollouts_per_example>` for
  RL eval reward and `val/loss` for SFT validation loss. Both follow the
  conventions already published by `orchestrator/envs.py` and
  `trainer/sft/train.py`.
- **Failure policy** — `continue_on_failure: bool = True` and
  `retry_budget: int = 1` are first-class config fields. Default is to keep
  exploring and to retry transient failures once.
- **Trial IDs** — `<index>-<hash8>`. Integer prefix preserves enumeration
  order; the hash makes IDs stable under deletion and regeneration.
- **W&B injection** — always injected when `wandb.enabled`. Matches the
  library convention that `[wandb]` blocks are overrides over defaults rather
  than gates on logging.
- **Manifest reproducibility** — record resolved-config checksum, base file
  checksums, git commit SHA, and a git dirty flag at study creation time.
- **SFT vs RL trial layout** — differ where the launchers already differ. RL
  trials reserve space for the split `configs/{trainer,orchestrator,inference}.toml`
  the launcher writes at runtime; SFT trials do not.
- **SLURM arrays** — own phase (Phase 8). Deferred past adaptive support.
- **Shared-trainer LoRA scope** — LoRA-only initial implementation. The
  scheduler and validator are designed against a generic per-run parameter set
  interface so future trainer architectures can plug in.
- **Active-run mutation** — dropped. The original Phase 8 design has been
  removed. There is no active-run control API and no controller-side mutation;
  trainer and orchestrator stay focused on training. If a concrete need
  appears later it can come back as a scoped, orchestrator-only feature.
- **Docs location** — `docs/hyperparam-optim/` is build-time / design history.
  The user-facing page lives at `docs/sweeps.md` and is created when the
  feature is ready for users.
