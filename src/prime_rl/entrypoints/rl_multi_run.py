"""Shared-trainer LoRA launcher.

Boots one trainer + one inference server + N orchestrators against a shared
output directory, one orchestrator per pre-materialized ``run_*`` directory.
The trainer's ``MultiRunManager`` discovers the run dirs, allocates per-run
LoRA adapter slots, and routes each orchestrator's training samples to the
right adapter.

The sweep controller is the expected caller: it materializes
``<output_dir>/run_<trial_id>/control/orch.toml`` for every trial up front
and then invokes::

    rl-multi-run @ shared.toml --runs-dir run_a:run_b:run_c

``shared.toml`` is an ordinary ``RLConfig`` whose trainer block has
``max_concurrent_runs = N`` and whose orchestrator block carries the shared
defaults trial overrides inherit from. The orchestrator block in the shared
file is *not* used to launch a process here; only the per-run ``orch.toml``
files referenced via ``--runs-dir`` produce orchestrators.
"""

import os
import signal
import sys
from pathlib import Path

from prime_rl.configs.rl import RLConfig
from prime_rl.entrypoints.launch import (
    LaunchSupervisor,
    build_wandb_shared_env,
    compute_gpu_mapping,
    start_inference,
    start_orchestrator,
    start_trainer,
    tail_trainer_log,
    wait_for_completion,
)
from prime_rl.entrypoints.rl import (
    INFERENCE_TOML,
    TEACHER_INFERENCE_TOML,
    TRAINER_TOML,
    check_gpus_available,
    get_physical_gpu_ids,
    write_subconfigs,
)
from prime_rl.entrypoints.rl_multi_run_args import RUNS_DIR_FLAG, parse_runs_dirs
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.monitor import SWEEP_METRICS_JSONL_ENV
from prime_rl.utils.process import cleanup_processes, cleanup_threads, set_proc_title
from prime_rl.utils.utils import get_log_dir


def _validate_run_layout(run_dirs: list[Path]) -> None:
    """Each run dir must already contain ``control/orch.toml``.

    The trainer's ``MultiRunManager`` will reject a run whose config file is
    missing; failing loudly here gives a much better error than waiting for
    the trainer to silently skip the run.
    """
    missing = [d for d in run_dirs if not (d / "control" / "orch.toml").exists()]
    if missing:
        raise SystemExit(
            f"{RUNS_DIR_FLAG} entries missing control/orch.toml: {[d.as_posix() for d in missing]}. "
            "The sweep launcher must pre-materialize each run before invoking rl-multi-run."
        )


def _validate_concurrency(config: RLConfig, run_dirs: list[Path]) -> None:
    """Trainer must be sized for at least len(run_dirs) concurrent runs."""
    max_runs = getattr(config.trainer, "max_concurrent_runs", None)
    if max_runs is None or max_runs < 1:
        raise SystemExit(
            "rl-multi-run requires trainer.max_concurrent_runs >= 1 in the shared config "
            "(use multi-run-LoRA training)."
        )
    if max_runs < len(run_dirs):
        raise SystemExit(
            f"trainer.max_concurrent_runs={max_runs} but {len(run_dirs)} run dirs were passed; "
            "set max_concurrent_runs to at least the number of concurrent trials."
        )


EXIT_CODE_FILENAME = "exit_code"


def _write_orchestrator_exit_code(run_dir: Path, returncode: int | None) -> None:
    """Write a per-orchestrator returncode for the sweep controller to reconcile.

    The sweep controller reads each ``<run_dir>/control/exit_code`` after the
    multi-run invocation exits, so it can attribute failures to the actual
    orchestrator that crashed instead of marking every trial in the wave
    failed (the Phase 7a behavior). ``None`` means "the launcher tore down the
    orchestrator before it produced an exit code"; we record ``-1`` so the
    controller treats it as an infrastructure failure.
    """
    control_dir = run_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    code = -1 if returncode is None else int(returncode)
    (control_dir / EXIT_CODE_FILENAME).write_text(f"{code}\n")


def _record_orchestrator_exit_codes(
    orchestrator_processes, run_dirs: list[Path]
) -> None:
    """Best-effort: write exit_code for every run dir, swallowing per-run write errors.

    A failure to write one exit_code must not prevent the others from being
    recorded — the controller falls back to "infrastructure failure" when the
    file is missing, which is at least diagnosable.
    """
    for proc, run_dir in zip(orchestrator_processes, run_dirs):
        try:
            _write_orchestrator_exit_code(run_dir, proc.returncode)
        except OSError:
            continue


def rl_multi_run(config: RLConfig, run_dirs: list[Path]) -> None:
    assert config.deployment.type == "single_node", "rl-multi-run is single-node only"
    _validate_concurrency(config, run_dirs)
    _validate_run_layout(run_dirs)

    logger = setup_logger(
        config.log.level or os.environ.get("PRIME_LOG_LEVEL", "info"),
        json_logging=config.log.json_logging,
    )

    config_dir = config.output_dir / "configs"
    write_subconfigs(config, config_dir)
    logger.info(f"Wrote subconfigs to {config_dir}")

    if config.dry_run:
        logger.success(
            "Dry run complete. To start a multi-run RL launch, remove --dry-run from your command."
        )
        return

    mapping = compute_gpu_mapping(config, get_physical_gpu_ids)
    logger.info(f"Using local->physical GPU mapping: {mapping.physical}")

    start_command = sys.argv
    logger.info(f"Starting multi-run RL launch with {len(run_dirs)} orchestrator(s)")
    logger.debug(f"Multi-run RL start command: {' '.join(start_command)}")

    wandb_shared_env = build_wandb_shared_env(config)

    all_gpu_ids = list(set(mapping.infer + mapping.trainer + mapping.teacher))
    check_gpus_available(all_gpu_ids)

    log_dir = get_log_dir(config.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    supervisor = LaunchSupervisor(logger=logger, log_dir=log_dir)

    orchestrator_labels: list[str] = []
    orchestrator_processes = []

    def sigterm_handler(signum, frame):
        logger.warning("Received SIGTERM, terminating all processes...")
        cleanup_threads(supervisor.monitor_threads)
        cleanup_processes(supervisor.processes)
        _record_orchestrator_exit_codes(orchestrator_processes, run_dirs)
        sys.exit(1)

    signal.signal(signal.SIGTERM, sigterm_handler)

    try:
        if config.inference:
            start_inference(
                cmd=["inference", "@", (config_dir / INFERENCE_TOML).as_posix()],
                gpu_ids=mapping.infer,
                label="inference",
                log_path=log_dir / "inference.log",
                supervisor=supervisor,
            )
        else:
            logger.warning(
                "No inference config specified, skipping starting inference server. "
                "Make sure your inference server is running."
            )

        if config.teacher_inference:
            if not mapping.teacher:
                raise ValueError(
                    "teacher_inference is configured but deployment.num_teacher_gpus is not set."
                )
            start_inference(
                cmd=["inference", "@", (config_dir / TEACHER_INFERENCE_TOML).as_posix()],
                gpu_ids=mapping.teacher,
                label="teacher_inference",
                log_path=log_dir / "teacher_inference.log",
                supervisor=supervisor,
            )

        for run_dir in run_dirs:
            run_id = run_dir.name
            label = f"orchestrator-{run_id}"
            orchestrator_labels.append(label)
            orchestrator_processes.append(
                start_orchestrator(
                    config_path=run_dir / "control" / "orch.toml",
                    label=label,
                    log_path=log_dir / f"orchestrator-{run_id}.log",
                    start_command=start_command,
                    wandb_shared_env=wandb_shared_env,
                    wandb_program="uv run rl-multi-run",
                    supervisor=supervisor,
                    # Per-run sweep sidecar metrics, so the controller can
                    # read this trial's objective from <run_dir>/metrics.jsonl
                    # without colliding with sibling orchestrators.
                    extra_env={SWEEP_METRICS_JSONL_ENV: (run_dir / "metrics.jsonl").as_posix()},
                )
            )

        trainer_process = start_trainer(
            config_path=config_dir / TRAINER_TOML,
            gpu_ids=mapping.trainer,
            ranks_filter=config.trainer.log.ranks_filter,
            log_path=log_dir / "trainer.log",
            torchrun_log_dir=log_dir / "trainer" / "torchrun",
            start_command=start_command,
            wandb_shared_env=wandb_shared_env,
            wandb_program="uv run rl-multi-run",
            supervisor=supervisor,
        )

        logger.success("Startup complete. Showing trainer logs...")
        tail_trainer_log(supervisor, log_dir / "trainer.log")

        # Trainer winding down implies all orchestrators completed; the
        # supervisor still requires every orchestrator's stop_event to fire,
        # which they do as their subprocesses exit.
        wait_for_completion(orchestrator_labels + ["trainer"], supervisor)

        # Per-orchestrator exit_code is the sweep controller's source of
        # truth for failure attribution; write it as soon as we've waited
        # for every orchestrator, before any cleanup that might mask codes.
        _record_orchestrator_exit_codes(orchestrator_processes, run_dirs)

        failed_orchestrators = [
            (label, proc.returncode)
            for label, proc in zip(orchestrator_labels, orchestrator_processes)
            if proc.returncode != 0
        ]
        if failed_orchestrators:
            for label, code in failed_orchestrators:
                logger.error(f"{label} failed with exit code {code}")
            cleanup_threads(supervisor.monitor_threads)
            cleanup_processes(supervisor.processes)
            sys.exit(1)

        if trainer_process.returncode != 0:
            logger.error(f"Trainer failed with exit code {trainer_process.returncode}")
            cleanup_threads(supervisor.monitor_threads)
            cleanup_processes(supervisor.processes)
            sys.exit(1)

        logger.success("Multi-run RL training finished!")
        cleanup_threads(supervisor.monitor_threads)
        cleanup_processes(supervisor.processes)

    except KeyboardInterrupt:
        logger.warning("Received interrupt signal, terminating all processes...")
        cleanup_threads(supervisor.monitor_threads)
        cleanup_processes(supervisor.processes)
        _record_orchestrator_exit_codes(orchestrator_processes, run_dirs)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        cleanup_threads(supervisor.monitor_threads)
        cleanup_processes(supervisor.processes)
        _record_orchestrator_exit_codes(orchestrator_processes, run_dirs)
        raise


def main():
    set_proc_title("MultiRunLauncher")
    run_dirs, remaining = parse_runs_dirs(sys.argv[1:])
    config = cli(RLConfig, args=remaining)
    rl_multi_run(config, run_dirs)


if __name__ == "__main__":
    main()
