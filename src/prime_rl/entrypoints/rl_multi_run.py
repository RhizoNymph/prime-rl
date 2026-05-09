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

import argparse
import os
import signal
import sys
import time
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
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.monitor import SWEEP_METRICS_JSONL_ENV
from prime_rl.utils.process import cleanup_processes, cleanup_threads, set_proc_title
from prime_rl.utils.utils import get_log_dir

RUNS_DIR_FLAG = "--runs-dir"
WATCH_SLOTS_FLAG = "--watch-slots"
DONE_MARKER_NAME = "done"
SLOT_POLL_INTERVAL_SECONDS = 2.0


def _parse_runs_dirs(argv: list[str]) -> tuple[list[Path], bool, list[str]]:
    """Peel ``--runs-dir`` and ``--watch-slots`` off argv before pydantic_config.

    Returns ``(run_dirs, watch_slots, remaining_argv)``. The remaining argv
    is passed to ``cli(RLConfig)`` so the standard ``@ shared.toml`` syntax
    keeps working. ``--watch-slots`` is the Phase 7c continuous-flow toggle:
    when set, the launcher keeps watching the parent of ``run_dirs`` for new
    ``run_*/control/orch.toml`` files and spawns orchestrators on demand.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(RUNS_DIR_FLAG, required=True)
    parser.add_argument(WATCH_SLOTS_FLAG, action="store_true")
    namespace, remaining = parser.parse_known_args(argv)
    raw = namespace.runs_dir
    if not raw:
        raise SystemExit(f"{RUNS_DIR_FLAG} must list at least one run directory")
    run_dirs = [Path(piece).resolve() for piece in raw.split(":") if piece]
    if not run_dirs:
        raise SystemExit(f"{RUNS_DIR_FLAG} parsed to no run directories: {raw!r}")
    return run_dirs, bool(namespace.watch_slots), remaining


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
LAUNCHER_PID_FILENAME = ".launcher.pid"
LAUNCHER_HEARTBEAT_FILENAME = ".launcher.heartbeat"


def _write_launcher_pid(shared_dir: Path) -> None:
    """Record the launcher's PID so resume can detect a still-running launcher.

    Phase 7e: when ``rl-multi-run --watch-slots`` is alive between sweeps the
    new controller can attach via the file protocol (drop run_*/control/orch.toml,
    let the watch-slots loop spawn the orchestrators) instead of re-launching
    trainer + inference. The PID file is paired with a heartbeat that the
    controller checks for freshness.
    """
    shared_dir.mkdir(parents=True, exist_ok=True)
    (shared_dir / LAUNCHER_PID_FILENAME).write_text(f"{os.getpid()}\n")


def _touch_launcher_heartbeat(shared_dir: Path) -> None:
    """Refresh the heartbeat file's mtime — cheap proof of life on each tick."""
    path = shared_dir / LAUNCHER_HEARTBEAT_FILENAME
    path.touch()


def _cleanup_launcher_pid_files(shared_dir: Path) -> None:
    """Remove PID and heartbeat files on orderly exit.

    Resume's freshness check rejects stale files, so leaving them behind is
    self-correcting — the heartbeat will look stale and the controller will
    fall back to stop+resume. But cleanup makes diagnosis easier and matches
    the intent of "this launcher is no longer running".
    """
    for filename in (LAUNCHER_PID_FILENAME, LAUNCHER_HEARTBEAT_FILENAME):
        path = shared_dir / filename
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue


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


def _watch_slots_loop(
    *,
    shared_dir: Path,
    log_dir: Path,
    start_command: list[str],
    wandb_shared_env: dict[str, str],
    supervisor: LaunchSupervisor,
    trainer_process,
    orchestrator_processes: list,
    orchestrator_labels: list[str],
    run_dirs: list[Path],
) -> None:
    """Watch for new ``run_*/control/orch.toml`` files and spawn orchestrators.

    Runs after the initial orchestrator + trainer spawn. Mutates the
    caller-owned lists in place so existing teardown / exit-code recording
    continues to work without changes. Exits when:

    1. The trainer exits — natural shutdown, no more progress possible.
    2. ``<shared_dir>/control/done`` exists *and* every tracked orchestrator
       has exited. The sweep controller writes ``done`` when it has no more
       trials to schedule, so the launcher can drain remaining orchestrators
       and tear down the trainer instead of waiting forever for new work.

    Per-orchestrator exit codes are written individually as each orchestrator
    exits, so the sweep controller can observe each completion in real time
    (continuous-flow needs that to know when to ask Optuna for a replacement).
    The ``_record_orchestrator_exit_codes`` batch call on shutdown re-writes
    them, which is harmless — the file content is identical.
    """
    # Lazy import to keep this helper testable without sweep dependencies.
    from prime_rl.utils.monitor import SWEEP_METRICS_JSONL_ENV

    seen_run_ids: set[str] = {d.name for d in run_dirs}
    finished_run_ids: set[str] = set()
    done_marker = shared_dir / "control" / DONE_MARKER_NAME

    _write_launcher_pid(shared_dir)

    try:
        _run_watch_slots_inner(
            shared_dir=shared_dir,
            done_marker=done_marker,
            log_dir=log_dir,
            start_command=start_command,
            wandb_shared_env=wandb_shared_env,
            supervisor=supervisor,
            trainer_process=trainer_process,
            orchestrator_processes=orchestrator_processes,
            orchestrator_labels=orchestrator_labels,
            run_dirs=run_dirs,
            seen_run_ids=seen_run_ids,
            finished_run_ids=finished_run_ids,
            sweep_env_var=SWEEP_METRICS_JSONL_ENV,
        )
    finally:
        _cleanup_launcher_pid_files(shared_dir)


def _run_watch_slots_inner(
    *,
    shared_dir: Path,
    done_marker: Path,
    log_dir: Path,
    start_command: list[str],
    wandb_shared_env: dict[str, str],
    supervisor: LaunchSupervisor,
    trainer_process,
    orchestrator_processes: list,
    orchestrator_labels: list[str],
    run_dirs: list[Path],
    seen_run_ids: set[str],
    finished_run_ids: set[str],
    sweep_env_var: str,
) -> None:
    """Body of the watch-slots loop, factored out so the outer wrapper can
    own PID/heartbeat lifecycle via try/finally without an extra indent."""
    while True:
        _touch_launcher_heartbeat(shared_dir)
        # 1. Reap finished orchestrators (write per-run exit_code).
        for proc, run_dir in zip(orchestrator_processes, run_dirs):
            if run_dir.name in finished_run_ids:
                continue
            if proc.poll() is None:
                continue
            try:
                _write_orchestrator_exit_code(run_dir, proc.returncode)
            except OSError:
                pass
            finished_run_ids.add(run_dir.name)

        # 2. Discover new run_* directories with control/orch.toml.
        for new_run_dir in sorted(shared_dir.glob("run_*")):
            if new_run_dir.name in seen_run_ids:
                continue
            orch_config = new_run_dir / "control" / "orch.toml"
            if not orch_config.exists():
                continue
            label = f"orchestrator-{new_run_dir.name}"
            new_proc = start_orchestrator(
                config_path=orch_config,
                label=label,
                log_path=log_dir / f"{label}.log",
                start_command=start_command,
                wandb_shared_env=wandb_shared_env,
                wandb_program="uv run rl-multi-run",
                supervisor=supervisor,
                extra_env={sweep_env_var: (new_run_dir / "metrics.jsonl").as_posix()},
            )
            orchestrator_processes.append(new_proc)
            orchestrator_labels.append(label)
            run_dirs.append(new_run_dir)
            seen_run_ids.add(new_run_dir.name)

        # 3. Surface monitor-thread errors as before.
        if supervisor.error_queue:
            return

        # 4. Exit conditions.
        if trainer_process.poll() is not None:
            return
        if done_marker.exists() and all(p.poll() is not None for p in orchestrator_processes):
            return

        time.sleep(SLOT_POLL_INTERVAL_SECONDS)


def rl_multi_run(config: RLConfig, run_dirs: list[Path], watch_slots: bool = False) -> None:
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

        if watch_slots:
            # Phase 7c continuous-flow: stay alive while the controller drops
            # new run_*/control/orch.toml files into the shared dir, spawning
            # orchestrators on demand. The trainer's MultiRunManager picks up
            # the same directories on its own discovery cycle.
            shared_dir = run_dirs[0].parent
            _watch_slots_loop(
                shared_dir=shared_dir,
                log_dir=log_dir,
                start_command=start_command,
                wandb_shared_env=wandb_shared_env,
                supervisor=supervisor,
                trainer_process=trainer_process,
                orchestrator_processes=orchestrator_processes,
                orchestrator_labels=orchestrator_labels,
                run_dirs=run_dirs,
            )
        else:
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
    run_dirs, watch_slots, remaining = _parse_runs_dirs(sys.argv[1:])
    config = cli(RLConfig, args=remaining)
    rl_multi_run(config, run_dirs, watch_slots=watch_slots)


if __name__ == "__main__":
    main()
