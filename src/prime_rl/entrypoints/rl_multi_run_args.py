import argparse
from pathlib import Path

RUNS_DIR_FLAG = "--runs-dir"


def parse_runs_dirs(argv: list[str]) -> tuple[list[Path], list[str]]:
    """Peel ``--runs-dir <colon-separated paths>`` off argv before pydantic_config.

    Returns ``(run_dirs, remaining_argv)``. The remaining argv is passed to
    ``cli(RLConfig)`` so the standard ``@ shared.toml`` syntax keeps working.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(RUNS_DIR_FLAG, required=True)
    namespace, remaining = parser.parse_known_args(argv)
    raw = namespace.runs_dir
    if not raw:
        raise SystemExit(f"{RUNS_DIR_FLAG} must list at least one run directory")
    run_dirs = [Path(piece).resolve() for piece in raw.split(":") if piece]
    if not run_dirs:
        raise SystemExit(f"{RUNS_DIR_FLAG} parsed to no run directories: {raw!r}")
    return run_dirs, remaining
