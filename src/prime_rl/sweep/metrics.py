"""Metric readers for sweep trials.

Phase 4 reads the final value of an objective from each trial's
``final_summary.json`` artifact. The W&B monitor writes that file to
``<run_dir>/run-<wandb_id>/final_summary.json`` (mirroring its own run-id
namespace), so the reader globs for it instead of assuming the wandb run id.
W&B step-indexed history and Prime Monitor fallback land alongside adaptive
strategies in later phases.
"""

import json
import math
from pathlib import Path
from typing import Any


def _final_summary_paths(run_dir: Path) -> list[Path]:
    """Return any ``run-*/final_summary.json`` files under ``run_dir``."""
    if not run_dir.exists():
        return []
    return sorted(run_dir.glob("run-*/final_summary.json"))


def _coerce_to_float(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None`` for anything else.

    NaN / +Inf / -Inf are rejected because they break later improvement and
    threshold comparisons (NaN compares False with everything, +/-Inf would
    pin best forever) and ``json.dumps`` writes them as non-standard
    ``NaN`` / ``Infinity`` tokens that other readers cannot parse.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    return None


def read_final_summary(run_dir: Path, metric: str) -> float | None:
    """Read ``metric`` from the most recently modified final_summary.json.

    Returns ``None`` if the file or key is absent, or if the value is not a
    finite scalar. Tolerating absence keeps the sweep alive when a trial
    legitimately ran without a summary (W&B disabled, run crashed, etc.) so
    the controller can record ``objective=None`` rather than abort.
    """
    paths = _final_summary_paths(run_dir)
    if not paths:
        return None
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    summary = json.loads(paths[0].read_text())
    return _coerce_to_float(summary.get(metric))
