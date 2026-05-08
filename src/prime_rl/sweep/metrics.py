"""Metric readers for sweep trials.

The canonical source for sweep objectives is ``<run_dir>/metrics.jsonl``,
written by ``FileMonitor`` whenever the sweep launcher sets
``PRIME_RL_SWEEP_METRICS_JSONL`` on the trial subprocess. One JSON line per
``monitor.log()`` call carries ``{step, ...metrics}``.

Two reader shapes:

- ``read_final_metric`` returns the value at the largest reported step. Used
  by Phase 4 once the trial completes; the last row of metrics.jsonl IS the
  final summary for sweep purposes.
- ``read_intermediate_metric`` returns the latest ``(step, value)`` pair while
  the trial is still running. Used by the Phase 5b Optuna pruning loop.

If metrics.jsonl is absent (e.g. an SFT run without FileMonitor for some
reason, or a crashed launch before the monitor was set up), the reader falls
back to the legacy ``final_summary.json`` so existing behavior is preserved.
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


def _metrics_jsonl_path(run_dir: Path) -> Path:
    return run_dir / "metrics.jsonl"


def _iter_metrics_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Return all JSON objects in metrics.jsonl, skipping malformed lines.

    A partially-flushed final line (e.g. controller polled mid-write) is
    silently skipped. This is preferable to raising because it lets the
    polling loop keep tolerating intermediate snapshots.
    """
    path = _metrics_jsonl_path(run_dir)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def read_final_summary(run_dir: Path, metric: str) -> float | None:
    """Read ``metric`` from the canonical sidecar, falling back to legacy file.

    Sweep trials always write ``metrics.jsonl``; the final value is the
    latest reported step's reading of ``metric``. If the sidecar is missing
    (legacy artifact, crashed launch), fall back to the most recently
    modified ``final_summary.json`` under the run directory.

    Returns ``None`` if no source supplies a finite scalar for ``metric``.
    Tolerating absence keeps the sweep alive when a trial legitimately ran
    without a summary (W&B disabled, run crashed, etc.) so the controller
    can record ``objective=None`` rather than abort.
    """
    rows = _iter_metrics_rows(run_dir)
    if rows:
        rows_with_metric = [row for row in rows if metric in row]
        if rows_with_metric:
            latest = max(rows_with_metric, key=lambda r: r.get("step", -1))
            return _coerce_to_float(latest.get(metric))

    paths = _final_summary_paths(run_dir)
    if not paths:
        return None
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    summary = json.loads(paths[0].read_text())
    return _coerce_to_float(summary.get(metric))


def read_intermediate_metric(run_dir: Path, metric: str) -> tuple[int, float] | None:
    """Return the latest ``(step, value)`` pair for ``metric``, or ``None``.

    Reads metrics.jsonl, which the trial subprocess streams in real time.
    Returns ``None`` when the file is absent, the metric has not been logged
    yet, or the latest value is not a finite scalar. The Optuna pruning loop
    treats ``None`` as "no new data, do not report this round".
    """
    rows = _iter_metrics_rows(run_dir)
    if not rows:
        return None
    rows_with_metric = [row for row in rows if metric in row]
    if not rows_with_metric:
        return None
    latest = max(rows_with_metric, key=lambda r: r.get("step", -1))
    value = _coerce_to_float(latest.get(metric))
    if value is None:
        return None
    raw_step = latest.get("step")
    if not isinstance(raw_step, int) or isinstance(raw_step, bool):
        return None
    return raw_step, value
