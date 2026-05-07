import json
import os
from pathlib import Path

import pytest

from prime_rl.utils.monitor.file import FileMonitor


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_file_monitor_writes_one_line_per_log_call(tmp_path: Path) -> None:
    monitor = FileMonitor(tmp_path / "metrics.jsonl")
    try:
        monitor.log({"reward": 0.1, "loss": 1.5}, step=1)
        monitor.log({"reward": 0.4, "loss": 1.2}, step=2)
    finally:
        monitor.close()

    rows = _read_lines(tmp_path / "metrics.jsonl")
    assert rows == [
        {"step": 1, "reward": 0.1, "loss": 1.5},
        {"step": 2, "reward": 0.4, "loss": 1.2},
    ]


def test_file_monitor_creates_parent_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "metrics.jsonl"
    monitor = FileMonitor(target)
    monitor.close()
    assert target.exists()


def test_file_monitor_flushes_after_each_write(tmp_path: Path) -> None:
    """Polling readers see partial files; writes must hit disk between log calls."""
    path = tmp_path / "metrics.jsonl"
    monitor = FileMonitor(path)
    try:
        monitor.log({"reward": 0.5}, step=10)
        # File must already contain the line on disk; we did not close yet.
        rows = _read_lines(path)
        assert rows == [{"step": 10, "reward": 0.5}]
    finally:
        monitor.close()


def test_file_monitor_replaces_non_finite_values_with_null(tmp_path: Path) -> None:
    monitor = FileMonitor(tmp_path / "metrics.jsonl")
    try:
        monitor.log({"reward": float("nan"), "loss": float("inf"), "ok": 0.5}, step=1)
    finally:
        monitor.close()
    rows = _read_lines(tmp_path / "metrics.jsonl")
    assert rows == [{"step": 1, "reward": None, "loss": None, "ok": 0.5}]


def test_file_monitor_no_op_on_non_master_rank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "1")
    target = tmp_path / "metrics.jsonl"
    monitor = FileMonitor(target)
    try:
        monitor.log({"reward": 1.0}, step=1)
    finally:
        monitor.close()
    assert not target.exists()


def test_file_monitor_appends_to_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps({"step": 0, "reward": 0.0}) + "\n")

    monitor = FileMonitor(path)
    try:
        monitor.log({"reward": 0.5}, step=1)
    finally:
        monitor.close()

    rows = _read_lines(path)
    assert rows == [
        {"step": 0, "reward": 0.0},
        {"step": 1, "reward": 0.5},
    ]


def test_file_monitor_history_buffer(tmp_path: Path) -> None:
    monitor = FileMonitor(tmp_path / "metrics.jsonl", keep_full_history=True)
    try:
        monitor.log({"reward": 0.1}, step=1)
        monitor.log({"reward": 0.2}, step=2)
    finally:
        monitor.close()
    assert monitor.history == [{"reward": 0.1}, {"reward": 0.2}]


def test_setup_monitor_adds_file_monitor_when_env_var_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sweep launcher injection: env var present => FileMonitor in the stack."""
    target = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("PRIME_RL_SWEEP_METRICS_JSONL", str(target))
    monkeypatch.setattr("prime_rl.utils.monitor._MONITOR", None)

    from prime_rl.utils.monitor import FileMonitor as FileMonitorClass
    from prime_rl.utils.monitor import setup_monitor

    monitor = setup_monitor()
    try:
        monitor.log({"reward": 0.7}, step=3)
    finally:
        if hasattr(monitor, "close"):
            monitor.close()

    # NoOpMonitor by itself if no other monitors. Single monitor path returns
    # the FileMonitor directly.
    assert isinstance(monitor, FileMonitorClass)
    assert target.exists()
