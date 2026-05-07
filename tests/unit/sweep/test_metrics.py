import json
from pathlib import Path

from prime_rl.sweep.metrics import read_final_summary


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_read_final_summary_returns_none_when_missing(tmp_path: Path) -> None:
    assert read_final_summary(tmp_path / "missing", "val/loss") is None
    (tmp_path / "no_summary").mkdir()
    assert read_final_summary(tmp_path / "no_summary", "val/loss") is None


def test_read_final_summary_finds_value_under_run_subdir(tmp_path: Path) -> None:
    write_summary(tmp_path / "run-abc123" / "final_summary.json", {"val/loss": 0.42, "step": 100})
    assert read_final_summary(tmp_path, "val/loss") == 0.42


def test_read_final_summary_picks_latest_when_multiple_runs(tmp_path: Path) -> None:
    older = tmp_path / "run-old" / "final_summary.json"
    newer = tmp_path / "run-new" / "final_summary.json"
    write_summary(older, {"val/loss": 1.0})
    write_summary(newer, {"val/loss": 2.0})
    import os

    os.utime(older, (1.0, 1.0))
    os.utime(newer, (10.0, 10.0))

    assert read_final_summary(tmp_path, "val/loss") == 2.0


def test_read_final_summary_returns_none_for_non_scalar(tmp_path: Path) -> None:
    write_summary(tmp_path / "run-x" / "final_summary.json", {"val/loss": "nope", "flag": True})
    assert read_final_summary(tmp_path, "val/loss") is None
    assert read_final_summary(tmp_path, "flag") is None
    assert read_final_summary(tmp_path, "missing.key") is None
