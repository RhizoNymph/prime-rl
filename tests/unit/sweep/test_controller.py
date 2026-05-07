import json
from pathlib import Path

import tomli_w

from prime_rl.configs.sweep import SweepConfig
from prime_rl.sweep.controller import run_sweep


def write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def test_run_sweep_dry_run_materializes_without_launching(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    launched = False

    def fake_local(*args, **kwargs):
        nonlocal launched
        launched = True

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    config = SweepConfig(
        name="unit",
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        dry_run=True,
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
    )

    run_sweep(config)

    assert not launched
    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    variant_ids = [variant["id"] for variant in manifest["variants"]]
    assert [vid[:4] for vid in variant_ids] == ["0000", "0001"]
    assert all(len(vid) == 13 and vid[4] == "-" for vid in variant_ids)
    assert (tmp_path / "study" / "trials" / variant_ids[0] / "resolved.toml").exists()


def test_run_sweep_dispatches_local_scheduler(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    called = {}

    def fake_local(artifacts, max_parallel):
        called["count"] = len(artifacts)
        called["max_parallel"] = max_parallel

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
    )

    run_sweep(config)

    assert called == {"count": 1, "max_parallel": 1}
