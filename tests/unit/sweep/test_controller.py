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
    assert "git" in manifest
    assert set(manifest["git"]) == {"sha", "dirty"}
    for variant in manifest["variants"]:
        assert len(variant["resolved_checksum"]) == 64
        assert variant["base_checksums"][base_path.as_posix()]


def test_run_sweep_dispatches_local_scheduler(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    called = {}

    def fake_local(artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None):
        called["count"] = len(artifacts)
        called["max_parallel"] = max_parallel
        called["gpu_groups"] = gpu_groups
        called["continue_on_failure"] = continue_on_failure
        called["retry_budget"] = retry_budget
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5]}},
    )

    run_sweep(config)

    assert called == {
        "count": 1,
        "max_parallel": 1,
        "gpu_groups": None,
        "continue_on_failure": True,
        "retry_budget": 1,
    }


def test_run_sweep_exits_nonzero_when_trials_fail(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    monkeypatch.setattr(
        "prime_rl.sweep.controller.run_trials_locally",
        lambda *args, **kwargs: 2,
    )

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5, 1e-4]}},
    )

    try:
        run_sweep(config)
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("Expected SystemExit when trials failed")


def test_run_sweep_random_strategy_dispatches_through_local_scheduler(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    captured = {}

    def fake_local(artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None):
        captured["count"] = len(artifacts)
        captured["parameters"] = [artifact.trial.parameters for artifact in artifacts]
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        strategy={"type": "random", "num_trials": 5, "seed": 13},
        parameters={
            "optim.lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4},
            "data.temperature": {"distribution": "uniform", "min": 0.6, "max": 1.2},
        },
    )

    run_sweep(config)

    assert captured["count"] == 5
    for params in captured["parameters"]:
        assert 1e-6 <= params["optim.lr"] <= 1e-4
        assert 0.6 <= params["data.temperature"] <= 1.2


def test_run_sweep_resume_skips_completed_trials(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    base_config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
    )

    runs: list[list[str]] = []

    def fake_local(artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None):
        runs.append([artifact.trial.id for artifact in artifacts])
        first_status = json.loads(artifacts[0].status_path.read_text())
        first_status.update({"state": "completed", "returncode": 0})
        artifacts[0].status_path.write_text(json.dumps(first_status, indent=2, sort_keys=True) + "\n")
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    run_sweep(SweepConfig(**base_config_kwargs))

    completed_id = runs[0][0]
    pending_id = runs[0][1]

    completed_status_path = tmp_path / "study" / "trials" / completed_id / "status.json"
    pending_status_path = tmp_path / "study" / "trials" / pending_id / "status.json"
    assert json.loads(completed_status_path.read_text())["state"] == "completed"
    pending_status = json.loads(pending_status_path.read_text())
    pending_status.update({"state": "failed", "returncode": 1})
    pending_status_path.write_text(json.dumps(pending_status, indent=2, sort_keys=True) + "\n")

    resume_runs: list[list[str]] = []

    def fake_local_resume(
        artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None
    ):
        resume_runs.append(
            [(artifact.trial.id, json.loads(artifact.status_path.read_text())["state"]) for artifact in artifacts]
        )
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)

    run_sweep(SweepConfig(**base_config_kwargs, resume=True))

    assert resume_runs[0] == [(completed_id, "completed"), (pending_id, "pending")]
    assert json.loads(completed_status_path.read_text())["state"] == "completed"


def test_run_sweep_resume_seeds_tracker_from_completed_trials(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    base_config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    captured_completion = []

    def fake_local(
        artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None
    ):
        # First run: mark both trials completed with recorded objectives.
        for value, artifact in zip([0.9, 0.7], artifacts):
            status = json.loads(artifact.status_path.read_text())
            status.update({"state": "completed", "returncode": 0, "objective": value})
            artifact.status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    run_sweep(SweepConfig(**base_config_kwargs))

    def fake_local_resume(
        artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None
    ):
        captured_completion.append([artifact.trial.id for artifact in artifacts])
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)

    run_sweep(SweepConfig(**base_config_kwargs, resume=True))

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    summary = manifest["summary"]
    assert summary["completed"] == 2
    assert summary["best_value"] == 0.9


def test_run_sweep_resume_short_circuits_when_seeding_triggers_halt(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    base_config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5, 1e-4]}},
        objective={"metric": "reward", "direction": "maximize"},
        early_stopping={"type": "threshold", "threshold": 0.5},
        wandb=None,
    )

    def fake_local(
        artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None
    ):
        for value, artifact in zip([0.9, 0.4], artifacts[:2]):
            status = json.loads(artifact.status_path.read_text())
            status.update({"state": "completed", "returncode": 0, "objective": value})
            artifact.status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)
    run_sweep(SweepConfig(**base_config_kwargs))

    invoked = []

    def fake_local_resume(*args, **kwargs):
        invoked.append(True)
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local_resume)
    run_sweep(SweepConfig(**base_config_kwargs, resume=True))

    assert invoked == []
    summary = json.loads((tmp_path / "study" / "manifest.json").read_text())["summary"]
    assert summary["halted_by_early_stopping"] is True
    assert summary["halt_reason"] == "threshold"


def test_run_sweep_skips_tracker_for_slurm_scheduler(tmp_path: Path, monkeypatch, capsys) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    monkeypatch.setattr("prime_rl.sweep.controller.submit_trials_to_slurm", lambda *args, **kwargs: 0)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        scheduler={"type": "slurm"},
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
        objective={"metric": "reward", "direction": "maximize"},
        wandb=None,
    )

    run_sweep(config)

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert manifest.get("summary") is None
    assert "objective tracking is only computed for the local scheduler" in capsys.readouterr().out


def test_run_sweep_records_objective_and_halts_on_threshold(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    import subprocess as real_subprocess
    from types import SimpleNamespace

    seq = iter([0.9, 0.8, 0.2])
    real_run = real_subprocess.run

    def fake_run(command, env=None, **kwargs):
        if command[:2] == ["git", "rev-parse"] or command[:2] == ["git", "status"]:
            return real_run(command, **kwargs)
        overrides = [part for part in command if part.endswith("overrides.toml")]
        if not overrides:
            return real_run(command, **kwargs)
        run_dir = Path(overrides[0]).parent / "run"
        summary_dir = run_dir / "run-fake"
        summary_dir.mkdir(parents=True, exist_ok=True)
        (summary_dir / "final_summary.json").write_text(json.dumps({"reward": next(seq)}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_run)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        parameters={"optim.lr": {"values": [1e-5, 3e-5, 1e-4, 1e-3]}},
        objective={"metric": "reward", "direction": "maximize"},
        early_stopping={"type": "threshold", "threshold": 0.5},
        wandb=None,
    )

    run_sweep(config)

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    summary = manifest["summary"]
    assert summary["completed"] == 3
    assert summary["best_value"] == 0.9
    assert summary["halted_by_early_stopping"] is True
    assert summary["halt_reason"] == "threshold"


def test_run_sweep_passes_gpu_groups_to_local_scheduler(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    captured = {}

    def fake_local(artifacts, max_parallel, gpu_groups, continue_on_failure, retry_budget, on_trial_complete=None):
        captured["max_parallel"] = max_parallel
        captured["gpu_groups"] = gpu_groups
        return 0

    monkeypatch.setattr("prime_rl.sweep.controller.run_trials_locally", fake_local)

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        scheduler={
            "type": "local",
            "max_parallel": 2,
            "gpu_assignment": {"visible_devices": [[0, 1], [2, 3]]},
        },
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
    )

    run_sweep(config)

    assert captured == {"max_parallel": 2, "gpu_groups": [[0, 1], [2, 3]]}


def test_run_sweep_dispatches_to_slurm_array_when_use_array(
    tmp_path: Path, monkeypatch
) -> None:
    """Phase 8: SlurmSweepSchedulerConfig.use_array=True dispatches to the
    array submitter, stamps array_task_index per variant, and records the
    SLURM job id at study level."""
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    captured: dict = {}

    def fake_array_submit(artifacts, *, study_dir, array_indices=None):
        captured["count"] = len(artifacts)
        captured["indices"] = list(array_indices) if array_indices is not None else None
        captured["study_dir"] = study_dir
        return "777", array_indices or list(range(len(artifacts)))

    monkeypatch.setattr(
        "prime_rl.sweep.controller.submit_trials_to_slurm_array", fake_array_submit
    )

    config = SweepConfig(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        scheduler={"type": "slurm", "use_array": True},
        parameters={"optim.lr": {"values": [1e-5, 3e-5, 1e-4]}},
    )

    run_sweep(config)

    assert captured["count"] == 3
    assert captured["indices"] == [0, 1, 2]
    assert captured["study_dir"] == tmp_path / "study"

    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert manifest["array_job_id"] == "777"
    indices_in_manifest = sorted(v["array_task_index"] for v in manifest["variants"])
    assert indices_in_manifest == [0, 1, 2]


def test_run_sweep_array_resume_skips_indices_still_running_in_squeue(
    tmp_path: Path, monkeypatch
) -> None:
    """Phase 8 resume: squeue says tasks 0,1 still alive on the prior array
    job; resume must not re-submit them. Only the failed/pending indices
    that aren't still on the cluster come back in a fresh sbatch."""
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        scheduler={"type": "slurm", "use_array": True},
        parameters={"optim.lr": {"values": [1e-5, 3e-5, 1e-4, 5e-5]}},
    )

    # Run 1: submit the initial array (job id "111"). Stash for run 2.
    initial_call: dict = {}

    def fake_array_submit_initial(artifacts, *, study_dir, array_indices=None):
        initial_call["indices"] = list(array_indices) if array_indices is not None else None
        return "111", array_indices or list(range(len(artifacts)))

    monkeypatch.setattr(
        "prime_rl.sweep.controller.submit_trials_to_slurm_array", fake_array_submit_initial
    )
    run_sweep(SweepConfig(**config_kwargs))

    assert initial_call["indices"] == [0, 1, 2, 3]

    # Mutate prior status.json: trials 0,1 are state=running on the cluster
    # (mid-execution), trial 2 completed, trial 3 failed.
    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    by_idx = {v["array_task_index"]: Path(v["status_path"]) for v in manifest["variants"]}
    states = {0: "running", 1: "running", 2: "completed", 3: "failed"}
    for idx, state in states.items():
        status_path = by_idx[idx]
        s = json.loads(status_path.read_text())
        s["state"] = state
        status_path.write_text(json.dumps(s, indent=2, sort_keys=True) + "\n")

    # Run 2 (resume): squeue claims tasks 0 and 1 are still pending/running on
    # job 111. Index 3 (failed) is no longer on the cluster and should be
    # the only one re-submitted.
    resume_call: dict = {}

    def fake_array_submit_resume(artifacts, *, study_dir, array_indices=None):
        resume_call["indices"] = list(array_indices) if array_indices is not None else None
        return "222", array_indices or list(range(len(artifacts)))

    def fake_squeue(command, **kwargs):
        from types import SimpleNamespace

        if command[:1] == ["squeue"]:
            assert "111" in command  # querying the prior job
            return SimpleNamespace(returncode=0, stdout="0\n1\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "prime_rl.sweep.controller.submit_trials_to_slurm_array", fake_array_submit_resume
    )
    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_squeue)

    run_sweep(SweepConfig(**config_kwargs, resume=True))

    # Index 0,1 still live on cluster → skip. Index 2 completed → skip.
    # Index 3 failed and not in squeue → resubmit.
    assert resume_call["indices"] == [3]

    # The prior array_job_id ("111") gets overwritten by the new submission's
    # ("222") because there *was* a new submission this run.
    manifest_after = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert manifest_after["array_job_id"] == "222"


def test_run_sweep_array_resume_preserves_array_job_id_when_nothing_to_submit(
    tmp_path: Path, monkeypatch
) -> None:
    """Phase 8 resume: every task either completed or still running on the
    prior array job → no new sbatch. The prior array_job_id must survive
    the resume's manifest rewrite so squeue can still find the job later."""
    base_path = tmp_path / "base.toml"
    write_toml(base_path, {"data": {"type": "fake"}, "max_steps": 1})

    config_kwargs = dict(
        entrypoint="sft",
        base=[base_path],
        output_dir=tmp_path / "study",
        scheduler={"type": "slurm", "use_array": True},
        parameters={"optim.lr": {"values": [1e-5, 3e-5]}},
    )

    monkeypatch.setattr(
        "prime_rl.sweep.controller.submit_trials_to_slurm_array",
        lambda artifacts, *, study_dir, array_indices=None: (
            "555",
            array_indices or list(range(len(artifacts))),
        ),
    )
    run_sweep(SweepConfig(**config_kwargs))

    # Mark trial 0 completed, trial 1 still running on cluster.
    manifest = json.loads((tmp_path / "study" / "manifest.json").read_text())
    by_idx = {v["array_task_index"]: Path(v["status_path"]) for v in manifest["variants"]}
    for idx, state in {0: "completed", 1: "running"}.items():
        s = json.loads(by_idx[idx].read_text())
        s["state"] = state
        by_idx[idx].write_text(json.dumps(s, indent=2, sort_keys=True) + "\n")

    submitted: list = []

    def fake_array_submit_should_not_be_called(*args, **kwargs):
        submitted.append(args)
        raise AssertionError("submit_trials_to_slurm_array should not run when nothing to submit")

    def fake_squeue(command, **kwargs):
        from types import SimpleNamespace

        if command[:1] == ["squeue"]:
            return SimpleNamespace(returncode=0, stdout="1\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "prime_rl.sweep.controller.submit_trials_to_slurm_array",
        fake_array_submit_should_not_be_called,
    )
    monkeypatch.setattr("prime_rl.sweep.schedulers.subprocess.run", fake_squeue)

    run_sweep(SweepConfig(**config_kwargs, resume=True))
    assert submitted == []

    manifest_after = json.loads((tmp_path / "study" / "manifest.json").read_text())
    assert manifest_after["array_job_id"] == "555"
