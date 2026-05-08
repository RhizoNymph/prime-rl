"""Shared fixtures for sweep tests.

The multi_run_lora driver suites both depend on the same fake
``rl-multi-run`` subprocess: a Popen that scans the shared dir for new
``run_*/control/orch.toml`` files, seeds metrics, writes per-orchestrator
``control/exit_code``, and exits when the controller drops the done marker.

Phase 7e routes static (grid/random) multi_run sweeps through the same
continuous-flow path, so the fake needs to live somewhere both test files
can reach it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


class FakeMultiRunPopen:
    """Fake ``subprocess.Popen`` that simulates ``rl-multi-run --watch-slots``.

    The launcher in production keeps watching the parent of ``--runs-dir``
    for new ``run_*/control/orch.toml`` files and spawns orchestrators on
    demand, writing each one's ``control/exit_code`` as it exits. This fake
    mirrors that without actually running orchestrators: each ``poll()`` it
    discovers any newly-materialized run dirs, seeds them with a single
    metrics row, and writes ``exit_code`` (``"1\\n"`` if status is pre-marked
    pruned, else ``"0\\n"``). ``poll()`` returns ``None`` until the
    controller drops ``<shared_dir>/control/done``, then ``0``.

    Test knobs:

    - ``fail_first_attempt``: when True, dirs without an ``-r`` suffix get
      ``exit_code=1`` (no metrics) so the controller's auto-retry kicks in.
    - ``exit_codes_by_index``: per-trial-index override for the exit code
      and metric. Keyed by the integer prefix in ``run_<NNNN>-<hash>``.
      ``None`` value means "don't write exit_code at all" (simulates a
      launcher death before the orchestrator started).
    - ``rewards_by_index``: per-trial-index reward to seed in metrics.jsonl
      for completed trials.

    Non-``rl-multi-run`` invocations (e.g. ``git rev-parse``) are delegated
    to the real ``subprocess.Popen``: patching
    ``multi_run.subprocess.Popen`` patches the stdlib module attribute, so
    every Popen call in the process flows through this fake while the test
    runs.
    """

    instances: list["FakeMultiRunPopen"] = []
    _real_popen: Any = None
    fail_first_attempt: bool = False
    exit_codes_by_index: dict[int, int | None] = {}
    rewards_by_index: dict[int, float] = {}
    default_reward: float = 0.5

    def __new__(cls, command, **kwargs):
        if not command or command[0] != "rl-multi-run":
            assert cls._real_popen is not None
            return cls._real_popen(command, **kwargs)
        return super().__new__(cls)

    def __init__(self, command, **kwargs) -> None:
        if not command or command[0] != "rl-multi-run":
            return
        FakeMultiRunPopen.instances.append(self)
        self.command = list(command)
        idx = command.index("--runs-dir")
        self.run_dirs = [Path(p) for p in command[idx + 1].split(":") if p]
        self.shared_dir = self.run_dirs[0].parent
        self.watch_slots = "--watch-slots" in command
        self.seen_run_ids: set[str] = set()
        self.returncode: int | None = None

    def _trial_index(self, run_dir: Path) -> int:
        suffix = run_dir.name.removeprefix("run_")
        return int(suffix.split("-", 1)[0])

    def _process_run_dirs(self) -> None:
        for run_dir in sorted(self.shared_dir.glob("run_*")):
            if run_dir.name in self.seen_run_ids:
                continue
            if not (run_dir / "control" / "orch.toml").exists():
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "control").mkdir(parents=True, exist_ok=True)
            status_path = run_dir / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            is_retry = "-r" in run_dir.name.removeprefix("run_")
            trial_index = self._trial_index(run_dir)

            if FakeMultiRunPopen.fail_first_attempt and not is_retry:
                code: int | None = 1
                reward: float | None = None
            elif trial_index in FakeMultiRunPopen.exit_codes_by_index:
                code = FakeMultiRunPopen.exit_codes_by_index[trial_index]
                reward = FakeMultiRunPopen.rewards_by_index.get(trial_index)
            elif status.get("state") == "pruned":
                code = 1
                reward = None
            else:
                code = 0
                reward = FakeMultiRunPopen.rewards_by_index.get(
                    trial_index, FakeMultiRunPopen.default_reward
                )

            if reward is not None:
                (run_dir / "metrics.jsonl").write_text(
                    json.dumps({"step": 1, "reward": reward}) + "\n"
                )
            if code is not None:
                (run_dir / "control" / "exit_code").write_text(f"{code}\n")
            self.seen_run_ids.add(run_dir.name)

    def poll(self) -> int | None:
        self._process_run_dirs()
        if self.watch_slots:
            done_marker = self.shared_dir / "control" / "done"
            if done_marker.exists():
                self.returncode = 0
                return 0
            return None
        # Wave-mode fallback for callers that haven't migrated; one tick of
        # work, then exit cleanly.
        self.returncode = 0
        return 0

    def wait(self) -> int:
        if self.returncode is None:
            self._process_run_dirs()
            self.returncode = 0
        return self.returncode  # type: ignore[return-value]


@pytest.fixture
def fake_multi_run_popen(monkeypatch):
    """Install ``FakeMultiRunPopen`` over ``multi_run.subprocess.Popen``.

    Tests get back the class so they can configure knobs (``fail_first_attempt``,
    ``exit_codes_by_index``, ``rewards_by_index``) and inspect ``instances``
    after ``run_sweep`` returns. State is reset between tests via the
    fixture's setup/teardown.
    """
    import subprocess as real_subprocess

    from prime_rl.sweep import multi_run as multi_run_mod

    FakeMultiRunPopen._real_popen = real_subprocess.Popen
    FakeMultiRunPopen.instances = []
    FakeMultiRunPopen.fail_first_attempt = False
    FakeMultiRunPopen.exit_codes_by_index = {}
    FakeMultiRunPopen.rewards_by_index = {}
    FakeMultiRunPopen.default_reward = 0.5

    monkeypatch.setattr(multi_run_mod.subprocess, "Popen", FakeMultiRunPopen)
    monkeypatch.setattr(multi_run_mod.time, "sleep", lambda *_a, **_kw: None)
    yield FakeMultiRunPopen
    FakeMultiRunPopen.instances = []
    FakeMultiRunPopen.exit_codes_by_index = {}
    FakeMultiRunPopen.rewards_by_index = {}
    FakeMultiRunPopen.fail_first_attempt = False
