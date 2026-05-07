import re

from prime_rl.configs.sweep import SweepParameterConfig
from prime_rl.sweep.materialize import build_nested_overrides, set_dotted_path
from prime_rl.sweep.search import expand_grid, parameters_hash


def test_expand_grid_uses_deterministic_insertion_order() -> None:
    trials = expand_grid(
        {
            "a": SweepParameterConfig(values=[1, 2]),
            "b": SweepParameterConfig(values=["x", "y"]),
        }
    )

    assert [trial.parameters for trial in trials] == [
        {"a": 1, "b": "x"},
        {"a": 1, "b": "y"},
        {"a": 2, "b": "x"},
        {"a": 2, "b": "y"},
    ]
    pattern = re.compile(r"^\d{4}-[0-9a-f]{8}$")
    assert [trial.id[:4] for trial in trials] == ["0000", "0001", "0002", "0003"]
    assert all(pattern.match(trial.id) for trial in trials)


def test_expand_grid_hash_suffix_is_stable() -> None:
    parameters = {"trainer.optim.lr": 1e-5, "orchestrator.train.sampling.temperature": 0.7}
    assert parameters_hash(parameters) == parameters_hash(dict(reversed(parameters.items())))


def test_expand_grid_hash_changes_with_values() -> None:
    a = parameters_hash({"trainer.optim.lr": 1e-5})
    b = parameters_hash({"trainer.optim.lr": 3e-5})
    assert a != b


def test_build_nested_overrides_from_dotted_paths() -> None:
    overrides = build_nested_overrides(
        {
            "trainer.optim.lr": 1e-5,
            "orchestrator.train.sampling.temperature": 0.7,
            "output_dir": "outputs/run",
        }
    )

    assert overrides == {
        "trainer": {"optim": {"lr": 1e-5}},
        "orchestrator": {"train": {"sampling": {"temperature": 0.7}}},
        "output_dir": "outputs/run",
    }


def test_set_dotted_path_rejects_table_conflict() -> None:
    data = {"trainer": 1}

    try:
        set_dotted_path(data, "trainer.optim.lr", 1e-5)
    except ValueError as exc:
        assert "non-table" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
