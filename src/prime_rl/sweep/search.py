from itertools import product
from typing import Any

from prime_rl.configs.sweep import SweepParameterConfig
from prime_rl.sweep.materialize import Trial, trial_label


def expand_grid(parameters: dict[str, SweepParameterConfig]) -> list[Trial]:
    paths = list(parameters.keys())
    value_lists = [parameters[path].values for path in paths]

    trials = []
    for idx, values in enumerate(product(*value_lists)):
        trial_parameters: dict[str, Any] = dict(zip(paths, values))
        trial_id = f"{idx:04d}"
        label = trial_label(trial_parameters) or trial_id
        trials.append(Trial(id=trial_id, label=label, parameters=trial_parameters))
    return trials
