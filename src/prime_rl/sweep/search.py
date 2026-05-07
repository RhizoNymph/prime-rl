import hashlib
import json
from itertools import product
from typing import Any

from prime_rl.configs.sweep import SweepParameterConfig
from prime_rl.sweep.materialize import Trial, trial_label


def parameters_hash(parameters: dict[str, Any]) -> str:
    """Stable 8-character hash of a trial's flat override dict.

    Sorted-key JSON keeps the hash stable across dict insertion order. Trial IDs
    use this as a suffix so identity survives deletion and regeneration.
    """
    serialized = json.dumps(parameters, sort_keys=True, default=str)
    return hashlib.blake2b(serialized.encode(), digest_size=4).hexdigest()


def expand_grid(parameters: dict[str, SweepParameterConfig]) -> list[Trial]:
    paths = list(parameters.keys())
    value_lists = [parameters[path].values for path in paths]

    trials = []
    for idx, values in enumerate(product(*value_lists)):
        trial_parameters: dict[str, Any] = dict(zip(paths, values))
        trial_id = f"{idx:04d}-{parameters_hash(trial_parameters)}"
        label = trial_label(trial_parameters) or trial_id
        trials.append(Trial(id=trial_id, label=label, parameters=trial_parameters))
    return trials
