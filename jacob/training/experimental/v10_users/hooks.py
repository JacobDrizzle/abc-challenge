"""Install v10_users on top of best: its config schema, its hashed files and, for each training run, SubsetWindows
with that run's two keys bound.

best.hooks.train_best looks TrainingWindows up in best.hooks when it is called, so the wrapper below swaps it there
for the length of one run. The model is best's, so MODEL_KEYS and LeanTCNN are unchanged.
"""
from contextlib import contextmanager
import functools

import run_desktop

from best import hooks as best_hooks
from best.hooks import _swapped

from experimental.v10_users.configuration import normalize_config
from experimental.v10_users.training import SubsetWindows

FILES = tuple(f"experimental/v10_users/{name}.py" for name in ("__init__", "configuration", "hooks", "training"))


@contextmanager
def installed():
    with (_swapped(best_hooks, normalize_config=normalize_config, BEST_FILES=best_hooks.BEST_FILES + FILES),
          best_hooks.installed()):
        train_best = run_desktop.train

        def train_v10(data, split, output, config, seed, *args, **kwargs):
            config = normalize_config(config)
            windows = functools.partial(SubsetWindows, train_users=config["train_users"],
                                        user_window_fraction=config["user_window_fraction"])
            with _swapped(best_hooks, TrainingWindows=windows):
                return train_best(data, split, output, config, seed, *args, **kwargs)

        with _swapped(run_desktop, train=train_v10):
            yield
