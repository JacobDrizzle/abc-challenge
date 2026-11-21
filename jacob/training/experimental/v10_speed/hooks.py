"""Install v10_speed on top of best: its model class, config schema, hashed files and, for each training run, the
window class and the validation function.

best.hooks.train_best looks LeanTCNN, MODEL_KEYS and TrainingWindows up in best.hooks when it is called, and
train_tcnn.train looks validate up in train_tcnn. LastPositionTCNN and its switch are swapped in before
best.hooks.installed() is entered, so the full-validation adapter loads checkpoints with it too; the window class
and validate are swapped for the length of one run, with that run's dense_validation bound.
"""
from contextlib import contextmanager
import functools

import run_desktop
import train_tcnn

from best import hooks as best_hooks
from best.hooks import _swapped

from experimental.v10_speed.configuration import normalize_config
from experimental.v10_speed.model import LastPositionTCNN
from experimental.v10_speed.training import HostWindows
from experimental.v10_speed.validation import dense_validate

FILES = tuple(f"experimental/v10_speed/{name}.py"
              for name in ("__init__", "configuration", "hooks", "model", "training", "validation"))


@contextmanager
def installed(model=LastPositionTCNN, normalize=normalize_config, model_keys=(), files=()):
    """The arguments are for a package built on this one: its LastPositionTCNN subclass, its config schema (which
    must pass the rest to this package's), its model keys and its hashed files."""
    with (_swapped(best_hooks, LeanTCNN=model, normalize_config=normalize,
                   MODEL_KEYS=best_hooks.MODEL_KEYS + ("last_position_forward",) + tuple(model_keys),
                   BEST_FILES=best_hooks.BEST_FILES + FILES + tuple(files)),
          best_hooks.installed()):
        train_best = run_desktop.train

        def train_v10(data, split, output, config, seed, *args, **kwargs):
            config = normalize(config)
            windows = functools.partial(HostWindows, dense_validation=config["dense_validation"])
            with _swapped(best_hooks, TrainingWindows=windows), _swapped(train_tcnn, validate=dense_validate):
                return train_best(data, split, output, config, seed, *args, **kwargs)

        with _swapped(run_desktop, train=train_v10):
            yield
