"""Install v10_fast on top of best: FastLeanTCNN in place of LeanTCNN, and this package's files in the hashed list.

best.hooks.installed() reads LeanTCNN and BEST_FILES from its own module when it is entered, and train_best looks
LeanTCNN up there again for every run, so both are swapped there for as long as this context is open. That covers
the three places a number is computed: training, the validation after each epoch (train_tcnn builds the model once
and validates its EMA copy) and the full validation that follows a run (tcnn_model_adapter). The config schema,
MODEL_KEYS and the recipe options are best's. Export and the benchmark still load LeanTCNN through
best.model.load_checkpoint: the checkpoints are the same.
"""
from contextlib import contextmanager

from best import hooks as best_hooks
from best.hooks import _swapped

from experimental.v10_fast.model import FastLeanTCNN

FILES = tuple(f"experimental/v10_fast/{name}.py" for name in ("__init__", "hooks", "model"))


@contextmanager
def installed():
    with (_swapped(best_hooks, LeanTCNN=FastLeanTCNN, BEST_FILES=best_hooks.BEST_FILES + FILES),
          best_hooks.installed()):
        yield
