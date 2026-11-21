"""Install v10_summary on top of best: SummaryTCNN, its config key and its hashed files.

best.hooks.installed() looks LeanTCNN, normalize_config, MODEL_KEYS and BEST_FILES up in best.hooks when it is
entered, so they are swapped there first. best.model.LeanTCNN is swapped too, so best.model.load_checkpoint (used by
the benchmark adapter) builds SummaryTCNN while a run is going.
"""
from contextlib import contextmanager

from best import hooks as best_hooks
from best import model as best_model
from best.hooks import _swapped

from experimental.v10_summary.configuration import normalize_config
from experimental.v10_summary.model import SummaryTCNN

FILES = tuple(f"experimental/v10_summary/{name}.py" for name in ("__init__", "configuration", "hooks", "model"))


@contextmanager
def installed():
    with (_swapped(best_hooks, LeanTCNN=SummaryTCNN, normalize_config=normalize_config,
                   MODEL_KEYS=best_hooks.MODEL_KEYS + ("summary_taus",), BEST_FILES=best_hooks.BEST_FILES + FILES),
          _swapped(best_model, LeanTCNN=SummaryTCNN),
          best_hooks.installed()):
        yield
