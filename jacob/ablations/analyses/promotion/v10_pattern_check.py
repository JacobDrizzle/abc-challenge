"""Check the experimental/README.md hook pattern: swaps on best.hooks before entering best.hooks.installed()."""
from contextlib import contextmanager
import json, os, sys, tempfile
from pathlib import Path
ROOT = Path(r"<repo>\jacob\training")
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import torch
import run_desktop, run_next, train_tcnn, tcnn_model_adapter
from smoke_tcnn import make_fixture
from best import hooks as best_hooks
from best.configuration import normalize_config as best_normalize
from best.hooks import _swapped
from best.model import LeanTCNN

FILES = ("experimental/__init__.py",)
built = []

class ScaledTCNN(LeanTCNN):
    def __init__(self, *args, output_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_scale = output_scale
        built.append(output_scale)

def normalize_config(config):
    config = dict(config)
    scale = config.pop("output_scale", 1.0)
    if not isinstance(scale, float):
        raise ValueError("output_scale must be a float")
    return {**best_normalize(config), "output_scale": scale}

@contextmanager
def installed():
    with (_swapped(best_hooks, LeanTCNN=ScaledTCNN, normalize_config=normalize_config,
                   MODEL_KEYS=best_hooks.MODEL_KEYS + ("output_scale",), BEST_FILES=best_hooks.BEST_FILES + FILES),
          best_hooks.installed()):
        yield

before = (run_desktop.SOURCE_FILES, tcnn_model_adapter.TinyTCNN, best_hooks.LeanTCNN, run_desktop.normalize_config)
base = json.loads((ROOT/"best/configs/base.json").read_text())
with tempfile.TemporaryDirectory() as folder, installed():
    assert run_desktop.SOURCE_FILES[-1] == FILES[0] and run_next.SOURCE_FILES[-1] == FILES[0]
    assert tcnn_model_adapter.TinyTCNN is ScaledTCNN
    data, split = make_fixture(Path(folder)/"fixture", samples=900)
    run_desktop.train(data, split, Path(folder)/"run", {**base, "output_scale": 0.5, "batch_size": 4,
                      "steps_per_epoch": 2, "epochs": 1, "validation_stride": 20}, 42, "cpu", synthetic=True)
    print("built with", built, "config output_scale", torch.load(Path(folder)/"run/last.pt", weights_only=True)["config"].get("output_scale"))
after = (run_desktop.SOURCE_FILES, tcnn_model_adapter.TinyTCNN, best_hooks.LeanTCNN, run_desktop.normalize_config)
print("restored:", all(a is b for a, b in zip(before, after)))
