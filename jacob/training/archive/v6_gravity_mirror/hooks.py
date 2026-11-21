"""Run the frozen v5 scripts with the v6 model and schema, inside this process only.

Each patch swaps a module-level name that those scripts look up when called: the config schema, the model
class used for training and for loading checkpoints, the streaming exporter, and the list of hashed source
files, which gains the v6 files, so experiment.json and suite.json catch any later edit to them.
"""
from contextlib import contextmanager
import functools

import benchmark_latency
import run_desktop
import run_next
import tcnn_model_adapter
import train_tcnn

from experimental.v6_gravity_mirror import export
from experimental.v6_gravity_mirror.configuration import normalize_config
from experimental.v6_gravity_mirror.model import GravityMirrorTCNN, load_checkpoint

V6_FILES = tuple(f"experimental/v6_gravity_mirror/{name}.py"
                 for name in ("__init__", "configuration", "export", "hooks", "model"))


@contextmanager
def _swapped(module, **names):
    original = {name: getattr(module, name) for name in names}
    for name, value in names.items():
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(module, name, value)


@contextmanager
def installed():
    train = run_desktop.train

    def train_v6(data, split, output, config, seed, *args, **kwargs):
        config = normalize_config(config)
        # train_tcnn builds TinyTCNN from the v5 keys; the two v6 switches are bound here.
        model = functools.partial(GravityMirrorTCNN, gravity_features=config["gravity_features"],
                                  mirror_equivariant=config["mirror_equivariant"])
        with _swapped(train_tcnn, TinyTCNN=model):
            return train(data, split, output, config, seed, *args, **kwargs)

    sources = run_desktop.SOURCE_FILES + V6_FILES
    with (_swapped(train_tcnn, normalize_config=normalize_config),
          _swapped(run_desktop, normalize_config=normalize_config, SOURCE_FILES=sources, train=train_v6),
          _swapped(run_next, normalize_config=normalize_config, SOURCE_FILES=sources),
          _swapped(tcnn_model_adapter, TinyTCNN=GravityMirrorTCNN),
          _swapped(benchmark_latency, load_checkpoint=load_checkpoint, export_onnx=export.export_onnx,
                   export_streaming_onnx=export.export_streaming_onnx)):
        yield
