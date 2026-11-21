"""Run the frozen v5 scripts with the v7 model, schema and exporter, inside this process only (as v6's hooks).

The hashed source files gain the v6 and v7 files, since v7 imports v6, so experiment.json and suite.json catch
any later edit to either.
"""
from contextlib import contextmanager
import functools

import benchmark_latency
import run_desktop
import run_next
import tcnn_model_adapter
import train_tcnn

from experimental.v6_gravity_mirror.hooks import V6_FILES, _swapped
from experimental.v7_compact import export
from experimental.v7_compact.configuration import V7_DEFAULTS, normalize_config
from experimental.v7_compact.model import CompactTCNN, load_checkpoint

V7_FILES = tuple(f"experimental/v7_compact/{name}.py"
                 for name in ("__init__", "configuration", "export", "hooks", "model"))
MODEL_KEYS = ("gravity_features", "mirror_equivariant", *V7_DEFAULTS)


@contextmanager
def installed():
    train = run_desktop.train

    def train_v7(data, split, output, config, seed, *args, **kwargs):
        config = normalize_config(config)
        # train_tcnn builds TinyTCNN from the v5 keys; the v6 and v7 switches are bound here.
        model = functools.partial(CompactTCNN, **{key: config[key] for key in MODEL_KEYS})
        with _swapped(train_tcnn, TinyTCNN=model):
            return train(data, split, output, config, seed, *args, **kwargs)

    sources = run_desktop.SOURCE_FILES + V6_FILES + V7_FILES
    with (_swapped(train_tcnn, normalize_config=normalize_config),
          _swapped(run_desktop, normalize_config=normalize_config, SOURCE_FILES=sources, train=train_v7),
          _swapped(run_next, normalize_config=normalize_config, SOURCE_FILES=sources),
          _swapped(tcnn_model_adapter, TinyTCNN=CompactTCNN),
          _swapped(benchmark_latency, load_checkpoint=load_checkpoint, export_onnx=export.export_onnx,
                   export_streaming_onnx=export.export_streaming_onnx)):
        yield
