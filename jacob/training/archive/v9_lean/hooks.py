"""Run the frozen v5 scripts with the v9 model, schema, recipe options and exporter, inside this process only.

As v7's hooks, plus two trainer swaps: DeviceWindows becomes TrainingWindows (sampling, tilt, jitter, teacher), and with
distill_beta > 0 rotation_loss becomes the distillation loss. With every v9 option at its default, training runs as
under v7's hooks. The hashed source files gain the v9 files that can change a training or validation number;
export.py is not one of them, and every export is checked against its model by stream parity instead.
"""
from contextlib import contextmanager
import functools

import benchmark_latency
import run_desktop
import run_next
import tcnn_model_adapter
import train_tcnn

from experimental.v6_gravity_mirror.hooks import V6_FILES, _swapped
from experimental.v7_compact.hooks import MODEL_KEYS as V7_MODEL_KEYS, V7_FILES
from experimental.v9_lean import export
from experimental.v9_lean.configuration import normalize_config
from experimental.v9_lean.model import LeanTCNN, deployed, load_checkpoint
from experimental.v9_lean.training import TeacherChannel, TrainingWindows, distillation_loss

V9_FILES = tuple(f"experimental/v9_lean/{name}.py"
                 for name in ("__init__", "configuration", "hooks", "model", "training"))
MODEL_KEYS = (*V7_MODEL_KEYS, "kernel_sizes", "dilations", "maps")


def load_deployed(path):
    """load_checkpoint for the benchmark: the model with the maps its v9 graph computes."""
    model, state = load_checkpoint(path)
    return deployed(model), state


@contextmanager
def installed():
    train = run_desktop.train

    def train_v9(data, split, output, config, seed, *args, **kwargs):
        config = normalize_config(config)
        # train_tcnn builds TinyTCNN from the v5 keys and DeviceWindows from the data; v9's options are bound here.
        model = functools.partial(LeanTCNN, **{key: config[key] for key in MODEL_KEYS})
        channel = TeacherChannel()
        windows = functools.partial(TrainingWindows, sampling=config["sampling"], channel=channel,
                                    tilt_deg=config["tilt_augmentation_deg"], teacher=config["teacher"],
                                    jitter_ms=config["jitter_augmentation_ms"])
        names = dict(TinyTCNN=model, DeviceWindows=windows)
        if config["distill_beta"]:
            names["rotation_loss"] = distillation_loss(config["distill_beta"], channel)
        with _swapped(train_tcnn, **names):
            return train(data, split, output, config, seed, *args, **kwargs)

    sources = run_desktop.SOURCE_FILES + V6_FILES + V7_FILES + V9_FILES
    with (_swapped(train_tcnn, normalize_config=normalize_config),
          _swapped(run_desktop, normalize_config=normalize_config, SOURCE_FILES=sources, train=train_v9),
          _swapped(run_next, normalize_config=normalize_config, SOURCE_FILES=sources),
          _swapped(tcnn_model_adapter, TinyTCNN=LeanTCNN),
          _swapped(benchmark_latency, load_checkpoint=load_deployed, export_onnx=export.export_onnx,
                   export_streaming_onnx=export.export_streaming_onnx)):
        yield
