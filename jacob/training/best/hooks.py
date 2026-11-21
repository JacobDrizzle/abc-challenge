"""Run the frozen v5 scripts with the best model, schema, recipe options and exporter, inside this process only.

Each patch swaps a module-level name that those scripts look up when called: the config schema, the model class used
for training and for loading checkpoints, the trainer's window sampler (DeviceWindows becomes TrainingWindows) and,
with distill_beta > 0, its loss, the streaming exporter, and the list of hashed source files. BEST_FILES, the files
that can change a training or validation number, join that list, so experiment.json and suite.json catch any later
edit to them. export.py is not one of them: every export is checked against its model by stream parity instead.
An experiment in experimental/ swaps names in this module (for example LeanTCNN) inside its own installed().
"""
from contextlib import contextmanager
import functools

import benchmark_latency
import run_desktop
import run_next
import tcnn_model_adapter
import train_tcnn

from best import export
from best.configuration import V7_DEFAULTS, normalize_config
from best.model import LeanTCNN, deployed, load_checkpoint
from best.training import TeacherChannel, TrainingWindows, distillation_loss

BEST_FILES = tuple(f"best/{name}.py" for name in ("__init__", "configuration", "hooks", "model", "training"))
MODEL_KEYS = ("gravity_features", "mirror_equivariant", *V7_DEFAULTS, "kernel_sizes", "dilations", "maps")


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


def load_deployed(path):
    """load_checkpoint for the benchmark: the model with the maps its streaming graph computes."""
    model, state = load_checkpoint(path)
    return deployed(model), state


@contextmanager
def installed():
    train = run_desktop.train

    def train_best(data, split, output, config, seed, *args, **kwargs):
        config = normalize_config(config)
        # train_tcnn builds TinyTCNN from the v5 keys and DeviceWindows from the data; the other options are bound here.
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

    sources = run_desktop.SOURCE_FILES + BEST_FILES
    with (_swapped(train_tcnn, normalize_config=normalize_config),
          _swapped(run_desktop, normalize_config=normalize_config, SOURCE_FILES=sources, train=train_best),
          _swapped(run_next, normalize_config=normalize_config, SOURCE_FILES=sources),
          _swapped(tcnn_model_adapter, TinyTCNN=LeanTCNN),
          _swapped(benchmark_latency, load_checkpoint=load_deployed, export_onnx=export.export_onnx,
                   export_streaming_onnx=export.export_streaming_onnx)):
        yield
