"""v7 configuration schema plus the v9 options; every other key is validated by the v7, v6 and v5 schemas.

Model: kernel_sizes, dilations and maps (model.py). Recipe: sampling, tilt_augmentation_deg, jitter_augmentation_ms,
distill_beta and teacher (training.py). v7's receptive-field rule assumes v7's blocks, so the real receptive field is
checked here.
"""
from experimental.v7_compact.configuration import normalize_config as normalize_v7
from experimental.v9_lean.model import MAPS, receptive_field

V9_DEFAULTS = {"kernel_sizes": None, "dilations": None, "maps": "exact",
               "sampling": "user", "tilt_augmentation_deg": 0, "jitter_augmentation_ms": 0, "distill_beta": 0,
               "teacher": None}
SAMPLING = ("user", "sqrt", "window")


def _per_block(value, blocks, low, high):
    """None, or a list of one integer in [low, high] per block."""
    return value is None or (isinstance(value, list) and len(value) == blocks
                             and all(type(v) is int and low <= v <= high for v in value))


def normalize_config(config):
    v7 = normalize_v7({k: v for k, v in config.items() if k not in V9_DEFAULTS})
    result = {**v7, **V9_DEFAULTS, **{k: config[k] for k in V9_DEFAULTS if k in config}}
    blocks = result["num_blocks"]
    if not _per_block(result["kernel_sizes"], blocks, 2, 16) or not _per_block(result["dilations"], blocks, 1, 399):
        raise ValueError("kernel_sizes and dilations must be null or one integer per block (kernels 2 to 16)")
    if (result["kernel_sizes"] or result["dilations"]) and not result["separable"]:
        raise ValueError("kernel_sizes and dilations apply to separable blocks only")
    if result["maps"] not in MAPS:
        raise ValueError("maps must be exact, lean or native")
    if result["sampling"] not in SAMPLING:
        raise ValueError("sampling must be user, sqrt or window")
    tilt = result["tilt_augmentation_deg"]
    if type(tilt) not in (int, float) or not 0 <= tilt <= 45:
        raise ValueError("tilt_augmentation_deg must be a number from 0 to 45")
    jitter = result["jitter_augmentation_ms"]
    if type(jitter) not in (int, float) or not 0 <= jitter < 5:  # under half a sample keeps the order
        raise ValueError("jitter_augmentation_ms must be a number from 0 to under 5")
    beta, teacher = result["distill_beta"], result["teacher"]
    if type(beta) not in (int, float) or not 0 <= beta <= 1:
        raise ValueError("distill_beta must be a number from 0 to 1")
    if (beta > 0) != (teacher is not None) or not (teacher is None or isinstance(teacher, str)):
        raise ValueError("teacher must name a cache folder exactly when distill_beta > 0")
    if beta > 0 and jitter > 0:  # the cached forecasts are for the windows on the exact grid
        raise ValueError("distillation and jitter augmentation cannot be combined")
    # A streaming graph equals the stateless one only if every input step it sees is in the window.
    if receptive_field(blocks, result["dilation_base"], result["kernel_sizes"],
                       result["dilations"]) + 1 > result["effective_context"]:
        raise ValueError("effective_context must be at least the receptive field + 1")
    return result
