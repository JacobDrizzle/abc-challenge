"""Config schema for the best model: the v5 schema (tcnn.configuration) plus the v6, v7 and v9 keys.

The keys are checked in three layers, in the order they were added, exactly as the v6, v7 and v9 schemas did:
- v6: gravity_features, mirror_equivariant.
- v7: dilation_base, separable, head_basis, speed_feature. Two v5 rules are replaced rather than inherited: v5 allows
  only channels 32 or 48, and checks the receptive field for base-2 dilations only.
- v9: kernel_sizes, dilations and maps (model.py); sampling, tilt_augmentation_deg, jitter_augmentation_ms,
  distill_beta and teacher (training.py). The v7 receptive-field rule assumes kernel-3 blocks, so the real receptive
  field is checked here.
A config without a key gets that key's default, which is the earlier model's behaviour.
"""
from tcnn import configuration

from best.model import MAPS, receptive_field

V6_DEFAULTS = {"gravity_features": False, "mirror_equivariant": False}
V7_DEFAULTS = {"dilation_base": 2, "separable": False, "head_basis": None, "speed_feature": "norm"}
V9_DEFAULTS = {"kernel_sizes": None, "dilations": None, "maps": "exact",
               "sampling": "user", "tilt_augmentation_deg": 0, "jitter_augmentation_ms": 0, "distill_beta": 0,
               "teacher": None}
SAMPLING = ("user", "sqrt", "window")


def normalize_v6(config):
    v5 = configuration.normalize_config({k: v for k, v in config.items() if k not in V6_DEFAULTS})
    result = {**v5, **V6_DEFAULTS, **{k: config[k] for k in V6_DEFAULTS if k in config}}
    for key in V6_DEFAULTS:
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if result["input_features"] != "body_velocity":
        raise ValueError("v6 models use input_features='body_velocity'")
    return result


def normalize_v7(config):
    channels = config.get("channels")
    # 48 only stands in for the width during the v5 check; the real width is checked below.
    v6 = normalize_v6({**{k: v for k, v in config.items() if k not in V7_DEFAULTS}, "channels": 48})
    result = {**v6, "channels": channels, **V7_DEFAULTS, **{k: config[k] for k in V7_DEFAULTS if k in config}}
    if type(channels) is not int or channels < 2 or (result["mirror_equivariant"] and channels % 2):
        raise ValueError("channels must be an integer of at least 2, and even with mirror_equivariant")
    if type(result["dilation_base"]) is not int or not 2 <= result["dilation_base"] <= 4:
        raise ValueError("dilation_base must be 2, 3 or 4")
    if type(result["separable"]) is not bool:
        raise ValueError("separable must be boolean")
    basis = result["head_basis"]
    if basis is not None and (type(basis) is not int or not 1 <= basis <= result["horizon"]):
        raise ValueError("head_basis must be null or an integer from 1 to horizon")
    if result["speed_feature"] not in ("norm", "l1"):
        raise ValueError("speed_feature must be norm or l1")
    # A streaming graph equals the stateless one only if every input step it sees is in the window.
    if receptive_field(result["num_blocks"], result["dilation_base"]) + 1 > result["effective_context"]:
        raise ValueError("effective_context must be at least the receptive field + 1")
    return result


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
