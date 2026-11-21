"""v6 configuration schema plus the v7 options; every other key is validated by the v6 and v5 schemas.

Two v5 rules are replaced here rather than inherited: v5 allows only channels 32 or 48, and checks the
receptive field for base-2 dilations only.
"""
from experimental.v6_gravity_mirror.configuration import normalize_config as normalize_v6
from experimental.v7_compact.model import receptive_field

V7_DEFAULTS = {"dilation_base": 2, "separable": False, "head_basis": None, "speed_feature": "norm"}


def normalize_config(config):
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
