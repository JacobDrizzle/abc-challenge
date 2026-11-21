"""Config schema for v10_narrow: v10_speed's schema plus block_inputs (model.py). null is v10_speed's model."""
from experimental.v10_speed.configuration import normalize_config as normalize_speed

V10_DEFAULTS = {"block_inputs": None}


def normalize_config(config):
    speed = normalize_speed({k: v for k, v in config.items() if k not in V10_DEFAULTS})
    result = {**speed, **V10_DEFAULTS, **{k: config[k] for k in V10_DEFAULTS if k in config}}
    inputs, stream_channels = result["block_inputs"], result["channels"] // (2 if result["mirror_equivariant"] else 1)
    if inputs is not None:
        if not (isinstance(inputs, list) and len(inputs) == result["num_blocks"]
                and all(type(c) is int and 1 <= c <= stream_channels for c in inputs)):
            raise ValueError("block_inputs must be null or one integer per block, 1 to the channels of a stream")
        if not result["separable"]:
            raise ValueError("block_inputs applies to separable blocks only")
    return result
