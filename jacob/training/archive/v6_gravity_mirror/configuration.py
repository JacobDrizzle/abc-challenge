"""v5 configuration schema plus the two v6 switches; every other key is validated by tcnn.configuration."""
from tcnn import configuration

V6_DEFAULTS = {"gravity_features": False, "mirror_equivariant": False}


def normalize_config(config):
    v5 = configuration.normalize_config({k: v for k, v in config.items() if k not in V6_DEFAULTS})
    result = {**v5, **V6_DEFAULTS, **{k: config[k] for k in V6_DEFAULTS if k in config}}
    for key in V6_DEFAULTS:
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if result["input_features"] != "body_velocity":
        raise ValueError("v6 models use input_features='body_velocity'")
    return result
