"""Validate experiment options without silently ignoring misspelled settings."""
import math

BASE_KEYS = {"context", "horizon", "channels", "dropout", "batch_size", "steps_per_epoch",
             "epochs", "learning_rate", "weight_decay", "gradient_clip", "patience",
             "minimum_improvement_deg", "validation_stride", "workers", "cache_mb_per_worker", "seeds"}
DEFAULTS = {"effective_context": 400, "velocity_features": False,
            "loss_kind": "quaternion", "endpoint_weight": .5,
            "damped_prior_tau_ms": None, "ema_decay": 0., "data_on_device": False,
            "input_features": "anchor", "num_blocks": 8}


def normalize_config(config):
    if not BASE_KEYS.issubset(config) or set(config) - BASE_KEYS - set(DEFAULTS):
        raise ValueError("Training configuration keys differ from the supported schema")
    result = {**DEFAULTS, **config}
    for key in ("context", "horizon", "channels", "effective_context", "batch_size",
                "steps_per_epoch", "epochs", "patience", "validation_stride"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (result["context"], result["horizon"]) != (400, 20):
        raise ValueError("External contract remains context=400, horizon=20")
    if not 6 <= result["effective_context"] <= 400:
        raise ValueError("effective_context must be between 6 and 400")
    if result["channels"] not in (32, 48):
        raise ValueError("Supported widths are 32 or 48 (below 100,000 parameters)")
    if type(result["velocity_features"]) is not bool:
        raise ValueError("velocity_features must be boolean")
    if result["loss_kind"] not in ("quaternion", "angular"):
        raise ValueError("loss_kind must be quaternion or angular")
    for key in ("learning_rate", "gradient_clip", "minimum_improvement_deg"):
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"Invalid {key}")
    for key in ("dropout", "weight_decay", "endpoint_weight", "cache_mb_per_worker"):
        if not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f"Invalid {key}")
    if result["dropout"] >= 1 or result["endpoint_weight"] > 1:
        raise ValueError("Invalid loss weight or dropout")
    tau = result["damped_prior_tau_ms"]
    if tau is not None and (type(tau) not in (int, float) or not math.isfinite(tau) or tau <= 0):
        raise ValueError("damped_prior_tau_ms must be null or a positive number")
    ema = result["ema_decay"]
    if type(ema) not in (int, float) or not 0 <= ema < 1:
        raise ValueError("ema_decay must be in [0, 1)")
    if type(result["data_on_device"]) is not bool:
        raise ValueError("data_on_device must be boolean")
    if result["input_features"] not in ("anchor", "body_velocity"):
        raise ValueError("input_features must be anchor or body_velocity")
    if type(result["num_blocks"]) is not int or not 1 <= result["num_blocks"] <= 8:
        raise ValueError("num_blocks must be an integer from 1 to 8")
    if result["input_features"] == "body_velocity":
        if result["velocity_features"]:
            raise ValueError("velocity_features applies to anchor inputs only")
        # A streaming graph equals the stateless one only if every input step it sees is in the window.
        if 2 ** (result["num_blocks"] + 1) > result["effective_context"]:
            raise ValueError("body_velocity needs effective_context >= 2**(num_blocks+1) (receptive field + 1)")
    if type(result["workers"]) is not int or result["workers"] < 0:
        raise ValueError("Invalid workers")
    seeds = result["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    return result
