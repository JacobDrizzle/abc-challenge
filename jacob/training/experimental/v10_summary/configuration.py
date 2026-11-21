"""Config schema for v10_summary: best's schema plus summary_taus (model.py). null is best's model, LeanTCNN."""
import math

from best.configuration import normalize_config as normalize_best

V10_DEFAULTS = {"summary_taus": None}


def normalize_config(config):
    best = normalize_best({k: v for k, v in config.items() if k not in V10_DEFAULTS})
    result = {**best, **V10_DEFAULTS, **{k: config[k] for k in V10_DEFAULTS if k in config}}
    taus = result["summary_taus"]
    if taus is not None:
        if not (isinstance(taus, list) and taus
                and all(type(t) in (int, float) and math.isfinite(t) and t > 0 for t in taus)):
            raise ValueError("summary_taus must be null or a non-empty list of positive time constants in samples")
        if not result["separable"]:
            raise ValueError("summary_taus applies to separable blocks only")
    return result
