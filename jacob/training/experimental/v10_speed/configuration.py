"""Config schema for v10_speed: best's schema plus two switches, both on by default.

last_position_forward: true trains and validates with model.LastPositionTCNN's forward (only the positions the last
  position reads); false uses LeanTCNN.forward, the reference.
dense_validation: true scores validation with one pass along each recording (validation.py); false cuts one window
  per forecast, as train_tcnn does.
With both false, training is best's: same batches, same forward, same validation.
"""
from best.configuration import normalize_config as normalize_best

V10_DEFAULTS = {"last_position_forward": True, "dense_validation": True}


def normalize_config(config):
    best = normalize_best({k: v for k, v in config.items() if k not in V10_DEFAULTS})
    result = {**best, **V10_DEFAULTS, **{k: config[k] for k in V10_DEFAULTS if k in config}}
    for key in V10_DEFAULTS:
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    return result
