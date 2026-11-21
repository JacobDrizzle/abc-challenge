"""Frozen damped angular-velocity predictor on SO(3)."""
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from baseline_pipeline import HoldLast, sha256_file, unit_quaternions


class DampedVelocity(HoldLast):
    name = "damped_velocity"

    def __init__(self, config, lag=1, tau_ms=100):
        super().__init__(config)
        if type(lag) is not int or not 1 <= lag < config.context:
            raise ValueError("Invalid velocity lag")
        if tau_ms is not None and (not np.isfinite(tau_ms) or tau_ms <= 0):
            raise ValueError("tau_ms must be positive or null for undamped velocity")
        self.lag, self.tau_ms = lag, tau_ms
        self.metadata = {"trainable_parameters": 0, "lag_samples": lag, "tau_ms": tau_ms,
                         "method": "Integrated exponentially decaying angular velocity on SO(3)"}

    def predict_steps(self, history, steps):
        endpoints = unit_quaternions(history[:, [-1, -1-self.lag]])
        now = Rotation.from_quat(endpoints[:, 0, [1, 2, 3, 0]])
        past = Rotation.from_quat(endpoints[:, 1, [1, 2, 3, 0]])
        velocity = (now * past.inv()).as_rotvec() / self.lag
        elapsed = np.asarray(steps, dtype=np.float64)
        if self.tau_ms is not None:
            tau = self.tau_ms * self.config.rate_hz / 1000
            elapsed = -tau * np.expm1(-elapsed/tau)
        turns = Rotation.from_rotvec((velocity[:, None] * elapsed[None, :, None]).reshape(-1, 3))
        initial = Rotation.from_quat(np.repeat(now.as_quat(), len(elapsed), axis=0))
        result = (turns * initial).as_quat().reshape(len(history), len(elapsed), 4)
        return result[..., [3, 0, 1, 2]]

    def predict_batch(self, history):
        return self.predict_steps(history, np.arange(1, self.config.horizon+1))


def build_model(config):
    path = os.environ.get("ABC_DAMPED_SELECTION")
    if not path:
        raise ValueError("ABC_DAMPED_SELECTION must name a frozen validation selection")
    selection = json.loads(Path(path).read_text(encoding="utf-8"))
    if selection["split"] != "validation" or selection["config"]["context"] != config.context or selection["config"]["horizon"] != config.horizon or selection["config"]["rate_hz"] != config.rate_hz:
        raise ValueError("Damped baseline selection has a different evaluation protocol")
    model = DampedVelocity(config, **selection["selected"])
    model.metadata.update(selection_sha256=sha256_file(path), selection=selection["selected"],
                          tuned_on="validation", data_sha256=selection["data_sha256"])
    return model
