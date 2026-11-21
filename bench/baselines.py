"""Fixed, parameter-free baselines. Settings are constants, never tuned on a benchmark.

Ported from ABC_local_test/baseline_pipeline.py (HoldLast, ConstantVelocity) and tcnn/damped.py
(DampedVelocity; lag 1 and tau 100 ms were chosen on Jacob's validation users only, see
outputs/tcnn_v5/damped_baseline/selection.json). Same maths, so the scores reproduce exactly.
Each predict_batch takes histories [N, C, 4] and returns [N, H, 4] (w, x, y, z).
"""
import numpy as np
from scipy.spatial.transform import Rotation

from bench.geometry import unit_quaternions


class HoldLast:
    name = "hold_last"
    stateful = False
    description = "Repeat the latest orientation"

    def __init__(self, horizon=20, rate_hz=100.0):
        self.horizon, self.rate_hz = horizon, rate_hz

    def predict_batch(self, history):
        return np.repeat(history[:, -1:, :], self.horizon, axis=1)


class ConstantVelocity(HoldLast):
    """D = R_now R_past^-1; prediction(h) = Exp((h/lag) Log(D)) R_now, with h and lag in samples."""
    description = "Shortest relative-rotation extrapolation on SO(3), lag 1 sample"

    def __init__(self, horizon=20, rate_hz=100.0, lag=1):
        super().__init__(horizon, rate_hz)
        self.lag = lag
        self.name = f"constant_velocity_{lag}step"

    def predict_batch(self, history):
        now = Rotation.from_quat(history[:, -1, [1, 2, 3, 0]])
        past = Rotation.from_quat(history[:, -1 - self.lag, [1, 2, 3, 0]])
        delta = (now * past.inv()).as_rotvec()
        scale = np.arange(1, self.horizon + 1) / self.lag
        step = Rotation.from_rotvec((delta[:, None, :] * scale[None, :, None]).reshape(-1, 3))
        repeated_now = Rotation.from_quat(np.repeat(now.as_quat(), self.horizon, axis=0))
        xyzw = (step * repeated_now).as_quat().reshape(len(history), self.horizon, 4)
        return xyzw[..., [3, 0, 1, 2]]


class DampedVelocity(HoldLast):
    """Angular velocity over the last lag samples, decaying as exp(-t/tau) and integrated forward."""
    name = "damped_velocity"
    description = "Integrated exponentially decaying angular velocity on SO(3), lag 1, tau 100 ms"

    def __init__(self, horizon=20, rate_hz=100.0, lag=1, tau_ms=100):
        super().__init__(horizon, rate_hz)
        self.lag, self.tau_ms = lag, tau_ms

    def predict_batch(self, history):
        endpoints = unit_quaternions(history[:, [-1, -1 - self.lag]])
        now = Rotation.from_quat(endpoints[:, 0, [1, 2, 3, 0]])
        past = Rotation.from_quat(endpoints[:, 1, [1, 2, 3, 0]])
        velocity = (now * past.inv()).as_rotvec() / self.lag
        tau = self.tau_ms * self.rate_hz / 1000
        elapsed = -tau * np.expm1(-np.arange(1, self.horizon + 1, dtype=np.float64) / tau)
        turns = Rotation.from_rotvec((velocity[:, None] * elapsed[None, :, None]).reshape(-1, 3))
        initial = Rotation.from_quat(np.repeat(now.as_quat(), len(elapsed), axis=0))
        result = (turns * initial).as_quat().reshape(len(history), len(elapsed), 4)
        return result[..., [3, 0, 1, 2]]


BASELINES = {"hold_last": HoldLast, "constant_velocity": ConstantVelocity, "damped_velocity": DampedVelocity}


def make_baseline(name, horizon=20, rate_hz=100.0):
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline {name!r}; choose from {', '.join(BASELINES)}")
    return BASELINES[name](horizon, rate_hz)
