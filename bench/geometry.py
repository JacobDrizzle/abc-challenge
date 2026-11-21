"""Quaternion checks and the challenge metric. Quaternions are (w, x, y, z); all scoring is float64.

unit_quaternions and geodesic_degrees are ported unchanged from ABC_local_test/baseline_pipeline.py
(protocol head-orientation-geodesic-v1), so scores match the team's earlier evaluator exactly.
"""
import numpy as np


def unit_quaternions(values, *, source="quaternions", tolerance=1e-3):
    """Reject invalid rotations; normalise only small numerical norm deviations."""
    q = np.asarray(values, dtype=np.float64)
    if q.ndim < 1 or q.shape[-1] != 4 or q.size == 0:
        raise ValueError(f"{source}: expected a nonempty array ending in 4, got {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError(f"{source}: NaN or infinity")
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError(f"{source}: zero/near-zero quaternion")
    deviation = float(np.max(np.abs(norms - 1)))
    if deviation > tolerance:
        raise ValueError(f"{source}: quaternion norm deviation {deviation:.6g} exceeds {tolerance}")
    return q / norms


def renormalise(values, *, source="prediction"):
    """What the harness does to a model's output before scoring: divide by the norm (any length is fine)."""
    q = np.asarray(values, dtype=np.float64)
    if not np.isfinite(q).all():
        raise ValueError(f"{source}: NaN or infinity (the harness rejects non-finite output)")
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError(f"{source}: zero/near-zero quaternion cannot be re-normalised")
    return q / norms


def geodesic_degrees(predicted, target):
    """Sign-invariant SO(3) angle 2 arccos|<p, q>|, in degrees; not quaternion MSE or RMS."""
    a = unit_quaternions(predicted, source="prediction")
    b = unit_quaternions(target, source="target")
    if a.shape != b.shape:
        raise ValueError(f"Prediction/target shapes differ: {a.shape} vs {b.shape}")
    dot = np.abs(np.sum(a * b, axis=-1))
    return np.rad2deg(2 * np.arccos(np.clip(dot, 0, 1)))


def axis_angle(axis, degrees):
    """Unit quaternion for a rotation of `degrees` about `axis` (used by tests and synthetic streams)."""
    axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    half = np.deg2rad(degrees) / 2
    return np.concatenate(([np.cos(half)], np.sin(half) * axis))


def qmul(a, b):
    """Hamilton product a * b of [..., 4] arrays."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a, np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, np.float64), -1, 0)
    return np.stack((aw*bw - ax*bx - ay*by - az*bz, aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw), -1)


def synthetic_stream(length, seed=0):
    """Smooth random head motion at 100 Hz: integrated, slowly decaying random angular velocity.

    Ported from ABC_local_test/benchmark_latency.py. Used where no data is needed (audit, MACs,
    determinism, latency), so those numbers are identical on every machine.
    """
    rng = np.random.default_rng(seed)
    velocity = np.cumsum(rng.normal(0, .02, (length, 3)), 0) * .98 ** np.arange(length)[:, None]
    q = np.empty((length, 4), np.float64)
    q[0] = (1, 0, 0, 0)
    for t in range(1, length):
        angle = np.linalg.norm(velocity[t]) / 100
        axis = velocity[t] / max(np.linalg.norm(velocity[t]), 1e-12)
        q[t] = qmul(np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis], q[t - 1])
        q[t] /= np.linalg.norm(q[t])
    return q.astype(np.float32)


def random_motion(length, seed=0, persistence=0.99, noise_rad_s=0.15):
    """Head-like motion that keeps moving: angular velocity is a mean-reverting random walk with a
    stationary spread of about 1 rad/s (60 deg/s). synthetic_stream above decays to stillness after a few
    hundred samples, which is fine for timing but too easy for parity and determinism checks."""
    rng = np.random.default_rng(seed)
    q = np.empty((length, 4), np.float64)
    q[0], velocity = axis_angle(rng.normal(size=3), rng.uniform(0, 180)), np.zeros(3)
    for t in range(1, length):
        velocity = persistence * velocity + rng.normal(0, noise_rad_s, 3)
        angle = np.linalg.norm(velocity) / 100
        axis = velocity / max(np.linalg.norm(velocity), 1e-12)
        q[t] = qmul(np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis], q[t - 1])
        q[t] /= np.linalg.norm(q[t])  # small steps from q[t-1] keep consecutive signs continuous
    return q.astype(np.float32)
