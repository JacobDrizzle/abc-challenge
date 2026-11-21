"""synthetic_motion.py: made-up head motion for the site, so it can show what BOXRR-23 recordings look like without
publishing any of them (BOXRR-23's data use agreement, clause 4). Used by tools/make_synthetic_data.py.

The model is deliberately simple and is not fitted to anyone's recordings:
  - gaze shifts: the head turns in quick minimum-jerk steps (the smooth profile of a human reach or head turn), with
    random sizes and a pull back towards straight ahead; a little tilt and lean go with each turn;
  - dodges: now and then the head leans sideways and back, like a Beat Saber player ducking a wall;
  - drift: slow wandering (an Ornstein-Uhlenbeck process, smoothly interpolated);
  - noise: a small random jitter on every sample, like tracking noise.
`activity` scales how often and how far the head turns: 0.01 is a head resting in a menu, 1 a calm player, 6 a very
busy one. Angles are turn (+ right), tilt (+ nose up) and lean (+ right ear up), the same three as the Data side's
pictures (dataview.js ttl). The quaternion is head -> world in the model's frame (Unity: +x right, +y up, +z forward):
q = q_y(turn) * q_x(-tilt) * q_z(lean), so ttl(q) gives the three angles back exactly.
"""
import numpy as np
from scipy.interpolate import CubicSpline


# A head turn lasts TURN_S[0] + TURN_S[1] * |size in deg| + up to TURN_S[2] seconds. Set on 20 Nov 2026 from one statistic
# of real public head motion, not from any model error: the velocity autocorrelation of QuestSet's four Beat Saber
# sessions in the sample (CC BY 4.0) is 0.47-0.74 at 100 ms and 0.12-0.43 at 200 ms (means 0.60 and 0.30); these
# values give 0.58-0.72 and 0.18-0.32 at activity 1 to 2.7. The first try, (0.15, 0.005, 0.08), gave 0.38-0.52 and
# about 0: turns that stopped too soon.
TURN_S = (0.25, 0.0075, 0.15)


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


class Steps:
    """value(t) = sum over k of size_k * min_jerk((t - start_k) / duration_k). t must be sorted."""

    def __init__(self, start, duration, size):
        self.start, self.duration, self.size = (np.asarray(v, np.float64) for v in (start, duration, size))

    def __call__(self, t):
        out = np.zeros(len(t))
        if not len(self.start):
            return out
        end = self.start + self.duration
        order = np.argsort(end, kind="stable")
        done = np.r_[0.0, np.cumsum(self.size[order])]
        out += done[np.searchsorted(end[order], t, side="right")]          # steps already finished
        first, stop = np.searchsorted(t, self.start, "left"), np.searchsorted(t, end, "left")
        for k in np.flatnonzero(stop > first):                              # steps under way
            part = slice(first[k], stop[k])
            out[part] += self.size[k] * min_jerk((t[part] - self.start[k]) / self.duration[k])
        return out


class HeadMotion:
    """Continuous head orientation on [t0, t0 + seconds]. angles(t) -> [N, 3] degrees (turn, tilt, lean)."""

    def __init__(self, seconds, activity, rng, t0=0.0, dodges=True, noise_deg=0.02, start=(0.0, -6.0, 0.0)):
        self.noise_deg, self.rng, self.start = noise_deg, rng, np.asarray(start, np.float64)
        root = np.sqrt(activity)
        n = rng.poisson(0.9 * root * seconds) + 1
        onsets = np.sort(rng.uniform(t0, t0 + seconds, n))
        turn, tilt, lean = (np.empty(n) for _ in range(3))
        at = self.start.copy()
        for k in range(n):
            size = rng.gamma(2.0, 6.0 * root)
            toward = rng.random() < 0.5 + 0.45 * np.tanh(abs(at[0]) / 25.0)
            sign = -np.sign(at[0]) if toward and at[0] else rng.choice((-1.0, 1.0))
            turn[k] = sign * size
            tilt[k] = -0.3 * (at[1] - self.start[1]) + rng.normal(0, 0.25 * size + 1.0 * root)
            lean[k] = -0.4 * at[2] + rng.normal(0, 0.06 * size + 0.6 * root)
            at += (turn[k], tilt[k], lean[k])
        duration = TURN_S[0] + TURN_S[1] * np.abs(turn) + rng.uniform(0, TURN_S[2], n)
        lean_start, lean_dur, lean_size = [onsets], [duration], [lean]
        tilt_start, tilt_dur, tilt_size = [onsets], [duration], [tilt]
        if dodges:                                                          # out and back again
            m = rng.poisson(0.06 * root * seconds)
            go = rng.uniform(t0, t0 + seconds, m)
            hold = rng.uniform(0.2, 0.6, m)
            side = rng.choice((-1.0, 1.0), m) * rng.uniform(8, 18, m) * min(1.0, root)
            duck = -rng.uniform(2, 7, m) * min(1.0, root)
            for starts, sizes, store_s, store_d, store_z in ((go, side, lean_start, lean_dur, lean_size),
                                                              (go, duck, tilt_start, tilt_dur, tilt_size)):
                store_s += [starts, starts + 0.25 + hold]
                store_d += [np.full(m, 0.25), np.full(m, 0.3)]
                store_z += [sizes, -sizes]
        self.turn = Steps(onsets, duration, turn)
        self.tilt = Steps(np.concatenate(tilt_start), np.concatenate(tilt_dur), np.concatenate(tilt_size))
        self.lean = Steps(np.concatenate(lean_start), np.concatenate(lean_dur), np.concatenate(lean_size))
        # drift: OU with a 1.5 s time constant on a 10 Hz grid, smoothly interpolated
        grid = np.arange(t0 - 1.0, t0 + seconds + 1.0, 0.1)
        sd = np.array([2.0, 1.2, 0.8]) * min(1.0, root)
        a = np.exp(-0.1 / 1.5)
        walk = np.zeros((len(grid), 3))
        for i in range(1, len(grid)):
            walk[i] = a * walk[i - 1] + rng.normal(0, sd * np.sqrt(1 - a * a))
        self.drift = CubicSpline(grid, walk)

    def angles(self, t, noise=True):
        t = np.asarray(t, np.float64)
        out = np.stack([self.turn(t), self.tilt(t), self.lean(t)], 1) + self.start + self.drift(t)
        if noise and self.noise_deg:
            out += self.rng.normal(0, self.noise_deg, out.shape)
        return out


def quaternions(angles):
    """[N, 3] (turn, tilt, lean) degrees -> [N, 4] unit quaternions (w, x, y, z), head -> world, model frame."""
    half = np.deg2rad(np.asarray(angles, np.float64)) / 2
    cy, sy = np.cos(half[:, 0]), np.sin(half[:, 0])
    cx, sx = np.cos(-half[:, 1]), np.sin(-half[:, 1])
    cz, sz = np.cos(half[:, 2]), np.sin(half[:, 2])
    # q_y * q_x: (cy cx, cy sx, sy cx, -sy sx); then * q_z
    w0, x0, y0, z0 = cy * cx, cy * sx, sy * cx, -sy * sx
    return np.stack([w0 * cz - z0 * sz, x0 * cz + y0 * sz, y0 * cz - x0 * sz, z0 * cz + w0 * sz], 1)


def ttl(q):
    """The inverse, as dataview.js computes it: [N, 4] wxyz -> [N, 3] (turn, tilt, lean) degrees."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.degrees(np.stack([np.arctan2(2 * (x * z + w * y), 1 - 2 * (x * x + y * y)),
                                np.arcsin(np.clip(2 * (y * z - w * x), -1, 1)),
                                np.arctan2(2 * (x * y + w * z), 1 - 2 * (x * x + z * z))], 1))
