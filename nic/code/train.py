"""Train a small GRU to forecast the next 20 head quaternions, then measure how well it generalises.

Needs only torch, numpy, matplotlib and windows.py (same folder). Runs on CPU or GPU.

  python train.py --data DIR [DIR ...] --out outputs/run1            # one train/val/test run
  python train.py --data DIR --cv-folds 5 --out outputs/cv           # leave-groups-of-users-out
  python train.py --selftest                                         # maths checks, no data needed

What it does, in plain words
  1. Loads every player's clean segments (.npz made by build_dataset.py: q float32 [N,4] w,x,y,z).
  2. Splits PLAYERS (not windows) into train / val / test from a hash of the player id (windows.user_split).
  3. Trains a GRU on train-player windows. Validation players decide when to stop. Test players are used once.
  4. On the test players it scores the model and three simple baselines (hold the last pose, constant velocity,
     damped constant velocity) with the challenge metric: angle in degrees at step 20, plus every other step.
  5. Generalisation numbers: error on unseen players, the train-vs-test gap, error per player, how many players
     the model beats the baseline on, and confidence intervals that resample PLAYERS (not windows).
  6. Writes metrics.json, per_window.npz and four figures into --out.

Choices that belong to the TEAM are flags (each default is a placeholder, not a recommendation).
"""
import argparse
import copy
import glob
import json
import math
import os
import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from windows import CONTEXT, HORIZON, get_window, make_index, user_split  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HZ = 100.0
CTX_LEN = [CONTEXT]   # how many of the 400 context readings the model sees (set by --context)


# ------------------------------------------------------------------------------------------------
# Quaternion maths (torch), scalar-first (w,x,y,z), Hamilton product
# ------------------------------------------------------------------------------------------------
def qconj(q):
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def _qmul_table():
    """Constant [4,16]: a @ T, reshaped to 4x4, is the transposed 'left-multiply by a' matrix, so qmul is two MatMuls."""
    M = [[(0, 1), (1, -1), (2, -1), (3, -1)],      # row i of L(a): (index of a, sign) for each column j
         [(1, 1), (0, 1), (3, -1), (2, 1)],
         [(2, 1), (3, 1), (0, 1), (1, -1)],
         [(3, 1), (2, -1), (1, 1), (0, 1)]]
    T = torch.zeros(4, 16)
    for i in range(4):
        for j in range(4):
            k, sgn = M[i][j]
            T[k, 4 * j + i] = sgn                  # transposed: entry (j, i)
    return T


QMUL_T = _qmul_table()


EXPORT_SAFE = [False]   # True for ONNX export: GADS-T max-pool and attention written with safe-list operators only (same results)
QMUL_MATRIX = [False]   # True for ONNX export (MatMul form); training uses the element-wise form (same maths, faster on GPU)


def qmul(a, b):
    """Hamilton product a*b. For export it is written as MatMul/Reshape only (both on the NPU safe list): b (as a row)
    times L(a)^T. The selftest checks both forms agree."""
    if not QMUL_MATRIX[0]:
        aw, ax, ay, az = a[..., 0:1], a[..., 1:2], a[..., 2:3], a[..., 3:4]
        bw, bx, by, bz = b[..., 0:1], b[..., 1:2], b[..., 2:3], b[..., 3:4]
        return torch.cat([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                          aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], dim=-1)
    shape = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    Lt = (a @ QMUL_T.to(a)).reshape(*a.shape[:-1], 4, 4)
    return (b.reshape(*b.shape[:-1], 1, 4) @ Lt).reshape(*shape, 4)


def angle(p, q):
    """Geodesic angle in radians between rotations p and q. Sign-safe, stable for tiny angles."""
    r = qmul(qconj(p), q)
    return 2.0 * torch.atan2(torch.linalg.vector_norm(r[..., 1:], dim=-1), r[..., 0].abs())


def to_rel(q, last, frame):
    return qmul(qconj(last), q) if frame == "body" else qmul(q, qconj(last))


def from_rel(d, last, frame):
    return qmul(last, d) if frame == "body" else qmul(d, last)


def qlog(q):
    """Unit quaternion -> rotation vector (3 numbers, radians)."""
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    n = v.norm(dim=-1, keepdim=True)
    a = 2.0 * torch.atan2(n, q[..., :1])
    return v * torch.where(n > 1e-9, a / n.clamp_min(1e-12), torch.full_like(n, 2.0))


def qexp(r):
    """Rotation vector -> unit quaternion."""
    a = r.norm(dim=-1, keepdim=True)
    s = torch.where(a > 1e-9, torch.sin(a / 2) / a.clamp_min(1e-12), torch.full_like(a, 0.5))
    return torch.cat([torch.cos(a / 2), r * s], dim=-1)


# ------------------------------------------------------------------------------------------------
# Baselines: predict the next 20 absolute quaternions from the context, no learning
# ------------------------------------------------------------------------------------------------
def hold_predict(ctx):
    return ctx[:, -1:].expand(-1, HORIZON, -1)


def cv_predict(ctx, k=5, damp=1.0):
    """Constant angular velocity from the last k steps. damp<1 shrinks the speed by that factor each step
    (total turn after h steps = speed * (d + d^2 + ... + d^h)); damp=1 is plain constant velocity."""
    c = ctx.double()
    rel = qmul(qconj(c[:, -k - 1:-1]), c[:, -k:])                    # body-frame turn per step, last k steps
    vel = qlog(rel).mean(1)                                           # [B,3] radians per step
    j = torch.arange(1, HORIZON + 1, dtype=torch.float64, device=ctx.device)
    s = j if damp == 1.0 else torch.cumsum(damp ** j, 0)
    return qmul(c[:, -1:], qexp(vel[:, None, :] * s[None, :, None])).to(ctx.dtype)


BASELINES = {
    "hold": lambda c: hold_predict(c),
    "const_vel": lambda c: cv_predict(c, k=5, damp=1.0),
    "damped_cv": lambda c: cv_predict(c, k=1, damp=0.9),     # k and damp were tuned on our first 11 players
}


# ------------------------------------------------------------------------------------------------
# Data: players -> segments -> windows
# ------------------------------------------------------------------------------------------------
def load_users(dirs, max_users=None):
    users = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(f)
            q, off = z["q"], z["offsets"]
            users[os.path.basename(f)[:-4]] = [q[a:b] for a, b in zip(off[:-1], off[1:])]
    if max_users:
        users = dict(list(users.items())[:max_users])
    return users


def warp_window(segments, seg, start, f):
    """The window at (seg, start) played at speed f (f>1 = the head moves faster): the 420 readings are re-sampled at
    start + f*i from the recording by interpolating neighbouring quaternions (nlerp). f=1 gives the normal window.
    If the faster window would run past the end of the recording, it is moved earlier."""
    q = segments[seg]
    n = CONTEXT + HORIZON
    if f * (n - 1) > len(q) - 1:
        f = 1.0
    start = min(float(start), len(q) - 1 - f * (n - 1))
    pos = start + f * np.arange(n)
    i = np.minimum(np.floor(pos).astype(int), len(q) - 2)
    t = (pos - i)[:, None].astype(np.float32)
    a, b = q[i], q[i + 1]
    b = np.where((a * b).sum(-1, keepdims=True) < 0, -b, b)           # same hemisphere before blending
    w = (1 - t) * a + t * b
    w = (w / np.linalg.norm(w, axis=-1, keepdims=True)).astype(np.float32)
    return w[:CONTEXT], w[CONTEXT:]


class Windows:
    """All windows of a group of players. rows = (segment, start); user_of_row says whose window it is."""

    def __init__(self, users, uids, stride):
        self.uids, self.segs, rows, who = list(uids), [], [], []
        for u, uid in enumerate(self.uids):
            base = len(self.segs)
            self.segs += users[uid]
            idx = make_index(users[uid], stride=stride)
            rows.append(idx + np.array([base, 0]))
            who.append(np.full(len(idx), u))
        self.rows = np.concatenate(rows) if rows else np.zeros((0, 2), np.int64)
        self.user_of_row = np.concatenate(who) if who else np.zeros(0, int)

    def __len__(self):
        return len(self.rows)

    warp, rng = 0.0, None        # set warp > 0 (and rng) on the TRAINING windows for time-warp augmentation

    def batch(self, ids):
        if self.warp > 0:
            fs = np.exp(self.rng.uniform(-np.log1p(self.warp), np.log1p(self.warp), len(ids)))   # speeds in [1/(1+a), 1+a]
            xs, ys = zip(*(warp_window(self.segs, *self.rows[i], f) for i, f in zip(ids, fs)))
        else:
            xs, ys = zip(*(get_window(self.segs, *self.rows[i]) for i in ids))
        return torch.from_numpy(np.stack(xs)).to(DEV), torch.from_numpy(np.stack(ys)).to(DEV)

    def sampler(self, rng, balance):
        if not balance:
            return lambda b: rng.integers(0, len(self), b)
        w = 1.0 / np.bincount(self.user_of_row)[self.user_of_row]
        return lambda b: rng.choice(len(self), b, p=w / w.sum())


# ------------------------------------------------------------------------------------------------
# Input features: extra columns worked out from the 4 rotation numbers (the challenge only gives us those)
# ------------------------------------------------------------------------------------------------
FEATURE_GROUPS = {"vel": 3, "acc": 3, "speed": 1, "mag": 2}      # extra columns each group adds to the 4 rotation numbers
SMALL_ANGLE = [False]  # True: velocity = 2*(x,y,z) of the step quaternion (no Atan; error ~angle^2/24, about 1e-4 at 100 Hz). Used for export
FEAT_DTYPE = [torch.float64]   # precision of the feature maths; onnx_speed.py switches it to float32 (ONNX Runtime has no double Atan)
ACC_GAP = 5          # acceleration = change in turning speed over this many readings (50 ms); 1 reading is too jittery


def feature_groups(mode):
    """'base' -> (); 'vel+acc' -> ('vel', 'acc'); old name 'velacc' = 'vel+acc'. Groups are always used in the order of FEATURE_GROUPS."""
    names = {"base": [], "velacc": ["vel", "acc"]}.get(mode) if mode in ("base", "velacc") else mode.split("+")
    names = [n for n in names if n != "noq"]
    if "noq" in mode.split("+") and not names:
        raise ValueError("noq (no quaternion columns) needs at least one other group")
    bad = [n for n in names if n not in FEATURE_GROUPS]
    if bad or len(set(names)) != len(names):
        raise ValueError(f"unknown or repeated feature group in {mode!r}; use base or a + list of {list(FEATURE_GROUPS)}")
    return tuple(g for g in FEATURE_GROUPS if g in names)


def uses_quat(mode):
    """False when the input set contains 'noq': the model then sees only motion (velocity etc.), not the 4 rotation numbers.
    Such a model can run one sample at a time (stateful), because velocity does not depend on the newest reading."""
    return "noq" not in mode.split("+")


def feature_dim(mode):
    return 4 * uses_quat(mode) + sum(FEATURE_GROUPS[g] for g in feature_groups(mode))


def featurize(rel, mode):
    """rel [B,T,4] (relative to the last reading) -> [B,T,feature_dim]. Extra columns, each a group (all worked out from rel only):
    vel    turning velocity: 3 numbers per reading (rad/s about x,y,z, measured in the head's own frame)
    acc    acceleration: 3 numbers (rad/s^2 / 10, change of velocity over ACC_GAP readings)
    speed  turning speed as one number (length of the velocity)
    mag    two numbers: speed and acceleration length
    The first readings (no earlier reading to compare with) get zeros."""
    groups = feature_groups(mode)
    if not groups:
        return rel
    q = rel.to(FEAT_DTYPE[0])
    step = qmul(qconj(q[:, :-1]), q[:, 1:])                         # the small turn between consecutive readings
    rv = 2.0 * step[..., 1:] if SMALL_ANGLE[0] else qlog(step)      # rotation vector of the step (small-angle form has no Atan)
    v = torch.cat([rv[:, :1] * 0, rv * HZ], dim=1)                   # [B,T,3] rad/s, zero for the first reading (Concat, not Pad)
    k = ACC_GAP
    a = torch.cat([v[:, :k] * 0, (v[:, k:] + v[:, :-k] * -1.0) * (HZ / k * 0.1)], dim=1)   # [B,T,3], change over k readings, x0.1
    cols = {"vel": lambda: v, "acc": lambda: a, "speed": lambda: v.norm(dim=-1, keepdim=True),
            "mag": lambda: torch.cat([v.norm(dim=-1, keepdim=True), a.norm(dim=-1, keepdim=True)], dim=-1)}
    extra = [cols[g]().to(rel.dtype) for g in groups]
    return torch.cat([rel, *extra] if uses_quat(mode) else extra, dim=-1)


# ------------------------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------------------------
def head_output(model, raw):
    """(raw, unit) for training; just raw when model.export_raw is set (no normalisation ops in the exported graph)."""
    if getattr(model, "export_raw", False):
        return raw
    return raw, F.normalize(raw, dim=-1, eps=1e-8)


def hold_init(head):
    """Output layer starts at 'no change' (identity quaternion for all 20 steps), so an untrained model equals hold."""
    nn.init.zeros_(head.weight)
    with torch.no_grad():
        head.bias.copy_(torch.tensor([1.0, 0, 0, 0]).repeat(HORIZON))


class GRUForecaster(nn.Module):
    """GRU over the context (relative to its last reading) -> 20 relative quaternions.
    The output layer starts at 'no change', so an untrained model equals the hold-last baseline."""

    def __init__(self, hidden=32, mode="base"):
        super().__init__()
        self.mode = mode
        self.gru = nn.GRU(feature_dim(mode), hidden, batch_first=True)
        self.head = nn.Linear(hidden, 4 * HORIZON)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([1.0, 0, 0, 0]).repeat(HORIZON))

    def forward(self, rel_ctx):
        _, h = self.gru(featurize(rel_ctx, self.mode))
        raw = self.head(h.reshape(h.shape[1], -1)).view(-1, HORIZON, 4)
        return head_output(self, raw)


def safe_max(h):
    """Max over the LAST axis using only Slice, Add, Mul, ReLU, Concat and Reshape (all on the NPU safe list), because
    max(a, b) = a + relu(b - a). Pairs are combined in a tree (20 -> 10 -> 5 -> 3 -> 2 -> 1), so it is only 5 levels deep."""
    while h.shape[-1] > 1:
        n = h.shape[-1]
        a, b = h[..., 0:n - n % 2:2], h[..., 1:n - n % 2:2]
        m = a + F.relu(b + a * -1.0)
        h = torch.cat([m, h[..., n - 1:n]], dim=-1) if n % 2 else m
    return h.reshape(h.shape[:-1])                                     # drop the size-1 axis (Reshape, not Squeeze)


class GADSForecaster(nn.Module):
    """Our adaptation of the GADS idea (Grouped Attention Deep Sets) to forecasting; the paper itself is face-landmark
    pose ESTIMATION with no time axis. All readings are processed in parallel, no loop over time:
      1. the last groups*size readings are cut into `groups` time slices of `size` readings
      2. per slice, a Deep Set: every reading -> small MLP (+ a sinusoidal code for its place in the slice), MAX-pool, MLP
      3. a sinusoidal code for the slice's place in the 4 s window is added (newest slice last)
      4. multi-head self-attention across the slices, then the newest slice's vector -> MLP -> 20 quaternions
    Without the position codes (pos=False) the model cannot tell earlier from later readings. The output layer starts at
    'no change', so an untrained model equals the hold-last baseline, like the GRU."""

    def __init__(self, dim=32, mode="base", groups=20, heads=4, pool="max", pos=True):
        super().__init__()
        self.mode, self.groups, self.pool, self.pos, self.dim = mode, groups, pool, pos, dim
        d_in = feature_dim(mode)
        self.enc1 = nn.Linear(d_in, dim)
        self.enc2 = nn.Linear(dim, dim)
        self.dec = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, 4 * HORIZON)
        hold_init(self.head)
        size = max(1, CTX_LEN[0] // groups)
        self.register_buffer("pe_in", self._sincos(size, dim), persistent=False)       # place inside a slice
        self.register_buffer("pe_group", self._sincos(groups, dim), persistent=False)  # place of the slice

    def _attend_last(self, z):
        """Same maths as self.attn for the newest slice's query, written as plain MatMul -> Softmax -> MatMul with the same
        weights (the challenge page says fused attention may not be supported; this form is)."""
        B, G, D = z.shape
        H = self.attn.num_heads
        dh = D // H
        W, b = self.attn.in_proj_weight, self.attn.in_proj_bias
        q = z[:, -1:] @ W[:D].T + b[:D]                                  # [B,1,D]
        k = z @ W[D:2 * D].T + b[D:2 * D]                                # [B,G,D]
        v = z @ W[2 * D:].T + b[2 * D:]
        q = q.reshape(B, 1, H, dh).transpose(1, 2)                       # [B,H,1,dh]
        k = k.reshape(B, G, H, dh).transpose(1, 2)                       # [B,H,G,dh]
        v = v.reshape(B, G, H, dh).transpose(1, 2)
        w = torch.softmax((q @ k.transpose(2, 3)) * (1.0 / math.sqrt(dh)), dim=-1)   # [B,H,1,G]
        o = (w @ v).transpose(1, 2).reshape(B, 1, D)
        return self.attn.out_proj(o)

    @staticmethod
    def _sincos(n, dim):
        p = torch.arange(n, dtype=torch.float32)[:, None]
        f = torch.exp(-math.log(10000.0) * torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        pe = torch.zeros(n, dim)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(p * f), torch.cos(p * f)
        return pe

    def forward(self, rel_ctx):
        x = featurize(rel_ctx, self.mode)                               # [B,T,d_in]
        B, T, _ = x.shape
        size = T // self.groups
        assert size >= 1, f"{T} readings cannot make {self.groups} groups"
        x = x[:, T - size * self.groups:].reshape(B, self.groups, size, -1)      # [B,G,S,d_in], newest slice last
        h = self.enc1(x)
        if self.pos:
            h = h + (self.pe_in if self.pe_in.shape[0] == size else self._sincos(size, self.dim).to(h))   # place inside its slice
        h = self.enc2(F.relu(h))
        if self.pool == "max":
            h = h.transpose(2, 3).contiguous() if EXPORT_SAFE[0] else h     # [B,G,dim,S] so the tree works on the last axis
            h = safe_max(h) if EXPORT_SAFE[0] else h.amax(2)          # Deep Set pooling: ignores order
        else:
            h = h.mean(2)
        z = self.dec(F.relu(h))                                         # [B,G,dim], one vector per slice
        if self.pos:
            z = z + self.pe_group                                       # place of the slice in the window
        if EXPORT_SAFE[0]:
            last = self.norm(z[:, -1:] + self._attend_last(z))            # only the newest slice is read out, so only its query is needed
        else:
            last = self.norm(z + self.attn(z, z, z, need_weights=False)[0])[:, -1:]
        raw = self.head(F.relu(self.mlp(last.reshape(z.shape[0], -1)))).view(-1, HORIZON, 4)
        return head_output(self, raw)


class TCNForecaster(nn.Module):
    """Convolution-only model (every layer is a Conv1d, a first-class operator on XR NPUs). Depthwise-separable blocks:
    a depthwise conv (each channel looks along time on its own) then a 1x1 'pointwise' conv (channels mix), ReLU, residual.
      kind='dilated': causal stack with dilation 1,2,4,...,64 and kernel 3 (sees the last 255 readings); no padding, the
                      sequence just gets shorter, and the newest output step is used.
      kind='strided': every block halves the sequence (kernel 4, stride 2) until about 1 step is left; far fewer MACs.
    Output layer starts at 'no change', like the other models."""

    def __init__(self, ch=24, mode="base", kind="dilated"):
        super().__init__()
        self.mode, self.kind = mode, kind
        self.inp = nn.Conv1d(feature_dim(mode), ch, 1)
        n = 7
        if kind == "dilated":
            self.dw = nn.ModuleList(nn.Conv1d(ch, ch, 3, dilation=2 ** i, groups=ch) for i in range(n))
        else:
            self.dw = nn.ModuleList(nn.Conv1d(ch, ch, 4, stride=2, groups=ch) for i in range(n))
        self.pw = nn.ModuleList(nn.Conv1d(ch, ch, 1) for _ in range(n))
        self.mlp = nn.Linear(ch, ch)
        self.head = nn.Linear(ch, 4 * HORIZON)
        hold_init(self.head)

    def forward(self, rel_ctx):
        x = self.inp(featurize(rel_ctx, self.mode).transpose(1, 2))       # [B,ch,T]
        for dw, pw in zip(self.dw, self.pw):
            y = pw(F.relu(dw(x)))
            if self.kind == "dilated":
                x = F.relu(x[:, :, x.shape[2] - y.shape[2]:] + y)         # residual: keep the newest part of the input
            else:
                x = F.relu(y)
        raw = self.head(F.relu(self.mlp(x[:, :, -1:].reshape(x.shape[0], -1)))).view(-1, HORIZON, 4)
        return head_output(self, raw)


def build_model(a):
    if a.model == "tcn":
        return TCNForecaster(a.dim, a.features, a.tcn_kind)
    if a.model == "gads":
        return GADSForecaster(a.dim, a.features, a.groups, a.heads, a.pool, not a.no_pos)
    return GRUForecaster(a.hidden, a.features)


def prepare(ctx, tgt, frame):
    ctx = ctx[:, -CTX_LEN[0]:]
    last = ctx[:, -1:]
    return to_rel(ctx, last, frame), to_rel(tgt, last, frame), last


def loss_fn(raw, unit, rel_tgt, loss_on, norm_penalty=0.01):
    ang = angle(unit, rel_tgt)                                          # [B,20] radians
    main = ang[:, -1].mean() if loss_on == "last" else ang.mean()
    return main + norm_penalty * ((raw.pow(2).sum(-1) - 1) ** 2).mean(), ang


# ------------------------------------------------------------------------------------------------
# Evaluation: per-window angle at every step, for the model and each baseline
# ------------------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, data, frame, max_windows=None):
    """dict: method -> degrees [N,20]; plus 'speed' (deg/s over the last 0.2 s) and 'user' (row -> player)."""
    model.eval()
    n = len(data) if not max_windows else min(len(data), max_windows)
    ids = np.linspace(0, len(data) - 1, n).astype(int)
    out = {m: [] for m in ["model", *BASELINES]}
    speed = []
    for k in range(0, n, 512):
        ctx, tgt = data.batch(ids[k:k + 512])
        rel_ctx, _, last = prepare(ctx, tgt, frame)
        _, unit = model(rel_ctx)
        out["model"].append(angle(from_rel(unit, last, frame), tgt))
        for name, fn in BASELINES.items():
            out[name].append(angle(fn(ctx), tgt))
        speed.append(torch.rad2deg(angle(ctx[:, -21], ctx[:, -1])) / 0.2)
    res = {m: torch.rad2deg(torch.cat(v)).cpu().numpy() for m, v in out.items()}
    res["speed"] = torch.cat(speed).cpu().numpy()
    res["user"] = data.user_of_row[ids]
    return res


def bootstrap_users(per_user_sum, per_user_n, n_boot=2000, seed=0):
    """95% interval for the pooled mean when we resample PLAYERS with replacement."""
    rng = np.random.default_rng(seed)
    u = len(per_user_n)
    idx = rng.integers(0, u, (n_boot, u))
    means = per_user_sum[idx].sum(1) / per_user_n[idx].sum(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarise(res, uids):
    """Headline and generalisation numbers from per-window errors (test players)."""
    methods = ["model", *BASELINES]
    user = res["user"]
    nu = len(uids)
    n_u = np.bincount(user, minlength=nu).astype(float)
    keep = n_u > 0
    s = {"n_windows": int(len(user)), "n_players": int(keep.sum()), "methods": {}}
    for m in methods:
        e20 = res[m][:, -1]
        sum_u = np.bincount(user, weights=e20, minlength=nu)
        s["methods"][m] = {
            "mean_step20": float(e20.mean()), "median_step20": float(np.median(e20)),
            "p90_step20": float(np.percentile(e20, 90)),
            "mean_per_step": [float(x) for x in res[m].mean(0)],
            "ci95_step20_players": bootstrap_users(sum_u[keep], n_u[keep]),
            "per_player_mean_step20": {uids[i]: float(sum_u[i] / n_u[i]) for i in np.flatnonzero(keep)},
        }
    # paired comparison with each baseline: model minus baseline, per player (negative = model better)
    for b in BASELINES:
        d = res["model"][:, -1] - res[b][:, -1]
        sum_u = np.bincount(user, weights=d, minlength=nu)
        per_player = sum_u[keep] / n_u[keep]
        s["methods"]["model"][f"vs_{b}"] = {
            "mean_diff_step20": float(d.mean()), "ci95_players": bootstrap_users(sum_u[keep], n_u[keep]),
            "players_model_better": int((per_player < 0).sum()), "players": int(keep.sum())}
    bins = [0, 10, 30, 60, 1e9]
    s["speed_bins_deg_per_s"] = bins
    s["by_speed"] = {m: [float(res[m][(res["speed"] >= lo) & (res["speed"] < hi), -1].mean())
                         if ((res["speed"] >= lo) & (res["speed"] < hi)).any() else None
                         for lo, hi in zip(bins[:-1], bins[1:])] for m in methods}
    s["share_by_speed"] = [float(((res["speed"] >= lo) & (res["speed"] < hi)).mean()) for lo, hi in zip(bins[:-1], bins[1:])]
    return s


# ------------------------------------------------------------------------------------------------
# One run: train on train players, stop on val players, report on test players
# ------------------------------------------------------------------------------------------------
def run_split(users, train_u, val_u, test_u, a, out, plots=True):
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    tr, va, te = (Windows(users, g, s) for g, s in ((train_u, a.stride_train), (val_u, a.stride_eval), (test_u, a.stride_test)))
    print(f"players train/val/test {len(train_u)}/{len(val_u)}/{len(test_u)} | windows {len(tr)}/{len(va)}/{len(te)} | device {DEV}", flush=True)
    model = build_model(a).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sample = tr.sampler(rng, a.balance_users)
    if getattr(a, "warp_aug", 0):
        tr.warp, tr.rng = a.warp_aug, rng

    log, best, best_state, bad, t0, recent, recent_loss = [], float("inf"), None, 0, time.time(), [], []
    for step in range(1, a.steps + 1):
        model.train()
        ctx, tgt = tr.batch(sample(a.batch))
        rel_ctx, rel_tgt, _ = prepare(ctx, tgt, a.frame)
        raw, unit = model(rel_ctx)
        loss, ang = loss_fn(raw, unit, rel_tgt, a.loss_on)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        recent.append(torch.rad2deg(ang[:, -1].mean()).item())
        recent_loss.append(loss.item())
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(model, va, a.frame, max_windows=3000)
            row = dict(step=step, train20=float(np.mean(recent)), val20=float(ev["model"][:, -1].mean()),
                       val20_hold=float(ev["hold"][:, -1].mean()), grad=float(gnorm), secs=time.time() - t0,
                       loss=float(np.mean(recent_loss)))
            log.append(row); recent = []; recent_loss = []
            print(f"epoch {len(log):3d} | loss {row['loss']:.6f} | validation endpoint {row['val20']:.4f} deg "
                  f"| hold {row['val20_hold']:.4f} deg | train {row['train20']:.3f} deg | {row['secs']:.1f}s", flush=True)
            if row["val20"] < best - 1e-6:
                best, best_state, bad = row["val20"], copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= a.patience:
                    print("early stop: validation players stopped improving", flush=True)
                    break
    model.load_state_dict(best_state)                                    # the checkpoint chosen on VALIDATION players
    torch.save({"state_dict": model.state_dict(), "config": {k: v for k, v in vars(a).items() if k != "data"}}, os.path.join(out, "model.pt"))

    n_seg = len({int(r) for r in te.rows[:, 0]})
    print(f"Scoring {len(te):,} windows from {n_seg} segments / {len(test_u)} players (test players, never used in training)", flush=True)
    res = evaluate(model, te, a.frame)                                   # test players, used once
    summ = summarise(res, te.uids)
    tr_ev = evaluate(model, tr, a.frame, max_windows=5000)               # for the train-vs-test gap
    summ["train_step20_eval_mode"] = float(tr_ev["model"][:, -1].mean())
    summ["val_step20_best"] = best
    summ["gap_test_minus_train"] = summ["methods"]["model"]["mean_step20"] - summ["train_step20_eval_mode"]
    summ.update(parameters=n_params, steps_run=log[-1]["step"], players={"train": len(train_u), "val": len(val_u), "test": len(test_u)},
                config={k: v for k, v in vars(a).items() if k not in ("data",)})
    json.dump(dict(summary=summ, log=log), open(os.path.join(out, "metrics.json"), "w"), indent=1)
    if not a.no_windows:
        np.savez_compressed(os.path.join(out, "per_window.npz"), user=res["user"], speed=res["speed"],
                            **{m: res[m].astype(np.float32) for m in ["model", *BASELINES]})
    if plots and not a.no_plots:
        make_plots(summ, log, out)
    rep_txt = format_report(summ)
    if not a.no_plots:
        open(os.path.join(out, "report.txt"), "w").write(rep_txt + "\n")
    print("\n" + rep_txt, flush=True)
    return summ


# ------------------------------------------------------------------------------------------------
# Plain-English report
# ------------------------------------------------------------------------------------------------
def format_report(s):
    """Compact table. Errors are angles in degrees between predicted and true head direction (smaller is better)."""
    mm = s["methods"]
    cols = [("model", "GRU"), ("hold", "Hold"), ("damped_cv", "DampedVel"), ("const_vel", "ConstVel")]
    row = lambda label, vals: f"  {label:36s}" + "".join(f"{v:>10s}" for v in vals)
    f2 = lambda key: [f"{mm[m][key]:.2f}" for m, _ in cols]
    hold = mm["hold"]["mean_step20"]
    pp = np.array(list(mm["model"]["per_player_mean_step20"].values()))
    v1, v2 = mm["model"]["vs_hold"], mm["model"]["vs_damped_cv"]
    L = ["", "=" * 74, f"TEST RESULTS  |  {s['n_players']} unseen players, {s['n_windows']:,} windows  |  error in degrees, smaller is better", "=" * 74,
         row("", [n for _, n in cols]),
         row("mean error at step 20 (0.2 s ahead)", f2("mean_step20")),
         row("median error at step 20", f2("median_step20")),
         row("90th percentile at step 20", f2("p90_step20")),
         row("mean error over all 20 steps", [f"{np.mean(mm[m]['mean_per_step']):.2f}" for m, _ in cols]),
         row("95% range of the mean (step 20)", [f"{mm[m]['ci95_step20_players'][0]:.1f}-{mm[m]['ci95_step20_players'][1]:.1f}" for m, _ in cols]),
         row("improvement vs Hold (step 20)", ["-" if m == "hold" else f"{100 * (hold - mm[m]['mean_step20']) / hold:+.0f}%" for m, _ in cols]),
         "",
         f"  players where the GRU beats Hold      {v1['players_model_better']}/{v1['players']}",
         f"  players where the GRU beats DampedVel  {v2['players_model_better']}/{v2['players']}",
         "",
         "GENERALISATION (training players vs players it never saw)",
         f"  train error (step 20)                 {s['train_step20_eval_mode']:.2f}",
         f"  validation error (stop decisions)     {s['val_step20_best']:.2f}",
         f"  TEST error (unseen players)           {mm['model']['mean_step20']:.2f}",
         f"  gap, test minus train                 {s['gap_test_minus_train']:+.2f}   (big positive = memorising the training players)",
         f"  per-player GRU error: best/median/worst  {pp.min():.2f} / {np.median(pp):.2f} / {pp.max():.2f}",
         f"  spread between players (std)          {pp.std():.2f}",
         f"  {s['parameters']:,} weights | {s['steps_run']:,} steps | players train/val/test {s['players']['train']}/{s['players']['val']}/{s['players']['test']}",
         "=" * 74]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------------------------
COL = {"model": "#1f77b4", "hold": "#444444", "const_vel": "#c0392b", "damped_cv": "#e67e22"}
NAME = {"model": "GRU", "hold": "Hold last pose", "const_vel": "Constant velocity", "damped_cv": "Damped constant velocity"}


def make_plots(s, log, out):
    mm = s["methods"]
    # 1 loss curves
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=160)
    st = [r["step"] for r in log]
    ax.plot(st, [r["train20"] for r in log], label="training players", color="#2ca02c")
    ax.plot(st, [r["val20"] for r in log], label="validation players", color=COL["model"])
    ax.plot(st, [r["val20_hold"] for r in log], "--", label="hold last pose (validation)", color=COL["hold"])
    ax.set_xlabel("training step"); ax.set_ylabel("angle error at step 20 (degrees)")
    ax.set_title("Learning curves: a widening gap means memorising"); ax.grid(alpha=.3); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig1_learning_curves.png")); plt.close(fig)
    # 2 error by step
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=160)
    for m in ["hold", "const_vel", "damped_cv", "model"]:
        ax.plot(range(1, HORIZON + 1), mm[m]["mean_per_step"], label=NAME[m], color=COL[m], lw=2 if m == "model" else 1.4)
    ax.set_xlabel("prediction step (100 per second)"); ax.set_ylabel("mean angle error (degrees)")
    ax.set_title("Test players: error by prediction step"); ax.grid(alpha=.3); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_error_by_step.png")); plt.close(fig)
    # 3 per-player
    pm, ph = mm["model"]["per_player_mean_step20"], mm["hold"]["per_player_mean_step20"]
    order = sorted(pm, key=lambda u: ph[u])
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(max(6, 0.22 * len(order) + 2), 3.6), dpi=160)
    ax.bar(x - 0.2, [ph[u] for u in order], 0.4, label=NAME["hold"], color=COL["hold"])
    ax.bar(x + 0.2, [pm[u] for u in order], 0.4, label="GRU", color=COL["model"])
    ax.set_xticks([]); ax.set_xlabel(f"{len(order)} held-out players (sorted by baseline error)"); ax.set_ylabel("angle error at step 20 (degrees)")
    v = mm["model"]["vs_hold"]
    ax.set_title(f"Generalisation: GRU better for {v['players_model_better']} of {v['players']} unseen players")
    ax.set_ylim(0, max(max(ph.values()), max(pm.values())) * 1.3)
    ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=2); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_per_player.png")); plt.close(fig)
    # 4 by speed
    bins = s["speed_bins_deg_per_s"]; labels = [f"{int(lo)}-{int(hi) if hi < 1e8 else '+'}" for lo, hi in zip(bins[:-1], bins[1:])]
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=160)
    for i, m in enumerate(["hold", "damped_cv", "model"]):
        ax.bar(np.arange(len(labels)) + (i - 1) * 0.27, [v or 0 for v in s["by_speed"][m]], 0.27, label=NAME[m], color=COL[m])
    ax.set_xticks(range(len(labels))); ax.set_xticklabels([f"{l}\n({100 * sh:.0f}% of windows)" for l, sh in zip(labels, s["share_by_speed"])], fontsize=7)
    ax.set_xlabel("head speed over the last 0.2 s (degrees per second)"); ax.set_ylabel("angle error at step 20 (degrees)")
    ax.set_title("Where the model helps: error by head speed"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_by_speed.png")); plt.close(fig)


# ------------------------------------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------------------------------------
def selftest():
    # a head turning at a constant 30 deg/s about one axis, sampled at 100 Hz
    t = torch.arange(CONTEXT + HORIZON, dtype=torch.float64) / HZ
    rv = torch.stack([0 * t, torch.deg2rad(torch.tensor(30.0, dtype=torch.float64)) * t, 0 * t], -1)
    q = qexp(rv)[None].float()
    ctx, tgt = q[:, :CONTEXT], q[:, CONTEXT:]
    e = lambda p: torch.rad2deg(angle(p, tgt))[0, -1].item()
    assert abs(e(hold_predict(ctx)) - 6.0) < 1e-3, "hold should be off by 30 deg/s x 0.2 s = 6 deg"
    assert e(cv_predict(ctx, 5, 1.0)) < 1e-2, "constant velocity should be exact on constant-velocity data"
    assert abs(e(cv_predict(ctx, 1, 0.9)) - 6.0 * (1 - sum(0.9 ** j for j in range(1, 21)) / 20)) < 0.05
    # sign safety and tiny angles
    assert torch.equal(angle(ctx[:, -1], tgt[:, -1]), angle(ctx[:, -1], -tgt[:, -1]))
    tiny = qexp(torch.tensor([[0.0, np.radians(0.01), 0.0]])).float()
    assert abs(torch.rad2deg(angle(torch.tensor([[1.0, 0, 0, 0]]), tiny)).item() - 0.01) < 1e-4
    # relative <-> absolute round trip, and an untrained model equals hold
    for fr in ("body", "world"):
        rel_ctx, rel_tgt, last = prepare(ctx, tgt, fr)
        assert torch.allclose(from_rel(rel_tgt, last, fr), tgt, atol=1e-6)
        _, unit = GRUForecaster(8)(rel_ctx)
        assert torch.allclose(from_rel(unit, last, fr), hold_predict(ctx), atol=1e-5)
    # features on a constant 30 deg/s turn: velocity = 30 deg/s = 0.5236 rad/s, acceleration = 0, speed = 0.5236
    rel_ctx, _, _ = prepare(ctx, tgt, "body")
    f = featurize(rel_ctx, "vel+acc")[0]
    assert f.shape == (CONTEXT, 10) and torch.equal(featurize(rel_ctx, "velacc")[0], f)
    assert abs(featurize(rel_ctx, "speed")[0, 100:, 4].mean().item() - np.radians(30)) < 2e-3
    assert featurize(rel_ctx, "vel+acc+mag").shape[-1] == 12 and featurize(rel_ctx, "acc+vel").shape[-1] == 10
    assert featurize(rel_ctx, "acc")[0, 50:, 4:].abs().max().item() < 5e-3
    for bad in ("foo", "vel+vel", "vel+"):
        try:
            feature_groups(bad); raise SystemExit("bad feature name accepted: " + bad)
        except ValueError:
            pass
    assert f.shape == (CONTEXT, 10) and abs(f[50:, 5].mean().item() - np.radians(30)) < 2e-3, f[50:, 4:7].mean(0)
    assert f[50:, 7:].abs().max().item() < 5e-3, "acceleration of a steady turn should be about 0"
    assert abs(featurize(rel_ctx, "mag")[0, 100:, 4].mean().item() - np.radians(30)) < 2e-3 and featurize(rel_ctx, "mag").shape[-1] == 6
    assert torch.equal(featurize(rel_ctx, "base"), rel_ctx)
    _, unit = GRUForecaster(8, "vel+acc+mag")(rel_ctx)
    assert torch.allclose(from_rel(unit, ctx[:, -1:], "body"), hold_predict(ctx), atol=1e-5), "untrained model with features should still equal hold"
    # GADS: shape, untrained = hold, a shuffled window changes the output only when position codes are on
    g = GADSForecaster(8, "vel+acc+mag", groups=20)
    _, unit = g(rel_ctx)
    assert unit.shape == (1, HORIZON, 4) and torch.allclose(from_rel(unit, ctx[:, -1:], "body"), hold_predict(ctx), atol=1e-5)
    # shuffling readings INSIDE each 20-reading slice: a pure Deep Set cannot notice (pos=False), position codes can (pos=True)
    perm = torch.cat([k * 20 + torch.randperm(20, generator=torch.Generator().manual_seed(k)) for k in range(CONTEXT // 20)])
    for p in (True, False):
        gg = GADSForecaster(8, "base", groups=CONTEXT // 20, pos=p)
        nn.init.normal_(gg.head.weight, std=0.5)
        same = torch.allclose(gg(rel_ctx)[0], gg(rel_ctx[:, perm])[0], atol=1e-5)
        assert same != p, f"pos={p}: shuffle inside slices should {'change' if p else 'not change'} the output"
    a_, b_ = F.normalize(torch.randn(5, 3, 4), dim=-1), F.normalize(torch.randn(5, 3, 4), dim=-1)
    aw, ax, ay, az = a_.unbind(-1); bw, bx, by, bz = b_.unbind(-1)
    ref = torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                       aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)
    for mode in (False, True):
        QMUL_MATRIX[0] = mode
        assert torch.allclose(qmul(a_, b_), ref, atol=1e-6) and torch.allclose(qmul(a_[:, :1], b_), qmul(a_[:, :1].expand(-1, 3, -1), b_), atol=1e-6)
    QMUL_MATRIX[0] = False
    # new pieces: velocity-only input, small-angle velocity, TCN, GADS fixed codes, time-warp
    fv = featurize(rel_ctx, "vel+noq")
    assert fv.shape[-1] == 3 and torch.allclose(fv, featurize(rel_ctx, "vel")[..., 4:])
    SMALL_ANGLE[0] = True
    fs = featurize(rel_ctx, "vel+mag")
    SMALL_ANGLE[0] = False
    assert (fs - featurize(rel_ctx, "vel+mag")).abs().max().item() < 1e-5, "small-angle velocity should match the exact one at 100 Hz"
    for kind in ("dilated", "strided"):
        t = TCNForecaster(8, "vel+mag", kind)
        _, unit = t(rel_ctx)
        assert unit.shape == (1, HORIZON, 4) and torch.allclose(from_rel(unit, ctx[:, -1:], "body"), hold_predict(ctx), atol=1e-5)
    g = GADSForecaster(8, "base", groups=20)
    assert torch.allclose(g.pe_in, g._sincos(20, 8)) and torch.allclose(g.pe_group, g._sincos(20, 8))
    x_ = torch.randn(3, 5, 7, 20)
    assert torch.allclose(safe_max(x_), x_.amax(-1)), "ReLU-tree max must equal max"
    g = GADSForecaster(32, "vel", groups=20)
    for p_ in g.parameters():
        nn.init.normal_(p_, std=0.3)
    with torch.no_grad():
        r0 = g(rel_ctx)[0]
        EXPORT_SAFE[0] = True
        r1 = g(rel_ctx)[0]
        EXPORT_SAFE[0] = False
    assert torch.allclose(r0, r1, atol=1e-4), (r0 - r1).abs().max()
    seg = [q[0].numpy()]
    w_ctx, w_tgt = warp_window(seg, 0, 0, 1.0)
    assert np.allclose(w_ctx, q[0, :CONTEXT].numpy(), atol=1e-6)
    w_ctx, w_tgt = warp_window(seg, 0, 0, 0.5)                       # half speed: 30 deg/s turn becomes 15 deg/s
    e = np.degrees(angle(torch.from_numpy(w_ctx[-1:]), torch.from_numpy(w_tgt[-1:])).item())
    assert abs(e - 3.0) < 0.05, e
    print("selftest ok")


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+")
    ap.add_argument("--out", default="outputs/run")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--frame", choices=["body", "world"], default="body")
    ap.add_argument("--loss-on", choices=["all", "last"], default="all")
    ap.add_argument("--stride-train", type=int, default=10)
    ap.add_argument("--stride-eval", type=int, default=50)
    ap.add_argument("--stride-test", type=int, default=10, help="window spacing when scoring the test players (1 = every window)")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--balance-users", action="store_true")
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--cv-folds", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--features", default="base", help="base, or extra column groups joined with +: vel acc speed mag (e.g. vel+acc)")
    ap.add_argument("--model", choices=["gru", "gads", "tcn"], default="gru", help="gru (default), gads (parallel Deep-Set + attention, our adaptation) or tcn (convolutions only)")
    ap.add_argument("--tcn-kind", choices=["dilated", "strided"], default="dilated", help="tcn: dilated causal stack or strided (halving) stack")
    ap.add_argument("--warp-aug", type=float, default=0.0, help="time-warp augmentation while training: play each training window at a random speed in [1/(1+a), 1+a]")
    ap.add_argument("--dim", type=int, default=32, help="gads: width of every layer and of the attention")
    ap.add_argument("--groups", type=int, default=20, help="gads: number of time slices the window is cut into")
    ap.add_argument("--heads", type=int, default=4, help="gads: attention heads")
    ap.add_argument("--pool", choices=["max", "mean"], default="max", help="gads: Deep Set pooling inside a slice")
    ap.add_argument("--no-pos", action="store_true", help="gads: remove both position codes (ablation: can the model still use order?)")
    ap.add_argument("--test-players-file", default=None, help="text file, one player id per line: these players are the test set; every other player is split train/val by hash")
    ap.add_argument("--val-players-file", default=None, help="with --test-players-file: these players are the validation set instead of the hash's; all others train")
    ap.add_argument("--no-plots", action="store_true", help="skip the four figures and report.txt")
    ap.add_argument("--no-windows", action="store_true", help="skip per_window.npz")
    ap.add_argument("--context", type=int, default=CONTEXT, help="use only the last N of the 400 context readings")
    a = ap.parse_args(argv)
    feature_groups(a.features)                                          # fail early on a typo
    return a


def run(a, users=None):
    CTX_LEN[0] = a.context
    users = users or load_users(a.data, a.max_users)
    uids = sorted(users)
    print(f"{len(uids)} players, {sum(len(s) for s in users.values())} segments, "
          f"{sum(len(x) for s in users.values() for x in s) / HZ / 3600:.2f} hours", flush=True)
    if a.cv_folds:
        order = list(np.random.default_rng(a.seed).permutation(uids))
        folds = [order[i::a.cv_folds] for i in range(a.cv_folds)]
        rows = []
        for i, test_u in enumerate(folds):
            rest = [u for u in order if u not in set(test_u)]
            val_u = rest[::10] or rest[:1]
            train_u = [u for u in rest if u not in set(val_u)]
            print(f"--- fold {i + 1}/{a.cv_folds}", flush=True)
            s = run_split(users, train_u, val_u, test_u, a, os.path.join(a.out, f"fold{i + 1}"), plots=False)
            rows.append({m: s["methods"][m]["mean_step20"] for m in ["model", *BASELINES]})
        agg = {m: [float(np.mean([r[m] for r in rows])), float(np.std([r[m] for r in rows]))] for m in rows[0]}
        json.dump(dict(folds=rows, mean_std_step20=agg), open(os.path.join(a.out, "cv_summary.json"), "w"), indent=1)
        nm = {"model": "GRU (our model)", "hold": "Hold the last pose", "const_vel": "Constant velocity", "damped_cv": "Damped constant velocity"}
        lines = ["", "=" * 92, f"CROSS-VALIDATION: {a.cv_folds} rounds, each tested on a different group of players the model never saw",
                 "Error 20 readings (0.2 s) ahead, degrees, mean over each group's windows", "=" * 92,
                 f"   {'':26s}" + "".join(f"{'group ' + str(i + 1):>9s}" for i in range(len(rows))) + f"{'average':>10s}{'+/-':>8s}"]
        for m in ["model", "hold", "const_vel", "damped_cv"]:
            lines.append(f"   {nm[m]:26s}" + "".join(f"{r[m]:9.2f}" for r in rows) + f"{agg[m][0]:10.2f}{agg[m][1]:8.2f}")
        lines.append("   (+/- is how much the score moves from one group of players to another)")
        open(os.path.join(a.out, "cv_report.txt"), "w").write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
    else:
        if a.test_players_file:
            fixed = {l.strip() for l in open(a.test_players_file) if l.strip()}
            missing = fixed - set(uids)
            assert not missing, f"{len(missing)} fixed test players are not in the data, e.g. {sorted(missing)[:3]}"
            rest = [u for u in uids if u not in fixed]
            groups = {"test": sorted(fixed), "val": [u for u in rest if user_split(u) == "val"],
                      "train": [u for u in rest if user_split(u) != "val"]}   # players the hash called 'test' but are not in the file train instead
            if a.val_players_file:                                    # a fixed validation list replaces the hash: train = everyone else
                val = {l.strip() for l in open(a.val_players_file) if l.strip()}
                assert val <= set(rest), "validation players missing from the data or also in the test file"
                groups["val"], groups["train"] = sorted(val), [u for u in rest if u not in val]
            assert not fixed & set(groups["train"] + groups["val"]), "test player leaked into train/val"
        else:
            groups = {s: [u for u in uids if user_split(u) == s] for s in ("train", "val", "test")}
        run_split(users, groups["train"], groups["val"], groups["test"], a, a.out)


def main():
    a = parse()
    selftest()
    if not a.selftest:
        run(a)


if __name__ == "__main__":
    main()
