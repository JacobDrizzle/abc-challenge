"""26,972-parameter causal TCNN; scalar-first Hamilton quaternion convention."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def qmul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)


def conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def rotvec_to_quaternion(v):
    angle = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    # sinc(x)=sin(pi*x)/(pi*x), including its stable value and derivative at zero.
    xyz = .5 * torch.sinc(angle / (2 * math.pi)) * v
    return F.normalize(torch.cat((torch.cos(angle / 2), xyz), dim=-1), dim=-1)


def relative_history(history):
    q = F.normalize(history, dim=-1)
    dots = (q[:, 1:] * q[:, :-1]).sum(-1)
    steps = torch.where(dots < 0, -torch.ones_like(dots), torch.ones_like(dots))
    signs = torch.cumprod(torch.cat((torch.ones_like(steps[:, :1]), steps), dim=1), dim=1)
    q = q * signs.unsqueeze(-1)
    last = q[:, -1:, :]
    return qmul(conjugate(last), q), last


def quaternion_to_rotvec(q):
    """Shortest SO(3) logarithm, with a finite derivative at identity."""
    q = F.normalize(q, dim=-1)
    q = torch.where(q[..., :1] < 0, -q, q)
    sine = torch.linalg.vector_norm(q[..., 1:], dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sine, q[..., :1])
    scale = torch.where(sine > 1e-6, angle / sine.clamp_min(1e-6), 2 + sine.square()/3)
    return q[..., 1:] * scale


def angular_velocity_features(relative, rate_hz=100.):
    # Rotation differences expressed in the fixed forecast-anchor coordinate frame.
    delta = qmul(relative[:, 1:], conjugate(relative[:, :-1]))
    velocity = quaternion_to_rotvec(delta) * rate_hz
    return torch.cat((torch.zeros_like(velocity[:, :1]), velocity), dim=1)


def body_velocity_features(history, rate_hz=100.):
    """[B, N, 4] features: per-step rotation conj(q[t-1]) q[t] in its own head frame (rad/s) and its
    magnitude; the first row is zero. Also the last orientation and the last step's rotation vector."""
    q = F.normalize(history, dim=-1)
    step = quaternion_to_rotvec(qmul(conjugate(q[:, :-1]), q[:, 1:]))
    velocity = step * rate_hz
    rows = torch.cat((velocity, torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)), dim=-1)
    return torch.cat((torch.zeros_like(rows[:, :1]), rows), dim=1), q[:, -1:], step[:, -1]


def damped_prior_rotvec(relative, horizon, tau_samples, lag=1):
    """Body-frame rotation vectors [B,H,3] of the damped-velocity baseline (tcnn/damped.py).

    Exp(e*v)*now == now*Exp(e*now^-1 v); with relative[-1] = identity, now^-1 past = relative[-1-lag].
    """
    velocity = quaternion_to_rotvec(conjugate(relative[:, -1-lag])) / lag
    steps = torch.arange(1, horizon+1, dtype=relative.dtype, device=relative.device)
    elapsed = -tau_samples * torch.expm1(-steps / tau_samples)
    return velocity[:, None, :] * elapsed[None, :, None]


class CausalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.left_padding, self.dilation = 2 * dilation, dilation
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def dilated_conv(self, x):
        # Same result and parameters as self.conv(F.pad(x, ...)). Under deterministic algorithms,
        # PyTorch runs dilated Conv1d as slow_conv_dilated2d (~50x slower); taps + matmul avoid it.
        padded, length, d = F.pad(x, (self.left_padding, 0)), x.shape[-1], self.dilation
        taps = torch.cat([padded[:, :, k*d:k*d + length] for k in range(3)], dim=1)
        weight = self.conv.weight.permute(0, 2, 1).reshape(self.conv.out_channels, -1)
        return torch.matmul(weight, taps) + self.conv.bias[:, None]

    def forward(self, x):
        update = self.dropout(F.relu(self.dilated_conv(x)))
        return F.relu(x + update)


def receptive_field(num_blocks):
    """Time steps one output sees through kernel-3 blocks with dilations 1, 2, ..., 2**(num_blocks-1)."""
    return 2 ** (num_blocks + 1) - 1


class TinyTCNN(nn.Module):
    def __init__(self, context=400, horizon=20, channels=32, dropout=.1,
                 effective_context=None, velocity_features=False, damped_prior_tau_ms=None,
                 input_features="anchor", num_blocks=8):
        super().__init__()
        self.context, self.horizon = context, horizon
        self.channels, self.dropout_rate = channels, dropout
        self.effective_context = context if effective_context is None else effective_context
        if not 2 <= self.effective_context <= context:
            raise ValueError("Invalid effective context")
        self.velocity_features = velocity_features
        # Residual on the parameter-free damped forecast; the zero-initialised head starts there.
        self.damped_prior_tau_ms = damped_prior_tau_ms
        # "anchor": orientations relative to the latest one (+ velocities in that frame) -- every call
        # recomputes everything. "body_velocity": each step's angular velocity in its own head frame and
        # its magnitude -- independent of the latest orientation, so a streaming graph can cache layers.
        if input_features not in ("anchor", "body_velocity"):
            raise ValueError("input_features must be anchor or body_velocity")
        if input_features == "body_velocity" and velocity_features:
            raise ValueError("velocity_features applies to anchor inputs only")
        self.input_features, self.num_blocks = input_features, num_blocks
        inputs = 4 if input_features == "body_velocity" else 7 if velocity_features else 4
        self.projection = nn.Conv1d(inputs, channels, 1)
        self.blocks = nn.Sequential(*(CausalBlock(channels, 2**i, dropout) for i in range(num_blocks)))
        self.head = nn.Linear(channels, horizon * 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, history):
        if history.ndim != 3 or history.shape[1:] != (self.context, 4):
            raise ValueError(f"Expected [B,{self.context},4], got {tuple(history.shape)}")
        if self.input_features == "body_velocity":
            features, last, step = body_velocity_features(history[:, -self.effective_context:])
        else:
            relative, last = relative_history(history[:, -self.effective_context:])
            features = torch.cat((relative, angular_velocity_features(relative)), dim=-1) if self.velocity_features else relative
        x = self.blocks(F.relu(self.projection(features.transpose(1, 2))))
        rotation_vectors = self.head(x[:, :, -1]).reshape(-1, self.horizon, 3)
        if self.damped_prior_tau_ms is not None:
            tau = self.damped_prior_tau_ms / 10  # ms to 100 Hz samples
            if self.input_features == "body_velocity":
                # Same rotation as damped_prior_rotvec: the last step's body-frame rotation, damped.
                steps = torch.arange(1, self.horizon+1, dtype=step.dtype, device=step.device)
                rotation_vectors = rotation_vectors + step[:, None, :] * (-tau * torch.expm1(-steps / tau))[None, :, None]
            else:
                rotation_vectors = rotation_vectors + damped_prior_rotvec(relative, self.horizon, tau)
        return F.normalize(qmul(last, rotvec_to_quaternion(rotation_vectors)), dim=-1)

    def specification(self):
        return dict(context=self.context, horizon=self.horizon, channels=self.channels,
                    dropout=self.dropout_rate, effective_context=self.effective_context,
                    velocity_features=self.velocity_features, damped_prior_tau_ms=self.damped_prior_tau_ms,
                    input_features=self.input_features, num_blocks=self.num_blocks)


def rotation_loss(predicted, target, kind="quaternion", endpoint_weight=.5):
    if kind not in ("quaternion", "angular") or not 0 <= endpoint_weight <= 1:
        raise ValueError("Invalid rotation loss settings")
    p, t = F.normalize(predicted, dim=-1), F.normalize(target, dim=-1)
    dot = (p * t).sum(-1)
    if kind == "quaternion":
        error = 1 - dot.square().clamp(0, 1)
    else:
        # Quaternion chord-angle formula: exact geodesic radians; no acos singularity.
        t = torch.where(dot[..., None] < 0, -t, t)
        error = 4 * torch.atan2(torch.linalg.vector_norm(p-t, dim=-1),
                               torch.linalg.vector_norm(p+t, dim=-1))
    return (1-endpoint_weight) * error.mean() + endpoint_weight * error[:, -1].mean()
