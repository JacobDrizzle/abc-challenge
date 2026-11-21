"""v7 CompactTCNN plus per-block kernel sizes and dilations, and a choice of input and output maps; with every
option at its default it is the v7 model (same parameters, created in the same order, same forward).

kernel_sizes, dilations: one per block, for separable blocks; None keeps v7's kernel 3 and dilation_base**i. The
  receptive field is 1 + sum((k - 1) * d). v7's best sees 243 samples of the 400 the rules allow; a kernel-5 last
  block with dilation 79 sees 397 for 48 more parameters, and fewer, wider blocks cost fewer graph nodes.
maps: how per-step rates come in and forecast rotations go out. For d = conj(q[t-1]) q[t] = (w, v), with angle
  theta, and a forecast rotation vector r:
  "exact"  (v7) rate = log(d) = theta n; quaternion = exp(r).
  "lean"   rate = 2 w v (1 + 2/3 |v|^2), the degree-1 series of theta / sin(theta) in 1 - cos(theta) = 2 |v|^2
           (relative error 5e-4 at 20 degrees a step); quaternion along (2 - |r|^2/6, r), the direction of exp(r)
           to second order (0.003 degrees off at 30, 0.09 at 60). The streaming graph computes these (Track A).
  "native" rate = 2 w v = sin(theta) n; quaternion along (2, r), the Cayley map (angle 2 atan(|r| / 2)). Trained
           with them, the graph needs neither series.
  The w v products ignore the signs of both quaternions, as the exact log map does.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from experimental.v6_gravity_mirror.model import gravity_in_head_frame
from experimental.v7_compact.model import CompactTCNN, SeparableBlock
from tcnn.model import conjugate, qmul, rotvec_to_quaternion

MAPS = ("exact", "lean", "native")


def map_quaternion(r, maps):
    """Unit quaternions [..., 4] for forecast rotation vectors r [..., 3] under the given maps."""
    if maps == "exact":
        return rotvec_to_quaternion(r)
    scalar = 2 - (r * r).sum(-1, keepdim=True) / 6 if maps == "lean" else torch.full_like(r[..., :1], 2.)
    return F.normalize(torch.cat((scalar, r), -1), dim=-1)


def block_layout(num_blocks, dilation_base=2, kernel_sizes=None, dilations=None):
    """Per-block kernel sizes and dilations; None means v7's kernel 3 and dilation_base**i."""
    kernels = [3] * num_blocks if kernel_sizes is None else list(kernel_sizes)
    spacing = [dilation_base ** i for i in range(num_blocks)] if dilations is None else list(dilations)
    if len(kernels) != num_blocks or len(spacing) != num_blocks:
        raise ValueError("kernel_sizes and dilations need one entry per block")
    return kernels, spacing


def receptive_field(num_blocks, dilation_base=2, kernel_sizes=None, dilations=None):
    """Time steps one output sees: 1 + sum over blocks of (kernel - 1) * dilation."""
    kernels, spacing = block_layout(num_blocks, dilation_base, kernel_sizes, dilations)
    return 1 + sum((k - 1) * d for k, d in zip(kernels, spacing))


class KernelBlock(SeparableBlock):
    """SeparableBlock with a depthwise kernel of any size k: taps at t - (k-1)d, ..., t - d, t."""

    def __init__(self, stream_channels, streams, dilation, dropout, kernel_size=3):
        super().__init__(stream_channels, streams, dilation, dropout)
        self.kernel_size, self.left_padding = kernel_size, (kernel_size - 1) * dilation
        # SeparableBlock built a kernel-3 depthwise conv; this one has kernel_size taps.
        self.depthwise = nn.Conv1d(stream_channels, stream_channels, kernel_size, dilation=dilation,
                                   groups=stream_channels, bias=False)

    def forward(self, x):
        # SeparableBlock.forward with kernel_size taps instead of 3.
        padded, length, d = F.pad(x, (self.left_padding, 0)), x.shape[-1], self.dilation
        taps = torch.stack([padded[:, :, k*d:k*d + length] for k in range(self.kernel_size)], -1)
        depthwise = (taps * self.depthwise.weight[:, 0].repeat(self.streams, 1)[:, None, :]).sum(-1)
        weight, bias = self.pointwise_weight()
        update = self.dropout(F.relu(torch.matmul(weight, depthwise) + bias[:, None]))
        return F.relu(x + update)


class LeanTCNN(CompactTCNN):
    """CompactTCNN plus kernel_sizes, dilations and maps (see the module docstring).

    The first four arguments keep train_tcnn's positional order (context, horizon, channels, dropout).
    """

    def __init__(self, context=400, horizon=20, channels=48, dropout=.1, effective_context=None,
                 velocity_features=False, damped_prior_tau_ms=None, input_features="body_velocity",
                 num_blocks=8, gravity_features=False, mirror_equivariant=False, dilation_base=2,
                 separable=False, head_basis=None, speed_feature="norm", kernel_sizes=None, dilations=None,
                 maps="exact"):
        super().__init__(context, horizon, channels, dropout, effective_context, velocity_features,
                         damped_prior_tau_ms, input_features, num_blocks, gravity_features, mirror_equivariant,
                         dilation_base, separable, head_basis, speed_feature)
        if maps not in MAPS:
            raise ValueError("maps must be exact, lean or native")
        self.kernel_sizes, self.dilations, self.maps = kernel_sizes, dilations, maps
        # v7's blocks stay unless a layout is given, so the defaults reproduce v7 exactly.
        if kernel_sizes is not None or dilations is not None:
            if not separable:
                raise ValueError("kernel_sizes and dilations apply to separable blocks only")
            kernels, spacing = block_layout(num_blocks, dilation_base, kernel_sizes, dilations)
            self.blocks = nn.Sequential(*(KernelBlock(self.stream_channels, self.streams, d, dropout, k)
                                          for k, d in zip(kernels, spacing)))

    def features(self, window):
        """As GravityMirrorTCNN.features, with the rate from this model's maps."""
        if self.maps == "exact":
            return super().features(window)
        q = F.normalize(window, dim=-1)
        delta = qmul(conjugate(q[:, :-1]), q[:, 1:])  # (cos(theta/2), sin(theta/2) n) per step
        w, v = delta[..., :1], delta[..., 1:]
        step = 2 * w * v  # sin(theta) n
        if self.maps == "lean":
            step = step * (1 + 2 / 3 * (v * v).sum(-1, keepdim=True))
        velocity = step * 100.  # rad/s at 100 Hz
        speed = (velocity.abs().sum(-1, keepdim=True) if self.speed_feature == "l1"
                 else torch.linalg.vector_norm(velocity, dim=-1, keepdim=True))
        rows = torch.cat((velocity, speed), -1)
        features = torch.cat((torch.zeros_like(rows[:, :1]), rows), 1)  # the first step has no rate
        if self.gravity_features:
            features = torch.cat((features, gravity_in_head_frame(q)), -1)
        return features, q[:, -1:], step[:, -1]

    def forward(self, history):
        if self.maps == "exact":
            return super().forward(history)
        # GravityMirrorTCNN.forward up to the rotation vectors; then this model's output map.
        if history.ndim != 3 or history.shape[1:] != (self.context, 4):
            raise ValueError(f"Expected [B,{self.context},4], got {tuple(history.shape)}")
        features, last, step = self.features(history[:, -self.effective_context:])
        x = features.transpose(1, 2)
        if self.mirror_equivariant:
            x = torch.cat((self.projection(x), self.projection(x * self.input_signs[:, None])), 1)
        else:
            x = self.projection(x)
        x = self.blocks(F.relu(x))
        r = self.readout(x[:, :, -1]).reshape(-1, self.horizon, 3)
        if self.damped_prior_tau_ms is not None:
            tau = self.damped_prior_tau_ms / 10  # ms to 100 Hz samples
            steps = torch.arange(1, self.horizon+1, dtype=step.dtype, device=step.device)
            r = r + step[:, None, :] * (-tau * torch.expm1(-steps / tau))[None, :, None]
        return F.normalize(qmul(last, map_quaternion(r, self.maps)), dim=-1)

    def specification(self):
        return dict(super().specification(), kernel_sizes=self.kernel_sizes, dilations=self.dilations,
                    maps=self.maps)


def deployed(model):
    """The model the v9 streaming graph reproduces: exact maps become lean ones (Track A); lean and native stay."""
    if model.maps == "exact":
        model.maps = "lean"
    return model


def load_checkpoint(path):
    """v9 checkpoints, and v7 and v6 ones: their specification lacks the v9 (and v7) keys, which take the defaults."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = LeanTCNN(**state["model_spec"])
    model.load_state_dict(state["model_state"])
    return model.eval(), state
