"""v6 GravityMirrorTCNN with four compression options; with all of them at their defaults it is the v6 model.

dilation_base: block i has dilation base**i. Base 3 with kernel 3 tiles every lag exactly once, so 5 blocks
  reach a receptive field of 243 samples where base 2 needs 7 blocks for 255.
separable: each block is a depthwise dilated kernel-3 conv (one filter per channel, shared by both mirror
  streams) followed by a pointwise 1x1 conv; on twin streams the pointwise weight is [[A, B], [B, A]] stored
  once, as in MirrorCausalBlock. No nonlinearity between the two, so the depthwise bias would be redundant.
head_basis: P, or None for v6's 60-output head. The head emits P coefficients per axis, and step k's rotation
  vector is sum_j c_j (k / horizon)**(j+1): corrections to the rate, acceleration and jerk of the damped prior.
  M2 (26 Oct): projecting v6's own 60 outputs onto P = 3 such terms changed validation error by +0.001 deg.
speed_feature: "norm" (v5/v6) or "l1", |wx| + |wy| + |wz|, which a graph can build from Relu and MatMul alone.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from experimental.v6_gravity_mirror.model import ROTATION_SIGNS, GravityMirrorTCNN, MirrorCausalBlock
from tcnn.model import CausalBlock


def receptive_field(num_blocks, base=2):
    """Time steps one output sees through kernel-3 blocks with dilations 1, base, ..., base**(num_blocks-1)."""
    return 1 + 2 * sum(base ** i for i in range(num_blocks))


class SeparableBlock(nn.Module):
    """Depthwise + pointwise version of CausalBlock / MirrorCausalBlock on [B, streams * K, T]."""

    def __init__(self, stream_channels, streams, dilation, dropout):
        super().__init__()
        self.left_padding, self.dilation, self.streams = 2 * dilation, dilation, streams
        self.depthwise = nn.Conv1d(stream_channels, stream_channels, 3, dilation=dilation,
                                   groups=stream_channels, bias=False)
        self.pointwise = nn.Conv1d(streams * stream_channels, stream_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def pointwise_weight(self):
        """[S*K, S*K] weight and [S*K] bias; for two streams [[A, B], [B, A]] built from the stored [A, B]."""
        w, b, k = self.pointwise.weight[:, :, 0], self.pointwise.bias, self.pointwise.out_channels
        if self.streams == 1:
            return w, b
        return torch.cat((w, torch.cat((w[:, k:], w[:, :k]), 1)), 0), torch.cat((b, b))

    def forward(self, x):
        # Taps + elementwise products, like CausalBlock.dilated_conv: no slow deterministic dilated Conv1d.
        padded, length, d = F.pad(x, (self.left_padding, 0)), x.shape[-1], self.dilation
        taps = torch.stack([padded[:, :, k*d:k*d + length] for k in range(3)], -1)  # [B, S*K, T, 3]
        depthwise = (taps * self.depthwise.weight[:, 0].repeat(self.streams, 1)[:, None, :]).sum(-1)
        weight, bias = self.pointwise_weight()
        update = self.dropout(F.relu(torch.matmul(weight, depthwise) + bias[:, None]))
        return F.relu(x + update)


class CompactTCNN(GravityMirrorTCNN):
    """GravityMirrorTCNN plus dilation_base, separable, head_basis and speed_feature (see the module docstring).

    The first four arguments keep train_tcnn's positional order (context, horizon, channels, dropout).
    """

    def __init__(self, context=400, horizon=20, channels=48, dropout=.1, effective_context=None,
                 velocity_features=False, damped_prior_tau_ms=None, input_features="body_velocity",
                 num_blocks=8, gravity_features=False, mirror_equivariant=False, dilation_base=2,
                 separable=False, head_basis=None, speed_feature="norm"):
        super().__init__(context, horizon, channels, dropout, effective_context, velocity_features,
                         damped_prior_tau_ms, input_features, num_blocks, gravity_features, mirror_equivariant)
        if speed_feature not in ("norm", "l1"):
            raise ValueError("speed_feature must be norm or l1")
        if head_basis is not None and not 1 <= head_basis <= horizon:
            raise ValueError("head_basis must be None or 1..horizon")
        self.dilation_base, self.separable = dilation_base, separable
        self.head_basis, self.speed_feature = head_basis, speed_feature
        # The v6 modules stay unless an option replaces them, so the defaults reproduce v6 exactly.
        if dilation_base != 2 or separable:
            if separable:
                blocks = (SeparableBlock(self.stream_channels, self.streams, dilation_base**i, dropout)
                          for i in range(num_blocks))
            else:
                block = MirrorCausalBlock if mirror_equivariant else CausalBlock
                blocks = (block(self.stream_channels, dilation_base**i, dropout) for i in range(num_blocks))
            self.blocks = nn.Sequential(*blocks)
        if head_basis is not None:
            self.head = nn.Linear(self.stream_channels, head_basis * 3, bias=not mirror_equivariant)
            nn.init.zeros_(self.head.weight)
            if mirror_equivariant:
                self.head_bias_x = nn.Parameter(torch.zeros(head_basis))
            else:
                nn.init.zeros_(self.head.bias)
            # GravityMirrorTCNN.readout applies these signs to the mirrored stream's outputs.
            self.register_buffer("output_signs", torch.tensor(ROTATION_SIGNS * head_basis), persistent=False)
            time = torch.arange(1, horizon + 1, dtype=torch.float64) / horizon
            self.register_buffer("basis", torch.stack([time ** (j + 1) for j in range(head_basis)], 1).float(),
                                 persistent=False)  # [horizon, P]

    def features(self, window):
        features, last, step = super().features(window)
        if self.speed_feature == "l1":
            features = torch.cat((features[..., :3], features[..., :3].abs().sum(-1, keepdim=True),
                                  features[..., 4:]), -1)
        return features, last, step

    def readout(self, last):
        """[B, horizon * 3] rotation-vector residuals, as GravityMirrorTCNN.readout."""
        outputs = super().readout(last)
        if self.head_basis is None:
            return outputs
        return torch.matmul(self.basis, outputs.reshape(-1, self.head_basis, 3)).flatten(1)

    def specification(self):
        return dict(super().specification(), dilation_base=self.dilation_base, separable=self.separable,
                    head_basis=self.head_basis, speed_feature=self.speed_feature)


def load_checkpoint(path):
    """v7 checkpoints, and v6 ones too: their specification lacks the v7 keys, which then take the v6 defaults."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = CompactTCNN(**state["model_spec"])
    model.load_state_dict(state["model_state"])
    return model.eval(), state
