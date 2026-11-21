"""Streaming ONNX graph for CompactTCNN (and v6 checkpoints) with fewer operators off the challenge's safe list.

Same interface as v6's StreamingGravityMirror: context [1, 400, 4] (last two samples read) and state_in_i in;
prediction [1, 20, 4] and state_out_i out. The maths differs in three ways, each relying on unit-norm inputs
("normalised head-motion quaternions", The Challenge page):
- rate: for the step d = conj(q[t-1]) q[t], s = 2 dw dv = sin(theta) n and u = 2 |dv|^2 = 1 - cos(theta)
  ignore the quaternions' signs, and the rotation vector is g(u) s with g = theta / sin(theta), a degree-8
  polynomial. u is clipped at 1, so steps up to 90 degrees are exact to float32 (data maximum: 66).
- squared norms are MatMuls with a ones vector instead of ReduceSum; gravity skips the division by |q|^2.
- the output is not normalised (unit norm to about 1e-6); predictions are re-normalised before scoring.
With speed_feature="l1" every operator is on the safe list; "norm" keeps one Sqrt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experimental.v6_gravity_mirror.export import _gravity_table
from experimental.v7_compact.model import SeparableBlock
from tcnn.onnx_export import COS_SINC_SERIES, OPSET, _bilinear_tables, _horner, _left_matrix, _outer_product, _power_series


def _theta_over_sine(u):
    """theta / sin(theta) for u = 1 - cos(theta), theta in [0, 90] degrees."""
    theta = np.arccos(1 - u)
    return np.where(u > 1e-12, theta / np.sqrt(np.maximum(u * (2 - u), 1e-300)), 1.)


# Relative error 3e-7 in float32 over 0-90 degrees (checked 26 Oct); built once, outside any trace.
RATE_SERIES = [float(c) for c in _power_series(_theta_over_sine, 1., 8)]
# COS_SINC_SERIES with the sinc column halved, so the exp map needs no separate multiplication by 1/2.
HALF_SINC_SERIES = [c * torch.tensor([1., .5]) for c in COS_SINC_SERIES]


class StreamingCompact(nn.Module):
    """Inputs context [1, 400, 4] and state_in_i [S, K, 2 * dilation_i]; S = 2 with mirror."""

    def __init__(self, model):
        super().__init__()
        if model.input_features != "body_velocity":
            raise ValueError("Streaming needs body_velocity inputs")
        self.model = model.eval()
        self.state_shapes = [(model.streams, model.stream_channels, block.left_padding) for block in model.blocks]
        tables = _bilinear_tables()
        self.register_buffer("table_aconj_b", tables["aconj_b"].float(), persistent=False)
        self.register_buffer("table_left", tables["left"].float(), persistent=False)
        # Unit inputs: the first three columns are already R(q)^T y, and the |q|^2 column is not needed.
        self.register_buffer("gravity_table", _gravity_table()[:, :3].contiguous(), persistent=False)
        self.register_buffer("twos", torch.full((3, 1), 2.), persistent=False)
        self.register_buffer("ones", torch.ones(3, 1), persistent=False)
        self.register_buffer("quarters", torch.full((3, 1), .25), persistent=False)
        self.register_buffer("plus_minus", torch.cat((torch.eye(3), -torch.eye(3)), 1), persistent=False)
        self.register_buffer("ones6", torch.ones(6, 1), persistent=False)
        # Stream 0 sees the input, stream 1 its mirror image; the readout of stream 1 is mirrored back.
        self.register_buffer("input_signs", torch.stack((torch.ones_like(model.input_signs), model.input_signs))
                             [:, :, None], persistent=False)
        if model.mirror_equivariant:
            bias = F.pad(model.head_bias_x.detach()[:, None], (0, 2)).reshape(1, -1)
            self.register_buffer("bias", bias, persistent=False)
        self.prior = model.damped_prior_tau_ms is not None
        if self.prior:
            tau = model.damped_prior_tau_ms / 10
            steps = torch.arange(1, model.horizon + 1, dtype=torch.float64)
            elapsed = (-tau * torch.expm1(-steps / tau)).float()[:, None]  # [horizon, 1]
        if model.head_basis is None:
            if self.prior:
                self.register_buffer("elapsed", elapsed[None], persistent=False)
        else:
            # basis @ coefficients + elapsed * step == [basis, elapsed] @ [coefficients; step]: one MatMul.
            basis = torch.cat((model.basis, elapsed), 1) if self.prior else model.basis
            self.register_buffer("basis", basis, persistent=False)

    def mirrored(self, h):
        """[2, K, T] -> [2, 2K, T]: each stream followed by the other, the input of a stored [A, B] weight."""
        if not self.model.mirror_equivariant:
            return h
        return torch.cat((h, torch.cat((h[1:], h[:1]), 0)), 1)

    def block_update(self, block, window):
        """The block's conv output at the newest step, from its inputs at t - 2d, t - d and t."""
        if isinstance(block, SeparableBlock):
            depthwise = F.conv1d(window, block.depthwise.weight, dilation=block.dilation, groups=block.depthwise.groups)
            return F.conv1d(self.mirrored(depthwise), block.pointwise.weight, block.pointwise.bias)
        return F.conv1d(self.mirrored(window), block.conv.weight, block.conv.bias, dilation=block.dilation)

    def forward(self, context, *states):
        m = self.model
        previous, now = context[:, -2:-1], context[:, -1:]
        delta = _outer_product(self.table_aconj_b, previous, now)  # conj(q[t-1]) q[t], [1, 1, 4]
        vector = delta[..., 1:4]
        sine = delta[..., 0:1] * vector * 2.  # sin(theta) n
        u = torch.matmul(vector * vector, self.twos).clamp(max=1.)  # 1 - cos(theta)
        step = sine * _horner(u, RATE_SERIES)  # rotation vector of the last step
        velocity = step * 100.
        if m.speed_feature == "l1":
            speed = torch.matmul(F.relu(torch.matmul(velocity, self.plus_minus)), self.ones6)
        else:
            speed = torch.sqrt(torch.matmul(velocity * velocity, self.ones))
        parts = [velocity, speed]
        if m.gravity_features:
            outer = torch.matmul(now.reshape(1, 4, 1), now).reshape(1, 1, 16)
            parts.append(torch.matmul(outer, self.gravity_table))
        x = torch.cat(parts, -1).reshape(1, -1, 1)
        if m.mirror_equivariant:
            x = x * self.input_signs  # [2, F, 1]
        x = F.relu(m.projection(x))
        new_states = []
        for block, state in zip(m.blocks, states):
            window = torch.cat((state, x), -1)  # block input at t - 2d ... t
            new_states.append(window[:, :, 1:])
            x = F.relu(x + F.relu(self.block_update(block, window)))
        if m.mirror_equivariant:
            out = F.linear(x.reshape(2, m.stream_channels), m.head.weight)
            outputs = out[0:1] + out[1:2] * m.output_signs + self.bias
        else:
            outputs = m.head(x.reshape(1, m.stream_channels))
        if m.head_basis is None:
            rotation_vectors = outputs.reshape(1, m.horizon, 3)
            if self.prior:
                rotation_vectors = rotation_vectors + step * self.elapsed
        else:
            coefficients = outputs.reshape(1, m.head_basis, 3)
            if self.prior:
                coefficients = torch.cat((coefficients, step), 1)
            rotation_vectors = torch.matmul(self.basis, coefficients)  # [1, horizon, 3]
        cos_sinc = _horner(torch.matmul(rotation_vectors * rotation_vectors, self.quarters), HALF_SINC_SERIES)
        exp = torch.cat((cos_sinc[..., 0:1], cos_sinc[..., 1:2] * rotation_vectors), -1)  # unnormalised
        return (torch.matmul(exp, _left_matrix(self.table_left, now)), *new_states)


def export_streaming_onnx(model, path, compact=True, trig_free=True):
    """Same signature and outputs as the v5/v6 exporters; the graph is always compact and trig-free."""
    if not (compact and trig_free):
        raise ValueError("The v7 streaming graph is always compact and trig-free")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = StreamingCompact(model).eval()
    example = (F.normalize(torch.randn(1, model.context, 4), dim=-1), *(torch.zeros(s) for s in wrapper.state_shapes))
    count = len(wrapper.state_shapes)
    with torch.no_grad():
        torch.onnx.export(wrapper, example, str(path), opset_version=OPSET, do_constant_folding=True, dynamo=False,
                          input_names=["context", *(f"state_in_{i}" for i in range(count))],
                          output_names=["prediction", *(f"state_out_{i}" for i in range(count))])
    model.eval()
    import onnx
    onnx.checker.check_model(onnx.load(str(path)))
    return path, wrapper.state_shapes


def export_onnx(*args, **kwargs):
    raise ValueError("v7 models export only the streaming graph: use --variants stream")
