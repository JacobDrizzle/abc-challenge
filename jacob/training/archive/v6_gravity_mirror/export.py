"""Streaming ONNX graph for GravityMirrorTCNN, with the harness interface of tcnn.onnx_export.StreamingTCNN.

The two mirror streams travel in the batch axis ([2, K, T]); each block's Conv reads cat(h, swap(h)) on the
channel axis with the stored W [K, 2K, 3], so the file holds the shared weights once and the parameter count
can be read from it. Gravity comes from the newest sample only; quaternion maths is MatMul and polynomials.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tcnn.onnx_export import (OPSET, _bilinear_tables, _compact_exp, _compact_rotvec, _exp_quaternion,
                              _left_matrix, _normalize, _outer_product, _rotvec)


def _gravity_table():
    """[16, 4]: outer(q, q) flattened @ table == (R(q)^T y * |q|^2, |q|^2), see model.gravity_in_head_frame."""
    table = torch.zeros(4, 4, 4)
    for (j, k), column in {(1, 2): (0, 1), (0, 3): (0, 1), (0, 0): (1, 1), (1, 1): (1, -1), (2, 2): (1, 1),
                           (3, 3): (1, -1), (2, 3): (2, 1), (0, 1): (2, -1)}.items():
        axis, sign = column
        table[j, k, axis] += sign
        if j != k:
            table[k, j, axis] += sign
    table[range(4), range(4), 3] = 1
    return table.reshape(16, 4)


class StreamingGravityMirror(nn.Module):
    """Inputs context [1, 400, 4] (last two samples read) and state_in_i [S, K, 2 * 2**i]; S = 2 with mirror."""

    def __init__(self, model, trig_free=True):
        super().__init__()
        if model.input_features != "body_velocity":
            raise ValueError("Streaming needs body_velocity inputs")
        self.model, self.trig_free = model.eval(), trig_free
        self.state_shapes = [(model.streams, model.stream_channels, block.left_padding) for block in model.blocks]
        for name in ("aconj_b", "left"):
            self.register_buffer(f"table_{name}", _bilinear_tables()[name].float(), persistent=False)
        self.register_buffer("gravity_table", _gravity_table(), persistent=False)
        # Stream 0 sees the input, stream 1 its mirror image; the readout of stream 1 is mirrored back.
        self.register_buffer("input_signs", torch.stack((torch.ones_like(model.input_signs), model.input_signs))
                             [:, :, None], persistent=False)
        if model.mirror_equivariant:
            self.register_buffer("bias", F.pad(model.head_bias_x.detach()[:, None], (0, 2)).reshape(1, -1),
                                 persistent=False)
        if model.damped_prior_tau_ms is not None:
            tau = model.damped_prior_tau_ms / 10
            steps = torch.arange(1, model.horizon + 1, dtype=torch.float64)
            self.register_buffer("elapsed", (-tau * torch.expm1(-steps / tau)).float()[None, :, None],
                                 persistent=False)

    def forward(self, context, *states):
        m = self.model
        previous, now = context[:, -2:-1], context[:, -1:]
        step = (_compact_rotvec if self.trig_free else _rotvec)(_outer_product(self.table_aconj_b, previous, now))
        velocity = step * 100.
        parts = [velocity, torch.sqrt((velocity * velocity).sum(-1, keepdim=True))]
        if m.gravity_features:
            squares = torch.matmul(torch.matmul(now.reshape(1, 4, 1), now).reshape(1, 1, 16), self.gravity_table)
            parts.append(squares[..., :3] / squares[..., 3:])  # scale-invariant: no normalisation needed
        x = torch.cat(parts, -1).reshape(1, -1, 1)
        if m.mirror_equivariant:
            x = x * self.input_signs  # [2, F, 1]
        x = F.relu(m.projection(x))
        new_states = []
        for block, state in zip(m.blocks, states):
            window = torch.cat((state, x), -1)  # block input at t - 2d ... t
            new_states.append(window[:, :, 1:])
            if m.mirror_equivariant:
                window = torch.cat((window, torch.cat((window[1:], window[:1]), 0)), 1)
            update = F.conv1d(window, block.conv.weight, block.conv.bias, dilation=block.dilation)
            x = F.relu(x + F.relu(update))
        if m.mirror_equivariant:
            out = F.linear(x.reshape(2, m.stream_channels), m.head.weight)
            rotation_vectors = out[0:1] + out[1:2] * m.output_signs + self.bias
        else:
            rotation_vectors = m.head(x.reshape(1, m.stream_channels))
        rotation_vectors = rotation_vectors.reshape(1, m.horizon, 3)
        if m.damped_prior_tau_ms is not None:
            rotation_vectors = rotation_vectors + step * self.elapsed
        exp = _compact_exp(rotation_vectors) if self.trig_free else _exp_quaternion(rotation_vectors)
        return (_normalize(torch.matmul(exp, _left_matrix(self.table_left, now))), *new_states)


def export_streaming_onnx(model, path, compact=True, trig_free=True):
    """Same signature and outputs as tcnn.onnx_export.export_streaming_onnx; the graph is always compact."""
    if not compact:
        raise ValueError("The v6 streaming graph is always compact")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = StreamingGravityMirror(model, trig_free).eval()
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
    raise ValueError("v6 models export only the streaming graph: use --variants stream")
