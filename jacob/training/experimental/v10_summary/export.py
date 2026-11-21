"""Streaming ONNX graph for a summary model: best/export.py's graph with the last block's history buffer replaced by
its moving averages.

Interface as best/export.py: context [1, 400, 4] (last two samples read) and state_in_i in; prediction [1, 20, 4]
and state_out_i out. For blocks 0 and 1 state_in_i is the block's input history, [1, K, 2, (k - 1) d], exactly as in
k8_combined's graph. For the summary block it holds the J moving averages, [1, K, 2, J], slowest first. Each call:

    averages = state * a + x * (1 - a)          one Mul, one Mul, one Add; a = exp(-1/tau), slowest first
    taps     = Concat(averages, x)              [1, K, 2, J + 1]
    then the KernelBlock arithmetic on taps     depthwise Conv (J + 1 taps, no dilation), pointwise Conv, ReLUs

state_out is `averages`, so the stream runs SummaryBlock.step: the recursion SummaryTCNN.streamed computes. Zero
states at the session start are the averages' zero start. The rest of the graph (features, blocks 0 and 1, head,
output) is StreamingLean.forward with its default options; st_layout is kept (ST Edge AI Core 4.1's LayerNorm bug).
fused_head, padded_rows and split_mirror are not supported here.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from best.export import OPSET, StreamingLean, _left_matrix

from experimental.v10_summary.model import SummaryBlock


class StreamingSummary(StreamingLean):
    def __init__(self, model, st_layout=False):
        if not isinstance(model.blocks[-1], SummaryBlock):
            raise ValueError("StreamingSummary needs a SummaryTCNN with summary_taus; use best/export.py otherwise")
        super().__init__(model, st_layout=st_layout)
        summary = model.blocks[-1]
        # StreamingLean sized the last state from left_padding = (J + 1 - 1) * 1 = J: one slot per average.
        assert self.state_shapes[-1] == (1, model.stream_channels, 2, len(summary.taus))
        self.register_buffer("keep", summary.decays.detach().clone(), persistent=False)  # a, slowest first, [J]
        self.register_buffer("take", 1 - summary.decays.detach(), persistent=False)  # 1 - a

    def forward(self, context, *states):
        # StreamingLean.forward with default options; only the last block's window is built differently.
        m, p = self.model, self.model.head_basis
        previous, now = context[:, -2:-1], context[:, -1:]
        delta = torch.matmul(now, _left_matrix(self.table_left_conj, previous))
        velocity = delta[..., 0:1] * delta[..., 1:4]
        if self.maps == "lean":
            velocity = velocity * torch.matmul(delta * delta, self.rate_weights)
        speed = torch.matmul(F.relu(torch.matmul(velocity, self.plus_minus)), self.ones6)
        parts = [velocity, speed]
        if m.gravity_features:
            outer = torch.matmul(now.reshape(1, 4, 1), now).reshape(1, 1, 16)
            parts.append(torch.matmul(outer, self.gravity_table))
        x = torch.cat(parts, -1).reshape(1, -1, 1, 1) * self.input_signs
        x = F.relu(F.conv2d(x, self.projection, m.projection.bias))
        new_states = []
        for block, state, depthwise, pointwise in zip(m.blocks, states, self.depthwise, self.pointwise):
            if isinstance(block, SummaryBlock):  # the change: averages updated, then read with the input now
                averages = state * self.keep + x * self.take  # [1, K, 2, J]
                new_states.append(averages)
                window = torch.cat((averages, x), -1)
            else:
                window = torch.cat((state, x), -1)
                new_states.append(window[..., 1:])
            h = F.conv2d(window, depthwise, dilation=(1, block.dilation), groups=m.stream_channels)
            update = F.conv2d(torch.cat((h, h[:, :, 0:1]), 2), pointwise, block.pointwise.bias)
            x = F.relu(x + F.relu(update))
        both = torch.matmul(m.head.weight, x.reshape(m.stream_channels, 2))
        coefficients = torch.matmul(both * self.head_mask, self.ones2).reshape(1, p, 3)
        r = torch.matmul(self.basis, torch.cat((coefficients, velocity, self.coefficient_bias), 1))
        scalar = self.twos if self.maps == "native" else torch.matmul(r * r, self.minus_sixth) + 2.
        both = torch.matmul(torch.cat((scalar, r), -1), torch.matmul(now, self.table_left_both).reshape(1, 4, 8))
        if self.st_layout:
            unit = F.layer_norm(both.reshape(1, 1, m.horizon, 8), (8,), self.halves, eps=1e-12)
            prediction = unit.reshape(1, m.horizon, 8)[..., 0:4]
        else:
            prediction = F.layer_norm(both, (8,), self.halves, eps=1e-12)[..., 0:4]
        return (prediction, *new_states)


def export_streaming_onnx(model, path, st_layout=False):
    """Writes the streaming graph; returns the path and the state shapes."""
    import onnx

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = StreamingSummary(model, st_layout).eval()
    example = (F.normalize(torch.randn(1, model.context, 4), dim=-1), *(torch.zeros(s) for s in wrapper.state_shapes))
    count = len(wrapper.state_shapes)
    with torch.no_grad():
        torch.onnx.export(wrapper, example, str(path), opset_version=OPSET, do_constant_folding=True, dynamo=False,
                          input_names=["context", *(f"state_in_{i}" for i in range(count))],
                          output_names=["prediction", *(f"state_out_{i}" for i in range(count))])
    onnx.checker.check_model(onnx.load(str(path)))
    return path, wrapper.state_shapes
