"""The v7 streaming graph with each mirror block's conv split into a sum part and a difference part.

A mirror block applies one stored weight [A, B] to both streams: stream 0 gets A h0 + B h1 + b and stream 1 gets
B h0 + A h1 + b. On the sum s = h0 + h1 and the difference d = h0 - h1 that matrix [[A, B], [B, A]] separates:
    P = (A + B)/2 s + b,   Q = (A - B)/2 d,   stream 0 = P + Q,   stream 1 = P - Q.
Two K x K convs replace two K x 2K ones, so each block's conv MACs halve. It is the same function (float32
rounding only) and the file still stores 2K^2 numbers per tap. Depthwise taps, projection and head are unchanged;
models without mirror streams export exactly as in export.py.
Sub is off the safe list, so d is h0 + h1 * -1 and the streams are P + Q * [1, -1], broadcast over the stream axis
as input_signs is; that costs 4 more nodes per block than the reduced graph's Slice, Slice, Concat, Concat, Conv.

Kept out of export.py, which is hashed into the v7 suites' manifests.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from experimental.v6_gravity_mirror.hooks import _swapped
from experimental.v7_compact import export
from experimental.v7_compact.model import SeparableBlock


class StreamingSplit(export.StreamingCompact):
    """StreamingCompact whose mirror blocks run on the sum and the difference of the two streams."""

    def __init__(self, model):
        super().__init__(model)
        if not model.mirror_equivariant:
            return
        k = model.stream_channels
        self.register_buffer("stream_signs", torch.tensor([1., -1.])[:, None, None], persistent=False)  # [2, 1, 1]
        for i, block in enumerate(model.blocks):
            conv = block.pointwise if isinstance(block, SeparableBlock) else block.conv
            own, other = conv.weight.detach()[:, :k], conv.weight.detach()[:, k:]  # A and B, [K, K, taps]
            # The 1/2 that turns P +- Q back into the two streams is folded into the stored weights. Parameters,
            # not buffers, so the file stores them as initializers, like the trained [A, B] they replace.
            setattr(self, f"half_sum_{i}", nn.Parameter((own + other) / 2, requires_grad=False))
            setattr(self, f"half_difference_{i}", nn.Parameter((own - other) / 2, requires_grad=False))

    def block_update(self, block, window):
        """The block's conv output at the newest step, as StreamingCompact.block_update."""
        if not self.model.mirror_equivariant:
            return super().block_update(block, window)
        i, dilation = list(self.model.blocks).index(block), block.dilation
        if isinstance(block, SeparableBlock):
            # Depthwise weights are shared by both streams, so they act on h0 and h1 before the split.
            window = F.conv1d(window, block.depthwise.weight, dilation=dilation, groups=block.depthwise.groups)
            conv, dilation = block.pointwise, 1
        else:
            conv = block.conv
        own, other = window[:1], window[1:]
        plus = F.conv1d(own + other, getattr(self, f"half_sum_{i}"), conv.bias, dilation=dilation)  # P
        minus = F.conv1d(own + other * -1., getattr(self, f"half_difference_{i}"), dilation=dilation)  # Q
        return plus + minus * self.stream_signs  # [P + Q; P - Q], [2, K, 1]


def export_streaming_onnx(model, path, compact=True, trig_free=True):
    """export.export_streaming_onnx with StreamingSplit in place of StreamingCompact; same files and returns."""
    with _swapped(export, StreamingCompact=StreamingSplit):
        return export.export_streaming_onnx(model, path, compact, trig_free)
