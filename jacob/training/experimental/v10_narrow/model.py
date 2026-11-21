"""NarrowTCNN: LastPositionTCNN whose blocks can read fewer channels than they write.

Why: in the streaming graph a block keeps its last (k - 1) d inputs as state, for every channel it reads. For
k8_combined's last block that is 329 samples x 64 channels = 21,056 floats, 84 % of the model's state, and every
call copies it twice (9 Nov profile: 27 % of the fast graph's time on the Mac). Long-range context may not need 32
channels a stream.

block_inputs: null (every block reads all K channels of each stream: LastPositionTCNN), or one integer per block,
1..K. A block with inputs c < K is a NarrowBlock: its depthwise conv filters the first c channels of each stream,
and its pointwise conv maps those 2c values to all K channels of each stream, [[A, B], [B, A]] with A and B [K, c].
The residual update still reaches every channel, so the later layers and the head are unchanged.

k8_combined with block_inputs [32, 32, 8]: 5,827 parameters instead of 7,555, and 9,296 state floats instead of
25,088.
"""
import torch
from torch import nn
from torch.nn import functional as F

from best.model import KernelBlock

from experimental.v10_speed.model import LastPositionTCNN


class NarrowBlock(KernelBlock):
    """KernelBlock that reads the first `inputs` channels of each stream and updates all of them."""

    def __init__(self, stream_channels, streams, dilation, dropout, kernel_size, inputs):
        super().__init__(stream_channels, streams, dilation, dropout, kernel_size)
        self.inputs = inputs
        self.depthwise = nn.Conv1d(inputs, inputs, kernel_size, dilation=dilation, groups=inputs, bias=False)
        self.pointwise = nn.Conv1d(streams * inputs, stream_channels, 1)  # [A | B], each [K, inputs]

    def pointwise_weight(self):
        """[S*K, S*inputs] weight and [S*K] bias; for two streams [[A, B], [B, A]] from the stored [A, B]."""
        w, b, c = self.pointwise.weight[:, :, 0], self.pointwise.bias, self.inputs
        if self.streams == 1:
            return w, b
        return torch.cat((w, torch.cat((w[:, c:], w[:, :c]), 1)), 0), torch.cat((b, b))

    def from_taps(self, taps):
        """taps [B, S*K, outputs, k], each output's k inputs oldest first, to the block output [B, S*K, outputs]."""
        read = taps.unflatten(1, (self.streams, -1))[:, :, :self.inputs].flatten(1, 2)  # [B, S*inputs, outputs, k]
        depthwise = (read * self.depthwise.weight[:, 0].repeat(self.streams, 1)[:, None, :]).sum(-1)
        weight, bias = self.pointwise_weight()
        update = self.dropout(F.relu(torch.matmul(weight, depthwise) + bias[:, None]))
        return F.relu(taps[..., -1] + update)

    def forward(self, x):
        padded, length, d = F.pad(x, (self.left_padding, 0)), x.shape[-1], self.dilation
        taps = torch.stack([padded[:, :, k*d:k*d + length] for k in range(self.kernel_size)], -1)
        return self.from_taps(taps)


class NarrowTCNN(LastPositionTCNN):
    def __init__(self, *args, block_inputs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_inputs = block_inputs
        if block_inputs is None:
            return
        if not self.separable or len(block_inputs) != len(self.blocks):
            raise ValueError("block_inputs needs separable blocks and one entry per block")
        if any(not 1 <= c <= self.stream_channels for c in block_inputs):
            raise ValueError("block_inputs entries must be 1..channels per stream")
        for i, (block, c) in enumerate(zip(self.blocks, block_inputs)):
            if c < self.stream_channels:
                self.blocks[i] = NarrowBlock(self.stream_channels, self.streams, block.dilation, self.dropout_rate,
                                             block.depthwise.kernel_size[0], c)

    def specification(self):
        return dict(super().specification(), block_inputs=self.block_inputs)


def load_checkpoint(path):
    """best.model.load_checkpoint for checkpoints with or without block_inputs."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = NarrowTCNN(**state["model_spec"])
    model.load_state_dict(state["model_state"])
    return model.eval(), state
