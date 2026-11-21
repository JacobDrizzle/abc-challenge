"""FastLeanTCNN: LeanTCNN with a forward that computes only what the last time step reads.

LeanTCNN.forward runs every block at all T = 400 time steps and then reads the last one, x[:, :, -1]. But a block's
output at step t depends on its input at that step's taps only: t, t - d, ..., t - (k - 1) d. Working back from the
last step, k8_combined (kernels 8, dilations 1, 8, 47) needs
  block 3 at 1 step                  (T - 1)
  block 2 at that step's 8 taps      (T - 1 - 47 j)
  block 1 at their 8 taps each, 64   (T - 1 - 47 j - 8 m)
  the projection at 8 taps of each   (T - 1 - 47 j - 8 m - n): 512 reads of 393 different steps, the receptive field
so 73 block evaluations a window where LeanTCNN does 1,200. FastLeanTCNN gathers the projected features at those 512
positions as a [8, 8, 8] grid and lets each block consume one axis of it. The arithmetic at every step it keeps is
LeanTCNN's, so the two give the same forecast up to float32 rounding (the sums run in another order); the parameters,
their order, specification() and checkpoints are LeanTCNN's.

Dropout: LeanTCNN draws one mask [B, C, T] a block. This forward draws the same mask from the same generator state
and reads it at the block's output steps, so with the same seed a training step drops the same activations.

Training is therefore the same recipe but not bit-identical: rounding differences grow over a run as they do between
two machines. A run made with this class is compared with a control made with this class.
"""
import torch
from torch.nn import functional as F

from tcnn.model import qmul

from best.model import LeanTCNN, map_quaternion


def tap_positions(blocks, length):
    """[k_n, ..., k_1] positions in 0 .. length - 1: the steps of the first block's input that the last output step
    reads. Block i owns axis i counted from the right; along an axis the taps run from the oldest to the newest
    (offset 0), the order of the depthwise weights. Negative entries would be left padding."""
    positions = torch.tensor(length - 1)
    for block in reversed(blocks):
        taps = block.depthwise.kernel_size[0]
        positions = positions[..., None] - torch.arange(taps - 1, -1, -1) * block.dilation
    return positions


class FastLeanTCNN(LeanTCNN):
    """LeanTCNN, same arguments. Separable models whose receptive field fits the window take the short forward; any
    other model (kernel-3 full blocks, or a window shorter than the field) runs LeanTCNN.forward unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        positions = tap_positions(self.blocks, self.effective_context) if self.separable else None
        if positions is not None and positions.min() < 0:
            positions = None  # LeanTCNN pads on the left there
        # Not persistent: the state dict stays LeanTCNN's, so checkpoints load in either class.
        self.register_buffer("positions", positions, persistent=False)

    def forward(self, history):
        if self.positions is None:
            return super().forward(history)
        if history.ndim != 3 or history.shape[1:] != (self.context, 4):
            raise ValueError(f"Expected [B,{self.context},4], got {tuple(history.shape)}")
        # As LeanTCNN.forward up to the blocks.
        features, last, step = self.features(history[:, -self.effective_context:])
        x = features.transpose(1, 2)
        if self.mirror_equivariant:
            x = torch.cat((self.projection(x), self.projection(x * self.input_signs[:, None])), 1)
        else:
            x = self.projection(x)
        x = F.relu(x)  # [B, C, T]
        batch, channels, length = x.shape
        x = x.index_select(2, self.positions.reshape(-1)).reshape(batch, channels, *self.positions.shape)
        for depth, block in enumerate(self.blocks, 1):
            # x is [B, C, k_n, ..., k_depth]: its last axis holds this block's taps for each step it must output.
            kernel = block.depthwise.weight[:, 0].repeat(block.streams, 1)  # [C, k]
            depthwise = (x * kernel.reshape(channels, *[1] * (x.ndim - 3), -1)).sum(-1)
            weight, bias = block.pointwise_weight()
            steps = depthwise.reshape(batch, channels, -1)  # the last block has one step and no axis left for it
            update = F.relu(torch.matmul(weight, steps) + bias[:, None])  # [B, C, steps]
            if self.training and block.dropout.p > 0:
                # LeanTCNN's mask for this block, [B, C, T], read at the steps this block outputs: the entries of
                # positions at offset 0 on this block's axis and on every earlier block's.
                mask = block.dropout(update.new_ones(batch, channels, length))
                update = update * mask.index_select(2, self.positions[(..., *[-1] * depth)].reshape(-1))
            x = F.relu(x[..., -1] + update.reshape(x.shape[:-1]))  # the residual is the tap at offset 0
        # x is [B, C]: the last block's output at the last step. From here on, LeanTCNN.forward again.
        r = self.readout(x).reshape(-1, self.horizon, 3)
        if self.damped_prior_tau_ms is not None:
            tau = self.damped_prior_tau_ms / 10  # ms to 100 Hz samples
            steps = torch.arange(1, self.horizon+1, dtype=step.dtype, device=step.device)
            r = r + step[:, None, :] * (-tau * torch.expm1(-steps / tau))[None, :, None]
        return F.normalize(qmul(last, map_quaternion(r, self.maps)), dim=-1)
