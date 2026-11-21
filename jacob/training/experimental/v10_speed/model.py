"""LastPositionTCNN: LeanTCNN with two more ways to run the same network.

LeanTCNN.forward runs every block at all 400 positions of a window and then reads only the last one. A block's output
at time t reads its input at t, t - d, ..., t - (k - 1) d, so the last position needs only a few of them:

  k8_combined (kernel 8, dilations 1, 8, 47)   positions needed
  block 2 (d = 47)                             1     (t)
  block 1 (d = 8)                              8     (t - 47 j)
  block 0 (d = 1)                              64    (t - 47 j - 8 i)
  projection                                   393   (t - 392 ... t, the receptive field)

forward (last_position_forward=True) computes exactly those: 73 block positions a window instead of 1,200. The
needed inputs are laid out so that the taps of one output sit in consecutive columns (lattice_lags), and each block
then reads k columns at a time. Same weights and same arithmetic per position, so the forecast equals LeanTCNN's up
to float32 rounding (about 1e-7); test_v10.py checks it. In training, dropout draws fewer random numbers, so a run
is not bit-identical to one on LeanTCNN.forward and needs its own control.

dense(q, anchors) is for validation: one pass along a whole stream gives the block outputs at every sample, and the
forecast for the window ending at sample a is read at position a. The receptive field (393) is shorter than a
window (400), so position a sees the same 393 samples either way, as long as they belong to a's recording.

Checkpoints are unchanged: the class adds no parameters and no specification key, so best.model.load_checkpoint
loads them as LeanTCNN.
"""
import torch
from torch.nn import functional as F

from tcnn.model import qmul

from best.model import LeanTCNN, SeparableBlock, map_quaternion, receptive_field


def lattice_lags(kernels, dilations):
    """Lags (samples before the last one) of the projected features the last position reads, one column per
    (block 0 tap, block 1 tap, ...) combination, ordered so that every block finds one output's taps in consecutive
    columns, oldest first. A lag reached twice is listed twice.

    Built from the last block down: its single output (lag 0) reads lags (k-1)d, ..., d, 0; each of those is an
    output of the block below, which reads its own k taps behind it; and so on."""
    lags = torch.zeros(1, dtype=torch.long)
    for k, d in zip(reversed(kernels), reversed(dilations)):
        lags = (lags[:, None] + torch.arange(k - 1, -1, -1) * d).reshape(-1)
    return lags


def block_on_taps(block, x, kernel_size):
    """SeparableBlock / KernelBlock.forward for inputs that are already taps: x [B, S*K, outputs * k] holds, for each
    output, its k inputs in consecutive columns, oldest first. Returns [B, S*K, outputs]."""
    taps = x.unflatten(2, (-1, kernel_size))  # [B, S*K, outputs, k]
    if hasattr(block, "from_taps"):  # a block with its own arithmetic (experimental/v10_narrow)
        return block.from_taps(taps)
    depthwise = (taps * block.depthwise.weight[:, 0].repeat(block.streams, 1)[:, None, :]).sum(-1)
    weight, bias = block.pointwise_weight()
    update = block.dropout(F.relu(torch.matmul(weight, depthwise) + bias[:, None]))
    return F.relu(taps[..., -1] + update)  # the newest tap is the block's own input at that position


class LastPositionTCNN(LeanTCNN):
    def __init__(self, *args, last_position_forward=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_position_forward = last_position_forward
        self.lattice_kernels = None
        if all(isinstance(block, SeparableBlock) for block in self.blocks):
            kernels = [block.depthwise.kernel_size[0] for block in self.blocks]
            dilations = [block.dilation for block in self.blocks]
            self.lattice_kernels = kernels
            self.span = receptive_field(len(kernels), kernel_sizes=kernels, dilations=dilations)
            # Column of each lag in the newest `span` positions: lag 0 is the last column.
            self.register_buffer("lattice", self.span - 1 - lattice_lags(kernels, dilations), persistent=False)

    def project(self, features):
        """[B, N, F] features to the block input [B, S*K, N]: the observed stream and, with mirror equivariance,
        the mirrored one."""
        x = features.transpose(1, 2)
        if self.mirror_equivariant:
            x = torch.cat((self.projection(x), self.projection(x * self.input_signs[:, None])), 1)
        else:
            x = self.projection(x)
        return F.relu(x)

    def forecast(self, hidden, last, step):
        """LeanTCNN.forward from the last block output on: hidden [B, S*K], the last orientation [B, 1, 4] and the
        last step's rotation vector [B, 3] to unit quaternions [B, horizon, 4]."""
        r = self.readout(hidden).reshape(-1, self.horizon, 3)
        if self.damped_prior_tau_ms is not None:
            tau = self.damped_prior_tau_ms / 10  # ms to 100 Hz samples
            steps = torch.arange(1, self.horizon+1, dtype=step.dtype, device=step.device)
            r = r + step[:, None, :] * (-tau * torch.expm1(-steps / tau))[None, :, None]
        return F.normalize(qmul(last, map_quaternion(r, self.maps)), dim=-1)

    def forward(self, history):
        # The oldest feature row of a window has no rate (LeanTCNN.features), so the receptive field must not reach it.
        if not self.last_position_forward or self.lattice_kernels is None or self.span >= self.effective_context:
            return super().forward(history)
        if history.ndim != 3 or history.shape[1:] != (self.context, 4):
            raise ValueError(f"Expected [B,{self.context},4], got {tuple(history.shape)}")
        features, last, step = self.features(history[:, -self.effective_context:])
        x = self.project(features[:, -self.span:])  # the receptive field: [B, S*K, span]
        x = x.index_select(2, self.lattice)  # every tap the last position reads, block 0's taps together
        for block, kernel_size in zip(self.blocks, self.lattice_kernels):
            x = block_on_taps(block, x, kernel_size)
        return self.forecast(x[:, :, 0], last, step)

    def dense(self, q, anchors):
        """Forecasts [len(anchors), horizon, 4] for the windows ending at samples `anchors` of one stream q [N, 4].

        Each anchor must have span - 1 samples of its own recording before it in q (and one more for the oldest
        rate): then no zero padding and no other recording reaches it."""
        features, _, _ = self.features(q[None])  # [1, N, F]
        x = self.blocks(self.project(features))  # [1, S*K, N]
        hidden = x[0].index_select(1, anchors).T  # [A, S*K]
        step = features[0].index_select(0, anchors)[:, :3] / 100.  # features carry the rate, 100 steps a second
        last = F.normalize(q.index_select(0, anchors), dim=-1)[:, None]
        return self.forecast(hidden, last, step)
