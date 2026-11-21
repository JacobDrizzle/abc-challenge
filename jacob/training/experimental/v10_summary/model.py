"""SummaryTCNN: LeanTCNN whose last block reads running averages instead of a long history buffer.

Why (13 Nov, ST Edge AI Developer Cloud profile of the final model): in the streaming graph k8_combined's last block
keeps its last 329 inputs for all 64 channels as state, 21,056 of the model's 25,088 state floats (84 %). Every call
passes the whole state in and out and shifts it by one sample. On the STM32N6 that copying was the largest single
cost, and on an NPU target the state also crosses between the CPU and the accelerator on every step.

summary_taus: null (LeanTCNN exactly), or J time constants in samples. The last block, a SummaryBlock, then reads
for each channel its input now and J exponential moving averages of its inputs,

    s_j(t) = a_j s_j(t - 1) + (1 - a_j) x(t),    a_j = exp(-1 / tau_j),

instead of k taps spaced d apart. In a streaming graph each average is one state value per channel, updated with a
Mul and an Add, so the block keeps J x channels floats (7 x 64 = 448) instead of (k - 1) d x channels (21,056).
With J = k - 1 the block has exactly the parameters of the KernelBlock it replaces: a depthwise weight of J + 1 taps
per channel, then the same pointwise conv, residual and ReLUs. The averages are linear and the same on both mirror
streams, so mirror equivariance is kept.

Training sees 400-sample windows, not whole sessions. In a window the block's input is valid only from position
`start` on (the blocks below need start - 1 earlier samples, and the oldest feature row has no rate), so the averages
are taken over those positions only, with weights (1 - a) a^k renormalised to sum to 1. A stream that has run longer
gives its older inputs a total weight of a^(number of valid positions): with tau <= 64 and 336 valid positions that is
at most 0.5 %, so training and streaming see nearly the same averages. Longer time constants would need longer
training windows. streamed() computes the forecasts exactly as a streaming graph would, to measure that difference.
"""
import torch
from torch.nn import functional as F

from tcnn.model import qmul

from best.model import KernelBlock, LeanTCNN, SeparableBlock, map_quaternion, receptive_field


def ema_weights(taus, length, start):
    """[length, J] weights that read J moving averages at the last of `length` positions, from the inputs at positions
    start ... length - 1: column j holds (1 - a_j) a_j^k for the input k positions back, renormalised to sum to 1, and
    0 before `start`. Columns in the order of `taus`."""
    valid = length - start
    if valid < 1:
        raise ValueError(f"no valid positions: length {length}, start {start}")
    back = torch.arange(valid - 1, -1, -1, dtype=torch.float64)  # position start is valid - 1 steps back
    a = torch.exp(-1 / torch.tensor(taus, dtype=torch.float64))
    weights = (1 - a) * a ** back[:, None]  # [valid, J]
    weights = weights / weights.sum(0, keepdim=True)
    return torch.cat((weights.new_zeros(start, len(taus)), weights)).float()


class SummaryBlock(KernelBlock):
    """The last block of a SummaryTCNN. Its taps are the J moving averages, slowest first, then the input now; from
    there the arithmetic is KernelBlock's: depthwise weights, pointwise conv, ReLU, residual, ReLU. The output is the
    last position only, [B, S*K, 1], which is all the head reads, so it must be the last block."""

    def __init__(self, stream_channels, streams, dropout, taus, start):
        super().__init__(stream_channels, streams, 1, dropout, len(taus) + 1)
        self.taus, self.start = tuple(float(t) for t in taus), start
        # Decays in tap order (slowest first), and the window weights per (length, device), built when first needed.
        self.register_buffer("decays", torch.exp(-1 / torch.tensor(self.taus)).flip(0), persistent=False)
        self._weights = {}

    def window_weights(self, length, device):
        key = (length, str(device))
        if key not in self._weights:
            self._weights[key] = ema_weights(self.taus, length, self.start).flip(1).to(device)  # slowest first
        return self._weights[key]

    def from_taps(self, taps):
        """taps [..., S*K, J + 1], the averages slowest first and then the input now, to the output [..., S*K, 1]."""
        depthwise = (taps * self.depthwise.weight[:, 0].repeat(self.streams, 1)).sum(-1, keepdim=True)
        weight, bias = self.pointwise_weight()
        update = self.dropout(F.relu(torch.matmul(weight, depthwise) + bias[:, None]))
        return F.relu(taps[..., -1:] + update)

    def forward(self, x):
        """x [B, S*K, T], the block input at every position of a window, to the output at the last one [B, S*K, 1]."""
        averages = torch.matmul(x, self.window_weights(x.shape[-1], x.device))  # [B, S*K, J]
        return self.from_taps(torch.cat((averages, x[:, :, -1:]), -1))

    def step(self, x, state):
        """One streaming step: x [B, S*K] the input now, state [B, S*K, J] the averages before it (zeros at the start
        of a session). Returns the output [B, S*K] and the new state."""
        state = self.decays * state + (1 - self.decays) * x[..., None]
        return self.from_taps(torch.cat((state, x[..., None]), -1))[..., 0], state


class SummaryTCNN(LeanTCNN):
    def __init__(self, *args, summary_taus=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.summary_taus = None if summary_taus is None else [float(t) for t in summary_taus]
        if summary_taus is None:
            return
        if not summary_taus or any(not t > 0 for t in self.summary_taus):
            raise ValueError("summary_taus must be null or positive time constants in samples")
        if not all(isinstance(block, SeparableBlock) for block in self.blocks):
            raise ValueError("summary_taus applies to separable blocks only")
        below = self.blocks[:-1]
        kernels = [block.depthwise.kernel_size[0] for block in below]
        start = receptive_field(len(below), kernel_sizes=kernels, dilations=[block.dilation for block in below])
        # The last block's kernel size and dilation in the config are no longer used.
        self.blocks[-1] = SummaryBlock(self.stream_channels, self.streams, self.dropout_rate, self.summary_taus, start)

    def specification(self):
        return dict(super().specification(), summary_taus=self.summary_taus)

    def forecast(self, hidden, last, step):
        """LeanTCNN.forward from the last block output on: hidden [N, S*K], the last orientation [N, 1, 4] and the
        last step's rotation vector [N, 3] to unit quaternions [N, horizon, 4]."""
        r = self.readout(hidden).reshape(-1, self.horizon, 3)
        if self.damped_prior_tau_ms is not None:
            tau = self.damped_prior_tau_ms / 10  # ms to 100 Hz samples
            steps = torch.arange(1, self.horizon+1, dtype=step.dtype, device=step.device)
            r = r + step[:, None, :] * (-tau * torch.expm1(-steps / tau))[None, :, None]
        return F.normalize(qmul(last, map_quaternion(r, self.maps)), dim=-1)

    def streamed(self, q):
        """Forecasts [N, horizon, 4] at every sample of one session q [N, 4], as a streaming graph computes them: the
        blocks below the summary block see the session from its start, and the averages start from zero at the first
        sample and are updated once per sample. Early forecasts are warm-up and not meaningful."""
        if self.summary_taus is None or self.maps == "exact":
            raise ValueError("streamed() is for summary models with lean or native maps")
        features, _, _ = self.features(q[None])  # [1, N, F]; columns 0-2 are the rate, 100 steps a second
        x = features.transpose(1, 2)
        if self.mirror_equivariant:
            x = torch.cat((self.projection(x), self.projection(x * self.input_signs[:, None])), 1)
        else:
            x = self.projection(x)
        x = F.relu(x)
        for block in self.blocks[:-1]:
            x = block(x)
        summary, rows = self.blocks[-1], x[0].T  # [N, S*K]
        state, averages = rows.new_zeros(rows.shape[1], len(summary.taus)), []
        for row in rows:
            state = summary.decays * state + (1 - summary.decays) * row[:, None]
            averages.append(state)
        hidden = summary.from_taps(torch.cat((torch.stack(averages), rows[:, :, None]), -1))[..., 0]
        return self.forecast(hidden, F.normalize(q, dim=-1)[:, None], features[0, :, :3] / 100.)


def load_checkpoint(path):
    """best.model.load_checkpoint for checkpoints with or without summary_taus."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = SummaryTCNN(**state["model_spec"])
    model.load_state_dict(state["model_state"])
    return model.eval(), state
