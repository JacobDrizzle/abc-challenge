"""v5 body-velocity TCNN plus gravity channels and exact left-right mirror equivariance.

Gravity (1a): world up (+y, gravity-aligned in BOXRR/Unity) in the head frame, R(q)^T y. It is quadratic in q,
so it ignores quaternion sign, and it ignores world yaw. It depends on q_t alone, so streaming stays exact.

Mirror (1b): reflecting the world across its x = 0 plane maps q = (w, x, y, z) to M(q) = (w, x, -y, -z)
(conjugation by i; it preserves quaternion products and the up axis). Under M the body rate keeps x and flips
y, z; the speed is unchanged; gravity flips x and keeps y, z; the forecast rotation keeps x and flips y, z.
The network carries two streams [h0; h1], the observed and the mirrored input, and every layer is equivariant
to swapping them: block weights [[A, B], [B, A]] stored once as W = [A, B]. Hence model(M h) == M model(h)
exactly, with half the parameters at the same width and MACs.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from tcnn.model import CausalBlock, body_velocity_features, qmul, rotvec_to_quaternion

# Sign of each channel under M: rate (x, y, z), speed, gravity (x, y, z); forecast rotation vector (x, y, z).
RATE_SIGNS, GRAVITY_SIGNS, ROTATION_SIGNS = (1., -1., -1., 1.), (-1., 1., 1.), (1., -1., -1.)


def mirror(q):
    """M: the same motion seen in a left-right mirrored world (scalar-first quaternions [..., 4])."""
    return q * q.new_tensor((1., 1., -1., -1.))


def gravity_in_head_frame(q):
    """World +y expressed in the head frame, R(q)^T y, for unit quaternions [..., 4]."""
    w, x, y, z = q.unbind(-1)
    return torch.stack((2*(x*y + w*z), w*w - x*x + y*y - z*z, 2*(y*z - w*x)), -1)


class MirrorCausalBlock(nn.Module):
    """CausalBlock on twin streams [B, 2K, T]: stream 0 is channels :K, stream 1 is channels K:.

    conv holds W [K, 2K, 3] = [A, B] and one bias [K]; the full 2K x 2K weight [[A, B], [B, A]] is assembled
    on every call, so it applies A to its own stream and B to the other one, for both streams alike.
    """

    def __init__(self, stream_channels, dilation, dropout):
        super().__init__()
        self.left_padding, self.dilation = 2 * dilation, dilation
        self.conv = nn.Conv1d(2 * stream_channels, stream_channels, 3, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def full_weight(self):
        w, k = self.conv.weight, self.conv.out_channels
        return torch.cat((w, torch.cat((w[:, k:], w[:, :k]), 1)), 0), torch.cat((self.conv.bias, self.conv.bias))

    def dilated_conv(self, x):
        # Taps + matmul, as CausalBlock.dilated_conv, to avoid the slow deterministic dilated Conv1d.
        weight, bias = self.full_weight()
        padded, length, d = F.pad(x, (self.left_padding, 0)), x.shape[-1], self.dilation
        taps = torch.cat([padded[:, :, k*d:k*d + length] for k in range(3)], dim=1)
        return torch.matmul(weight.permute(0, 2, 1).reshape(len(weight), -1), taps) + bias[:, None]

    def forward(self, x):
        update = self.dropout(F.relu(self.dilated_conv(x)))
        return F.relu(x + update)


class GravityMirrorTCNN(nn.Module):
    """TinyTCNN(input_features="body_velocity") with optional gravity channels and mirror equivariance.

    With both options off it is the v5 model: same parameters, created in the same order, same forward.
    Accepts every TinyTCNN specification key, so checkpoints load through TinyTCNN-style **model_spec.
    """

    def __init__(self, context=400, horizon=20, channels=48, dropout=.1, effective_context=None,
                 velocity_features=False, damped_prior_tau_ms=None, input_features="body_velocity",
                 num_blocks=8, gravity_features=False, mirror_equivariant=False):
        super().__init__()
        if input_features != "body_velocity" or velocity_features:
            raise ValueError("v6 models use body_velocity inputs only")
        if mirror_equivariant and channels % 2:
            raise ValueError("Mirror equivariance splits the channels into two equal streams")
        self.context, self.horizon = context, horizon
        self.channels, self.dropout_rate = channels, dropout
        self.effective_context = context if effective_context is None else effective_context
        if not 2 <= self.effective_context <= context:
            raise ValueError("Invalid effective context")
        self.velocity_features, self.input_features, self.num_blocks = False, input_features, num_blocks
        self.damped_prior_tau_ms = damped_prior_tau_ms
        self.gravity_features, self.mirror_equivariant = gravity_features, mirror_equivariant
        self.streams = 2 if mirror_equivariant else 1
        self.stream_channels = channels // self.streams
        inputs = 7 if gravity_features else 4
        self.projection = nn.Conv1d(inputs, self.stream_channels, 1)
        block = MirrorCausalBlock if mirror_equivariant else CausalBlock
        self.blocks = nn.Sequential(*(block(self.stream_channels, 2**i, dropout) for i in range(num_blocks)))
        # Mirrored stream: its readout is sign-flipped, and only the x components may carry a bias.
        self.head = nn.Linear(self.stream_channels, horizon * 3, bias=not mirror_equivariant)
        nn.init.zeros_(self.head.weight)
        if mirror_equivariant:
            self.head_bias_x = nn.Parameter(torch.zeros(horizon))
        else:
            nn.init.zeros_(self.head.bias)
        signs = RATE_SIGNS + (GRAVITY_SIGNS if gravity_features else ())
        self.register_buffer("input_signs", torch.tensor(signs), persistent=False)
        self.register_buffer("output_signs", torch.tensor(ROTATION_SIGNS * horizon), persistent=False)

    def features(self, window):
        """[B, N, F] per-step inputs, the last orientation and the last step's rotation vector."""
        features, last, step = body_velocity_features(window)
        if self.gravity_features:
            features = torch.cat((features, gravity_in_head_frame(F.normalize(window, dim=-1))), -1)
        return features, last, step

    def readout(self, last):
        if not self.mirror_equivariant:
            return self.head(last)
        own, other = last.split(self.stream_channels, 1)
        bias = F.pad(self.head_bias_x[:, None], (0, 2)).reshape(-1)
        return self.head(own) + self.output_signs * self.head(other) + bias

    def forward(self, history):
        if history.ndim != 3 or history.shape[1:] != (self.context, 4):
            raise ValueError(f"Expected [B,{self.context},4], got {tuple(history.shape)}")
        features, last, step = self.features(history[:, -self.effective_context:])
        x = features.transpose(1, 2)
        if self.mirror_equivariant:
            x = torch.cat((self.projection(x), self.projection(x * self.input_signs[:, None])), 1)
        else:
            x = self.projection(x)
        x = self.blocks(F.relu(x))
        rotation_vectors = self.readout(x[:, :, -1]).reshape(-1, self.horizon, 3)
        if self.damped_prior_tau_ms is not None:
            tau = self.damped_prior_tau_ms / 10  # ms to 100 Hz samples
            steps = torch.arange(1, self.horizon+1, dtype=step.dtype, device=step.device)
            rotation_vectors = rotation_vectors + step[:, None, :] * (-tau * torch.expm1(-steps / tau))[None, :, None]
        return F.normalize(qmul(last, rotvec_to_quaternion(rotation_vectors)), dim=-1)

    def specification(self):
        return dict(context=self.context, horizon=self.horizon, channels=self.channels,
                    dropout=self.dropout_rate, effective_context=self.effective_context,
                    velocity_features=self.velocity_features, damped_prior_tau_ms=self.damped_prior_tau_ms,
                    input_features=self.input_features, num_blocks=self.num_blocks,
                    gravity_features=self.gravity_features, mirror_equivariant=self.mirror_equivariant)


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = GravityMirrorTCNN(**state["model_spec"])
    model.load_state_dict(state["model_state"])
    return model.eval(), state
