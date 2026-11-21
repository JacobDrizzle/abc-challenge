"""The best model's network, LeanTCNN, built in three layers; each keeps the layer below as its default.

GravityMirrorTCNN (v6) is the v5 body-velocity TCNN plus gravity channels and exact left-right mirror equivariance.
CompactTCNN (v7) adds dilation_base, separable blocks, a polynomial head basis and an L1 speed channel. LeanTCNN (v9)
adds per-block kernel sizes and dilations and a choice of maps. With its new options at their defaults each class is
the one below it (same parameters, created in the same order, same forward), so every group-a-v2 checkpoint since v5
loads with load_checkpoint, and the classes are moved here unchanged from archive/v6_gravity_mirror, v7_compact and
v9_lean so that training is unchanged too.

Gravity (v6): world up (+y, gravity-aligned in BOXRR/Unity) in the head frame, R(q)^T y. It is quadratic in q, so it
  ignores quaternion sign, and it ignores world yaw. It depends on q_t alone, so streaming stays exact.
Mirror (v6): reflecting the world across its x = 0 plane maps q = (w, x, y, z) to M(q) = (w, x, -y, -z) (conjugation
  by i; it preserves quaternion products and the up axis). Under M the body rate keeps x and flips y, z; the speed is
  unchanged; gravity flips x and keeps y, z; the forecast rotation keeps x and flips y, z. The network carries two
  streams [h0; h1], the observed and the mirrored input, and every layer is equivariant to swapping them: block
  weights [[A, B], [B, A]] stored once as W = [A, B]. Hence model(M h) == M model(h) exactly, with half the
  parameters at the same width and MACs.
dilation_base (v7): block i has dilation base**i. Base 3 with kernel 3 tiles every lag exactly once, so 5 blocks reach
  a receptive field of 243 samples where base 2 needs 7 blocks for 255.
separable (v7): each block is a depthwise dilated conv (one filter per channel, shared by both mirror streams)
  followed by a pointwise 1x1 conv; on twin streams the pointwise weight is [[A, B], [B, A]] stored once, as in
  MirrorCausalBlock. No nonlinearity between the two, so the depthwise bias would be redundant.
head_basis (v7): P, or None for v6's 60-output head. The head emits P coefficients per axis, and step k's rotation
  vector is sum_j c_j (k / horizon)**(j+1): corrections to the rate, acceleration and jerk of the damped prior.
  M2 (26 Oct): projecting v6's own 60 outputs onto P = 3 such terms changed validation error by +0.001 deg.
speed_feature (v7): "norm" (v5/v6) or "l1", |wx| + |wy| + |wz|, which a graph can build from Relu and MatMul alone.
kernel_sizes, dilations (v9): one per block, for separable blocks; None keeps kernel 3 and dilation_base**i. The
  receptive field is 1 + sum((k - 1) * d). k8_combined's three kernel-8 blocks with dilations 1, 8, 47 see 393 of the
  400 samples the rules allow, where v7's five kernel-3 blocks saw 243, and fewer blocks cost fewer graph nodes.
maps (v9): how per-step rates come in and forecast rotations go out. For d = conj(q[t-1]) q[t] = (w, v), with angle
  theta, and a forecast rotation vector r:
  "exact"  (v5-v8) rate = log(d) = theta n; quaternion = exp(r).
  "lean"   rate = 2 w v (1 + 2/3 |v|^2), the degree-1 series of theta / sin(theta) in 1 - cos(theta) = 2 |v|^2
           (relative error 5e-4 at 20 degrees a step); quaternion along (2 - |r|^2/6, r), the direction of exp(r)
           to second order (0.003 degrees off at 30, 0.09 at 60). The streaming graph computes these for exact-map
           checkpoints.
  "native" rate = 2 w v = sin(theta) n; quaternion along (2, r), the Cayley map (angle 2 atan(|r| / 2)). Trained
           with them (k8_combined is), the graph needs neither series.
  The w v products ignore the signs of both quaternions, as the exact log map does.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from tcnn.model import CausalBlock, body_velocity_features, conjugate, qmul, rotvec_to_quaternion

# Sign of each channel under M: rate (x, y, z), speed, gravity (x, y, z); forecast rotation vector (x, y, z).
RATE_SIGNS, GRAVITY_SIGNS, ROTATION_SIGNS = (1., -1., -1., 1.), (-1., 1., 1.), (1., -1., -1.)
MAPS = ("exact", "lean", "native")


def mirror(q):
    """M: the same motion seen in a left-right mirrored world (scalar-first quaternions [..., 4])."""
    return q * q.new_tensor((1., 1., -1., -1.))


def gravity_in_head_frame(q):
    """World +y expressed in the head frame, R(q)^T y, for unit quaternions [..., 4]."""
    w, x, y, z = q.unbind(-1)
    return torch.stack((2*(x*y + w*z), w*w - x*x + y*y - z*z, 2*(y*z - w*x)), -1)


def map_quaternion(r, maps):
    """Unit quaternions [..., 4] for forecast rotation vectors r [..., 3] under the given maps."""
    if maps == "exact":
        return rotvec_to_quaternion(r)
    scalar = 2 - (r * r).sum(-1, keepdim=True) / 6 if maps == "lean" else torch.full_like(r[..., :1], 2.)
    return F.normalize(torch.cat((scalar, r), -1), dim=-1)


def block_layout(num_blocks, dilation_base=2, kernel_sizes=None, dilations=None):
    """Per-block kernel sizes and dilations; None means kernel 3 and dilation_base**i."""
    kernels = [3] * num_blocks if kernel_sizes is None else list(kernel_sizes)
    spacing = [dilation_base ** i for i in range(num_blocks)] if dilations is None else list(dilations)
    if len(kernels) != num_blocks or len(spacing) != num_blocks:
        raise ValueError("kernel_sizes and dilations need one entry per block")
    return kernels, spacing


def receptive_field(num_blocks, dilation_base=2, kernel_sizes=None, dilations=None):
    """Time steps one output sees: 1 + sum over blocks of (kernel - 1) * dilation."""
    kernels, spacing = block_layout(num_blocks, dilation_base, kernel_sizes, dilations)
    return 1 + sum((k - 1) * d for k, d in zip(kernels, spacing))


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


class KernelBlock(SeparableBlock):
    """SeparableBlock with a depthwise kernel of any size k: taps at t - (k-1)d, ..., t - d, t."""

    def __init__(self, stream_channels, streams, dilation, dropout, kernel_size=3):
        super().__init__(stream_channels, streams, dilation, dropout)
        self.kernel_size, self.left_padding = kernel_size, (kernel_size - 1) * dilation
        # SeparableBlock built a kernel-3 depthwise conv; this one has kernel_size taps.
        self.depthwise = nn.Conv1d(stream_channels, stream_channels, kernel_size, dilation=dilation,
                                   groups=stream_channels, bias=False)

    def forward(self, x):
        # SeparableBlock.forward with kernel_size taps instead of 3.
        padded, length, d = F.pad(x, (self.left_padding, 0)), x.shape[-1], self.dilation
        taps = torch.stack([padded[:, :, k*d:k*d + length] for k in range(self.kernel_size)], -1)
        depthwise = (taps * self.depthwise.weight[:, 0].repeat(self.streams, 1)[:, None, :]).sum(-1)
        weight, bias = self.pointwise_weight()
        update = self.dropout(F.relu(torch.matmul(weight, depthwise) + bias[:, None]))
        return F.relu(x + update)


class LeanTCNN(CompactTCNN):
    """CompactTCNN plus kernel_sizes, dilations and maps (see the module docstring).

    The first four arguments keep train_tcnn's positional order (context, horizon, channels, dropout).
    """

    def __init__(self, context=400, horizon=20, channels=48, dropout=.1, effective_context=None,
                 velocity_features=False, damped_prior_tau_ms=None, input_features="body_velocity",
                 num_blocks=8, gravity_features=False, mirror_equivariant=False, dilation_base=2,
                 separable=False, head_basis=None, speed_feature="norm", kernel_sizes=None, dilations=None,
                 maps="exact"):
        super().__init__(context, horizon, channels, dropout, effective_context, velocity_features,
                         damped_prior_tau_ms, input_features, num_blocks, gravity_features, mirror_equivariant,
                         dilation_base, separable, head_basis, speed_feature)
        if maps not in MAPS:
            raise ValueError("maps must be exact, lean or native")
        self.kernel_sizes, self.dilations, self.maps = kernel_sizes, dilations, maps
        # The CompactTCNN blocks stay unless a layout is given, so the defaults reproduce v7 exactly.
        if kernel_sizes is not None or dilations is not None:
            if not separable:
                raise ValueError("kernel_sizes and dilations apply to separable blocks only")
            kernels, spacing = block_layout(num_blocks, dilation_base, kernel_sizes, dilations)
            self.blocks = nn.Sequential(*(KernelBlock(self.stream_channels, self.streams, d, dropout, k)
                                          for k, d in zip(kernels, spacing)))

    def features(self, window):
        """As GravityMirrorTCNN.features, with the rate from this model's maps."""
        if self.maps == "exact":
            return super().features(window)
        q = F.normalize(window, dim=-1)
        delta = qmul(conjugate(q[:, :-1]), q[:, 1:])  # (cos(theta/2), sin(theta/2) n) per step
        w, v = delta[..., :1], delta[..., 1:]
        step = 2 * w * v  # sin(theta) n
        if self.maps == "lean":
            step = step * (1 + 2 / 3 * (v * v).sum(-1, keepdim=True))
        velocity = step * 100.  # rad/s at 100 Hz
        speed = (velocity.abs().sum(-1, keepdim=True) if self.speed_feature == "l1"
                 else torch.linalg.vector_norm(velocity, dim=-1, keepdim=True))
        rows = torch.cat((velocity, speed), -1)
        features = torch.cat((torch.zeros_like(rows[:, :1]), rows), 1)  # the first step has no rate
        if self.gravity_features:
            features = torch.cat((features, gravity_in_head_frame(q)), -1)
        return features, q[:, -1:], step[:, -1]

    def forward(self, history):
        if self.maps == "exact":
            return super().forward(history)
        # GravityMirrorTCNN.forward up to the rotation vectors; then this model's output map.
        if history.ndim != 3 or history.shape[1:] != (self.context, 4):
            raise ValueError(f"Expected [B,{self.context},4], got {tuple(history.shape)}")
        features, last, step = self.features(history[:, -self.effective_context:])
        x = features.transpose(1, 2)
        if self.mirror_equivariant:
            x = torch.cat((self.projection(x), self.projection(x * self.input_signs[:, None])), 1)
        else:
            x = self.projection(x)
        x = self.blocks(F.relu(x))
        r = self.readout(x[:, :, -1]).reshape(-1, self.horizon, 3)
        if self.damped_prior_tau_ms is not None:
            tau = self.damped_prior_tau_ms / 10  # ms to 100 Hz samples
            steps = torch.arange(1, self.horizon+1, dtype=step.dtype, device=step.device)
            r = r + step[:, None, :] * (-tau * torch.expm1(-steps / tau))[None, :, None]
        return F.normalize(qmul(last, map_quaternion(r, self.maps)), dim=-1)

    def specification(self):
        return dict(super().specification(), kernel_sizes=self.kernel_sizes, dilations=self.dilations,
                    maps=self.maps)


def deployed(model):
    """The model the streaming graph reproduces: exact maps become lean ones (Track A); lean and native stay."""
    if model.maps == "exact":
        model.maps = "lean"
    return model


def load_checkpoint(path):
    """Any group-a-v2 checkpoint since v5: a specification without the v6, v7 or v9 keys takes their defaults."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = LeanTCNN(**state["model_spec"])
    model.load_state_dict(state["model_state"])
    return model.eval(), state
