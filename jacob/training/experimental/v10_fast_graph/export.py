"""StreamingFast: the streaming graph of a native-maps LeanTCNN in its fastest known form, straight from a checkpoint.

best/export.py's StreamingLean is the reference graph. On 6 Nov a chain of exact rewrites of its ONNX file
(~/Dev/experiments/wsp-v8, edits e01 ... e13) cut local CPU latency by about 30 % at identical accuracy, but only as
hand edits to one file. This module builds that graph for any checkpoint and layout, so no file has to be edited.

Same interface as StreamingLean, except the state layout: context [1, 400, 4] (the last two samples are read) and
state_in_i [1, c_i, (k_i - 1) d_i, 2] in; prediction [1, 20, 4] and state_out_i out. c_i is the block's channels a
stream, or its `inputs` if the block reads fewer (experimental/v10_narrow).

What differs from StreamingLean (each step is exact algebra, no weight is refitted):
- Front end. Every input feature is a product of two numbers that are linear in the 32 products of
  [q[t-1], q[t]] with q[t]: the rate is w v from conj(q[t-1]) q[t] = (w, v), gravity is quadratic in q[t] (times
  |q[t]|^2 = 1), and 1 is |q[t]|^2 squared. One MatMul (K1) gives both factors of 13 products, and
  r = ReLU(products) = [rate+, rate-, gravity+, gravity-, 1] (3, 3, 3, 3, 1). rate = rate+ - rate-, the L1 speed is
  rate+ + rate-, and the mirrored stream's sign flips are a permutation of r, so the projection of both streams is
  one Gemm on r.
- Residual stream. It stays 2-D, [1, 2K] in (channel, stream) order. A block is: Reshape to [1, K, 1, 2], Concat
  with its state, depthwise Conv over time, Reshape, one dense Gemm with the mirror weight [[A, B], [B, A]], ReLU,
  Add. On ORT a Gemm costs far less fixed time than a Conv.
- No ReLU after the Add: both terms are ReLU outputs already, so it changed nothing.
- Tail. The head, the polynomial basis, the damped prior and the x biases are all linear in [stream, r], so one
  Gemm gives the 20 forecast rotations as quaternions (1, r_k / 2), the Cayley map's (2, r_k) halved. One MatMul
  with q[t]'s left-multiplication matrix composes them onto the last pose.
- Output. now * (1, r_k / 2) is not normalised: its norm is sqrt(1 + |r_k|^2 / 4), 1.00 to about 1.03. The
  organisers renormalise predictions before scoring ("Submitting artefacts"); a harness that refuses non-unit
  outputs needs the reference graph.

fold=True stores the dense matrices (k8_combined: 20,304 numbers, each learned weight up to four times) and the
graph has no builder nodes. fold=False stores the learned weights as the checkpoint has them (7,555 numbers) and
builds the dense matrices with constant-only nodes; ONNX Runtime folds those when the session is created, a
runtime that does not would run them on every call.

Needs what StreamingLean needs (separable mirror blocks, basis head, damped prior, L1 speed, gravity) and native
maps: lean maps add a correction that is not a product of two linear terms.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tcnn.model import conjugate, qmul
from tcnn.onnx_export import OPSET

from best.export import _gravity_table

FEATURES = 13  # rate+ (3), rate- (3), gravity+ (3), gravity- (3), one


def _front_end_table():
    """K1 [32, 26]: products [q[t-1]; q[t]] (8) x q[t] (4), flattened, to the two factors of each of the 13 features.
    Columns 0..12 are the first factors, 13..25 the second."""
    basis = torch.eye(4)
    delta = torch.stack([torch.stack([qmul(conjugate(basis[i]), basis[j]) for j in range(4)]) for i in range(4)])
    scale = 200 ** .5  # rate = 100 * 2 w v rad/s, split evenly between the two factors
    gravity = _gravity_table().reshape(4, 4, 4)  # [i, j, (gx, gy, gz, |q|^2)] over q_i q_j
    table = torch.zeros(8, 4, 2 * FEATURES)
    for axis in range(3):
        for sign, feature in ((1., axis), (-1., 3 + axis)):
            table[:4, :, feature] = sign * scale * delta[:, :, 0]  # w
            table[:4, :, FEATURES + feature] = scale * delta[:, :, 1 + axis]  # v
        for sign, feature in ((1., 6 + axis), (-1., 9 + axis)):
            table[4:, :, feature] = sign * gravity[:, :, axis]
            table[4:, :, FEATURES + feature] = gravity[:, :, 3]  # |q|^2
    table[4:, :, 12] = table[4:, :, FEATURES + 12] = gravity[:, :, 3]
    return table.reshape(32, 2 * FEATURES)


def _feature_table():
    """[7, 13]: r to the model's features (rate xyz, L1 speed, gravity xyz)."""
    table = torch.zeros(7, FEATURES)
    for axis in range(3):
        table[axis, axis], table[axis, 3 + axis] = 1., -1.
        table[3, axis] = table[3, 3 + axis] = 1.
        table[4 + axis, 6 + axis], table[4 + axis, 9 + axis] = 1., -1.
    return table


def _compose_table():
    """[4, 16]: q @ table, as [4, 4], is the matrix M with (q * b) = b @ M for a quaternion b as a row."""
    basis = torch.eye(4)
    return torch.stack([torch.stack([qmul(basis[i], basis[n]) for n in range(4)]) for i in range(4)]).reshape(4, 16)


class StreamingFast(nn.Module):
    def __init__(self, model, fold=True):
        super().__init__()
        if not (model.input_features == "body_velocity" and model.mirror_equivariant and model.separable
                and model.head_basis is not None and model.damped_prior_tau_ms is not None
                and model.speed_feature == "l1" and model.gravity_features and model.maps == "native"):
            raise ValueError("The fast graph needs separable mirror blocks, gravity, a basis head, the damped prior, "
                             "L1 speed and native maps")
        model = model.eval()
        k, p, horizon = model.stream_channels, model.head_basis, model.horizon
        self.channels, self.horizon = k, horizon
        self.dilations = [block.dilation for block in model.blocks]
        self.inputs = [getattr(block, "inputs", k) for block in model.blocks]
        self.state_shapes = [(1, c, block.left_padding, 2) for c, block in zip(self.inputs, model.blocks)]
        # Learned weights, in the checkpoint's shapes: with fold=False the file stores exactly these.
        self.projection = nn.Parameter(model.projection.weight.detach()[:, :, 0])  # [K, 7]
        self.projection_bias = nn.Parameter(model.projection.bias.detach())
        self.depthwise = nn.ParameterList(nn.Parameter(block.depthwise.weight.detach()[..., None])
                                          for block in model.blocks)  # [c, 1, k, 1]: time is the image height
        self.pointwise = nn.ParameterList(nn.Parameter(block.pointwise.weight.detach()[:, :, 0])
                                          for block in model.blocks)  # [K, 2c] = [A | B]
        self.pointwise_bias = nn.ParameterList(nn.Parameter(block.pointwise.bias.detach()) for block in model.blocks)
        self.head = nn.Parameter(model.head.weight.detach())  # [3P, K]
        self.head_bias_x = nn.Parameter(model.head_bias_x.detach())  # [P]
        # Fixed tables.
        self.register_buffer("front_end", _front_end_table(), persistent=False)
        self.register_buffer("compose", _compose_table(), persistent=False)
        signs = torch.stack((torch.ones_like(model.input_signs), model.input_signs))  # [2, 7]: stream 0, stream 1
        self.register_buffer("stream_features", signs[:, :, None] * _feature_table(), persistent=False)  # [2, 7, 13]
        self.register_buffer("output_signs", model.output_signs.clone(), persistent=False)  # [3P]
        self.register_buffer("no_bias", torch.zeros(p, 2), persistent=False)  # the y and z components have none
        one = torch.zeros(FEATURES)
        one[12] = 1.
        self.register_buffer("one", one, persistent=False)
        tau = model.damped_prior_tau_ms / 10
        steps = torch.arange(1, horizon + 1, dtype=torch.float64)
        elapsed = (-tau * torch.expm1(-steps / tau)).float() / 100  # the prior's step is rate / 100
        width = 2 * k + FEATURES
        # Rows (step, component) of the quaternions (1, r / 2). basis_rows: head coefficients (term, axis) to the
        # rotation rows, halved. fixed_rows: the scalar 1 and the damped prior on rate+ - rate-, halved.
        basis_rows, fixed_rows = torch.zeros(horizon * 4, 3 * p), torch.zeros(horizon * 4, width)
        for step in range(horizon):
            fixed_rows[step * 4, 2 * k + 12] = 1.
            for axis in range(3):
                row = step * 4 + 1 + axis
                basis_rows[row, torch.arange(p) * 3 + axis] = model.basis[step] / 2
                fixed_rows[row, 2 * k + axis], fixed_rows[row, 2 * k + 3 + axis] = elapsed[step] / 2, -elapsed[step] / 2
        self.register_buffer("basis_rows", basis_rows, persistent=False)
        self.register_buffer("fixed_rows", fixed_rows, persistent=False)
        self.folded = None
        if fold:  # the dense matrices become the stored weights; the checkpoint-shaped ones leave the graph
            with torch.no_grad():
                projection, projection_bias, blocks, tail = self.matrices()
            self.folded = nn.ParameterList([projection, projection_bias, tail, *(t for pair in blocks for t in pair)])

    def matrices(self):
        """The dense matrices the graph multiplies by, built from the stored weights (constant-only nodes)."""
        k = self.channels
        # Projection of both streams from r: rows (channel, stream).
        projection = torch.matmul(self.projection[:, None, None, :], self.stream_features).reshape(2 * k, FEATURES)
        projection_bias = torch.stack((self.projection_bias, self.projection_bias), 1).reshape(2 * k)
        blocks = []
        for weight, bias, c in zip(self.pointwise, self.pointwise_bias, self.inputs):
            a, b = weight[:, :c], weight[:, c:]
            # Rows (output channel, stream), columns (input channel, stream): A within a stream, B across.
            dense = torch.stack((torch.stack((a, b), -1), torch.stack((b, a), -1)), 1).reshape(2 * k, 2 * c)
            blocks.append((dense, torch.stack((bias, bias), 1).reshape(2 * k)))
        # Head coefficients (term, axis) from [stream, r]: W h0 + signs * W h1, and the x biases on the 1.
        head = torch.stack((self.head, self.output_signs[:, None] * self.head), -1).reshape(-1, 2 * k)
        bias = torch.cat((self.head_bias_x[:, None], self.no_bias), 1).reshape(-1)  # [3P]: the x components only
        coefficients = torch.cat((head, bias[:, None] * self.one), 1)  # [3P, 2K + 13]
        tail = (torch.matmul(self.basis_rows, coefficients) + self.fixed_rows).t()  # [2K + 13, horizon * 4]
        return projection, projection_bias, blocks, tail

    def forward(self, context, *states):
        k = self.channels
        if self.folded is None:
            projection, projection_bias, blocks, tail = self.matrices()
        else:
            projection, projection_bias, tail = self.folded[:3]
            blocks = list(zip(self.folded[3::2], self.folded[4::2]))
        now = context[:, -1:]  # [1, 1, 4]
        products = torch.matmul(context[:, -2:].reshape(1, 8, 1), now).reshape(1, 32)
        factors = torch.matmul(products, self.front_end)  # [1, 26]
        r = F.relu(factors[:, :FEATURES] * factors[:, FEATURES:])  # [1, 13]
        x = F.relu(F.linear(r, projection, projection_bias))  # [1, 2K], (channel, stream)
        new_states = []
        for state, depthwise, (dense, bias), dilation, c in zip(states, self.depthwise, blocks, self.dilations,
                                                                self.inputs):
            newest = x.reshape(1, k, 1, 2)
            if c < k:
                newest = newest[:, :c]
            window = torch.cat((state, newest), 2)  # the block's input at t - (k-1)d ... t
            new_states.append(window[:, :, 1:])
            h = F.conv2d(window, depthwise, dilation=(dilation, 1), groups=c).reshape(1, 2 * c)
            x = x + F.relu(F.linear(h, dense, bias))
        forecast = torch.matmul(torch.cat((x, r), 1), tail).reshape(1, self.horizon, 4)  # (1, r_k / 2)
        prediction = torch.matmul(forecast, torch.matmul(now, self.compose).reshape(1, 4, 4))  # now * (1, r_k / 2)
        return (prediction, *new_states)


def export_fast_onnx(model, path, fold=True):
    """Writes the fast streaming graph of `model` to `path`; returns (path, state shapes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = StreamingFast(model, fold).eval()
    example = (F.normalize(torch.randn(1, model.context, 4), dim=-1), *(torch.zeros(s) for s in wrapper.state_shapes))
    count = len(wrapper.state_shapes)
    with torch.no_grad():
        # fold=False: the builder nodes over the stored weights must stay in the file, so torch may not fold them.
        torch.onnx.export(wrapper, example, str(path), opset_version=OPSET, do_constant_folding=fold, dynamo=False,
                          input_names=["context", *(f"state_in_{i}" for i in range(count))],
                          output_names=["prediction", *(f"state_out_{i}" for i in range(count))])
    model.eval()
    import onnx
    onnx.checker.check_model(onnx.load(str(path)))
    return path, wrapper.state_shapes
