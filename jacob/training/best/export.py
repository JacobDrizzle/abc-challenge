"""Streaming ONNX graph for LeanTCNN (v9 Track A): one new sample per call, the challenge's stateful interface.

Interface: context [1, 400, 4] (last two samples read) and state_in_i in; prediction [1, 20, 4] and state_out_i out.
On the local CPU latency follows the node count (about 0.7 us a node), so:
- A1, rate: for d = conj(q[t-1]) q[t] = (w, v), the lean rate is 100 * 2 w v (1 + 2/3 |v|^2)
  = w v (200 w^2 + 1000/3 |v|^2), using w^2 + |v|^2 = 1 for unit inputs, so the bracket is one MatMul of d * d.
  With native maps the d table carries sqrt(200), and w v is already the rate 100 sin(theta) n.
- A2, output: (2 - |r|^2/6, r), or (2, r) with native maps, instead of the exp map's 18 nodes. Its norm is about
  2, and baseline_pipeline refuses norms off by more than 1e-3, as a strict harness might, so the graph normalises:
  one MatMul gives [p, -p] (mean 0, mean square |p|^2 / 4) and LayerNormalization with scale 1/2 gives p / |p|.
  st_layout=True (9 Nov, off by default) runs that LayerNormalization on [1, 1, horizon, 8], between two Reshapes.
  ST Edge AI Core 4.1 stores a [1, horizon, 8] tensor channel-last and then normalises over the wrong groups of 8:
  on 6 Nov the STM32N6, H7 and SR6 boards gave a wrong prediction (cosine 0.796 with ST's reference). With a
  second axis of size 1 the two layouts are the same memory, and the N6 gave cosine 1.000000. ONNX Runtime's output
  is the same either way. The challenge's test hardware is not known, so whether it needs this is not known either.
- A3, mirror layout: the two streams are the rows of [1, K, 2, T]. A block is a depthwise (1, k) Conv over both
  rows and a (2, 1) Conv with weight [A | B] over the rows [h0, h1, h0] (a Slice and a Concat): [[A, B], [B, A]]
  in one Conv. padded_rows=True drops the Slice and the Concat: a (3, 1) Conv with kernel rows [B, A, B] over
  [0, h0, h1, 0] (one zero row padded above and below) gives A h0 + B h1 and B h0 + A h1 directly, for 2K^2 more
  MACs a block (a third of its taps meet the padding).
- split_mirror=True (v7's mirror split, for this layout): on the sum s = h0 + h1 and the difference d = h0 - h1 the
  matrix [[A, B], [B, A]] separates: P = (A + B)/2 s + b, Q = (A - B)/2 d, and the rows are P + Q and P - Q. Two
  K x K Convs replace the 2K^2-tap one, so each block's pointwise MACs halve, for 6 more nodes a block (Sub is off
  the safe list, so d is h0 + h1 * -1, and the rows are P + Q * [1, -1]). The file stores (A + B)/2 and (A - B)/2,
  the same 2K^2 numbers as [A | B]. It cannot be combined with padded_rows, which also replaces that Conv.
- Head: one MatMul gives [W h0, W h1]; a sign mask and a ones MatMul give W h0 + S W h1. The x biases and the
  damped prior's rate are extra rows of the basis MatMul's coefficients. A4 (fused_head=True): one (2, 1) Conv
  with kernel rows W and S W, and the x biases as its bias.
- Tied weights are stored once. With padded_rows or fused_head the file holds [A | B], W and the x biases once,
  and Concat nodes over those initializers build [B, A, B] and [W; S W]. ORT's constant folding runs them once,
  when the session is created, so they cost nothing per call; a runtime that does not fold constants would run
  them on every call. padded_rows and fused_head were measured on 28 Oct, split_mirror is untimed on this layout;
  none is the default, and adopting one is a team call (it waits for ST hardware timings).
The graph reproduces deployed(model): exact maps are exported as lean ones. Built for the models trained since
27 Oct (separable mirror blocks, basis head, damped prior, L1 speed); others raise.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tcnn.onnx_export import OPSET, _bilinear_tables, _left_matrix


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


class StreamingLean(nn.Module):
    """Inputs context [1, 400, 4] and state_in_i [1, K, 2, (k_i - 1) * d_i]."""

    def __init__(self, model, fused_head=False, padded_rows=False, split_mirror=False, st_layout=False):
        super().__init__()
        if not (model.input_features == "body_velocity" and model.mirror_equivariant and model.separable
                and model.head_basis is not None and model.damped_prior_tau_ms is not None
                and model.speed_feature == "l1"):
            raise ValueError("The streaming graph needs separable mirror blocks, a basis head, the damped prior and "
                             "L1 speed")
        if padded_rows and split_mirror:
            raise ValueError("padded_rows and split_mirror both replace the pointwise Conv: choose one")
        self.model, self.fused_head, self.padded_rows = model.eval(), fused_head, padded_rows
        self.split_mirror, self.st_layout = split_mirror, st_layout
        self.maps = "native" if getattr(model, "maps", "exact") == "native" else "lean"
        k, p, signs = model.stream_channels, model.head_basis, model.output_signs
        self.state_shapes = [(1, k, 2, block.left_padding) for block in model.blocks]
        tables = _bilinear_tables()
        # now @ left_matrix(table_left_conj, previous) == conj(previous) * now.
        scale = 200 ** .5 if self.maps == "native" else 1.
        self.register_buffer("table_left_conj", tables["left_conj"].float() * scale, persistent=False)
        # now @ table_left -> [4, 4] matrix of b -> now * b; this table gives [M, -M] as [4, 8].
        left = tables["left"].float().reshape(4, 4, 4)
        self.register_buffer("table_left_both", torch.cat((left, -left), 2).reshape(4, 32), persistent=False)
        self.register_buffer("rate_weights", torch.tensor([[200.], [1000 / 3], [1000 / 3], [1000 / 3]]),
                             persistent=False)
        self.register_buffer("gravity_table", _gravity_table()[:, :3].contiguous(), persistent=False)
        self.register_buffer("plus_minus", torch.cat((torch.eye(3), -torch.eye(3)), 1), persistent=False)
        self.register_buffer("ones6", torch.ones(6, 1), persistent=False)
        # Row 0 sees the input, row 1 its mirror image.
        self.register_buffer("input_signs", torch.stack((torch.ones_like(model.input_signs), model.input_signs), 1)
                             [None, :, :, None], persistent=False)  # [1, F, 2, 1]
        # Weights in Conv2d shapes. nn.Parameters, so the file stores them as initializers, each value once.
        self.projection = nn.Parameter(model.projection.weight.detach()[..., None])  # [K, F, 1, 1]
        self.depthwise = nn.ParameterList(nn.Parameter(block.depthwise.weight.detach()[:, :, None])
                                          for block in model.blocks)  # [K, 1, 1, k]
        # pointwise [K, 2K, 1] = [A | B] -> [K, K, 2, 1]: kernel row 0 holds A, row 1 holds B.
        pointwise = [block.pointwise.weight.detach().reshape(k, 2, k).transpose(1, 2)[..., None]
                     for block in model.blocks]
        if padded_rows:  # kernel rows B, A, B; the file stores A and B once (_store_tied_weights_once)
            pointwise = [torch.cat((w[:, :, 1:2], w[:, :, 0:1], w[:, :, 1:2]), 2) for w in pointwise]
        if split_mirror:  # (A - B)/2 here and (A + B)/2 as pointwise, [K, K, 1, 1] each: the 1/2 is folded in
            self.difference = nn.ParameterList(nn.Parameter(((w[:, :, 0:1] - w[:, :, 1:2]) / 2).contiguous())
                                               for w in pointwise)
            pointwise = [(w[:, :, 0:1] + w[:, :, 1:2]) / 2 for w in pointwise]
            self.register_buffer("stream_signs", torch.tensor([[1.], [-1.]]), persistent=False)  # rows P+Q, P-Q
        self.pointwise = nn.ParameterList(nn.Parameter(w.contiguous()) for w in pointwise)
        x_bias = F.pad(model.head_bias_x.detach()[:, None], (0, 2))  # [P, 3]: the x components only
        tau = model.damped_prior_tau_ms / 10
        steps = torch.arange(1, model.horizon + 1, dtype=torch.float64)
        elapsed = (-tau * torch.expm1(-steps / tau)).float()[:, None] / 100  # the prior's step is rate / 100
        if fused_head:
            self.head_weight = nn.Parameter(torch.stack((model.head.weight.detach(),
                                                         signs[:, None] * model.head.weight.detach()), -1)[..., None])
            self.head_bias = nn.Parameter(x_bias.reshape(-1))
            basis = torch.cat((model.basis, elapsed), 1)  # [horizon, P + 1]
        else:
            self.register_buffer("head_mask", torch.stack((torch.ones_like(signs), signs), 1), persistent=False)
            self.register_buffer("ones2", torch.ones(2, 1), persistent=False)
            self.register_buffer("coefficient_bias", x_bias[None], persistent=False)  # [1, P, 3]
            basis = torch.cat((model.basis, elapsed, model.basis), 1)  # [horizon, 2P + 1]
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer("minus_sixth", torch.full((3, 1), -1 / 6), persistent=False)
        self.register_buffer("twos", torch.full((1, model.horizon, 1), 2.), persistent=False)
        self.register_buffer("halves", torch.full((8,), .5), persistent=False)

    def forward(self, context, *states):
        m, p = self.model, self.model.head_basis
        previous, now = context[:, -2:-1], context[:, -1:]
        delta = torch.matmul(now, _left_matrix(self.table_left_conj, previous))  # conj(q[t-1]) q[t], [1, 1, 4]
        velocity = delta[..., 0:1] * delta[..., 1:4]  # native: 100 sin(theta) n (scaled table)
        if self.maps == "lean":
            velocity = velocity * torch.matmul(delta * delta, self.rate_weights)
        speed = torch.matmul(F.relu(torch.matmul(velocity, self.plus_minus)), self.ones6)  # |wx| + |wy| + |wz|
        parts = [velocity, speed]
        if m.gravity_features:
            outer = torch.matmul(now.reshape(1, 4, 1), now).reshape(1, 1, 16)
            parts.append(torch.matmul(outer, self.gravity_table))
        x = torch.cat(parts, -1).reshape(1, -1, 1, 1) * self.input_signs  # [1, F, 2, 1]
        x = F.relu(F.conv2d(x, self.projection, m.projection.bias))  # [1, K, 2, 1]
        new_states = []
        differences = self.difference if self.split_mirror else [None] * len(m.blocks)
        for block, state, depthwise, pointwise, difference in zip(m.blocks, states, self.depthwise, self.pointwise,
                                                                  differences):
            window = torch.cat((state, x), -1)  # block input at t - (k-1)d ... t
            new_states.append(window[..., 1:])
            h = F.conv2d(window, depthwise, dilation=(1, block.dilation), groups=m.stream_channels)
            if self.split_mirror:  # P from h0 + h1 (with the bias), Q from h0 - h1; rows P + Q and P - Q
                h0, h1 = h[:, :, 0:1], h[:, :, 1:2]
                plus = F.conv2d(h0 + h1, pointwise, block.pointwise.bias)
                minus = F.conv2d(h0 + h1 * -1., difference)
                update = plus + minus * self.stream_signs  # [1, K, 2, 1]
            elif self.padded_rows:  # [B, A, B] over [0, h0, h1, 0]
                update = F.conv2d(h, pointwise, block.pointwise.bias, padding=(1, 0))
            else:  # [A | B] over [h0, h1, h0]
                update = F.conv2d(torch.cat((h, h[:, :, 0:1]), 2), pointwise, block.pointwise.bias)
            x = F.relu(x + F.relu(update))
        if self.fused_head:
            coefficients = F.conv2d(x, self.head_weight, self.head_bias).reshape(1, p, 3)
            rows = (coefficients, velocity)
        else:
            both = torch.matmul(m.head.weight, x.reshape(m.stream_channels, 2))  # [3P, 2]: W h0, W h1
            coefficients = torch.matmul(both * self.head_mask, self.ones2).reshape(1, p, 3)  # W h0 + S W h1
            rows = (coefficients, velocity, self.coefficient_bias)
        r = torch.matmul(self.basis, torch.cat(rows, 1))  # [1, horizon, 3], prior included
        scalar = self.twos if self.maps == "native" else torch.matmul(r * r, self.minus_sixth) + 2.
        both = torch.matmul(torch.cat((scalar, r), -1), torch.matmul(now, self.table_left_both).reshape(1, 4, 8))
        if self.st_layout:  # the same LayerNormalization on [1, 1, horizon, 8], between two Reshapes
            unit = F.layer_norm(both.reshape(1, 1, m.horizon, 8), (8,), self.halves, eps=1e-12)
            prediction = unit.reshape(1, m.horizon, 8)[..., 0:4]
        else:
            prediction = F.layer_norm(both, (8,), self.halves, eps=1e-12)[..., 0:4]  # now * (scalar, r), unit norm
        return (prediction, *new_states)


def _store_tied_weights_once(path, wrapper):
    """Rewrites the exported file: each [B, A, B] kernel, the fused head's [W; S W] and its bias (x biases and zeros)
    become Concat nodes over initializers holding A and B, W, and the x biases, each stored once."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(path))
    graph = model.graph
    stored = {t.name: t for t in graph.initializer}
    nodes = []

    def replace(name, initializers, built):
        graph.initializer.remove(stored[name])
        graph.initializer.extend(initializers)
        nodes.extend(built)

    def constant(name, value):
        return helper.make_node("Constant", [], [name], value=numpy_helper.from_array(value, name))

    if wrapper.padded_rows:
        for i in range(len(wrapper.pointwise)):
            name = f"pointwise.{i}"
            w = numpy_helper.to_array(stored[name])  # [K, K, 3, 1], rows B, A, B
            assert np.array_equal(w[:, :, 0], w[:, :, 2])
            a = numpy_helper.from_array(w[:, :, 1:2].copy(), f"{name}.A")
            b = numpy_helper.from_array(w[:, :, 0:1].copy(), f"{name}.B")
            replace(name, [a, b], [helper.make_node("Concat", [b.name, a.name, b.name], [name], axis=2)])
    if wrapper.fused_head:
        w = numpy_helper.to_array(stored["head_weight"])  # [3P, K, 2, 1], rows W, S W
        signs = wrapper.model.output_signs.numpy().astype(np.float32).reshape(-1, 1, 1, 1)
        assert np.array_equal(w[:, :, 1:2], signs * w[:, :, 0:1])
        weight = numpy_helper.from_array(w[:, :, 0:1].copy(), "head_weight.W")
        replace("head_weight", [weight], [constant("head_weight.signs", signs),
                                          helper.make_node("Mul", [weight.name, "head_weight.signs"], ["head_weight.SW"]),
                                          helper.make_node("Concat", [weight.name, "head_weight.SW"], ["head_weight"],
                                                           axis=2)])
        bias = numpy_helper.to_array(stored["head_bias"]).reshape(-1, 3)  # [P, 3]: x bias, 0, 0
        assert not bias[:, 1:].any()
        x_bias = numpy_helper.from_array(bias[:, :1].copy(), "head_bias.x")
        replace("head_bias", [x_bias], [constant("head_bias.zeros", np.zeros_like(bias[:, 1:])),
                                        constant("head_bias.shape", np.array([bias.size], np.int64)),
                                        helper.make_node("Concat", [x_bias.name, "head_bias.zeros"], ["head_bias.rows"],
                                                         axis=1),
                                        helper.make_node("Reshape", ["head_bias.rows", "head_bias.shape"], ["head_bias"])])
    rest = list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes + rest)  # the weight-building nodes come first, so the graph stays in order
    onnx.save(model, str(path))


def export_streaming_onnx(model, path, compact=True, trig_free=True, fused_head=False, padded_rows=False,
                          split_mirror=False, st_layout=False):
    """Same signature and outputs as the v5/v6/v7 exporters, plus fused_head (A4), padded_rows, split_mirror and
    st_layout."""
    if not (compact and trig_free):
        raise ValueError("The streaming graph is always compact and trig-free")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = StreamingLean(model, fused_head, padded_rows, split_mirror, st_layout).eval()
    example = (F.normalize(torch.randn(1, model.context, 4), dim=-1), *(torch.zeros(s) for s in wrapper.state_shapes))
    count = len(wrapper.state_shapes)
    with torch.no_grad():
        torch.onnx.export(wrapper, example, str(path), opset_version=OPSET, do_constant_folding=True, dynamo=False,
                          input_names=["context", *(f"state_in_{i}" for i in range(count))],
                          output_names=["prediction", *(f"state_out_{i}" for i in range(count))])
    if fused_head or padded_rows:
        _store_tied_weights_once(path, wrapper)
    model.eval()
    import onnx
    onnx.checker.check_model(onnx.load(str(path)))
    return path, wrapper.state_shapes


def export_onnx(*args, **kwargs):
    raise ValueError("The best model exports only the streaming graph: use --variants stream")
