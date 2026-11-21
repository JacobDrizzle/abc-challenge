"""ONNX export of a trained TinyTCNN: same weights and maths, rebuilt from portable operators.

TinyTCNN.forward uses cumprod, sinc, atan2 and expm1, which have no ONNX operator (cumprod) or
export as slow/unsupported patterns. ExportableTCNN reuses the trained submodules and expresses
the same computation with Conv, MatMul, Gemm, Concat, Slice and elementwise operators. The graph
has the harness interface: input "context" [1, 400, 4] -> output "prediction" [1, 20, 4], (w, x, y, z).
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tcnn.model import TinyTCNN

OPSET = 17
# Submitting Artifacts page, "Operator constraints", as ONNX operator names.
SAFE_OPS = {"Conv", "MatMul", "Gemm", "LayerNormalization", "BatchNormalization", "Relu", "Clip", "HardSwish",
            "Gelu", "Sigmoid", "Tanh", "Add", "Mul", "Concat", "Slice", "Reshape", "Transpose", "GlobalAveragePool"}
AVOID_OPS = {"Loop", "If", "Scan", "NonZero", "TopK", "SequenceConstruct", "SequenceAt", "SequenceInsert",
             "SequenceEmpty", "SplitToSequence", "ConcatFromSequence"}
SLOW_ON_NPU = {"Sin", "Cos", "Atan", "Acos", "Asin", "Tan", "Exp", "Log"}


def _qmul(a, b):
    aw, ax, ay, az = a[..., 0:1], a[..., 1:2], a[..., 2:3], a[..., 3:4]
    bw, bx, by, bz = b[..., 0:1], b[..., 1:2], b[..., 2:3], b[..., 3:4]
    return torch.cat((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                      aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)


def _conjugate(q):
    return torch.cat((q[..., 0:1], -q[..., 1:4]), -1)


def _normalize(q):
    return q / torch.sqrt((q * q).sum(-1, keepdim=True)).clamp_min(1e-12)


def _rotvec(q):
    """Shortest SO(3) logarithm; matches tcnn.model.quaternion_to_rotvec."""
    q = _normalize(q)
    q = q * torch.where(q[..., 0:1] < 0, -1., 1.)
    sine = torch.sqrt((q[..., 1:4] * q[..., 1:4]).sum(-1, keepdim=True))
    # w >= 0 and sine >= 0, so atan2(sine, w) == atan(sine / w); w == 0 gives atan(inf) == pi/2.
    angle = 2 * torch.atan(sine / q[..., 0:1].clamp_min(1e-30))
    scale = torch.where(sine > 1e-6, angle / sine.clamp_min(1e-6), 2 + sine * sine / 3)
    return q[..., 1:4] * scale


def _exp_quaternion(v):
    """Exp map; matches tcnn.model.rotvec_to_quaternion (sinc written as sin(a/2)/a)."""
    angle = torch.sqrt((v * v).sum(-1, keepdim=True))
    half = torch.where(angle > 1e-6, torch.sin(angle / 2) / angle.clamp_min(1e-6), .5 - angle * angle / 48)
    return _normalize(torch.cat((torch.cos(angle / 2), half * v), -1))


# ---- compact graph: quaternion products as MatMul, trigonometry as polynomials (Mul/Add only) ----

def _power_series(function, high, degree):
    """Chebyshev least-squares fit on [0, high], as power-series coefficients for Horner's rule."""
    from numpy.polynomial import Chebyshev, Polynomial
    nodes = high * (np.cos(np.pi * (np.arange(4000) + .5) / 4000) + 1) / 2
    return Chebyshev.fit(nodes, function(nodes), degree, domain=[0, high]).convert(kind=Polynomial).coef


def _safe_sqrt_ratio(function, x):
    root = np.sqrt(np.maximum(x, 1e-300))
    return np.where(x > 0, function(root) / root, 1.)


# atan(r)/r as a polynomial in r*r on [0, 1]; float32 error 9e-8. Coefficients are built once here,
# outside any trace, so the exporter stores them as constants.
ATAN_SERIES = [float(c) for c in _power_series(lambda x: _safe_sqrt_ratio(np.arctan, x), 1., 8)]
# cos(h) and sin(h)/h as polynomials in h*h for h in [0, pi] (rotations up to 360 degrees); error 5e-7.
COS_SINC_SERIES = list(torch.tensor(np.stack((
    _power_series(lambda x: np.cos(np.sqrt(x)), np.pi**2, 6),
    _power_series(lambda x: _safe_sqrt_ratio(np.sin, x), np.pi**2, 6)), 1), dtype=torch.float32))


def _horner(x, coefficients):
    total = coefficients[-1] * x + coefficients[-2]
    for c in reversed(coefficients[:-2]):
        total = total * x + c
    return total


def _structure_constants():
    """T[j, k] = e_j * e_k for the quaternion basis, so a * b = sum_jk a_j b_k T[j, k]."""
    from tcnn.model import qmul
    basis = torch.eye(4, dtype=torch.float64)
    return qmul(basis[:, None, :].expand(4, 4, 4), basis[None, :, :].expand(4, 4, 4))


def _bilinear_tables():
    t, conj = _structure_constants(), torch.tensor([1., -1., -1., -1.], dtype=torch.float64)
    return {"left": t.reshape(4, 16),                          # last -> M with last*b == b @ M
            "left_conj": (conj[:, None, None] * t).reshape(4, 16),  # last -> M with conj(last)*b == b @ M
            "a_bconj": (t * conj[None, :, None]).reshape(16, 4),    # outer(a, b) @ . == a * conj(b)
            "aconj_b": (conj[:, None, None] * t).reshape(16, 4)}    # outer(a, b) @ . == conj(a) * b


def _left_matrix(table, q):
    """[1, 1, 4] quaternion -> [1, 4, 4] matrix of the linear map b -> q * b (or conj(q) * b)."""
    return torch.matmul(q, table).reshape(1, 4, 4)


def _outer_product(table, a, b):
    """Row-wise bilinear quaternion product of [1, N, 4] tensors: per-row matrices of a, applied to b."""
    n = a.shape[1]
    per_row = torch.matmul(a, table.reshape(4, 4, 4).reshape(4, 16)).reshape(1, n, 4, 4)
    return torch.matmul(b.reshape(1, n, 1, 4), per_row).reshape(1, n, 4)


def _compact_rotvec(q):
    """Shortest SO(3) logarithm without Atan; scale-invariant, so q need not be unit length."""
    w = q[..., 0:1]
    sign = torch.where(w < 0, -1., 1.)
    w, vector = w * sign, q[..., 1:4] * sign
    sine2 = (vector * vector).sum(-1, keepdim=True)
    small = sine2 <= w * w  # rotation of at most 90 degrees, so atan's argument is at most 1
    sine = torch.sqrt(sine2).clamp_min(1e-30)
    ratio2 = torch.where(small, sine2 / (w * w).clamp_min(1e-30), w * w / sine2.clamp_min(1e-30))
    series = _horner(ratio2, ATAN_SERIES)
    # angle / sine with angle = 2 atan(sine / w), or 2 (pi/2 - atan(w / sine)) beyond 90 degrees.
    scale = torch.where(small, 2 * series / w.clamp_min(1e-30), (math.pi - 2 * (w / sine) * series) / sine)
    return vector * scale


def _compact_exp(v):
    """Unnormalised exp map (cos(h), sin(h)/h * v/2) with h = |v|/2, using Mul/Add only."""
    half2 = (v * v).sum(-1, keepdim=True) * .25
    cos_sinc = _horner(half2, COS_SINC_SERIES)
    return torch.cat((cos_sinc[..., 0:1], cos_sinc[..., 1:2] * (.5 * v)), -1)


class ExportableTCNN(nn.Module):
    """Batch-1 graph equal to TinyTCNN.forward in eval mode.

    strided=True computes, in each block, only the time steps the final prediction depends on. Block i
    (dilation 2**i) reads its input every 2**i steps back from the last one, so in those grid coordinates
    it is an ordinary kernel-3 convolution, and the next block needs every second output: a stride-2 Conv.

    compact=True writes the quaternion products (with the last orientation, and row by row) as MatMul.
    trig_free=True replaces atan, sin and cos with polynomials (Mul/Add), which the challenge prefers
    for NPUs; on a desktop CPU these are slightly slower than the built-in operators. All options keep
    the weights and the predictions (to float32 rounding).
    """

    def __init__(self, model: TinyTCNN, strided=False, compact=False, trig_free=False):
        super().__init__()
        self.model, self.strided, self.compact, self.trig_free = model.eval(), strided, compact, trig_free
        self.body = getattr(model, "input_features", "anchor") == "body_velocity"
        n = model.effective_context
        # Inclusive prefix count of sign flips as a MatMul: flips[1, n-1] @ upper[n-1, n-1].
        if not self.body:
            self.register_buffer("prefix", torch.triu(torch.ones(n - 1, n - 1)), persistent=False)
        for name, table in _bilinear_tables().items():
            self.register_buffer(f"table_{name}", table.float(), persistent=False)
        self.starts, length = [], n
        for i, block in enumerate(model.blocks):
            if strided and block.dilation != 2 ** i:
                raise ValueError("Strided export needs dilations 1, 2, 4, ...")
            # First kept grid index so that the kept outputs end on the last time step; the zero
            # padding shrinks by the same amount, so no Slice is needed to drop it.
            self.starts.append((length - 1) % 2)
            length = (length - 1 - self.starts[-1]) // 2 + 1
            width = 2 - self.starts[-1] if strided else block.left_padding
            self.register_buffer(f"pad_{i}", torch.zeros(1, model.channels, width), persistent=False)
        if model.damped_prior_tau_ms is not None:
            tau = model.damped_prior_tau_ms / 10  # ms to 100 Hz samples, as in TinyTCNN.forward
            steps = torch.arange(1, model.horizon + 1, dtype=torch.float64)
            self.register_buffer("elapsed", (-tau * torch.expm1(-steps / tau)).float()[None, :, None],
                                 persistent=False)

    def features(self, context):
        """(features [1, N, F], last orientation [1, 1, 4], body-frame velocity of the last step [1, 1, 3])."""
        m, rotvec = self.model, _compact_rotvec if self.trig_free else _rotvec
        if self.body:
            # Scale-invariant maths: the window needs no normalisation and no sign alignment.
            q = context[:, -m.effective_context:] if self.compact else _normalize(context[:, -m.effective_context:])
            delta = (_outer_product(self.table_aconj_b, q[:, :-1], q[:, 1:]) if self.compact
                     else _qmul(_conjugate(q[:, :-1]), q[:, 1:]))
            omega = rotvec(delta) * 100.
            speed = torch.sqrt((omega * omega).sum(-1, keepdim=True))
            features = torch.cat((torch.zeros_like(context[:, :1]), torch.cat((omega, speed), -1)), 1)
            return features, q[:, -1:], omega[:, -1:] / 100.
        q = _normalize(context[:, -m.effective_context:])
        # cumprod of +/-1 step signs == (-1)^(number of flips so far).
        flips = torch.where((q[:, 1:] * q[:, :-1]).sum(-1) < 0, 1., 0.)
        count = torch.matmul(flips, self.prefix)
        signs = torch.cat((torch.ones_like(count[:, :1]), 1 - 2 * (count - 2 * torch.floor(count / 2))), 1)
        q = q * signs.reshape(1, -1, 1)
        last = q[:, -1:]
        relative = (torch.matmul(q, _left_matrix(self.table_left_conj, last)) if self.compact
                    else _qmul(_conjugate(last), q))
        if not m.velocity_features:
            conj_previous = relative[:, -2:-1] * torch.tensor([1., -1., -1., -1.])
            return relative, last, rotvec(conj_previous)
        delta = (_outer_product(self.table_a_bconj, relative[:, 1:], relative[:, :-1]) if self.compact
                 else _qmul(relative[:, 1:], _conjugate(relative[:, :-1])))
        velocity = rotvec(delta) * 100.
        # relative[-1] is the identity, so the last velocity row is the damped prior's rotation per step.
        features = torch.cat((relative, torch.cat((torch.zeros_like(velocity[:, :1]), velocity), 1)), -1)
        return features, last, velocity[:, -1:] / 100.

    def network(self, features):
        m = self.model
        x = F.relu(m.projection(features.transpose(1, 2)))
        for i, block in enumerate(m.blocks):
            padded = torch.cat((getattr(self, f"pad_{i}"), x), -1)
            if self.strided:
                update = F.conv1d(padded, block.conv.weight, block.conv.bias, stride=2)
                x = F.relu(x[:, :, self.starts[i]::2] + F.relu(update))
            else:
                x = F.relu(x + F.relu(F.conv1d(padded, block.conv.weight, block.conv.bias, dilation=block.dilation)))
        return m.head(x[:, :, -1:].reshape(1, m.channels)).reshape(1, m.horizon, 3)

    def output(self, rotation_vectors, last):
        exp = _compact_exp(rotation_vectors) if self.trig_free else _exp_quaternion(rotation_vectors)
        if self.compact:
            return _normalize(torch.matmul(exp, _left_matrix(self.table_left, last)))
        return _normalize(_qmul(last, exp))

    def forward(self, context):
        features, last, step = self.features(context)
        rotation_vectors = self.network(features)
        if self.model.damped_prior_tau_ms is not None:
            rotation_vectors = rotation_vectors + step * self.elapsed
        return self.output(rotation_vectors, last)


class StreamingTCNN(nn.Module):
    """Stateful graph for a body_velocity TinyTCNN: one new time step per call, layers cached as state.

    Inputs: context [1, 400, 4] (only the last two samples are read), then state_in_i [1, C, 2 * 2**i]:
    block i's inputs at the previous 2 * 2**i steps. Outputs: prediction [1, 20, 4], then state_out_i.
    Zero-initialised states equal the stateless model's zero padding, so once a session has as many
    samples as the window, every prediction equals TinyTCNN.forward on that window (float32 rounding).
    """

    def __init__(self, model: TinyTCNN, compact=True, trig_free=True):
        super().__init__()
        if getattr(model, "input_features", "anchor") != "body_velocity":
            raise ValueError("Streaming needs input_features='body_velocity': anchor-relative inputs change "
                             "for every past step whenever a new sample arrives")
        self.model, self.compact, self.trig_free = model.eval(), compact, trig_free
        self.state_shapes = [(1, model.channels, block.left_padding) for block in model.blocks]
        for name, table in _bilinear_tables().items():
            self.register_buffer(f"table_{name}", table.float(), persistent=False)
        if model.damped_prior_tau_ms is not None:
            tau = model.damped_prior_tau_ms / 10
            steps = torch.arange(1, model.horizon + 1, dtype=torch.float64)
            self.register_buffer("elapsed", (-tau * torch.expm1(-steps / tau)).float()[None, :, None],
                                 persistent=False)

    def forward(self, context, *states):
        m = self.model
        previous, now = context[:, -2:-1], context[:, -1:]
        if self.compact:
            delta = _outer_product(self.table_aconj_b, previous, now)
        else:
            delta = _qmul(_conjugate(_normalize(previous)), _normalize(now))
        step = _compact_rotvec(delta) if self.trig_free else _rotvec(delta)
        velocity = step * 100.
        speed = torch.sqrt((velocity * velocity).sum(-1, keepdim=True))
        x = F.relu(m.projection(torch.cat((velocity, speed), -1).reshape(1, 4, 1)))
        new_states = []
        for block, state in zip(m.blocks, states):
            window = torch.cat((state, x), -1)  # block input at t - 2d ... t
            new_states.append(window[:, :, 1:])
            update = F.conv1d(window, block.conv.weight, block.conv.bias, dilation=block.dilation)
            x = F.relu(x + F.relu(update))
        rotation_vectors = m.head(x.reshape(1, m.channels)).reshape(1, m.horizon, 3)
        if m.damped_prior_tau_ms is not None:
            rotation_vectors = rotation_vectors + step * self.elapsed
        exp = _compact_exp(rotation_vectors) if self.trig_free else _exp_quaternion(rotation_vectors)
        if self.compact:
            prediction = _normalize(torch.matmul(exp, _left_matrix(self.table_left, now)))
        else:
            prediction = _normalize(_qmul(_normalize(now), exp))
        return (prediction, *new_states)


def export_streaming_onnx(model: TinyTCNN, path, compact=True, trig_free=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = StreamingTCNN(model, compact, trig_free).eval()
    example = (F.normalize(torch.randn(1, model.context, 4), dim=-1), *(torch.zeros(s) for s in wrapper.state_shapes))
    count = len(wrapper.state_shapes)
    with torch.no_grad():
        torch.onnx.export(wrapper, example, str(path), opset_version=OPSET, do_constant_folding=True, dynamo=False,
                          input_names=["context", *(f"state_in_{i}" for i in range(count))],
                          output_names=["prediction", *(f"state_out_{i}" for i in range(count))])
    model.eval()
    import onnx
    onnx.checker.check_model(onnx.load(str(path)))
    return path, wrapper.state_shapes


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyTCNN(**state["model_spec"])
    model.load_state_dict(state["model_state"])
    return model.eval(), state


def _fold_zero_padding(graph):
    """Concat(zeros, x) feeding only Conv nodes -> the Conv's own left padding (one node less, no copy)."""
    import onnx
    from onnx import numpy_helper
    zeros = {t.name for t in graph.initializer if not numpy_helper.to_array(t).any()}
    zeros |= {n.output[0] for n in graph.node if n.op_type == "Constant" and n.attribute[0].name == "value"
              and not numpy_helper.to_array(n.attribute[0].t).any()}
    def last_dim(array):
        return array.shape[-1] if array.ndim else 0
    width = {t.name: last_dim(numpy_helper.to_array(t)) for t in graph.initializer}
    width.update({n.output[0]: last_dim(numpy_helper.to_array(n.attribute[0].t))
                  for n in graph.node if n.op_type == "Constant" and n.attribute[0].name == "value"})
    consumers = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    removed = []
    for node in graph.node:
        axis = next((a.i for a in node.attribute if a.name == "axis"), None)
        if (node.op_type != "Concat" or len(node.input) != 2 or node.input[0] not in zeros or axis not in (2, -1)
                or any(c.op_type != "Conv" for c in consumers.get(node.output[0], []))):
            continue
        for conv in consumers[node.output[0]]:
            pads = next((a for a in conv.attribute if a.name == "pads"), None)
            if pads is not None and any(pads.ints):
                break
            if pads is not None:
                conv.attribute.remove(pads)
            conv.attribute.append(onnx.helper.make_attribute("pads", [int(width[node.input[0]]), 0]))
            conv.input[0] = node.input[1]
        else:
            removed.append(node)
    for node in removed:
        graph.node.remove(node)
    used = {name for node in graph.node for name in node.input}
    for node in [n for n in graph.node if n.op_type == "Constant" and n.output[0] not in used]:
        graph.node.remove(node)
    for tensor in [t for t in graph.initializer if t.name not in used]:
        graph.initializer.remove(tensor)
    return len(removed)


def export_onnx(model: TinyTCNN, path, strided=False, compact=False, trig_free=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    example = F.normalize(torch.randn(1, model.context, 4), dim=-1)
    # The exporter restores the wrapper's training flag on all submodules afterwards; a fresh module
    # is in training mode, which would silently re-enable dropout in `model`.
    with torch.no_grad():
        torch.onnx.export(ExportableTCNN(model, strided, compact, trig_free).eval(), (example,), str(path),
                          input_names=["context"],
                          output_names=["prediction"], opset_version=OPSET, do_constant_folding=True, dynamo=False)
    model.eval()
    import onnx
    graph = onnx.load(str(path))
    if compact:
        _fold_zero_padding(graph.graph)
        onnx.save(graph, str(path))
    onnx.checker.check_model(graph)
    return path


def audit_operators(path):
    """Operator counts against the challenge's safe/avoid lists, and any float64 tensor."""
    import onnx
    graph = onnx.load(str(path)).graph
    counts = Counter(node.op_type for node in graph.node)
    double = onnx.TensorProto.DOUBLE
    float64 = [t.name for t in graph.initializer if t.data_type == double]
    float64 += [v.name for v in (*graph.input, *graph.output, *graph.value_info) if v.type.tensor_type.elem_type == double]
    return {"counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "safe": sorted(set(counts) & SAFE_OPS), "avoid": sorted(set(counts) & AVOID_OPS),
            "other": sorted(set(counts) - SAFE_OPS - AVOID_OPS - {"Constant"}),  # Constant holds weights, not work
            "slow_on_npu": sorted(set(counts) & SLOW_ON_NPU), "float64_tensors": float64}


def count_macs(path):
    """Multiply-accumulates per call (one streaming step) from Conv, MatMul and Gemm nodes."""
    import onnx
    model = onnx.shape_inference.infer_shapes(onnx.load(str(path)))
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in (*model.graph.input, *model.graph.output, *model.graph.value_info)}
    shapes.update({t.name: list(t.dims) for t in model.graph.initializer})
    per_op = Counter()
    for node in model.graph.node:
        if node.op_type == "Conv":
            weight, out = shapes[node.input[1]], shapes[node.output[0]]
            per_op["Conv"] += math.prod(out) * math.prod(weight[1:])
        elif node.op_type in ("MatMul", "Gemm"):
            a, out = shapes[node.input[0]], shapes[node.output[0]]
            transposed = node.op_type == "Gemm" and any(x.name == "transA" and x.i for x in node.attribute)
            per_op[node.op_type] += math.prod(out) * (a[-2] if transposed else a[-1])
    return {"total": sum(per_op.values()), "by_operator": dict(per_op)}


def geodesic_degrees(a, b):
    """Angle between quaternion arrays [..., 4] in float64, stable for tiny angles."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    b = np.where((a * b).sum(-1, keepdims=True) < 0, -b, b)
    return np.rad2deg(4 * np.arctan2(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1)))
