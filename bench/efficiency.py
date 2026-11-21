"""Efficiency of an ONNX graph: parameters, MACs per call, operator audit, output checks, stream parity.

Everything here runs on generated motion (bench.geometry.random_motion), never on benchmark data, so
`python -m bench audit` needs no data and gives the same numbers on every machine.

MACs per call: one harness call, i.e. one streaming step. A stateful graph processes one new sample; a
stateless one the whole 400-sample window. Counted from the tensor shapes seen at run time, so dynamic
dimensions are resolved. Ported from ABC_local_test/experimental/team_comparison/compare_nic.py
(runtime_macs: MatMul, Gemm, Conv, GRU), extended with LSTM, RNN and Einsum. Only multiply-accumulates of
these operators are counted (no elementwise work, no recurrent gate products or biases), as is usual.

Operator lists: the Submitting Artifacts page as summarised in nic/FINDINGS.md R4 (25 Oct). The raw page
text is not saved in ABC_local_test/references, so this is the only copy we have.
"""
import math
from collections import Counter
from pathlib import Path

import numpy as np
import onnx

from bench.data import sha256_file
from bench.drivers import OnnxModel, interface, ort_session
from bench.geometry import geodesic_degrees, random_motion, renormalise

SAFE_OPS = {"Conv", "MatMul", "Gemm", "LayerNormalization", "BatchNormalization", "Relu", "Clip",  # Clip = ReLU6
            "HardSwish", "Gelu", "Sigmoid", "Tanh", "Add", "Mul", "Concat", "Slice", "Reshape", "Transpose",
            "GlobalAveragePool"}
AVOID_OPS = {"Loop", "If", "Scan", "NonZero", "TopK", "SequenceConstruct", "SequenceAt", "SequenceInsert",
             "SequenceEmpty", "SequenceErase", "SequenceLength", "SplitToSequence", "ConcatFromSequence"}
CONDITIONAL = {"Softmax": "allowed only as plain MatMul-Softmax-MatMul attention",
               "Einsum": "'big einsum' is on the avoid list"}
SLOW_ON_NPU = {"Sin", "Cos", "Tan", "Asin", "Acos", "Atan"}  # "trig / arccos: supported but slow on NPUs"
RECURRENT = {"GRU", "LSTM", "RNN"}
MAX_RECURRENCE = 64  # "unrolled recurrences deeper than ~64 steps" are on the avoid list


def parameter_counts(model):
    """Initializer elements (what the leaderboard calls ONNX parameters) and elements held in Constant nodes.

    Neither has to equal the trainable parameter count: an export can fold weights into Constants (v6's head
    bias) or store tensors derived from them. Both are reported next to the declared count.
    """
    initializer = sum(math.prod(t.dims) for t in model.graph.initializer)
    constant = 0
    for node in model.graph.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                constant += math.prod(a.t.dims) if a.name == "value" else len(a.floats) or len(a.ints) or 1
    return {"initializer_elements": int(initializer), "constant_node_elements": int(constant)}


def zero_feeds(session, window):
    context, _, states = interface(session)
    return {context: window, **{i: np.zeros(shape, np.float32) for i, _, shape in states}}


def runtime_shapes(model, feeds):
    """Shape of every intermediate tensor, by exposing all node outputs and running the graph once."""
    model = onnx.ModelProto.FromString(model.SerializeToString())
    existing = {o.name for o in model.graph.output}
    for node in model.graph.node:
        for name in node.output:
            if name and name not in existing:
                model.graph.output.append(onnx.ValueInfoProto(name=name))
                existing.add(name)
    session = ort_session(model.SerializeToString(), 1)
    shapes = {o.name: tuple(v.shape) for o, v in zip(session.get_outputs(), session.run(None, feeds))}
    shapes.update({k: tuple(v.shape) for k, v in feeds.items()})
    shapes.update({t.name: tuple(t.dims) for t in model.graph.initializer})
    return shapes


def _einsum_macs(node, shape):
    """Product of all index sizes for a multi-operand contraction (e.g. bij,bjk->bik: b*i*j*k)."""
    equation = next(a.s.decode() for a in node.attribute if a.name == "equation").replace(" ", "")
    operands = equation.split("->")[0].split(",")
    if len(operands) < 2 or "..." in equation:
        return 0 if len(operands) < 2 else None
    sizes = {}
    for letters, name in zip(operands, node.input):
        sizes.update(zip(letters, shape(name)))
    return math.prod(sizes.values())


def count_macs(model, shapes):
    shape = shapes.__getitem__
    by_op, recurrences, unknown = Counter(), [], []
    for node in model.graph.node:
        op = node.op_type
        if op in ("MatMul", "Gemm"):
            a, out = shape(node.input[0]), shape(node.output[0])
            transposed = op == "Gemm" and any(x.name == "transA" and x.i for x in node.attribute)
            macs = math.prod(out) * (a[-2] if transposed else a[-1])
        elif op == "Conv":
            macs = math.prod(shape(node.output[0])) * math.prod(shape(node.input[1])[1:])
        elif op in RECURRENT:
            # W [directions, gates*hidden, input], R [directions, gates*hidden, hidden]; X [seq, batch, input]
            # (layout 0) or [batch, seq, input] (layout 1). Per step: gates*hidden*(input + hidden) per direction.
            w, r, x = shape(node.input[1]), shape(node.input[2]), shape(node.input[0])
            layout = next((a.i for a in node.attribute if a.name == "layout"), 0)
            steps, batch = (x[1], x[0]) if layout else (x[0], x[1])
            macs = steps * batch * w[0] * w[1] * (w[2] + r[2])
            recurrences.append({"op": op, "name": node.name, "steps": int(steps)})
        elif op == "Einsum":
            macs = _einsum_macs(node, shape)
            if macs is None:
                unknown.append(node.name or op)
                continue
        else:
            continue
        by_op[op] += int(macs)
    return {"total": int(sum(by_op.values())), "by_operator": dict(by_op), "recurrences": recurrences,
            "uncounted_einsum": unknown}


def audit_operators(model, recurrences=()):
    """Operators against the safe/avoid lists, plus float64 anywhere, custom domains and long recurrences."""
    counts = Counter(n.op_type for n in model.graph.node)
    custom = sorted({f"{n.domain}:{n.op_type}" for n in model.graph.node if n.domain not in ("", "ai.onnx")})
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:  # custom operators can stop shape inference; the raw graph is still audited
        inferred = model
    g, double = inferred.graph, onnx.TensorProto.DOUBLE
    float64 = [t.name for t in g.initializer if t.data_type == double]
    float64 += [v.name for v in (*g.input, *g.output, *g.value_info) if v.type.tensor_type.elem_type == double]
    float64 += [n.name or n.output[0] for n in g.node if n.op_type == "Cast"
                and any(a.name == "to" and a.i == double for a in n.attribute)]
    float64 += [n.output[0] for n in g.node if n.op_type == "Constant"
                and any(a.name == "value" and a.t.data_type == double for a in n.attribute)]
    sequences = [v.name for v in (*g.input, *g.output, *g.value_info) if v.type.HasField("sequence_type")]
    long = [r for r in recurrences if r["steps"] > MAX_RECURRENCE]
    ops = set(counts) - {"Constant"}  # Constant holds weights, not work
    issues = sorted(ops & AVOID_OPS) + custom
    issues += ["float64"] * bool(float64) + ["sequence types"] * bool(sequences)
    issues += [f"{r['op']} over {r['steps']} steps (> {MAX_RECURRENCE})" for r in long]
    return {"counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "safe": sorted(ops & SAFE_OPS), "avoid": sorted(ops & AVOID_OPS),
            "conditional": {op: CONDITIONAL[op] for op in sorted(ops & set(CONDITIONAL))},
            "other": sorted(ops - SAFE_OPS - AVOID_OPS - set(CONDITIONAL)),
            "slow_on_npu": sorted(ops & SLOW_ON_NPU), "custom_domain": custom,
            "float64_tensors": float64, "sequence_tensors": sequences, "long_recurrences": long,
            "issues": issues}


def output_checks(path, calls=64, seed=1):
    """Shape and finiteness, bitwise repeatability in a fresh session, and drift between 1 and 4 threads."""
    q = random_motion(400 + calls + 19, seed)  # exactly `calls` full windows
    runs = {}
    for key, threads in (("first", 1), ("again", 1), ("threads4", 4)):
        runs[key] = OnnxModel(path, threads=threads).stream(q)
    first = runs["first"]
    norms = np.linalg.norm(first.astype(np.float64), axis=-1)
    return {"calls": int(len(first)), "shape_ok": first.shape[1:] == (20, 4), "finite": bool(np.isfinite(first).all()),
            "output_norm_range": [float(norms.min()), float(norms.max())],
            "deterministic": bool(np.array_equal(first, runs["again"])),
            "max_abs_diff_rerun": float(np.abs(first - runs["again"]).max()),
            "max_abs_diff_1_vs_4_threads": float(np.abs(first - runs["threads4"]).max())}


def stream_parity(stateful_path, stateless_path, windows=1000, seed=2):
    """A stateful graph streamed from the first sample vs a stateless reference on every full window."""
    q = random_motion(400 + windows + 19, seed)
    a = renormalise(OnnxModel(stateful_path).stream(q))
    b = renormalise(OnnxModel(stateless_path).stream(q))
    if a.shape != b.shape:
        raise ValueError(f"Parity: shapes differ {a.shape} vs {b.shape}")
    difference = geodesic_degrees(a, b)
    return {"windows": int(len(a)), "max_difference_deg": float(difference.max()),
            "mean_difference_deg": float(difference.mean()), "step20_max_difference_deg": float(difference[:, -1].max()),
            "reference": Path(stateless_path).name, "reference_sha256": sha256_file(stateless_path)}


def report(path, stateless_reference=None):
    model = onnx.load(str(path))
    session = ort_session(path, 1)
    window = random_motion(400, 0)[None]
    feeds = zero_feeds(session, window)
    macs = count_macs(model, runtime_shapes(model, feeds))
    _, _, states = interface(session)
    result = {"opset": [{"domain": o.domain or "ai.onnx", "version": o.version} for o in model.opset_import],
              "stateful": bool(states),
              "states": [{"name": i, "shape": list(s)} for i, _, s in states],
              "state_elements": int(sum(math.prod(s) for _, _, s in states)),
              **parameter_counts(model), "macs_per_call": macs,
              "operators": audit_operators(model, macs["recurrences"]),
              "output_checks": output_checks(path), "stream_parity": None}
    if stateless_reference:
        result["stream_parity"] = stream_parity(path, stateless_reference)
    return result

