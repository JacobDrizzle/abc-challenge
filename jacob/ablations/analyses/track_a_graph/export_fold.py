"""Load-time folding for the bab layout: the file stores each block's [A | B] once and builds the (3, 1) kernel
[B, A, B] with a Concat over initializers, which ORT's constant folding evaluates once at session start. Same for
the A4 head ([W; S W] from W) and its bias (the x biases plus zeros). Checks: stored floats, outputs identical to
the unfolded file, ORT's optimised node count equal to the unfolded file's, and timing in alternating rounds.
Reads the graphs written by export_variants.py; the CPU is contended: relative numbers only.
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort

TRAINING = Path(r"<repo>\jacob\training")
sys.path.insert(0, str(TRAINING))
from benchmark_latency import Streamer, ort_session, time_stream  # noqa: E402

HERE = Path(__file__).resolve().parent / "export_variants"


def fold_at_load(source, target):
    model = onnx.load(str(source))
    graph, built = model.graph, []
    for init in list(graph.initializer):
        w = numpy_helper.to_array(init)
        if init.name.startswith("pointwise.") and w.shape[2] == 3:  # rows B, A, B
            a, b = numpy_helper.from_array(w[:, :, 1:2], init.name + ".A"), numpy_helper.from_array(w[:, :, 0:1], init.name + ".B")
            built += [(init, [a, b], [helper.make_node("Concat", [b.name, a.name, b.name], [init.name], axis=2)])]
        elif init.name == "head_weight":  # [3P, K, 2, 1]: W, S W
            sw = w[:, :, 1:2] / np.where(w[:, :, 0:1] == 0, 1, w[:, :, 0:1])
            signs = np.sign(np.median(sw, axis=1, keepdims=True)).astype(np.float32)  # [3P, 1, 1, 1]
            assert np.allclose(w[:, :, 1:2], signs * w[:, :, 0:1])
            base = numpy_helper.from_array(w[:, :, 0:1], "head_weight.W")
            nodes = [helper.make_node("Constant", [], ["head_signs"], value=numpy_helper.from_array(signs, "s")),
                     helper.make_node("Mul", [base.name, "head_signs"], ["head_weight.SW"]),
                     helper.make_node("Concat", [base.name, "head_weight.SW"], [init.name], axis=2)]
            built += [(init, [base], nodes)]
        elif init.name == "head_bias":  # [3P]: x bias, 0, 0 per basis term
            p = w.size // 3
            assert np.all(w.reshape(p, 3)[:, 1:] == 0)
            xb = numpy_helper.from_array(w.reshape(p, 3)[:, :1].copy(), "head_bias.x")
            nodes = [helper.make_node("Constant", [], ["head_bias.zeros"], value=numpy_helper.from_array(np.zeros((p, 2), np.float32), "z")),
                     helper.make_node("Constant", [], ["head_bias.shape"], value=numpy_helper.from_array(np.array([3 * p], np.int64), "s")),
                     helper.make_node("Concat", [xb.name, "head_bias.zeros"], ["head_bias.rows"], axis=1),
                     helper.make_node("Reshape", ["head_bias.rows", "head_bias.shape"], [init.name])]
            built += [(init, [xb], nodes)]
    prefix = []
    for init, stored, nodes in built:
        graph.initializer.remove(init)
        graph.initializer.extend(stored)
        prefix += nodes
    rest = list(graph.node)
    del graph.node[:]
    graph.node.extend(prefix + rest)
    onnx.checker.check_model(model)
    onnx.save(model, str(target))
    return target


def optimised_nodes(path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC  # constant folding, no layout
    out = path.with_suffix(".basic_opt.onnx")
    options.optimized_model_filepath = str(out)
    ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    graph = onnx.load(str(out)).graph
    return sum(n.op_type != "Constant" for n in graph.node), int(sum(np.prod(t.dims) for t in graph.initializer))


def stored(path):
    graph = onnx.load(str(path)).graph
    return sum(n.op_type != "Constant" for n in graph.node), int(sum(np.prod(t.dims) for t in graph.initializer))


def main():
    files = {"now": HERE / "now.onnx", "bab": HERE / "bab.onnx", "bab_fused": HERE / "bab_fused.onnx"}
    files["bab_fold"] = fold_at_load(files["bab"], HERE / "bab_fold.onnx")
    files["bab_fused_fold"] = fold_at_load(files["bab_fused"], HERE / "bab_fused_fold.onnx")
    rng = np.random.default_rng(0)
    stream = np.cumsum(rng.normal(0, .01, (3000, 4)), 0) + [1, 0, 0, 0]
    stream = (stream / np.linalg.norm(stream, axis=1, keepdims=True)).astype(np.float32)
    calls = {}
    for name, path in files.items():
        session = ort_session(path, 1)
        calls[name] = Streamer(session, [i.shape for i in session.get_inputs()[1:]])
        print(f"{name:15s} file nodes/stored {stored(path)}  after ORT basic optimisation {optimised_nodes(path)}")
    # Identical outputs: the folded files against their unfolded twins on one stream.
    for a, b in (("bab", "bab_fold"), ("bab_fused", "bab_fused_fold"), ("now", "bab")):
        calls[a].reset(); calls[b].reset()
        worst = 0.
        for t in range(400, 1400):
            pa, pb = calls[a](stream[t - 400:t][None]), calls[b](stream[t - 400:t][None])
            worst = max(worst, float(np.abs(pa - pb).max()))
        print(f"max |{a} - {b}| over 1000 steps: {worst:.2e}")
    timings = {k: [] for k in calls}
    for _ in range(5):
        for name, call in calls.items():
            call.reset()
            timings[name] += time_stream(call, stream, 400, 300, 2000)
    base = np.median(timings["now"])
    for name, t in timings.items():
        print(f"{name:15s} median {np.median(t) * 1e6:5.1f} us ({100 * (np.median(t) / base - 1):+.1f}%)  p90 {np.percentile(t, 90) * 1e6:5.1f}")


if __name__ == "__main__":
    main()
