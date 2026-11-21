"""Per-operator time inside ORT (CPU, 1 thread) for the v7 and v9 lean graphs exported by trackA_latency.py.

Kernel times from ORT's profiler, summed by operator type over 2,000 streaming calls. Run while training is on the
GPU, so absolute numbers are inflated; the shares are what this is for.
"""
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort

HERE = Path(__file__).resolve().parent / "trackA_latency"
CALLS = 2000


def profile(path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads, options.inter_op_num_threads = 1, 1
    options.enable_profiling = True
    options.profile_file_prefix = str(HERE / f"profile_{path.stem}")
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    states = [np.zeros([d for d in i.shape], np.float32) for i in inputs[1:]]
    rng = np.random.default_rng(0)
    q = rng.normal(size=(1, 400, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    for _ in range(CALLS):
        outputs = session.run(None, {"context": q, **{i.name: s for i, s in zip(inputs[1:], states)}})
        states = outputs[1:]
    trace = json.loads(Path(session.end_profiling()).read_text())
    by_op, nodes, runs = defaultdict(float), set(), []
    for event in trace:
        if event.get("cat") == "Node" and event["name"].endswith("_kernel_time"):
            by_op[event["args"]["op_name"]] += event["dur"]
            nodes.add(event["name"])
        if event.get("cat") == "Session" and event["name"] == "model_run":
            runs.append(event["dur"])
    return {op: t / CALLS for op, t in by_op.items()}, len(nodes), float(np.median(runs[200:]))


for name in sys.argv[1:]:
    ops, executed, run = profile(HERE / f"{name}.onnx")
    kernels = sum(ops.values())
    print(f"\n{name}: {executed} executed nodes after ORT fusion; model_run median {run:.1f} us; kernels {kernels:.1f} us "
          f"({100 * kernels / run:.0f}%); the rest is per-run overhead")
    for op, t in sorted(ops.items(), key=lambda kv: -kv[1]):
        print(f"  {op:22s} {t:6.2f} us  {100 * t / kernels:4.1f}%")
