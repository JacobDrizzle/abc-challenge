"""v7 vs lean graphs at ORT_ENABLE_ALL (adds NCHWc layout for 4-D Conv) and ORT_ENABLE_EXTENDED (no layout pass),
1 thread, alternating rounds. Contended CPU (training on the GPU): relative numbers only."""
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, r"<repo>\jacob\training")
from benchmark_latency import Streamer, time_stream  # noqa: E402

HERE = Path(__file__).resolve().parent / "trackA_latency"
LEVELS = {"ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL, "EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED}


def session(path, level):
    options = ort.SessionOptions()
    options.graph_optimization_level = LEVELS[level]
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads, options.inter_op_num_threads = 1, 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


rng = np.random.default_rng(0)
stream = np.cumsum(rng.normal(0, .01, (4000, 4)), 0) + [1, 0, 0, 0]
stream = (stream / np.linalg.norm(stream, axis=1, keepdims=True)).astype(np.float32)
calls = {}
for name in sys.argv[1:]:
    for level in LEVELS:
        s = session(HERE / f"{name}.onnx", level)
        shapes = [i.shape for i in s.get_inputs()[1:]]
        calls[(name, level)] = Streamer(s, shapes)
times = {k: [] for k in calls}
for _ in range(5):
    for k, call in calls.items():
        call.reset()
        times[k] += time_stream(call, stream, 400, 300, 2000)
for (name, level), t in times.items():
    print(f"{name:12s} {level:9s} median {np.median(t) * 1e6:6.1f} us  p90 {np.percentile(t, 90) * 1e6:6.1f}")
