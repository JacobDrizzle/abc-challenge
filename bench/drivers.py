"""Run an ONNX graph (or a submission folder) through the challenge interface.

Stateless graph: one input [1, 400, 4] -> one output [1, 20, 4]; any tensor names. Re-run on every
shifted window.
Stateful graph: additionally state_in_k inputs and state_out_k outputs (k = 0, 1, ...), static float32
shapes. The harness zero-initialises the states and calls the graph once per new sample, feeding each
state_out_k back into state_in_k (matched by name, not position). Ported and generalised from
ABC_local_test/benchmark_latency.py (Streamer) and experimental/team_comparison/compare_nic.py
(ONNXPredictor); unlike those, no input name is assumed.

Before a stateful session has 400 samples, the context is left-padded with the first sample (the
convention behind Jacob's stream-parity checks). What the real harness feeds then is still an open
question to the organisers (nic/FINDINGS.md O8).
"""
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

STATE_IN = re.compile(r"^state_in_(\d+)$")
STATE_OUT = re.compile(r"^state_out_(\d+)$")


def ort_session(path, threads=1):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    source = path if isinstance(path, bytes) else str(path)
    return ort.InferenceSession(source, options, providers=["CPUExecutionProvider"])


def interface(session):
    """Find the context input, the prediction output and the state pairs of a graph."""
    inputs = {i.name: i for i in session.get_inputs()}
    outputs = {o.name: o for o in session.get_outputs()}
    states_in = {int(m.group(1)): n for n in inputs if (m := STATE_IN.match(n))}
    states_out = {int(m.group(1)): n for n in outputs if (m := STATE_OUT.match(n))}
    context = [n for n in inputs if not STATE_IN.match(n)]
    prediction = [n for n in outputs if not STATE_OUT.match(n)]
    if len(context) != 1 or len(prediction) != 1:
        raise ValueError(f"Expected one context input and one prediction output besides state tensors; "
                         f"got inputs {list(inputs)}, outputs {list(outputs)}")
    if set(states_in) != set(states_out) or sorted(states_in) != list(range(len(states_in))):
        raise ValueError(f"state_in_k / state_out_k must pair up as k = 0..K-1; got {sorted(states_in)} / {sorted(states_out)}")
    shape = inputs[context[0]].shape
    if len(shape) != 3 or shape[-1] != 4 or any(isinstance(d, int) and d != want for d, want in zip(shape, (1, 400))):
        raise ValueError(f"Context input {context[0]} must be [1, 400, 4]; got {shape}")
    states = []
    for k in sorted(states_in):
        i, o = inputs[states_in[k]], outputs[states_out[k]]
        if not all(isinstance(d, int) and d > 0 for d in i.shape):
            raise ValueError(f"{i.name} needs a static shape for zero initialisation; got {i.shape}")
        if i.type != "tensor(float)":
            raise ValueError(f"{i.name} must be float32; got {i.type}")
        states.append((i.name, o.name, tuple(i.shape)))
    return context[0], prediction[0], states


class OnnxModel:
    """One ONNX graph behind the predictor interface used by bench.scoring."""

    def __init__(self, path, name=None, threads=1):
        self.path = Path(path)
        self.name = name or self.path.stem
        self.threads = threads
        self.session = ort_session(self.path, threads)
        self.context_name, self.prediction_name, self.states = interface(self.session)
        self.stateful = bool(self.states)
        self.reset()

    def reset(self):
        self.state = {i: np.zeros(shape, np.float32) for i, _, shape in self.states}

    def step(self, window):
        """One harness call on a [1, 400, 4] float32 window; carries state. Returns the raw [1, 20, 4] output."""
        names = [self.prediction_name, *(o for _, o, _ in self.states)]
        outputs = self.session.run(names, {self.context_name: window, **self.state})
        for (i, _, _), value in zip(self.states, outputs[1:]):
            self.state[i] = value
        return outputs[0]

    def predict_segment(self, q32, context, horizon, protocol="stateless", block=1024):
        if protocol not in ("stateless", "from-start", "cold-start"):
            raise ValueError(f"Unknown protocol {protocol!r}")
        # Calls start at sample 0 only from segment start; scoring always starts at the first full window.
        calls = range(0 if protocol == "from-start" else context - 1, len(q32) - horizon)
        self.reset()
        out = []
        for t in calls:
            if t >= context - 1:
                window = q32[t - context + 1:t + 1][None]
            else:
                window = np.concatenate((np.repeat(q32[:1], context - 1 - t, 0), q32[:t + 1]))[None]
            prediction = self.step(window)
            if t >= context - 1:
                if prediction.shape != (1, horizon, 4):
                    raise ValueError(f"{self.name}: output shape {prediction.shape}, expected (1, {horizon}, 4)")
                out.append(prediction[0])
                if len(out) == block:
                    yield np.stack(out)
                    out = []
        if out:
            yield np.stack(out)

    def stream(self, q32, context=400, horizon=20):
        """Raw outputs for every full window of q32: from segment start if stateful, else window by window."""
        protocol = "from-start" if self.stateful else "stateless"
        return np.concatenate(list(self.predict_segment(q32, context, horizon, protocol)))


class Submission:
    """A submission folder: run_inference.py with load_model(model_dir) and predict(model, context[400,4]) -> [20,4]."""
    stateful = False

    def __init__(self, folder, name=None):
        self.folder = Path(folder).resolve()
        self.name = name or self.folder.name
        script = self.folder / "run_inference.py"
        if not script.is_file():
            raise ValueError(f"{script} not found")
        spec = importlib.util.spec_from_file_location(f"bench_submission_{abs(hash(str(self.folder)))}", script)
        self.module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(self.folder))
        try:
            spec.loader.exec_module(self.module)
        finally:
            sys.path.remove(str(self.folder))
        for fn in ("load_model", "predict"):
            if not callable(getattr(self.module, fn, None)):
                raise ValueError(f"{script} must define {fn}()")
        self.reset()

    def reset(self):
        """A new session: load_model again, so a model that keeps state internally starts clean."""
        self.model = self.module.load_model(str(self.folder))

    def predict_segment(self, q32, context, horizon, protocol="stateless", block=1024):
        """Calls predict() on every shifted window from the segment's first full window, in stream order."""
        self.reset()
        out = []
        for t in range(context - 1, len(q32) - horizon):
            prediction = np.asarray(self.module.predict(self.model, np.ascontiguousarray(q32[t - context + 1:t + 1])))
            if prediction.shape != (horizon, 4):
                raise ValueError(f"{self.name}: predict() returned {prediction.shape}, expected ({horizon}, 4)")
            out.append(prediction)
            if len(out) == block:
                yield np.stack(out)
                out = []
        if out:
            yield np.stack(out)
