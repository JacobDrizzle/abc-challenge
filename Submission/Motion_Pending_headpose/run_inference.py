"""Motion_Pending head-pose predictor (9th ABC Challenge).

    model = load_model(model_dir)
    prediction = predict(model, context)   # context: float32 [400, 4] (w, x, y, z) -> float32 [20, 4], same order

or from the command line:

    python run_inference.py --model_dir . --input sample.npy --output pred.npy

model_fp32.onnx is a streaming (stateful) graph: each call takes the full [1, 400, 4] context plus three state tensors
and returns the [1, 20, 4] prediction plus the updated states. predict() gets no state from its caller, so the
model object keeps it:
- If the context is the previous context shifted on by one sample (the next window of the same stream), the graph
  runs once, with the carried state.
- Otherwise (the first call, a new session, or a jump) the state is rebuilt from this context alone: zero state, then
  one call per sample of the context, each with the context left-padded by its first sample, exactly as for a stream
  that starts at this window's first sample. That costs 400 graph calls instead of one.
Either way the output only depends on the context passed in and the earlier contexts of the same stream, never on
anything later. Runs on the CPU (onnxruntime), with no network access.
"""
import argparse
import os
import re

import numpy as np
import onnxruntime as ort

CONTEXT, HORIZON = 400, 20
STATE_IN = re.compile(r"^state_in_(\d+)$")


class StreamingModel:
    def __init__(self, path, threads=1):
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        self.session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
        inputs = {i.name: i for i in self.session.get_inputs()}
        k = sorted(int(m.group(1)) for n in inputs if (m := STATE_IN.match(n)))
        self.states = [(f"state_in_{i}", f"state_out_{i}", tuple(inputs[f"state_in_{i}"].shape)) for i in k]
        self.outputs = ["prediction"] + [o for _, o, _ in self.states]
        self.reset()

    def reset(self):
        self.state = {i: np.zeros(shape, np.float32) for i, _, shape in self.states}
        self.previous = None

    def step(self, window):
        """One graph call on a [400, 4] window, carrying the state. Returns the [20, 4] prediction."""
        out = self.session.run(self.outputs, {"context": window[None], **self.state})
        for (i, _, _), value in zip(self.states, out[1:]):
            self.state[i] = value
        return out[0][0]


def load_model(model_dir):
    return StreamingModel(os.path.join(model_dir, "model_fp32.onnx"))


def predict(model, context):
    q = np.ascontiguousarray(context, dtype=np.float32).reshape(CONTEXT, 4)
    if model.previous is not None and np.array_equal(q[:-1], model.previous[1:]):
        prediction = model.step(q)                       # next window of the same stream: one new sample
    else:
        model.reset()                                    # rebuild the state from this context alone
        for t in range(CONTEXT - 1):
            model.step(np.concatenate((np.repeat(q[:1], CONTEXT - 1 - t, axis=0), q[:t + 1])))
        prediction = model.step(q)
    model.previous = q
    return prediction.astype(np.float32)                 # [20, 4], (w, x, y, z); not re-normalised


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model_dir", default=".")
    parser.add_argument("--input", required=True, help=".npy with a [400, 4] or [1, 400, 4] float32 context (w, x, y, z)")
    parser.add_argument("--output", required=True, help=".npy to write the [20, 4] prediction to")
    args = parser.parse_args()
    context = np.load(args.input)
    np.save(args.output, predict(load_model(args.model_dir), context))


if __name__ == "__main__":
    main()
