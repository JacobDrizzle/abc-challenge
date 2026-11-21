"""Run both main models on one recording and print the mean error at step 20 (0.2 s ahead).

  python run_example.py recording.npy      # float array [N, 4]: quaternions (w, x, y, z) at 100 Hz, N > 420

Needs numpy and onnxruntime. GADS-T sees each 400-sample window on its own. The stateful GRU is streamed: one call per
sample, its state fed back each time, starting from zeros at the beginning of the recording (as the challenge harness does).
"""
import os
import sys

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
CONTEXT, HORIZON = 400, 20


def angle_deg(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))


def main(path):
    q = np.load(path).astype(np.float32)
    gads = ort.InferenceSession(os.path.join(HERE, "gadst.onnx"), providers=["CPUExecutionProvider"])
    gru = ort.InferenceSession(os.path.join(HERE, "gru_stateful_window.onnx"), providers=["CPUExecutionProvider"])
    h, prev = np.zeros((1, 32), np.float32), np.zeros((1, 4), np.float32)
    err_gru, err_gads, err_hold = [], [], []
    for t in range(len(q) - HORIZON):                     # stream every sample through the stateful GRU
        ctx = np.zeros((1, CONTEXT, 4), np.float32)
        ctx[0, -1] = q[t]                                 # only the newest sample is used by this model
        pred, h, prev = gru.run(None, {"context": ctx, "state_in_0": h, "state_in_1": prev})
        if t >= CONTEXT - 1:                              # a full 4 s window ends at t
            truth = q[t + HORIZON]
            err_gru.append(angle_deg(pred[0, -1], truth))
            window = q[None, t - CONTEXT + 1:t + 1]
            err_gads.append(angle_deg(gads.run(None, {"context": window})[0][0, -1], truth))
            err_hold.append(angle_deg(q[t], truth))
    print(f"{len(err_gru)} windows | step-20 error: stateful GRU {np.mean(err_gru):.3f} deg, "
          f"GADS-T {np.mean(err_gads):.3f} deg, hold-last {np.mean(err_hold):.3f} deg")


if __name__ == "__main__":
    main(sys.argv[1])
