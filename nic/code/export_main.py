"""Export the two main models for the team (chip-safe, checked):
  gadst.onnx                  context [1,400,4] -> prediction [1,20,4]                          (stateless)
  gru_stateful_window.onnx    context [1,400,4] (uses only the newest sample), state_in_0 [1,32], state_in_1 [1,4]
                              -> prediction [1,20,4], state_out_0 [1,32], state_out_1 [1,4]  (stateful, harness feeds 400)
  gru_stateful_step.onnx      same, but context [1,1,4] (if the harness feeds exactly one sample)
  ../abc_onnx/bin/python export_main.py --gads RUN/model.pt --stateful RUN/model.pt --out DIR"""
import argparse
import json
import os

import numpy as np
import torch

import onnx_check as oc
import stateful
import train


class WindowStep(torch.nn.Module):
    def __init__(self, st):
        super().__init__()
        self.st = st

    def forward(self, ctx, h, prev):
        return self.st(ctx[:, -1:], h, prev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gads", required=True)
    ap.add_argument("--stateful", required=True)
    ap.add_argument("--out", required=True)
    g = ap.parse_args()
    os.makedirs(g.out, exist_ok=True)
    rng = np.random.default_rng(0)
    x = train.qexp(torch.tensor(np.cumsum(rng.normal(0, 0.02, (400, 3)), 0) * 0.3)).float()[None]
    rows = []
    m, a = oc.load(g.gads)
    p = oc.Pipeline(m, a.frame).eval()
    with torch.no_grad():
        ref = p(x).numpy()
    f = os.path.join(g.out, "gadst.onnx")
    torch.onnx.export(p, (x,), f, input_names=["context"], output_names=["prediction"], opset_version=17, dynamo=False)
    rows.append(oc.check("gadst", f, {"context": x.numpy()}, ref / np.linalg.norm(ref, axis=-1, keepdims=True)))
    gm, st = stateful.check(g.stateful)
    H = gm.gru.hidden_size
    h, prev = torch.zeros(1, H), torch.zeros(1, 4)
    with torch.no_grad():
        for t in range(399):
            _, h, prev = st(x[:, t:t + 1], h, prev)
        ref = st(x[:, 399:400], h, prev)[0].numpy()
    ref = ref / np.linalg.norm(ref, axis=-1, keepdims=True)
    io = dict(input_names=["context", "state_in_0", "state_in_1"], output_names=["prediction", "state_out_0", "state_out_1"], opset_version=17, dynamo=False)
    f = os.path.join(g.out, "gru_stateful_step.onnx")
    torch.onnx.export(st, (x[:, 399:400], h, prev), f, **io)
    rows.append(oc.check("stateful_step", f, {"context": x[:, 399:400].numpy(), "state_in_0": h.numpy(), "state_in_1": prev.numpy()}, ref))
    f = os.path.join(g.out, "gru_stateful_window.onnx")
    torch.onnx.export(WindowStep(st).eval(), (x, h, prev), f, **io)
    rows.append(oc.check("stateful_window", f, {"context": x.numpy(), "state_in_0": h.numpy(), "state_in_1": prev.numpy()}, ref))
    json.dump(rows, open(os.path.join(g.out, "export_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
