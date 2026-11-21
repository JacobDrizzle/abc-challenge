"""Export models the way the challenge wants them and check them against the published operator rules.
For each model: export (float32, small-angle velocity = no Atan, no output normalisation), simplify with onnxsim (folds
shape bookkeeping into constants), list operators outside the safe list, count MACs, check ONNX == PyTorch, time on CPU.
  ../abc_onnx/bin/python onnx_check.py --out DIR NAME=run/model.pt [NAME=run/model.pt ...] [--stateful NAME=run/model.pt]
MACs per streaming step: stateless = one full 400-sample pass; stateful = one new sample. Only MatMul/Gemm/Conv/GRU are
counted (element-wise ops are small); this is the convention most profilers use."""
import argparse
import collections
import json
import os
import statistics
import time

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import torch
from onnx import shape_inference

import train

train.FEAT_DTYPE[0] = torch.float32
train.SMALL_ANGLE[0] = True
train.QMUL_MATRIX[0] = True    # quaternion products as MatMul for the exported graph
train.EXPORT_SAFE[0] = True    # GADS-T max-pool and attention with safe operators only

SAFE = {"Conv", "MatMul", "Gemm", "LayerNormalization", "BatchNormalization", "Relu", "Clip", "HardSwish", "Gelu",
        "Sigmoid", "Tanh", "Add", "Mul", "Concat", "Slice", "Reshape", "Transpose", "GlobalAveragePool"}
ATTENTION_OK = {"Softmax"}            # allowed inside plain MatMul -> Softmax -> MatMul attention
AVOID = {"Loop", "If", "Scan", "NonZero", "TopK", "Einsum", "SequenceConstruct"}
HARMLESS = {"Constant", "Identity"}    # no computation


class Pipeline(torch.nn.Module):
    def __init__(self, model, frame):
        super().__init__()
        self.model, self.frame = model, frame
        self.model.export_raw = True

    def forward(self, ctx):
        last = ctx[:, -1:]
        rel = train.to_rel(ctx, last, self.frame)
        return train.from_rel(self.model(rel), last, self.frame)


def load(path):
    ck = torch.load(path, map_location="cpu")
    a = train.parse([])
    for k, v in ck["config"].items():
        setattr(a, k, v)
    train.CTX_LEN[0] = a.context
    m = train.build_model(a)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), a


def macs(model):
    g = shape_inference.infer_shapes(model).graph
    shp = {}
    for v in list(g.value_info) + list(g.input) + list(g.output):
        shp[v.name] = [d.dim_value for d in v.type.tensor_type.shape.dim]
    for t in g.initializer:
        shp[t.name] = list(t.dims)
    total, by = 0, collections.Counter()
    for n in g.node:
        out = shp.get(n.output[0])
        if n.op_type == "MatMul" and out:
            k = shp[n.input[0]][-1]
            c = int(np.prod(out)) * k
        elif n.op_type == "Gemm" and out:
            a = shp[n.input[0]]
            transA = any(x.name == "transA" and x.i for x in n.attribute)
            c = int(np.prod(out)) * (a[0] if transA else a[-1])
        elif n.op_type == "Conv" and out:
            w = shp[n.input[1]]                                           # [Cout, Cin/groups, k]
            c = int(np.prod(out)) * int(np.prod(w[1:]))
        elif n.op_type == "GRU":
            x, w, r = shp[n.input[0]], shp[n.input[1]], shp[n.input[2]]   # x [T,B,I], W [1,3H,I], R [1,3H,H]
            c = x[0] * x[1] * (w[1] * w[2] + r[1] * r[2])
        else:
            continue
        total += c
        by[n.op_type] += c
    return total, dict(by)


def bench(sess, feeds, threads, n=300):
    for _ in range(30):
        sess.run(None, feeds)
    ts = []
    for _ in range(n):
        t = time.perf_counter(); sess.run(None, feeds); ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def check(name, f, feeds, ref):
    m = onnx.load(f)
    m, ok = onnxsim.simplify(m)
    assert ok, "onnxsim check failed"
    onnx.save(m, f)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    off = {k: v for k, v in ops.items() if k not in SAFE | ATTENTION_OK | HARMLESS}
    avoid = [k for k in ops if k in AVOID]
    dbl = [t.name for t in m.graph.initializer if t.data_type == 11]
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(f, so, providers=["CPUExecutionProvider"])
    got = sess.run(None, feeds)[0]
    got = got / np.linalg.norm(got, axis=-1, keepdims=True)
    deg = float(np.degrees(np.arccos(np.clip(np.abs((got * ref).sum(-1)), -1, 1))).max())
    total, by = macs(m)
    ms = bench(sess, feeds, 1)
    row = dict(name=name, macs=total, macs_by_op=by, ms_1thread=ms, max_deg_vs_torch=deg, n_ops=sum(ops.values()),
               off_list=off, avoid=avoid, float64=len(dbl), kb=os.path.getsize(f) / 1024)
    print(f"{name:14s} MACs {total:>10,d}  {ms:6.3f} ms  ONNX vs torch {deg:.4f} deg  ops {row['n_ops']:4d}  {row['kb']:.0f} KB"
          f"  off-list {off or 'none'}  avoid {avoid or 'none'}  float64 {len(dbl)}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stateful", nargs="*", default=[])
    ap.add_argument("models", nargs="*")
    g = ap.parse_args()
    os.makedirs(g.out, exist_ok=True)
    rng = np.random.default_rng(0)
    rv = np.cumsum(rng.normal(0, 0.02, (400, 3)), 0) * 0.3
    x = train.qexp(torch.tensor(rv)).float()[None]                        # a wiggly, realistic-looking window
    rows = []
    for spec in g.models:
        name, path = spec.split("=", 1)
        m, a = load(path)
        p = Pipeline(m, a.frame).eval()
        with torch.no_grad():
            ref = p(x).numpy()
        ref = ref / np.linalg.norm(ref, axis=-1, keepdims=True)
        f = os.path.join(g.out, f"{name}.onnx")
        torch.onnx.export(p, (x,), f, input_names=["context"], output_names=["prediction"], opset_version=17, dynamo=False)
        rows.append(check(name, f, {"context": x.numpy()}, ref))
    for spec in g.stateful:
        import stateful
        name, path = spec.split("=", 1)
        m, st = stateful.check(path)
        H = m.gru.hidden_size
        h, prev = torch.zeros(1, H), torch.zeros(1, 4)
        with torch.no_grad():
            for t in range(399):
                _, h, prev = st(x[:, t:t + 1], h, prev)
            ref = st(x[:, 399:400], h, prev)[0].numpy()
        ref = ref / np.linalg.norm(ref, axis=-1, keepdims=True)
        f = os.path.join(g.out, f"{name}.onnx")
        torch.onnx.export(st, (x[:, 399:400], h, prev), f, input_names=["context", "state_in_0", "state_in_1"],
                          output_names=["prediction", "state_out_0", "state_out_1"], opset_version=17, dynamo=False)
        rows.append(check(name + " (1 step)", f, {"context": x[:, 399:400].numpy(), "state_in_0": h.numpy(), "state_in_1": prev.numpy()}, ref))
    json.dump(rows, open(os.path.join(g.out, "onnx_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
