"""Stateful (streaming) version of a GRU trained on velocity-only input (features 'vel+noq').
The challenge harness can feed ONE new sample per step and carry state between steps (state_in_k / state_out_k).
Because velocity only depends on the previous and the new sample, the GRU can update its memory once per sample.

  python stateful.py check  RUN/model.pt                 # step-by-step == window mode? (CPU, quick)
  python stateful.py export RUN/model.pt OUT.onnx        # ONNX graph: context[1,1,4], state_in_0 h[1,H], state_in_1 prev[1,4]
  python stateful.py stream --data D --test-players-file F --out res.json NAME=RUN/model.pt ...
      # error when the memory has run from the start of the recording (what the harness does) vs fresh 4 s windows

The GRU cell is written out with MatMul/Sigmoid/Tanh/Mul/Add (all on the safe-operator list) instead of the GRU operator."""
import argparse
import json
import sys

import numpy as np
import torch
import torch.nn as nn

import train


def load(path):
    ck = torch.load(path, map_location="cpu")
    a = train.parse([])
    for k, v in ck["config"].items():
        setattr(a, k, v)
    assert a.model == "gru" and not train.uses_quat(a.features) and train.feature_groups(a.features) == ("vel",), \
        "stateful mode needs a GRU trained on 'vel+noq'"
    m = train.build_model(a)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), a


class Step(nn.Module):
    """One streaming step. Inputs: the new sample q [1,1,4] (w,x,y,z), state h [1,H], previous sample prev [1,4].
    Outputs: prediction [1,20,4], new h, new prev (= q). With prev = 0 (the harness's zero start) the velocity is 0,
    exactly like the first reading of a training window."""

    def __init__(self, model):
        super().__init__()
        g = model.gru
        H = g.hidden_size
        self.H = H
        self.w_ih, self.w_hh = g.weight_ih_l0.detach().T.contiguous(), g.weight_hh_l0.detach().T.contiguous()   # [I,3H], [H,3H]
        self.b_ih, self.b_hh = g.bias_ih_l0.detach(), g.bias_hh_l0.detach()
        self.head = model.head

    def forward(self, q, h, prev):
        q = q.reshape(1, 4)                                               # [1,1,4] -> [1,4]
        d = train.qmul(train.qconj(prev), q)                              # turn since the previous sample
        v = 2.0 * d[:, 1:] * train.HZ                                     # small-angle velocity, rad/s (no Atan)
        gi = v @ self.w_ih + self.b_ih
        gh = h @ self.w_hh + self.b_hh
        H = self.H
        r = torch.sigmoid(gi[:, :H] + gh[:, :H])
        z = torch.sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
        n = torch.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:])
        h_new = n + z * h + (z * n) * -1.0                               # = (1-z)*n + z*h, written with Add/Mul only
        raw = self.head(h_new).view(1, train.HORIZON, 4)
        return train.qmul(q.reshape(1, 1, 4), raw), h_new, q                       # absolute prediction (not normalised: harness does it)


def check(path):
    m, a = load(path)
    st = Step(m)
    rng = np.random.default_rng(0)
    rv = np.cumsum(rng.normal(0, 0.02, (420, 3)), 0) * 0.3
    q = train.qexp(torch.tensor(rv)).float()                             # a wiggly test trajectory
    ctx = q[None, :400]
    rel, _, last = train.prepare(ctx, q[None, 400:], a.frame)
    with torch.no_grad():
        _, unit = m(rel)
        ref = train.from_rel(unit, last, a.frame)
        h, prev = torch.zeros(1, m.gru.hidden_size), torch.zeros(1, 4)
        for t in range(400):
            out, h, prev = st(ctx[:, t:t + 1], h, prev)
        out = out / out.norm(dim=-1, keepdim=True)
    diff = np.degrees(train.angle(out, ref).max().item())
    print(f"max difference step-by-step vs window mode: {diff:.5f} deg")
    assert diff < 0.01
    return m, st


def export(path, out):
    m, st = check(path)
    H = m.gru.hidden_size
    torch.onnx.export(st, (torch.randn(1, 1, 4), torch.zeros(1, H), torch.zeros(1, 4)), out,
                      input_names=["context", "state_in_0", "state_in_1"], output_names=["prediction", "state_out_0", "state_out_1"],
                      opset_version=17, dynamo=False)
    print("exported", out)


@torch.no_grad()
def stream(g):
    test = [l.strip() for l in open(g.test_players_file) if l.strip()]
    users = train.load_users(g.data, None)
    w = train.Windows(users, test, g.stride)
    res = {}
    for spec in g.models:
        name, path = spec.split("=", 1)
        m, a = load(path)
        m = m.to(train.DEV)
        e_win, e_str, e_hold = [], [], []
        by_seg = {}
        for s, st in w.rows:
            by_seg.setdefault(int(s), []).append(int(st))
        for s, starts in by_seg.items():
            seg = torch.from_numpy(w.segs[s]).to(train.DEV)[None]          # [1,N,4] whole recording
            v = train.featurize(seg, a.features)                           # velocity of every sample (invariant to the frame)
            hs, _ = m.gru(v)                                               # memory after every sample, run from the start
            ends = torch.tensor(starts, device=train.DEV) + train.CONTEXT - 1
            raw = m.head(hs[0, ends]).view(-1, train.HORIZON, 4)
            last = seg[0, ends][:, None]
            tgt = torch.stack([seg[0, e + 1:e + 1 + train.HORIZON] for e in ends.tolist()])
            pred = train.from_rel(nn.functional.normalize(raw, dim=-1), last, a.frame)
            e_str.append(train.angle(pred[:, -1], tgt[:, -1]))
            ctx = torch.stack([seg[0, e - train.CONTEXT + 1:e + 1] for e in ends.tolist()])
            rel, _, lst = train.prepare(ctx, tgt, a.frame)
            _, unit = m(rel)
            e_win.append(train.angle(train.from_rel(unit, lst, a.frame)[:, -1], tgt[:, -1]))
            e_hold.append(train.angle(ctx[:, -1], tgt[:, -1]))
        r = {k: float(torch.rad2deg(torch.cat(v)).mean()) for k, v in (("window_4s", e_win), ("streaming_from_start", e_str), ("hold", e_hold))}
        res[name] = r
        print(name, r, flush=True)
    json.dump(dict(windows=len(w.rows), results=res), open(g.out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] in ("check", "export"):
        train.FEAT_DTYPE[0] = torch.float32
        train.QMUL_MATRIX[0] = True
        (check if sys.argv[1] == "check" else export)(*sys.argv[2:])
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--data", nargs="+", required=True)
        ap.add_argument("--test-players-file", required=True)
        ap.add_argument("--stride", type=int, default=10)
        ap.add_argument("--out", required=True)
        ap.add_argument("models", nargs="+")
        stream(ap.parse_args())
