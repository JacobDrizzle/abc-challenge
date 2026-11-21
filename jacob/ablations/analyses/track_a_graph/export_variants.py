"""Prototype block layouts for the lean graph (same seed-44 weights), timed against the current lean.onnx in
alternating rounds. The screen is training, so the CPU is contended: relative numbers only.

- dup: per block, one broadcast MatMul [[1,0],[0,1],[1,0]] @ h gives rows [h0, h1, h0] (now: Slice + Concat).
- bab: per block, a (3, 1) pointwise Conv with kernel rows [B, A, B] over [0, h0, h1, 0] (pads 1 above and below)
  gives A h0 + B h1 and B h0 + A h1 directly. Timing-only here: B is stored twice. A file could store [A | B]
  once and build [B, A, B] with nodes ORT folds at load (a team call).
- bab + A4: bab with the fused head.
Each graph's stream parity is checked against the PyTorch model on the longest validation session.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from torch.nn import functional as F

TRAINING = Path(r"<repo>\jacob\training")
sys.path.insert(0, str(TRAINING))

from baseline_pipeline import DEFAULT_H5, manifest_for  # noqa: E402
from benchmark_latency import Streamer, ort_session, stream_parity, summarise, time_stream, validation_segments  # noqa: E402
from tcnn.onnx_export import OPSET, audit_operators, count_macs  # noqa: E402
from experimental.v9_lean.export import StreamingLean  # noqa: E402
from experimental.v9_lean.model import deployed, load_checkpoint  # noqa: E402

OUT = Path(__file__).resolve().parent / "export_variants"
SEP44 = TRAINING / "outputs/v8_recipe_sep_base3_addons/angular_loss/seed_44/best.pt"


class Variant(StreamingLean):
    def __init__(self, model, layout, fused_head=False):
        super().__init__(model, fused_head)
        self.layout = layout
        self.register_buffer("dup", torch.tensor([[1., 0.], [0., 1.], [1., 0.]]), persistent=False)
        if layout == "bab":
            for w in self.pointwise:  # [K, K, 2, 1] rows A, B -> [K, K, 3, 1] rows B, A, B
                w.data = torch.cat((w.data[:, :, 1:2], w.data[:, :, 0:1], w.data[:, :, 1:2]), 2)

    def forward(self, context, *states):
        m, p = self.model, self.model.head_basis
        previous, now = context[:, -2:-1], context[:, -1:]
        from tcnn.onnx_export import _left_matrix
        delta = torch.matmul(now, _left_matrix(self.table_left_conj, previous))
        velocity = delta[..., 0:1] * delta[..., 1:4]
        if self.maps == "lean":
            velocity = velocity * torch.matmul(delta * delta, self.rate_weights)
        speed = torch.matmul(F.relu(torch.matmul(velocity, self.plus_minus)), self.ones6)
        parts = [velocity, speed]
        if m.gravity_features:
            outer = torch.matmul(now.reshape(1, 4, 1), now).reshape(1, 1, 16)
            parts.append(torch.matmul(outer, self.gravity_table))
        x = torch.cat(parts, -1).reshape(1, -1, 1, 1) * self.input_signs
        x = F.relu(F.conv2d(x, self.projection, m.projection.bias))
        new_states = []
        for block, state, depthwise, pointwise in zip(m.blocks, states, self.depthwise, self.pointwise):
            window = torch.cat((state, x), -1)
            new_states.append(window[..., 1:])
            h = F.conv2d(window, depthwise, dilation=(1, block.dilation), groups=m.stream_channels)
            if self.layout == "dup":
                y = F.conv2d(torch.matmul(self.dup, h), pointwise, block.pointwise.bias)
            elif self.layout == "bab":
                y = F.conv2d(h, pointwise, block.pointwise.bias, padding=(1, 0))
            else:
                y = F.conv2d(torch.cat((h, h[:, :, 0:1]), 2), pointwise, block.pointwise.bias)
            x = F.relu(x + F.relu(y))
        if self.fused_head:
            coefficients = F.conv2d(x, self.head_weight, self.head_bias).reshape(1, p, 3)
            rows = (coefficients, velocity)
        else:
            both = torch.matmul(m.head.weight, x.reshape(m.stream_channels, 2))
            coefficients = torch.matmul(both * self.head_mask, self.ones2).reshape(1, p, 3)
            rows = (coefficients, velocity, self.coefficient_bias)
        r = torch.matmul(self.basis, torch.cat(rows, 1))
        scalar = self.twos if self.maps == "native" else torch.matmul(r * r, self.minus_sixth) + 2.
        both = torch.matmul(torch.cat((scalar, r), -1), torch.matmul(now, self.table_left_both).reshape(1, 4, 8))
        prediction = F.layer_norm(both, (8,), self.halves, eps=1e-12)[..., 0:4]
        return (prediction, *new_states)


def export(wrapper, path):
    example = (F.normalize(torch.randn(1, 400, 4), dim=-1), *(torch.zeros(s) for s in wrapper.state_shapes))
    count = len(wrapper.state_shapes)
    with torch.no_grad():
        torch.onnx.export(wrapper.eval(), example, str(path), opset_version=OPSET, do_constant_folding=True, dynamo=False,
                          input_names=["context", *(f"state_in_{i}" for i in range(count))],
                          output_names=["prediction", *(f"state_out_{i}" for i in range(count))])
    onnx.checker.check_model(onnx.load(str(path)))
    return path, wrapper.state_shapes


def main():
    OUT.mkdir(exist_ok=True)
    segments = validation_segments(DEFAULT_H5, manifest_for(DEFAULT_H5), 420)
    stream = max(segments, key=len)
    lean = deployed(load_checkpoint(SEP44)[0])
    graphs = {"lean (now)": export(Variant(lean, "now"), OUT / "now.onnx"),
              "dup": export(Variant(lean, "dup"), OUT / "dup.onnx"),
              "bab": export(Variant(lean, "bab"), OUT / "bab.onnx"),
              "lean + A4": export(Variant(lean, "now", True), OUT / "now_fused.onnx"),
              "bab + A4": export(Variant(lean, "bab", True), OUT / "bab_fused.onnx")}
    report = {}
    for name, (path, shapes) in graphs.items():
        graph = onnx.load(str(path)).graph
        check = stream_parity(lean, Streamer(ort_session(path, 1), shapes), segments, 400, 20, 1000)
        report[name] = {"nodes": sum(n.op_type != "Constant" for n in graph.node), "macs": count_macs(path)["total"],
                        "stored": int(sum(np.prod(t.dims) for t in graph.initializer)),
                        "other_ops": audit_operators(path)["other"],
                        "parity_max_deg": check["max_difference_deg"]}
    calls = {name: Streamer(ort_session(path, 1), shapes) for name, (path, shapes) in graphs.items()}
    timings = {name: [] for name in calls}
    for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 5):
        for name, call in calls.items():
            call.reset()
            timings[name] += time_stream(call, stream, 400, 300, 2000)
    base = np.median(timings["lean (now)"])
    for name, r in report.items():
        lat = summarise(timings[name])
        r["median_us"], r["p90_us"] = lat["median_ms"] * 1e3, lat["p90_ms"] * 1e3
        print(f"{name:12s} nodes {r['nodes']:3d} stored {r['stored']:5d} MACs {r['macs']:6d} other {r['other_ops']} "
              f"parity {r['parity_max_deg']:.1e} deg | median {r['median_us']:5.1f} us ({100 * (np.median(timings[name]) / base - 1):+.1f}%)"
              f" p90 {r['p90_us']:5.1f}")
    (OUT / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
