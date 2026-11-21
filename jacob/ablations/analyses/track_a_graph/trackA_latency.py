"""Track A + L0 latency: the v7 graph vs v9 lean graphs (same seed-44 weights), and random-weight v9 graphs of the
B-screen layouts, timed like benchmark_latency (ORT CPU, 1 thread, one sample per call) in alternating rounds.

Also stream parity on the longest validation session (1,000 full windows): each graph vs the PyTorch model it
reproduces, and the lean graph vs the exact model. Read-only on the repo; writes into trackA_latency/ here.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import torch

TRAINING = Path(r"<repo>\jacob\training")
sys.path.insert(0, str(TRAINING))

from baseline_pipeline import DEFAULT_H5, manifest_for  # noqa: E402
from benchmark_latency import Streamer, ort_session, stream_parity, summarise, time_stream, validation_segments  # noqa: E402
from tcnn.onnx_export import audit_operators, count_macs  # noqa: E402
from experimental.v7_compact.export import export_streaming_onnx as export_v7  # noqa: E402
from experimental.v7_compact.model import load_checkpoint as load_v7  # noqa: E402
from experimental.v9_lean.export import export_streaming_onnx  # noqa: E402
from experimental.v9_lean.model import LeanTCNN, deployed, load_checkpoint  # noqa: E402

OUT = Path(__file__).resolve().parent / "trackA_latency"
SEP44 = TRAINING / "outputs/v8_recipe_sep_base3_addons/angular_loss/seed_44/best.pt"
BEST = dict(input_features="body_velocity", num_blocks=5, effective_context=256, damped_prior_tau_ms=100,
            gravity_features=True, mirror_equivariant=True, dilation_base=3, separable=True, head_basis=3,
            speed_feature="l1", channels=48, dropout=0)
RANDOM = {"L0 control layout": {}, "L0 rf_324": dict(kernel_sizes=[3, 3, 3, 3, 4], dilations=[1, 3, 9, 27, 81], effective_context=400),
          "L0 rf_397": dict(kernel_sizes=[3, 3, 3, 3, 5], dilations=[1, 3, 9, 27, 79], effective_context=400),
          "L0 k5_4blocks": dict(num_blocks=4, kernel_sizes=[5] * 4, dilations=[1, 5, 25, 68], effective_context=400),
          "L0 k8_3blocks_w32": dict(num_blocks=3, kernel_sizes=[8] * 3, dilations=[1, 8, 47], channels=64, effective_context=400),
          "L0 native maps": dict(maps="native")}
ROUNDS, RUNS, WARMUP = 5, 3000, 300


def describe(path, shapes):
    graph = onnx.load(str(path)).graph
    return {"nodes": sum(n.op_type != "Constant" for n in graph.node), "macs": count_macs(path)["total"],
            "state_floats": int(sum(np.prod(s) for s in shapes)), "other_ops": audit_operators(path)["other"],
            "stored_weights": int(sum(np.prod(t.dims) for t in graph.initializer))}


def main():
    OUT.mkdir(exist_ok=True)
    segments = validation_segments(DEFAULT_H5, manifest_for(DEFAULT_H5), 420)
    stream = max(segments, key=len)
    graphs, report = {}, {}
    exact7 = load_v7(SEP44)[0]
    graphs["v7 graph (seed 44)"] = export_v7(exact7, OUT / "v7.onnx") + (exact7,)
    lean = deployed(load_checkpoint(SEP44)[0])
    graphs["v9 lean (seed 44)"] = export_streaming_onnx(lean, OUT / "lean.onnx") + (lean,)
    graphs["v9 lean + A4 fused head (seed 44)"] = export_streaming_onnx(lean, OUT / "lean_fused.onnx", fused_head=True) + (lean,)
    for name, options in RANDOM.items():
        torch.manual_seed(0)
        model = LeanTCNN(**{**BEST, **options}).eval()
        with torch.no_grad():
            for p in model.parameters():
                p.normal_(0, .05)
        model = deployed(model)
        graphs[name] = export_streaming_onnx(model, OUT / f"{name.split(' ', 1)[1].replace(' ', '_')}.onnx") + (model,)
    for name, (path, shapes, model) in graphs.items():
        report[name] = {**describe(path, shapes), "params": sum(p.numel() for p in model.parameters())}
        if "seed 44" in name:
            check = stream_parity(model, Streamer(ort_session(path, 1), shapes), segments, 400, 20, 1000)
            report[name]["parity_vs_own_model"] = {k: check[k] for k in ("max_difference_deg", "mean_difference_deg",
                                                                        "onnx_endpoint_error_deg")}
            if "lean" in name:
                versus = stream_parity(exact7, Streamer(ort_session(path, 1), shapes), segments, 400, 20, 1000)
                report[name]["vs_exact_model"] = {k: versus[k] for k in ("max_difference_deg", "mean_difference_deg")}
    calls = {name: Streamer(ort_session(path, 1), shapes) for name, (path, shapes, _) in graphs.items()}
    timings = {name: [] for name in calls}
    for _ in range(ROUNDS):
        for name, call in calls.items():
            call.reset()
            timings[name] += time_stream(call, stream, 400, WARMUP, RUNS)
    for name in calls:
        report[name]["latency_1thread"] = summarise(timings[name])
    (OUT / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    base = report["v7 graph (seed 44)"]["latency_1thread"]["median_ms"]
    print(f"{'graph':38s} {'nodes':>5s} {'params':>6s} {'stored':>6s} {'MACs':>6s} {'state':>6s} {'median us':>9s} {'p90 us':>7s} {'vs v7':>6s}")
    for name, r in report.items():
        lat = r["latency_1thread"]
        print(f"{name:38s} {r['nodes']:5d} {r['params']:6d} {r['stored_weights']:6d} {r['macs']:6d} {r['state_floats']:6d} "
              f"{lat['median_ms']*1e3:9.1f} {lat['p90_ms']*1e3:7.1f} {100*(lat['median_ms']/base-1):+5.0f}%")
    for name, r in report.items():
        if "parity_vs_own_model" in r:
            print(name, "parity", r["parity_vs_own_model"], r.get("vs_exact_model", ""))


if __name__ == "__main__":
    main()
