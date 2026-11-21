"""Finalists' graphs, measured on an idle machine: each checkpoint as the v9 lean graph, with padded rows, and with
padded rows and the fused head (load-time folding); the reference also as its v7 graph. Per graph: nodes in the file
and after ORT's constant folding, parameters, stored floats, MACs, state floats, operator audit, stream parity against
its own model on the longest validation session, and ORT CPU latency (1 thread, alternating rounds).
Usage (from jacob/training): python <this script> label=checkpoint [label=checkpoint ...]   (reference first)
Writes final_latency/report.json next to this script.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

TRAINING = Path(r"<repo>\jacob\training")
sys.path.insert(0, str(TRAINING))

from baseline_pipeline import DEFAULT_H5, manifest_for  # noqa: E402
from benchmark_latency import Streamer, ort_session, stream_parity, summarise, time_stream, validation_segments  # noqa: E402
from tcnn.onnx_export import audit_operators, count_macs  # noqa: E402
from experimental.v7_compact.export import export_streaming_onnx as export_v7  # noqa: E402
from experimental.v7_compact.model import load_checkpoint as load_v7  # noqa: E402
from experimental.v9_lean.export import export_streaming_onnx  # noqa: E402
from experimental.v9_lean.model import deployed, load_checkpoint  # noqa: E402

OUT = Path(__file__).resolve().parent / "final_latency"
ROUNDS, RUNS, WARMUP = 5, 3000, 300


def executed_nodes(path):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(path.with_suffix(".folded.onnx"))
    ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    return sum(n.op_type != "Constant" for n in onnx.load(options.optimized_model_filepath).graph.node)


def describe(path, shapes, model):
    graph = onnx.load(str(path)).graph
    audit = audit_operators(path)
    return {"file_nodes": sum(n.op_type != "Constant" for n in graph.node), "executed_nodes": executed_nodes(path),
            "params": sum(p.numel() for p in model.parameters()),
            "stored_floats": int(sum(np.prod(t.dims) for t in graph.initializer)),
            "macs": count_macs(path)["total"], "state_floats": int(sum(np.prod(s) for s in shapes)),
            "ops": audit["counts"], "off_list_ops": audit["other"] + audit["avoid"]}


def main():
    OUT.mkdir(exist_ok=True)
    segments = validation_segments(DEFAULT_H5, manifest_for(DEFAULT_H5), 420)
    stream = max(segments, key=len)
    graphs = {}
    for i, spec in enumerate(sys.argv[1:]):
        label, checkpoint = spec.split("=", 1)
        checkpoint = TRAINING / checkpoint
        if i == 0:  # the reference, as its v7 graph too
            exact = load_v7(checkpoint)[0]
            graphs[f"{label} | v7 graph"] = export_v7(exact, OUT / f"{i}_v7.onnx") + (exact,)
        model = deployed(load_checkpoint(checkpoint)[0])
        for name, options in {"lean": {}, "padded": dict(padded_rows=True),
                              "padded+A4": dict(padded_rows=True, fused_head=True)}.items():
            graphs[f"{label} | {name}"] = export_streaming_onnx(model, OUT / f"{i}_{name}.onnx", **options) + (model,)
    report = {}
    for name, (path, shapes, model) in graphs.items():
        report[name] = describe(path, shapes, model)
        check = stream_parity(model, Streamer(ort_session(path, 1), shapes), segments, 400, 20, 1000)
        report[name]["parity_max_deg"] = check["max_difference_deg"]
        print(name, {k: report[name][k] for k in ("file_nodes", "executed_nodes", "params", "stored_floats", "macs",
                                                  "state_floats", "off_list_ops", "parity_max_deg")}, flush=True)
    calls = {name: Streamer(ort_session(path, 1), shapes) for name, (path, shapes, _) in graphs.items()}
    timings = {name: [] for name in calls}
    for _ in range(ROUNDS):
        for name, call in calls.items():
            call.reset()
            timings[name] += time_stream(call, stream, 400, WARMUP, RUNS)
    base = np.median(timings[next(iter(calls))])
    print(f"\n{'graph':44s} {'run':>4s} {'params':>6s} {'stored':>6s} {'MACs':>6s} {'state':>6s} {'median us':>9s} {'p90':>6s} {'vs first':>8s}")
    for name in calls:
        r, lat = report[name], summarise(timings[name])
        r["latency_1thread"] = lat
        print(f"{name:44s} {r['executed_nodes']:4d} {r['params']:6d} {r['stored_floats']:6d} {r['macs']:6d} {r['state_floats']:6d} "
              f"{lat['median_ms'] * 1e3:9.1f} {lat['p90_ms'] * 1e3:6.1f} {100 * (np.median(timings[name]) / base - 1):+7.0f}%")
    (OUT / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
