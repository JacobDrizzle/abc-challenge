"""Post-training int8 quantisation of the streaming graph (export.py), and its validation error against float.

python -m best.quantize --checkpoint outputs/v9_confirm/k8_combined/selected_model.pt [--output DIR]

E3 (v7, 26 Oct) ran this on v6 control through the v7 graph: int8 minmax +0.021 deg, int8 percentile +0.012, w8a16
+0.006 (outputs/v7_e3_int8). That graph computes exact maps, so it cannot reproduce a native-map model such as
k8_combined; this version quantises the default lean graph instead, the only change being which nodes hold the
weights (network_nodes). It has not been run on k8_combined yet.

Only the network is quantised: every block Conv (depthwise and pointwise) and the head's MatMul (ONNX Runtime QDQ
format, int8 weights per output channel). The feature maths, the projection (its inputs mix rad/s rates with unit gravity
in one tensor, so one int8 scale would serve gravity poorly; --include-projection overrides) and the output
quaternion maths stay float, and so do the state tensors, which the harness requires to be float32.
Calibration streams TRAIN users through the float graph and records real (context, state) inputs; the error is
measured by streaming every VALIDATION segment from its first sample, as the harness does, and scoring every
full window at step 20. Test users are never read. The chosen schemes are compared in one report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import onnx
from onnx import numpy_helper

from baseline_pipeline import DEFAULT_H5, EvaluationConfig, inventory, load_split, manifest_for
from benchmark_latency import Streamer, ort_session, summarise, time_stream
from tcnn.onnx_export import audit_operators, count_macs, geodesic_degrees

CONTEXT, HORIZON = 400, 20
# name: (activation type, calibration method, ORT extra options)
SCHEMES = {"int8_minmax": ("QInt8", "MinMax", {}),
           "int8_percentile": ("QInt8", "Percentile", {"CalibPercentile": 99.999}),
           "w8a16_minmax": ("QInt16", "MinMax", {})}


def split_segments(data, split_path, split):
    """Float32 quaternion arrays of one split's segments that hold at least one full window."""
    _, users = load_split(split_path, split)
    with h5py.File(data, "r") as h5:
        rows, _ = inventory(h5, users, EvaluationConfig(), False)
        segments = [h5[row["path"]]["quaternion_wxyz"][:].astype(np.float32) for row in rows]
    return [q for q in segments if len(q) >= CONTEXT + HORIZON]


def windows(q, end):
    """Harness-style calls t = 1 .. end - 1: the latest 400 samples, the first one repeated before the start."""
    padded = np.concatenate((np.repeat(q[:1], CONTEXT - 1, 0), q))
    for t in range(1, end):
        yield t, padded[t:t + CONTEXT][None]


def calibration_inputs(path, shapes, segments, count, seed=0, every=40):
    """count graph inputs (context and warm states), every `every` steps after the window first fills."""
    streamer, rng, inputs = Streamer(ort_session(path, 1), shapes), np.random.default_rng(seed), []
    for index in rng.permutation(len(segments)):
        q = segments[index]
        streamer.reset()
        for t, window in windows(q, min(len(q), CONTEXT + every * 40)):
            if t >= CONTEXT - 1 and (t - CONTEXT + 1) % every == 0:
                inputs.append({"context": window, **dict(zip(streamer.names, streamer.states))})
                if len(inputs) == count:
                    return inputs
            streamer(window)
    return inputs


def network_nodes(path, model, include_projection=False):
    """Names of the nodes that apply the network's weights in the default streaming graph: each block's depthwise
    and pointwise Conv (and the projection Conv) and the head's MatMul, W @ [h0, h1], whose weight is input 0."""
    graph = onnx.load(str(path)).graph
    constants = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    constants.update({node.output[0]: numpy_helper.to_array(node.attribute[0].t) for node in graph.node
                      if node.op_type == "Constant" and node.attribute and node.attribute[0].name == "value"})
    head = model.head.weight.detach().numpy()
    projection = model.projection.weight.detach().numpy()[..., None]  # [K, F, 1, 1], as the graph's Conv2d holds it

    def equal(a, b):
        return a is not None and a.shape == b.shape and np.array_equal(a, b)

    names = []
    for node in graph.node:
        weight = constants.get(node.input[1]) if len(node.input) > 1 else None
        if node.op_type == "Conv" and weight is not None and (include_projection or not equal(weight, projection)):
            names.append(node.name)
        elif node.op_type == "MatMul" and equal(constants.get(node.input[0]), head):
            names.append(node.name)
    expected = 2 * len(model.blocks) + 1 + include_projection
    if len(names) != expected or len(set(names)) != len(names):
        raise ValueError(f"Expected {expected} uniquely named network nodes, found {names}")
    return names


def quantize_stream(float_path, out_path, model, inputs, scheme, include_projection=False):
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.items = iter(inputs)

        def get_next(self):
            return next(self.items, None)

    activations, method, extra = SCHEMES[scheme]
    quantize_static(str(float_path), str(out_path), Reader(), quant_format=QuantFormat.QDQ,
                    op_types_to_quantize=["Conv", "MatMul", "Gemm"],
                    nodes_to_quantize=network_nodes(float_path, model, include_projection), per_channel=True,
                    activation_type=getattr(QuantType, activations), weight_type=QuantType.QInt8,
                    calibrate_method=getattr(CalibrationMethod, method), extra_options=extra)
    onnx.checker.check_model(onnx.load(str(out_path)))
    return out_path


def stream_errors(graphs, segments):
    """Step-20 geodesic errors (degrees) of every full window, per graph, plus each graph's drift from the first."""
    streamers = {name: Streamer(ort_session(path, 1), shapes) for name, (path, shapes) in graphs.items()}
    errors = {name: [] for name in graphs}
    endpoints = {name: [] for name in graphs}
    for q in segments:
        for streamer in streamers.values():
            streamer.reset()
        for t, window in windows(q, len(q) - HORIZON):
            for name, streamer in streamers.items():
                prediction = streamer(window)
                if t >= CONTEXT - 1:
                    endpoints[name].append(prediction[0, -1])
        targets = q[CONTEXT - 1 + HORIZON:]
        for name in graphs:
            errors[name].append(geodesic_degrees(np.stack(endpoints[name]), targets))
            endpoints[name] = []
    return {name: np.concatenate(value) for name, value in errors.items()}


def main():
    from best.export import export_streaming_onnx
    from best.model import deployed, load_checkpoint

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="A checkpoint the streaming graph exports (.pt)")
    p.add_argument("--output", type=Path, default=None, help="Default: <checkpoint folder>/int8")
    p.add_argument("--data", type=Path, default=DEFAULT_H5)
    p.add_argument("--schemes", nargs="+", choices=list(SCHEMES), default=list(SCHEMES))
    p.add_argument("--calibration-inputs", type=int, default=1000)
    p.add_argument("--include-projection", action="store_true")
    p.add_argument("--runs", type=int, default=3000, help="Timed calls per graph (3 alternating rounds)")
    args = p.parse_args()

    output = args.output or args.checkpoint.parent / "int8"
    output.mkdir(parents=True, exist_ok=True)
    split = manifest_for(args.data)
    model = deployed(load_checkpoint(args.checkpoint)[0])  # the maps the graph computes
    float_path, shapes = export_streaming_onnx(model, output / "model_fp32_stream.onnx")
    print(f"Float graph: {float_path}", flush=True)
    inputs = calibration_inputs(float_path, shapes, split_segments(args.data, split, "train"), args.calibration_inputs)
    print(f"Calibration: {len(inputs)} streamed inputs from train users", flush=True)
    graphs = {"float32": (float_path, shapes)}
    for scheme in args.schemes:
        path = quantize_stream(float_path, output / f"model_{scheme}_stream.onnx", model, inputs, scheme,
                               args.include_projection)
        graphs[scheme] = (path, shapes)
        print(f"Quantised: {path}", flush=True)

    validation = split_segments(args.data, split, "validation")
    errors = stream_errors(graphs, validation)
    stream = max(validation, key=len)
    sessions = {name: Streamer(ort_session(path, 1), shapes) for name, (path, shapes) in graphs.items()}
    timings = {name: [] for name in graphs}
    per_round = -(-args.runs // 3)
    for _ in range(3):
        for name, streamer in sessions.items():
            streamer.reset()
            timings[name] += time_stream(streamer, stream, CONTEXT, 200, per_round)

    rows, reference = [], errors["float32"].mean()
    for name, (path, _) in graphs.items():
        audit = audit_operators(path)
        rows.append({"graph": name, "file": path.name, "kib": round(path.stat().st_size / 1024, 1),
                     "windows": int(len(errors[name])), "mean_endpoint_deg": float(errors[name].mean()),
                     "delta_vs_float_deg": float(errors[name].mean() - reference),
                     "per_window_abs_delta_p99_deg": float(np.percentile(np.abs(errors[name] - errors["float32"]), 99)),
                     "macs": count_macs(path)["total"], "counts": audit["counts"], "off_list": audit["other"],
                     "latency": summarise(timings[name])})
    report = {"checkpoint": str(args.checkpoint), "data": str(args.data), "calibration_inputs": len(inputs),
              "include_projection": args.include_projection, "schemes": {k: SCHEMES[k][:2] for k in args.schemes},
              "rows": rows}
    (output / "quantization.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [f"# Int8 post-training quantisation: {args.checkpoint}", "",
             f"Validation users, every full window streamed from each segment's first sample "
             f"({rows[0]['windows']:,} windows). Calibration: {len(inputs)} streamed inputs from train users. "
             f"Quantised: block Convs and head{' and projection' if args.include_projection else ''}; "
             "features, output maths and states stay float. Latency: ORT CPU, 1 thread (relative only).", "",
             "| Graph | Step-20 mean (°) | vs float (°) | p99 per-window change (°) | Median (ms) | KiB | Off-list |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    lines += [f"| {r['graph']} | {r['mean_endpoint_deg']:.4f} | {r['delta_vs_float_deg']:+.4f} | "
              f"{r['per_window_abs_delta_p99_deg']:.4f} | {r['latency']['median_ms']:.4f} | {r['kib']} | "
              f"{', '.join(r['off_list']) or 'none'} |" for r in rows]
    (output / "quantization.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
