"""Export the selected TCNN to ONNX, check it against PyTorch, and time it like the grading harness.

ST measures latency on an XR edge-AI target at batch size 1, in steady state, as the median of repeated
runs after warm-up. This script reproduces that protocol on this PC with ONNX Runtime (CPU): a stream
slides forward one sample per call and each call predicts the next 20 samples. Local CPU timings are a
relative guide for comparing models, not the official figure; MACs per step and the operator audit are
what the challenge asks you to report.

Exports graphs of the same weights: "full" (every layer computes all 400 steps), "strided" (each layer
computes only the steps the prediction depends on), "fast" (strided, quaternion maths as MatMul) and
"fast_notrig" (fast without trigonometric operators, as the challenge advises for NPUs). Writes
model_fp32_<variant>.onnx, latency.json and latency.md to --output.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

from baseline_pipeline import DEFAULT_H5, EvaluationConfig, inventory, load_split, manifest_for
from tcnn.onnx_export import (OPSET, audit_operators, count_macs, export_onnx, export_streaming_onnx,
                               geodesic_degrees, load_checkpoint)

ROOT = Path(__file__).resolve().parent


def validation_segments(data_path, split_path, minimum):
    """Quaternion arrays of validation-user segments long enough for one window (test users untouched)."""
    _, users = load_split(split_path, "validation")
    segments = []
    with h5py.File(data_path, "r") as h5:
        rows, _ = inventory(h5, users, EvaluationConfig(), False)
        for row in rows:
            q = h5[row["path"]]["quaternion_wxyz"][:].astype(np.float32)
            if len(q) >= minimum:
                segments.append(q)
    if not segments:
        raise ValueError(f"No validation segment of at least {minimum} samples in {data_path}")
    return segments


def synthetic_stream(length, seed=0):
    """Smooth random head motion: integrated, low-pass-filtered angular velocity at 100 Hz."""
    rng = np.random.default_rng(seed)
    velocity = np.cumsum(rng.normal(0, .02, (length, 3)), 0) * .98 ** np.arange(length)[:, None]
    q = np.empty((length, 4), np.float64)
    q[0] = (1, 0, 0, 0)
    for t in range(1, length):
        angle = np.linalg.norm(velocity[t]) / 100
        axis = velocity[t] / max(np.linalg.norm(velocity[t]), 1e-12)
        dq = np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]
        w1, x1, y1, z1 = dq
        w2, x2, y2, z2 = q[t - 1]
        q[t] = (w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2)
        q[t] /= np.linalg.norm(q[t])
    return q.astype(np.float32)


def summarise(seconds):
    ms = np.asarray(seconds) * 1e3
    return {"calls": len(ms), "median_ms": float(np.median(ms)), "mean_ms": float(ms.mean()),
            "p90_ms": float(np.percentile(ms, 90)), "p99_ms": float(np.percentile(ms, 99)),
            "min_ms": float(ms.min()), "max_ms": float(ms.max()), "calls_per_second": float(1e3 / np.median(ms))}


def time_stream(call, stream, context, warmup, runs):
    """Slide one sample per call; warm-up calls are excluded. Returns per-call seconds."""
    if len(stream) < context + warmup + runs:
        raise ValueError("Stream too short for the requested warm-up and runs")
    timings = []
    for step in range(warmup + runs):
        window = np.ascontiguousarray(stream[step:step + context])[None]
        began = time.perf_counter()
        call(window)
        elapsed = time.perf_counter() - began
        if step >= warmup:
            timings.append(elapsed)
    return timings


def ort_session(path, threads):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def parity(model, session, segments, context, horizon, windows, seed=0):
    """ONNX vs PyTorch on validation windows; both also scored against the true future at step 20."""
    rng = np.random.default_rng(seed)
    picks = [(s, rng.integers(0, len(segments[s]) - context - horizon + 1))
             for s in rng.integers(0, len(segments), windows)]
    history = np.stack([segments[s][a:a + context] for s, a in picks])
    future = np.stack([segments[s][a + context:a + context + horizon] for s, a in picks])
    with torch.inference_mode():
        reference = model(torch.from_numpy(history)).numpy()
    exported = np.concatenate([session.run(None, {"context": h[None]})[0] for h in history])
    difference = geodesic_degrees(exported, reference)
    return {"windows": windows, "max_difference_deg": float(difference.max()),
            "mean_difference_deg": float(difference.mean()),
            "pytorch_endpoint_error_deg": float(geodesic_degrees(reference[:, -1], future[:, -1]).mean()),
            "onnx_endpoint_error_deg": float(geodesic_degrees(exported[:, -1], future[:, -1]).mean()),
            "finite": bool(np.isfinite(exported).all()), "output_shape": list(session.run(None, {"context": history[:1]})[0].shape)}


class Streamer:
    """Calls a stateful graph like the harness: zero states at session start, each state_out fed back."""

    def __init__(self, session, shapes):
        self.session, self.names = session, [f"state_in_{i}" for i in range(len(shapes))]
        self.shapes = shapes
        self.reset()

    def reset(self):
        self.states = [np.zeros(shape, np.float32) for shape in self.shapes]

    def __call__(self, window):
        outputs = self.session.run(None, {"context": window, **dict(zip(self.names, self.states))})
        self.states = outputs[1:]
        return outputs[0]


def stream_parity(model, streamer, segments, context, horizon, windows):
    """Stream one validation session from its first sample; compare every full-window prediction."""
    segment = max(segments, key=len)
    length = min(len(segment) - horizon, context - 1 + windows)
    streamer.reset()
    predictions = []
    for t in range(length):
        window = segment[max(0, t - context + 1):t + 1]
        window = np.concatenate((np.repeat(window[:1], context - len(window), 0), window))[None]
        prediction = streamer(np.ascontiguousarray(window))
        if t >= context - 1:
            predictions.append(prediction)
    steps = range(context - 1, length)
    history = np.stack([segment[t - context + 1:t + 1] for t in steps])
    future = np.stack([segment[t + 1:t + 1 + horizon] for t in steps])
    with torch.inference_mode():
        reference = model(torch.from_numpy(history)).numpy()
    exported = np.concatenate(predictions)
    difference = geodesic_degrees(exported, reference)
    return {"windows": len(steps), "max_difference_deg": float(difference.max()),
            "mean_difference_deg": float(difference.mean()),
            "pytorch_endpoint_error_deg": float(geodesic_degrees(reference[:, -1], future[:, -1]).mean()),
            "onnx_endpoint_error_deg": float(geodesic_degrees(exported[:, -1], future[:, -1]).mean()),
            "finite": bool(np.isfinite(exported).all()), "output_shape": list(predictions[0].shape),
            "note": "one session streamed from its first sample"}


STREAM = ("model_fp32_stream.onnx", "stateful: one new sample per call, per-layer inputs cached as state (no trig ops)")
VARIANTS = {"full": ("model_fp32_full.onnx", dict(),
                     "every layer computes all time steps (the trained graph as written)"),
            "strided": ("model_fp32_strided.onnx", dict(strided=True),
                        "each layer computes only the time steps the prediction depends on (stride-2 Conv)"),
            "fast": ("model_fp32_fast.onnx", dict(strided=True, compact=True),
                     "strided, plus quaternion products as MatMul and zero padding inside Conv"),
            "fast_notrig": ("model_fp32_fast_notrig.onnx", dict(strided=True, compact=True, trig_free=True),
                            "fast, with atan/sin/cos as Mul/Add polynomials: no trigonometric operators")}


def write_markdown(path, report):
    proto = report["protocol"]
    lines = [f"# Latency benchmark: {report['checkpoint']}", "",
             f"{report['created_utc'][:19]} UTC · {report['machine']['processor']} · ONNX Runtime "
             f"{report['machine']['onnxruntime']} (CPU) · batch 1 · {proto['runs']} timed calls per setting in "
             f"{proto['rounds']} alternating rounds, {proto['warmup']} warm-up calls before each · stream: {proto['stream']}", "",
             "Local CPU timings compare models on this PC. ST's measurement on the XR target is the official figure.", "",
             "| Model | Threads | Median (ms) | p90 (ms) | p99 (ms) | Max (ms) | Calls/s |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["latency"]:
        lines.append(f"| {row['model']} | {row['threads']} | {row['median_ms']:.3f} | {row['p90_ms']:.3f} | "
                     f"{row['p99_ms']:.3f} | {row['max_ms']:.3f} | {row['calls_per_second']:,.0f} |")
    medians = {(r["model"], r["threads"]): r["median_ms"] for r in report["latency"]}
    if "full" in report["variants"] and len(report["variants"]) > 1:
        lines.append("")
        for threads in proto["threads"]:
            full = medians[("full", threads)]
            lines.append(f"- {threads} thread(s), speed-up over full: " + ", ".join(
                f"{name} {full / medians[(name, threads)]:.1f}x" for name in report["variants"] if name != "full") + ".")
    stream = report["variants"].get("stream")
    lines += ["", "## Models", "",
              f"All files hold the same {report['trainable_parameters']:,} trained parameters. Stateless graphs process "
              f"the {report['effective_context']}-sample window on every call"
              + (f"; the stream graph carries {len(stream['state_shapes'])} state tensors ({stream['state_floats']:,} "
                 f"float32 values) and processes one new sample per call." if stream else "."), "",
              "| Model | What it computes | MACs per step | ONNX (KiB) | Max diff vs PyTorch (°) | Error at 200 ms (°) |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name, v in report["variants"].items():
        p = v["parity"]
        checked = (f"{p['max_difference_deg']:.1e} | {p['onnx_endpoint_error_deg']:.4f}" if p
                   else "not checked | not checked")
        lines.append(f"| {name} (`{Path(v['onnx']).name}`) | {v['description']} | {v['macs_per_step']['total']:,} | "
                     f"{v['onnx_bytes'] / 1024:.0f} | {checked} |")
    first = next((v["parity"] for n, v in report["variants"].items() if n != "stream"), None)
    if first:
        lines += ["", f"Stateless graphs: checked on {first['windows']:,} random validation windows; PyTorch's own "
                  f"error at 200 ms on them is {first['pytorch_endpoint_error_deg']:.4f}°."]
    if stream and stream["parity"]:
        lines += ["", f"Stream graph: one validation session streamed from its first sample, compared on its "
                  f"{stream['parity']['windows']:,} full windows (different windows, so its error is not comparable "
                  f"with the row above); PyTorch's error there is {stream['parity']['pytorch_endpoint_error_deg']:.4f}°."]
    for name, v in report["variants"].items():
        a = v["operators"]
        lines += ["", f"## Operators: {name}", "",
                  f"- On the safe list: {', '.join(a['safe']) or 'none'}",
                  f"- On the avoid list: {', '.join(a['avoid']) or 'none'}",
                  f"- On neither list (elementwise; check with organisers if unsure): {', '.join(a['other']) or 'none'}",
                  f"- Often slow on NPUs: {', '.join(a['slow_on_npu']) or 'none'}",
                  f"- float64 tensors: {len(a['float64_tensors'])}",
                  f"- Counts: {', '.join(f'{k} {c}' for k, c in a['counts'].items())}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/v5_group_a_v2/selected_model.pt")
    p.add_argument("--output", type=Path, default=None, help="Default: <checkpoint folder>/onnx")
    p.add_argument("--variants", nargs="+", choices=[*VARIANTS, "stream"], default=None,
                   help="Default: all; 'stream' only for body_velocity models")
    p.add_argument("--data", type=Path, default=DEFAULT_H5)
    p.add_argument("--split-file", type=Path, help="Default: the manifest_<run_id>.json next to --data")
    p.add_argument("--synthetic", action="store_true", help="Use generated motion instead of validation data")
    p.add_argument("--threads", type=int, nargs="+", default=[1, 4], help="ONNX Runtime intra-op threads to time")
    p.add_argument("--warmup", type=int, default=200, help="Untimed calls before each timed round")
    p.add_argument("--runs", type=int, default=3000, help="Timed calls per model and thread setting")
    p.add_argument("--rounds", type=int, default=3, help="Models alternate in this many rounds, so drift hits all equally")
    p.add_argument("--parity-windows", type=int, default=1000, help="0 skips the ONNX vs PyTorch check")
    p.add_argument("--skip-torch", action="store_true", help="Do not time PyTorch eager for comparison")
    args = p.parse_args()
    import onnxruntime as ort

    output = args.output or args.checkpoint.parent / "onnx"
    model, state = load_checkpoint(args.checkpoint)
    context, horizon = model.context, model.horizon
    per_round = -(-args.runs // args.rounds)

    if args.synthetic or not args.data.exists():
        segments, source = [synthetic_stream(context + args.warmup + per_round + horizon)], "synthetic motion"
    else:
        segments = validation_segments(args.data, args.split_file or manifest_for(args.data), context + horizon)
        source = f"validation users in {args.data.relative_to(ROOT) if args.data.is_relative_to(ROOT) else args.data}"
    stream = max(segments, key=len)
    if len(stream) < context + args.warmup + per_round:
        stream = np.concatenate([stream, synthetic_stream(context + args.warmup + per_round)])

    streamable = getattr(model, "input_features", "anchor") == "body_velocity"
    requested = args.variants or [*VARIANTS, *(["stream"] if streamable else [])]
    if "stream" in requested and not streamable:
        raise SystemExit("Only body_velocity models can stream; anchor inputs change for every past step.")
    variants, streamers = {}, {}
    if "stream" in requested:
        path, shapes = export_streaming_onnx(model, output / STREAM[0])
        operators, macs = audit_operators(path), count_macs(path)
        streamers["stream"] = lambda threads, path=path, shapes=shapes: Streamer(ort_session(path, threads), shapes)
        check = None
        if args.parity_windows:
            check = stream_parity(model, streamers["stream"](1), segments, context, horizon, args.parity_windows)
            if not check["finite"] or check["max_difference_deg"] > 1e-2 or check["output_shape"] != [1, horizon, 4]:
                raise SystemExit("stream: streaming graph does not match the PyTorch model; not timing it.")
        variants["stream"] = {"onnx": str(path), "onnx_bytes": path.stat().st_size, "options": {"stateful": True},
                              "state_shapes": [list(s) for s in shapes], "state_floats": int(sum(np.prod(s) for s in shapes)),
                              "description": STREAM[1], "macs_per_step": macs, "operators": operators, "parity": check}
        print(f"stream: exported {path}; {macs['total']:,} MACs per step; {len(shapes)} state tensors "
              f"({variants['stream']['state_floats']:,} floats)"
              + (f"; max diff vs PyTorch {check['max_difference_deg']:.1e} deg" if check else ""), flush=True)
    for name in [n for n in requested if n != "stream"]:
        filename, options, description = VARIANTS[name]
        path = export_onnx(model, output / filename, **options)
        operators, macs = audit_operators(path), count_macs(path)
        check = None
        if args.parity_windows:
            check = parity(model, ort_session(path, 1), segments, context, horizon, args.parity_windows)
            if not check["finite"] or check["max_difference_deg"] > 1e-2 or check["output_shape"] != [1, horizon, 4]:
                raise SystemExit(f"{name}: ONNX export does not match the PyTorch model; not timing it.")
        variants[name] = {"onnx": str(path), "onnx_bytes": path.stat().st_size, "options": options,
                          "description": description, "macs_per_step": macs, "operators": operators, "parity": check}
        summary = f"{name}: exported {path}; {macs['total']:,} MACs per step; avoid-list ops {operators['avoid'] or 'none'}"
        if check:
            summary += (f"; max diff vs PyTorch {check['max_difference_deg']:.1e} deg; 200 ms error "
                        f"{check['onnx_endpoint_error_deg']:.4f} deg (PyTorch {check['pytorch_endpoint_error_deg']:.4f})")
        print(summary, flush=True)

    def runner(session):
        return lambda window: session.run(None, {"context": window})

    torch.set_num_threads(1)
    calls = {(name, threads): streamers[name](threads) if name in streamers else runner(ort_session(v["onnx"], threads))
             for threads in args.threads for name, v in variants.items()}
    if not args.skip_torch:
        calls[("PyTorch eager (reference)", 1)] = lambda window: model(torch.from_numpy(window))
    timings = {key: [] for key in calls}
    with torch.inference_mode():
        for _ in range(args.rounds):
            for key, call in calls.items():
                timings[key] += time_stream(call, stream, context, args.warmup, per_round)
    results = [{"model": name, "threads": threads,
                "runtime": "PyTorch" if name.startswith("PyTorch") else "ONNX Runtime CPU", **summarise(seconds)}
               for (name, threads), seconds in timings.items()]
    for row in results:
        print(f"{row['model']}, {row['threads']} thread(s): median {row['median_ms']:.3f} ms, p99 {row['p99_ms']:.3f} ms")

    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": str(args.checkpoint),
              "protocol": {"batch_size": 1, "warmup": args.warmup, "runs": per_round * args.rounds, "rounds": args.rounds,
                           "threads": args.threads, "stream": source,
                           "timed": "session.run per call, window slides by one sample"},
              "machine": {"processor": platform.processor() or platform.machine(), "platform": platform.platform(),
                          "cpu_count": os.cpu_count(), "onnxruntime": ort.__version__, "torch": str(torch.__version__)},
              "model_spec": state["model_spec"], "context": context, "effective_context": model.effective_context,
              "opset": OPSET,
              "trainable_parameters": sum(t.numel() for t in model.parameters()),
              "variants": variants, "latency": results}
    (output / "latency.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(output / "latency.md", report)
    print(f"Report: {output / 'latency.md'}")


if __name__ == "__main__":
    main()
