"""Every metric of every v7 variant, from the trained seed-42 checkpoints, in one comparison.

Accuracy: read from each run's full stride-1 validation (657,189 windows; test is never read).
Training: from history.json. Cost: each trained checkpoint exported through the v7 reduced streaming graph, plus
its stream parity against PyTorch on a real validation segment. Latency: ORT CPU, 1 thread, every graph in one
session with alternating rounds; the v6 export of control is the reference. Untrained add-on combinations are
included with random weights (cost only). Trained graphs stay in outputs/ (gitignored) and must not be uploaded.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"<repo>\jacob\training")
sys.path.insert(0, str(ROOT))
import run_next  # noqa: E402
from baseline_pipeline import DEFAULT_H5, geodesic_degrees, manifest_for  # noqa: E402
from benchmark_latency import Streamer, ort_session, summarise, synthetic_stream, time_stream  # noqa: E402
from experimental.v6_gravity_mirror.export import export_streaming_onnx as export_v6  # noqa: E402
from experimental.v6_gravity_mirror.test_v6 import predict  # noqa: E402
from experimental.v7_compact import hooks  # noqa: E402
from experimental.v7_compact.export import export_streaming_onnx  # noqa: E402
from experimental.v7_compact.model import CompactTCNN, load_checkpoint  # noqa: E402
from experimental.v7_compact.quantize import split_segments  # noqa: E402
from tcnn.onnx_export import audit_operators, count_macs  # noqa: E402

OUT = ROOT / "outputs/v7_metrics"
SCREEN, EXTRA = ROOT / "outputs/v7_group_a_v2", ROOT / "outputs/v7_extra"
TRAINED = {name: SCREEN / name for name in ("control", "speed_l1", "basis_head", "base3_5", "width_20",
                                            "narrow_full", "separable_24", "narrow_base3")}
TRAINED["separable_base3_addons"] = EXTRA / "separable_base3_addons"
ADDONS = {"speed_feature": "l1", "head_basis": 3}
UNTRAINED = {"base3_5_addons": {"dilation_base": 3, "num_blocks": 5, **ADDONS},
             "separable_24_addons": {"separable": True, **ADDONS},
             "narrow_full_addons": {"channels": 32, **ADDONS}}
KEYS = ("context", "horizon", "channels", "dropout", "effective_context", "damped_prior_tau_ms",
        "input_features", "num_blocks", *hooks.MODEL_KEYS)
SUITE = ROOT / "experimental/v7_compact/configs"
WARMUP, RUNS, ROUNDS = 200, 3000, 5
TOLERANCE = 4.382  # 3-seed validation mean allowed by the plan (v6 control mean 4.342 + 0.04)


def accuracy(folder):
    """Diagnostics of the seed-42 checkpoint from its saved full validation."""
    results = json.loads((folder / "seed_42/full_validation/results.json").read_text(encoding="utf-8"))
    model, hold = results["models"]["tiny_tcnn"], results["models"]["hold_last"]
    users = [u["mean_deg_by_horizon"][-1] for u in model["by_user"].values()]
    curve = model["mean_deg_by_horizon"]
    return {"endpoint_mean": model["mean_endpoint_deg"], "endpoint_median": model["median_endpoint_deg"],
            "endpoint_p95": model["p95_endpoint_deg"], "endpoint_p99": model["p99_endpoint_deg"],
            "all_horizon_mean": model["mean_all_horizons_deg"], "step5": curve[4], "step10": curve[9],
            "equal_user_mean": float(np.mean(users)), "worst_user": float(np.max(users)),
            "vs_hold_last_pct": 100 * (model["mean_endpoint_deg"] / hold["mean_endpoint_deg"] - 1)}


def training(folder):
    history = json.loads((folder / "seed_42/history.json").read_text(encoding="utf-8"))
    return {"epochs": len(history), "best_epoch": history[-1]["best_epoch"],
            "train_minutes": sum(h["seconds"] for h in history) / 60,
            "peak_gpu_mb": max(h["peak_gpu_memory_mb"] for h in history)}


def random_model(config):
    torch.manual_seed(0)
    model = CompactTCNN(**config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, .05)
    return model


def stream_parity(model, path, shapes, stream):
    """Largest geodesic gap (degrees) between the streamed graph and PyTorch over every full window."""
    streamer, got = Streamer(ort_session(path, 1), shapes), []
    padded = np.concatenate((np.repeat(stream[:1], 399, 0), stream))
    for t in range(1, len(stream)):
        prediction = streamer(padded[t:t + 400][None])
        if t >= 399:
            got.append(prediction[0])
    reference = predict(model, np.stack([stream[t - 399:t + 1] for t in range(399, len(stream))]))
    return float(geodesic_degrees(np.stack(got), reference).max())


def graph_costs(model, path, shapes):
    audit = audit_operators(path)
    return {"params": sum(p.numel() for p in model.parameters()), "macs": count_macs(path)["total"],
            "state_floats": int(sum(np.prod(s) for s in shapes)),
            "nodes": sum(n for op, n in audit["counts"].items() if op != "Constant"),
            "off_list": audit["other"], "kib": round(path.stat().st_size / 1024, 1)}


(OUT / "graphs_trained").mkdir(parents=True, exist_ok=True)
(OUT / "graphs_random").mkdir(exist_ok=True)
segment = split_segments(DEFAULT_H5, manifest_for(DEFAULT_H5), "validation")[0][:650]
rows, control = {}, None
for name, folder in TRAINED.items():
    model = load_checkpoint(folder / "seed_42/best.pt")[0].eval()
    path, shapes = export_streaming_onnx(model, OUT / "graphs_trained" / f"{name}.onnx")
    rows[name] = {"trained": True, "path": str(path), "shapes": shapes, **accuracy(folder), **training(folder),
                  **graph_costs(model, path, shapes), "parity_deg": stream_parity(model, path, shapes, segment)}
    control = control or model
    print(f"{name}: {rows[name]['endpoint_mean']:.4f} deg, {rows[name]['params']:,} params", flush=True)
path, shapes = export_v6(control, OUT / "graphs_trained" / "control_v6maths.onnx")
reference = {"trained": True, "path": str(path), "shapes": shapes, **graph_costs(control, path, shapes)}
with hooks.installed():
    configs = run_next.experiment_configs(
        {**json.loads((SUITE / "v7_run.json").read_text()), "variants": {"control": {}, **UNTRAINED}},
        json.loads((SUITE / "tcnn_v7.json").read_text()))
for name in UNTRAINED:
    model = random_model({k: configs[name][k] for k in KEYS})
    path, shapes = export_streaming_onnx(model, OUT / "graphs_random" / f"{name}.onnx")
    rows[name] = {"trained": False, "path": str(path), "shapes": shapes, **graph_costs(model, path, shapes)}

graphs = {"v6 export (control weights)": reference, **rows}
stream = synthetic_stream(400 + WARMUP + RUNS)
sessions = {name: Streamer(ort_session(g["path"], 1), g["shapes"]) for name, g in graphs.items()}
timings = {name: [] for name in graphs}
for _ in range(ROUNDS):
    for name, streamer in sessions.items():
        streamer.reset()
        timings[name] += time_stream(streamer, stream, 400, WARMUP, RUNS)
for name, g in graphs.items():
    g["latency"] = summarise(timings[name])
v6_ms, control_deg = reference["latency"]["median_ms"], rows["control"]["endpoint_mean"]

order = sorted(rows, key=lambda n: -rows[n]["params"])
lines = ["# v7 variants: every metric (seed 42)", "",
         "Validation users only, 657,189 stride-1 windows, step 20 = 200 ms ahead. Single seed: control's three "
         "v6 seeds span 0.023 deg, so smaller gaps are not resolved. Plan tolerance for a 3-seed mean: "
         f"{TOLERANCE} deg. Latency: ORT CPU, 1 thread, one session, {ROUNDS} alternating rounds of {RUNS:,} calls "
         "(relative only; not the challenge target).", "",
         "## Accuracy (degrees, lower is better)", "",
         "| Variant | Step-20 mean | vs control | Median | p95 | p99 | Step 5 | Step 10 | All-horizon mean | "
         "Equal-user mean | Worst user | vs hold-last |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for name in sorted((n for n in order if rows[n]["trained"]), key=lambda n: rows[n]["endpoint_mean"]):
    r = rows[name]
    lines.append(f"| {name} | {r['endpoint_mean']:.4f} | {r['endpoint_mean'] - control_deg:+.4f} | "
                 f"{r['endpoint_median']:.4f} | {r['endpoint_p95']:.3f} | {r['endpoint_p99']:.3f} | {r['step5']:.4f} | "
                 f"{r['step10']:.4f} | {r['all_horizon_mean']:.4f} | {r['equal_user_mean']:.4f} | "
                 f"{r['worst_user']:.3f} | {r['vs_hold_last_pct']:+.1f}% |")
lines += ["", "## Size, graph and latency", "",
          "| Variant | Step-20 mean | Params | MACs/step | State floats | Nodes | Off-list | KiB | Median (ms) | "
          "p90 (ms) | vs v6 graph | Parity (deg) | Best epoch | Train (min) |",
          "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for name in ["v6 export (control weights)", *order]:
    g = graphs[name]
    trained = name in rows and rows[name]["trained"]
    lines.append(f"| {name}{'' if name not in rows or trained else ' (not trained)'} | "
                 f"{g['endpoint_mean']:.4f} | " if trained else
                 f"| {name}{'' if name not in rows else ' (not trained)'} | — | ")
    lines[-1] += (f"{g['params']:,} | {g['macs']:,} | {g['state_floats']:,} | {g['nodes']} | "
                  f"{', '.join(g['off_list']) or 'none'} | {g['kib']} | {g['latency']['median_ms']:.4f} | "
                  f"{g['latency']['p90_ms']:.4f} | {100 * (g['latency']['median_ms'] / v6_ms - 1):+.1f}% | " +
                  (f"{g['parity_deg']:.1e} | {g['best_epoch']} | {g['train_minutes']:.1f} |" if trained else
                   "— | — | — |"))
print("\n".join(lines))
(OUT / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(OUT / "metrics.json").write_text(json.dumps(graphs, indent=2, default=str), encoding="utf-8")
