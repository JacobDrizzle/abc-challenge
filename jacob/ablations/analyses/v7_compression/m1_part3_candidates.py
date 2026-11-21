"""M1 (part 3): local latency of the confirm-stage candidates, separable + base 3 among them, timed in one
session with alternating rounds against the screened variants they come from. Random weights (std .05, as in
part 2). Also checks the full-width separable + base 3 stream against PyTorch, since the unit test covers it at
8 channels only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"<repo>\jacob\training")
sys.path.insert(0, str(ROOT))
import run_next  # noqa: E402
from baseline_pipeline import geodesic_degrees  # noqa: E402
from benchmark_latency import Streamer, ort_session, summarise, synthetic_stream, time_stream  # noqa: E402
from experimental.v6_gravity_mirror.export import export_streaming_onnx as export_v6  # noqa: E402
from experimental.v6_gravity_mirror.test_v6 import predict  # noqa: E402
from experimental.v7_compact import hooks  # noqa: E402
from experimental.v7_compact.export import export_streaming_onnx as export_v7  # noqa: E402
from experimental.v7_compact.model import CompactTCNN  # noqa: E402
from tcnn.onnx_export import audit_operators, count_macs  # noqa: E402

HERE = Path(__file__).parent
GRAPHS = HERE / "m1_graphs_v7"
WARMUP, RUNS, ROUNDS = 200, 3000, 5
SUITE = ROOT / "experimental/v7_compact/configs"
KEYS = ("context", "horizon", "channels", "dropout", "effective_context", "damped_prior_tau_ms",
        "input_features", "num_blocks", *hooks.MODEL_KEYS)
ADDONS = {"speed_feature": "l1", "head_basis": 3}
VARIANTS = {"control": {}, "base3_5": {"dilation_base": 3, "num_blocks": 5}, "separable_24": {"separable": True},
            "narrow_base3": {"channels": 32, "dilation_base": 3, "num_blocks": 5},
            "separable_base3": {"separable": True, "dilation_base": 3, "num_blocks": 5},
            "base3_5_addons": {"dilation_base": 3, "num_blocks": 5, **ADDONS},
            "separable_24_addons": {"separable": True, **ADDONS},
            "narrow_full_addons": {"channels": 32, **ADDONS},
            "separable_base3_addons": {"separable": True, "dilation_base": 3, "num_blocks": 5, **ADDONS}}


def build(config):
    torch.manual_seed(0)
    model = CompactTCNN(**{k: config[k] for k in KEYS}).eval()
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


with hooks.installed():
    configs = run_next.experiment_configs({**json.loads((SUITE / "v7_run.json").read_text()), "variants": VARIANTS},
                                          json.loads((SUITE / "tcnn_v7.json").read_text()))
GRAPHS.mkdir(exist_ok=True)
entries, check = [], synthetic_stream(650, seed=1).astype(np.float32)
for name, config in [("control (v6 export)", configs["control"]), *configs.items()]:
    model = build(config)
    exporter = export_v6 if name.endswith("(v6 export)") else export_v7
    path, shapes = exporter(model, GRAPHS / f"{name.split()[0]}{'_v6maths' if exporter is export_v6 else ''}.onnx")
    audit = audit_operators(path)
    entries.append(dict(name=name, path=str(path), shapes=shapes, params=sum(p.numel() for p in model.parameters()),
                        macs=count_macs(path)["total"], state=int(sum(np.prod(s) for s in shapes)),
                        nodes=sum(n for op, n in audit["counts"].items() if op != "Constant"),
                        counts=audit["counts"], other=audit["other"],
                        parity_deg=stream_parity(model, path, shapes, check) if name.startswith("separable_base3")
                        else None))

stream = synthetic_stream(400 + WARMUP + RUNS)
sessions = [(e, Streamer(ort_session(e["path"], 1), e["shapes"])) for e in entries]
timings = {e["name"]: [] for e in entries}
rounds = {e["name"]: [] for e in entries}
for _ in range(ROUNDS):
    for entry, streamer in sessions:
        streamer.reset()
        calls = time_stream(streamer, stream, 400, WARMUP, RUNS)
        timings[entry["name"]] += calls
        rounds[entry["name"]].append(float(np.median(calls)) * 1e3)
reference = None
lines = ["| Graph | Params | MACs/step | State floats | Nodes | Off-list | Median (ms) | p90 (ms) | "
         "Round medians spread (ms) | vs v6 graph |",
         "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
for entry in entries:
    entry.update(summarise(timings[entry["name"]]), round_medians_ms=rounds[entry["name"]])
    reference = reference or entry["median_ms"]
    spread = max(rounds[entry["name"]]) - min(rounds[entry["name"]])
    lines.append(f"| {entry['name']} | {entry['params']:,} | {entry['macs']:,} | {entry['state']:,} | {entry['nodes']} | "
                 f"{', '.join(entry['other']) or 'none'} | {entry['median_ms']:.4f} | {entry['p90_ms']:.4f} | "
                 f"{spread:.4f} | {100 * (entry['median_ms'] / reference - 1):+.1f}% |")
lines += ["", *(f"Stream parity, {e['name']}: max {e['parity_deg']:.2e} deg over 251 windows"
                for e in entries if e["parity_deg"] is not None)]
print("\n".join(lines))
(HERE / "m1_part3_candidates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(HERE / "m1_part3_candidates.json").write_text(json.dumps(entries, indent=2, default=str))
