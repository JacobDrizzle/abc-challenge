"""M1 (part 2): local latency of every v7 suite variant's streaming graph (v7 reduced export), plus v6's own
export of the control architecture as the reference, timed in one session with alternating rounds.
Random weights; graphs are kept in m1_graphs_v7/ for the M3 uploads.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"<repo>\jacob\training")
sys.path.insert(0, str(ROOT))
import run_next  # noqa: E402
from benchmark_latency import Streamer, ort_session, summarise, synthetic_stream, time_stream  # noqa: E402
from experimental.v6_gravity_mirror.export import export_streaming_onnx as export_v6  # noqa: E402
from experimental.v7_compact import hooks  # noqa: E402
from experimental.v7_compact.export import export_streaming_onnx as export_v7  # noqa: E402
from experimental.v7_compact.model import CompactTCNN  # noqa: E402
from tcnn.onnx_export import audit_operators, count_macs  # noqa: E402

HERE = Path(__file__).parent
GRAPHS = HERE / "m1_graphs_v7"
WARMUP, RUNS, ROUNDS = 200, 3000, 3
SUITE = ROOT / "experimental/v7_compact/configs"
KEYS = ("context", "horizon", "channels", "dropout", "effective_context", "damped_prior_tau_ms",
        "input_features", "num_blocks", *hooks.MODEL_KEYS)

with hooks.installed():
    configs = run_next.experiment_configs(json.loads((SUITE / "v7_run.json").read_text()),
                                          json.loads((SUITE / "tcnn_v7.json").read_text()))
GRAPHS.mkdir(exist_ok=True)
entries = []
for name, config in [("control (v6 export)", configs["control"]), *configs.items()]:
    torch.manual_seed(0)
    model = CompactTCNN(**{k: config[k] for k in KEYS}).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, .05)
    exporter = export_v6 if name.endswith("(v6 export)") else export_v7
    path, shapes = exporter(model, GRAPHS / f"{name.split()[0]}{'_v6maths' if exporter is export_v6 else ''}.onnx")
    audit = audit_operators(path)
    entries.append(dict(name=name, path=str(path), shapes=shapes, params=sum(p.numel() for p in model.parameters()),
                        macs=count_macs(path)["total"], state=int(sum(np.prod(s) for s in shapes)),
                        nodes=sum(n for op, n in audit["counts"].items() if op != "Constant"), other=audit["other"]))

stream = synthetic_stream(400 + WARMUP + RUNS)
sessions = [(e, Streamer(ort_session(e["path"], 1), e["shapes"])) for e in entries]
timings = {e["name"]: [] for e in entries}
for _ in range(ROUNDS):
    for entry, streamer in sessions:
        streamer.reset()
        timings[entry["name"]] += time_stream(streamer, stream, 400, WARMUP, RUNS)
reference = None
lines = ["| Graph | Params | MACs/step | State floats | Nodes | Off-list | Median (ms) | vs v6 graph |",
         "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |"]
for entry in entries:
    entry.update(summarise(timings[entry["name"]]))
    reference = reference or entry["median_ms"]
    lines.append(f"| {entry['name']} | {entry['params']:,} | {entry['macs']:,} | {entry['state']:,} | {entry['nodes']} | "
                 f"{', '.join(entry['other']) or 'none'} | {entry['median_ms']:.4f} | "
                 f"{100 * (entry['median_ms'] / reference - 1):+.1f}% |")
print("\n".join(lines))
(HERE / "m1_part2_suite.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(HERE / "m1_part2_suite.json").write_text(json.dumps(entries, indent=2, default=str))
