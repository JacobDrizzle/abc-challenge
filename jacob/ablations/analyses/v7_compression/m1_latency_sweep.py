"""M1 (part 1): local latency of random-weight v6 streaming graphs against MACs, node count and state size.

Grid: width per stream x layout (7 blocks, dilations 2**i; or 5 blocks, dilations 3**i) x mirror on/off.
Mirror off uses the same total channels, so its MACs match and its parameters double. Graphs come from the
v6 exporter unchanged; weights are random because latency does not depend on their values.
Timing follows benchmark_latency.py: ORT CPU, 1 thread, batch 1, synthetic motion, models alternate rounds.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

ROOT = Path(r"<repo>\jacob\training")
sys.path.insert(0, str(ROOT))
from benchmark_latency import Streamer, ort_session, summarise, synthetic_stream, time_stream  # noqa: E402
from experimental.v6_gravity_mirror.export import export_streaming_onnx  # noqa: E402
from experimental.v6_gravity_mirror.model import GravityMirrorTCNN, MirrorCausalBlock  # noqa: E402
from tcnn.model import CausalBlock  # noqa: E402
from tcnn.onnx_export import audit_operators, count_macs  # noqa: E402

HERE = Path(__file__).parent
GRAPHS = HERE / "m1_graphs"
WIDTHS, LAYOUTS = (8, 12, 16, 20, 24, 32), {"base2_7": (2, 7), "base3_5": (3, 5)}
WARMUP, RUNS, ROUNDS = 200, 2000, 3


def build(width, layout, mirror):
    base, blocks = LAYOUTS[layout]
    torch.manual_seed(0)
    model = GravityMirrorTCNN(channels=2 * width, num_blocks=blocks, effective_context=256,
                              damped_prior_tau_ms=100, gravity_features=True, mirror_equivariant=mirror)
    block = MirrorCausalBlock if mirror else CausalBlock
    model.blocks = nn.Sequential(*(block(model.stream_channels, base ** i, .1) for i in range(blocks)))
    nn.init.normal_(model.head.weight, std=.01)  # the zero init would leave the head inert
    return model.eval()


def artefact_floats(path):
    """Float values stored in the file: initializers plus Constant-node tensors (tables included)."""
    graph = onnx.load(str(path)).graph
    total = sum(int(np.prod(t.dims)) for t in graph.initializer)
    for node in graph.node:
        if node.op_type == "Constant":
            total += sum(int(np.prod(a.t.dims)) for a in node.attribute if a.type == onnx.AttributeProto.TENSOR)
    return total


GRAPHS.mkdir(exist_ok=True)
entries = []
for layout in LAYOUTS:
    for mirror in (True, False):
        for width in WIDTHS:
            name = f"{layout}_w{width}_{'mirror' if mirror else 'plain'}"
            model = build(width, layout, mirror)
            path, shapes = export_streaming_onnx(model, GRAPHS / f"{name}.onnx")
            audit = audit_operators(path)
            entries.append(dict(
                name=name, layout=layout, mirror=mirror, width_per_stream=width, path=str(path), shapes=shapes,
                params=sum(p.numel() for p in model.parameters()), artefact_floats=artefact_floats(path),
                macs=count_macs(path)["total"], state_floats=int(sum(np.prod(s) for s in shapes)),
                nodes=sum(n for op, n in audit["counts"].items() if op != "Constant"),
                off_list=audit["other"]))

stream = synthetic_stream(400 + WARMUP + RUNS)
sessions = [(e, Streamer(ort_session(e["path"], 1), e["shapes"])) for e in entries]
timings = {e["name"]: [] for e in entries}
for _ in range(ROUNDS):
    for entry, streamer in sessions:
        streamer.reset()
        timings[entry["name"]] += time_stream(streamer, stream, 400, WARMUP, RUNS)
for entry in entries:
    entry.update(summarise(timings[entry["name"]]))

# Which cost explains latency? Least squares: median_ms ~ a + b * nodes + c * MACs + d * state floats.
X = np.array([[1, e["nodes"], e["macs"] / 1e3, e["state_floats"] / 1e3] for e in entries], float)
y = np.array([e["median_ms"] for e in entries])
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
fit = dict(intercept_ms=coef[0], ms_per_node=coef[1], ms_per_kmac=coef[2], ms_per_kstate=coef[3],
           r2=1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum())
corr = {k: float(np.corrcoef([e[k] for e in entries], y)[0, 1]) for k in ("nodes", "macs", "state_floats")}

lines = ["| Graph | Params | Artefact floats | MACs/step | State floats | Nodes | Median (ms) | p90 (ms) |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
for e in entries:
    lines.append(f"| {e['name']} | {e['params']:,} | {e['artefact_floats']:,} | {e['macs']:,} | "
                 f"{e['state_floats']:,} | {e['nodes']} | {e['median_ms']:.4f} | {e['p90_ms']:.4f} |")
lines += ["", "Correlation of median latency with: " + ", ".join(f"{k} {v:+.3f}" for k, v in corr.items()),
          "Fit median_ms = {intercept_ms:.4f} + {ms_per_node:.5f}*nodes + {ms_per_kmac:.5f}*kMACs + "
          "{ms_per_kstate:.5f}*kState  (R^2 {r2:.3f})".format(**fit),
          "Off-list ops (all graphs): " + ", ".join(sorted({op for e in entries for op in e["off_list"]}))]
print("\n".join(lines))
(HERE / "m1_latency_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(HERE / "m1_latency_sweep.json").write_text(json.dumps(dict(entries=entries, fit=fit, correlation=corr),
                                                        indent=2, default=str))
