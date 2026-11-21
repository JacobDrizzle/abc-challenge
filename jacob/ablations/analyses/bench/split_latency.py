"""A/B latency: sep+angular through the reduced graph vs the mirror-split graph, one alternating bench session.

Run from the repo root. Uses bench.latency.session (same protocol as `python -m bench latency`) but writes the
record next to the ONNX files instead of bench/results/latency/, so no committed results change.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from bench.latency import session  # noqa: E402

V = Path("jacob/training/outputs/v8_bench")
models = {"v7_sep_angular_reduced": V/"sep_base3_addons_angular/model_fp32_stream.onnx",
          "v7_sep_angular_split": V/"sep_base3_addons_angular_split/model_fp32_stream.onnx"}
record = session(models, "jacob-amd", [1, 4], 200, 3000, 5)
out = V/"split_latency.json"
out.write_text(json.dumps(record, indent=1), encoding="utf-8")
for name, entry in record["models"].items():
    print(name, ", ".join(f"{t} thr median {v['median_ms'] * 1e3:.1f} us (p99 {v['p99_ms'] * 1e3:.1f})"
                          for t, v in entry["threads"].items()))
print("Wrote", out)
