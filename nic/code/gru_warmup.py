"""Warm-up of a stateful graph: how many calls after zero states until it scores like the warm stream.

Warm reference: the bench's from-start protocol (zero states at the segment's first sample, so the state
has seen everything). Cold runs: zero states at a full 400-sample window, as the organisers' harness
starts, then one sample per call; one cold run starts every --stride windows. gap(k) is the cold minus
the warm step-20 error on the same window, k calls after the reset, averaged over all cold runs.

  .venv-bench/bin/python nic_scratch/warmup/gru_warmup.py nic/models/group_a_v2/s45/gru_stateful_window.onnx
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.data import Benchmark  # noqa: E402
from bench.drivers import OnnxModel  # noqa: E402
from bench.geometry import geodesic_degrees, renormalise  # noqa: E402

C, H = 400, 20
CHECKPOINTS = [1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 750, 1000]


def step20_error(pred, q, anchors):
    """pred [n, 20, 4] raw outputs for these anchors -> step-20 geodesic error in degrees, as the bench scores it."""
    return geodesic_degrees(renormalise(pred[:, -1]), q[anchors + H])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="stateful ONNX graph, relative to the repo root")
    ap.add_argument("--bench", default="bench/definitions/group-a-v2-test20.json")
    ap.add_argument("--h5", default="data/qualified_head_100hz_20261026_131953.h5")
    ap.add_argument("--calls", type=int, default=1000, help="length of each cold run")
    ap.add_argument("--stride", type=int, default=1000, help="windows between cold-run starts")
    ap.add_argument("--segments", type=int, default=None, help="only the first N segments (quick test)")
    ap.add_argument("--out", default="nic_scratch/warmup/warmup.json")
    a = ap.parse_args()
    model = OnnxModel(ROOT / a.model)
    if not model.stateful:
        sys.exit(f"{a.model} has no state_in/state_out tensors")
    # Per call index k (0-based): sums over cold runs. "start" = only the runs reset at the segment's first full window.
    sums = {key: np.zeros(a.calls) for key in ("cold", "warm", "n", "start_gap", "start_n")}
    t0 = time.time()
    with Benchmark(ROOT / a.bench, ROOT / a.h5) as b:
        segments = b.segments[:a.segments]
        for i, s in enumerate(segments):
            q = b.segment(s)
            q32 = q.astype(np.float32)
            anchors = np.arange(C - 1, len(q) - H)
            warm = step20_error(np.concatenate(list(model.predict_segment(q32, C, H, "from-start"))), q, anchors)
            for start in range(0, len(anchors), a.stride):
                run = anchors[start:start + a.calls]
                model.reset()
                cold = step20_error(np.stack([model.step(q32[t - C + 1:t + 1][None])[0] for t in run]), q, run)
                n = len(run)
                sums["cold"][:n] += cold
                sums["warm"][:n] += warm[start:start + n]
                sums["n"][:n] += 1
                if start == 0:
                    sums["start_gap"][:n] += cold - warm[:n]
                    sums["start_n"][:n] += 1
            print(f"segment {i + 1}/{len(segments)}  {time.time() - t0:.0f} s", flush=True)
    cold, warm = sums["cold"] / sums["n"], sums["warm"] / sums["n"]
    gap, start_gap = cold - warm, sums["start_gap"] / np.maximum(sums["start_n"], 1)
    # First k from which the gap stays under each threshold for the rest of the run.
    settled = {f"{th:g}": int(np.max(np.nonzero(gap >= th)[0], initial=-1)) + 2 for th in (0.1, 0.05, 0.02, 0.01)}
    rows = [{"k": k, "cold": cold[k - 1], "warm": warm[k - 1], "gap": gap[k - 1], "start_gap": start_gap[k - 1],
             "runs": int(sums["n"][k - 1])} for k in CHECKPOINTS if k <= a.calls]
    print(f"\n{a.model}: {len(segments)} segments, cold runs of {a.calls} calls every {a.stride} windows")
    print(f"{'k':>5} {'cold':>8} {'warm':>8} {'gap':>8} {'gap@start':>10} {'runs':>6}")
    for r in rows:
        print(f"{r['k']:>5} {r['cold']:>8.4f} {r['warm']:>8.4f} {r['gap']:>+8.4f} {r['start_gap']:>+10.4f} {r['runs']:>6}")
    print("gap stays below threshold (deg) from call k:", settled)
    Path(ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "bench": a.bench, "calls": a.calls, "stride": a.stride, "segments": len(segments),
               "settled_from_call": settled, "checkpoints": rows,
               "gap_per_call": gap.tolist(), "start_gap_per_call": start_gap.tolist()},
              open(ROOT / a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
