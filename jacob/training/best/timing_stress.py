"""Timing stress test: validation streams sampled at uneven times, as the organisers' test stream may be ("roughly
~100 Hz (not exactly 100 Hz)", "some jitter, but mostly 100 Hz", 28 Oct).

Each validation segment is our exact 100 Hz grid. A stressed stream takes its samples at grid positions tau (10 ms
units) by SLERP between the neighbouring grid samples, with training.resampled's maths:
  rate_hz    the stream's nominal ticks are 100 / rate_hz grid steps apart
  drop       each tick after the first is lost with this probability; the stream skips it
  jitter_ms  each sample is taken uniform within +-jitter_ms of its tick
  repeat     with this probability a sample repeats the previous one (a late frame); the next is on time again
  noise_deg  sensor noise (added 9 Nov): after the timing above, every sample is turned by its own small random
             rotation, whose angle has this RMS (with_noise). The noise is independent from sample to sample, the
             worst case for a model that differences neighbouring samples; BOXRR's SLERP-resampled poses are smooth.
The model sees the samples as if evenly spaced, and the target is the stressed stream's 20th next sample, as the
organisers score it (so with noise_deg the target is noisy too: the damped baseline's change in the same row shows
what any model would pay). A segment ends at its last tick inside the grid (jittered positions are clipped to it), and
segments left shorter than one window are dropped. So every model scores the same windows within a scenario, but rate
and drop scenarios have their own windows: deltas across those rows are approximate.

python -m best.timing_stress --name <run> label=<checkpoint> ... [--reference <label>] [--data <h5>]
  [--stride 5] [--batch 1024] [--device cuda]   (from jacob/training; checkpoints relative to outputs/)
Writes outputs/timing_stress/<run>.json and .md (the 29 Oct runs are in outputs/v9_timing_stress). Reads only the
validation users of --data (default: the 100-user file, whose 15 validation users the 29 Oct runs scored).
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from baseline_pipeline import DEFAULT_H5, manifest_for
from jsonio import save_json
from tcnn.data import DeviceWindows, check_corpus
from tcnn.model import conjugate, qmul, quaternion_to_rotvec, rotvec_to_quaternion
from train_tcnn import reference_predictors, score_windows

from best.model import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026  # as the 28 Oct jitter runs, so their rows reproduce
SCENARIOS = {"exact 100 Hz grid": {},
             "jitter +-0.5 ms": {"jitter_ms": .5}, "jitter +-1 ms": {"jitter_ms": 1},
             "jitter +-1.5 ms": {"jitter_ms": 1.5}, "jitter +-2 ms": {"jitter_ms": 2}, "jitter +-3 ms": {"jitter_ms": 3},
             "rate 90 Hz": {"rate_hz": 90}, "rate 95 Hz": {"rate_hz": 95}, "rate 98 Hz": {"rate_hz": 98},
             "rate 102 Hz": {"rate_hz": 102}, "rate 105 Hz": {"rate_hz": 105}, "rate 110 Hz": {"rate_hz": 110},
             "rate 98 Hz + jitter +-1 ms": {"rate_hz": 98, "jitter_ms": 1},
             "repeats 1%": {"repeat": .01}, "drops 1%": {"drop": .01},
             "jitter +-2 ms + repeats 1%": {"jitter_ms": 2, "repeat": .01},
             "noise 0.005 deg": {"noise_deg": .005}, "noise 0.01 deg": {"noise_deg": .01},
             "noise 0.02 deg": {"noise_deg": .02}, "noise 0.05 deg": {"noise_deg": .05}}


def sample_positions(length, rng, rate_hz=100, jitter_ms=0, repeat=0, drop=0):
    """Grid positions (float64, in 10 ms steps) of a stressed stream over a segment of `length` grid samples."""
    step = 100 / rate_hz
    ticks = np.arange(math.floor((length - 1) / step + 1e-9) + 1)
    if drop:
        ticks = ticks[np.r_[True, rng.random(len(ticks) - 1) >= drop]]
    tau = ticks * step
    if jitter_ms:
        tau = tau + rng.uniform(-1, 1, len(tau)) * jitter_ms / 10
    if repeat:
        hit = np.flatnonzero(rng.random(len(tau)) < repeat)
        hit = hit[hit > 0]
        tau[hit] = tau[hit - 1]
    return np.clip(tau, 0, length - 1)


def slerp_at(q, tau):
    """q [L, 4] on the grid, tau [n] positions in [0, L - 1] -> [n, 4]. training.resampled's maths, for any n."""
    low = tau.floor().long().clamp(0, len(q) - 2)
    a, b = q[low], q[low + 1]
    step = quaternion_to_rotvec(qmul(conjugate(a), b))  # shortest rotation from a to b
    return F.normalize(qmul(a, rotvec_to_quaternion((tau - low)[:, None] * step)), dim=-1)


def with_noise(q, noise_deg, rng):
    """q [n, 4] float64 with every sample turned, in its own head frame, by an independent random rotation: a
    Gaussian rotation vector, the same in every direction, whose angle has an RMS of noise_deg."""
    sigma = math.radians(noise_deg) / math.sqrt(3)  # per axis: the angle's mean square is 3 sigma^2
    turn = torch.from_numpy(rng.standard_normal((len(q), 3)) * sigma).to(q.device)
    return F.normalize(qmul(q, rotvec_to_quaternion(turn)), dim=-1)


def stressed(buffer, bounds, span, scenario, seed):
    """(buffer, bounds) of the stressed streams; segments shorter than span samples are left out."""
    rng = np.random.default_rng(seed)
    timing = {key: value for key, value in scenario.items() if key != "noise_deg"}
    noise = scenario.get("noise_deg", 0)
    parts, kept, offset = [], [], 0
    for start, length in bounds:
        tau = torch.from_numpy(sample_positions(length, rng, **timing)).to(buffer.device)
        if len(tau) >= span:
            q = slerp_at(buffer[start:start + length].double(), tau)
            # The noise draws come after this segment's timing draws, so a scenario without noise is unchanged.
            parts.append((with_noise(q, noise, rng) if noise else q).float())
            kept.append((offset, len(tau)))
            offset += len(tau)
    return torch.cat(parts), kept


def tables(results, windows, reference):
    names = list(next(iter(results.values())))
    grid = results["exact 100 Hz grid"]
    head = "| Stream | Windows | " + " | ".join(names) + " |\n| --- | ---: |" + " ---: |" * len(names)
    rows = [f"| exact 100 Hz grid | {windows['exact 100 Hz grid']:,} | " +
            " | ".join(f"{grid[n]:.4f}" for n in names) + " |"]
    rows += [f"| {label} | {windows[label]:,} | " + " | ".join(f"{row[n] - grid[n]:+.4f}" for n in names) + " |"
             for label, row in results.items() if label != "exact 100 Hz grid"]
    text = ["Step-20 mean geodesic error (degrees) on the exact grid; the other rows are each model's change.", "",
            head, *rows]
    if reference:
        others = [n for n in names if n != reference]
        text += ["", f"Each model minus {reference} on the same stream (negative: better than {reference}).", "",
                 "| Stream | " + " | ".join(others) + " |", "| --- |" + " ---: |" * len(others)]
        text += [f"| {label} | " + " | ".join(f"{row[n] - row[reference]:+.4f}" for n in others) + " |"
                 for label, row in results.items()]
    return "\n".join(text) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", help="label=checkpoint, the checkpoint relative to outputs/")
    p.add_argument("--name", required=True, help="Output file name in outputs/timing_stress/")
    p.add_argument("--reference", help="A model label: also tabulate every model minus this one")
    p.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), help="Default: all")
    p.add_argument("--data", type=Path, default=DEFAULT_H5, help="ray/ export HDF5; its validation users are scored")
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    models = dict(spec.split("=", 1) for spec in args.models)
    if args.reference and args.reference not in models:
        p.error("--reference must be one of the model labels")
    device = torch.device(args.device)
    audit = check_corpus(args.data, manifest_for(args.data))
    data = DeviceWindows(args.data, [r for r in audit["segments"] if r["split"] == "validation"], device)
    grid_buffer, grid_bounds, span = data.buffer, data.bounds, data.context + data.horizon
    predictors = {"damped velocity (baseline)": reference_predictors(data.horizon)["damped_velocity"]}
    for label, path in models.items():
        predictors[label] = load_checkpoint(ROOT/"outputs"/path)[0].to(device).eval()
    same, _ = stressed(grid_buffer, grid_bounds, span, {}, SEED)
    print("exact-grid resample max |dq|:", float((same - grid_buffer).abs().max()), flush=True)

    results, windows = {}, {}
    for label in args.scenarios or SCENARIOS:
        start = time.time()
        scenario = SCENARIOS[label]
        data.buffer, data.bounds = stressed(grid_buffer, grid_bounds, span, scenario, SEED) if scenario else \
            (grid_buffer, grid_bounds)
        scores = score_windows(predictors, data.windows(args.stride, args.batch), data.horizon)
        results[label] = {name: s["mean_endpoint_deg"] for name, s in scores.items()}
        windows[label] = next(iter(scores.values()))["windows"]
        print(f"{label} ({time.time() - start:.0f} s, {windows[label]:,} windows)",
              {k: round(v, 4) for k, v in results[label].items()}, flush=True)
    data.buffer, data.bounds = grid_buffer, grid_bounds

    report = tables(results, windows, args.reference) if "exact 100 Hz grid" in results else ""
    folder = ROOT/"outputs/timing_stress"
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder/f"{args.name}.json", {"data": str(args.data), "stride": args.stride, "seed": SEED,
                                           "models": models, "scenarios": {k: SCENARIOS[k] for k in results},
                                           "windows": windows, "results": results, "reference": args.reference})
    (folder/f"{args.name}.md").write_text(f"# Timing stress: {args.name}\n\n" + report, encoding="utf-8")
    print("\n" + report + "\nwrote", folder/f"{args.name}.json", flush=True)


if __name__ == "__main__":
    main()
