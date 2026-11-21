"""Timing jitter (organiser answer 2, 28 Oct: the test stream is "roughly ~100 Hz", "some jitter, but mostly 100 Hz").

Each validation segment is our exact 100 Hz grid. A jittered stream takes its k-th sample at time k + e_k (in 10 ms
units) by SLERP between the neighbouring grid samples, with e_k uniform in +-J ms; a glitch makes a sample repeat the
previous one (a late frame), and the next is on time again. The model sees the jittered samples as if evenly spaced,
and the target is the jittered stream's 20th next sample, as the organisers score it. Every model scores the same
windows of the same jittered buffer. Read-only; CPU only (the GPU is training); writes JSON next to this script.
Usage (from jacob/training): python <this script> [stride] [threads] [device] [model label substring ...]
"""
import json
from pathlib import Path
import sys
import time

TRAINING = Path(r"<repo>\jacob\training")
sys.path.insert(0, str(TRAINING))

import numpy as np
import torch
from torch.nn import functional as F

from baseline_pipeline import DEFAULT_H5, manifest_for
from experimental.v9_lean.model import load_checkpoint
from tcnn.data import DeviceWindows, check_corpus
from tcnn.model import conjugate, qmul
from train_tcnn import reference_predictors, score_windows

OUT = TRAINING / "outputs"
MODELS = {"sep+angular s44 (current best)": "v8_recipe_sep_base3_addons/angular_loss/seed_44/best.pt",
          "v9 control s42": "v9_screen/control/seed_42/best.pt",
          "rf_397 s42": "v9_screen/rf_397/seed_42/best.pt",
          "sampling_window s42": "v9_screen/sampling_window/seed_42/best.pt",
          "native_maps s42": "v9_screen/native_maps/seed_42/best.pt",
          "k5_4blocks s42": "v9_screen/k5_4blocks/seed_42/best.pt",
          "k8_3blocks_w32 s42": "v9_screen/k8_3blocks_w32/seed_42/best.pt",
          "k8 jitter_3ms s42": "v9_jitter/jitter_3ms/seed_42/best.pt",
          "k8_combined s42": "v9_confirm/k8_combined/seed_42/best.pt",
          "k5_combined s42": "v9_confirm/k5_combined/seed_42/best.pt"}
# label -> (uniform jitter half-width in ms, glitch probability)
SCENARIOS = {"exact 100 Hz grid": (0, 0), "jitter +-1 ms": (1, 0), "jitter +-2 ms": (2, 0), "jitter +-3 ms": (3, 0),
             "glitches 1%": (0, .01), "jitter +-2 ms + glitches 1%": (2, .01)}


def slerp_at(q, tau):
    """q [L, 4] unit quaternions on the grid; tau [L] positions in [0, L-1] -> [L, 4], by SLERP."""
    lo = tau.floor().long().clamp(0, len(q) - 2)
    frac = (tau - lo).clamp(0, 1)[:, None]
    a, b = q[lo], q[lo + 1]
    b = torch.where((a * b).sum(-1, keepdim=True) < 0, -b, b)
    d = qmul(conjugate(a), b)
    angle = 2 * torch.atan2(d[:, 1:].norm(dim=-1, keepdim=True), d[:, :1])
    half = frac * angle / 2
    return F.normalize(qmul(a, torch.cat((half.cos(), half.sin() * F.normalize(d[:, 1:], dim=-1)), -1)), dim=-1)


def sample_times(length, jitter_ms, glitch, rng):
    tau = np.arange(length, dtype=np.float64)
    if jitter_ms:
        tau += rng.uniform(-1, 1, length) * jitter_ms / 10
    if glitch:
        hit = np.flatnonzero(rng.random(length) < glitch)
        hit = hit[hit > 0]
        tau[hit] = tau[hit - 1]  # a late frame: the previous pose again
    return np.clip(tau, 0, length - 1)


def jittered(buffer, bounds, jitter_ms, glitch, seed):
    rng = np.random.default_rng(seed)
    out = buffer.clone()
    for offset, length in bounds:
        q = buffer[offset:offset + length].double()
        tau = torch.from_numpy(sample_times(length, jitter_ms, glitch, rng)).to(q.device)
        out[offset:offset + length] = slerp_at(q, tau).float()
    return out


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    device = torch.device(sys.argv[3] if len(sys.argv) > 3 else "cpu")
    wanted = sys.argv[4:]
    audit = check_corpus(DEFAULT_H5, manifest_for(DEFAULT_H5))
    data = DeviceWindows(DEFAULT_H5, [r for r in audit["segments"] if r["split"] == "validation"], device)
    original = data.buffer.clone()
    predictors = {"damped velocity (baseline)": reference_predictors(20)["damped_velocity"]}
    for name, path in MODELS.items():
        if (OUT / path).exists() and (not wanted or any(w in name for w in wanted)):
            predictors[name] = load_checkpoint(OUT / path)[0].to(device).eval()
    # The resampler itself, at zero jitter, must give back the grid.
    same = jittered(original, data.bounds, 0, 0, 0)
    print("zero-jitter resample max |dq|:", float((same - original).abs().max()), flush=True)
    results = {}
    for label, (jitter_ms, glitch) in SCENARIOS.items():
        start = time.time()
        data.buffer = jittered(original, data.bounds, jitter_ms, glitch, seed=2026) if (jitter_ms or glitch) else original
        scores = score_windows(predictors, data.windows(stride, 4096), 20)
        results[label] = {name: s["mean_endpoint_deg"] for name, s in scores.items()}
        print(f"{label} ({time.time() - start:.0f} s)", {k: round(v, 4) for k, v in results[label].items()}, flush=True)
    data.buffer = original
    names = list(predictors)
    base = results["exact 100 Hz grid"]
    lines = ["| Stream | " + " | ".join(names) + " |", "| --- |" + " ---: |" * len(names),
             "| exact 100 Hz grid | " + " | ".join(f"{base[n]:.4f}" for n in names) + " |"]
    for label, row in results.items():
        if label != "exact 100 Hz grid":
            lines.append(f"| {label} | " + " | ".join(f"{row[n] - base[n]:+.4f}" for n in names) + " |")
    table = "\n".join(lines)
    print("\nStep-20 mean (deg) on the exact grid; other rows are the change.\n" + table)
    out = Path(__file__).with_name("jitter_robustness" + ("_subset" if wanted else "") + ".json")
    out.write_text(json.dumps({"stride": stride, "windows": next(iter(scores.values()))["windows"],
                               "models": {n: MODELS[n] for n in predictors if n in MODELS},
                               "scenarios": SCENARIOS, "results": results, "table_md": table}, indent=1),
                   encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
