"""Track A accuracy: sep_base3_addons + angular (seeds 42-44) with exact maps vs the lean maps its v9 graph computes.

Validation, stride 1 (657,189 windows, 15 users), step-20 error as train_tcnn scores it. Paired per user, with a
user bootstrap. Also the step-20 forecast angle distribution, which sets the output map's approximation error.
Read-only on the repo; writes trackA_accuracy.json next to this script.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

TRAINING = Path(r"<repo>\jacob\training")
sys.path.insert(0, str(TRAINING))

from baseline_pipeline import DEFAULT_H5, manifest_for  # noqa: E402
from tcnn.data import DeviceWindows, check_corpus  # noqa: E402
from train_tcnn import geodesic_degrees_torch  # noqa: E402
from experimental.v9_lean.model import deployed, load_checkpoint  # noqa: E402

OUT = Path(__file__).resolve().parent / "trackA_accuracy.json"
FOLDER = TRAINING / "outputs/v8_recipe_sep_base3_addons/angular_loss"
SEEDS = (42, 43, 44)


def main():
    began = time.time()
    audit = check_corpus(DEFAULT_H5, manifest_for(DEFAULT_H5))
    rows = [r for r in audit["segments"] if r["split"] == "validation"]
    windows = DeviceWindows(DEFAULT_H5, rows, "cuda", 400, 20)
    users = sorted({r["user"] for r in rows})
    owner = torch.from_numpy(np.concatenate([np.full(length - 419, users.index(r["user"]))
                                             for r, (_, length) in zip(rows, windows.bounds)])).cuda()
    models = {}
    for seed in SEEDS:
        models[f"exact_s{seed}"] = load_checkpoint(FOLDER / f"seed_{seed}/best.pt")[0].cuda()
        models[f"lean_s{seed}"] = deployed(load_checkpoint(FOLDER / f"seed_{seed}/best.pt")[0]).cuda()
    sums = {n: torch.zeros(len(users), dtype=torch.float64, device="cuda") for n in models}
    gap = {s: [] for s in SEEDS}
    angles, counts, seen = [], torch.zeros(len(users), dtype=torch.float64, device="cuda"), 0
    with torch.inference_mode():
        for history, target in windows.windows(1, 4096):
            who = owner[seen:seen + len(history)]
            seen += len(history)
            counts += torch.bincount(who, minlength=len(users)).double()
            predictions = {n: m(history) for n, m in models.items()}
            for n, p in predictions.items():
                sums[n].index_add_(0, who, geodesic_degrees_torch(p[:, -1], target[:, -1]))
            for s in SEEDS:
                gap[s].append(geodesic_degrees_torch(predictions[f"lean_s{s}"][:, -1], predictions[f"exact_s{s}"][:, -1]).float().cpu())
            angles.append(geodesic_degrees_torch(predictions["exact_s44"][:, -1], history[:, -1]).float().cpu())
    counts = counts.cpu().numpy()
    angles = torch.cat(angles).numpy()
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(users), size=(10000, len(users)))
    result = {"windows": seen, "users": len(users), "forecast_angle_step20_deg": {
        q: float(np.percentile(angles, p)) for q, p in (("p50", 50), ("p90", 90), ("p99", 99), ("p99.9", 99.9), ("max", 100))}}
    for s in SEEDS:
        e, l = (sums[f"{k}_s{s}"].cpu().numpy() for k in ("exact", "lean"))
        diff = l - e
        boot = diff[draws].sum(1) / counts[draws].sum(1)
        g = torch.cat(gap[s]).numpy()
        result[f"seed_{s}"] = {"exact": float(e.sum() / counts.sum()), "lean": float(l.sum() / counts.sum()),
                               "delta": float(diff.sum() / counts.sum()),
                               "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                               "users_worse": int((diff > 0).sum()),
                               "lean_vs_exact_forecast_deg": {"mean": float(g.mean()), "p99": float(np.percentile(g, 99)),
                                                              "max": float(g.max())}}
    result["seconds"] = time.time() - began
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
