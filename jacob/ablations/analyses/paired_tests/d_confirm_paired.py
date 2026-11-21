"""Stage D: each confirmed candidate vs the reference (sep_base3_addons + angular) on seeds 42-44, per seed and per
validation user (3-seed mean per user, window-weighted user bootstrap). Adapted from the 27 Oct
angular_confirm_paired.py. Read-only: reads each seed's full_validation/, writes one JSON next to this script.
Usage (from jacob/training): python <this script> <reference folder> <candidate folder> [<candidate folder> ...]
"""
import json
from pathlib import Path
import sys

import numpy as np

SEEDS = (42, 43, 44)
MODEL = "tiny_tcnn"


def seed_result(folder, seed):
    fv = Path(folder) / f"seed_{seed}" / "full_validation"
    res = json.loads((fv / "results.json").read_text(encoding="utf-8"))["models"][MODEL]
    return res, np.load(fv / f"endpoint_{MODEL}.npy"), np.load(fv / "window_index.npy")


def per_seed(folder, index):
    rows = {}
    for seed in SEEDS:
        res, err, idx = seed_result(folder, seed)
        assert np.array_equal(idx, index), f"{folder} seed {seed}: window order differs"
        rows[seed] = {"mean": res["mean_endpoint_deg"], "median": float(np.median(err)),
                      "p99": float(np.percentile(err, 99)), "by_user": {u: v["mean_endpoint_deg"] for u, v in
                                                                         res["by_user"].items()},
                      "windows": {u: v["windows"] for u, v in res["by_user"].items()}}
    return rows


def main():
    rng = np.random.default_rng(0)
    reference_dir, candidates = sys.argv[1], sys.argv[2:]
    index = seed_result(reference_dir, SEEDS[0])[2]
    ref = per_seed(reference_dir, index)
    users = sorted(ref[SEEDS[0]]["by_user"])
    weights = np.array([ref[SEEDS[0]]["windows"][u] for u in users], dtype=float)
    ref_user = np.array([[ref[s]["by_user"][u] for u in users] for s in SEEDS]).mean(0)
    ref_mean = float(np.mean([ref[s]["mean"] for s in SEEDS]))
    report = {"reference": {"folder": reference_dir, "per_seed": {s: ref[s]["mean"] for s in SEEDS},
                            "mean": ref_mean}}
    print(f"reference {reference_dir}: " + ", ".join(f"{ref[s]['mean']:.4f}" for s in SEEDS) + f"  mean {ref_mean:.4f}")
    for folder in candidates:
        rows = per_seed(folder, index)
        user = np.array([[rows[s]["by_user"][u] for u in users] for s in SEEDS]).mean(0)
        d = user - ref_user
        picks = rng.integers(0, len(users), size=(20000, len(users)))
        boot = (d[picks] * weights[picks]).sum(1) / weights[picks].sum(1)
        mean = float(np.mean([rows[s]["mean"] for s in SEEDS]))
        report[folder] = {"per_seed": {s: rows[s]["mean"] for s in SEEDS}, "mean": mean,
                          "sd": float(np.std([rows[s]["mean"] for s in SEEDS], ddof=1)),
                          "delta_vs_reference": mean - ref_mean,
                          "seed_deltas": {s: rows[s]["mean"] - ref[s]["mean"] for s in SEEDS},
                          "median_3seed": float(np.mean([rows[s]["median"] for s in SEEDS])),
                          "p99_3seed": float(np.mean([rows[s]["p99"] for s in SEEDS])),
                          "users_better": int((d < 0).sum()), "users": len(users),
                          "user_bootstrap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                          "per_user_delta": dict(zip(users, map(float, d)))}
        r = report[folder]
        print(f"{folder}: " + ", ".join(f"{rows[s]['mean']:.4f}" for s in SEEDS) +
              f"  mean {mean:.4f} (sd {r['sd']:.4f})  vs reference {r['delta_vs_reference']:+.4f}  "
              f"users better {r['users_better']}/{r['users']}  CI95 [{r['user_bootstrap_ci95'][0]:+.4f}, "
              f"{r['user_bootstrap_ci95'][1]:+.4f}]")
    out = Path(__file__).with_name("d_confirm_paired.json")
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
