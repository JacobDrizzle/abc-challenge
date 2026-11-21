"""B screen: each recipe variant vs its own control, paired per user on the 15 validation users.

Read-only: reads full_validation/ and history.json of each run, writes one JSON next to this script.
Usage: python <this script> <outputs dir> <suite name> [<suite name> ...]
"""
import json
from pathlib import Path
import sys

import numpy as np

MODEL = "tiny_tcnn"
RNG = np.random.default_rng(0)
BOOT = 20000


def load(run):
    fv = run / "full_validation"
    res = json.loads((fv / "results.json").read_text(encoding="utf-8"))["models"][MODEL]
    hist = json.loads((run / "history.json").read_text(encoding="utf-8"))
    errors = np.load(fv / f"endpoint_{MODEL}.npy")
    index = np.load(fv / "window_index.npy")
    return res, hist, errors, index


def paired(control, variant):
    users = sorted(control["by_user"])
    w = np.array([control["by_user"][u]["windows"] for u in users], dtype=float)
    c = np.array([control["by_user"][u]["mean_endpoint_deg"] for u in users])
    v = np.array([variant["by_user"][u]["mean_endpoint_deg"] for u in users])
    d = v - c
    picks = RNG.integers(0, len(users), size=(BOOT, len(users)))
    boot = (d[picks] * w[picks]).sum(1) / w[picks].sum(1)
    signs = RNG.choice([-1.0, 1.0], size=(BOOT, len(users)))
    observed = (d * w).sum() / w.sum()
    null = (signs * d * w).sum(1) / w.sum()
    return {"window_weighted_delta": float(observed),
            "equal_user_delta": float(d.mean()),
            "users_better": int((d < 0).sum()), "users": len(users),
            "user_bootstrap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "sign_flip_p": float((np.abs(null) >= abs(observed) - 1e-12).mean()),
            "per_user_delta": dict(zip(users, map(float, d)))}


def main():
    outputs = Path(sys.argv[1])
    report = {}
    for suite in sys.argv[2:]:
        root = outputs / suite
        variants = sorted(p.parent.parent.name for p in root.glob("*/seed_42/full_validation"))
        base_res, base_hist, base_err, base_idx = load(root / "control" / "seed_42")
        rows = {}
        for name in ["control"] + [v for v in variants if v != "control"]:
            res, hist, err, idx = load(root / name / "seed_42")
            assert np.array_equal(idx, base_idx), f"{suite}/{name}: window order differs from control"
            best = min(hist, key=lambda h: h["validation_endpoint_deg"])
            row = {"mean": res["mean_endpoint_deg"], "equal_user": res["equal_user_mean_endpoint_deg"],
                   "median": float(np.median(err)), "p90": float(np.percentile(err, 90)),
                   "p99": float(np.percentile(err, 99)),
                   "step10_100ms": res["mean_deg_by_horizon"][9],
                   "best_epoch": best["epoch"], "epochs_run": len(hist),
                   "last_epoch_val_stride5": hist[-1]["validation_endpoint_deg"],
                   "train_end_deg_last": hist[-1]["training_endpoint_deg"],
                   "window_level_share_better": float((err < base_err).mean())}
            if name != "control":
                row["paired"] = paired(base_res, res)
            rows[name] = row
        report[suite] = rows
        print(f"\n== {suite}")
        print(f"{'variant':<13}{'mean':>8}{'d_mean':>8}{'eq_user':>9}{'median':>8}{'p90':>8}{'p99':>8}"
              f"{'100ms':>8}{'best@':>7}{'win':>7}{'CI95 (user bootstrap)':>24}{'p':>7}")
        c = rows["control"]
        for name, r in rows.items():
            p = r.get("paired")
            extra = (f"{p['users_better']:>3}/{p['users']:<3}"
                     f"  [{p['user_bootstrap_ci95'][0]:+.4f}, {p['user_bootstrap_ci95'][1]:+.4f}]"
                     f"{p['sign_flip_p']:>7.3f}") if p else ""
            print(f"{name:<13}{r['mean']:>8.4f}{r['mean'] - c['mean']:>+8.4f}{r['equal_user']:>9.4f}"
                  f"{r['median']:>8.4f}{r['p90']:>8.3f}{r['p99']:>8.3f}{r['step10_100ms']:>8.4f}"
                  f"{r['best_epoch']:>4}/{r['epochs_run']:<3}{extra}")
    out = Path(__file__).with_name("b_screen_paired.json")
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
