"""Angular-loss confirm: angular vs quaternion loss on seeds 42-44, per seed and per validation user.

Read-only: reads each seed's full_validation/, writes one JSON next to this script.
Usage (from jacob/training): python <this script>
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

PAIRS = {"separable_base3_addons": ("outputs/v7_extra/separable_base3_addons",
                                    "outputs/v8_recipe_sep_base3_addons/angular_loss"),
         "base3_5_addons": ("outputs/v7_confirm/base3_5_addons",
                            "outputs/v8_recipe_base3_5_addons/angular_loss")}
SEEDS = (42, 43, 44)
MODEL = "tiny_tcnn"


def seed_result(folder, seed):
    fv = Path(folder) / f"seed_{seed}" / "full_validation"
    res = json.loads((fv / "results.json").read_text(encoding="utf-8"))
    return res["models"][MODEL], np.load(fv / f"endpoint_{MODEL}.npy"), np.load(fv / "window_index.npy")


def main():
    rng = np.random.default_rng(0)
    report, seed_deltas = {}, []
    for arch, (base_dir, angular_dir) in PAIRS.items():
        rows, user_base, user_ang, index = {}, [], [], None
        for seed in SEEDS:
            stats_seed = {}
            for kind, folder in (("quaternion", base_dir), ("angular", angular_dir)):
                res, err, idx = seed_result(folder, seed)
                index = idx if index is None else index
                assert np.array_equal(idx, index), f"{folder} seed {seed}: window order differs"
                stats_seed[kind] = {"mean": res["mean_endpoint_deg"], "median": float(np.median(err)),
                                    "p99": float(np.percentile(err, 99)), "step10": res["mean_deg_by_horizon"][9],
                                    "by_user": {u: v["mean_endpoint_deg"] for u, v in res["by_user"].items()},
                                    "windows": {u: v["windows"] for u, v in res["by_user"].items()}}
            users = sorted(stats_seed["quaternion"]["by_user"])
            user_base.append([stats_seed["quaternion"]["by_user"][u] for u in users])
            user_ang.append([stats_seed["angular"]["by_user"][u] for u in users])
            weights = np.array([stats_seed["quaternion"]["windows"][u] for u in users], dtype=float)
            rows[seed] = {k: {m: v for m, v in s.items() if m not in ("by_user", "windows")}
                          for k, s in stats_seed.items()}
            seed_deltas.append(stats_seed["angular"]["mean"] - stats_seed["quaternion"]["mean"])
        d = np.array(user_ang).mean(0) - np.array(user_base).mean(0)
        picks = rng.integers(0, len(users), size=(20000, len(users)))
        boot = (d[picks] * weights[picks]).sum(1) / weights[picks].sum(1)
        summary = {m: {k: float(np.mean([rows[s][k][m] for s in SEEDS])) for k in ("quaternion", "angular")}
                   for m in ("mean", "median", "p99", "step10")}
        report[arch] = {"per_seed": rows, "three_seed": summary,
                        "users_better_3seed_mean": int((d < 0).sum()), "users": len(users),
                        "user_bootstrap_ci95_3seed": [float(np.percentile(boot, 2.5)),
                                                      float(np.percentile(boot, 97.5))],
                        "per_user_delta_3seed": dict(zip(users, map(float, d)))}
        print(f"\n== {arch}")
        for s in SEEDS:
            q, a = rows[s]["quaternion"]["mean"], rows[s]["angular"]["mean"]
            print(f"  seed {s}: {q:.4f} -> {a:.4f}  ({a - q:+.4f})")
        for m, v in summary.items():
            print(f"  3-seed {m:<7}: {v['quaternion']:.4f} -> {v['angular']:.4f}  ({v['angular'] - v['quaternion']:+.4f})")
        r = report[arch]
        print(f"  users better (3-seed mean): {r['users_better_3seed_mean']}/{r['users']}, user-bootstrap CI95 "
              f"[{r['user_bootstrap_ci95_3seed'][0]:+.4f}, {r['user_bootstrap_ci95_3seed'][1]:+.4f}]")
    t = stats.ttest_1samp(seed_deltas, 0.0)
    report["all_seed_pairs"] = {"deltas": seed_deltas, "mean": float(np.mean(seed_deltas)),
                                "sd": float(np.std(seed_deltas, ddof=1)), "t": float(t.statistic),
                                "p_two_sided": float(t.pvalue)}
    print(f"\nall 6 seed pairs: mean {np.mean(seed_deltas):+.4f}, sd {np.std(seed_deltas, ddof=1):.4f}, "
          f"t={t.statistic:.2f}, p={t.pvalue:.4f}")
    out = Path(__file__).with_name("angular_confirm_paired.json")
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
