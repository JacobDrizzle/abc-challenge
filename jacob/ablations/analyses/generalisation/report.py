"""Step 2 of the generalisation report: how well the two k8 models do on users they never saw.

Reads the per-segment sums that score_users.py wrote (scores/*.json), both data manifests, the bench's test20 result
JSONs and the ablation ledger (runs.csv). Writes generalisation_report.json and generalisation_report.md next to this
script. CPU only. Every verification check runs first, and the report is not written if one fails.

  A  Unseen-user accuracy on test20 and test160 (and the 100-user model's 965-user unseen pool)
  B  Seen against unseen: the per-user model/damped ratio by role (trained, selection, unseen)
  C  Unseen users broken down by headset family, source rate and motion intensity
  D  Validation to test: the ledger models' val15 against test20, and k8-800 minus k8-100 on every cohort

Run from the repo root with a Python that has numpy and scipy:
    python jacob/ablations/analyses/generalisation/report.py
"""
import argparse
import csv
import json
from collections import namedtuple
from pathlib import Path
import sys

import numpy as np
from scipy import stats as scipy_stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from bench import stats  # noqa: E402  (paired per-user bootstrap, the one the leaderboard uses)

DATA = REPO / "data" / "h5"
OLD_MANIFEST = DATA / "manifest_20261026_131953.json"
MERGED_MANIFEST = DATA / "manifest_merged_20261101_140915.json"
BENCH = REPO / "bench" / "results" / "group-a-v2-test20"
LEDGER = HERE.parents[1] / "runs.csv"
OUTPUTS = REPO / "jacob" / "training" / "outputs"

MODELS = {
    "k8_100": {"labels": {42: "k8_100_s42", 43: "k8_100_s43", 44: "k8_100_s44"}, "selected": 42, "trained_on": "old",
               "selection_json": OUTPUTS / "v9_confirm/k8_combined/selection.json",
               "text": "k8_combined trained on the 100-user file (v9_confirm, selected seed 42)"},
    "k8_800": {"labels": {42: "k8_800_s42", 43: "k8_800_s43", 44: "k8_800_s44"}, "selected": 44,
               "trained_on": "merged",
               "selection_json": OUTPUTS / "v10_k8_800users_full_pass/full_pass/selection.json",
               "text": "k8_combined trained on the 800-user file, one full pass (v10_k8_800users_full_pass, selected "
                       "seed 44)"},
}
CONFIRM_800 = OUTPUTS / "v10_k8_800users_full_pass_confirm/confirmation.json"
BASELINES = {"hold_last": "hold_last", "constant_velocity": "constant_velocity_1step",
             "damped_velocity": "damped_velocity"}  # score_users name -> bench result name
TEST20_WINDOWS = 931_051

# Section D: ledger rows (suite, variant; the selected row) and the bench result scored from that same checkpoint.
# Sources: bench/models.json notes, outputs/v8_bench/*/latency.json and analyses/bench/v9_bench_export.py.
# Graph variants of the same weights (padded, A4, split, lean re-exports) are left out, so each model counts once.
LEDGER_TO_BENCH = [
    ("v5_group_a_v2", "control", "v5_tcnn_ga2"),
    ("v6_group_a_v2", "control", "v6_tcnn_ga2"),
    ("v7_extra", "separable_base3_addons", "v7_sep_base3_addons_ga2"),
    ("v7_confirm", "base3_5_addons", "v7_base3_5_addons_ga2"),
    ("v8_recipe_sep_base3_addons", "angular_loss", "v7_sep_base3_addons_angular_ga2"),
    ("v8_recipe_base3_5_addons", "angular_loss", "v7_base3_5_addons_angular_ga2"),
    ("v9_confirm", "k5_combined", "v9_k5_combined_ga2"),
    ("v9_confirm", "k8_combined", "v9_k8_combined_ga2"),
    ("v9_jitter", "jitter_3ms", "v9_k8_jitter3_s42_ga2"),
]

# Section C. An analysis choice for the team to check: exact manifest strings to families. A string not listed here
# stops the report, so a new export can't silently fall into the wrong group.
HEADSET_FAMILIES = {
    "Quest 2": ["Oculus Quest 2", "Oculus Quest2", "Oculus_Link_Quest_2"],
    "Quest 1": ["Oculus Quest"],
    "Rift CV1": ["Oculus Rift CV1", "Rift_CV1"],
    "Rift S": ["Oculus Rift S", "Rift_S"],
    "Index": ["Index"],
    "Vive family": ["Vive", "Vive MV", "Vive MV.", "Vive. MV", "VIVE_Pro MV", "VIVE Pro 2", "Vive Elite", "vive_cosmos"],
    "Windows MR": ["HP Reverb Virtual Reality Headset G20", "HP Reverb G2 Omnicept0", "HP Reverb VR Headset VR1000-2xxx0",
                   "Samsung Windows Mixed Reality 800ZAA0", "Samsung Windows Mixed Reality 800ZBA0",
                   "DELL VISOR VR1180", "Acer AH1010", "Lenovo Explorer0"],
    "Pico": ["PICO 4", "Pico Neo 2", "Pico Neo 3"],
    "Pimax": ["Pimax 5K Plus", "Pimax 5K Super"],
    "Other": ["3GLASSES S20", "Miramar", "hedy"],
    "Unknown": ["unknown"],
}
FAMILY_OF = {name: family for family, names in HEADSET_FAMILIES.items() for name in names}
MIN_GROUP_USERS = 5
DRAWS = 10_000

Users = namedtuple("Users", "ids mean weight")  # per-user mean error and weight (all windows of the user)


# ---------------------------------------------------------------- statistics

def weighted_mean(values, weights):
    return float((values * weights).sum() / weights.sum())


def bootstrap_ci(values, weights=None, draws=DRAWS, seed=0):
    """95% interval of the weighted mean, resampling users (windows of one user are strongly correlated)."""
    weights = np.ones_like(values) if weights is None else weights
    picks = np.random.default_rng(seed).integers(0, len(values), (draws, len(values)))
    boots = (values[picks] * weights[picks]).sum(1) / weights[picks].sum(1)
    return [float(x) for x in np.percentile(boots, [2.5, 97.5])]


def two_groups(a, b, draws=DRAWS, seed=0):
    """Equal-user mean of b minus a, a bootstrap 95% interval (each group resampled on its own) and a two-sided
    permutation p (group labels shuffled over the pooled users)."""
    rng = np.random.default_rng(seed)
    diff = float(b.mean() - a.mean())
    boots = (b[rng.integers(0, len(b), (draws, len(b)))].mean(1)
             - a[rng.integers(0, len(a), (draws, len(a)))].mean(1))
    pooled = np.concatenate([a, b])
    shuffled = np.array([rng.permutation(pooled) for _ in range(draws)])
    perm = shuffled[:, len(a):].mean(1) - shuffled[:, :len(a)].mean(1)
    p = (1 + int((np.abs(perm) >= abs(diff) - 1e-12).sum())) / (draws + 1)
    return {"diff": diff, "ci95": [float(x) for x in np.percentile(boots, [2.5, 97.5])], "permutation_p": p,
            "users_a": len(a), "users_b": len(b)}


def paired(a, b, draws=DRAWS, seed=0):
    """bench.stats.paired on per-user means: window-weighted mean of a - b, user-bootstrap CI, users better."""
    if not (np.array_equal(a.ids, b.ids) and np.array_equal(a.weight, b.weight)):
        raise ValueError("paired comparison on different users or windows")

    def as_accuracy(u):
        return {"per_player": {i: {"windows": float(w), "sum_deg_by_horizon": [float(m * w)]}
                               for i, m, w in zip(u.ids, u.mean, u.weight)}}
    return stats.paired(as_accuracy(a), as_accuracy(b), draws, seed)


def roles(trained_on, old, merged):
    """Users per role for one model. Every merged-file user gets exactly one role."""
    everyone = set().union(*merged.values())
    if trained_on == "old":
        trained, selection = set(old["train"]), set(old["validation"])
    else:
        trained, selection = set(merged["train"]), set(merged["validation"])
    result = {"trained": trained, "selection": selection, "unseen": everyone - trained - selection}
    if trained & selection or not (trained | selection) <= everyone:
        raise ValueError("roles overlap, or a trained/selection user is missing from the merged file")
    return result


# ---------------------------------------------------------------- data

class Scores:
    """Per-segment step-10/20 error sums of the baselines and every scored checkpoint, on one segment list."""

    def __init__(self, folder, labels):
        base = json.loads((folder / "baselines.json").read_text(encoding="utf-8"))
        if base.get("subset_users"):
            raise ValueError("scores/baselines.json is a subset run; rerun score_users.py baselines on every user")
        self.meta = {"baselines": {k: v for k, v in base.items() if k not in ("segments", "sums")}}
        self.segments = base["segments"]
        self.users = sorted({s["user"] for s in self.segments})
        index = {u: i for i, u in enumerate(self.users)}
        self.segment_user = np.array([index[s["user"]] for s in self.segments])
        self.scored = np.array([s["windows_scored"] for s in self.segments], float)
        self.total = np.array([s["windows_total"] for s in self.segments], float)
        self.sums = {(n, step): np.array(base["sums"][n][f"step{step}"]) for n in base["predictors"] for step in (10, 20)}
        self.arrays = {n: base["per_window_file"] for n in base["predictors"]}
        self.labels = []
        for label in labels:
            path = folder / f"{label}.json"
            if not path.exists():
                continue
            d = json.loads(path.read_text(encoding="utf-8"))
            if d.get("subset_users") or [(s["key"], s["windows_scored"]) for s in d["segments"]] != \
                    [(s["key"], s["windows_scored"]) for s in self.segments]:
                raise ValueError(f"{label}: not scored on the same segments and windows as the baselines")
            for step in (10, 20):
                self.sums[(label, step)] = np.array(d["sums"]["model"][f"step{step}"])
            self.arrays[label] = d["per_window_file"]
            self.meta[label] = {k: v for k, v in d.items() if k not in ("segments", "sums")}
            self.labels.append(label)

    def mask(self, users, recordings=None):
        """Segments of these users (and, optionally, only of these (user, recording) pairs)."""
        keep = np.array([s["user"] in users for s in self.segments])
        if recordings is not None:
            keep &= np.array([(s["user"], s["recording"]) in recordings for s in self.segments])
        return keep

    def per_user(self, name, mask, step=20):
        """Users(ids, mean over scored windows, weight = all windows) for the users with a segment in mask."""
        n = np.bincount(self.segment_user[mask], self.scored[mask], len(self.users))
        w = np.bincount(self.segment_user[mask], self.total[mask], len(self.users))
        s = np.bincount(self.segment_user[mask], self.sums[(name, step)][mask], len(self.users))
        keep = n > 0
        return Users(np.array(self.users)[keep], s[keep] / n[keep], w[keep])

    def per_window(self, name, mask):
        """Step-20 errors of every window in the masked segments (stride-1 splits only)."""
        if not all(s["stride"] == 1 for s, m in zip(self.segments, mask) if m):
            return None
        with np.load(self.arrays[name]) as f:
            return f[name if name in BASELINES else "model"][mask[f["segment"]]]


def load_ledger():
    with LEDGER.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- verification

def verify(scores, old, merged, manifest):
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            raise SystemExit(f"Verification failed: {name}: {detail}")

    for model, spec in MODELS.items():
        if spec["labels"][spec["selected"]] not in scores.labels:
            raise SystemExit(f"{spec['labels'][spec['selected']]} (the selected {model} checkpoint) is not scored yet")
    shas = {m["data_sha256"] for m in scores.meta.values()}
    check("scored the merged file", shas == {manifest["data_sha256"]}, f"data sha256 {sorted(shas)}")
    totals = scores.meta["baselines"]["split_windows_total"]
    for split in ("train", "validation", "test"):
        mine = [s for s in scores.segments if s["split"] == split]
        total = sum(s["windows_total"] for s in mine)
        check(f"{split} windows match check_corpus", total == totals[split], f"{total:,} vs {totals[split]:,}")
        if split != "train":
            check(f"{split} scored at stride 1", all(s["windows_scored"] == s["windows_total"] for s in mine),
                  "every window scored")
    test20 = scores.mask(set(old["test"]))
    check("test20 windows", scores.total[test20].sum() == TEST20_WINDOWS, f"{int(scores.total[test20].sum()):,}")

    def against_bench(name, bench_name):
        acc = json.loads((BENCH / f"{bench_name}@jacob-amd.json").read_text(encoding="utf-8"))["accuracy"]
        mine = scores.per_user(name, test20)
        theirs = np.array([acc["per_player"][u]["sum_deg_by_horizon"][-1] / acc["per_player"][u]["windows"]
                           for u in mine.ids])
        worst = float(np.abs(mine.mean - theirs).max())
        check(f"{name} on test20 = bench {bench_name}", worst <= 1e-3, f"max per-user difference {worst:.2e} deg")

    for name, bench_name in BASELINES.items():
        against_bench(name, bench_name)
    selected_100 = MODELS["k8_100"]["labels"][MODELS["k8_100"]["selected"]]
    against_bench(selected_100, "v9_k8_combined_ga2")

    ledger = load_ledger()
    val15, val120 = scores.mask(set(old["validation"])), scores.mask(set(merged["validation"]))
    confirm = json.loads(CONFIRM_800.read_text(encoding="utf-8"))["rows"][0]["scores"]
    for model, spec in MODELS.items():
        for seed, label in spec["labels"].items():
            if label not in scores.labels:
                continue
            if model == "k8_100":
                row = [r for r in ledger if (r["suite"], r["variant"], r["seed"]) == ("v9_confirm", "k8_combined",
                                                                                         str(seed))]
                expected, mask, cohort = float(row[0]["val_step20_deg"]), val15, "val15"
            else:
                expected, mask, cohort = confirm[str(seed)], val120, "val120"
            got = weighted_mean(*scores.per_user(label, mask)[1:])
            check(f"{label} on {cohort} = trainer", abs(got - expected) <= 1e-4, f"{got:.6f} vs {expected:.6f}")
        selection = json.loads(spec["selection_json"].read_text(encoding="utf-8"))
        selected = spec["labels"][spec["selected"]]
        sha = scores.meta[selected]["checkpoint_sha256"]
        check(f"{selected} is the selected checkpoint", sha == selection["checkpoint_sha256"], sha[:16])
        r = roles(spec["trained_on"], old, merged)
        check(f"{model} roles are disjoint and cover every user",
              sum(len(x) for x in r.values()) == len(scores.users) and set().union(*r.values()) == set(scores.users),
              {k: len(v) for k, v in r.items()})
    return checks


# ---------------------------------------------------------------- sections

def family_of_users(scores, mask, recording_meta):
    """Each user's headset family with the most windows in the masked segments."""
    windows = {}
    for s, m in zip(scores.segments, mask):
        if m:
            family = FAMILY_OF[recording_meta[(s["user"], s["recording"])]["headset"]]
            windows.setdefault(s["user"], {}).setdefault(family, 0)
            windows[s["user"]][family] += s["windows_total"]
    return {u: max(f, key=f.get) for u, f in windows.items()}


def accuracy(scores, model, cohorts, recording_meta):
    spec, out = MODELS[model], {}
    selected = spec["labels"][spec["selected"]]
    for cohort, users in cohorts.items():
        mask = scores.mask(users)
        m = scores.per_user(selected, mask)
        base = {b: scores.per_user(b, mask) for b in BASELINES}
        errors = scores.per_window(selected, mask)
        families = family_of_users(scores, mask, recording_meta)
        ratio = m.mean / base["damped_velocity"].mean
        worst = np.argsort(-ratio)[:5]
        seeds = {seed: weighted_mean(*scores.per_user(label, mask)[1:])
                 for seed, label in spec["labels"].items() if label in scores.labels}
        out[cohort] = {
            "users": len(m.ids), "windows": int(m.weight.sum()),
            "mean_deg_step20": weighted_mean(m.mean, m.weight), "ci95": bootstrap_ci(m.mean, m.weight),
            "equal_user_mean_deg_step20": float(m.mean.mean()),
            "mean_deg_step10": weighted_mean(scores.per_user(selected, mask, 10).mean, m.weight),
            "median_p95_p99_deg_step20": None if errors is None else
            [float(x) for x in np.percentile(errors, [50, 95, 99])],
            "between_user_sd": float(m.mean.std(ddof=1)), "user_iqr": [float(x) for x in np.percentile(m.mean, [25, 75])],
            "baselines_deg_step20": {b: weighted_mean(u.mean, u.weight) for b, u in base.items()},
            "skill": {b: {"equal_user_mean": float((1 - m.mean / u.mean).mean()),
                          "users_better": int((m.mean < u.mean).sum())} for b, u in base.items()},
            "paired_vs": {b: paired(m, u) for b, u in base.items() if b != "hold_last"},
            "worst_users": [{"user": str(m.ids[i]), "family": families[m.ids[i]], "windows": int(m.weight[i]),
                             "model": float(m.mean[i]), "damped": float(base["damped_velocity"].mean[i]),
                             "ratio": float(ratio[i])} for i in worst],
            "seeds": seeds,
            "seed_mean_sd": [float(np.mean(list(seeds.values()))), float(np.std(list(seeds.values()), ddof=1))]
            if len(seeds) > 1 else None,
        }
    return out


def ratio_by_role(scores, label, role_users):
    out = {}
    for role, users in role_users.items():
        mask = scores.mask(users)
        m, d = scores.per_user(label, mask), scores.per_user("damped_velocity", mask)
        out[role] = {"users": len(m.ids), "model_deg": weighted_mean(m.mean, m.weight),
                     "damped_deg": weighted_mean(d.mean, d.weight), "ratio": m.mean / d.mean}
    return out


def gap(scores, model, old, merged):
    spec = MODELS[model]
    role_users = roles(spec["trained_on"], old, merged)
    if model == "k8_100":  # same split, same stride, same export: the cleanest seen/unseen pair
        role_users["unseen_train_split"] = set(merged["train"]) - set(old["train"])
    pairs = [("trained", "unseen"), ("trained", "selection"), ("selection", "unseen")]
    if model == "k8_100":
        pairs.append(("trained", "unseen_train_split"))
    out = {"roles": {k: len(v) for k, v in role_users.items()}, "seeds": {}}
    for seed, label in spec["labels"].items():
        if label not in scores.labels:
            continue
        by_role = ratio_by_role(scores, label, role_users)
        out["seeds"][seed] = {
            "by_role": {r: {"users": v["users"], "model_deg": v["model_deg"], "damped_deg": v["damped_deg"],
                            "mean_ratio": float(v["ratio"].mean()), "ratio_ci95": bootstrap_ci(v["ratio"])}
                        for r, v in by_role.items()},
            "differences": {f"{b} - {a}": two_groups(by_role[a]["ratio"], by_role[b]["ratio"]) for a, b in pairs}}
    out["selected"] = out["seeds"][spec["selected"]]
    return out


def group_order(group):
    """Table row order: motion terciles low to high, rates numerically, headset families alphabetically."""
    return ({"low": 0, "mid": 1, "high": 2}.get(group, 0), int(group) if group.isdigit() else 0, group)


def breakdown(scores, model, cohorts, recording_meta, trained_users):
    """Per group of recordings: users, windows, model and damped error, and the per-user ratio with a CI."""
    tracked = [m for m in recording_meta if m[0] in trained_users]
    cuts = np.percentile([recording_meta[k]["velocity_deg_s"] for k in tracked], [100 / 3, 200 / 3])
    groupings = {
        "headset family": lambda r: FAMILY_OF[r["headset"]],
        "source rate (Hz, nearest 5)": lambda r: f"{int(round(r['source_rate_hz'] / 5) * 5)}",
        "motion intensity (velocity_deg_s tercile)": lambda r: ["low", "mid", "high"][
            int(np.searchsorted(cuts, r["velocity_deg_s"]))],
    }
    selected = MODELS[model]["labels"][MODELS[model]["selected"]]
    out = {"motion_cut_points": [float(c) for c in cuts]}
    for cohort, users in cohorts.items():
        cohort_out = {}
        for grouping, key in groupings.items():
            groups = {}
            for k, meta in recording_meta.items():
                groups.setdefault(key(meta), set()).add(k)
            rows = {}
            for group, recordings in sorted(groups.items(), key=lambda item: group_order(item[0])):
                mask = scores.mask(users, recordings)
                if not mask.any():
                    continue
                m, d = scores.per_user(selected, mask), scores.per_user("damped_velocity", mask)
                ratio = m.mean / d.mean
                rows[group] = {"users": len(m.ids), "recordings": len({(s["user"], s["recording"]) for s, x in
                                                                       zip(scores.segments, mask) if x}),
                               "windows": int(m.weight.sum()), "model_deg": weighted_mean(m.mean, m.weight),
                               "damped_deg": weighted_mean(d.mean, d.weight), "mean_ratio": float(ratio.mean()),
                               "ratio_ci95": bootstrap_ci(ratio) if len(ratio) >= MIN_GROUP_USERS else None,
                               "trained_users_in_group": len({k[0] for k in recordings if k[0] in trained_users})}
            cohort_out[grouping] = rows
        trained_headsets = {recording_meta[k]["headset"] for k in tracked}
        unseen = {}
        for s, x in zip(scores.segments, scores.mask(users)):
            headset = recording_meta[(s["user"], s["recording"])]["headset"]
            if x and headset not in trained_headsets:
                unseen.setdefault(headset, set()).add(s["user"])
        cohort_out["headsets absent from training users"] = {h: sorted(u) for h, u in sorted(unseen.items())}
        out[cohort] = cohort_out
    return out


def validation_to_test(scores, old, merged):
    ledger = load_ledger()
    rows = []
    for suite, variant, bench_name in LEDGER_TO_BENCH:
        picked = [r for r in ledger if r["suite"] == suite and r["variant"] == variant and r["selected"] == "1"]
        if len(picked) != 1:
            raise SystemExit(f"ledger: expected one selected row for {suite}/{variant}, found {len(picked)}")
        test = json.loads((BENCH / f"{bench_name}@jacob-amd.json").read_text(encoding="utf-8"))
        val = float(picked[0]["val_step20_deg"])
        rows.append({"bench": bench_name, "ledger": f"{suite}/{variant}", "seed": int(picked[0]["seed"]),
                     "val15": val, "test20": test["accuracy"]["mean_deg_step20"],
                     "test_minus_val": test["accuracy"]["mean_deg_step20"] - val})
    val, test = [r["val15"] for r in rows], [r["test20"] for r in rows]
    shift = np.array([r["test_minus_val"] for r in rows])
    result = {"models": rows, "spearman": float(scipy_stats.spearmanr(val, test)[0]),
              "kendall": float(scipy_stats.kendalltau(val, test)[0]), "pearson": float(np.corrcoef(val, test)[0, 1]),
              "test_minus_val_mean_sd": [float(shift.mean()), float(shift.std(ddof=1))]}

    cohorts = {"val15": set(old["validation"]), "val105_new": set(merged["validation"]) - set(old["validation"]),
               "val120": set(merged["validation"]), "test20": set(old["test"]), "test160": set(merged["test"])}
    a, b = (MODELS[m]["labels"][MODELS[m]["selected"]] for m in ("k8_800", "k8_100"))
    deltas = {}
    for cohort, users in cohorts.items():
        mask = scores.mask(users)
        entry = {"selected": paired(scores.per_user(a, mask), scores.per_user(b, mask))}
        seed_users = {m: [scores.per_user(label, mask) for label in MODELS[m]["labels"].values()
                          if label in scores.labels] for m in MODELS}
        if all(len(v) == 3 for v in seed_users.values()):
            averaged = {m: Users(v[0].ids, np.mean([u.mean for u in v], 0), v[0].weight) for m, v in seed_users.items()}
            entry["seed_averaged"] = paired(averaged["k8_800"], averaged["k8_100"])
        deltas[cohort] = entry
    result["k8_800_minus_k8_100"] = deltas
    return result


# ---------------------------------------------------------------- markdown

def f3(x):
    return f"{x:.3f}"


def ci(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def markdown(report):
    lines = ["# Generalisation report: k8 trained on 100 and on 800 users", "",
             f"Made by `report.py` from `scores/*.json` ({report['created']}). Every number is at step 20 (200 ms) in "
             "degrees, window-weighted unless it says equal-user. CIs are 95%, from resampling users.", ""]
    for model, spec in MODELS.items():
        lines += [f"- **{model}**: {spec['text']}"]
    lines += ["", "## A. Unseen-user accuracy", ""]
    for model, cohorts in report["A_unseen_accuracy"].items():
        lines += [f"### {model}", "",
                  "| Cohort | Users | Windows | Mean (CI) | Equal-user | 100 ms | Median / p95 / p99 | Between-user SD | "
                  "Seeds mean ± SD | Δ vs CV (CI), users better | Δ vs damped (CI), users better |",
                  "| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | --- | --- | --- |"]
        for cohort, r in cohorts.items():
            q = "–" if r["median_p95_p99_deg_step20"] is None else " / ".join(f"{x:.2f}" for x in r["median_p95_p99_deg_step20"])
            seeds = "–" if r["seed_mean_sd"] is None else f"{r['seed_mean_sd'][0]:.4f} ± {r['seed_mean_sd'][1]:.4f}"
            vs = {b: f"{p['mean_diff_deg']:+.3f} {ci(p['ci95'])}, {p['players_better']}/{p['players']}"
                  for b, p in r["paired_vs"].items()}
            lines.append(f"| {cohort} | {r['users']} | {r['windows']:,} | {f3(r['mean_deg_step20'])} "
                         f"[{f3(r['ci95'][0])}, {f3(r['ci95'][1])}] | {f3(r['equal_user_mean_deg_step20'])} | "
                         f"{f3(r['mean_deg_step10'])} | {q} | {f3(r['between_user_sd'])} | {seeds} | "
                         f"{vs['constant_velocity']} | {vs['damped_velocity']} |")
        lines += ["", "Baselines on the same windows, and the 5 worst users by model/damped ratio:", ""]
        for cohort, r in cohorts.items():
            b = r["baselines_deg_step20"]
            worst = "; ".join(f"{w['user'][:8]} ({w['family']}) {w['model']:.2f} vs {w['damped']:.2f}, ratio "
                              f"{w['ratio']:.3f}" for w in r["worst_users"])
            lines.append(f"- {cohort}: hold {f3(b['hold_last'])}, CV {f3(b['constant_velocity'])}, damped "
                         f"{f3(b['damped_velocity'])}. Worst: {worst}")
        lines.append("")

    lines += ["## B. Seen against unseen users", "",
              "Mean of the per-user model/damped ratio (equal-user). Lower is better. The ratio corrects for users "
              "whose motion is simply harder to predict. A positive unseen − trained difference whose CI excludes 0 "
              "means the model does better on the users it trained on.", ""]
    for model, g in report["B_gap"].items():
        sel = g["selected"]
        lines += [f"### {model} (selected seed)", "", "| Role | Users | Model | Damped | Mean ratio (CI) |",
                  "| --- | ---: | ---: | ---: | --- |"]
        for role, r in sel["by_role"].items():
            lines.append(f"| {role} | {r['users']} | {f3(r['model_deg'])} | {f3(r['damped_deg'])} | "
                         f"{r['mean_ratio']:.4f} [{r['ratio_ci95'][0]:.4f}, {r['ratio_ci95'][1]:.4f}] |")
        lines += ["", "| Difference | Δ ratio | CI | Permutation p |", "| --- | ---: | --- | ---: |"]
        for name, d in sel["differences"].items():
            lines.append(f"| {name} | {d['diff']:+.4f} | [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | "
                         f"{d['permutation_p']:.4f} |")
        seeds = {s: v["differences"]["unseen - trained"]["diff"] for s, v in g["seeds"].items()}
        lines += ["", "unseen − trained by seed: " + ", ".join(f"{s}: {v:+.4f}" for s, v in seeds.items()), ""]

    lines += ["## C. Unseen users by headset, source rate and motion", "",
              f"Per-user model/damped ratio inside each group. CIs only for groups with ≥ {MIN_GROUP_USERS} users. "
              "Groups are of recordings, so a user with recordings in two groups counts in both. 'Trained' counts the "
              "model's training users who have a recording in the group. Small groups are anecdotes, not findings.",
              ""]
    for model, b in report["C_breakdown"].items():
        lines.append(f"### {model} (motion tercile cut points {b['motion_cut_points'][0]:.1f} / "
                     f"{b['motion_cut_points'][1]:.1f} °/s, set on its training users)")
        for cohort, groupings in ((k, v) for k, v in b.items() if k != "motion_cut_points"):
            lines += ["", f"**{cohort}**", ""]
            for grouping, rows in groupings.items():
                if grouping == "headsets absent from training users":
                    absent = "; ".join(f"{h} ({len(u)} user{'s' * (len(u) != 1)})" for h, u in rows.items()) or "none"
                    lines += [f"Headsets absent from the training users: {absent}", ""]
                    continue
                lines += [f"| {grouping} | Users | Recordings | Windows | Model | Damped | Mean ratio (CI) | Trained |",
                          "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
                for group, r in rows.items():
                    c = "" if r["ratio_ci95"] is None else f" [{r['ratio_ci95'][0]:.3f}, {r['ratio_ci95'][1]:.3f}]"
                    lines.append(f"| {group} | {r['users']} | {r['recordings']} | {r['windows']:,} | "
                                 f"{f3(r['model_deg'])} | {f3(r['damped_deg'])} | {r['mean_ratio']:.3f}{c} | "
                                 f"{r['trained_users_in_group']} |")
                lines.append("")

    d = report["D_validation_to_test"]
    lines += ["## D. Validation to test", "",
              f"Ledger models (selected checkpoint, val15 from `runs.csv`, test20 from the bench): Spearman "
              f"{d['spearman']:.3f}, Kendall {d['kendall']:.3f}, Pearson {d['pearson']:.3f}; test − val "
              f"{d['test_minus_val_mean_sd'][0]:+.4f} ± {d['test_minus_val_mean_sd'][1]:.4f}. With "
              f"{len(d['models'])} models that share one cohort, this is weak evidence.", "",
              "| Bench model | Ledger | Seed | val15 | test20 | test − val |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in d["models"]:
        lines.append(f"| {r['bench']} | {r['ledger']} | {r['seed']} | {r['val15']:.4f} | {r['test20']:.4f} | "
                     f"{r['test_minus_val']:+.4f} |")
    lines += ["", "k8-800 − k8-100, paired per user (negative = the 800-user model is better). val15 is k8-100's "
              "selection set and val120 is k8-800's; val105_new and the test sets are unseen by both.", "",
              "| Cohort | Selected seeds: Δ (CI), users better | 3-seed average: Δ (CI), users better |",
              "| --- | --- | --- |"]
    for cohort, e in d["k8_800_minus_k8_100"].items():
        cells = [f"{p['mean_diff_deg']:+.4f} {ci(p['ci95'])}, {p['players_better']}/{p['players']}"
                 for p in (e["selected"], e.get("seed_averaged")) if p]
        lines.append(f"| {cohort} | " + " | ".join(cells + ["–"] * (2 - len(cells))) + " |")
    lines += ["", "## Verification", ""] + [f"- {'OK' if c['ok'] else 'FAILED'}: {c['check']} ({c['detail']})"
                                            for c in report["verification"]]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores", type=Path, default=HERE / "scores")
    p.add_argument("--out", type=Path, default=HERE / "generalisation_report")
    args = p.parse_args()

    manifest = json.loads(MERGED_MANIFEST.read_text(encoding="utf-8"))
    merged, old = manifest["splits"], json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))["splits"]
    recording_meta = {(r["user"], r["recording"]): r for r in manifest["recordings"] if r["status"] == "qualified"}
    unknown = {r["headset"] for r in recording_meta.values()} - set(FAMILY_OF)
    if unknown:
        raise SystemExit(f"headsets without a family in HEADSET_FAMILIES: {sorted(unknown)}")
    scores = Scores(args.scores, [label for spec in MODELS.values() for label in spec["labels"].values()])
    if {(s["user"], s["recording"]) for s in scores.segments} - set(recording_meta):
        raise SystemExit("a scored recording is missing from the merged manifest")

    from datetime import datetime
    report = {"created": datetime.now().isoformat(timespec="seconds"), "labels": scores.labels,
              "verification": verify(scores, old, merged, manifest)}
    test_cohorts = {"test20": set(old["test"]), "test160": set(merged["test"])}
    pool = roles("old", old, merged)["unseen"]
    report["A_unseen_accuracy"] = {
        model: accuracy(scores, model, {**test_cohorts, **({"unseen pool (965)": pool} if model == "k8_100" else {})},
                        recording_meta) for model in MODELS}
    report["B_gap"] = {model: gap(scores, model, old, merged) for model in MODELS}
    report["C_breakdown"] = {
        model: breakdown(scores, model, {"test160": set(merged["test"]),
                                         **({"unseen pool (965)": pool} if model == "k8_100" else {})},
                         recording_meta, roles(MODELS[model]["trained_on"], old, merged)["trained"])
        for model in MODELS}
    report["D_validation_to_test"] = validation_to_test(scores, old, merged)

    args.out.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(f"wrote {args.out.with_suffix('.json')} and .md")


if __name__ == "__main__":
    main()
