"""Jacob's v7 angular picks vs Nic's group-a-v2 models: accuracy, size, latency, ops and paired CIs.

Run from the repo root: python <this file> <latency session json> <out dir>
Paired tests are bench.stats.paired (player bootstrap, 10,000 draws, seed 0). For step 10 and for
Nic's seed means, the per-player sums are swapped into a copy of the accuracy dict, so the same
function is used everywhere.
"""
import copy
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from bench import results, stats  # noqa: E402

BENCH = "group-a-v2-test20"
MINE = ["v7_sep_base3_addons_angular_ga2", "v7_base3_5_addons_angular_ga2", "v6_control_v7graph_ga2"]
CONTEXT = ["v7_sep_base3_addons_ga2", "v7_base3_5_addons_ga2", "v6_tcnn_ga2", "v5_tcnn_ga2", "damped_velocity"]
SEEDS = range(42, 47)
NIC = {g: [f"nic_ga2_{g}_s{s}" for s in SEEDS] for g in ("gru", "gadst")}
LATENCY_NAME = {"nic_ga2_gru": "nic_ga2_gru_s45", "nic_ga2_gadst": "nic_ga2_gadst_s42"}

latency_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
latency = json.loads(latency_path.read_text(encoding="utf-8"))["models"]
R = {n: results.find_result(BENCH, n) for n in MINE + CONTEXT + NIC["gru"] + NIC["gadst"]}
fingerprints = {r["bench"]["cohort_fingerprint"] for r in R.values()}
data = {r["inputs"]["data_sha256"] for r in R.values()}
assert len(fingerprints) == 1 and len(data) == 1, (fingerprints, data)


def at_step(accuracy, step):
    """Copy of an accuracy dict whose last horizon entry is the given step (1-based), for stats.paired."""
    a = copy.deepcopy(accuracy)
    for p in a["per_player"].values():
        p["sum_deg_by_horizon"] = p["sum_deg_by_horizon"][:step]
    return a


def seed_mean(names, key):
    """Per-player sums averaged over seeds: the expected score of a randomly chosen seed."""
    base = copy.deepcopy(acc(names[0], key))
    for u, p in base["per_player"].items():
        cols = [acc(n, key)["per_player"][u]["sum_deg_by_horizon"] for n in names]
        p["sum_deg_by_horizon"] = [statistics.fmean(c) for c in zip(*cols)]
    return base


def acc(name, key="accuracy"):
    return R[name][key] or R[name]["accuracy"]


def paired(a, b):
    p = stats.paired(a, b)
    return {k: p[k] for k in ("mean_diff_deg", "ci95", "players_better", "players", "equal_player_mean_diff_deg")}


def unsafe_ops(name):
    o = R[name]["efficiency"]["operators"]
    bits = o["avoid"] + o["other"] + o["slow_on_npu"] + o["custom_domain"] + [f"{k} (conditional)" for k in o["conditional"]]
    return ", ".join(bits) or "none"


def lat(name):
    t = latency.get(name, {}).get("threads", {})
    return {th: {"median_us": v["median_ms"] * 1e3, "p99_us": v["p99_ms"] * 1e3} for th, v in t.items()}


# ---- per-model table ----
rows = []
for n in MINE + NIC["gru"] + NIC["gadst"] + CONTEXT:
    r, a = R[n], R[n]["accuracy"]
    c = r.get("accuracy_cold_start")
    eff = r["efficiency"] or {}
    rows.append({
        "model": n, "machine": r["machine"]["label"],
        "step20": a["mean_deg_step20"], "step10": a["mean_deg_step10"], "median20": a["median_deg_step20"],
        "p99_20": a["p99_deg_step20"], "equal_player20": a["equal_player_mean_deg_step20"],
        "cold20": c["mean_deg_step20"] if c else None,
        "params": r["model"].get("declared_trainable_parameters"),
        "macs": (eff.get("macs_per_call") or {}).get("total"), "stateful": eff.get("stateful"),
        "ops": unsafe_ops(n) if r["model"]["kind"] == "onnx" else "-",
        "out_norm": eff.get("output_checks", {}).get("output_norm_range") if eff.get("output_checks") else None,
        "latency": lat(n) or lat(next((LATENCY_NAME[g] for g in LATENCY_NAME if n.startswith(g)), "")),
    })

groups = {}
for g, names in NIC.items():
    for key, label in (("accuracy", "step20"), ("accuracy_cold_start", "cold20")):
        vals = [acc(n, key)["mean_deg_step20"] for n in names]
        groups.setdefault(g, {})[label] = (statistics.fmean(vals), statistics.stdev(vals))
    vals = [R[n]["accuracy"]["mean_deg_step10"] for n in names]
    groups[g]["step10"] = (statistics.fmean(vals), statistics.stdev(vals))

# ---- paired tests ----
pairs = []
for m in MINE:
    for g, names in NIC.items():
        targets = [(f"nic_ga2_{g} seed mean", None)] + [(n, n) for n in names]
        for label, n in targets:
            row = {"a": m, "b": label}
            for tag, key, step in (("from_start_s20", "accuracy", 20), ("cold_start_s20", "accuracy_cold_start", 20),
                                   ("from_start_s10", "accuracy", 10)):
                b = seed_mean(names, key) if n is None else acc(n, key)
                row[tag] = paired(at_step(acc(m, key), step), at_step(b, step))
            pairs.append(row)

# sanity: the step-20 path through at_step must equal the bench's own compare
direct = stats.paired(acc(MINE[0]), acc(NIC["gru"][0]))["mean_diff_deg"]
assert abs(direct - next(p for p in pairs if p["b"] == NIC["gru"][0] and p["a"] == MINE[0])["from_start_s20"]["mean_diff_deg"]) < 1e-12

out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "nic_comparison.json").write_text(json.dumps(
    {"bench": BENCH, "cohort_fingerprint": fingerprints.pop(), "latency_session": latency_path.name,
     "models": rows, "nic_groups": groups, "paired": pairs}, indent=1), encoding="utf-8")

# ---- markdown ----
def f(x, d=4):
    return "-" if x is None else f"{x:.{d}f}"


def ci(p):
    return f"{p['mean_diff_deg']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}], {p['players_better']}/{p['players']}"


L = [f"# v7 angular vs Nic's group-a-v2 models ({BENCH})", "",
     "Generated by nic_compare.py from the committed/untracked result JSONs plus latency session "
     f"`{latency_path.name}`. All results share cohort fingerprint and data file; test is for reporting only.", "",
     "## Per model", "",
     "| Model | Scored on | 200 ms (s20) | 100 ms (s10) | Median s20 | p99 s20 | Equal-player s20 | Cold-start s20 | Params | MACs/call | 1-thr median µs | 4-thr median µs | Non-safe ops |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    l1, l4 = r["latency"].get("1", {}), r["latency"].get("4", {})
    L.append(f"| {r['model']} | {r['machine']} | {f(r['step20'])} | {f(r['step10'])} | {f(r['median20'])} | {f(r['p99_20'], 3)} | "
             f"{f(r['equal_player20'])} | {f(r['cold20'])} | {r['params'] or '-'} | {r['macs'] or '-'} | "
             f"{f(l1.get('median_us'), 1)} | {f(l4.get('median_us'), 1)} | {r['ops']} |")
L += ["", "Latency for Nic's other seeds is the timed seed of the same architecture (GRU s45, GADS-T s42).", "",
      "## Nic's groups (5 seeds)", "", "| Group | s20 mean ± sd | s10 mean ± sd | Cold-start s20 mean ± sd |", "|---|---|---|---|"]
for g, v in groups.items():
    L.append(f"| nic_ga2_{g} | {v['step20'][0]:.4f} ± {v['step20'][1]:.4f} | {v['step10'][0]:.4f} ± {v['step10'][1]:.4f} | "
             f"{v['cold20'][0]:.4f} ± {v['cold20'][1]:.4f} |")
L += ["", "## Paired per-player bootstrap (A − B, negative = A better)", "",
      "Each cell: mean difference [95% CI], players where A is better. 20 players, 10,000 draws, seed 0.", "",
      "| A | B | From-start s20 | Cold-start s20 | From-start s10 |", "|---|---|---|---|---|"]
for p in pairs:
    L.append(f"| {p['a']} | {p['b']} | {ci(p['from_start_s20'])} | {ci(p['cold_start_s20'])} | {ci(p['from_start_s10'])} |")
(out_dir / "nic_comparison.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
