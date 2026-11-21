"""Every Jacob model from v7 on and Nic's group-a-v2 models: accuracy, size, latency, ops and paired CIs.

Run from the repo root: python jacob/training/outputs/v9_bench/full_compare.py <latency session json> <out dir>
Adapted from outputs/v8_bench/nic_compare.py. Results are the jacob-amd scores of 29 Oct (one machine, one bench
commit); v5/v6 and the baselines are earlier jacob-amd results, kept as context. Paired tests are bench.stats.paired
(player bootstrap, 10,000 draws, seed 0); for step 10 and for seed means the per-player sums are swapped into a copy
of the accuracy dict. Test is for reporting only.
"""
import copy
import json
import statistics
import sys
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path.cwd()))
from bench import results, stats  # noqa: E402

BENCH, MACHINE = "group-a-v2-test20", "jacob-amd"
REFERENCE = "v7_sep_base3_addons_angular_ga2"
# name -> (short label, graph, validation 3-seed mean ± SD from the training runs)
MINE = {
    "v9_k8_combined_ga2": ("k8_combined", "v9 lean", "4.2501 ± 0.0041"),
    "v9_k8_combined_padded_ga2": ("k8_combined", "v9 padded rows", "4.2501 ± 0.0041"),
    "v9_k8_combined_padded_a4_ga2": ("k8_combined", "v9 padded + A4", "4.2501 ± 0.0041"),
    "v9_k5_combined_ga2": ("k5_combined", "v9 lean", "4.2717 ± 0.0067"),
    "v9_k5_combined_padded_ga2": ("k5_combined", "v9 padded rows", "4.2717 ± 0.0067"),
    "v9_k5_combined_padded_a4_ga2": ("k5_combined", "v9 padded + A4", "4.2717 ± 0.0067"),
    "v9_k8_jitter3_s42_ga2": ("k8 + jitter 3 ms (screen)", "v9 lean", "4.3212 (seed 42 only)"),
    "v9_sep_angular_lean_ga2": ("sep + angular", "v9 lean", "4.3297 ± 0.0142"),
    "v7_sep_base3_addons_angular_ga2": ("sep + angular (current best)", "v7", "4.3297 ± 0.0142"),
    "v7_sep_base3_addons_angular_split_ga2": ("sep + angular", "v7 mirror-split", "4.3297 ± 0.0142"),
    "v7_base3_5_addons_angular_ga2": ("base3_5 + angular", "v7", "4.3247 ± 0.0104"),
    "v7_sep_base3_addons_ga2": ("sep (quaternion loss)", "v7", "4.3829 ± 0.0119"),
    "v7_base3_5_addons_ga2": ("base3_5 (quaternion loss)", "v7", "4.3673 ± 0.0121"),
    "v6_control_v7graph_ga2": ("v6 control", "v7", "4.3423 ± 0.0121"),
}
SEEDS = range(42, 47)
NIC = {g: [f"nic_ga2_{g}_s{s}" for s in SEEDS] for g in ("gru", "gadst")}
NIC_LABEL = {"gru": ("Nic stateful GRU", "GRU, 32 units"), "gadst": ("Nic GADS-T", "GADS-T")}
CONTEXT = {"v6_tcnn_ga2": ("v6 control (registered)", "v6", "4.3423 ± 0.0121"),
           "v5_tcnn_ga2": ("v5 control", "v5", "4.4049 ± 0.0042"),
           "damped_velocity": ("damped velocity", "baseline", "–"), "hold_last": ("hold last", "baseline", "–")}

latency_path, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
latency = json.loads(latency_path.read_text(encoding="utf-8"))["models"]
names = [*MINE, *NIC["gru"], *NIC["gadst"], *CONTEXT]
R = {n: results.find_result(BENCH, n, MACHINE) for n in names}
fingerprints = {r["bench"]["cohort_fingerprint"] for r in R.values()}
data = {r["inputs"]["data_sha256"] for r in R.values()}
assert len(fingerprints) == 1 and len(data) == 1, (fingerprints, data)


def at_step(accuracy, step):
    a = copy.deepcopy(accuracy)
    for p in a["per_player"].values():
        p["sum_deg_by_horizon"] = p["sum_deg_by_horizon"][:step]
    return a


def acc(name, key="accuracy"):
    return R[name][key] or R[name]["accuracy"]


def seed_mean(group, key):
    """Per-player sums averaged over seeds: the expected score of a randomly chosen seed."""
    base = copy.deepcopy(acc(group[0], key))
    for u, p in base["per_player"].items():
        cols = [acc(n, key)["per_player"][u]["sum_deg_by_horizon"] for n in group]
        p["sum_deg_by_horizon"] = [statistics.fmean(c) for c in zip(*cols)]
    return base


def paired(a, b):
    p = stats.paired(a, b)
    return {k: p[k] for k in ("mean_diff_deg", "ci95", "players_better", "players")}


def compare(a_name, b, step=20, key="accuracy"):
    """b is a model name or a Nic group key."""
    other = seed_mean(NIC[b], key) if b in NIC else acc(b, key)
    return paired(at_step(acc(a_name, key), step), at_step(other, step))


def nodes(path):
    """Non-Constant nodes in the file, and after ONNX Runtime's basic (constant-folding) optimisation."""
    graph = onnx.load(str(path)).graph
    with tempfile.TemporaryDirectory() as tmp:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.optimized_model_filepath = str(Path(tmp) / "folded.onnx")
        ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        folded = onnx.load(options.optimized_model_filepath).graph
    count = lambda g: sum(n.op_type != "Constant" for n in g.node)  # noqa: E731
    return count(graph), count(folded)


def unsafe_ops(name):
    o = R[name]["efficiency"]["operators"]
    bits = o["avoid"] + o["other"] + o["slow_on_npu"] + o["custom_domain"] + [f"{k} (conditional)" for k in o["conditional"]]
    return ", ".join(bits) or "none"


def lat(name):
    t = latency.get(name, {}).get("threads", {})
    return {th: {"median_us": v["median_ms"] * 1e3, "p99_us": v["p99_ms"] * 1e3} for th, v in t.items()}


def label(name):
    if name in MINE:
        return MINE[name]
    if name in CONTEXT:
        return CONTEXT[name]
    g = "gru" if "_gru_" in name else "gadst"
    return (f"{NIC_LABEL[g][0]} s{name[-2:]}", NIC_LABEL[g][1], "–")


rows = []
for n in names:
    r, a, c = R[n], R[n]["accuracy"], R[n].get("accuracy_cold_start")
    eff, onnx_path = r["efficiency"] or {}, r["model"].get("onnx")
    file_nodes, run_nodes = nodes(Path(onnx_path)) if onnx_path else (None, None)
    rows.append({
        "name": n, "label": label(n)[0], "graph": label(n)[1], "validation": label(n)[2],
        "owner": "nic" if n.startswith("nic_") else ("–" if r["model"]["kind"] != "onnx" else "jacob"),
        "rerun_29oct": n not in CONTEXT, "created_utc": r["created_utc"],
        "step20": a["mean_deg_step20"], "step10": a["mean_deg_step10"], "median20": a["median_deg_step20"],
        "p95_20": a["p95_deg_step20"], "p99_20": a["p99_deg_step20"], "equal_player20": a["equal_player_mean_deg_step20"],
        "cold20": c["mean_deg_step20"] if c else None,
        "vs_reference": compare(n, REFERENCE) if n != REFERENCE else None,
        "vs_nic_gru": compare(n, "gru") if not n.startswith("nic_ga2_gru") else None,
        "params": r["model"].get("declared_trainable_parameters"),
        "onnx_elements": (eff.get("initializer_elements") or 0) + (eff.get("constant_node_elements") or 0) if eff else None,
        "initializers": eff.get("initializer_elements"), "macs": (eff.get("macs_per_call") or {}).get("total"),
        "state": eff.get("state_elements"), "stateful": eff.get("stateful"),
        "file_nodes": file_nodes, "run_nodes": run_nodes,
        "ops": unsafe_ops(n) if r["model"]["kind"] == "onnx" else "–",
        "deterministic": (eff.get("output_checks") or {}).get("deterministic"),
        "thread_drift": (eff.get("output_checks") or {}).get("max_abs_diff_1_vs_4_threads"),
        "latency": lat(n),
    })
by_name = {r["name"]: r for r in rows}

groups = {}
for g, group in NIC.items():
    for key, tag in (("accuracy", "step20"), ("accuracy_cold_start", "cold20")):
        vals = [acc(n, key)["mean_deg_step20"] for n in group]
        groups.setdefault(g, {})[tag] = (statistics.fmean(vals), statistics.stdev(vals))
    vals = [R[n]["accuracy"]["mean_deg_step10"] for n in group]
    groups[g]["step10"] = (statistics.fmean(vals), statistics.stdev(vals))

KEY_PAIRS = [("v9_k8_combined_ga2", REFERENCE), ("v9_k5_combined_ga2", REFERENCE), ("v9_k8_combined_ga2", "v9_k5_combined_ga2"),
             ("v9_k8_jitter3_s42_ga2", "v9_k8_combined_ga2"), ("v9_k8_combined_ga2", "gru"), ("v9_k8_combined_ga2", "gadst"),
             ("v9_k5_combined_ga2", "gru"), (REFERENCE, "gru"), (REFERENCE, "gadst"),
             ("v9_k8_combined_padded_ga2", "v9_k8_combined_ga2"), ("v9_k8_combined_padded_a4_ga2", "v9_k8_combined_ga2"),
             ("v9_sep_angular_lean_ga2", REFERENCE)]
pairs = []
for a, b in KEY_PAIRS:
    pairs.append({"a": a, "b": b if b not in NIC else f"nic_ga2_{b} (5-seed mean)",
                  "from_start_s20": compare(a, b), "cold_start_s20": compare(a, b, key="accuracy_cold_start"),
                  "from_start_s10": compare(a, b, step=10)})
# The step-20 path through at_step must equal the bench's own compare.
direct = stats.paired(acc("v9_k8_combined_ga2"), acc(REFERENCE))["mean_diff_deg"]
assert abs(direct - pairs[0]["from_start_s20"]["mean_diff_deg"]) < 1e-12

out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "full_comparison.json").write_text(json.dumps(
    {"bench": BENCH, "machine": MACHINE, "cohort_fingerprint": fingerprints.pop(), "latency_session": latency_path.name,
     "reference": REFERENCE, "models": rows, "nic_groups": groups, "paired": pairs}, indent=1), encoding="utf-8")


def f(x, d=4):
    return "–" if x is None else f"{x:.{d}f}"


def ci(p):
    if p is None:
        return "reference"
    body = f"{p['mean_diff_deg']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"
    return body + f", {p['players_better']}/{p['players']}"


def num(x):
    return "–" if x is None else f"{x:,}"


order = sorted([r for r in rows if r["name"] not in CONTEXT], key=lambda r: r["step20"]) + [by_name[c] for c in CONTEXT]
L = [f"# Model comparison on {BENCH} ({MACHINE}, 29 Oct 2026)", "",
     "Every model of Jacob's from v7 on and all ten of Nic's group-a-v2 models, scored with `python -m bench score` "
     f"on {MACHINE} at one bench commit, plus one latency session (`{latency_path.name}`). v5, v6 and the baselines are "
     "earlier jacob-amd scores, shown for context. All rows share one cohort fingerprint (20 test players, 931,051 "
     "windows). Test is for reporting; choose between models on validation.", "",
     "## Accuracy (degrees, lower is better)", "",
     "Test = step 20 (200 ms), streamed from segment start. vs current best and vs Nic GRU: paired per-player "
     "difference [95% CI], players better (negative = this model better). Nic GRU = the 5-seed mean per player.", "",
     "| Model | Graph | Validation (3 seeds) | Test s20 | vs current best | vs Nic GRU | Cold start s20 | Test s10 | Median s20 | p99 s20 | Equal-player s20 |",
     "|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|"]
for r in order:
    L.append(f"| {r['label']} | {r['graph']} | {r['validation']} | **{f(r['step20'])}** | {ci(r['vs_reference'])} | "
             f"{ci(r['vs_nic_gru']) if r['vs_nic_gru'] else '–'} | {f(r['cold20'])} | {f(r['step10'])} | {f(r['median20'])} | "
             f"{f(r['p99_20'], 3)} | {f(r['equal_player20'])} |")
L += ["", "## Size, compute and latency", "",
      "Latency: ONNX Runtime CPU on jacob-amd, one call = one new sample, median / p99 in µs, alternating rounds in one "
      "session. Relative only, not ST's hardware. Nodes: non-Constant nodes in the file, then after ONNX Runtime folds "
      "constants at load.", "",
      "| Model | Graph | Declared params | ONNX weight elements | MACs / call | State floats | Nodes (file / run) | Off safe list | 1 thread median / p99 | 4 threads median | Deterministic, 1 = 4 threads |",
      "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|"]
for r in order:
    if r["owner"] == "–":
        continue
    l1, l4 = r["latency"].get("1", {}), r["latency"].get("4", {})
    det = "–" if r["deterministic"] is None else ("yes" if r["deterministic"] and not r["thread_drift"] else f"drift {r['thread_drift']:.1e}")
    L.append(f"| {r['label']} | {r['graph']} | {num(r['params'])} | {num(r['onnx_elements'])} | {num(r['macs'])} | {num(r['state'])} | "
             f"{r['file_nodes']} / {r['run_nodes']} | {r['ops']} | {f(l1.get('median_us'), 1)} / {f(l1.get('p99_us'), 1)} | "
             f"{f(l4.get('median_us'), 1)} | {det} |")
L += ["", "## Nic's seed groups (5 seeds)", "", "| Group | Test s20 mean ± sd | Test s10 mean ± sd | Cold-start s20 mean ± sd |",
      "|---|---:|---:|---:|"]
for g, v in groups.items():
    L.append(f"| {NIC_LABEL[g][0]} | {v['step20'][0]:.4f} ± {v['step20'][1]:.4f} | {v['step10'][0]:.4f} ± {v['step10'][1]:.4f} | "
             f"{v['cold20'][0]:.4f} ± {v['cold20'][1]:.4f} |")
L += ["", "## Key paired comparisons (A − B, negative = A better)", "",
      "Each cell: mean difference [95% CI], players where A is better. 20 players, 10,000 draws, seed 0.", "",
      "| A | B | From start s20 | Cold start s20 | From start s10 |", "|---|---|---|---|---|"]
for p in pairs:
    L.append(f"| {p['a']} | {p['b']} | {ci(p['from_start_s20'])} | {ci(p['cold_start_s20'])} | {ci(p['from_start_s10'])} |")
(out_dir / "full_comparison.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
