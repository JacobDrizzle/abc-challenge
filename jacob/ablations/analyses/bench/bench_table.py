"""One comparison table for the v5/v6/v7 and angular-loss models from bench results + training validation.

Read-only on bench/results and jacob/training/outputs; writes the table to the path given (or prints it).
Usage (from the repo root): python <this script> [out.md]
"""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, ".")
from bench import stats

BENCH = Path("bench/results/group-a-v2-test20")
OUT = Path("jacob/training/outputs")
REFERENCE = "v6_tcnn_ga2"
# bench name -> (label, loss, graph, folder whose selection.json holds the 3 validation seeds)
MODELS = {
    "v5_tcnn_ga2": ("v5 control", "quaternion", "v5", "v5_group_a_v2/control"),
    "v6_tcnn_ga2": ("v6 control (registered)", "quaternion", "v6", "v6_group_a_v2/control"),
    "v6_control_v7graph_ga2": ("v6 control, v7 graph", "quaternion", "v7", "v6_group_a_v2/control"),
    "v7_base3_5_addons_ga2": ("base3_5_addons", "quaternion", "v7", "v7_confirm/base3_5_addons"),
    "v7_base3_5_addons_angular_ga2": ("base3_5_addons + angular", "angular", "v7",
                                      "v8_recipe_base3_5_addons/angular_loss"),
    "v7_sep_base3_addons_ga2": ("separable_base3_addons", "quaternion", "v7", "v7_extra/separable_base3_addons"),
    "v7_sep_base3_addons_angular_ga2": ("separable_base3_addons + angular", "angular", "v7",
                                        "v8_recipe_sep_base3_addons/angular_loss"),
}
BASELINES = {"damped_velocity": "damped velocity (baseline)", "hold_last": "hold last (baseline)"}


def validation(folder):
    sel = json.loads((OUT / folder / "selection.json").read_text(encoding="utf-8"))
    scores = [c["validation_endpoint_deg"] for c in sel["validation_candidates"]]
    return float(np.mean(scores)), float(np.std(scores, ddof=1)), sel["selected_seed"]


def main():
    session = sorted(Path("bench/results/latency").glob("jacob-amd_*.json"))[-1]
    latency = json.loads(session.read_text(encoding="utf-8"))["models"]
    results = {name: json.loads((BENCH / f"{name}@jacob-amd.json").read_text(encoding="utf-8"))
               for name in [*MODELS, *BASELINES]}
    fingerprints = {r["bench"]["cohort_fingerprint"] for r in results.values()}
    assert len(fingerprints) == 1, "results were scored on different windows"
    ref = results[REFERENCE]

    eff_rows, acc_rows = [], []
    for name, (label, loss, graph, folder) in MODELS.items():
        r, e = results[name], results[name]["efficiency"]
        lat = latency[name]["threads"]
        if r["model"]["onnx_sha256"] != latency[name]["onnx_sha256"]:
            raise ValueError(f"{name}: latency session timed a different ONNX file")
        off = e["operators"]["other"] + e["operators"]["avoid"]
        eff_rows.append(
            f"| {label} | {graph} | {r['model']['declared_trainable_parameters']:,} | "
            f"{e['initializer_elements'] + e['constant_node_elements']:,} | {e['macs_per_call']['total']:,} | "
            f"{e['state_elements']:,} | {sum(e['operators']['counts'].values())} | {', '.join(off) or 'none'} | "
            f"{lat['1']['median_ms'] * 1000:.0f} / {lat['1']['p99_ms'] * 1000:.0f} | "
            f"{lat['4']['median_ms'] * 1000:.0f} | "
            f"{'yes' if e['output_checks']['deterministic'] and e['output_checks']['max_abs_diff_1_vs_4_threads'] == 0 else 'no'} |")
        mean, sd, seed = validation(folder)
        a, c = r["accuracy"], r["accuracy_cold_start"]
        if name == REFERENCE:
            vs = "reference"
        else:
            p = stats.paired(a, ref["accuracy"], 10000, 0)
            vs = (f"{p['mean_diff_deg']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}], "
                  f"{p['players_better']}/{p['players']}")
        acc_rows.append(
            f"| {label} | {loss} | {mean:.4f} ± {sd:.4f} | {seed} | **{a['mean_deg_step20']:.4f}** | {vs} | "
            f"{c['mean_deg_step20']:.4f} | {a['mean_deg_step10']:.4f} | {a['median_deg_step20']:.4f} | "
            f"{a['p95_deg_step20']:.3f} | {a['p99_deg_step20']:.3f} | {a['equal_player_mean_deg_step20']:.4f} |")
    for name, label in BASELINES.items():
        a = results[name]["accuracy"]
        acc_rows.append(f"| {label} | – | – | – | {a['mean_deg_step20']:.4f} | – | – | {a['mean_deg_step10']:.4f} | "
                        f"{a['median_deg_step20']:.4f} | {a['p95_deg_step20']:.3f} | {a['p99_deg_step20']:.3f} | "
                        f"{a['equal_player_mean_deg_step20']:.4f} |")

    lines = [
        "# Benchmark: v5 / v6 / v7 and angular-loss models", "",
        f"Bench `group-a-v2-test20` ({ref['bench']['windows']:,} windows, 20 unseen players, cohort "
        f"`{next(iter(fingerprints))[:16]}`), machine `jacob-amd`, latency session `{session.name}`.", "",
        "## Size, compute and latency", "",
        "Latency: ONNX Runtime CPU, one call = one new sample, median / p99 in microseconds; alternating rounds in "
        "one session. Relative only, not the challenge hardware.", "",
        "| Model | Graph | Declared params | ONNX weight elements | MACs / call | State floats | Nodes | "
        "Off safe list | 1 thread median / p99 (µs) | 4 threads median (µs) | Deterministic, 1 = 4 threads |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
        *eff_rows, "",
        "## Accuracy (degrees, lower is better)", "",
        "Validation: 15 users, 3-seed mean ± SD; the selected seed is the ONNX file scored here (chosen on validation). "
        "Test: the selected seed only, step 20 = 200 ms, streamed from segment start; cold start = zero states at the "
        f"first full window. vs {REFERENCE}: paired per-player difference, bootstrap 95% CI, players better.", "",
        "| Model | Loss | Validation 3-seed | Seed | Test step 20 | vs v6 (test) | Test cold start | Test step 10 "
        "(100 ms) | Test median | Test p95 | Test p99 | Test equal-player |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *acc_rows]
    text = "\n".join(lines) + "\n"
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
