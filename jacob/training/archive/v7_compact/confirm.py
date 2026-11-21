"""Confirm v7 candidates on every seed and compare their 3-seed validation means with the plan's tolerance.

python -m experimental.v7_compact.confirm [--candidates NAME ...] [--dry-run]

Plan rule (26 Oct): the finalist is the smallest candidate whose 3-seed validation mean is within the tolerance
(v6 control's 4.342 + 0.04 = 4.382 degrees). Each candidate trains through run_desktop.run_experiment, as the frozen
runner's confirm stage does: finished seeds are reused (a screened candidate is confirmed in its screen folder), and
the candidate's best seed is locked in selected_model.pt. This script only reports. Choosing the submission stays
with the team, after the hardware latencies. Test users are never read.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from baseline_pipeline import DEFAULT_H5, manifest_for
from jsonio import save_json
import run_desktop
import run_next

from experimental.v7_compact.hooks import installed
from experimental.v7_compact.model import load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def candidate_configs(plan, base):
    """Each candidate's config, built exactly as the screen built it (experiment_configs requires a control)."""
    suite = {"version": 1, "seeds": plan["seeds"],
             "variants": {"control": {}, **{name: c["overrides"] for name, c in plan["candidates"].items()}}}
    configs = run_next.experiment_configs(suite, base)
    return {name: configs[name] for name in plan["candidates"]}


def locked_result(folder, seeds):
    """Params, per-seed validation errors and selected seed of a confirmed folder, or None if not confirmed yet."""
    path = folder / "selection.json"
    if not path.exists():
        return None
    locked = json.loads(path.read_text(encoding="utf-8"))
    if locked["evaluated_seeds"] != seeds:
        raise ValueError(f"{path} covers seeds {locked['evaluated_seeds']}, not {seeds}")
    model = load_checkpoint(folder / "selected_model.pt")[0]
    candidates = locked["validation_candidates"]
    return {"params": sum(p.numel() for p in model.parameters()), "selected_seed": locked["selected_seed"],
            "scores": {str(c["seed"]): c["validation_endpoint_deg"] for c in candidates},
            "cohort_fingerprints": sorted({c["cohort_fingerprint"] for c in candidates})}


def summarise(results, tolerance):
    """3-seed statistics per candidate, and the plan rule's finalist (None if no candidate is within tolerance)."""
    rows = []
    for name, result in results.items():
        scores = np.array(list(result["scores"].values()))
        rows.append({"candidate": name, **result, "mean_deg": float(scores.mean()),
                     "std_deg": float(scores.std(ddof=1)), "within_tolerance": bool(scores.mean() <= tolerance)})
    passing = [row for row in rows if row["within_tolerance"]]
    finalist = min(passing, key=lambda row: (row["params"], row["mean_deg"]))["candidate"] if passing else None
    return {"tolerance_deg": tolerance, "rows": rows, "plan_rule_finalist": finalist}


def report(summary, reference, seeds):
    lines = ["# v7 confirmation: validation, every seed", "",
             "Step-20 mean geodesic error over all validation windows (degrees). Test was not evaluated. "
             f"Plan tolerance for the 3-seed mean: {summary['tolerance_deg']}.", "",
             "| Candidate | Params | " + " | ".join(f"Seed {s}" for s in seeds) + " | Mean | SD | Within tolerance | "
             "Selected seed |", "| --- | ---: | " + "---: | " * len(seeds) + "---: | ---: | --- | ---: |"]
    for row in [*([reference] if reference else []), *summary["rows"]]:
        verdict = "reference" if row is reference else "yes" if row["within_tolerance"] else "no"
        lines.append(f"| {row['candidate']} | {row['params']:,} | " +
                     " | ".join(f"{row['scores'][str(s)]:.4f}" for s in seeds) +
                     f" | {row['mean_deg']:.4f} | {row['std_deg']:.4f} | {verdict} | {row['selected_seed']} |")
    finalist = summary["plan_rule_finalist"]
    lines += ["", f"Plan rule (smallest candidate within tolerance): **{finalist}**." if finalist else
              "Plan rule: no candidate is within tolerance; fall back to v6 control seed 44.",
              "This is the plan's rule, not a decision: the submission is chosen by the team after the hardware "
              "latencies."]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", type=Path, default=HERE/"configs/v7_confirm.json")
    p.add_argument("--base-config", type=Path, default=HERE/"configs/tcnn_v7.json")
    p.add_argument("--data", type=Path, default=DEFAULT_H5)
    p.add_argument("--output", type=Path, default=ROOT/"outputs/v7_confirm", help="Where the comparison is written")
    p.add_argument("--candidates", nargs="+", help="Train only these now (default: all); the report covers all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true", help="Print each candidate's folder and config, then stop")
    args = p.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("version") != 1 or not set(args.candidates or []).issubset(plan["candidates"]):
        p.error("Unknown plan version or candidate")
    with installed():
        configs = candidate_configs(plan, json.loads(args.base_config.read_text(encoding="utf-8")))
        if args.dry_run:
            print(json.dumps({name: {"output": c["output"], "config": configs[name]}
                              for name, c in plan["candidates"].items()}, indent=2))
            return
        split = manifest_for(args.data)
        for name in args.candidates or plan["candidates"]:
            print(f"Confirming {name} on seeds {plan['seeds']}", flush=True)
            run_desktop.run_experiment(args.data, split, ROOT/plan["candidates"][name]["output"], configs[name],
                                       args.device)

    results = {name: locked_result(ROOT/c["output"], plan["seeds"]) for name, c in plan["candidates"].items()}
    results = {name: result for name, result in results.items() if result is not None}
    reference = locked_result(ROOT/plan["reference"], plan["seeds"])
    if len({f for r in [*results.values(), reference] if r for f in r["cohort_fingerprints"]}) > 1:
        raise ValueError("Candidates were scored on different validation windows")
    summary = summarise(results, plan["tolerance_deg"])
    if reference:
        reference = summarise({"v6 control (reference)": reference}, plan["tolerance_deg"])["rows"][0]
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output/"confirmation.json", {"plan": plan, "reference": reference, **summary})
    (args.output/"confirmation.md").write_text(report(summary, reference, plan["seeds"]), encoding="utf-8")
    print(report(summary, reference, plan["seeds"]), flush=True)


if __name__ == "__main__":
    main()
