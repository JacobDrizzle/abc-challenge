"""Confirm v9 candidates on every seed and compare their 3-seed validation means with the reference's.

python -m experimental.v9_lean.confirm --plan experimental/v9_lean/configs/<plan>.json [--candidates ...] [--dry-run]

Each candidate trains through run_desktop.run_experiment, as v7's confirm.py: finished seeds are reused (a screened
candidate is confirmed in its screen folder) and the best seed is locked in selected_model.pt. Plan rule (28 Oct):
a candidate improves on the reference (sep_base3_addons + angular) if its 3-seed mean is lower; the paired per-user
comparison is reported with the analysis. This script only reports: the submission choice stays with the team.
Test users are never read.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from baseline_pipeline import DEFAULT_H5, manifest_for
from jsonio import save_json
import run_desktop

from experimental.v7_compact.confirm import candidate_configs
from experimental.v9_lean.hooks import installed
from experimental.v9_lean.model import load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


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


def summarise(name, result, reference_mean=None):
    scores = np.array(list(result["scores"].values()))
    mean = float(scores.mean())
    return {"candidate": name, **result, "mean_deg": mean, "std_deg": float(scores.std(ddof=1)),
            "delta_vs_reference_deg": None if reference_mean is None else mean - reference_mean}


def report(rows, reference, seeds):
    lines = ["# v9 confirmation: validation, every seed", "",
             "Step-20 mean geodesic error over all validation windows (degrees). Test was not evaluated.", "",
             "| Candidate | Params | " + " | ".join(f"Seed {s}" for s in seeds) + " | Mean | SD | vs reference | "
             "Selected seed |", "| --- | ---: | " + "---: | " * len(seeds) + "---: | ---: | ---: | ---: |"]
    for row in [*([reference] if reference else []), *rows]:
        delta = "reference" if row is reference else f"{row['delta_vs_reference_deg']:+.4f}"
        lines.append(f"| {row['candidate']} | {row['params']:,} | " +
                     " | ".join(f"{row['scores'][str(s)]:.4f}" for s in seeds) +
                     f" | {row['mean_deg']:.4f} | {row['std_deg']:.4f} | {delta} | {row['selected_seed']} |")
    lines += ["", "A lower 3-seed mean is the plan's bar, not a decision: the paired per-user comparison and the "
              "submission choice follow, with the team."]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--base-config", type=Path, default=HERE/"configs/tcnn_v9.json")
    p.add_argument("--data", type=Path, default=DEFAULT_H5)
    p.add_argument("--output", type=Path, default=ROOT/"outputs/v9_confirm", help="Where the comparison is written")
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
    if reference:
        reference = summarise(f"{Path(plan['reference']).name} (reference)", reference)
    rows = [summarise(name, result, reference and reference["mean_deg"]) for name, result in results.items()]
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output/"confirmation.json", {"plan": plan, "reference": reference, "rows": rows})
    (args.output/"confirmation.md").write_text(report(rows, reference, plan["seeds"]), encoding="utf-8")
    print(report(rows, reference, plan["seeds"]), flush=True)


if __name__ == "__main__":
    main()
