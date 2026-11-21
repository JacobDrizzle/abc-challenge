"""Screen declared ablations on validation, confirm the winner, never score test."""
import argparse
import json
from pathlib import Path
import re
import shutil

from baseline_pipeline import DEFAULT_H5, manifest_for, sha256_file
from jsonio import save_json
from run_desktop import run_experiment, SOURCE_FILES
from tcnn.configuration import normalize_config
from tune_damped_baseline import tune

ROOT = Path(__file__).resolve().parent


def experiment_configs(suite, base, recipe=None):
    if suite.get("version") != 1 or not suite.get("variants") or "control" not in suite["variants"]:
        raise ValueError("Invalid experiment suite")
    base = normalize_config({**base, "seeds": suite["seeds"]})
    if recipe is not None:
        # Isolate the data expansion effect using the original control and locked recipe.
        chosen = normalize_config(recipe["selected_config"])
        if chosen["seeds"] != base["seeds"]:
            raise ValueError("Recipe seed policy differs")
        return {"control": base, **({"selected_recipe": chosen} if chosen != base else {})}
    configs = {}
    for name, overrides in suite["variants"].items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", name):
            raise ValueError("Invalid variant name")
        configs[name] = normalize_config({**base, **overrides})
        if configs[name]["seeds"] != base["seeds"]:
            raise ValueError("Variants must share the same seeds")
    return configs


def run_suite(data, output, suite, base, *, stage="all", device="cuda", variants=None,
              recipe=None, synthetic=False):
    """data is the ray/ export's HDF5; its manifest_<run_id>.json next to it gives the split."""
    data, output = Path(data).resolve(), Path(output).resolve()
    configs = experiment_configs(suite, base, recipe)
    if variants and not set(variants).issubset(configs):
        raise ValueError("Unknown variant")
    if stage == "plan":
        print(json.dumps({"data": str(data), "output": str(output), "variants": configs,
                          "screen_seed": suite["seeds"][0], "confirmation_seeds": suite["seeds"],
                          "test_is_not_run": True}, indent=2))
        return
    split = manifest_for(data)
    protocol = {"suite": suite, "configs": configs, "data_sha256": sha256_file(data),
                "split_sha256": sha256_file(split), "device": device,
                "recipe": recipe, "code_sha256": {p: sha256_file(ROOT/p) for p in
                    (*SOURCE_FILES, "run_next.py", "tune_damped_baseline.py", "tcnn/damped.py")}}
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output/"suite.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Suite changed after starting; choose a fresh output directory")
    elif any(output.iterdir()):
        raise ValueError("Suite output is nonempty without a manifest")
    else:
        save_json(protocol_path, protocol)
    if stage in ("screen", "all"):
        tune(data, split, output/"damped_baseline", tuple(suite["baseline_lags"]), tuple(suite["baseline_tau_ms"]))
        for name, config in configs.items():
            if variants and name not in variants:
                continue
            print(f"Screening variant: {name}", flush=True)
            run_experiment(data, split, output/name, config, device, seeds=[suite["seeds"][0]],
                           selection_name="screen_selection.json", synthetic=synthetic)
    rows = []
    for name in configs:
        path = output/name/"screen_selection.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            row = saved["validation_candidates"][0]
            if saved["data_sha256"] != protocol["data_sha256"] or saved["split_sha256"] != protocol["split_sha256"]:
                raise ValueError("Screen result belongs to a different dataset")
            rows.append({"variant": name, **row})
    if rows:
        if len({r["cohort_fingerprint"] for r in rows}) != 1:
            raise ValueError("Variants used different evaluation windows")
        rows.sort(key=lambda r: (r["validation_endpoint_deg"], list(configs).index(r["variant"])))
        save_json(output/"comparison.json", {"complete": len(rows) == len(configs), "rows": rows})
        table = ["# Next-run validation comparison", "", "Lower step-20 mean error is better. Test was not evaluated.", "",
                 "| Variant | Seed | Mean endpoint error (degrees) |", "| --- | ---: | ---: |"]
        table += [f"| {r['variant']} | {r['seed']} | {r['validation_endpoint_deg']:.6f} |" for r in rows]
        (output/"comparison.md").write_text("\n".join(table)+"\n", encoding="utf-8")
    if stage in ("confirm", "all"):
        if len(rows) != len(configs):
            raise ValueError("Complete every declared screening variant before confirmation")
        winner = rows[0]["variant"]
        chosen = run_experiment(data, split, output/winner, configs[winner], device, synthetic=synthetic)
        lock = {**chosen, "selected_variant": winner, "selected_config": configs[winner],
                "screening": rows, "suite_sha256": sha256_file(protocol_path),
                "data": str(data), "damped_selection": "damped_baseline/selection.json"}
        path = output/"selection.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != lock or sha256_file(output/"selected_model.pt") != lock["checkpoint_sha256"]:
                raise ValueError("Final suite selection changed")
        else:
            temp = output/"selected_model.tmp"
            shutil.copyfile(output/winner/"selected_model.pt", temp)
            temp.replace(output/"selected_model.pt")
            save_json(path, lock)
        print(f"Confirmed {winner}. Model and selection saved in {output}. Test remains separate.", flush=True)
        return lock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("plan", "screen", "confirm", "all"), default="plan")
    p.add_argument("--data", type=Path, default=DEFAULT_H5, help="ray/ export HDF5; its manifest gives the split")
    p.add_argument("--output", type=Path, default=ROOT/"outputs/v5_group_a_v2")
    p.add_argument("--suite", type=Path, default=ROOT/"configs/v5_run.json")
    p.add_argument("--base-config", type=Path, default=ROOT/"configs/tcnn_v5.json")
    p.add_argument("--recipe-from", type=Path, help="Use control plus an already selected recipe on expanded data")
    p.add_argument("--variants", nargs="+", help="Run only these screen variants now; full confirmation still requires all")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run_suite(args.data, args.output, json.loads(args.suite.read_text()), json.loads(args.base_config.read_text()),
              stage=args.stage, device=args.device, variants=args.variants,
              recipe=json.loads(args.recipe_from.read_text()) if args.recipe_from else None)


if __name__ == "__main__":
    main()
