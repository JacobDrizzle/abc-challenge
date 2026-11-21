"""Train declared seeds, compare full validation, and lock a checkpoint for testing."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import torch
from baseline_pipeline import DEFAULT_H5, EvaluationConfig, evaluate, manifest_for, sha256_file
from jsonio import save_json
from tcnn.configuration import normalize_config
from tcnn.data import check_corpus
from tcnn.runtime import configure_runtime
from train_tcnn import train

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ("train_tcnn.py", "run_desktop.py", "tcnn/model.py", "tcnn/data.py",
                "baseline_pipeline.py", "tcnn_model_adapter.py", "tcnn/configuration.py", "tcnn/runtime.py")


def run_experiment(data, split, output, config, device="cuda", *, seeds=None,
                   selection_name="selection.json", synthetic=False):
    config = normalize_config(config)
    seeds = config["seeds"] if seeds is None else seeds
    if not seeds or any(s not in config["seeds"] for s in seeds):
        raise ValueError("Requested seed is outside the frozen experiment")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    runtime = configure_runtime()
    data, split, output = Path(data).resolve(), Path(split).resolve(), Path(output).resolve()
    audit = check_corpus(data, split, synthetic=synthetic,
                         minimum_users=3 if synthetic else 30, minimum_recordings=3 if synthetic else 100)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"data_sha256": audit["data_sha256"], "split_sha256": audit["split_sha256"],
                "config": config, "device": str(device), "runtime": runtime,
                "code_sha256": {name: sha256_file(ROOT/name) for name in SOURCE_FILES}}
    protocol_path = output / "experiment.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing experiment uses different data, code, device or settings. Choose a new output directory.")
    else:
        if any(output.iterdir()):
            raise ValueError("Output contains files but no experiment manifest")
        save_json(protocol_path, protocol)
    selection_path = output / selection_name
    checkpoint_name = "selected_model.pt" if selection_name == "selection.json" else "screen_model.pt"
    if selection_path.exists():
        locked = json.loads(selection_path.read_text(encoding="utf-8"))
        if sha256_file(output/checkpoint_name) != locked["checkpoint_sha256"] or locked["evaluated_seeds"] != seeds:
            raise ValueError("Locked selection differs from requested run")
        return locked
    scores = []
    for seed in seeds:
        run = output / f"seed_{seed}"
        if run.exists() and not (run/"last.pt").exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            run.rename(output/f"seed_{seed}_interrupted_{stamp}")
        checkpoint = train(data, split, run, config, seed, device, resume=run.exists(), synthetic=synthetic)
        valdir = run / "full_validation"
        checkpoint_sha = sha256_file(checkpoint)
        if valdir.exists():
            report = json.loads((valdir/"results.json").read_text(encoding="utf-8"))
            metadata = report["models"]["tiny_tcnn"]["metadata"]
            if (metadata["checkpoint_sha256"] != checkpoint_sha or metadata.get("runtime") != runtime
                    or report["input"]["sha256"] != audit["data_sha256"]
                    or report["input"]["split_sha256"] != audit["split_sha256"]
                    or report["environment"]["pipeline_sha256"] != protocol["code_sha256"]["baseline_pipeline.py"]
                    or report["config"] != dict(context=400, horizon=20, rate_hz=100., stride=1, batch_size=config["batch_size"])
                    or report["split"] != "validation"):
                raise ValueError("Saved validation report does not match this experiment")
        else:
            os.environ["ABC_TCNN_CHECKPOINT"] = str(checkpoint)
            os.environ["ABC_TCNN_DEVICE"] = str(device)
            report = evaluate(data, split, valdir, config=EvaluationConfig(batch_size=config["batch_size"]),
                              model_specs=["tcnn_model_adapter:build_model"], latency_steps=200)
        scores.append({"seed": seed, "checkpoint": f"seed_{seed}/best.pt", "checkpoint_sha256": checkpoint_sha,
                       "validation_endpoint_deg": report["models"]["tiny_tcnn"]["mean_endpoint_deg"],
                       "hold_last_endpoint_deg": report["models"]["hold_last"]["mean_endpoint_deg"],
                       "cohort_fingerprint": report["cohort_fingerprint"]})
    winner = min(scores, key=lambda item: (item["validation_endpoint_deg"], item["seed"]))
    temporary = output / (checkpoint_name + ".tmp")
    shutil.copyfile(output/winner["checkpoint"], temporary)
    temporary.replace(output/checkpoint_name)
    selection = {"selection_rule": "Lowest full stride-1 validation mean step-20 geodesic error; seed breaks ties",
                 "selected_seed": winner["seed"], "checkpoint_sha256": winner["checkpoint_sha256"],
                 "data_sha256": audit["data_sha256"], "split_sha256": audit["split_sha256"],
                 "validation_candidates": scores, "evaluated_seeds": seeds, "runtime": runtime,
                 "test_receipt": "Written separately after successful test evaluation"}
    save_json(selection_path, selection)
    print(f"Selected seed {winner['seed']}: validation endpoint {winner['validation_endpoint_deg']:.4f} degrees.", flush=True)
    return selection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT/"configs/tcnn_v5.json")
    p.add_argument("--data", type=Path, default=DEFAULT_H5)
    p.add_argument("--split-file", type=Path, help="Default: the manifest_<run_id>.json next to --data")
    args = p.parse_args()
    run_experiment(args.data, args.split_file or manifest_for(args.data), args.output, json.loads(args.config.read_text()), args.device)


if __name__ == "__main__":
    main()
