"""Compare a frozen checkpoint with all baselines; test split requires a locked selection."""
import argparse
from dataclasses import asdict
import json
import os
import torch
from pathlib import Path

from baseline_pipeline import DEFAULT_H5, EvaluationConfig, evaluate, manifest_for, sha256_file
from tcnn.data import check_corpus
from tcnn.runtime import configure_runtime
from jsonio import save_json

ROOT = Path(__file__).resolve().parent


def render_figures(report, output, data, checkpoint_path, selection_path, damped_selection, device):
    """Figures are a view of the saved results; a drawing failure never discards an evaluation."""
    try:
        from baseline_pipeline import EvaluationConfig as Config
        from tcnn.plots import experiment_files, render_evaluation
        from tcnn_model_adapter import TCNNPredictor
        histories, comparison = experiment_files(selection_path) if selection_path else ([], None)
        if not histories and (checkpoint_path.parent/"history.json").exists():
            histories = [checkpoint_path.parent/"history.json"]
        examples = {"tiny_tcnn": TCNNPredictor(checkpoint_path, Config(**report["config"]), device)}
        if damped_selection:
            from tcnn.damped import build_model
            examples["damped_velocity"] = build_model(Config(**report["config"]))
        figures = render_evaluation(output, data, histories=histories, comparison=comparison, examples=examples)
        print(f"Saved {len(figures)} figures and figures.md in {output}.", flush=True)
    except Exception as error:
        print(f"Warning: figures were not drawn: {error}", flush=True)


def evaluate_checkpoint(checkpoint_path, data, split_file, output, *, split="validation", selection_path=None,
                        damped_selection=None, device="cpu", batch_size=128, latency_steps=200, synthetic=False):
    checkpoint_path, data, split_file, output = map(Path, (checkpoint_path, data, split_file, output))
    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test")
    runtime = configure_runtime()
    audit = check_corpus(data, split_file, synthetic=synthetic,
                         minimum_users=3 if synthetic else 30, minimum_recordings=3 if synthetic else 100)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    provenance = checkpoint["provenance"]
    if (provenance["synthetic"] != synthetic or provenance["data_sha256"] != audit["data_sha256"]
            or provenance["split_sha256"] != audit["split_sha256"]):
        raise ValueError("Checkpoint was not trained on this frozen cohort and split")
    if split == "test":
        if selection_path is None:
            raise ValueError("Test evaluation requires the frozen validation selection")
        selection = json.loads(Path(selection_path).read_text())
        if selection["checkpoint_sha256"] != sha256_file(checkpoint_path):
            raise ValueError("Checkpoint differs from the locked selection")
        if selection["data_sha256"] != audit["data_sha256"] or selection["split_sha256"] != audit["split_sha256"]:
            raise ValueError("Dataset or split differs from the locked selection")
    os.environ["ABC_TCNN_CHECKPOINT"] = str(checkpoint_path.resolve())
    os.environ["ABC_TCNN_DEVICE"] = device
    models = ["tcnn_model_adapter:build_model"]
    if damped_selection:
        damped_selection = Path(damped_selection)
        damped = json.loads(damped_selection.read_text(encoding="utf-8"))
        if (damped["split"] != "validation" or damped["data_sha256"] != audit["data_sha256"]
                or damped["split_sha256"] != audit["split_sha256"]):
            raise ValueError("Damped baseline selection belongs to a different cohort")
        os.environ["ABC_DAMPED_SELECTION"] = str(damped_selection.resolve())
        models.append("tcnn.damped:build_model")
    config = EvaluationConfig(batch_size=batch_size)
    if output.exists():
        report = json.loads((output/"results.json").read_text(encoding="utf-8"))
        metadata = report["models"]["tiny_tcnn"]["metadata"]
        if (report["input"]["sha256"] != audit["data_sha256"]
                or report["input"]["split_sha256"] != audit["split_sha256"]
                or report["split"] != split or report["config"] != asdict(config)
                or metadata["checkpoint_sha256"] != sha256_file(checkpoint_path)
                or metadata.get("runtime") != runtime or metadata["device"] != device
                or report["environment"]["pipeline_sha256"] != sha256_file(ROOT/"baseline_pipeline.py")
                or ("damped_velocity" in report["models"]) != bool(damped_selection)):
            raise ValueError("Existing evaluation differs from the requested frozen experiment")
        if damped_selection and report["models"]["damped_velocity"]["metadata"]["selection_sha256"] != sha256_file(damped_selection):
            raise ValueError("Existing evaluation used a different damped baseline")
        print("Verified existing evaluation; results preserved.", flush=True)
    else:
        report = evaluate(data, split_file, output, config=config,
                          split_name=split, model_specs=models, latency_steps=latency_steps)
    render_figures(report, output, data, checkpoint_path, selection_path, damped_selection, device)
    if split == "test":
        receipt = {"test_evaluated": True, "split": "test", "synthetic": synthetic,
                   "selection_sha256": sha256_file(selection_path), "checkpoint_sha256": sha256_file(checkpoint_path),
                   "results_sha256": sha256_file(output/"results.json"), "runtime": runtime,
                   "cohort_fingerprint": report["cohort_fingerprint"], "windows": report["windows"]}
        path = output/"test_receipt.json"
        if path.exists() and json.loads(path.read_text()) != receipt:
            raise ValueError("Existing test receipt differs; results were preserved")
        if not path.exists():
            save_json(path, receipt)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=DEFAULT_H5)
    p.add_argument("--split-file", type=Path, help="Default: the manifest_<run_id>.json next to --data")
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--selection", type=Path, help="Required for test; produced by run_desktop.py or run_next.py")
    p.add_argument("--damped-selection", type=Path, help="Optional baseline tuned exclusively on validation")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--latency-steps", type=int, default=200)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    evaluate_checkpoint(args.checkpoint, args.data, args.split_file or manifest_for(args.data), args.output, split=args.split,
                        selection_path=args.selection, damped_selection=args.damped_selection,
                        device=args.device, batch_size=args.batch_size, latency_steps=args.latency_steps)


if __name__ == "__main__":
    main()
