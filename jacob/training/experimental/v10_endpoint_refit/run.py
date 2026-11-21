"""Run the frozen 97-coefficient refit protocol in a NEW output directory."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import h5py
import numpy as np
import torch

from baseline_pipeline import EvaluationConfig, inventory, sha256_file, unit_quaternions
from best.model import load_checkpoint
from tcnn.runtime import configure_runtime
from train_tcnn import geodesic_degrees_torch
from experimental.v10_weight_edit.metrics import ErrorAccumulator, paired_comparison, promising_gate
from experimental.v10_weight_edit.run import TRAINING, REPO, SOURCE, DATA, SPLIT, save, log
from .fit import PENALTIES, STRENGTHS, NormalEquations, apply_delta, extract, predict, candidate, candidates

PREVIOUS = TRAINING / "outputs/v10_weight_edit_20261104"
CONTROLS = ("identity", "global_105")


def restricted_batches(data, rows, split, device, stride, batch_size):
    """No test access; fitting and evaluation explicitly request different splits."""
    if split not in ("train", "validation") or any(not r["path"].startswith(split + "/") for r in rows):
        raise ValueError("Requested rows must all belong to the permitted train/validation split")
    span = torch.arange(420, device=device)
    with h5py.File(data, "r") as h5:
        for row in rows:
            group = h5[row["path"]]
            times = np.asarray(group["time_s"], dtype=np.float64)
            if not np.isfinite(times).all() or not np.allclose(np.diff(times), .01, rtol=0, atol=1e-6):
                raise ValueError(f"Invalid time grid: {row['key']}")
            q = unit_quaternions(group["quaternion_wxyz"][:], source=row["key"]).astype(np.float32)
            if np.any(np.sum(q[1:] * q[:-1], axis=-1) < -1e-8):
                raise ValueError(f"Discontinuous signs: {row['key']}")
            buffer = torch.from_numpy(q).to(device)
            starts = torch.arange(0, len(q) - 419, stride, device=device)
            for begin in range(0, len(starts), batch_size):
                indices = starts[begin:begin + batch_size]
                values = buffer[indices[:, None] + span]
                yield indices, values[:, :400], values[:, 400:]


def prepare(output, batch_size):
    output.mkdir(parents=True, exist_ok=False)
    for name in ("originals", "cache", "candidates"):
        (output / name).mkdir()
    selection = json.loads((SOURCE / "selection.json").read_text())
    previous = json.loads((PREVIOUS / "manifest.json").read_text())
    for file, key in ((DATA, "data_sha256"), (SPLIT, "split_sha256")):
        if sha256_file(file) != selection[key] or previous[key] != selection[key]:
            raise ValueError("Data provenance differs between model and experiments")
    split_manifest = json.loads(SPLIT.read_text())
    cohorts, rows = {}, {}
    with h5py.File(DATA, "r") as h5:
        for split, expected in (("train", 800), ("validation", 120)):
            users = sorted(h5[split])
            if len(users) != expected or users != sorted(split_manifest["splits"][split]):
                raise ValueError(f"Unexpected {split} users")
            included, skipped = inventory(h5, users, EvaluationConfig(), False)
            if skipped or {r["user"] for r in included} != set(users):
                raise ValueError("Missing eligible users or segments")
            cohorts[split] = included
            rows[split] = defaultdict(list)
            for row in included:
                rows[split][row["user"]].append(row)
    discovery, confirmation = previous["discovery_users"], previous["confirmation_users"]
    if len(discovery) != 80 or len(confirmation) != 40 or set(discovery) & set(confirmation) or set(discovery + confirmation) != set(rows["validation"]):
        raise ValueError("The previous 80/40 development partition has changed")
    train_count = sum((r["samples"] - 420) // 20 + 1 for r in cohorts["train"])
    validation_count = sum(r["windows"] for r in cohorts["validation"])
    if train_count != 1620028 or validation_count != 4755761:
        raise ValueError("Unexpected frozen-window counts")
    sources = {}
    for item in selection["validation_candidates"]:
        seed = item["seed"]
        source = SOURCE / ("selected_model.pt" if seed == 44 else item["checkpoint"])
        digest = sha256_file(source)
        if digest != item["checkpoint_sha256"]:
            raise ValueError("Original checkpoint hash mismatch")
        copy = output / "originals" / f"seed_{seed}.pt"
        shutil.copy2(source, copy)
        if sha256_file(copy) != digest:
            raise ValueError("Checkpoint copy verification failed")
        sources[str(seed)] = {"source": str(source), "copy": str(copy), "sha256": digest}
    dependencies = [TRAINING / p for p in ("baseline_pipeline.py", "train_tcnn.py", "tcnn/model.py", "tcnn/runtime.py",
                    "tcnn/onnx_export.py", "best/model.py", "best/export.py")]
    files = [*Path(__file__).parent.glob("*.py"), *(TRAINING / "experimental/v10_weight_edit").glob("*.py"), *dependencies]
    manifest = {
        "version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
        "data_sha256": selection["data_sha256"], "split_sha256": selection["split_sha256"],
        "sources": sources, "train_users": sorted(rows["train"]), "train_stride": 20, "train_examples": train_count,
        "discovery_users": discovery, "confirmation_users": confirmation, "validation_windows": validation_count,
        "penalties": PENALTIES, "strengths": STRENGTHS, "rms_cap_fraction": .1,
        "angle_weight_floor_rad": float(np.pi / 180), "feature_rms_floor": 1e-6,
        "training_performed": "Regularized endpoint coefficient fit only; no backpropagation",
        "batch_size": batch_size, "runtime": configure_runtime(), "test_samples_evaluated": False,
        "source_code_sha256": {str(p.relative_to(TRAINING)): sha256_file(p) for p in sorted(set(files))},
        "previous_artifact_sha256": {str(p): sha256_file(p) for p in sorted(PREVIOUS.iterdir()) if p.is_file()},
        "cohort_sha256": hashlib.sha256(json.dumps(cohorts, sort_keys=True).encode()).hexdigest(),
        "selection_rule": "Endpoint mean, then smaller training RMS correction, then identifier; stop if a control wins",
        "evaluation_rule": "Only the frozen finalist; same penalty, strength and fitting rule on all seeds",
        "interpretation": "Both validation subsets are reused development data; all evidence is exploratory",
    }
    save(output / "manifest.json", manifest)
    save(output / "cohorts.json", cohorts)
    return manifest, rows


@torch.inference_mode()
def baseline_cache(output, manifest, rows, device, batch_size):
    model, _ = load_checkpoint(output / "originals/seed_44.pt")
    model.to(device)
    acc = ErrorAccumulator()
    max_difference = 0.
    for i, user in enumerate(sorted(rows)):
        arrays = defaultdict(list)
        checked = False
        for starts, history, target in restricted_batches(DATA, rows[user], "validation", device, 1, batch_size):
            parts = extract(model, history)
            prediction = predict(parts)
            if not checked:
                difference = float((prediction - model(history)).abs().max())
                if difference > 1e-6:
                    raise ValueError("Original cached prediction does not match actual inference")
                max_difference = max(max_difference, difference)
                checked = True
            acc.update(user, geodesic_degrees_torch(prediction, target).cpu().numpy())
            if user in manifest["discovery_users"]:
                keep = starts.remainder(5) == 0
                for name, value in {**parts, "target": target}.items():
                    arrays[name].append(value[keep].cpu().numpy())
        if arrays:
            np.savez(output / "cache" / f"{user}.npz", **{name: np.concatenate(values) for name, values in arrays.items()})
        if (i + 1) % 5 == 0:
            log(f"Baseline: {i + 1}/120 users")
    summary = acc.summary()
    summary["recorded_error_deg"] = 4.403628579547263
    summary["difference_from_recorded_deg"] = summary["mean_endpoint_deg"] - summary["recorded_error_deg"]
    summary["max_cached_component_difference"] = max_difference
    save(output / "baseline_reproduction.json", summary)
    if abs(summary["difference_from_recorded_deg"]) > .001:
        raise ValueError("Baseline reproduction failed")
    log(f"Baseline reproduced: {summary['mean_endpoint_deg']:.9f} degrees")


@torch.inference_mode()
def fit_seed(output, manifest, seed, rows, device, batch_size):
    model, _ = load_checkpoint(output / "originals" / f"seed_{seed}.pt")
    model.to(device)
    equations = NormalEquations(device)
    for i, user in enumerate(manifest["train_users"]):
        for _, history, target in restricted_batches(DATA, rows[user], "train", device, 20, batch_size):
            equations.update_parts(extract(model, history), target)
        if (i + 1) % 50 == 0:
            log(f"Fit statistics seed{seed}: {i + 1}/800 training users")
    arrays = equations.arrays()
    if int(arrays["count"]) != manifest["train_examples"]:
        raise ValueError("Fitting sample count changed")
    np.savez(output / f"training_statistics_seed_{seed}.npz", **arrays)
    save(output / f"fit_seed_{seed}.json", {"seed": seed, "train_examples": int(arrays["count"]),
         "train_users": len(rows), "residual_rms": float(arrays["residual_rms"]),
         "statistics_sha256": sha256_file(output / f"training_statistics_seed_{seed}.npz"),
         "fit_target": "200 ms endpoint", "fitting_split": "train", "feature_extractor_frozen": True})
    return arrays


def save_candidate(output, seed, definition):
    source = output / "originals" / f"seed_{seed}.pt"
    original, state = load_checkpoint(source)
    edited = apply_delta(original, definition["delta"])
    folder = output / "candidates" / f"seed_{seed}"
    folder.mkdir(exist_ok=True)
    path = folder / (definition["id"] + ".pt")
    if path.exists():
        raise FileExistsError(path)
    torch.save({"model_spec": state["model_spec"], "model_state": edited.state_dict(),
                "endpoint_refit": {"candidate": definition, "source_sha256": sha256_file(source),
                                   "fitting_split": "train", "training_performed": True,
                                   "backpropagation_performed": False}}, path)
    return path


@torch.inference_mode()
def screen(output, manifest, definitions, device, batch_size):
    acc = {item["id"]: ErrorAccumulator() for item in definitions}
    for i, user in enumerate(manifest["discovery_users"]):
        with np.load(output / "cache" / f"{user}.npz", allow_pickle=False) as file:
            arrays = {k: file[k] for k in file.files}
        for start in range(0, len(arrays["target"]), batch_size):
            parts = {key: torch.from_numpy(value[start:start + batch_size]).to(device) for key, value in arrays.items()}
            target = parts.pop("target")
            for item in definitions:
                prediction = predict(parts, item.get("delta"), item.get("gain", 1.))
                acc[item["id"]].update(user, geodesic_degrees_torch(prediction, target).cpu().numpy())
        if (i + 1) % 10 == 0:
            log(f"Screen: {i + 1}/80 users, 21 fits and two controls")
    summaries = {name: value.summary() for name, value in acc.items()}
    ranking = sorted(definitions, key=lambda x: (summaries[x["id"]]["mean_endpoint_deg"], x["train_rms_correction"], x["id"]))
    save(output / "screen.json", {"stride": 5, "ranking": ranking, "results": summaries})
    log(f"Screen winner: {ranking[0]['id']}")
    return ranking[0]


def compare_summaries(summaries, label):
    comparisons = {name: paired_comparison(summaries["candidate"], summaries[name]) for name in CONTROLS}
    gate = None
    if label == "confirmation":
        gate = promising_gate(summaries["candidate"], summaries["identity"], comparisons["identity"])
        gate["checks"]["beats_previous_gain_endpoint"] = comparisons["global_105"]["mean_diff_deg"] < 0
        gate["passed"] = all(gate["checks"].values())
    return comparisons, gate


@torch.inference_mode()
def evaluate(output, seed, definition, users, rows, device, batch_size, label):
    original, _ = load_checkpoint(output / "originals" / f"seed_{seed}.pt")
    edited, _ = load_checkpoint(output / "candidates" / f"seed_{seed}" / (definition["id"] + ".pt"))
    original.to(device)
    edited.to(device)
    acc = {name: ErrorAccumulator() for name in (*CONTROLS, "candidate")}
    max_difference = 0.
    for i, user in enumerate(users):
        checked = False
        for _, history, target in restricted_batches(DATA, rows[user], "validation", device, 1, batch_size):
            parts = extract(original, history)
            predictions = {"identity": predict(parts), "global_105": predict(parts, gain=1.05), "candidate": edited(history)}
            if not checked:
                difference = float((predictions["candidate"] - predict(parts, definition["delta"])).abs().max())
                original_difference = float((predictions["identity"] - original(history)).abs().max())
                if difference > 3e-6 or original_difference > 1e-6:
                    raise ValueError("Cached and actual model predictions disagree")
                max_difference = max(max_difference, difference)
                checked = True
            for name, prediction in predictions.items():
                acc[name].update(user, geodesic_degrees_torch(prediction, target).cpu().numpy())
        if (i + 1) % 5 == 0:
            log(f"{label} seed{seed}: {i + 1}/{len(users)} users")
    summaries = {name: value.summary() for name, value in acc.items()}
    comparisons, gate = compare_summaries(summaries, label)
    result = {"seed": seed, "stride": 1, "results": summaries, "comparisons": comparisons, "gate": gate,
              "max_candidate_cached_component_difference": max_difference, "interpretation": "Exploratory reused validation data"}
    save(output / f"{label}_seed_{seed}.json", result)
    log(f"{label} seed{seed}: delta vs original {comparisons['identity']['mean_diff_deg']:+.9f} degrees")
    return result


def verify_preservation(output, manifest):
    sources = {seed: {kind: sha256_file(item[kind]) == item["sha256"] for kind in ("source", "copy")}
               for seed, item in manifest["sources"].items()}
    code = {path: sha256_file(TRAINING / path) == digest for path, digest in manifest["source_code_sha256"].items()}
    previous = {path: sha256_file(path) == digest for path, digest in manifest["previous_artifact_sha256"].items()}
    audit = {"originals": sources, "source_code": code, "previous_experiment": previous}
    save(output / "preservation.json", audit)
    if not (all(all(values.values()) for values in sources.values()) and all(code.values()) and all(previous.values())):
        raise ValueError("Preservation or source-code check failed")


def run(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    output = Path(args.output).resolve()
    manifest, rows = prepare(output, args.batch_size)
    start = time.monotonic()
    report = {"status": "running", "test_samples_evaluated": False, "interpretation": "Exploratory development experiment"}
    save(output / "status.json", report)
    try:
        baseline_cache(output, manifest, rows["validation"], args.device, args.batch_size)
        arrays = fit_seed(output, manifest, 44, rows["train"], args.device, args.batch_size)
        definitions = candidates(arrays)
        for item in definitions:
            save_candidate(output, 44, item)
        definitions = [{"id": "identity", "gain": 1., "train_rms_correction": 0.},
                       {"id": "global_105", "gain": 1.05, "train_rms_correction": .05 * float(arrays["residual_rms"])}, *definitions]
        save(output / "candidates.json", definitions)
        winner = screen(output, manifest, definitions, args.device, args.batch_size)
        report["winner"] = winner
        if winner["id"] in CONTROLS:
            report["status"] = "control_won_screen"
            return
        full = evaluate(output, 44, winner, manifest["discovery_users"], rows["validation"], args.device, args.batch_size, "discovery")
        if any(full["comparisons"][name]["mean_diff_deg"] >= 0 for name in CONTROLS):
            report["status"] = "no_full_discovery_improvement"
            return
        lock = {"candidate": winner, "locked_utc": datetime.now(timezone.utc).isoformat(), "further_tuning_allowed": False,
                "seed_transfer": "Refit training features using this same penalty, strength and RMS cap; never retune"}
        save(output / "frozen_fit.json", lock)
        shutil.copy2(output / "candidates/seed_44" / (winner["id"] + ".pt"), output / "candidate_model.pt")
        results = {44: evaluate(output, 44, winner, manifest["confirmation_users"], rows["validation"], args.device, args.batch_size, "confirmation")}
        for seed in (42, 43):
            arrays = fit_seed(output, manifest, seed, rows["train"], args.device, args.batch_size)
            fitted = candidate(arrays, winner["penalty"], winner["strength"])
            save_candidate(output, seed, fitted)
            save(output / f"frozen_fit_seed_{seed}.json", fitted)
            results[seed] = evaluate(output, seed, fitted, manifest["confirmation_users"], rows["validation"], args.device, args.batch_size, "confirmation")
        report["confirmation"] = {str(seed): {"comparisons": result["comparisons"], "gate": result["gate"]} for seed, result in results.items()}
        passed = results[44]["gate"]["passed"] and all(results[s]["comparisons"]["identity"]["mean_diff_deg"] < 0 for s in (42, 43))
        report["status"] = "promising_exploratory" if passed else "not_promising"
        if passed:
            from experimental.v10_weight_edit.export_check import verify_exports
            report["exports"] = verify_exports(output / "originals/seed_44.pt", output / "candidate_model.pt", output / "onnx")
            save(output / "export_checks.json", report["exports"])
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        try:
            verify_preservation(output, manifest)
        except Exception as exc:
            report.update(status="failed", preservation_error=str(exc))
            raise
        finally:
            save(output / "status.json", report)
            log(f"Endpoint refit status: {report['status']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    run(args)


if __name__ == "__main__":
    main()
