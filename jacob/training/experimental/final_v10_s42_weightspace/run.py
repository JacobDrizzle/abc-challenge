"""One approved fit, three strengths, two controls, and one frozen assessment.

Run with `--pilot` first. The main run requires the passing parity/tests/pilot
records and refuses to reuse an execution directory. Historical runners are
not imported; their pure fitting and metric functions are reused unchanged.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import h5py
import numpy as np
import torch

from baseline_pipeline import EvaluationConfig, inventory
from best.model import GravityMirrorTCNN, load_checkpoint
from experimental.v10_endpoint_refit.fit import NormalEquations, candidate, predict
from experimental.v10_weight_edit.metrics import ErrorAccumulator, paired_comparison
from tcnn.runtime import configure_runtime
from train_tcnn import geodesic_degrees_torch
from .common import (DATA, DATA_SHA, MANIFEST, MANIFEST_SHA, OLD_MANIFEST, ORIGINAL, OUTPUT, ROOT,
                     TRAINING, REPO, PENALTY, STRENGTHS, MAX_SECONDS, ANGLE_TOLERANCE,
                     assessment_gate, check_originals, log, save, sha256, validation_partition, window_count)
from .engine import CHUNK, batches
from .recover import recover, edited_model, save_inference, write_patch
from .verify import angles, latency_pair, verify_model


def corpus():
    manifest = json.loads(MANIFEST.read_text())
    splits = {k: set(v) for k, v in manifest["splits"].items()}
    if any(splits[a] & splits[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Split overlap")
    old = json.loads(OLD_MANIFEST.read_text())["splits"]["validation"]
    discovery, assessment = validation_partition(splits["validation"], old)
    rows = {}
    with h5py.File(DATA, "r") as handle:
        for split in ("train", "validation"):
            if set(handle[split]) != splits[split]:
                raise ValueError("HDF5 membership mismatch")
            included, skipped = inventory(handle, sorted(splits[split]), EvaluationConfig(), False)
            if skipped or {r["user"] for r in included} != splits[split]:
                raise ValueError("Ineligible segments/users")
            rows[split] = included
    rows["discovery"] = [r for r in rows["validation"] if r["user"] in set(discovery)]
    rows["assessment"] = [r for r in rows["validation"] if r["user"] in set(assessment)]
    if window_count(rows["train"], 20) != 10237229 or window_count(rows["validation"], 1) != 30406800:
        raise ValueError("Final corpus counts changed")
    return rows, {"train": sorted(splits["train"]), "validation": sorted(splits["validation"]),
                  "discovery": discovery, "assessment": assessment}


def protocol_record(rows, users):
    return {"arm": "endpoint_only", "approved_by": "User: Please proceed, after endpoint-only recommendation",
            "source": "ONNX-derived inference weights; original training checkpoint unavailable",
            "penalty": PENALTY, "strengths": list(STRENGTHS), "rms_cap_fraction": 0.1,
            "minimum_gain_degrees": 0.005, "bootstrap_draws": 10000, "bootstrap_seed": 0,
            "train_stride": 20, "screen_stride": 5, "assessment_stride": 1,
            "maximum_main_run_seconds": MAX_SECONDS, "chunk_positions": CHUNK,
            "partition_salt": "final-v10-s42-weightspace-v1:", "users": users,
            "counts": {k: {"users": len(users[k]), "segments": len(v),
                           "samples": sum(r["samples"] for r in v),
                           "stride1": window_count(v, 1), "stride5": window_count(v, 5),
                           "stride20": window_count(v, 20)} for k, v in rows.items()},
            "selection": "Discovery endpoint mean, smaller training RMS correction, identifier; stop if control wins",
            "inference": "Dense float32, complete temporal samples, segment boundaries, 400-position left overlap",
            "targets": "Input HDF5 normalized in float64; model inputs float32; geodesic metrics float64",
            "assessment": "One frozen finalist only; no retuning", "test_access": "No test samples",
            "interpretation": "Reused validation/development data; original final-model selection history unavailable"}


def source_hashes():
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [TRAINING / p for p in (
        "baseline_pipeline.py", "best/model.py", "best/export.py", "train_tcnn.py", "tcnn/model.py",
        "tcnn/runtime.py", "tcnn/onnx_export.py", "experimental/v10_endpoint_refit/fit.py",
        "experimental/v10_weight_edit/metrics.py", "experimental/v10_weight_edit/export_check.py")]
    return {str(p.relative_to(REPO)).replace("\\", "/"): sha256(p) for p in sorted(paths)}


def checkpoint_statistics(path, equations):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **equations.arrays())
    temporary.replace(path)


@torch.inference_mode()
def throughput_pilot():
    check_originals()
    if not json.loads((ROOT / "preflight/parity.json").read_text())["passed"]:
        raise ValueError("Parity prerequisite missing")
    rows, users = corpus()
    # Deterministically sample training segments, with no validation feedback.
    selected = sorted(rows["train"], key=lambda r: __import__("hashlib").sha256(r["key"].encode()).hexdigest())[:32]
    model = recover().cuda()
    # Exclude initialization kernels from the timing.
    first = next(batches(model, DATA, selected[:1], "train", users["train"], 20))
    predict(first[2]); torch.cuda.synchronize()
    equations = NormalEquations("cuda")
    start = time.monotonic()
    for _, _, parts, target in batches(model, DATA, selected, "train", users["train"], 20):
        equations.update_parts(parts, target)
    torch.cuda.synchronize()
    fit_seconds = time.monotonic() - start
    timings = {}
    for label, stride, models in (("baseline", 1, 1), ("screen", 5, 5), ("comparison", 1, 3)):
        start = time.monotonic()
        for _, _, parts, target in batches(model, DATA, selected, "train", users["train"], stride):
            for _ in range(models):
                # Measure metric/transfer cost, but retain no training accuracy scores.
                geodesic_degrees_torch(predict(parts), target).cpu().numpy()
        torch.cuda.synchronize()
        timings[label] = time.monotonic() - start
    sample_positions = sum(r["samples"] for r in selected)
    scale = lambda split: sum(r["samples"] for r in rows[split]) / sample_positions
    estimate = fit_seconds * scale("train") + timings["baseline"] * scale("validation")
    estimate += timings["screen"] * scale("discovery") + timings["comparison"] * scale("validation")
    report = {"passed": estimate * 2 <= MAX_SECONDS, "training_segments": len(selected),
              "training_positions": sample_positions, "fit_examples": equations.count,
              "fit_seconds": fit_seconds, "evaluation_seconds": timings,
              "estimated_main_seconds": estimate, "conservative_estimate_seconds": 2 * estimate,
              "budget_seconds": MAX_SECONDS, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "chunk_positions": CHUNK, "protocol": protocol_record(rows, users),
              "runtime": configure_runtime(), "source_code_sha256": source_hashes(),
              "note": "Training-only timing; temporary sufficient statistics discarded, no coefficients solved"}
    save(ROOT / "preflight/throughput.json", report)
    log(f"Pilot: conservative estimate {estimate * 2 / 60:.1f} minutes; within budget={report['passed']}")
    if not report["passed"]:
        raise RuntimeError("Estimated runtime exceeds approved budget; revise protocol before fitting")


@torch.inference_mode()
def evaluate(base, rows, users, definitions, stride, deadline, actual=False, output=None):
    accumulators = {item["id"]: ErrorAccumulator() for item in definitions}
    actual_models = {}
    if actual:
        for item in definitions:
            saved = OUTPUT / "candidates" / (item.get("artifact_id", item["id"]) + ".pt")
            if saved.exists():
                actual_models[item["id"]] = load_checkpoint(saved)[0].cuda().requires_grad_(False)
            else:
                if item["id"] != "identity":
                    raise FileNotFoundError(saved)
                actual_models[item["id"]] = edited_model(base, item)
    checked_users, max_cached = set(), 0.
    for row, anchors, parts, target in batches(base, DATA, rows, "validation", users, stride, deadline=deadline):
        for item in definitions:
            cached = predict(parts, item.get("delta"), item.get("gain", 1.0))
            if actual:
                # The saved edited head is evaluated on the bit-identical frozen trunk.
                # This avoids a second temporal pass without using the delta shortcut.
                edited = actual_models[item["id"]]
                actual_parts = dict(parts, coefficients=GravityMirrorTCNN.readout(edited, parts["hidden"]).reshape(-1, 3, 3))
                prediction = predict(actual_parts)
                if row["user"] not in checked_users:
                    reference = edited(parts["first_history"])
                    error = float(angles(prediction[:1].cpu().numpy(), reference.cpu().numpy()).max())
                    cache_error = float(angles(prediction.cpu().numpy(), cached.cpu().numpy()).max())
                    max_cached = max(max_cached, error, cache_error)
                    if max_cached > ANGLE_TOLERANCE:
                        raise ValueError("Actual dense/window/cached inference mismatch")
            else:
                prediction = cached
            accumulators[item["id"]].update(row["user"], geodesic_degrees_torch(prediction, target).cpu().numpy())
        if row["user"] not in checked_users:
            checked_users.add(row["user"])
            if len(checked_users) % 50 == 0:
                log(f"Evaluate stride {stride}: {len(checked_users)}/{len(users)} users, {len(definitions)} models")
    summaries = {name: acc.summary() for name, acc in accumulators.items()}
    expected = window_count(rows, stride)
    if any(v["windows"] != expected or set(v["per_user"]) != set(users) for v in summaries.values()):
        raise ValueError("Evaluation membership/window count mismatch")
    result = {"stride": stride, "results": summaries, "actual_edited_heads": actual,
              "max_dense_window_cached_angle_deg": max_cached}
    if output:
        save(output, result)
    return result


@torch.inference_mode()
def fit(base, rows, users, deadline):
    equations, seen = NormalEquations("cuda"), set()
    for row, _, parts, target in batches(base, DATA, rows, "train", users, 20, deadline=deadline):
        equations.update_parts(parts, target)
        if row["user"] not in seen:
            seen.add(row["user"])
            if len(seen) % 100 == 0:
                checkpoint_statistics(OUTPUT / "training_statistics_partial.npz", equations)
                save(OUTPUT / "fit_progress.json", {"users_started": len(seen), "examples_accumulated": equations.count,
                                                    "current_segment": row["key"], "complete": False})
                log(f"Fit: {len(seen)}/{len(users)} users; {equations.count:,} examples")
    if equations.count != 10237229 or seen != set(users):
        raise ValueError("Fit membership/count mismatch")
    checkpoint_statistics(OUTPUT / "training_statistics.npz", equations)
    save(OUTPUT / "fit_progress.json", {"users": len(seen), "examples": equations.count, "complete": True})
    return equations.arrays()


def preservation(frozen_hashes):
    check_originals()
    original_record = json.loads((ROOT / "verification.json").read_text())
    checks = {p: sha256(REPO / p) == digest for p, digest in original_record["source_file_sha256"].items()}
    code = {p: sha256(REPO / p) == digest for p, digest in frozen_hashes.items()}
    save(OUTPUT / "preservation.json", {"preexisting_files": checks, "experiment_sources": code,
                                        "original_onnx_and_copies_unchanged": True})
    if not all(checks.values()) or not all(code.values()):
        raise ValueError("Pre-existing files or active source code changed")


def main_run():
    check_originals()
    for name in ("recovery", "parity", "tests", "throughput"):
        record = json.loads((ROOT / f"preflight/{name}.json").read_text())
        if name != "recovery" and not record["passed"]:
            raise ValueError(f"Missing prerequisite: {name}")
        if name in ("tests", "throughput") and record["source_code_sha256"] != source_hashes():
            raise ValueError(f"Sources changed after the {name} check")
    if sha256(DATA) != DATA_SHA or sha256(MANIFEST) != MANIFEST_SHA:
        raise ValueError("Data provenance changed")
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT / "candidates").mkdir()
    (OUTPUT / "source_snapshot").mkdir()
    rows, users = corpus()
    frozen_hashes = source_hashes()
    for relative in frozen_hashes:
        destination = OUTPUT / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, destination)
    protocol = protocol_record(rows, users)
    protocol.update(created_utc=datetime.now(timezone.utc).isoformat(), source_code_sha256=frozen_hashes,
                    data_sha256=DATA_SHA, manifest_sha256=MANIFEST_SHA, runtime=configure_runtime(),
                    packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
                    git_status=subprocess.check_output(["git", "status", "--short", "--branch"], cwd=REPO, text=True))
    save(OUTPUT / "protocol.json", protocol)
    save(OUTPUT / "cohorts.json", rows)
    base = recover().cuda()
    start = time.monotonic()
    deadline = start + MAX_SECONDS
    status = {"status": "running", "stage": "baseline", "test_samples_evaluated": False,
              "source_onnx_sha256": sha256(ORIGINAL), "started_utc": datetime.now(timezone.utc).isoformat()}
    save(OUTPUT / "status.json", status)
    try:
        identity = {"id": "identity", "gain": 1., "train_rms_correction": 0.}
        log("Establishing the unchanged baseline on all 786 validation users")
        baseline = evaluate(base, rows["validation"], users["validation"], [identity], 1, deadline,
                            actual=True, output=OUTPUT / "baseline_validation.json")
        log(f"Validation baseline: {baseline['results']['identity']['mean_endpoint_deg']:.9f} degrees")
        status.update(stage="fit"); save(OUTPUT / "status.json", status)
        arrays = fit(base, rows["train"], users["train"], deadline)
        fitted = [candidate(arrays, PENALTY, strength) for strength in STRENGTHS]
        gain = {"id": "global_105", "gain": 1.05, "train_rms_correction": .05 * float(arrays["residual_rms"])}
        definitions = [identity, gain, *fitted]
        artifacts = {}
        for item in [gain, *fitted]:
            edited = edited_model(base.cpu(), item)
            stem = OUTPUT / "candidates" / item["id"]
            save_inference(edited, Path(str(stem) + ".pt"), {"definition": item, "backpropagation": False})
            artifacts[item["id"]] = write_patch(edited, Path(str(stem) + ".onnx"), "cubic" if "delta" in item else "gain")
        base.cuda()
        save(OUTPUT / "candidates.json", {"definitions": definitions, "artifacts": artifacts,
                                          "training_residual_rms": float(arrays["residual_rms"])})
        status.update(stage="screen"); save(OUTPUT / "status.json", status)
        screen = evaluate(base, rows["discovery"], users["discovery"], definitions, 5, deadline,
                          output=OUTPUT / "screen.json")
        ranked = sorted(definitions, key=lambda item: (screen["results"][item["id"]]["mean_endpoint_deg"],
                                                       item["train_rms_correction"], item["id"]))
        save(OUTPUT / "ranking.json", ranked)
        winner = ranked[0]
        status["winner"] = winner
        log(f"Screen winner: {winner['id']}")
        if winner["id"] in ("identity", "global_105"):
            status.update(status="control_won_screen", stage="complete")
            return
        finalist = dict(winner, id="candidate", artifact_id=winner["id"])
        chosen = [identity, gain, finalist]
        status.update(stage="full_discovery"); save(OUTPUT / "status.json", status)
        discovery = evaluate(base, rows["discovery"], users["discovery"], chosen, 1, deadline, actual=True,
                             output=OUTPUT / "discovery.json")
        means = {k: v["mean_endpoint_deg"] for k, v in discovery["results"].items()}
        if any(means["candidate"] >= means[k] for k in ("identity", "global_105")):
            status.update(status="no_full_discovery_improvement", stage="complete")
            return
        lock = {"candidate": winner, "locked_utc": datetime.now(timezone.utc).isoformat(),
                "further_tuning_allowed": False, "assessment_users": users["assessment"]}
        save(OUTPUT / "frozen_fit.json", lock)
        status.update(stage="assessment"); save(OUTPUT / "status.json", status)
        assessment = evaluate(base, rows["assessment"], users["assessment"], chosen, 1, deadline, actual=True,
                              output=OUTPUT / "assessment.json")
        summaries = assessment["results"]
        comparisons = {k: paired_comparison(summaries["candidate"], summaries[k]) for k in ("identity", "global_105")}
        gate = assessment_gate(summaries, comparisons)
        save(OUTPUT / "assessment_decision.json", {"comparisons": comparisons, "gate": gate})
        status.update(status="promising_exploratory" if gate["passed"] else "not_promising", stage="complete",
                      comparisons=comparisons, gate=gate)
        if gate["passed"]:
            status.update(stage="deployment_checks"); save(OUTPUT / "status.json", status)
            candidate_path = Path(artifacts[winner["id"]]["path"])
            model, _ = load_checkpoint(candidate_path.with_suffix(".pt"))
            parity = verify_model(model, candidate_path, training_rows=rows["train"])
            report = {"passed": True, "parity": parity, "patch": artifacts[winner["id"]],
                      "latency": latency_pair(candidate_path)}
            save(OUTPUT / "deployment_checks.json", report)
            status.update(stage="complete", deployment_verified=True)
    except TimeoutError as exc:
        status.update(status="budget_incomplete", error=str(exc))
        raise
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - start
        try:
            preservation(frozen_hashes)
        except Exception as exc:
            status.update(status="failed", preservation_error=str(exc))
            raise
        finally:
            save(OUTPUT / "status.json", status)
            log(f"Run ended: {status['status']}; elapsed {status['elapsed_seconds'] / 60:.1f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    configure_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError("This approved main run requires the verified CUDA environment")
    throughput_pilot() if args.pilot else main_run()
