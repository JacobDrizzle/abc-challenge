"""Run the preregistered final-head pilot without accessing test samples.

From jacob/training: python -m experimental.v10_weight_edit.run --output outputs/NEW_RUN
The output directory must be new. Originals are hashed and copied before evaluation.
"""
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
from tcnn.model import conjugate, qmul
from tcnn.runtime import configure_runtime
from train_tcnn import geodesic_degrees_torch
from .edits import apply_gains, candidates, decompose, reconstruct
from .metrics import ErrorAccumulator, paired_comparison, promising_gate

TRAINING = Path(__file__).resolve().parents[2]
REPO = TRAINING.parents[1]
SOURCE = TRAINING / "outputs/v10_k8_800users_full_pass/full_pass"
DATA = REPO / "data/h5/qualified_head_100hz_merged_20261101_140915.h5"
SPLIT = REPO / "data/h5/manifest_merged_20261101_140915.json"
IDENTITY = {"id": "identity", "gains": [1.0, 1.0, 1.0]}


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}", flush=True)


def split_users(users):
    if len(users) != 120 or len(set(users)) != 120:
        raise ValueError("This preregistered experiment requires exactly 120 validation users")
    ordered = sorted(users, key=lambda u: (hashlib.sha256(f"v10-weight-edit-v1:{u}".encode()).hexdigest(), u))
    return ordered[:80], ordered[80:]


def motion_groups(history):
    """Input-only diagnostics: speed and world-axis change over the last 100 ms."""
    now = qmul(history[:, -1], conjugate(history[:, -2]))
    earlier = qmul(history[:, -11], conjugate(history[:, -12]))
    now = torch.where(now[:, :1] < 0, -now, now)
    earlier = torch.where(earlier[:, :1] < 0, -earlier, earlier)
    vector, previous = now[:, 1:], earlier[:, 1:]
    norm, previous_norm = vector.norm(dim=-1), previous.norm(dim=-1)
    speed = torch.rad2deg(2 * torch.atan2(norm, now[:, 0])) * 100
    previous_speed = torch.rad2deg(2 * torch.atan2(previous_norm, earlier[:, 0])) * 100
    cosine = (vector * previous).sum(-1) / (norm * previous_norm).clamp_min(1e-12)
    group = torch.ones(len(history), dtype=torch.int64, device=history.device)
    group[speed >= 90] = 2
    group[(cosine < 2 ** -.5) & (speed >= 5) & (previous_speed >= 5)] = 3
    group[speed < 5] = 0
    return group


def batches(data, rows, device, stride, batch_size):
    """Same per-segment anchors and normalisation as DeviceWindows.windows()."""
    if any(not row["path"].startswith("validation/") for row in rows):
        raise ValueError("Only validation sample paths are permitted")
    span = torch.arange(420, device=device)
    with h5py.File(data, "r") as h5:
        for row in rows:
            group = h5[row["path"]]
            times = np.asarray(group["time_s"], dtype=np.float64)
            if not np.isfinite(times).all() or not np.allclose(np.diff(times), .01, rtol=0, atol=1e-6):
                raise ValueError(f"Invalid time grid: {row['key']}")
            q = unit_quaternions(group["quaternion_wxyz"][:], source=row["key"]).astype(np.float32)
            if np.any(np.sum(q[1:] * q[:-1], axis=-1) < -1e-8):
                raise ValueError(f"Discontinuous quaternion signs: {row['key']}")
            buffer = torch.from_numpy(q).to(device)
            starts = torch.arange(0, len(q) - 419, stride, device=device)
            for begin in range(0, len(starts), batch_size):
                indices = starts[begin:begin + batch_size]
                values = buffer[indices[:, None] + span]
                yield indices, values[:, :400], values[:, 400:]


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "originals").mkdir()
    (output / "cache").mkdir()
    selection = json.loads((SOURCE / "selection.json").read_text())
    if sha256_file(DATA) != selection["data_sha256"] or sha256_file(SPLIT) != selection["split_sha256"]:
        raise ValueError("Dataset or split hash differs from checkpoint provenance")
    with h5py.File(DATA, "r") as h5:
        users = sorted(h5["validation"])
        discovery, confirmation = split_users(users)
        rows, skipped = inventory(h5, users, EvaluationConfig(), False)
    if skipped:
        raise ValueError("Validation contains skipped segments")
    sources = {}
    for item in selection["validation_candidates"]:
        seed = item["seed"]
        source = SOURCE / ("selected_model.pt" if seed == 44 else item["checkpoint"])
        digest = sha256_file(source)
        if digest != item["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch for seed {seed}")
        destination = output / "originals" / f"seed_{seed}.pt"
        shutil.copy2(source, destination)
        if sha256_file(destination) != digest:
            raise ValueError("Checkpoint copy verification failed")
        sources[str(seed)] = {"path": str(source), "copy": str(destination), "sha256": digest}
    shutil.copy2(output / "originals/seed_44.pt", output / "candidate_model.pt")
    owning_files = [TRAINING / name for name in ("best/model.py", "best/export.py", "tcnn/model.py", "tcnn/runtime.py",
                                                "train_tcnn.py", "baseline_pipeline.py", "tcnn/onnx_export.py")]
    code_files = [*Path(__file__).parent.glob("*.py"), *owning_files]
    cohort = {"context": 400, "horizon": 20, "stride": 1, "segments": rows}
    manifest = {
        "version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
        "data": str(DATA), "data_sha256": selection["data_sha256"], "split_sha256": selection["split_sha256"],
        "sources": sources, "discovery_users": discovery, "confirmation_users": confirmation,
        "candidates": [IDENTITY, *candidates()], "baseline_expected_deg": 4.403628579547263,
        "baseline_tolerance_deg": .001, "runtime": configure_runtime(),
        "source_code_sha256": {str(p.relative_to(TRAINING)): sha256_file(p) for p in sorted(code_files)},
        "cohort_sha256": hashlib.sha256(json.dumps(cohort, sort_keys=True).encode()).hexdigest(),
        "validation_windows": sum(row["windows"] for row in rows),
        "test_samples_accessed": False,
        "interpretation": "40 users withheld from edit selection, not from original early stopping or seed selection",
        "selection_rule": "stride5 endpoint mean, then Euclidean gain distance from identity, then candidate ID; one winner checked at stride1",
        "confirmation_rule": "seed44 primary and equal-user means lower, majority improved, CI upper<0, p95 and all-horizon average no worse; endpoint means lower on seeds42/43",
    }
    save(output / "manifest.json", manifest)
    save(output / "cohort.json", cohort)
    by_user = defaultdict(list)
    for row in rows:
        by_user[row["user"]].append(row)
    return manifest, by_user


@torch.inference_mode()
def baseline_and_cache(output, manifest, rows, device, batch_size):
    model, _ = load_checkpoint(output / "originals/seed_44.pt")
    model.to(device)
    accumulator = ErrorAccumulator()
    discovery = set(manifest["discovery_users"])
    checks = []
    for i, user in enumerate(sorted(rows)):
        cached = defaultdict(list)
        checked = False
        for starts, history, target in batches(DATA, rows[user], device, 1, batch_size):
            residual, prior, last = decompose(model, history)
            prediction = reconstruct(residual, prior, last)
            if not checked:
                difference = float((prediction - model(history)).abs().max())
                if difference > 1e-6:
                    raise ValueError(f"Cached identity differs from model by {difference}")
                checks.append(difference)
                checked = True
            accumulator.update(user, geodesic_degrees_torch(prediction, target).cpu().numpy())
            if user in discovery:
                keep = starts.remainder(5) == 0
                for key, value in (("residual", residual), ("prior", prior), ("last", last), ("target", target)):
                    cached[key].append(value[keep].cpu().numpy())
                cached["motion_group"].append(motion_groups(history[keep]).cpu().numpy())
        if cached:
            np.savez(output / "cache" / f"{user}.npz", **{k: np.concatenate(v) for k, v in cached.items()})
        if (i + 1) % 5 == 0:
            log(f"Baseline: {i + 1}/120 validation users")
    summary = accumulator.summary()
    summary["max_cached_identity_component_difference"] = max(checks)
    summary["difference_from_recorded_deg"] = summary["mean_endpoint_deg"] - manifest["baseline_expected_deg"]
    save(output / "baseline_reproduction.json", summary)
    if abs(summary["difference_from_recorded_deg"]) > manifest["baseline_tolerance_deg"]:
        raise ValueError("Baseline reproduction failed; do not screen edits")
    log(f"Baseline reproduced: {summary['mean_endpoint_deg']:.9f} degrees")
    return summary


@torch.inference_mode()
def screen(output, manifest, device, batch_size):
    definitions = manifest["candidates"]
    accumulators = {item["id"]: ErrorAccumulator() for item in definitions}
    names = ["below_5_deg_s", "slow_5_to_90_deg_s", "fast_at_least_90_deg_s", "axis_change_over_45_deg_in_100ms"]
    diagnostics = {"by_motion": {name: {"windows": 0, "axis_error_times_correction_sum": [0., 0., 0.],
                                      "axis_correction_square_sum": [0., 0., 0.]} for name in names},
                   "description": "Positive error-times-correction suggests reducing that correction; diagnostic only, in native-map coordinates. Motion groups use inputs only; axis-change group takes precedence over slow/fast."}
    for i, user in enumerate(manifest["discovery_users"]):
        with np.load(output / "cache" / f"{user}.npz", allow_pickle=False) as cache:
            arrays = {key: cache[key] for key in cache.files}
        for start in range(0, len(arrays["target"]), batch_size):
            tensors = {key: torch.from_numpy(value[start:start + batch_size]).to(device) for key, value in arrays.items()}
            residual, prior, last, target = (tensors[k] for k in ("residual", "prior", "last", "target"))
            for item in definitions:
                prediction = reconstruct(residual, prior, last, item["gains"])
                accumulators[item["id"]].update(user, geodesic_degrees_torch(prediction, target).cpu().numpy())
            relative = qmul(conjugate(last), target[:, -1:])[:, 0]
            relative = torch.where(relative[:, :1] < 0, -relative, relative)
            valid = relative[:, 0] > .1
            desired = 2 * relative[valid, 1:] / relative[valid, :1]
            correction = residual[valid, -1]
            error = correction + prior[valid, -1] - desired
            for index, name in enumerate(names):
                mask = tensors["motion_group"][valid] == index
                group = diagnostics["by_motion"][name]
                group["windows"] += int(mask.sum())
                for key, value in (("axis_error_times_correction_sum", error * correction), ("axis_correction_square_sum", correction.square())):
                    group[key] = (np.array(group[key]) + value[mask].double().sum(0).cpu().numpy()).tolist()
        if (i + 1) % 10 == 0:
            log(f"Screen: {i + 1}/80 discovery users, all 16 edits")
    summaries = {name: acc.summary() for name, acc in accumulators.items()}
    ranked = sorted(definitions, key=lambda item: (summaries[item["id"]]["mean_endpoint_deg"],
                    sum((g - 1.) ** 2 for g in item["gains"]), item["id"]))
    result = {"stride": 5, "ranking": ranked, "results": summaries, "diagnostics": diagnostics}
    save(output / "screen.json", result)
    winner = ranked[0]
    log(f"Screen winner: {winner['id']}, {summaries[winner['id']]['mean_endpoint_deg']:.9f} degrees")
    return winner


@torch.inference_mode()
def evaluate_pair(output, seed, users, rows, gains, device, batch_size, label):
    original, _ = load_checkpoint(output / "originals" / f"seed_{seed}.pt")
    original.to(device)
    edited = apply_gains(original, gains).to(device)
    baseline, candidate = ErrorAccumulator(), ErrorAccumulator()
    for i, user in enumerate(users):
        for _, history, target in batches(DATA, rows[user], device, 1, batch_size):
            baseline.update(user, geodesic_degrees_torch(original(history), target).cpu().numpy())
            candidate.update(user, geodesic_degrees_torch(edited(history), target).cpu().numpy())
        if (i + 1) % 5 == 0:
            log(f"{label} seed{seed}: {i + 1}/{len(users)} users")
    baseline, candidate = baseline.summary(), candidate.summary()
    paired = paired_comparison(candidate, baseline)
    result = {"seed": seed, "stride": 1, "baseline": baseline, "candidate": candidate, "paired": paired,
              "gate": promising_gate(candidate, baseline, paired) if label == "confirmation" else None}
    save(output / f"{label}_seed_{seed}.json", result)
    log(f"{label} seed{seed}: delta={paired['mean_diff_deg']:+.9f} degrees")
    return result


def write_candidate(output, winner, selection):
    original, state = load_checkpoint(output / "originals/seed_44.pt")
    edited = apply_gains(original, winner["gains"])
    # Inference-only artifact: no stale optimizer, raw training weights or old scores.
    payload = {"model_spec": state["model_spec"], "model_state": edited.state_dict(),
               "weight_edit": {"candidate": winner, "source_sha256": sha256_file(output / "originals/seed_44.pt"),
                               "selection": selection, "training_performed": False}}
    temporary = output / "candidate_model.tmp"
    torch.save(payload, temporary)
    temporary.replace(output / "candidate_model.pt")
    return sha256_file(output / "candidate_model.pt")


def preserve_check(output, manifest):
    results = {}
    for seed, item in manifest["sources"].items():
        results[seed] = {"source_unchanged": sha256_file(item["path"]) == item["sha256"],
                         "copy_unchanged": sha256_file(item["copy"]) == item["sha256"]}
    save(output / "preservation.json", results)
    if not all(all(item.values()) for item in results.values()):
        raise ValueError("Original checkpoint preservation check failed")


def run(args):
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise ValueError("CUDA is unavailable")
    output = Path(args.output).resolve()
    manifest, rows = prepare(output)
    start = time.monotonic()
    report = {"status": "running", "test_samples_accessed": False}
    save(output / "status.json", report)
    try:
        baseline_and_cache(output, manifest, rows, args.device, args.batch_size)
        winner = screen(output, manifest, args.device, args.batch_size)
        report["winner"] = winner
        if winner["id"] == "identity":
            report["status"] = "no_screen_improvement"
            return
        discovery = evaluate_pair(output, 44, manifest["discovery_users"], rows, winner["gains"], args.device, args.batch_size, "discovery")
        if discovery["paired"]["mean_diff_deg"] >= 0:
            report["status"] = "no_full_discovery_improvement"
            return
        # This file and inference copy are frozen before any candidate sees confirmation users.
        lock = {"candidate": winner, "discovery_delta_deg": discovery["paired"]["mean_diff_deg"],
                "locked_utc": datetime.now(timezone.utc).isoformat(), "further_tuning_allowed": False}
        save(output / "frozen_edit.json", lock)
        report["candidate_sha256"] = write_candidate(output, winner, lock)
        results = {seed: evaluate_pair(output, seed, manifest["confirmation_users"], rows, winner["gains"], args.device, args.batch_size, "confirmation")
                   for seed in (44, 42, 43)}
        report["confirmation"] = {str(seed): {"paired": r["paired"], "gate": r["gate"]} for seed, r in results.items()}
        passed = results[44]["gate"]["passed"] and all(results[seed]["paired"]["mean_diff_deg"] < 0 for seed in (42, 43))
        report["status"] = "promising" if passed else "not_confirmed"
        if passed:
            from .export_check import verify_exports
            report["export_checks"] = verify_exports(output / "originals/seed_44.pt", output / "candidate_model.pt", output / "onnx")
            save(output / "export_checks.json", report["export_checks"])
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        preserve_check(output, manifest)
        save(output / "status.json", report)
        log(f"Experiment status: {report['status']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    run(args)


if __name__ == "__main__":
    main()
