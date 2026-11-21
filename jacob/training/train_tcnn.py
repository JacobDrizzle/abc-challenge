"""Train one TCNN seed on frozen qualified data; resume at saved epoch boundaries."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from baseline_pipeline import DEFAULT_H5, manifest_for, sha256_file
from jsonio import save_json
from tcnn.data import check_corpus, DeviceWindows, WindowDataset
from tcnn.model import (TinyTCNN, damped_prior_rotvec, qmul, relative_history, rotation_loss,
                        rotvec_to_quaternion)
from tcnn.configuration import normalize_config
from tcnn.runtime import configure_runtime

ROOT = Path(__file__).resolve().parent


def atomic_checkpoint(path, state):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


REFERENCE_TAU_MS = 100  # damped baseline selected on both earlier cohorts: lag 1, tau 100 ms


def geodesic_degrees_torch(predicted, target):
    """Sign-invariant SO(3) angle in float64, as baseline_pipeline.geodesic_degrees."""
    p = torch.nn.functional.normalize(predicted.double(), dim=-1)
    t = torch.nn.functional.normalize(target.double(), dim=-1)
    return torch.rad2deg(2 * torch.arccos((p * t).sum(-1).abs().clamp(0, 1)))


def reference_predictors(horizon):
    def hold_last(history):
        return history[:, -1:].expand(-1, horizon, -1)

    def damped(history):
        relative, last = relative_history(history)
        return qmul(last, rotvec_to_quaternion(damped_prior_rotvec(relative, horizon, REFERENCE_TAU_MS / 10)))
    return {"hold_last": hold_last, "damped_velocity": damped}


def score_windows(predictors, windows, horizon):
    """Window-weighted mean error per horizon step for each predictor on identical windows."""
    sums, total = None, 0
    with torch.inference_mode():
        for history, target in windows:
            if sums is None:
                sums = {name: torch.zeros(horizon, dtype=torch.float64, device=history.device) for name in predictors}
            for name, predict in predictors.items():
                sums[name] += geodesic_degrees_torch(predict(history), target).sum(0)
            total += len(history)
    if not total:
        raise ValueError("No validation windows")
    results = {}
    for name, value in sums.items():
        means = (value / total).cpu().numpy()
        results[name] = {"windows": total, "mean_endpoint_deg": float(means[-1]),
                         "mean_deg_by_horizon": means.tolist()}
    return results


def validate(model, windows, horizon):
    model.eval()
    return score_windows({"model": model}, windows, horizon)["model"]


def format_duration(seconds):
    minutes = int(round(seconds / 60))
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def train(data, split, output, config, seed, device="cuda", resume=False, synthetic=False):
    data, split, output = Path(data).resolve(), Path(split).resolve(), Path(output).resolve()
    config = normalize_config(config)
    for key in ("epochs", "steps_per_epoch", "batch_size", "patience", "validation_stride"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["workers"] < 0 or config["cache_mb_per_worker"] < 0:
        raise ValueError("Invalid loader settings")
    if not all(math.isfinite(config[k]) and config[k] > 0
               for k in ("learning_rate", "gradient_clip", "minimum_improvement_deg")):
        raise ValueError("Invalid optimiser settings")
    if not 0 <= config["dropout"] < 1 or config["weight_decay"] < 0:
        raise ValueError("Invalid dropout/weight decay")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu only for diagnostics.")
    device = torch.device(device)
    runtime = configure_runtime()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    audit = check_corpus(data, split, synthetic=synthetic,
                         minimum_users=3 if synthetic else 30,
                         minimum_recordings=3 if synthetic else 100)
    if synthetic:
        with h5py.File(data) as h5:
            if not h5.attrs.get("synthetic", False):
                raise ValueError("Smoke-test override requires an explicitly synthetic fixture")
    provenance = {"data_sha256": audit["data_sha256"], "split_sha256": audit["split_sha256"],
                  "train_users": audit["users"]["train"], "validation_users": audit["users"]["validation"],
                  "synthetic": synthetic, "runtime": runtime, "sampler": "sha256-seed-epoch-index-v2",
                  "code_sha256": {name: sha256_file(ROOT / name) for name in
                                  ("train_tcnn.py", "tcnn/model.py", "tcnn/data.py", "baseline_pipeline.py",
                                   "tcnn/configuration.py", "tcnn/runtime.py")}}
    model = TinyTCNN(config["context"], config["horizon"], config["channels"], config["dropout"],
                     effective_context=config["effective_context"], velocity_features=config["velocity_features"],
                     damped_prior_tau_ms=config["damped_prior_tau_ms"], input_features=config["input_features"],
                     num_blocks=config["num_blocks"]).to(device)
    # Validated and saved as model_state; raw weights continue training underneath.
    ema = copy.deepcopy(model).requires_grad_(False) if config["ema_decay"] > 0 else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    start, best, patience_best, stale, history = 0, float("inf"), float("inf"), 0, []
    if resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["config"] != config or checkpoint["seed"] != seed or checkpoint["provenance"] != provenance:
            raise ValueError("Resume requires identical configuration, seed, code, data and split")
        model.load_state_dict(checkpoint.get("train_model_state", checkpoint["model_state"]))
        if ema is not None:
            ema.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start, best = checkpoint["epoch"], checkpoint["best_validation_deg"]
        patience_best, stale = checkpoint["patience_best"], checkpoint["stale_epochs"]
        history = checkpoint["history"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    else:
        if output.exists():
            raise ValueError("Run directory already exists. Use --resume or a new output directory.")
        output.mkdir(parents=True)
        save_json(output / "config.json", config)
        save_json(output / "environment.json", {
            "python": platform.python_version(), "platform": platform.platform(), "torch": str(torch.__version__),
            "cuda_build": torch.version.cuda,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "numpy": np.__version__, "h5py": h5py.__version__, "runtime": runtime,
            "created_utc": datetime.now(timezone.utc).isoformat()})
        (output / "environment_freeze.txt").write_text(
            subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
        save_json(output / "provenance.json", provenance)
    train_rows = [r for r in audit["segments"] if r["split"] == "train"]
    val_rows = [r for r in audit["segments"] if r["split"] == "validation"]
    horizon, steps = config["horizon"], config["steps_per_epoch"]
    validation_data = DeviceWindows(data, val_rows, device, config["context"], horizon)

    def validation_windows():
        return validation_data.windows(config["validation_stride"], 2048)
    reference = score_windows(reference_predictors(horizon), validation_windows(), horizon)
    save_json(output / "validation_reference.json", {
        "validation_stride": config["validation_stride"], "damped_velocity_settings": {"lag": 1, "tau_ms": REFERENCE_TAU_MS},
        **reference})
    damped_score = reference["damped_velocity"]["mean_endpoint_deg"]
    if config["data_on_device"]:
        train_data, dataset = DeviceWindows(data, train_rows, device, config["context"], horizon), None
    else:
        dataset = WindowDataset(data, train_rows, config["batch_size"]*steps, seed,
                                cache_mb=config["cache_mb_per_worker"])
    print(f"Training seed {seed} on {device}; {sum(p.numel() for p in model.parameters()):,} parameters; "
          f"{steps} steps x {config['batch_size']} windows per epoch; "
          f"{'GPU-resident' if config['data_on_device'] else 'DataLoader'} sampling; "
          f"EMA {config['ema_decay'] if ema is not None else 'off'}; "
          f"damped prior {config['damped_prior_tau_ms'] or 'off'}.", flush=True)
    print(f"Validation references ({reference['hold_last']['windows']:,} windows, stride {config['validation_stride']}): "
          f"hold-last {reference['hold_last']['mean_endpoint_deg']:.4f} deg | damped velocity {damped_score:.4f} deg", flush=True)
    try:
        for epoch in range(start, config["epochs"]):
            if stale >= config["patience"]:
                break
            if dataset is None:
                batches = train_data.batches(seed, epoch, steps, config["batch_size"])
            else:
                dataset.epoch = epoch
                batches = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                                     num_workers=config["workers"], pin_memory=device.type == "cuda",
                                     persistent_workers=False,
                                     generator=torch.Generator().manual_seed(seed + epoch))
            model.train()
            began = time.perf_counter()
            # Device-side sums; a single synchronisation per epoch.
            total_loss = torch.zeros((), dtype=torch.float64, device=device)
            total_endpoint = torch.zeros((), dtype=torch.float64, device=device)
            count = 0
            for step, (x, target) in enumerate(batches):
                x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x)
                loss = rotation_loss(prediction, target, config["loss_kind"], config["endpoint_weight"])
                loss.backward()
                # Also stops on a non-finite loss: its gradients are non-finite.
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                if ema is not None:
                    done = epoch * steps + step
                    decay = min(config["ema_decay"], (1 + done) / (10 + done))
                    with torch.no_grad():
                        for average, current in zip(ema.parameters(), model.parameters()):
                            average.lerp_(current, 1 - decay)
                total_loss += loss.detach() * len(x)
                total_endpoint += geodesic_degrees_torch(prediction.detach()[:, -1], target[:, -1]).sum()
                count += len(x)
            training_loss = total_loss.item() / count
            if not math.isfinite(training_loss):
                raise ValueError("Non-finite training loss")
            training_seconds = time.perf_counter() - began
            validation = validate(ema if ema is not None else model, validation_windows(), horizon)
            score = validation["mean_endpoint_deg"]
            if not math.isfinite(score):
                raise ValueError("Non-finite validation score")
            improved = score < best
            best = min(best, score)
            if score < patience_best - config["minimum_improvement_deg"]:
                patience_best, stale = score, 0
            else:
                stale += 1
            best_epoch = epoch + 1 if improved else min(history, key=lambda r: r["validation_endpoint_deg"])["epoch"]
            seconds = time.perf_counter() - began
            record = {"epoch": epoch+1, "training_loss": training_loss,
                      "training_endpoint_deg": total_endpoint.item() / count,
                      "validation_endpoint_deg": score, "validation": validation,
                      "best_validation_deg": best, "best_epoch": best_epoch, "stale_epochs": stale,
                      "validation_vs_damped_deg": score - damped_score,
                      "learning_rate": optimizer.param_groups[0]["lr"],
                      "seconds": seconds, "training_windows_per_second": count / training_seconds,
                      "ema": ema is not None,
                      "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device)/1e6 if device.type == "cuda" else None}
            history.append(record)
            scheduler.step()
            saved = ema if ema is not None else model
            state = {"format_version": 2, "model_spec": model.specification(), "model_state": saved.state_dict(),
                     **({"train_model_state": model.state_dict()} if ema is not None else {}),
                     "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                     "config": config, "seed": seed, "epoch": epoch+1, "provenance": provenance,
                     "best_validation_deg": best, "patience_best": patience_best, "stale_epochs": stale,
                     "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                     "history": history}
            # best precedes last: resuming always has a matching best checkpoint.
            if improved:
                atomic_checkpoint(output / "best.pt", state)
            atomic_checkpoint(output / "last.pt", state)
            save_json(output / "history.json", history)
            mean_seconds = sum(r["seconds"] for r in history[start:]) / (len(history) - start)
            step10 = f" | val 100ms {validation['mean_deg_by_horizon'][9]:.3f} deg" if horizon >= 10 else ""
            print(f"epoch {epoch+1:02d}/{config['epochs']} | loss {training_loss:.6f} | train end {record['training_endpoint_deg']:.3f} deg"
                  f" | val end {score:.4f} deg (best {best:.4f} @{best_epoch}, stale {stale}/{config['patience']}){step10}"
                  f" | vs damped {score-damped_score:+.3f} deg ({100*(score-damped_score)/damped_score:+.1f}%)"
                  f" | lr {record['learning_rate']:.2e} | {record['training_windows_per_second']/1000:.1f}k win/s"
                  f" | {seconds:.1f}s | ETA <= {format_duration(mean_seconds*(config['epochs']-epoch-1))}", flush=True)
    finally:
        if dataset is not None:
            dataset.close()
    try:
        from tcnn.plots import plot_training
        plot_training([output / "history.json"], output / "training_curves.png")
    except Exception as error:  # A finished run must not fail on a figure.
        print(f"Warning: training_curves.png was not drawn: {error}", flush=True)
    save_json(output / "complete.json", {"best_validation_deg": best, "epochs": len(history),
                                       "seed": seed, "synthetic": synthetic})
    return output / "best.pt"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_H5)
    p.add_argument("--split-file", type=Path, help="Default: the manifest_<run_id>.json next to --data")
    p.add_argument("--config", type=Path, default=ROOT / "configs/tcnn_v5.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    try:
        train(args.data, args.split_file or manifest_for(args.data), args.output, json.loads(args.config.read_text()),
              args.seed, args.device, args.resume)
    except (ValueError, OSError) as error:
        p.exit(2, f"Training stopped: {error}\n")


if __name__ == "__main__":
    main()
