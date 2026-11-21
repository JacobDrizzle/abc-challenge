"""Score parameter-free predictors, and optionally the submitted ONNX model, on one split of the final dataset.

Written 13 Nov 2026 for the paper's full-test table (1,049 test users). Nic's PC run gave the rows for the model,
hold last and constant velocity; its results file is not in this repo. This script re-derives those rows as a check
and adds two rows a reviewer asked for: the damped prior alone, and double exponential smoothing (LaViola 2003).

What is scored, and how:
- Windows are the bench's: every anchor t = 399 .. N-21 of every segment (400 samples of history, 20 targets).
- Hold last, constant velocity and the damped prior are the classes of bench/baselines.py, unchanged. They only read
  the newest sample and the one `lag` samples before it, so they are given the last 6 samples of each window, not 400.
- The ONNX model is streamed from each segment's first sample, one call per sample (bench/drivers.py, "from-start").
- Errors are bench/geometry.py's float64 geodesic angle, after re-normalising the output as the harness does.
- Double exponential smoothing follows LaViola (2003): each quaternion component is smoothed twice with factor alpha,
  s1[t] = alpha q[t] + (1 - alpha) s1[t-1], s2[t] = alpha s1[t] + (1 - alpha) s2[t-1], both starting at q[0], and the
  forecast k samples ahead is (2 + alpha k / (1 - alpha)) s1[t] - (1 + alpha k / (1 - alpha)) s2[t], re-normalised.
  It runs from the segment's first sample, like the model. alpha is chosen on the validation split (--tune-des) and
  then passed in for the test split (--des-alpha): it is never tuned on test users.

Usage (from the repo root, with the bench's Python environment):
  python jacob/ablations/analyses/full_test_baselines/score_full_test.py --h5 DATA.h5 --split validation --tune-des
  python jacob/ablations/analyses/full_test_baselines/score_full_test.py --h5 DATA.h5 --split test \
      --des-alpha ALPHA --onnx Submission/Motion_Pending_headpose/model_fp32.onnx --out OUT_DIR
"""
import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

# One maths thread per worker process: the workers are the parallelism. Without this, every worker's OpenBLAS
# reserves memory for all 24 threads and 20 workers run Windows out of memory. Must be set before numpy is imported.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import h5py
import numpy as np
from scipy.signal import lfilter

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from bench.baselines import ConstantVelocity, DampedVelocity, HoldLast  # noqa: E402
from bench.drivers import OnnxModel  # noqa: E402
from bench.geometry import geodesic_degrees, renormalise, unit_quaternions  # noqa: E402

CONTEXT, HORIZON = 400, 20
DES_GRID = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
SHORT = 6  # samples of history handed to the bench baselines: the newest one and up to 5 before it

_state = {}


def start_worker(h5_path, onnx_path, des_alphas, tune):
    """Runs once in each worker process: open the data file and load the model there."""
    _state["h5"] = h5py.File(h5_path, "r")
    _state["model"] = OnnxModel(onnx_path, threads=1) if onnx_path else None
    _state["des_alphas"], _state["tune"] = des_alphas, tune
    _state["baselines"] = {"hold_last": HoldLast(HORIZON), "constant_velocity_1step": ConstantVelocity(HORIZON, lag=1),
                           "constant_velocity_5step": ConstantVelocity(HORIZON, lag=5),
                           "damped_velocity": DampedVelocity(HORIZON, lag=1, tau_ms=100)}


def des_forecasts(q, alpha, steps):
    """Double exponential smoothing forecasts [N, len(steps), 4] for every sample t of q [N, 4], from samples 0..t."""
    signs = np.sign(np.sum(q[1:] * q[:-1], axis=1))
    signs[signs == 0] = 1
    q = q * np.concatenate(([1.0], np.cumprod(signs)))[:, None]  # q and -q are one rotation: keep the signs continuous
    b, a = [alpha], [1.0, -(1.0 - alpha)]
    s1, _ = lfilter(b, a, q, axis=0, zi=(1 - alpha) * q[:1])   # zi makes the filter start at s1[0] = q[0]
    s2, _ = lfilter(b, a, s1, axis=0, zi=(1 - alpha) * s1[:1])
    gain = alpha * np.asarray(steps, np.float64) / (1 - alpha)
    return (2 + gain)[None, :, None] * s1[:, None, :] - (1 + gain)[None, :, None] * s2[:, None, :]


def score_segment(task):
    """Errors of every predictor on one segment. Returns sums per forecast step and the per-window step-20 errors."""
    path, user = task
    q = unit_quaternions(_state["h5"][path]["quaternion_wxyz"][...], source=path)
    q32 = q.astype(np.float32)
    anchors = np.arange(CONTEXT - 1, len(q) - HORIZON)
    targets = q[anchors[:, None] + np.arange(1, HORIZON + 1)[None, :]]
    out = {"user": user, "windows": len(anchors), "sums": {}, "endpoint": {}}
    if _state["tune"]:
        for alpha in _state["des_alphas"]:
            forecast = des_forecasts(q, alpha, [HORIZON])[anchors]
            out["sums"][f"des_{alpha}"] = float(geodesic_degrees(renormalise(forecast), targets[:, -1:]).sum())
        return out
    short = np.lib.stride_tricks.sliding_window_view(q32[CONTEXT - SHORT:len(q) - HORIZON], SHORT, axis=0)
    short = short.transpose(0, 2, 1)  # [windows, 6, 4]; row i ends at anchor 399 + i
    for name, model in _state["baselines"].items():
        errors = np.concatenate([geodesic_degrees(renormalise(model.predict_batch(np.array(short[i:i + 8192]))),
                                                  targets[i:i + 8192]) for i in range(0, len(anchors), 8192)])
        out["sums"][name], out["endpoint"][name] = errors.sum(0), errors[:, -1].astype(np.float32)
    for alpha in _state["des_alphas"]:
        forecast = des_forecasts(q, alpha, [10, HORIZON])[anchors]
        errors = geodesic_degrees(renormalise(forecast), targets[:, [9, HORIZON - 1]])
        sums = np.full(HORIZON, np.nan)
        sums[[9, HORIZON - 1]] = errors.sum(0)
        out["sums"][f"des_{alpha}"], out["endpoint"][f"des_{alpha}"] = sums, errors[:, -1].astype(np.float32)
    if _state["model"] is not None:
        prediction = np.concatenate(list(_state["model"].predict_segment(q32, CONTEXT, HORIZON, "from-start")))
        errors = geodesic_degrees(renormalise(prediction), targets)
        out["sums"]["model"], out["endpoint"]["model"] = errors.sum(0), errors[:, -1].astype(np.float32)
    return out


def paired_bootstrap(a_sum, b_sum, windows, draws=10000, seed=0):
    """bench/stats.py's paired(): the window-weighted mean of a - b, with users (not windows) resampled."""
    per_user = (a_sum - b_sum) / windows
    picks = np.random.default_rng(seed).integers(0, len(windows), (draws, len(windows)))
    boots = (per_user[picks] * windows[picks]).sum(1) / windows[picks].sum(1)
    low, high = np.percentile(boots, [2.5, 97.5])
    return {"mean_diff_deg": float((per_user * windows).sum() / windows.sum()), "ci95": [float(low), float(high)],
            "equal_user_mean_diff_deg": float(per_user.mean()), "users_better": int((per_user < 0).sum()),
            "users": len(windows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--onnx")
    parser.add_argument("--tune-des", action="store_true", help="only score the alpha grid at step 20")
    parser.add_argument("--des-alpha", type=float, nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit-users", type=int, help="first N users only, for a quick check")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()

    with h5py.File(args.h5, "r") as h5:
        users = sorted(h5[args.split])[:args.limit_users]
        tasks = [(f"{args.split}/{u}/{r}/{s}", u, h5[args.split][u][r][s]["quaternion_wxyz"].shape[0])
                 for u in users for r in sorted(h5[args.split][u]) for s in sorted(h5[args.split][u][r])]
    tasks = sorted((t for t in tasks if t[2] >= CONTEXT + HORIZON), key=lambda t: -t[2])  # long segments first
    alphas = list(DES_GRID) if args.tune_des else args.des_alpha
    print(f"{args.split}: {len(users):,} users, {len(tasks):,} segments, "
          f"{sum(t[2] - CONTEXT - HORIZON + 1 for t in tasks):,} windows", flush=True)

    index = {u: i for i, u in enumerate(users)}
    windows = np.zeros(len(users))
    sums, endpoints, started = {}, {}, time.time()
    with multiprocessing.Pool(args.workers, start_worker, (args.h5, args.onnx, alphas, args.tune_des)) as pool:
        for done, result in enumerate(pool.imap_unordered(score_segment, [t[:2] for t in tasks]), 1):
            i = index[result["user"]]
            windows[i] += result["windows"]
            for name, value in result["sums"].items():
                shape = (len(users),) if args.tune_des else (len(users), HORIZON)
                sums.setdefault(name, np.zeros(shape))[i] += value
            for name, value in result["endpoint"].items():
                endpoints.setdefault(name, []).append(value)
            if done % 200 == 0 or done == len(tasks):
                print(f"  {done:,}/{len(tasks):,} segments, {time.time() - started:.0f} s", flush=True)

    total = windows.sum()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.tune_des:
        scores = {name: float(value.sum() / total) for name, value in sums.items()}
        report = {"split": args.split, "users": len(users), "windows": int(total), "step20_mean_deg": scores,
                  "selected": min(scores, key=scores.get)}
        (out_dir / f"des_tuning_{args.split}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(json.dumps(report, indent=1))
        return

    report = {"split": args.split, "users": len(users), "windows": int(total), "h5": Path(args.h5).name,
              "onnx": Path(args.onnx).name if args.onnx else None, "predictors": {}}
    for name, value in sums.items():
        endpoint = np.concatenate(endpoints[name])
        by_step = value.sum(0) / total
        report["predictors"][name] = {
            "mean_deg_step20": float(by_step[-1]), "mean_deg_step10": float(by_step[9]),
            "median_deg_step20": float(np.median(endpoint)), "p95_deg_step20": float(np.percentile(endpoint, 95)),
            "equal_user_mean_deg_step20": float((value[:, -1] / windows).mean()),
            "mean_deg_by_step": [None if np.isnan(x) else float(x) for x in by_step]}
    for name, against in [(n, b) for n in sums for b in ("hold_last", "damped_velocity") if n != b]:
        report["predictors"][name][f"vs_{against}"] = paired_bootstrap(sums[name][:, -1], sums[against][:, -1], windows)
    (out_dir / f"full_{args.split}_summary.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for name, row in report["predictors"].items():
        print(f"{name:26s} 200 ms {row['mean_deg_step20']:.4f}  median {row['median_deg_step20']:.3f}  "
              f"p95 {row['p95_deg_step20']:.3f}  100 ms {row['mean_deg_step10']:.4f}  "
              f"equal-user {row['equal_user_mean_deg_step20']:.4f}")
    for name, row in report["predictors"].items():
        for key in ("vs_hold_last", "vs_damped_velocity"):
            if key in row:
                print(f"{name} {key}: {row[key]}")


if __name__ == "__main__":
    main()
