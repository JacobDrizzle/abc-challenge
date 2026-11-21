"""Step 1 of the generalisation report: per-segment errors of checkpoints and baselines on every merged-file user.

For every filter-qualified segment of every user in the 800/120/160 merged HDF5, this writes the number of windows
scored and the sums of the step-10 and step-20 geodesic errors. Train users are scored at stride 25 (per-user means
stay unbiased, at 1/25 of the cost); validation and test users at stride 1, which is every window, as the bench
scores them. For validation and test users it also keeps the per-window step-20 errors, for medians and quantiles.

Two kinds of run:
- `baselines`: hold last, constant velocity (lag 1) and damped velocity (lag 1, tau 100 ms). Model-independent,
  so run it once.
- `label=checkpoint` pairs: one predictor, "model", per checkpoint.

Outputs:
- scores/<label>.json next to this script: the per-segment sums (small; the report reads these)
- jacob/training/outputs/generalisation/<label>.npz: per-window step-20 errors (gitignored; about 45 MB a predictor)

Nothing is written to bench/results/: the leaderboard globs that folder and only compares same-data models.

Run from jacob/training with the CUDA venv:
    python ../ablations/analyses/generalisation/score_users.py baselines
    python ../ablations/analyses/generalisation/score_users.py k8_100_s42=outputs/v9_confirm/k8_combined/seed_42/best.pt
"""
import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parents[2] / "training"
sys.path.insert(0, str(TRAINING))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from baseline_pipeline import manifest_for, sha256_file  # noqa: E402
from best.model import load_checkpoint  # noqa: E402
from tcnn.data import DeviceWindows, check_corpus  # noqa: E402
from tcnn.runtime import configure_runtime  # noqa: E402
from tcnn.model import conjugate, qmul, quaternion_to_rotvec, relative_history, rotvec_to_quaternion  # noqa: E402
from train_tcnn import geodesic_degrees_torch, reference_predictors  # noqa: E402

MERGED = TRAINING.parents[1] / "data" / "h5" / "qualified_head_100hz_merged_20261101_140915.h5"
STRIDES = {"train": 25, "validation": 1, "test": 1}
PER_WINDOW_SPLITS = ("validation", "test")
SCORES = HERE / "scores"
ARRAYS = TRAINING / "outputs" / "generalisation"
HORIZON = 20


def constant_velocity(horizon):
    """Lag-1 constant velocity, the FAQ's required baseline: the damped prior with tau -> infinity.

    relative[:, -2] is the previous sample in the latest head frame, so conj() of it is the last step's rotation in
    that frame, and step h is last * Exp(h * step). This equals bench.baselines.ConstantVelocity, which applies the
    same rotation in the world frame: Exp(h Log(now past^-1)) now = now Exp(h Log(past^-1 now)).
    """
    def predict(history):
        relative, last = relative_history(history)
        step = quaternion_to_rotvec(conjugate(relative[:, -2]))
        h = torch.arange(1, horizon + 1, dtype=step.dtype, device=step.device)
        return qmul(last, rotvec_to_quaternion(step[:, None, :] * h[None, :, None]))
    return predict


def segment_counts(bounds, stride, context=400, horizon=HORIZON):
    """Windows that DeviceWindows.windows(stride) yields per segment, in its order."""
    return [(length - context - horizon) // stride + 1 for _, length in bounds]


def score_user(predictors, path, rows, device, stride, batch):
    """{predictor: [n_windows, 2] float64 errors at steps 10 and 20} and the per-segment window counts."""
    data = DeviceWindows(path, rows, device)
    counts = segment_counts(data.bounds, stride, data.context, data.horizon)
    errors = {name: [] for name in predictors}
    with torch.inference_mode():
        for history, target in data.windows(stride, batch):
            for name, predict in predictors.items():
                errors[name].append(geodesic_degrees_torch(predict(history), target)[:, [9, 19]].cpu().numpy())
    errors = {name: np.concatenate(parts) for name, parts in errors.items()}
    if any(len(e) != sum(counts) for e in errors.values()):
        raise RuntimeError(f"{rows[0]['user']}: window count does not match the segment bounds")
    return errors, counts


def run(label, predictors, args, checkpoint=None):
    device = torch.device(args.device)
    audit = check_corpus(args.h5, manifest_for(args.h5))
    rows = sorted(audit["segments"], key=lambda r: (r["split"], r["user"], r["key"]))
    if args.users:
        rows = [r for r in rows if r["user"] in args.users]
    by_user = {}
    for row in rows:
        by_user.setdefault(row["user"], []).append(row)

    segments = []
    sums = {name: {"step10": [], "step20": []} for name in predictors}
    per_window = {name: [] for name in predictors}
    per_window_segment = []
    start = time.time()
    for done, (user, user_rows) in enumerate(by_user.items(), 1):
        split = user_rows[0]["split"]
        stride = STRIDES[split]
        errors, counts = score_user(predictors, args.h5, user_rows, device, stride, args.batch)
        edges = np.cumsum([0] + counts)
        for row, count, lo, hi in zip(user_rows, counts, edges[:-1], edges[1:]):
            if stride == 1 and count != row["windows"]:
                raise RuntimeError(f"{row['key']}: {count} windows at stride 1, the corpus audit says {row['windows']}")
            segments.append({"key": row["key"], "user": user, "recording": row["recording"], "split": split,
                             "stride": stride, "windows_scored": int(count), "windows_total": int(row["windows"])})
            for name, e in errors.items():
                sums[name]["step10"].append(float(e[lo:hi, 0].sum()))
                sums[name]["step20"].append(float(e[lo:hi, 1].sum()))
                if split in PER_WINDOW_SPLITS:
                    per_window[name].append(e[lo:hi, 1].astype(np.float32))
            if split in PER_WINDOW_SPLITS:
                per_window_segment.append(np.full(count, len(segments) - 1, np.int32))
        if done % 50 == 0 or done == len(by_user):
            print(f"{label}: {done}/{len(by_user)} users, {time.time() - start:.0f} s", flush=True)

    SCORES.mkdir(parents=True, exist_ok=True)
    ARRAYS.mkdir(parents=True, exist_ok=True)
    arrays = ARRAYS / f"{label}.npz"
    np.savez(arrays, segment=np.concatenate(per_window_segment),
             **{name: np.concatenate(parts) for name, parts in per_window.items()})
    record = {
        "schema": "generalisation-scores-v1", "label": label,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
        "data_file": Path(args.h5).name, "data_sha256": audit["data_sha256"],
        "split_windows_total": audit["windows"], "strides": STRIDES, "horizon": HORIZON,
        "predictors": list(predictors), "subset_users": sorted(args.users) if args.users else None,
        "per_window_file": str(arrays), "per_window_splits": list(PER_WINDOW_SPLITS),
        "seconds": round(time.time() - start),
        "segments": segments, "sums": sums}
    out = SCORES / f"{label}.json"
    out.write_text(json.dumps(record) + "\n", encoding="utf-8")
    print(f"wrote {out} and {arrays}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="'baselines' or label=checkpoint (the checkpoint relative to jacob/training)")
    p.add_argument("--h5", type=Path, default=MERGED)
    # k8 peaks at about 2.2 GiB per 1,024 windows. On the 10 GB RTX 3080, 4,096 spills into shared memory and runs
    # 3x slower; 512-2,048 all run at about 13k windows/s.
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--users", nargs="+", help="Only these users (for a quick check; the report needs every user)")
    args = p.parse_args()
    args.users = set(args.users) if args.users else None
    configure_runtime()  # the trainer's policy for validation and test: float32, no TF32, deterministic

    for spec in args.runs:
        if spec == "baselines":
            predictors = {**reference_predictors(HORIZON), "constant_velocity": constant_velocity(HORIZON)}
            run("baselines", predictors, args)
        else:
            label, _, path = spec.partition("=")
            if not path:
                p.error(f"{spec}: expected label=checkpoint")
            checkpoint = (TRAINING / path).resolve()
            model = load_checkpoint(checkpoint)[0].to(args.device).eval()
            run(label, {"model": model}, args, checkpoint)


if __name__ == "__main__":
    main()
