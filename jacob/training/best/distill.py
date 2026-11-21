"""Cache a teacher's forecasts for every training window, the targets of distillation (training.py).

python -m best.distill --output outputs/<new teacher folder> [--device cuda]

The teacher is the ensemble C0 scored on 28 Oct: the sign-aligned mean of nine checkpoints (sep_base3_addons and
base3_5_addons with angular loss, and v6 control; seeds 42-44), 4.2725 degrees on validation, 0.041 better than the
best of them. Its cache is outputs/v9_teacher. Distilling k8 from it failed on 28 Oct (beta 0.5: +0.021 degrees), so
a next try needs a stronger teacher: edit TEACHERS. Row i of teacher.npy, [windows, 20, 3] float32, holds the
forecast for TrainingWindows.starts[i] as rotation vectors in the body frame of the window's last sample.
teacher.json records the window fingerprint, the file's SHA-256 and each teacher checkpoint's. Only training users
are read; an existing cache is never replaced.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from baseline_pipeline import DEFAULT_H5, manifest_for, sha256_file
from jsonio import save_json
from tcnn.data import DeviceWindows, check_corpus
from tcnn.model import conjugate, qmul, quaternion_to_rotvec

from best.model import load_checkpoint
from best.training import windows_fingerprint

ROOT = Path(__file__).resolve().parents[1]
TEACHERS = tuple(f"{folder}/seed_{seed}/best.pt" for folder in (
    "outputs/v8_recipe_sep_base3_addons/angular_loss", "outputs/v8_recipe_base3_5_addons/angular_loss",
    "outputs/v6_group_a_v2/control") for seed in (42, 43, 44))


def ensemble(predictions):
    """Sign-aligned mean of unit quaternions [B, H, 4], renormalised (the chordal mean; C0's rule)."""
    reference, total = predictions[0], torch.zeros_like(predictions[0])
    for p in predictions:
        total += p * torch.where((p * reference).sum(-1, keepdim=True) < 0, -1., 1.)
    return F.normalize(total, dim=-1)


def forecast_cache(windows, predict, batch_size=4096):
    """[windows, horizon, 3]: predict's forecast for each of windows.starts, in its last sample's body frame."""
    cache = np.empty((len(windows.starts), windows.horizon, 3), np.float32)
    with torch.inference_mode():
        for begin in range(0, len(windows.starts), batch_size):
            history, _ = windows._gather(windows.starts[begin:begin + batch_size])
            last = F.normalize(history[:, -1:], dim=-1)
            cache[begin:begin + len(history)] = quaternion_to_rotvec(qmul(conjugate(last), predict(history))).cpu().numpy()
    return cache


def save_cache(output, cache, windows, details):
    """teacher.npy and teacher.json in a new folder."""
    output = Path(output)
    output.mkdir(parents=True)
    np.save(output / "teacher.npy", cache)
    save_json(output / "teacher.json", {"windows_fingerprint": windows_fingerprint(windows), "shape": list(cache.shape),
                                        "sha256": sha256_file(output / "teacher.npy"), **details})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_H5)
    p.add_argument("--output", type=Path, required=True, help="A new folder, e.g. outputs/teacher_<name>")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.output.exists():
        p.error(f"{args.output} exists; a teacher cache is never replaced")
    audit = check_corpus(args.data, manifest_for(args.data))
    # The rows and order train_tcnn gives its training DeviceWindows, so row i matches starts[i] there too.
    windows = DeviceWindows(args.data, [r for r in audit["segments"] if r["split"] == "train"], args.device)
    models = [load_checkpoint(ROOT / path)[0].to(args.device) for path in TEACHERS]
    print(f"Caching the {len(models)}-model ensemble's forecasts for {len(windows.starts):,} training windows", flush=True)
    cache = forecast_cache(windows, lambda history: ensemble([F.normalize(m(history), dim=-1) for m in models]))
    save_cache(args.output, cache, windows, {
        "data_sha256": audit["data_sha256"], "split_sha256": audit["split_sha256"], "rule": ensemble.__doc__,
        "teachers": {path: sha256_file(ROOT / path) for path in TEACHERS}})
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
