"""A1: eval-mode step-20 error of a trained checkpoint on train users vs validation users, per user.

Read-only: loads the checkpoint and the group-a-v2 HDF5, writes one JSON next to this script.
Usage (from jacob/training): python <this script> <checkpoint> [stride]
"""
import json
from pathlib import Path
import sys

TRAINING = Path(r"<repo>\jacob\training")
sys.path.insert(0, str(TRAINING))

import numpy as np
import torch

from baseline_pipeline import DEFAULT_H5, manifest_for
from experimental.v6_gravity_mirror.model import load_checkpoint
from tcnn.data import DeviceWindows, check_corpus
from tcnn.runtime import configure_runtime
from train_tcnn import reference_predictors, score_windows


def main():
    checkpoint = Path(sys.argv[1]).resolve()
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    configure_runtime()
    device = torch.device("cuda")
    model, state = load_checkpoint(checkpoint)
    model = model.to(device).eval()
    audit = check_corpus(DEFAULT_H5, manifest_for(DEFAULT_H5))
    predictors = {"model": model, **reference_predictors(20)}
    result = {"checkpoint": str(checkpoint), "stride": stride, "splits": {}}
    for split in ("train", "validation"):
        rows = [r for r in audit["segments"] if r["split"] == split]
        users = {}
        for user in sorted({r["user"] for r in rows}):
            data = DeviceWindows(DEFAULT_H5, [r for r in rows if r["user"] == user], device)
            scores = score_windows(predictors, data.windows(stride, 4096), 20)
            users[user] = {name: {"windows": s["windows"], "step20": s["mean_endpoint_deg"],
                                  "step10": s["mean_deg_by_horizon"][9]} for name, s in scores.items()}
            del data
        windows = np.array([u["model"]["windows"] for u in users.values()], dtype=float)
        summary = {}
        for name in predictors:
            e20 = np.array([u[name]["step20"] for u in users.values()])
            summary[name] = {"window_weighted_step20": float((e20 * windows).sum() / windows.sum()),
                             "equal_user_step20": float(e20.mean())}
        ratio = np.array([u["model"]["step20"] / u["damped_velocity"]["step20"] for u in users.values()])
        summary["model_over_damped_equal_user"] = float(ratio.mean())
        summary["model_over_damped_window_weighted"] = (summary["model"]["window_weighted_step20"]
                                                        / summary["damped_velocity"]["window_weighted_step20"])
        summary["users"], summary["windows"] = len(users), int(windows.sum())
        result["splits"][split] = {"summary": summary, "users": users}
        print(split, json.dumps(summary, indent=1), flush=True)
    out = Path(__file__).with_name(f"a1_{checkpoint.parent.parent.name}_{checkpoint.parent.name}.json")
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
