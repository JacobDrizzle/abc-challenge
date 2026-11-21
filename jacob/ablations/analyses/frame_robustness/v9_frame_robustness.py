"""v9 copy of the 27 Oct frame_robustness.py (v9 checkpoints). Frame robustness: validation error when the stream arrives in another coordinate convention.

The same transform is applied to every sample of a segment (context and targets), as a harness in that convention
would deliver it. Read-only: loads checkpoints and the group-a-v2 HDF5, writes one JSON next to this script.
Usage (from jacob/training): python <this script> [stride] [label substring ...]  (identity always runs)
"""
import json
import math
from pathlib import Path
import sys

TRAINING = Path(r"<repo>\jacob\training")
sys.path.insert(0, str(TRAINING))

import numpy as np
import torch
from torch.nn import functional as F

from baseline_pipeline import DEFAULT_H5, manifest_for
from experimental.v6_gravity_mirror.model import gravity_in_head_frame
from experimental.v9_lean.model import load_checkpoint as load_v9
from tcnn.data import DeviceWindows, check_corpus
from tcnn.model import qmul
from tcnn.runtime import configure_runtime
from train_tcnn import reference_predictors, score_windows

OUT = TRAINING / "outputs"
# name -> (loader, checkpoint, gravity, mirror)
MODELS = {  # the current best and the v9 screen runs it is compared with (seed 42)
    "sep+angular s44": (load_v9, "v8_recipe_sep_base3_addons/angular_loss/seed_44/best.pt", True, True),
    "v9 control s42": (load_v9, "v9_screen/control/seed_42/best.pt", True, True),
    "v9 tilt_15 s42": (load_v9, "v9_screen/tilt_15/seed_42/best.pt", True, True),
    "k8 screen s42": (load_v9, "v9_screen/k8_3blocks_w32/seed_42/best.pt", True, True),
    "k8_combined s42": (load_v9, "v9_confirm/k8_combined/seed_42/best.pt", True, True),
    "k5_combined s42": (load_v9, "v9_confirm/k5_combined/seed_42/best.pt", True, True),
}


def rotation(axis, degrees, device):
    """Unit quaternion (w, x, y, z) for a rotation about a coordinate axis."""
    half = math.radians(degrees) / 2
    q = torch.zeros(4, device=device)
    q[0], q[1 + "xyz".index(axis)] = math.cos(half), math.sin(half)
    return q


def conj(q):
    return q * torch.tensor([1., -1., -1., -1.], device=q.device)


def per_segment(bounds, fn):
    """Apply fn(segment) -> segment to each segment of the concatenated buffer."""
    def apply(buffer):
        out = buffer.clone()
        for offset, length in bounds:
            out[offset:offset + length] = fn(buffer[offset:offset + length])
        return out
    return apply


def yaw_only(q):
    """Twist of q about world +y (swing-twist decomposition): the heading part of an orientation."""
    return F.normalize(q * torch.tensor([1., 0., 1., 0.], device=q.device), dim=-1)


def transforms(bounds, device):
    left = lambda g: (lambda b: qmul(g.expand_as(b), b))
    signs = lambda *s: (lambda b: b * torch.tensor(s, device=device))
    ros = lambda b: torch.stack((b[:, 0], -b[:, 3], b[:, 1], -b[:, 2]), -1)
    return {
        # Controls: every model should be unchanged (mirror_x: only mirror-equivariant ones).
        "identity": lambda b: b,
        "heading +90 deg (world yaw)": left(rotation("y", 90, device)),
        "re-centre heading on first pose": per_segment(bounds, lambda s: qmul(conj(yaw_only(s[0])).expand_as(s), s)),
        "left-right mirror (x = 0 plane)": signs(1., 1., -1., -1.),
        # World frame only: only the gravity channels see these.
        "world tilted 5 deg": left(rotation("x", 5, device)),
        "world tilted 15 deg": left(rotation("x", 15, device)),
        "z-up world (up axis off by 90 deg)": left(rotation("x", 90, device)),
        "re-centre full first pose": per_segment(bounds, lambda s: qmul(conj(s[0]).expand_as(s), s)),
        # Whole conventions (world and head axes): every learned model sees these.
        "OpenXR handedness (flip z)": signs(1., -1., -1., 1.),
        "ROS/ENU (z-up, x fwd, y left)": ros,
        # BOXRR players face one way, so in ROS axes the world's +y (left) is nearly the head's +y; turn them first.
        "ROS/ENU, player heading +90 deg": lambda b: ros(left(rotation("y", 90, device))(b)),
        "ROS/ENU, player heading +180 deg": lambda b: ros(left(rotation("y", 180, device))(b)),
    }


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    configure_runtime()
    device = torch.device("cuda")
    audit = check_corpus(DEFAULT_H5, manifest_for(DEFAULT_H5))
    data = DeviceWindows(DEFAULT_H5, [r for r in audit["segments"] if r["split"] == "validation"], device)
    original = data.buffer.clone()
    first = torch.stack([original[o] for o, _ in data.bounds])
    tilt = torch.rad2deg(torch.arccos(gravity_in_head_frame(first)[:, 1].clamp(-1, 1))).cpu().numpy()
    print(f"{len(data.bounds)} segments; head tilt from upright at each segment's first sample: "
          f"median {np.median(tilt):.1f} deg, range {tilt.min():.1f}-{tilt.max():.1f}", flush=True)

    predictors = {"damped velocity (baseline)": reference_predictors(20)["damped_velocity"]}
    for name, (loader, path, _, _) in MODELS.items():
        predictors[name] = loader(OUT / path)[0].to(device).eval()
    results = {}
    wanted = sys.argv[2:]
    for label, transform in transforms(data.bounds, device).items():
        if wanted and label != "identity" and not any(w in label for w in wanted):
            continue
        data.buffer = F.normalize(transform(original), dim=-1)
        scores = score_windows(predictors, data.windows(stride, 4096), 20)
        results[label] = {name: s["mean_endpoint_deg"] for name, s in scores.items()}
        print(label, {k: round(v, 4) for k, v in results[label].items()}, flush=True)
    data.buffer = original

    names = list(predictors)
    base = results["identity"]
    header = "| Stream convention | " + " | ".join(names) + " |"
    lines = [header, "| --- |" + " ---: |" * len(names),
             "| identity (BOXRR/Unity) | " + " | ".join(f"{base[n]:.4f}" for n in names) + " |"]
    for label, row in results.items():
        if label != "identity":
            lines.append(f"| {label} | " + " | ".join(f"{row[n] - base[n]:+.4f}" for n in names) + " |")
    table = "\n".join(lines)
    print("\nStep-20 mean (deg) on identity; other rows are the change vs identity.\n" + table)
    meta = {n: {"checkpoint": p, "gravity": g, "mirror": m} for n, (_, p, g, m) in MODELS.items()}
    out = Path(__file__).with_name("v9_frame_robustness" + ("_subset" if wanted else "") + ".json")
    out.write_text(json.dumps({"stride": stride, "windows": next(iter(scores.values()))["windows"],
                               "first_pose_tilt_deg": {"median": float(np.median(tilt)), "min": float(tilt.min()),
                                                       "max": float(tilt.max())},
                               "models": meta, "results": results, "table_md": table}, indent=1),
                   encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
