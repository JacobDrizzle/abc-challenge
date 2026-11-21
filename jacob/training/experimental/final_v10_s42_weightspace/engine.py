"""Dense frozen features with real segment boundaries and explicit split guards.

Every temporal sample is processed. Strides select anchors only. A 400-sample
left overlap exceeds the 393-feature receptive field, so chunk boundaries do
not change scored predictions. Targets are never inputs to the feature model.
"""
import time

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from baseline_pipeline import unit_quaternions
from best.model import GravityMirrorTCNN
from experimental.v10_endpoint_refit.fit import check_model

CHUNK = 16384
OVERLAP = 400


def validate_rows(rows, split, allowed_users):
    if split not in ("train", "validation"):
        raise ValueError("Test access is not permitted")
    allowed = set(allowed_users)
    if any(row["user"] not in allowed or not row["path"].startswith(f"{split}/{row['user']}/") for row in rows):
        raise ValueError("Row is outside the explicitly permitted split/user cohort")


def read_segment(handle, row):
    group = handle[row["path"]]
    times = np.asarray(group["time_s"], dtype=np.float64)
    if not np.isfinite(times).all() or not np.allclose(np.diff(times), 0.01, rtol=0, atol=1e-6):
        raise ValueError(f"Invalid time grid: {row['path']}")
    q = unit_quaternions(group["quaternion_wxyz"][:], source=row["path"])
    if len(q) != row["samples"] or len(q) < 420:
        raise ValueError("Segment length differs from the frozen inventory")
    if np.any(np.sum(q[1:] * q[:-1], axis=-1) < -1e-8):
        raise ValueError(f"Discontinuous quaternion signs: {row['path']}")
    return q


@torch.inference_mode()
def dense_parts(model, q, anchors):
    """The actual model trunk and polynomial readout at selected positions."""
    check_model(model)
    if anchors.numel() == 0 or int(anchors.min()) < 399 or int(anchors.max()) >= len(q):
        raise ValueError("Dense anchors must have a complete, same-segment history")
    if sum(block.left_padding for block in model.blocks) != 392:
        raise ValueError("This dense path is restricted to the verified k8 receptive field")
    features, _, _ = model.features(q[None])
    x = features.transpose(1, 2)
    x = F.relu(torch.cat((model.projection(x), model.projection(x * model.input_signs[:, None])), 1))
    hidden = model.blocks(x)[0].index_select(1, anchors).T
    own, other = hidden.split(32, 1)
    return {
        "hidden": hidden,
        "z": torch.cat((own + other, torch.ones_like(own[:, :1]), own - other), 1),
        "coefficients": GravityMirrorTCNN.readout(model, hidden).reshape(-1, 3, 3),
        "last": F.normalize(q.index_select(0, anchors), dim=-1)[:, None],
        "step": features[0].index_select(0, anchors)[:, :3] / 100,
    }


@torch.inference_mode()
def batches(model, data, rows, split, allowed_users, stride, chunk=CHUNK, deadline=None):
    validate_rows(rows, split, allowed_users)
    if stride not in (1, 5, 20) or chunk < 420:
        raise ValueError("Unsupported stride/chunk")
    device = next(model.parameters()).device
    future = torch.arange(1, 21, device=device)
    with h5py.File(data, "r") as handle:
        for row in rows:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Approved six-hour main-run budget reached")
            q64 = read_segment(handle, row)
            q = torch.as_tensor(q64, dtype=torch.float32, device=device)
            target_buffer = torch.as_tensor(q64, dtype=torch.float64, device=device)
            all_anchors = np.arange(399, len(q) - 20, stride, dtype=np.int64)
            for begin in range(0, len(q) - 20, chunk):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Approved six-hour main-run budget reached")
                end = min(begin + chunk, len(q) - 20)
                selected = all_anchors[(all_anchors >= begin) & (all_anchors < end)]
                if not len(selected):
                    continue
                start = max(0, begin - OVERLAP)
                anchors = torch.as_tensor(selected, device=device)
                parts = dense_parts(model, q[start:end], anchors - start)
                parts["first_history"] = q[int(selected[0]) - 399:int(selected[0]) + 1][None]
                target = target_buffer[anchors[:, None] + future]
                yield row, selected, parts, target
