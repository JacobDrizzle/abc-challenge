"""Is head tilt a constant bias (learnable) or does it vary? True gravity in the head frame, validation users."""
import sys
sys.path.insert(0, r"<repo>\jacob\training")
import numpy as np
import torch
from baseline_pipeline import DEFAULT_H5, manifest_for
from experimental.v6_gravity_mirror.model import gravity_in_head_frame
from tcnn.data import DeviceWindows, check_corpus

audit = check_corpus(DEFAULT_H5, manifest_for(DEFAULT_H5))
rows = [r for r in audit["segments"] if r["split"] == "validation"]
data = DeviceWindows(DEFAULT_H5, rows, "cpu")
g = gravity_in_head_frame(data.buffer.double())          # world up in head frame, per sample
mean = g.mean(0); mean = mean / mean.norm()
deg = lambda a, b: torch.rad2deg(torch.arccos((a * b).sum(-1).clamp(-1, 1))).numpy()
pitch = torch.rad2deg(torch.atan2(g[:, 2], g[:, 1])).numpy(); roll = torch.rad2deg(torch.atan2(-g[:, 0], g[:, 1])).numpy()
print(f"mean tilt direction in head frame {mean.numpy().round(3)}: {deg(mean, torch.tensor([0., 1, 0], dtype=torch.float64)).item():.1f} deg from head-up")
print(f"pitch (+ = gravity toward face front) mean {pitch.mean():.1f} sd {pitch.std():.1f}; roll mean {roll.mean():.1f} sd {roll.std():.1f}")
a = deg(g, mean[None]); print(f"per-sample tilt vs that mean: median {np.median(a):.1f}, p90 {np.percentile(a, 90):.1f}")
users = {}
for (offset, length), row in zip(data.bounds, rows):
    users.setdefault(row["user"], []).append(g[offset:offset + length])
um = torch.stack([torch.cat(v).mean(0) for v in users.values()]); um = um / um.norm(dim=-1, keepdim=True)
print(f"per-user mean posture vs overall mean: " + ", ".join(f"{x:.1f}" for x in sorted(deg(um, mean[None]))))
# Slow posture: 4 s mean of true gravity in head frame, vs the overall mean (what a learned bias could not fix).
slow = []
for offset, length in data.bounds:
    c = torch.cat((torch.zeros(1, 3, dtype=g.dtype), torch.cumsum(g[offset:offset + length], 0)))
    ends = torch.arange(399, length, 50)
    m = c[ends + 1] - c[ends + 1 - 400]; slow.append(deg(m / m.norm(dim=-1, keepdim=True), mean[None]))
s = np.concatenate(slow); print(f"4 s posture vs overall mean: median {np.median(s):.1f}, p90 {np.percentile(s, 90):.1f}")
