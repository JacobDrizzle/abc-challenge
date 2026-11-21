"""How well does the mean head-up direction over a trailing window recover world up? Validation users, read-only."""
import sys
from pathlib import Path
sys.path.insert(0, r"<repo>\jacob\training")
import numpy as np
import torch
from baseline_pipeline import DEFAULT_H5, manifest_for
from tcnn.data import DeviceWindows, check_corpus

def head_up_in_world(q):  # second column of R(q): the head's +y axis in world coordinates
    w, x, y, z = q.unbind(-1)
    return torch.stack((2*(x*y - w*z), w*w - x*x + y*y - z*z, 2*(y*z + w*x)), -1)

audit = check_corpus(DEFAULT_H5, manifest_for(DEFAULT_H5))
data = DeviceWindows(DEFAULT_H5, [r for r in audit["segments"] if r["split"] == "validation"], "cpu")
print(f"{'window':>18} {'median':>7} {'p90':>6} {'p99':>6}   angle (deg) between mean head-up and world up")
for span, spacing in ((1, 1), (25, 1), (50, 1), (100, 1), (144, 1), (144, 4), (144, 8), (158, 1), (400, 1), (400, 8)):
    angles = []
    for offset, length in data.bounds:
        u = head_up_in_world(data.buffer[offset:offset + length].double())
        c = torch.cat((torch.zeros(1, 3, dtype=u.dtype), torch.cumsum(u, 0)))
        ends = torch.arange(span - 1, length, 7)  # every 7th sample
        if spacing == 1:
            m = c[ends + 1] - c[ends + 1 - span]
        else:
            idx = ends[:, None] - torch.arange(0, span, spacing)[None]
            m = u[idx].sum(1)
        m = m / m.norm(dim=-1, keepdim=True)
        angles.append(torch.rad2deg(torch.arccos(m[:, 1].clamp(-1, 1))).numpy())
    a = np.concatenate(angles)
    print(f"{span:>5} samples, d={spacing}: {np.median(a):7.2f} {np.percentile(a, 90):6.2f} {np.percentile(a, 99):6.2f}")
