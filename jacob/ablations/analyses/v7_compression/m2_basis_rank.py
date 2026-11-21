"""M2: could the forecast head emit K coefficients per axis on a fixed 20-step basis instead of 60 numbers?

The head's job is the residual on top of the damped prior. For an anchor t (last context sample):
  truth residual  = log(conj(q_t) q_{t+k}) - step * elapsed_k        k = 1..20, body frame of q_t
  model residual  = v6 control's head output (what readout() returns before the prior is added)
Each (anchor, axis) is one row of 20 numbers. Bases are the top-K eigenvectors of the uncentred Gram matrix
of those rows on TRAIN users; they are scored on VALIDATION users by the step-20 geodesic error after
projecting the residual onto them. Test users are never read.
"""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(r"<repo>")
sys.path.insert(0, str(ROOT / "jacob" / "training"))
from baseline_pipeline import EvaluationConfig, inventory, load_split  # noqa: E402
from experimental.v6_gravity_mirror.model import load_checkpoint  # noqa: E402
from tcnn.model import conjugate, qmul, quaternion_to_rotvec, rotvec_to_quaternion  # noqa: E402
from tcnn.onnx_export import geodesic_degrees  # noqa: E402

DATA = ROOT / "data" / "h5" / "qualified_head_100hz_20261026_131953.h5"
MANIFEST = ROOT / "data" / "h5" / "manifest_20261026_131953.json"
CHECKPOINT = ROOT / "jacob" / "training" / "outputs" / "v6_group_a_v2" / "selected_model.pt"  # control seed 44
OUT = Path(__file__).with_name("m2_basis_rank.json")
STRIDE, CONTEXT, HORIZON, KS = 25, 400, 20, (1, 2, 3, 4, 5, 6)
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _ = load_checkpoint(CHECKPOINT)
model = model.to(device)
assert model.mirror_equivariant and model.damped_prior_tau_ms == 100
tau = model.damped_prior_tau_ms / 10  # samples
elapsed = -tau * torch.expm1(-torch.arange(1, HORIZON + 1, dtype=torch.float64) / tau)  # [20]


@torch.no_grad()
def head_output(windows):
    """GravityMirrorTCNN.forward up to the readout: the residual added to the damped prior, [B, 20, 3]."""
    features, _, _ = model.features(windows[:, -model.effective_context:])
    x = features.transpose(1, 2)
    x = torch.cat((model.projection(x), model.projection(x * model.input_signs[:, None])), 1)
    x = model.blocks(F.relu(x))
    return model.readout(x[:, :, -1]).reshape(-1, HORIZON, 3)


def collect(split):
    _, users = load_split(MANIFEST, split)
    parts = {name: [] for name in ("truth", "head", "prior", "last", "target")}
    with h5py.File(DATA, "r") as h5:
        rows, skipped = inventory(h5, sorted(users), EvaluationConfig(), False)
        assert not skipped
        for row in rows:
            q = np.asarray(h5[row["path"]]["quaternion_wxyz"], np.float64)
            q /= np.linalg.norm(q, axis=-1, keepdims=True)
            anchors = np.arange(CONTEXT - 1, len(q) - HORIZON, STRIDE)
            windows = torch.from_numpy(np.stack([q[t - CONTEXT + 1:t + 1] for t in anchors]))  # [A, 400, 4]
            futures = torch.from_numpy(np.stack([q[t + 1:t + HORIZON + 1] for t in anchors]))  # [A, 20, 4]
            last = windows[:, -1:]
            step = quaternion_to_rotvec(qmul(conjugate(windows[:, -2]), windows[:, -1]))  # [A, 3]
            prior = step[:, None, :] * elapsed[None, :, None]
            truth = quaternion_to_rotvec(qmul(conjugate(last), futures))  # broadcasts over the 20 steps
            head = torch.cat([head_output(chunk.float().to(device)).double().cpu()
                              for chunk in windows.split(4096)])
            for name, value in (("truth", truth - prior), ("head", head), ("prior", prior),
                                ("last", last[:, 0]), ("target", futures[:, -1])):
                parts[name].append(value)
    return {name: torch.cat(value) for name, value in parts.items()}, len(rows), len(users)


def basis(residual):
    """Eigenvalues (descending) and eigenvectors [20, 20] of the uncentred Gram matrix of the 20-step rows."""
    rows = residual.permute(0, 2, 1).reshape(-1, HORIZON)
    values, vectors = torch.linalg.eigh(rows.T @ rows)
    return values.flip(0), vectors.flip(1)


def step20_error(residual, data, projector=None):
    if projector is not None:
        residual = torch.einsum("akc,kj->ajc", residual, projector)
    rotation = residual[:, -1] + data["prior"][:, -1]
    prediction = qmul(data["last"], rotvec_to_quaternion(rotation))
    return float(geodesic_degrees(prediction.numpy(), data["target"].numpy()).mean())


start = time.time()
train, train_segments, train_users = collect("train")
validation, val_segments, val_users = collect("validation")
print(f"train {len(train['last']):,} anchors ({train_users} users, {train_segments} segments); "
      f"validation {len(validation['last']):,} anchors ({val_users} users, {val_segments} segments); "
      f"stride {STRIDE}; {time.time() - start:.0f} s")

report = {"stride": STRIDE, "checkpoint": str(CHECKPOINT), "train_anchors": len(train["last"]),
          "validation_anchors": len(validation["last"])}
full_model = step20_error(validation["head"], validation)
damped = step20_error(torch.zeros_like(validation["head"]), validation)
print(f"\nValidation step-20 mean error: damped prior alone {damped:.4f} deg, v6 control (full 60-number head) "
      f"{full_model:.4f} deg")
report.update(damped_deg=damped, full_head_deg=full_model)

for source in ("truth", "head"):
    values, vectors = basis(train[source])
    energy = (values.cumsum(0) / values.sum()).tolist()
    print(f"\nBasis fitted on train {source} residuals: cumulative energy "
          + ", ".join(f"K={k}: {100 * energy[k - 1]:.4f}%" for k in KS))
    rows = []
    for k in KS:
        projector = vectors[:, :k] @ vectors[:, :k].T
        projected_head = step20_error(validation["head"], validation, projector)
        oracle = step20_error(validation["truth"], validation, projector)
        rows.append(dict(k=k, energy=energy[k - 1], head_projected_deg=projected_head,
                         head_delta_deg=projected_head - full_model, truth_projected_deg=oracle))
        print(f"  K={k}: model head projected {projected_head:.4f} deg ({projected_head - full_model:+.4f}); "
              f"truth projected (representation error alone) {oracle:.4f} deg")
    report[source] = dict(eigenvalues=values.tolist(), basis=vectors[:, :max(KS)].T.tolist(), rows=rows)

# Physical alternative with no fitted constants: corrections to rate, acceleration and jerk, t**1..K, t = k/20.
t = torch.arange(1, HORIZON + 1, dtype=torch.float64) / HORIZON
report["polynomial"] = []
print("\nPolynomial basis t, t^2, ... (orthonormalised only for the projection):")
for k in (1, 2, 3):
    q, _ = torch.linalg.qr(torch.stack([t ** (j + 1) for j in range(k)], 1))
    projected_head = step20_error(validation["head"], validation, q @ q.T)
    rows = validation["head"].permute(0, 2, 1).reshape(-1, HORIZON)
    captured = float(((rows @ q) ** 2).sum() / (rows ** 2).sum())
    report["polynomial"].append(dict(k=k, head_energy_validation=captured, head_projected_deg=projected_head,
                                     head_delta_deg=projected_head - full_model))
    print(f"  K={k}: head energy captured {100 * captured:.4f}%; model head projected {projected_head:.4f} deg "
          f"({projected_head - full_model:+.4f})")

OUT.write_text(json.dumps(report, indent=2))
print(f"\nWrote {OUT} ({time.time() - start:.0f} s)")
