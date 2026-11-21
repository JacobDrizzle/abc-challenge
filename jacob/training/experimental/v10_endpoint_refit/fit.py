"""Frozen features, quaternion-alignment normal equations, and cubic-only edits.

Coefficient order is x's 32 weights and bias, y's 32 weights, z's 32 weights.
All fitting statistics use train samples only. No gradients are constructed.
"""
from copy import deepcopy
import math

import numpy as np
import torch
from torch.nn import functional as F

from best.model import GravityMirrorTCNN, map_quaternion
from tcnn.model import conjugate, qmul

PENALTIES = (1e-4, 1e-3, 1e-2, .1, 1., 10., 100.)
STRENGTHS = (.25, .5, 1.)


def check_model(model):
    if not (model.maps == "native" and model.mirror_equivariant and model.head_basis == 3
            and model.stream_channels == 32 and model.context == 400 and model.horizon == 20
            and model.damped_prior_tau_ms == 100 and not model.training):
        raise ValueError("Expected the eval-mode, native, mirror-equivariant 800-user k8 model")


@torch.inference_mode()
def extract(model, history):
    """Run the frozen trunk once and retain its exact original polynomial head."""
    check_model(model)
    features, last, step = model.features(history[:, -model.effective_context:])
    x = features.transpose(1, 2)
    x = torch.cat((model.projection(x), model.projection(x * model.input_signs[:, None])), 1)
    x = model.blocks(F.relu(x))[:, :, -1]
    own, other = x.split(32, 1)
    z = torch.cat((own + other, torch.ones_like(own[:, :1]), own - other), 1)
    coefficients = GravityMirrorTCNN.readout(model, x).reshape(-1, 3, 3)
    return {"z": z, "coefficients": coefficients, "step": step, "last": last}


def residual_and_prior(parts):
    time = torch.arange(1, 21, device=parts["step"].device, dtype=torch.float64) / 20
    basis = torch.stack([time ** j for j in (1, 2, 3)], 1).to(parts["step"].dtype)
    residual = torch.matmul(basis, parts["coefficients"])
    steps = torch.arange(1, 21, device=time.device, dtype=parts["step"].dtype)
    prior = parts["step"][:, None] * (-10 * torch.expm1(-steps / 10))[None, :, None]
    return residual, prior


def endpoint_change(z, delta):
    delta = torch.as_tensor(delta, dtype=z.dtype, device=z.device)
    if delta.shape != (97,) or not torch.isfinite(delta).all():
        raise ValueError("Expected 97 finite endpoint coefficient changes")
    return torch.stack((z[:, :33] @ delta[:33], z[:, 33:] @ delta[33:65],
                        z[:, 33:] @ delta[65:]), -1)


@torch.inference_mode()
def predict(parts, delta=None, gain=1.):
    residual, prior = residual_and_prior(parts)
    r = gain * residual + prior
    if delta is not None:
        time = torch.arange(1, 21, dtype=torch.float64, device=r.device) / 20
        cubic = time.pow(3).to(r.dtype)
        r = r + cubic[None, :, None] * endpoint_change(parts["z"], delta)[:, None]
    return F.normalize(qmul(parts["last"], map_quaternion(r, "native")), dim=-1)


@torch.no_grad()
def apply_delta(model, delta):
    """Make an independent copy; change only the 97 cubic-term coefficients."""
    check_model(model)
    delta = torch.as_tensor(delta, dtype=model.head.weight.dtype, device=model.head.weight.device)
    if delta.shape != (97,) or not torch.isfinite(delta).all():
        raise ValueError("Expected 97 finite changes")
    edited = deepcopy(model).eval()
    edited.head.weight[6].add_(delta[:32])
    edited.head_bias_x[2].add_(delta[32])
    edited.head.weight[7].add_(delta[33:65])
    edited.head.weight[8].add_(delta[65:])
    return edited


class NormalEquations:
    """Stream the exact 97-variable quadratic surrogate in float64.

For original a=(2,r), target unit quaternion q and correction Jacobian J,
minimize sum_i u_i ||(I-q_i q_i^T)(a_i+J_i delta)||^2 / ||a_i||^2.
Here u=1/max(original angle,1 degree), subsequently normalized to sum to one.
The projector is invariant to q's sign and never divides by q's scalar part.
"""
    def __init__(self, device="cpu"):
        self.gram = torch.zeros(97, 97, dtype=torch.float64, device=device)
        self.rhs = torch.zeros(97, dtype=torch.float64, device=device)
        self.covariance = torch.zeros_like(self.gram)
        self.weight_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.residual_square_sum = torch.zeros_like(self.weight_sum)
        self.count = 0

    @torch.inference_mode()
    def update(self, z, r, residual, relative_target):
        z, r, residual = z.double(), r.double(), residual.double()
        if not torch.isfinite(relative_target).all() or torch.any(relative_target.norm(dim=-1) < 1e-8):
            raise ValueError("Invalid target quaternion")
        q = F.normalize(relative_target.double(), dim=-1)
        if z.ndim != 2 or z.shape[1] != 65 or r.shape != (len(z), 3) or residual.shape != r.shape or q.shape != (len(z), 4):
            raise ValueError("Invalid fitting batch shape")
        if not all(torch.isfinite(t).all() for t in (z, r, residual, q)):
            raise ValueError("Nonfinite fitting inputs")
        if not torch.all(z[:, 32] == 1):
            raise ValueError("The x intercept feature must be one")
        a = torch.cat((torch.full_like(r[:, :1], 2.), r), 1)
        norm_squared = a.square().sum(1)
        angle = 2 * torch.acos(((a * q).sum(1) / norm_squared.sqrt()).abs().clamp(0, 1))
        u = angle.clamp_min(math.pi / 180).reciprocal()
        weight = u / norm_squared
        even, odd = z[:, :33], z[:, 33:]
        features = (even, odd, odd)
        slices = (slice(0, 33), slice(33, 65), slice(65, 97))
        # J^T P J = J^T J - (J^T q)(J^T q)^T, exploiting J's three sparse rows.
        jq = torch.cat([feature * q[:, axis + 1:axis + 2] for axis, feature in enumerate(features)], 1)
        projected = r - q[:, 1:] * (a * q).sum(1, keepdim=True)
        self.gram -= jq.T @ (weight[:, None] * jq)
        for axis, (feature, block) in enumerate(zip(features, slices)):
            self.gram[block, block] += feature.T @ (weight[:, None] * feature)
            self.rhs[block] -= feature.T @ (weight * projected[:, axis])
            self.covariance[block, block] += feature.T @ feature
        self.weight_sum += u.sum()
        self.residual_square_sum += residual.square().sum()
        self.count += len(z)

    @torch.inference_mode()
    def update_parts(self, parts, target):
        residual, prior = residual_and_prior(parts)
        relative = qmul(conjugate(parts["last"].double()), target[:, -1:].double())[:, 0]
        self.update(parts["z"], (residual + prior)[:, -1], residual[:, -1], relative)

    def arrays(self):
        if self.count == 0:
            raise ValueError("Cannot fit an empty training set")
        covariance = self.covariance.cpu().numpy() / self.count
        scales = np.sqrt(np.maximum(np.diag(covariance), 0)).clip(1e-6)
        scales[32] = 1.
        gram = self.gram.cpu().numpy() / float(self.weight_sum)
        gram = (gram + gram.T) / 2
        return {"gram": gram, "rhs": self.rhs.cpu().numpy() / float(self.weight_sum),
                "covariance": covariance, "scales": scales, "count": np.array(self.count),
                "residual_rms": np.array(math.sqrt(float(self.residual_square_sum) / self.count))}


def solve(arrays, penalty):
    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("Penalty must be finite and positive")
    scale = arrays["scales"]
    gram = arrays["gram"] / scale[:, None] / scale[None, :]
    standardized = np.linalg.solve(gram + penalty * np.eye(97), arrays["rhs"] / scale)
    delta = standardized / scale
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite fitted correction")
    return delta


def candidate(arrays, penalty, strength):
    if strength not in STRENGTHS:
        raise ValueError("Unexpected correction strength")
    delta = solve(arrays, penalty)
    rms = math.sqrt(max(0., float(delta @ arrays["covariance"] @ delta)))
    limit = .1 * float(arrays["residual_rms"])
    cap = min(1., limit / rms) if rms > 0 else 1.
    delta = delta * cap * strength
    final_rms = math.sqrt(max(0., float(delta @ arrays["covariance"] @ delta)))
    return {"id": f"ridge_{penalty:g}_strength_{strength:g}", "penalty": penalty,
            "strength": strength, "cap_factor": cap, "train_rms_correction": final_rms,
            "train_rms_limit": limit, "delta": delta.tolist()}


def candidates(arrays):
    return [candidate(arrays, penalty, strength) for penalty in PENALTIES for strength in STRENGTHS]
