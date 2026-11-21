"""Small, exportable edits to the native k8 model's learned residual head.

Each axis gain scales all three polynomial coefficients of that axis. The
body-velocity prior and feature extractor stay unchanged. These are inference
operations: they do not compute gradients or modify the source model.
"""
from __future__ import annotations

from copy import deepcopy
import math

import torch
from torch.nn import functional as F

from best.model import map_quaternion
from tcnn.model import qmul


IDENTITY_GAINS = (1.0, 1.0, 1.0)


def candidates():
    """The 16 predeclared non-identity candidates, in stable ID order."""
    result = []
    for group in ("global", "x", "y", "z"):
        for label, gain in (("090", .90), ("095", .95), ("105", 1.05), ("110", 1.10)):
            gains = [gain] * 3 if group == "global" else list(IDENTITY_GAINS)
            if group != "global":
                gains[("x", "y", "z").index(group)] = gain
            result.append({"id": f"{group}_{label}", "gains": tuple(gains)})
    return result


def _gains(values):
    gains = tuple(float(value) for value in values)
    if len(gains) != 3 or not all(math.isfinite(value) and value > 0 for value in gains):
        raise ValueError("Gains must be three finite positive numbers in x, y, z order")
    return gains


def _check_model(model):
    if not (model.maps == "native" and model.mirror_equivariant and model.head_basis == 3
            and model.damped_prior_tau_ms is not None):
        raise ValueError("Weight edits require native maps, mirror symmetry, a 3-term head and a damped prior")
    if model.head.weight.shape != (9, model.stream_channels) or model.head_bias_x.shape != (3,):
        raise ValueError("Unexpected residual-head parameter shapes")


@torch.no_grad()
def apply_gains(model, gains):
    """Return an independent model with scaled residual weights and x biases.

    Row 3*j+a predicts axis a of polynomial order j. Both mirror streams use
    that same row, so this edit preserves their sign rule. Only x has a bias.
    """
    _check_model(model)
    gains = _gains(gains)
    edited = deepcopy(model).eval()
    row_gains = edited.head.weight.new_tensor(gains).repeat(edited.head_basis)
    edited.head.weight.mul_(row_gains[:, None])
    edited.head_bias_x.mul_(gains[0])
    return edited


@torch.no_grad()
def decompose(model, history):
    """Return residual [B,20,3], prior [B,20,3], last quaternion [B,1,4].

    This follows the existing model's feature, block and head operations once;
    it does not run a second forward pass. A cache is valid only in eval mode.
    """
    _check_model(model)
    if model.training:
        raise ValueError("Put the model in eval mode before caching its residual")
    if history.ndim != 3 or history.shape[1:] != (model.context, 4):
        raise ValueError(f"Expected [B,{model.context},4], got {tuple(history.shape)}")
    features, last, step = model.features(history[:, -model.effective_context:])
    x = features.transpose(1, 2)
    x = torch.cat((model.projection(x), model.projection(x * model.input_signs[:, None])), 1)
    x = model.blocks(F.relu(x))
    residual = model.readout(x[:, :, -1]).reshape(-1, model.horizon, 3)
    tau = model.damped_prior_tau_ms / 10
    steps = torch.arange(1, model.horizon + 1, dtype=step.dtype, device=step.device)
    prior = step[:, None, :] * (-tau * torch.expm1(-steps / tau))[None, :, None]
    return residual, prior, last


@torch.no_grad()
def reconstruct(residual, prior, last, gains=IDENTITY_GAINS):
    """Apply an axis gain to cached residuals and reconstruct every quaternion."""
    gains = _gains(gains)
    if residual.ndim != 3 or residual.shape[-1] != 3 or prior.shape != residual.shape:
        raise ValueError("Residual and prior must have the same [B,H,3] shape")
    if last.shape != (residual.shape[0], 1, 4):
        raise ValueError("Last orientation must have shape [B,1,4]")
    r = residual * residual.new_tensor(gains) + prior
    return F.normalize(qmul(last, map_quaternion(r, "native")), dim=-1)
