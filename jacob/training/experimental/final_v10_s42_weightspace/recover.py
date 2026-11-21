"""Recover every learned value from the exact ONNX; patch only approved values.

The 7,552 initializer elements plus three x-bias entries account for all 7,555
learned parameters. A round-trip export checks the graph, constants and weights,
not just tensor shapes. This produces an inference model, never a training resume.
"""
from copy import deepcopy

import numpy as np
import onnx
from onnx import numpy_helper
import torch

from best.export import export_streaming_onnx
from best.model import LeanTCNN
from experimental.v10_endpoint_refit.fit import apply_delta
from experimental.v10_weight_edit.export_check import _signature
from .common import MODEL_SHA, ORIGINAL, sha256

HEAD = "model.head.weight"
BIAS_CONSTANT = "/Constant_53_output_0"


def tensors(graph):
    initializers = {v.name: v for v in graph.initializer}
    constants = {n.output[0]: n.attribute[0].t for n in graph.node if n.op_type == "Constant"}
    return initializers, constants


def recover(path=ORIGINAL):
    if sha256(path) != MODEL_SHA:
        raise ValueError("Recovery is restricted to the verified final ONNX")
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    initializers, constants = tensors(graph.graph)
    arrays = {name: numpy_helper.to_array(t).copy() for name, t in initializers.items()}
    convs = {n.input[1]: n for n in graph.graph.node if n.op_type == "Conv"}
    dilations = [list(next(a.ints for a in convs[f"depthwise.{i}"].attribute if a.name == "dilations"))[1]
                 for i in range(3)]
    kernels = [arrays[f"depthwise.{i}"].shape[-1] for i in range(3)]
    if kernels != [8, 8, 8] or dilations != [1, 8, 47]:
        raise ValueError("Unexpected temporal layout")
    model = LeanTCNN(context=400, horizon=20, channels=64, dropout=0, num_blocks=3,
                     effective_context=400, damped_prior_tau_ms=100, gravity_features=True,
                     mirror_equivariant=True, separable=True, head_basis=3, speed_feature="l1",
                     kernel_sizes=kernels, dilations=dilations, maps="native")
    recovered = {
        "projection.weight": arrays.pop("projection")[..., 0],
        "projection.bias": arrays.pop("model.projection.bias"),
        "head.weight": arrays.pop(HEAD),
    }
    bias = numpy_helper.to_array(constants[BIAS_CONSTANT])
    if bias.shape != (1, 3, 3) or np.any(bias[0, :, 1:]):
        raise ValueError("Unexpected x-only coefficient bias")
    recovered["head_bias_x"] = bias[0, :, 0].copy()
    for i in range(3):
        recovered[f"blocks.{i}.depthwise.weight"] = arrays.pop(f"depthwise.{i}")[:, :, 0, :]
        recovered[f"blocks.{i}.pointwise.weight"] = arrays.pop(f"pointwise.{i}")[:, :, :, 0].transpose(0, 2, 1).reshape(32, 64, 1)
        recovered[f"blocks.{i}.pointwise.bias"] = arrays.pop(f"model.blocks.{i}.pointwise.bias")
    if arrays or set(recovered) != set(model.state_dict()):
        raise ValueError("Unaccounted initializer or model parameter")
    state = {k: torch.from_numpy(v.copy()) for k, v in recovered.items()}
    model.load_state_dict(state, strict=True)
    for name, value in model.state_dict().items():
        if not torch.equal(value, state[name]) or not torch.isfinite(value).all():
            raise ValueError(f"Recovery failed for {name}")
    if sum(p.numel() for p in model.parameters()) != 7555:
        raise ValueError("Parameter count mismatch")
    model.eval().requires_grad_(False)
    return model


def roundtrip(model, path):
    """Export to a new diagnostic file; require all graph constants to agree."""
    if path.exists():
        raise FileExistsError(path)
    export_streaming_onnx(model, path)
    original, exported = onnx.load(str(ORIGINAL)), onnx.load(str(path))
    if _signature(original.graph) != _signature(exported.graph):
        raise ValueError("Round-trip graph structure differs from the original")
    checks = {}
    for kind, a, b in zip(("initializer", "constant"), tensors(original.graph), tensors(exported.graph)):
        if a.keys() != b.keys():
            raise ValueError(f"Round-trip {kind} names differ")
        for name in a:
            equal = np.array_equal(numpy_helper.to_array(a[name]), numpy_helper.to_array(b[name]))
            checks[f"{kind}:{name}"] = bool(equal)
    if not all(checks.values()):
        raise ValueError(f"Round-trip tensor values differ: {[k for k,v in checks.items() if not v]}")
    return {"all_graph_tensors_exact": True, "graph_structure_exact": True, "tensor_checks": checks,
            "parameters": 7555, "nodes": len(original.graph.node), "roundtrip_sha256": sha256(path)}


def edited_model(model, definition):
    if "delta" in definition:
        return apply_delta(model, definition["delta"]).requires_grad_(False)
    edited = deepcopy(model)
    with torch.no_grad():
        edited.head.weight.mul_(definition.get("gain", 1.0))
        edited.head_bias_x.mul_(definition.get("gain", 1.0))
    return edited.eval().requires_grad_(False)


def write_patch(model, path, kind="cubic"):
    """Write head values into a source-graph copy; every other byte must match."""
    if path.exists() or path.resolve() == ORIGINAL.resolve():
        raise FileExistsError(path)
    baseline = onnx.load(str(ORIGINAL))
    changed = deepcopy(baseline)
    source_initializers, source_constants = tensors(baseline.graph)
    initializers, constants = tensors(changed.graph)
    old_head = numpy_helper.to_array(source_initializers[HEAD])
    old_bias = numpy_helper.to_array(source_constants[BIAS_CONSTANT])
    head = model.head.weight.detach().cpu().numpy().copy()
    bias = old_bias.copy()
    bias[0, :, 0] = model.head_bias_x.detach().cpu().numpy()
    allowed_head, allowed_bias = np.zeros_like(head, bool), np.zeros_like(bias, bool)
    if kind == "cubic":
        allowed_head[6:9] = True
        allowed_bias[0, 2, 0] = True
    elif kind == "gain":
        allowed_head[:] = True
        allowed_bias[0, :, 0] = True
    else:
        raise ValueError("Unsupported edit kind")
    if np.any(head[~allowed_head] != old_head[~allowed_head]) or np.any(bias[~allowed_bias] != old_bias[~allowed_bias]):
        raise ValueError("Edit escaped the permitted parameter mask")
    initializers[HEAD].CopyFrom(numpy_helper.from_array(head, name=initializers[HEAD].name))
    constants[BIAS_CONSTANT].CopyFrom(numpy_helper.from_array(bias, name=constants[BIAS_CONSTANT].name))
    restored = deepcopy(changed)
    restored_i, restored_c = tensors(restored.graph)
    restored_i[HEAD].CopyFrom(source_initializers[HEAD])
    restored_c[BIAS_CONSTANT].CopyFrom(source_constants[BIAS_CONSTANT])
    if restored.SerializeToString() != baseline.SerializeToString():
        raise ValueError("Non-head graph content changed")
    # All non-head PyTorch parameters must also remain exact.
    recovered = recover()
    for name, value in recovered.state_dict().items():
        if name not in ("head.weight", "head_bias_x") and not torch.equal(value, model.state_dict()[name].cpu()):
            raise ValueError(f"Frozen parameter changed: {name}")
    onnx.checker.check_model(changed)
    onnx.save(changed, str(path))
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
            "changed_scalars": int(np.count_nonzero(head != old_head) + np.count_nonzero(bias != old_bias)),
            "allowed_scalars": int(allowed_head.sum() + allowed_bias.sum()),
            "all_other_graph_content_exact": True, "kind": kind}


def save_inference(model, path, provenance):
    if path.exists():
        raise FileExistsError(path)
    torch.save({"model_spec": model.specification(),
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "provenance": {"source_onnx_sha256": MODEL_SHA, "inference_only": True,
                               "original_training_checkpoint": False, **provenance}}, path)
