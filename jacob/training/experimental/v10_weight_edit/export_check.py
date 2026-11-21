"""Verify unchanged deployment structure after a frozen, head-only weight edit.

Only generated motion is used here. Both source checkpoints are read-only and
every export is written beneath a new output directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from best.export import StreamingLean, export_streaming_onnx
from best.model import load_checkpoint
from tcnn.onnx_export import audit_operators, count_macs, geodesic_degrees


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _signature(graph):
    """Compare structure and constant shapes, while allowing weight values to differ."""
    def attribute(item):
        if item.type == onnx.AttributeProto.TENSOR:
            return [item.name, item.type, item.t.data_type, list(item.t.dims)]
        return [item.name, item.type, item.SerializeToString().hex()]

    return {
        "inputs": [[v.name, v.type.SerializeToString().hex()] for v in graph.input],
        "outputs": [[v.name, v.type.SerializeToString().hex()] for v in graph.output],
        "initializers": [[v.name, v.data_type, list(v.dims)] for v in graph.initializer],
        "nodes": [[n.domain, n.op_type, list(n.input), list(n.output),
                   [attribute(a) for a in n.attribute]] for n in graph.node],
    }


def _motion(length=650, seed=73):
    """The project's random-walk test construction, including quaternion sign flips."""
    rng = np.random.default_rng(seed)
    q = np.cumsum(rng.normal(0, .05, (length, 4)), axis=0) + [1., 0., 0., 0.]
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q *= np.where(rng.random((length, 1)) < .1, -1, 1)
    return q.astype(np.float32)


def _windows(stream):
    for t in range(1, len(stream)):
        window = stream[max(0, t - 399):t + 1]
        if len(window) < 400:
            window = np.concatenate((np.repeat(window[:1], 400 - len(window), axis=0), window))
        yield t, np.ascontiguousarray(window[None])


def _parity(model, session, shapes, stream):
    wrapper = StreamingLean(model).eval()
    torch_states = [torch.zeros(s) for s in shapes]
    ort_states = [np.zeros(s, np.float32) for s in shapes]
    input_names = [i.name for i in session.get_inputs()]
    full_predictions, max_wrapper_deg, max_state_diff, max_norm_error = [], 0., 0., 0.
    with torch.inference_mode():
        for t, window in _windows(stream):
            actual = session.run(None, dict(zip(input_names, [window, *ort_states])))
            expected = wrapper(torch.from_numpy(window), *torch_states)
            ort_states, torch_states = actual[1:], expected[1:]
            if not all(np.isfinite(value).all() for value in actual):
                raise AssertionError("Non-finite ONNX prediction or state")
            max_wrapper_deg = max(max_wrapper_deg,
                                  float(geodesic_degrees(actual[0], expected[0].numpy()).max()))
            max_state_diff = max(max_state_diff, max(
                float(np.abs(a - b.numpy()).max()) for a, b in zip(ort_states, torch_states)))
            max_norm_error = max(max_norm_error, float(np.abs(
                np.linalg.norm(actual[0].astype(np.float64), axis=-1) - 1).max()))
            if t >= 399:
                full_predictions.append(actual[0])
        histories = np.stack([stream[t - 399:t + 1] for t in range(399, len(stream))])
        reference = model(torch.from_numpy(histories)).numpy()
    predictions = np.concatenate(full_predictions)
    max_window_deg = float(geodesic_degrees(predictions, reference).max())
    if predictions.shape != (len(stream) - 399, 20, 4):
        raise AssertionError(f"Unexpected forecast shape: {predictions.shape}")
    # Match the existing best.test_best streaming/window tolerance.
    if max(max_window_deg, max_wrapper_deg) >= 1e-3 or max_norm_error >= 1e-5:
        raise AssertionError(f"Export parity failed: window={max_window_deg}, "
                             f"wrapper={max_wrapper_deg}, norm={max_norm_error}")
    return {"passed": True, "synthetic_samples": len(stream), "full_windows": len(predictions),
            "max_ort_vs_window_deg": max_window_deg, "max_ort_vs_streaming_wrapper_deg": max_wrapper_deg,
            "max_state_absolute_difference": max_state_diff, "max_unit_norm_error": max_norm_error,
            "angle_tolerance_deg": 1e-3, "norm_tolerance": 1e-5,
            "includes_quaternion_sign_flips": True}


def _timing(sessions, shapes, stream, rounds=5, calls=200):
    """Alternate order between paired rounds; time one ORT call including state returns."""
    windows = [w for t, w in _windows(stream) if t >= 450]
    values, round_results = {name: [] for name in sessions}, []
    for round_index in range(rounds):
        order = ["baseline", "edited"] if round_index % 2 == 0 else ["edited", "baseline"]
        row = {"round": round_index + 1, "order": order}
        for name in order:
            session = sessions[name]
            names = [item.name for item in session.get_inputs()]
            states = [np.zeros(s, np.float32) for s in shapes[name]]
            # Fill the complete receptive field, then leave additional warm-up calls.
            for _, window in _windows(stream[:450]):
                states = session.run(None, dict(zip(names, [window, *states])))[1:]
            durations = []
            for i in range(calls):
                feeds = dict(zip(names, [windows[i % len(windows)], *states]))
                started = time.perf_counter_ns()
                outputs = session.run(None, feeds)
                elapsed = (time.perf_counter_ns() - started) / 1000
                states = outputs[1:]
                durations.append(elapsed)
            values[name].extend(durations)
            row[name] = {"median_us": float(np.median(durations)),
                         "p95_us": float(np.percentile(durations, 95))}
        round_results.append(row)
    summary = {name: {"calls": len(v), "median_us": float(np.median(v)),
                      "p95_us": float(np.percentile(v, 95))} for name, v in values.items()}
    summary["edited_over_baseline_median"] = summary["edited"]["median_us"] / summary["baseline"]["median_us"]
    summary.update(rounds=round_results, cpu_threads=1, providers=["CPUExecutionProvider"],
                   scope="Local single-thread ORT calls; not organiser hardware. Alternating paired rounds, "
                         "but background load is not controlled; small differences may be timing noise.")
    return summary


def verify_exports(baseline_path, edited_path, output_dir):
    """Export and verify a frozen edit; raise on structural, parity, or preservation failure.

    Returns a JSON-safe report and writes it to export_verification.json. Refuses
    to reuse a populated output directory, or to overwrite either checkpoint.
    """
    sources = {"baseline": Path(baseline_path).resolve(), "edited": Path(edited_path).resolve()}
    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Export directory must be new or empty: {output}")
    if any(path == output or output in path.parents for path in sources.values()):
        raise ValueError("Source checkpoints must be outside the export directory")
    hashes = {name: _sha256(path) for name, path in sources.items()}
    output.mkdir(parents=True, exist_ok=True)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        stream = _motion()
        models, sessions, shapes, signatures, reports = {}, {}, {}, {}, {}
        for name, source in sources.items():
            model, _ = load_checkpoint(source)
            if model.maps != "native":
                raise ValueError("This pilot verifies native-map k8 checkpoints only")
            models[name] = model
            path, state_shapes = export_streaming_onnx(model, output / f"{name}.onnx")
            shapes[name] = state_shapes
            graph = onnx.load(str(path))
            onnx.checker.check_model(graph)
            signatures[name] = _signature(graph.graph)
            audit, macs = audit_operators(path), count_macs(path)
            if any(audit[k] for k in ("other", "avoid", "slow_on_npu", "float64_tensors")):
                raise AssertionError(f"Operator audit failed: {audit}")
            sessions[name] = _session(path)
            reports[name] = {
                "source": str(source), "source_sha256": hashes[name], "onnx": str(path),
                "onnx_sha256": _sha256(path), "onnx_bytes": path.stat().st_size,
                "declared_parameters": sum(p.numel() for p in model.parameters()),
                "initializer_parameters": sum(int(np.prod(t.dims)) for t in graph.graph.initializer),
                "nodes": len(graph.graph.node), "state_shapes": [list(s) for s in state_shapes],
                "operators": audit, "macs_per_call": macs,
                "parity": _parity(model, sessions[name], state_shapes, stream),
            }
        if signatures["baseline"] != signatures["edited"]:
            raise AssertionError("Weight edit changed graph topology or interfaces")
        for key in ("declared_parameters", "initializer_parameters", "nodes", "macs_per_call", "state_shapes"):
            if reports["baseline"][key] != reports["edited"][key]:
                raise AssertionError(f"Weight edit changed {key}")
        result = {"passed": True, "graph_structure_unchanged": True, "models": reports,
                  "latency": _timing(sessions, shapes, stream),
                  "runtime": {"torch": str(torch.__version__), "onnx": onnx.__version__,
                              "onnxruntime": ort.__version__}, "test_data_used": False}
        for name, path in sources.items():
            if _sha256(path) != hashes[name]:
                raise AssertionError(f"Source checkpoint changed: {path}")
        result["source_checkpoint_hashes_unchanged"] = True
        (output / "export_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        torch.set_num_threads(original_threads)
