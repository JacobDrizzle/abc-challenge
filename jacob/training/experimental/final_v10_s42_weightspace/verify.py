"""Independent streaming, dense, symmetry and deployment checks; no test data."""
from copy import deepcopy
import time

import h5py
import numpy as np
import torch

from best.export import StreamingLean
from best.model import mirror
from experimental.v10_endpoint_refit.fit import predict
from experimental.v10_weight_edit.export_check import _session, _timing
from tcnn.model import qmul, rotvec_to_quaternion
from .common import ANGLE_TOLERANCE, DATA, ORIGINAL, log
from .engine import CHUNK, batches, dense_parts, read_segment


def angles(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1)))


def generated_streams():
    n = 900
    t = np.arange(n)
    identity = np.tile([1., 0., 0., 0.], (n, 1))
    constant = np.stack((np.cos(t * .01), np.sin(t * .01), np.zeros(n), np.zeros(n)), 1)
    theta = np.cumsum(.03 * np.sin(t * .025))
    braking = np.stack((np.cos(theta / 2), np.zeros(n), np.sin(theta / 2), np.zeros(n)), 1)
    rng = np.random.default_rng(153)
    extreme = rng.normal(size=(n, 4))
    extreme /= np.linalg.norm(extreme, axis=1, keepdims=True)
    # Include exactly and nearly 180-degree relative rotations, and sign flips.
    extreme[:4] = [[1, 0, 0, 0], [0, 1, 0, 0], [1e-12, 0, 1, 0], [-1, 0, 0, 0]]
    long = np.cumsum(rng.normal(0, .005, (CHUNK + 650, 4)), axis=0) + [2., 0., 0., 0.]
    long /= np.linalg.norm(long, axis=1, keepdims=True)
    return {"stationary": identity, "constant_rotation": constant, "acceleration_braking": braking,
            "sign_flips": braking * np.where(t[:, None] % 3, 1, -1), "extreme_near_pi": extreme,
            "long_chunk_boundary": long}


@torch.inference_mode()
def streaming_parity(model, path, stream, label, check_wrapper=True):
    """Start at sample zero, including padded context, as the benchmark does."""
    model = deepcopy(model).cpu().eval()
    session = _session(path)
    wrapper = StreamingLean(model).eval()
    states = [np.zeros(s, np.float32) for s in wrapper.state_shapes]
    reference_states = [torch.zeros(s) for s in wrapper.state_shapes]
    q = np.asarray(stream, dtype=np.float32)
    predictions, max_angle, max_state, max_norm = [], 0., 0., 0.
    names = [v.name for v in session.get_inputs()]
    for t in range(len(q)):
        history = q[max(0, t - 399):t + 1]
        if len(history) < 400:
            history = np.concatenate((np.repeat(q[:1], 400 - len(history), axis=0), history))
        window = np.ascontiguousarray(history[None])
        actual = session.run(None, dict(zip(names, [window, *states])))
        if not all(np.isfinite(v).all() for v in actual):
            raise ValueError(f"Nonfinite ONNX output: {label}/{t}")
        states = actual[1:]
        max_norm = max(max_norm, float(abs(np.linalg.norm(actual[0].astype(np.float64), axis=-1) - 1).max()))
        if check_wrapper:
            reference = wrapper(torch.from_numpy(window), *reference_states)
            reference_states = reference[1:]
            max_angle = max(max_angle, float(angles(actual[0], reference[0].numpy()).max()))
            for a, b in zip(states, reference_states):
                max_state = max(max_state, float(np.abs(a - b.numpy()).max()))
                if not np.allclose(a, b.numpy(), atol=1e-5, rtol=1e-5):
                    raise ValueError(f"Streaming state mismatch: {label}/{t}; max absolute {max_state}")
        if t >= 399:
            predictions.append(actual[0][0])
    expected = np.stack(predictions)
    maxima = {}
    for device in ("cpu", "cuda"):
        device_model = deepcopy(model).to(device)
        tensor = torch.as_tensor(q, device=device)
        dense = []
        for begin in range(399, len(q), CHUNK):
            end = min(begin + CHUNK, len(q))
            start = max(0, begin - 400)
            anchors = torch.arange(begin, end, device=device)
            dense.append(predict(dense_parts(device_model, tensor[start:end], anchors - start)).cpu().numpy())
        maxima[device] = float(angles(np.concatenate(dense), expected).max())
    if max(max_angle, *maxima.values()) > ANGLE_TOLERANCE or max_norm > 1e-5:
        raise ValueError(f"Prediction parity failed: {label}, wrapper {max_angle}, dense {maxima}")
    return {"label": label, "samples": len(q), "scored_predictions": len(expected),
            "max_wrapper_angle_deg": max_angle, "max_dense_angle_deg": maxima,
            "max_state_absolute_difference": max_state, "state_atol": 1e-5, "state_rtol": 1e-5,
            "max_unit_norm_error": max_norm, "passed": True}


@torch.inference_mode()
def invariances(model):
    model = deepcopy(model).cuda().eval()
    stream = generated_streams()["acceleration_braking"]
    history = torch.as_tensor(np.stack([stream[i:i + 400] for i in (0, 200, 400)]), dtype=torch.float32, device="cuda")
    expected = model(history)
    signs = torch.where(torch.arange(400, device="cuda") % 3 == 0, -1., 1.)[None, :, None]
    yaw = rotvec_to_quaternion(history.new_tensor([0., .7, 0.])).reshape(1, 1, 4)
    reports = {}
    for name, actual, desired in (
            ("quaternion_sign", model(history * signs), expected),
            ("mirror", model(mirror(history)), mirror(expected)),
            ("world_yaw", model(qmul(yaw, history)), qmul(yaw, expected))):
        error = float(angles(actual.cpu().numpy(), desired.cpu().numpy()).max())
        if error > ANGLE_TOLERANCE:
            raise ValueError(f"Invariant failed: {name}={error}")
        reports[name] = error
    return reports


def verify_model(model, path=ORIGINAL, training_rows=(), full=True):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        streams = generated_streams()
        if not full:
            streams.pop("long_chunk_boundary")
        if training_rows:
            with h5py.File(DATA, "r") as handle:
                for i, row in enumerate(training_rows[:3]):
                    if not row["path"].startswith("train/"):
                        raise ValueError("Parity samples must come from training")
                    streams[f"training_segment_{i}"] = read_segment(handle, row)[:1800]
        result = []
        for name, q in streams.items():
            log(f"Verifying {name}: {len(q)} samples")
            # The long sequence still checks all ORT steps/dense predictions;
            # short sequences compare every recurrent state to the wrapper too.
            result.append(streaming_parity(model, path, q, name, check_wrapper=name != "long_chunk_boundary"))
        return {"passed": True, "cases": result, "invariance_max_degrees": invariances(model),
                "prediction_tolerance_degrees": ANGLE_TOLERANCE, "test_data_used": False}
    finally:
        torch.set_num_threads(threads)


def latency_pair(edited_path):
    sessions = {"baseline": _session(ORIGINAL), "edited": _session(edited_path)}
    shapes = {k: [v.shape for v in s.get_inputs()[1:]] for k, s in sessions.items()}
    stream = generated_streams()["acceleration_braking"].astype(np.float32)
    return _timing(sessions, shapes, stream, rounds=5, calls=500)
