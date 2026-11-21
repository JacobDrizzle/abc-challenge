"""v5: streamable body-velocity TCNN, compact and trig-free ONNX graphs, and the streaming export."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

from baseline_pipeline import EvaluationConfig, geodesic_degrees
from run_next import experiment_configs
from tcnn.configuration import normalize_config
from tcnn.damped import DampedVelocity
from tcnn.model import TinyTCNN, receptive_field
from tcnn.onnx_export import (ExportableTCNN, audit_operators, count_macs, export_onnx, export_streaming_onnx,
                              _compact_exp, _compact_rotvec, _exp_quaternion, _rotvec)
from tests.test_onnx import random_walk, trained_like

ROOT = Path(__file__).resolve().parents[1]
BODY = dict(input_features="body_velocity", num_blocks=7, effective_context=256)


def smooth_histories(count, length=400, seed=3):
    rng = np.random.default_rng(seed)
    histories = []
    for b in range(count):
        rate = rng.normal(0, 1.5, 3)
        steps = Rotation.from_rotvec(np.cumsum(np.tile(rate * .01, (length, 1)) + rng.normal(0, .002, (length, 3)), 0))
        histories.append((steps * Rotation.random(random_state=b)).as_quat()[:, [3, 0, 1, 2]])
    return np.array(histories, dtype=np.float32)


class ConfigTests(unittest.TestCase):
    def test_v5_suite_is_valid_and_streamable(self):
        configs = experiment_configs(json.loads((ROOT/"configs/v5_run.json").read_text()),
                                     json.loads((ROOT/"configs/tcnn_v5.json").read_text()))
        self.assertIn("control", configs)
        for config in configs.values():
            self.assertEqual(config["input_features"], "body_velocity")
            self.assertLessEqual(receptive_field(config["num_blocks"]) + 1, config["effective_context"])

    def test_new_keys_default_to_v4_behaviour_and_are_checked(self):
        base = json.loads((ROOT/"configs/tcnn_v3.json").read_text())
        self.assertEqual((normalize_config(base)["input_features"], normalize_config(base)["num_blocks"]), ("anchor", 8))
        body = {**base, "velocity_features": False, "input_features": "body_velocity", "num_blocks": 7,
                "effective_context": 256}
        normalize_config(body)
        for bad in ({"input_features": "world"}, {"num_blocks": 9}, {"num_blocks": 0},
                    {"effective_context": 255},  # receptive field 255 needs 256 samples
                    {"velocity_features": True}):
            with self.assertRaises(ValueError):
                normalize_config({**body, **bad})

    def test_old_checkpoint_specs_still_load(self):
        old_spec = dict(context=400, horizon=20, channels=48, dropout=.1, effective_context=400,
                        velocity_features=True, damped_prior_tau_ms=100)
        model = TinyTCNN(**old_spec)
        self.assertEqual((model.input_features, model.num_blocks, sum(p.numel() for p in model.parameters())),
                         ("anchor", 8, 59004))


class BodyModelTests(unittest.TestCase):
    def test_untrained_body_model_equals_damped_baseline(self):
        history = smooth_histories(16)
        model = TinyTCNN(channels=48, dropout=0, damped_prior_tau_ms=100, **BODY).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 51900)
        with torch.no_grad():
            prediction = model(torch.from_numpy(history)).numpy()
        expected = DampedVelocity(EvaluationConfig(), 1, 100).predict_batch(history)
        self.assertLess(geodesic_degrees(prediction, expected).max(), 1e-3)

    def test_body_features_ignore_the_world_frame_and_quaternion_signs(self):
        history = smooth_histories(4)
        model = trained_like(channels=8, damped_prior_tau_ms=100, **BODY)
        turned = (Rotation.random(random_state=9) * Rotation.from_quat(history.reshape(-1, 4)[:, [1, 2, 3, 0]]))
        turned = turned.as_quat()[:, [3, 0, 1, 2]].reshape(history.shape).astype(np.float32)
        turned *= np.where(np.random.default_rng(1).random(turned.shape[:2] + (1,)) < .3, -1, 1)
        with torch.no_grad():
            a = model(torch.from_numpy(history)).numpy()
            b = model(torch.from_numpy(turned)).numpy()
        # Same head motion seen from a rotated world: the forecast rotates with it.
        rotate = Rotation.random(random_state=9)
        a_turned = (rotate * Rotation.from_quat(a.reshape(-1, 4)[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
        self.assertLess(geodesic_degrees(a_turned.reshape(a.shape), b).max(), 1e-3)


class PolynomialTests(unittest.TestCase):
    def test_trig_free_maths_matches_trigonometric_maths(self):
        rng = np.random.default_rng(0)
        # Rotations from 0 to 180 degrees, unnormalised, both quaternion signs.
        axis = rng.normal(size=(4000, 3))
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        angle = np.concatenate((rng.uniform(0, 1e-3, 1000), rng.uniform(0, np.pi, 3000)))[:, None]
        q = np.concatenate((np.cos(angle / 2), np.sin(angle / 2) * axis), 1) * rng.uniform(.5, 2, (4000, 1))
        q *= np.where(rng.random((4000, 1)) < .5, -1, 1)
        q = torch.tensor(q, dtype=torch.float32)
        self.assertLess((_compact_rotvec(q) - _rotvec(q)).abs().max().item(), 2e-6)
        v = torch.tensor(rng.normal(size=(4000, 3)), dtype=torch.float32)
        v *= torch.rand(4000, 1) * 2 * np.pi / v.norm(dim=1, keepdim=True)  # rotations up to 360 degrees
        exp = _compact_exp(v)
        exp = exp / exp.norm(dim=-1, keepdim=True)
        self.assertLess(geodesic_degrees(exp.numpy(), _exp_quaternion(v).numpy()).max(), 1e-3)


@unittest.skipIf(ort is None, "onnxruntime is not installed")
class ExportTests(unittest.TestCase):
    OPTIONS = [dict(), dict(strided=True), dict(strided=True, compact=True),
               dict(strided=True, compact=True, trig_free=True)]
    CASES = [dict(channels=8, velocity_features=True, damped_prior_tau_ms=100),
             dict(channels=8, velocity_features=False, damped_prior_tau_ms=100),
             dict(channels=8, damped_prior_tau_ms=100, **BODY),
             dict(channels=8, input_features="body_velocity", num_blocks=6, effective_context=150)]

    def test_every_export_option_matches_pytorch(self):
        histories = random_walk(8, 400, 2)
        for case in self.CASES:
            model = trained_like(**case)
            with torch.inference_mode():
                reference = model(torch.from_numpy(histories)).numpy()
            for options in self.OPTIONS:
                with self.subTest(case=case, **options), tempfile.TemporaryDirectory() as folder:
                    wrapper = ExportableTCNN(model, **options).eval()
                    with torch.inference_mode():
                        wrapped = np.concatenate([wrapper(torch.from_numpy(h[None])).numpy() for h in histories])
                    path = export_onnx(model, Path(folder) / "m.onnx", **options)
                    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                    exported = np.concatenate([session.run(None, {"context": h[None]})[0] for h in histories])
                    self.assertLess(geodesic_degrees(wrapped, reference).max(), 1e-3)
                    self.assertLess(geodesic_degrees(exported, reference).max(), 1e-3)
                    audit = audit_operators(path)
                    self.assertEqual(audit["avoid"], [])
                    if options.get("trig_free"):
                        self.assertEqual(audit["slow_on_npu"], [])
                    if options.get("compact"):
                        self.assertLessEqual(audit["counts"].get("Concat", 0), 4)

    def test_streaming_graph_equals_stateless_model_once_the_window_is_full(self):
        stream = random_walk(1, 650, 4)[0]
        for case in (dict(channels=8, damped_prior_tau_ms=100, **BODY),
                     dict(channels=8, input_features="body_velocity", num_blocks=5, effective_context=64)):
            model = trained_like(**case)
            for compact, trig_free in ((False, False), (True, True)):
                with self.subTest(case=case, compact=compact), tempfile.TemporaryDirectory() as folder:
                    path, shapes = export_streaming_onnx(model, Path(folder) / "s.onnx", compact, trig_free)
                    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                    names = [i.name for i in session.get_inputs()]
                    self.assertEqual(names, ["context", *(f"state_in_{i}" for i in range(len(shapes)))])
                    self.assertEqual([o.name for o in session.get_outputs()],
                                     ["prediction", *(f"state_out_{i}" for i in range(len(shapes)))])
                    states, predictions = [np.zeros(s, np.float32) for s in shapes], {}
                    for t in range(1, len(stream)):
                        window = stream[max(0, t - 399):t + 1]
                        window = np.concatenate((np.repeat(window[:1], 400 - len(window), 0), window))[None]
                        outputs = session.run(None, {"context": window, **dict(zip(names[1:], states))})
                        predictions[t], states = outputs[0], outputs[1:]
                    steps = range(399, len(stream))
                    with torch.inference_mode():
                        reference = model(torch.from_numpy(np.stack([stream[t - 399:t + 1] for t in steps]))).numpy()
                    got = np.concatenate([predictions[t] for t in steps])
                    self.assertLess(geodesic_degrees(got, reference).max(), 1e-3)
                    c, n = case["channels"], case["num_blocks"]
                    # projection, blocks, head; compact adds the quaternion MatMuls (delta 64+16, output 64+20*16).
                    expected = 4*c + n*c*c*3 + c*60 + (64 + 16 + 64 + 20*16 if compact else 0)
                    self.assertEqual(count_macs(path)["total"], expected)
                    if trig_free:
                        self.assertEqual(audit_operators(path)["slow_on_npu"], [])

    def test_anchor_models_refuse_to_stream(self):
        with self.assertRaises(ValueError):
            export_streaming_onnx(trained_like(channels=8, velocity_features=True), Path("unused.onnx"))


if __name__ == "__main__":
    unittest.main()
