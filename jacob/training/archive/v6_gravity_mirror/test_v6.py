"""v6: gravity channels, mirror equivariance, streaming export, and isolation from the frozen v5 code.

python -m unittest experimental.v6_gravity_mirror.test_v6 -v   (from the repository root)
"""
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

from baseline_pipeline import EvaluationConfig, geodesic_degrees, sha256_file
import run_next
from tcnn.damped import DampedVelocity
from tcnn.model import TinyTCNN
from tcnn.onnx_export import audit_operators, count_macs
from tests.test_onnx import random_walk
from tests.test_v5 import smooth_histories

from experimental.v6_gravity_mirror import hooks
from experimental.v6_gravity_mirror.configuration import normalize_config
from experimental.v6_gravity_mirror.export import export_streaming_onnx
from experimental.v6_gravity_mirror.model import GravityMirrorTCNN, gravity_in_head_frame, mirror

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BODY = dict(input_features="body_velocity", num_blocks=7, effective_context=256, damped_prior_tau_ms=100)
OPTIONS = [dict(gravity_features=g, mirror_equivariant=m) for g in (True, False) for m in (True, False)]


def trained_like(**kwargs):
    torch.manual_seed(0)
    model = GravityMirrorTCNN(**{"dropout": 0, **kwargs})
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, .05)  # Non-zero head, so the network path is tested too.
    return model.eval()


def predict(model, history):
    with torch.inference_mode():
        return model(torch.from_numpy(np.ascontiguousarray(history, dtype=np.float32))).numpy()


def rotate_world(rotation, q):
    """rotation * q for scalar-first quaternion arrays [..., 4]."""
    turned = rotation * Rotation.from_quat(q.reshape(-1, 4)[:, [1, 2, 3, 0]])
    return turned.as_quat()[:, [3, 0, 1, 2]].reshape(q.shape).astype(np.float32)


def suite_configs():
    return run_next.experiment_configs(json.loads((HERE/"configs/v6_run.json").read_text()),
                                       json.loads((HERE/"configs/tcnn_v6.json").read_text()))


class ConfigTests(unittest.TestCase):
    def test_suite_is_valid_and_only_the_v6_switches_differ_from_v5(self):
        with hooks.installed():
            configs = suite_configs()
        v5 = json.loads((ROOT/"configs/tcnn_v5.json").read_text())
        self.assertEqual(list(configs), ["control", "no_gravity", "no_mirror", "plain", "narrow"])
        self.assertEqual({k: v for k, v in configs["plain"].items() if k not in ("gravity_features", "mirror_equivariant")},
                         run_next.normalize_config(v5))
        self.assertEqual((configs["control"]["gravity_features"], configs["control"]["mirror_equivariant"]), (True, True))
        self.assertEqual((configs["plain"]["gravity_features"], configs["plain"]["mirror_equivariant"]), (False, False))

    def test_v6_keys_are_checked_and_the_v5_schema_still_rejects_them(self):
        base = json.loads((HERE/"configs/tcnn_v6.json").read_text())
        self.assertEqual(normalize_config(normalize_config(base)), normalize_config(base))
        for bad in ({"gravity_features": 1}, {"mirror_equivariant": "yes"}, {"input_features": "anchor"},
                    {"channels": 40}, {"colour": True}):
            with self.assertRaises(ValueError):
                normalize_config({**base, **bad})
        with self.assertRaises(ValueError):
            run_next.normalize_config(base)  # outside hooks.installed() the frozen schema is in force

    def test_hooks_restore_every_name(self):
        import benchmark_latency, run_desktop, tcnn_model_adapter, train_tcnn
        modules = (benchmark_latency, run_desktop, run_next, tcnn_model_adapter, train_tcnn)
        before = [dict(vars(m)) for m in modules]
        with hooks.installed():
            self.assertIs(tcnn_model_adapter.TinyTCNN, GravityMirrorTCNN)
            self.assertEqual(run_desktop.SOURCE_FILES[-len(hooks.V6_FILES):], hooks.V6_FILES)
        for module, names in zip(modules, before):
            self.assertEqual({k: v for k, v in vars(module).items() if k in names}, names)
        for name in hooks.V6_FILES:
            self.assertTrue((ROOT/name).exists(), name)

    def test_frozen_v5_code_still_matches_its_suite_manifest(self):
        manifest = ROOT/"outputs/v5_group_a_v2/suite.json"
        if not manifest.exists():
            self.skipTest("No v5 suite on this machine")
        for name, digest in json.loads(manifest.read_text())["code_sha256"].items():
            self.assertEqual(sha256_file(ROOT/name), digest, name)


class ModelTests(unittest.TestCase):
    def test_parameter_counts(self):
        with hooks.installed():
            configs = suite_configs()
        expected = {"control": 26012, "no_gravity": 25940, "no_mirror": 52044, "plain": 51900, "narrow": 11972}
        for name, config in configs.items():
            model = GravityMirrorTCNN(channels=config["channels"], gravity_features=config["gravity_features"],
                                      mirror_equivariant=config["mirror_equivariant"], **BODY)
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected[name], name)

    def test_plain_model_is_the_v5_model(self):
        history = torch.from_numpy(random_walk(4, 400, 5))
        torch.manual_seed(7)
        v5 = TinyTCNN(channels=48, **BODY).eval()
        torch.manual_seed(7)
        v6 = GravityMirrorTCNN(channels=48, **BODY).eval()
        self.assertEqual(v5.state_dict().keys(), v6.state_dict().keys())
        for key, value in v5.state_dict().items():
            self.assertTrue(torch.equal(value, v6.state_dict()[key]), key)
        with torch.no_grad():
            for p in (*v5.parameters(), *v6.parameters()):
                p.copy_(torch.sin(torch.arange(p.numel()).reshape(p.shape) * .7) * .05)
            self.assertTrue(torch.equal(v5(history), v6(history)))
        self.assertEqual(TinyTCNN(**{k: v for k, v in v6.specification().items()
                                     if k not in ("gravity_features", "mirror_equivariant")}).specification(),
                         v5.specification())

    def test_untrained_models_equal_the_damped_baseline(self):
        history = smooth_histories(8)
        expected = DampedVelocity(EvaluationConfig(), 1, 100).predict_batch(history)
        for options in OPTIONS:
            with self.subTest(**options):
                prediction = predict(GravityMirrorTCNN(channels=48, dropout=0, **BODY, **options).eval(), history)
                self.assertLess(geodesic_degrees(prediction, expected).max(), 1e-3)

    def test_gravity_is_world_up_in_the_head_frame_and_sign_invariant(self):
        rotations = Rotation.random(500, random_state=2)
        q = torch.tensor(rotations.as_quat()[:, [3, 0, 1, 2]])
        expected = rotations.inv().apply([0, 1, 0])
        self.assertLess((gravity_in_head_frame(q).numpy() - expected).__abs__().max(), 1e-12)
        self.assertTrue(torch.equal(gravity_in_head_frame(q), gravity_in_head_frame(-q)))

    def test_mirror_equivariance_is_exact(self):
        history = smooth_histories(6)
        for gravity in (True, False):
            with self.subTest(gravity=gravity):
                model = trained_like(channels=16, gravity_features=gravity, mirror_equivariant=True, **BODY)
                mirrored = predict(model, mirror(torch.from_numpy(history)).numpy())
                expected = mirror(torch.from_numpy(predict(model, history))).numpy()
                self.assertLess(geodesic_degrees(mirrored, expected).max(), 1e-3)
        # Without the constraint, a trained-like model is far from equivariant.
        model = trained_like(channels=16, gravity_features=True, **BODY)
        mirrored = predict(model, mirror(torch.from_numpy(history)).numpy())
        self.assertGreater(geodesic_degrees(mirrored, mirror(torch.from_numpy(predict(model, history))).numpy()).max(), .1)

    def test_yaw_and_sign_invariant_but_tilt_sensitive(self):
        history = smooth_histories(4)
        flips = np.where(np.random.default_rng(1).random(history.shape[:2] + (1,)) < .3, -1, 1).astype(np.float32)
        yaw, tilt = Rotation.from_rotvec([0, 2.1, 0]), Rotation.from_rotvec([.5, 0, .3])
        for options in OPTIONS:
            model = trained_like(channels=16, **BODY, **options)
            base = predict(model, history)
            for name, rotation in (("yaw", yaw), ("tilt", tilt)):
                with self.subTest(turn=name, **options):
                    got = predict(model, rotate_world(rotation, history) * flips)
                    error = geodesic_degrees(got, rotate_world(rotation, base)).max()
                    if name == "tilt" and options["gravity_features"]:
                        self.assertGreater(error, .1)  # absolute tilt is now an input
                    else:
                        self.assertLess(error, 1e-3)


@unittest.skipIf(ort is None, "onnxruntime is not installed")
class StreamingTests(unittest.TestCase):
    def test_streaming_graph_equals_stateless_model_once_the_window_is_full(self):
        stream = random_walk(1, 650, 4)[0]
        for structure in (dict(num_blocks=7, effective_context=256), dict(num_blocks=5, effective_context=64)):
            for options in OPTIONS:
                model = trained_like(channels=8, input_features="body_velocity", damped_prior_tau_ms=100,
                                     **structure, **options)
                for trig_free in (True, False):
                    with self.subTest(trig_free=trig_free, **structure, **options), tempfile.TemporaryDirectory() as folder:
                        self.check_stream(model, stream, Path(folder)/"s.onnx", trig_free)

    def check_stream(self, model, stream, path, trig_free):
        path, shapes = export_streaming_onnx(model, path, trig_free=trig_free)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        names = [i.name for i in session.get_inputs()]
        self.assertEqual(names, ["context", *(f"state_in_{i}" for i in range(len(shapes)))])
        states, predictions = [np.zeros(s, np.float32) for s in shapes], {}
        for t in range(1, len(stream)):
            window = stream[max(0, t - 399):t + 1]
            window = np.concatenate((np.repeat(window[:1], 400 - len(window), 0), window))[None]
            outputs = session.run(None, {"context": window, **dict(zip(names[1:], states))})
            predictions[t], states = outputs[0], outputs[1:]
        steps = range(399, len(stream))
        reference = predict(model, np.stack([stream[t - 399:t + 1] for t in steps]))
        got = np.concatenate([predictions[t] for t in steps])
        self.assertEqual(got.shape[1:], (20, 4))
        self.assertLess(geodesic_degrees(got, reference).max(), 1e-3)
        s, k, f, n = model.streams, model.stream_channels, 7 if model.gravity_features else 4, model.num_blocks
        # projection, blocks, head, quaternion MatMuls (delta 64+16, output 64+20*16), gravity (16+64).
        expected = s*k*f + n*3*(s*k)**2 + 60*s*k + 464 + (80 if model.gravity_features else 0)
        self.assertEqual(count_macs(path)["total"], expected)
        self.assertEqual([tuple(s) for s in shapes], [(s, k, 2**(i+1)) for i in range(n)])
        audit = audit_operators(path)
        self.assertEqual(audit["avoid"], [])
        self.assertEqual(audit["float64_tensors"], [])
        if trig_free:
            self.assertEqual(audit["slow_on_npu"], [])
        if model.mirror_equivariant:
            # The file stores the shared block weights once: [K, 2K, 3].
            import onnx
            from onnx import numpy_helper
            weights = [numpy_helper.to_array(t) for t in onnx.load(str(path)).graph.initializer]
            weights += [numpy_helper.to_array(node.attribute[0].t) for node in onnx.load(str(path)).graph.node
                        if node.op_type == "Constant" and node.attribute[0].name == "value"]
            self.assertEqual(sum(w.shape == (k, 2*k, 3) for w in weights), n)
            self.assertFalse(any(w.shape == (2*k, 2*k, 3) for w in weights))


if __name__ == "__main__":
    unittest.main()
