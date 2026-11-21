"""Geometry, causality and experiment selection checks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from baseline_pipeline import ConstantVelocity, EvaluationConfig, geodesic_degrees, sha256_file
from evaluate_tcnn import evaluate_checkpoint
from desktop_next import display_progress
from run_next import experiment_configs, run_suite
from smoke_tcnn import make_fixture
from tcnn.configuration import normalize_config
from tcnn.damped import DampedVelocity
from tcnn.data import WindowDataset, check_corpus
from tcnn.model import TinyTCNN, angular_velocity_features, qmul, relative_history, rotation_loss, rotvec_to_quaternion
from tcnn.runtime import configure_runtime
from tune_damped_baseline import tune

ROOT = Path(__file__).resolve().parents[1]


class MotionTests(unittest.TestCase):
    def test_damping_integrates_velocity_and_undamped_matches_existing(self):
        axis = np.array([.2, -.3, .7]); axis /= np.linalg.norm(axis)
        initial = Rotation.from_euler("xyz", [20, -35, 12], degrees=True)
        q = (Rotation.from_rotvec(np.arange(420)[:, None]*.01*axis)*initial).as_quat()[:, [3, 0, 1, 2]]
        history = q[:400][None].astype(np.float32)
        config = EvaluationConfig()
        for lag in (1, 3, 5, 10):
            a = DampedVelocity(config, lag, None).predict_batch(history)
            b = ConstantVelocity(config, lag).predict_batch(history)
            np.testing.assert_allclose(geodesic_degrees(a, b), 0, atol=5e-6)
        model = DampedVelocity(config, 1, 100)
        pred = model.predict_batch(history)
        displacement = geodesic_degrees(pred, np.repeat(history[:, -1:], 20, axis=1))
        expected = np.rad2deg(.01*10*(-np.expm1(-np.arange(1, 21)/10)))
        np.testing.assert_allclose(displacement[0], expected, atol=1e-4)
        np.testing.assert_allclose(model.predict(history[0]), pred[0], atol=1e-12)

    def test_velocity_features_recover_known_rotation_rate_and_sign_invariance(self):
        angle = torch.arange(400, dtype=torch.float64)*.01
        v = torch.zeros(1, 400, 3, dtype=torch.float64); v[0, :, 2] = angle
        q = rotvec_to_quaternion(v)
        relative, _ = relative_history(q)
        velocity = angular_velocity_features(relative)
        torch.testing.assert_close(velocity[:, 1:, 2], torch.ones(1, 399, dtype=torch.float64))
        torch.testing.assert_close(velocity[:, 1:, :2], torch.zeros(1, 399, 2, dtype=torch.float64))
        q[:, ::3] *= -1
        actual, _ = relative_history(q)
        torch.testing.assert_close(angular_velocity_features(actual), velocity)

    def test_short_context_ignores_older_history_even_with_velocity(self):
        torch.manual_seed(3)
        model = TinyTCNN(effective_context=50, velocity_features=True, dropout=0).eval()
        torch.nn.init.normal_(model.head.weight, std=.02)
        q = rotvec_to_quaternion(torch.randn(2, 400, 3)*.1)
        other = q.clone(); other[:, :350] = rotvec_to_quaternion(torch.randn(2, 350, 3))
        torch.testing.assert_close(model(q), model(other), rtol=0, atol=0)
        world = rotvec_to_quaternion(torch.tensor([.4, -.2, .3]))
        np.testing.assert_allclose(geodesic_degrees(model(qmul(world, q)).detach(), qmul(world, model(q)).detach()), 0, atol=1e-4)

    def test_angular_loss_is_geodesic_sign_invariant_and_has_finite_gradients(self):
        angles = torch.tensor([0., .1, 1., np.pi], dtype=torch.float64)
        rv = torch.zeros(4, 20, 3, dtype=torch.float64); rv[:, :, 0] = angles[:, None]
        target = rotvec_to_quaternion(rv)
        pred = torch.zeros_like(target); pred[..., 0] = 1; pred.requires_grad_()
        loss = rotation_loss(pred, target, "angular", .75)
        torch.testing.assert_close(loss, angles.mean())
        torch.testing.assert_close(loss, rotation_loss(pred, -target, "angular", .75))
        loss.backward(); self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad[1, :, 1:].abs().sum()), 0)
        identity = pred.detach().clone().requires_grad_()
        rotation_loss(identity, identity.detach(), "angular").backward()
        self.assertTrue(torch.isfinite(identity.grad).all())

    def test_new_variants_have_finite_training_gradients(self):
        for context in (50, 100, 200, 400):
            model = TinyTCNN(effective_context=context, velocity_features=True)
            q = rotvec_to_quaternion(torch.randn(2, 400, 3)*.03)
            target = rotvec_to_quaternion(torch.randn(2, 20, 3)*.03)
            loss = rotation_loss(model(q), target, "angular", .75)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertEqual(tuple(model(q).shape), (2, 20, 4))


class ExperimentTests(unittest.TestCase):
    def test_configuration_contract_and_parameter_budget(self):
        base = json.loads((ROOT/"configs/tcnn_3080.json").read_text())
        suite = json.loads((ROOT/"configs/next_run.json").read_text())
        configs = experiment_configs(suite, base)
        self.assertEqual(len(configs), 8)
        for config in configs.values():
            self.assertEqual((config["context"], config["horizon"]), (400, 20))
        for kwargs in ({"channels": 64}, {"effective_context": 401}, {"loss_kind": "unknown"}, {"velocity_features": "true"}):
            with self.assertRaises(ValueError): normalize_config({**base, **kwargs})
        self.assertEqual(sum(p.numel() for p in TinyTCNN(channels=48).parameters()), 58860)

    def test_sampling_is_repeatable_without_shifted_seed_streams(self):
        with tempfile.TemporaryDirectory() as temp:
            data, split = make_fixture(Path(temp)/"data", samples=1000)
            audit = check_corpus(data, split, synthetic=True, minimum_users=3, minimum_recordings=3)
            rows = [r for r in audit["segments"] if r["split"] == "train"]
            a, b = WindowDataset(data, rows, 100, 42), WindowDataset(data, rows, 100, 43)
            try:
                matches = sum(torch.equal(a[i+1][0], b[i][0]) for i in range(40))
                self.assertLess(matches, 5)
                torch.testing.assert_close(a[8][0], a[8][0], atol=0, rtol=0)
            finally: a.close(); b.close()

    def test_runtime_policy_is_explicit(self):
        state = configure_runtime()
        self.assertTrue(state["deterministic_algorithms"])
        self.assertFalse(state["cudnn_allow_tf32"])
        self.assertFalse(state["matmul_allow_tf32"])

    def test_suite_selects_validation_only_and_can_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); data, split = make_fixture(root/"data")
            base = json.loads((ROOT/"configs/tcnn_3080.json").read_text())
            base.update(epochs=1, steps_per_epoch=1, batch_size=2, workers=0)
            suite = {"version": 1, "seeds": [42, 43], "variants": {"control": {}, "velocity": {"velocity_features": True}},
                     "baseline_lags": [1], "baseline_tau_ms": [100, None]}
            with display_progress(experiment_configs(suite, base)):
                first = run_suite(data, root/"out", suite, base, device="cpu", synthetic=True)
            again = run_suite(data, root/"out", suite, base, device="cpu", synthetic=True)
            self.assertEqual(first, again)
            self.assertEqual(first["evaluated_seeds"], [42, 43])
            self.assertFalse((root/"out/test").exists())
            for path in (root/"out").rglob("results.json"):
                report = json.loads(path.read_text())
                self.assertEqual(report["split"], "validation")
                self.assertEqual(report["windows"], 81)
            with self.assertRaisesRegex(ValueError, "Suite changed"):
                run_suite(data, root/"out", suite, dict(base, epochs=2), device="cpu", synthetic=True)
            model, lock = root/"out/selected_model.pt", root/"out/selection.json"
            with self.assertRaisesRegex(ValueError, "requires the frozen"):
                evaluate_checkpoint(model, data, split, root/"test", split="test", synthetic=True)
            kwargs = dict(split="test", selection_path=lock, synthetic=True, batch_size=8, latency_steps=2,
                          damped_selection=root/"out/damped_baseline/selection.json")
            report = evaluate_checkpoint(model, data, split, root/"test", **kwargs)
            receipt = json.loads((root/"test/test_receipt.json").read_text())
            self.assertTrue(receipt["test_evaluated"])
            self.assertTrue(receipt["synthetic"])
            self.assertEqual(len(report["models"]), 5)
            result_hash = sha256_file(root/"test/results.json")
            (root/"test/test_receipt.json").unlink()  # Recover a crash after publishing results.
            evaluate_checkpoint(model, data, split, root/"test", **kwargs)
            self.assertEqual(result_hash, sha256_file(root/"test/results.json"))
            self.assertEqual(receipt, json.loads((root/"test/test_receipt.json").read_text()))
            with self.assertRaisesRegex(ValueError, "Existing evaluation differs"):
                evaluate_checkpoint(model, data, split, root/"test", **{**kwargs, "batch_size": 9})


if __name__ == "__main__":
    unittest.main()
