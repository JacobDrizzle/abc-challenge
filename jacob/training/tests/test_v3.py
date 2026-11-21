"""v3 training upgrades: damped prior, GPU-resident sampling, EMA, metrics and figures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from baseline_pipeline import EvaluationConfig, geodesic_degrees, window_batches
from evaluate_tcnn import evaluate_checkpoint
from run_next import experiment_configs, run_suite
from smoke_tcnn import make_fixture
from tcnn.configuration import normalize_config
from tcnn.damped import DampedVelocity
from tcnn.data import DeviceWindows, check_corpus
from tcnn.model import TinyTCNN
from tcnn.plots import user_bootstrap
import train_tcnn

ROOT = Path(__file__).resolve().parents[1]


def base_config(**overrides):
    config = json.loads((ROOT/"configs/tcnn_v3.json").read_text())
    config.update(batch_size=4, steps_per_epoch=3, epochs=2, validation_stride=20, **overrides)
    return config


class PriorTests(unittest.TestCase):
    def test_v3_configs_are_valid_and_new_keys_are_checked(self):
        configs = experiment_configs(json.loads((ROOT/"configs/v3_run.json").read_text()),
                                     json.loads((ROOT/"configs/tcnn_v3.json").read_text()))
        self.assertEqual(set(configs), {"control", "damped_prior", "damped_prior_48"})
        self.assertIsNone(configs["control"]["damped_prior_tau_ms"])
        base = json.loads((ROOT/"configs/tcnn_3080.json").read_text())
        self.assertEqual(normalize_config(base)["ema_decay"], 0)
        for bad in ({"damped_prior_tau_ms": 0}, {"damped_prior_tau_ms": "100"}, {"ema_decay": 1},
                    {"ema_decay": -.1}, {"data_on_device": 1}):
            with self.assertRaises(ValueError):
                normalize_config({**base, **bad})

    def test_untrained_damped_prior_equals_damped_baseline_without_new_parameters(self):
        rng = np.random.default_rng(3)
        histories = []
        for b in range(16):
            rate = rng.normal(0, 1.5, 3)
            steps = Rotation.from_rotvec(np.cumsum(np.tile(rate * .01, (400, 1)) + rng.normal(0, .002, (400, 3)), 0))
            histories.append((steps * Rotation.random(random_state=b)).as_quat()[:, [3, 0, 1, 2]])
        history = np.array(histories, dtype=np.float32)
        for velocity, channels, parameters in ((False, 32, 26972), (True, 32, 27068), (True, 48, 59004)):
            model = TinyTCNN(channels=channels, dropout=0, velocity_features=velocity, damped_prior_tau_ms=100).eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()), parameters)
            with torch.no_grad():
                prediction = model(torch.from_numpy(history)).numpy()
            expected = DampedVelocity(EvaluationConfig(), 1, 100).predict_batch(history)
            self.assertLess(geodesic_degrees(prediction, expected).max(), 1e-3)
        spec = TinyTCNN(damped_prior_tau_ms=100).specification()
        self.assertEqual(TinyTCNN(**spec).damped_prior_tau_ms, 100)


class DeviceWindowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.data, self.split = make_fixture(self.directory/"fixture", samples=900)
        audit = check_corpus(self.data, self.split, minimum_users=3, minimum_recordings=3, synthetic=True)
        self.rows = audit["segments"]

    def tearDown(self):
        self.temp.cleanup()

    def test_batches_are_real_future_windows_and_reproducible(self):
        rows = [r for r in self.rows if r["split"] == "train"]
        windows = DeviceWindows(self.data, rows, "cpu")
        with h5py.File(self.data) as h5:
            full = h5[rows[0]["path"]]["quaternion_wxyz"][:].astype(np.float64)
        full /= np.linalg.norm(full, axis=1, keepdims=True)
        x, y = next(windows.batches(42, 0, 1, 16))
        for history, target in zip(x.numpy(), y.numpy()):
            start = int(np.flatnonzero(np.all(np.isclose(full, history[0], atol=1e-7), axis=1))[0])
            np.testing.assert_allclose(history, full[start:start+400], atol=1e-7)
            np.testing.assert_allclose(target, full[start+400:start+420], atol=1e-7)
        again = next(windows.batches(42, 0, 1, 16))[0]
        torch.testing.assert_close(x, again, rtol=0, atol=0)
        self.assertFalse(torch.equal(x, next(windows.batches(42, 1, 1, 16))[0]))
        self.assertFalse(torch.equal(x, next(windows.batches(43, 0, 1, 16))[0]))

    def test_users_are_balanced_despite_unequal_lengths(self):
        path = self.directory/"unequal.h5"
        rows = []
        with h5py.File(path, "w") as h5:
            for name, n, identity in [("short", 500, [1, 0, 0, 0]), ("long", 20000, [0, 1, 0, 0])]:
                key = f"{name}/recording/segment_000"
                h5.create_dataset(key+"/quaternion_wxyz", data=np.tile(np.float32(identity), (n, 1)))
                rows.append({"user": name, "key": key, "path": key, "samples": n})
        x = torch.cat([b[0] for b in DeviceWindows(path, rows, "cpu").batches(42, 0, 10, 100)])
        share = float((x[:, 0, 0] == 1).double().mean())
        self.assertGreater(share, .4)
        self.assertLess(share, .6)

    def test_validation_windows_match_shared_evaluator_anchors(self):
        rows = [r for r in self.rows if r["split"] == "validation"]
        ours = DeviceWindows(self.data, rows, "cpu").windows(20, 7)
        x = torch.cat([h for h, _ in ours]).numpy()
        with h5py.File(self.data) as h5:
            reference = np.concatenate([h for _, h, _ in window_batches(h5[rows[0]["path"]], EvaluationConfig(stride=20))])
        np.testing.assert_allclose(x, reference, atol=1e-7)


class TrainingTests(unittest.TestCase):
    def test_device_ema_training_records_metrics_and_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, split = make_fixture(root/"fixture")
            config = base_config(damped_prior_tau_ms=100)
            train_tcnn.train(data, split, root/"full", config, 42, "cpu", synthetic=True)
            history = json.loads((root/"full/history.json").read_text())
            for key in ("training_endpoint_deg", "best_epoch", "validation_vs_damped_deg", "training_windows_per_second"):
                self.assertIn(key, history[-1])
            reference = json.loads((root/"full/validation_reference.json").read_text())
            self.assertLess(reference["damped_velocity"]["mean_endpoint_deg"], reference["hold_last"]["mean_endpoint_deg"])
            self.assertTrue((root/"full/training_curves.png").exists())
            state = torch.load(root/"full/last.pt", weights_only=True)
            self.assertIn("train_model_state", state)
            self.assertFalse(all(torch.equal(state["model_state"][k], state["train_model_state"][k]) for k in state["model_state"]))
            original = train_tcnn.atomic_checkpoint

            def stop(path, saved):
                original(path, saved)
                if Path(path).name == "last.pt" and saved["epoch"] == 1:
                    raise InterruptedError
            with patch.object(train_tcnn, "atomic_checkpoint", side_effect=stop):
                with self.assertRaises(InterruptedError):
                    train_tcnn.train(data, split, root/"resumed", config, 42, "cpu", synthetic=True)
            train_tcnn.train(data, split, root/"resumed", config, 42, "cpu", resume=True, synthetic=True)
            resumed = torch.load(root/"resumed/last.pt", weights_only=True)
            for part in ("model_state", "train_model_state"):
                for name in state[part]:
                    torch.testing.assert_close(state[part][name], resumed[part][name], rtol=0, atol=0)

    def test_device_validation_matches_shared_evaluator(self):
        with tempfile.TemporaryDirectory() as temp:
            data, split = make_fixture(Path(temp)/"fixture", samples=900)
            rows = [r for r in check_corpus(data, split, minimum_users=3, minimum_recordings=3, synthetic=True)["segments"]
                    if r["split"] == "validation"]
            score = train_tcnn.score_windows(train_tcnn.reference_predictors(20),
                                             DeviceWindows(data, rows, "cpu").windows(1, 64), 20)
            damped = DampedVelocity(EvaluationConfig(), 1, 100)
            with h5py.File(data) as h5:
                errors = np.concatenate([geodesic_degrees(damped.predict_batch(h), t)
                                         for _, h, t in window_batches(h5[rows[0]["path"]], EvaluationConfig())])
            self.assertAlmostEqual(score["damped_velocity"]["mean_endpoint_deg"], errors[:, -1].mean(), places=4)


class FigureTests(unittest.TestCase):
    def test_user_bootstrap_resamples_users(self):
        diff = np.array([-1., -1, -1, 1])
        users = np.array(["a", "a", "a", "b"])
        result = user_bootstrap(diff, users, rounds=500)
        self.assertAlmostEqual(result["mean"], -.5)
        self.assertEqual(result["per_user"], {"a": -1., "b": 1.})
        self.assertLessEqual(result["ci95"][0], -.99)
        self.assertGreaterEqual(result["ci95"][1], .99)

    def test_test_evaluation_writes_every_figure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, split = make_fixture(root/"data")
            suite = {"version": 1, "seeds": [42], "variants": {"control": {}, "damped_prior": {"damped_prior_tau_ms": 100}},
                     "baseline_lags": [1], "baseline_tau_ms": [100]}
            run_suite(data, root/"out", suite, base_config(), device="cpu", synthetic=True)
            evaluate_checkpoint(root/"out/selected_model.pt", data, split, root/"test", split="test",
                                selection_path=root/"out/selection.json", synthetic=True, batch_size=8, latency_steps=2,
                                damped_selection=root/"out/damped_baseline/selection.json")
            for name in ("horizon_relative", "endpoint_cdf", "per_user_endpoint", "paired_difference", "error_vs_speed",
                         "example_forecasts", "training_curves", "variant_comparison"):
                self.assertTrue((root/"test"/f"{name}.png").exists(), name)
            text = (root/"test/figures.md").read_text(encoding="utf-8")
            self.assertIn("user-bootstrap 95% CI", text)
            self.assertTrue((root/"test/test_receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
