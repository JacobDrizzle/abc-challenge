"""Analytical geometry, leakage, streaming and end-to-end scorer checks."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from baseline_pipeline import (ConstantVelocity, EvaluationConfig, HoldLast, evaluate,
                               geodesic_degrees, inventory, load_split, make_predictors,
                               manifest_for, unit_quaternions, window_batches)


def manifest(validation, train=(), test=()):
    """The split part of a ray/ export manifest."""
    return json.dumps({"splits": {"train": list(train), "validation": list(validation), "test": list(test)}})


def segment(user):
    return f"validation/{user}/recording/segment_000"


def turning(n, degrees_per_sample=1):
    q = Rotation.from_euler("z", (np.arange(n) * degrees_per_sample)[:, None], degrees=True).as_quat()
    return q[:, [3, 0, 1, 2]]


STREAM_MODELS = []


class StreamingProbe(HoldLast):
    name = "stream_probe"
    stateful = True

    def __init__(self, config):
        super().__init__(config)
        self.reset_count = 0
        self.calls = 0
        self.previous = None

    def reset(self):
        self.reset_count += 1
        self.previous = None

    def predict(self, history):
        if self.previous is not None:
            np.testing.assert_array_equal(self.previous[1:], history[:-1])
        self.previous = history.copy()
        self.calls += 1
        return np.repeat(history[-1:], self.config.horizon, axis=0)

    def predict_batch(self, history):
        raise AssertionError("Stateful models must not be batched")


def streaming_factory(config):
    model = StreamingProbe(config)
    STREAM_MODELS.append(model)
    return model


class InvalidModel(HoldLast):
    name = "invalid_model"

    def predict_batch(self, history):
        return np.full((len(history), self.config.horizon, 4), np.nan)


def invalid_factory(config):
    return InvalidModel(config)


class GeometryTests(unittest.TestCase):
    def test_known_angles_and_double_cover(self):
        a = np.tile([1., 0, 0, 0], (4, 1))
        b = Rotation.from_euler("x", np.array([0, 30, 90, 180])[:, None], degrees=True).as_quat()[:, [3, 0, 1, 2]]
        np.testing.assert_allclose(geodesic_degrees(a, b), [0, 30, 90, 180], atol=1e-5)
        np.testing.assert_allclose(geodesic_degrees(b, -b), 0, atol=1e-5)

    def test_invalid_rotations_are_rejected(self):
        for q in ([0, 0, 0, 0], [2, 0, 0, 0], [np.nan, 0, 0, 1], [np.inf, 0, 0, 1], [1, 0, 0]):
            with self.subTest(q=q), self.assertRaises(ValueError):
                unit_quaternions(q)
        np.testing.assert_allclose(unit_quaternions([1.0000001, 0, 0, 0]), [1, 0, 0, 0])

    def test_shapes_cannot_broadcast_silently(self):
        with self.assertRaises(ValueError):
            geodesic_degrees(np.array([[1, 0, 0, 0]]), np.tile([1, 0, 0, 0], (3, 1)))

    def test_constant_rotation_with_nonidentity_start(self):
        config = EvaluationConfig()
        axis = np.array([1., -2., 3.])
        axis /= np.linalg.norm(axis)
        initial = Rotation.from_euler("xyz", [23, -31, 48], degrees=True)
        rotations = Rotation.from_rotvec(np.arange(440)[:, None] * .006 * axis) * initial
        q = rotations.as_quat()[:, [3, 0, 1, 2]]
        for lag in (1, 5):
            model = ConstantVelocity(config, lag)
            prediction = model.predict(q[:400])
            self.assertLess(geodesic_degrees(prediction, q[400:420]).max(), 1e-5)
            flipped = q[:400].copy()
            flipped[::2] *= -1
            self.assertLess(geodesic_degrees(model.predict(flipped), prediction).max(), 1e-5)

    def test_stationary_predictions(self):
        config = EvaluationConfig(context=8, horizon=3)
        history = np.tile([.5, .5, .5, .5], (8, 1))
        for model in make_predictors(config):
            self.assertLess(geodesic_degrees(model.predict(history), history[:3]).max(), 1e-5)

    def test_invalid_configuration(self):
        for kwargs in ({"context": 5}, {"horizon": 0}, {"stride": 0}, {"batch_size": 0}, {"rate_hz": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EvaluationConfig(**kwargs)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data.h5"
        self.split = self.root / "manifest_test.json"
        self.config = EvaluationConfig(context=8, horizon=3, batch_size=2)
        self.split.write_text(manifest(["alice", "bob"]))
        with h5py.File(self.data, "w") as h5:
            # ray/'s group-a-v2 layout: users inside the split groups listed in the root attribute splits.
            h5.attrs.update(quaternion_order="wxyz", rate_hz=100, run_id="test")
            h5.attrs.create("splits", ["train", "validation", "test"], dtype=h5py.string_dtype())
            h5.create_group("train")
            h5.create_group("test")
            for user, n, speed in (("alice", 11, 1), ("bob", 13, 2)):
                g = h5.create_group(segment(user))
                g.attrs["challenge_filter_pass"] = True
                g.create_dataset("time_s", data=np.arange(n) / 100)
                g.create_dataset("quaternion_wxyz", data=turning(n, speed))

    def tearDown(self):
        self.temp.cleanup()

    def run_eval(self, name="run", **kwargs):
        return evaluate(self.data, self.split, self.root / name, config=kwargs.pop("config", self.config), progress=None, **kwargs)

    def test_window_boundaries_and_no_cross_segment_windows(self):
        with h5py.File(self.data) as h5:
            batches = list(window_batches(h5[segment("alice")], self.config))
            self.assertEqual(len(batches), 1)
            anchors, history, target = batches[0]
            np.testing.assert_array_equal(anchors, [7])
            np.testing.assert_allclose(history[0], turning(11)[:8], atol=3e-8)
            np.testing.assert_allclose(target[0], turning(11)[8:11], atol=1e-14)
            anchors_b = np.concatenate([x[0] for x in window_batches(h5[segment("bob")], self.config)])
            np.testing.assert_array_equal(anchors_b, [7, 8, 9])

    def test_weighted_metric_and_per_user_diagnostics(self):
        report = self.run_eval(latency_steps=2)
        hold = report["models"]["hold_last"]
        # One window at 3 degrees; three windows at 6 degrees. Equal-user mean differs.
        self.assertEqual(report["windows"], 4)
        self.assertAlmostEqual(hold["mean_endpoint_deg"], 5.25, places=5)
        self.assertAlmostEqual(hold["equal_user_mean_endpoint_deg"], 4.5, places=5)
        self.assertAlmostEqual(hold["p95_endpoint_deg"], 6, places=5)
        self.assertEqual(report["horizons_ms"], [10, 20, 30])
        for name in ("constant_velocity_1step", "constant_velocity_5step"):
            self.assertLess(report["models"][name]["mean_endpoint_deg"], 1e-4)
        stored = np.load(self.root / "run/endpoint_hold_last.npy")
        np.testing.assert_allclose(stored, [3, 6, 6, 6], atol=2e-6)
        index = np.load(self.root / "run/window_index.npy")
        np.testing.assert_array_equal(index["segment_id"], [0, 1, 1, 1])
        np.testing.assert_array_equal(index["anchor"], [7, 7, 8, 9])
        self.assertTrue((self.root / "run/report.md").is_file())
        self.assertTrue((self.root / "run/horizon_errors.png").is_file())
        self.assertGreaterEqual(hold["local_latency"]["median_ms"], 0)

    def test_batch_size_does_not_change_cohort_or_metrics(self):
        a = self.run_eval("a")
        b = self.run_eval("b", config=EvaluationConfig(context=8, horizon=3, batch_size=1))
        self.assertEqual(a["cohort_fingerprint"], b["cohort_fingerprint"])
        for name in a["models"]:
            np.testing.assert_allclose(a["models"][name]["mean_deg_by_horizon"], b["models"][name]["mean_deg_by_horizon"], atol=1e-12)

    def test_model_adapter_uses_same_scorer(self):
        result = self.run_eval(model_specs=["example_model_adapter:build_model"])
        self.assertEqual(result["models"]["hold_last"]["mean_endpoint_deg"], result["models"]["example_model"]["mean_endpoint_deg"])
        np.testing.assert_array_equal(np.load(self.root / "run/endpoint_hold_last.npy"), np.load(self.root / "run/endpoint_example_model.npy"))

    def test_stateful_models_advance_once_and_reset_between_segments(self):
        self.run_eval(model_specs=[__name__ + ":streaming_factory"])
        self.assertEqual(STREAM_MODELS[-1].reset_count, 2)
        self.assertEqual(STREAM_MODELS[-1].calls, 4)
        with self.assertRaisesRegex(ValueError, "stride=1"):
            self.run_eval("skipped", config=EvaluationConfig(context=8, horizon=3, stride=2), model_specs=[__name__ + ":streaming_factory"])

    def test_future_target_changes_do_not_change_input(self):
        with h5py.File(self.data, "r+") as h5:
            g = h5[segment("alice")]
            before = next(window_batches(g, self.config))
            g["quaternion_wxyz"][8:] = turning(3, 10)
            after = next(window_batches(g, self.config))
            np.testing.assert_array_equal(before[1], after[1])
            self.assertFalse(np.array_equal(before[2], after[2]))
            for model in make_predictors(self.config):
                np.testing.assert_array_equal(model.predict(before[1][0]), model.predict(after[1][0]))

    def test_qualification_is_required_and_exploration_is_explicit(self):
        with h5py.File(self.data, "r+") as h5:
            for u in h5["validation"]:
                h5[segment(u)].attrs["challenge_filter_pass"] = False
        with self.assertRaisesRegex(ValueError, "No eligible"):
            self.run_eval()
        report = self.run_eval(allow_exploratory=True)
        self.assertTrue(report["exploratory"])
        self.assertIn("EXPLORATORY", (self.root / "run/report.md").read_text())

    def test_overlap_and_duplicates_in_split_are_rejected(self):
        for doc in (manifest(["alice"], train=["alice"]), manifest(["alice", "alice"])):
            self.split.write_text(doc)
            with self.assertRaises(ValueError):
                load_split(self.split, "validation")

    def test_missing_user_and_empty_split_are_rejected(self):
        for users in ([], ["unknown"]):
            self.split.write_text(manifest(users))
            with self.assertRaises(ValueError):
                self.run_eval()

    def test_clock_order_and_qualification_types_are_checked(self):
        with h5py.File(self.data, "r+") as h5:
            h5.attrs["rate_hz"] = 90
            with self.assertRaisesRegex(ValueError, "clock"):
                inventory(h5, ["alice"], self.config, False)
            h5.attrs["rate_hz"] = 100
            h5.attrs["quaternion_order"] = "xyzw"
            with self.assertRaisesRegex(ValueError, "wxyz"):
                inventory(h5, ["alice"], self.config, False)
            h5.attrs["quaternion_order"] = "wxyz"
            h5[segment("alice")].attrs["challenge_filter_pass"] = "False"
            with self.assertRaisesRegex(ValueError, "boolean"):
                inventory(h5, ["alice"], self.config, False)

    def test_timestamp_gaps_fail_and_leave_no_partial_result(self):
        with h5py.File(self.data, "r+") as h5:
            h5[segment("alice") + "/time_s"][9] = 1
        with self.assertRaisesRegex(ValueError, "timestamps"):
            self.run_eval()
        self.assertFalse((self.root / "run").exists())
        self.assertFalse(list(self.root.glob(".baseline-run-*")))

    def test_nonfinite_model_output_fails_without_partial_results(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            self.run_eval(model_specs=[__name__ + ":invalid_factory"])
        self.assertFalse((self.root / "run").exists())

    def test_split_layout_keys_paths_and_manifest(self):
        self.assertEqual(manifest_for(self.data), self.root / "manifest_test.json")
        with h5py.File(self.data) as h5:
            rows, _ = inventory(h5, ["alice", "bob"], self.config, False)
        # The key has no split, as in the old root layout, so fingerprints keep their form; the path finds the data.
        self.assertEqual([r["key"] for r in rows], ["alice/recording/segment_000", "bob/recording/segment_000"])
        self.assertEqual([r["path"] for r in rows], [segment("alice"), segment("bob")])

    def test_user_in_two_splits_and_root_layout_are_rejected(self):
        with h5py.File(self.data, "r+") as h5:
            h5.copy(h5["validation/alice"], h5["test"], name="alice")
            with self.assertRaisesRegex(ValueError, "two splits"):
                inventory(h5, ["alice"], self.config, False)
            del h5["test/alice"]
            h5.copy(h5["validation/alice"], h5, name="alice")
            with self.assertRaisesRegex(ValueError, "split groups"):
                inventory(h5, ["alice"], self.config, False)

    def test_existing_run_is_preserved(self):
        (self.root / "run").mkdir()
        marker = self.root / "run/do-not-overwrite.txt"
        marker.write_text("original")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_eval()
        self.assertEqual(marker.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
