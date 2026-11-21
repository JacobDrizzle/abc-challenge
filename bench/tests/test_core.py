"""Metric, baselines, data enumeration, fingerprints and the scorer."""
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from bench import data
from bench.baselines import ConstantVelocity, DampedVelocity, HoldLast
from bench.geometry import axis_angle, geodesic_degrees, qmul, renormalise
from bench.scoring import BaselinePredictor, score
from bench.tests import helpers

LAYOUT = {"b_user": [("rec1", 1000, True), ("rec1", 419, True), ("rec2", 500, False)],
          "a_user": [("r", 420, True)],
          "c_user": [("r", 600, True)]}


class Metric(unittest.TestCase):
    def test_hand_made_rotations(self):
        identity = np.array([1., 0, 0, 0])
        self.assertAlmostEqual(geodesic_degrees(axis_angle([0, 0, 1], 90), identity), 90, places=10)
        self.assertAlmostEqual(geodesic_degrees(axis_angle([1, 1, 0], 180), identity), 180, places=10)
        self.assertAlmostEqual(geodesic_degrees(axis_angle([0, 1, 0], 0.01), identity), 0.01, places=8)
        a, b = axis_angle([0, 0, 1], 30), axis_angle([0, 0, 1], -20)
        self.assertAlmostEqual(geodesic_degrees(a, b), 50, places=10)

    def test_sign_invariance_and_random_rotations(self):
        rng = np.random.default_rng(1)
        p = Rotation.random(500, random_state=2).as_quat()[:, [3, 0, 1, 2]]
        q = Rotation.random(500, random_state=3).as_quat()[:, [3, 0, 1, 2]]
        np.testing.assert_allclose(geodesic_degrees(p, q), geodesic_degrees(-p, q), atol=1e-12)
        np.testing.assert_allclose(geodesic_degrees(p, q), geodesic_degrees(p, -q * rng.choice([1, -1], (500, 1))), atol=1e-12)
        expected = np.rad2deg((Rotation.from_quat(p[:, [1, 2, 3, 0]]).inv() * Rotation.from_quat(q[:, [1, 2, 3, 0]])).magnitude())
        np.testing.assert_allclose(geodesic_degrees(p, q), expected, atol=1e-6)

    def test_renormalise_like_the_harness(self):
        q = axis_angle([1, 0, 0], 10)
        np.testing.assert_allclose(renormalise(3.7 * q), q)
        for bad in (np.array([np.nan, 0, 0, 1]), np.zeros(4)):
            with self.assertRaises(ValueError):
                renormalise(bad)

    def test_qmul_composes_rotations(self):
        a, b = axis_angle([0, 0, 1], 20), axis_angle([0, 0, 1], 25)
        self.assertAlmostEqual(geodesic_degrees(qmul(a, b), axis_angle([0, 0, 1], 45)), 0, places=6)


class Baselines(unittest.TestCase):
    """On a constant-rate rotation each baseline's error is known in closed form."""
    rate = 0.5  # degrees per sample, i.e. 50 deg/s

    def setUp(self):
        axis = np.array([0.3, 1.0, -0.2])
        steps = np.arange(420)
        self.q = np.stack([qmul(axis_angle(axis, self.rate * t), axis_angle([1, 0, 0], 40)) for t in steps])
        self.history, self.future = self.q[None, :400].astype(np.float32), self.q[400:]

    def errors(self, model):
        return geodesic_degrees(model.predict_batch(self.history)[0], self.future)

    def test_hold_last(self):
        np.testing.assert_allclose(self.errors(HoldLast()), self.rate * np.arange(1, 21), atol=2e-3)

    def test_constant_velocity_is_exact(self):
        np.testing.assert_allclose(self.errors(ConstantVelocity()), 0, atol=2e-3)

    def test_damped_velocity(self):
        h, tau = np.arange(1, 21), 10.0  # 100 ms at 100 Hz
        np.testing.assert_allclose(self.errors(DampedVelocity()), self.rate * (h - tau * (1 - np.exp(-h / tau))), atol=2e-3)


class DataAndScoring(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.h5 = helpers.make_h5(Path(self.tmp.name) / "d.h5", LAYOUT)
        self.definition = helpers.definition(["a_user", "b_user"])

    def tearDown(self):
        self.tmp.cleanup()

    def test_inventory_order_and_skips(self):
        with data.Benchmark(self.definition, self.h5) as b:
            self.assertEqual([s["key"] for s in b.segments], ["a_user/r/segment_000", "b_user/rec1/segment_000"])
            self.assertEqual([s["windows"] for s in b.segments], [1, 581])
            self.assertEqual(b.windows, 582)
            self.assertEqual(sorted(x["reason"] for x in b.skipped), ["motion_filter_failed", "too_short"])

    def test_missing_player_is_an_error(self):
        with self.assertRaises(ValueError):
            data.Benchmark(helpers.definition(["a_user", "zz"]), self.h5)

    def test_fingerprint_follows_content_not_file(self):
        with data.Benchmark(self.definition, self.h5) as b:
            fingerprint, legacy = b.fingerprint, b.legacy_fingerprint
        extract = data.extract(self.h5, ["a_user", "b_user"], Path(self.tmp.name) / "x.h5")
        with data.Benchmark({**self.definition, "cohort_fingerprint": fingerprint}, extract) as b:
            self.assertEqual(b.fingerprint, fingerprint)          # same windows and values
            self.assertNotEqual(b.legacy_fingerprint, legacy)     # legacy one hashes the file
        with h5py.File(extract, "r+") as h5:
            h5["b_user/rec1/segment_000/quaternion_wxyz"][500] *= -1  # same rotation, different bytes
        with self.assertRaises(ValueError):
            data.Benchmark({**self.definition, "cohort_fingerprint": fingerprint}, extract)

    def test_split_layout_reads_like_the_flat_one(self):
        nested = helpers.nest_in_splits(self.h5, Path(self.tmp.name) / "nested.h5",
                                        {"train": ["c_user"], "validation": [], "test": ["a_user", "b_user"]})
        with data.Benchmark(self.definition, self.h5) as flat, data.Benchmark(self.definition, nested) as b:
            self.assertEqual([s["key"] for s in b.segments], [s["key"] for s in flat.segments])
            self.assertEqual([s["path"] for s in b.segments], ["test/a_user/r/segment_000", "test/b_user/rec1/segment_000"])
            self.assertEqual(b.fingerprint, flat.fingerprint)
            np.testing.assert_array_equal(b.segment(b.segments[1]), flat.segment(flat.segments[1]))
            fingerprint = b.fingerprint
        extract = data.extract(nested, ["a_user", "b_user"], Path(self.tmp.name) / "x.h5")
        with h5py.File(extract, "r") as h5:
            self.assertEqual(sorted(h5), ["a_user", "b_user"])     # players back at the root
            self.assertNotIn("splits", h5.attrs)
        with data.Benchmark({**self.definition, "cohort_fingerprint": fingerprint}, extract) as b:
            self.assertEqual(b.fingerprint, fingerprint)

    def test_split_attribute_alone_does_not_change_a_flat_file(self):
        with h5py.File(self.h5, "r+") as h5:
            h5.attrs["splits"] = ["train", "validation", "test"]   # but the players are still at the root
        with data.Benchmark(self.definition, self.h5) as b:
            self.assertEqual(b.windows, 582)

    def test_player_in_two_splits_is_an_error(self):
        nested = helpers.nest_in_splits(self.h5, Path(self.tmp.name) / "nested.h5",
                                        {"train": ["a_user"], "validation": [], "test": ["a_user", "b_user"]})
        with self.assertRaises(ValueError):
            data.Benchmark(self.definition, nested)

    def test_read_segment_rejects_gaps_and_sign_flips(self):
        with h5py.File(self.h5, "r+") as h5:
            h5["b_user/rec1/segment_000/time_s"][700:] += 0.5
            h5["a_user/r/segment_000/quaternion_wxyz"][100:] *= -1
        with h5py.File(self.h5, "r") as h5:
            for key in ("b_user/rec1/segment_000", "a_user/r/segment_000"):
                with self.assertRaises(ValueError):
                    data.read_segment(h5, key)

    def test_score_counts_and_per_player_sums(self):
        with data.Benchmark(self.definition, self.h5) as b:
            accuracy, endpoint, segment = score(BaselinePredictor(HoldLast()), b, progress=None)
        self.assertEqual(accuracy["windows"], 582)
        self.assertEqual(accuracy["per_player"]["a_user"]["windows"], 1)
        self.assertAlmostEqual(accuracy["mean_deg_step20"], endpoint.mean(), places=10)
        total = sum(p["sum_deg_by_horizon"][-1] for p in accuracy["per_player"].values())
        self.assertAlmostEqual(total / 582, accuracy["mean_deg_step20"], places=10)
        self.assertEqual(list(np.bincount(segment)), [1, 581])
        self.assertAlmostEqual(accuracy["equal_player_mean_deg_step20"],
                               (endpoint[0] + endpoint[1:].mean()) / 2, places=10)


if __name__ == "__main__":
    unittest.main()
