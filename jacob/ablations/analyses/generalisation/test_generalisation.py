"""Tests for score_users.py and report.py. Synthetic data, except two checks that read the real manifests when present.

Run with the CUDA venv (it has torch, h5py, scipy), with TEMP/TMP pointing somewhere writable:
    python -m unittest discover -s jacob/ablations/analyses/generalisation -p "test_*.py" -t jacob/ablations/analyses/generalisation
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import report  # noqa: E402
import score_users  # noqa: E402
from bench.baselines import ConstantVelocity, DampedVelocity  # noqa: E402  (report put the repo on sys.path)
from bench.geometry import geodesic_degrees, random_motion  # noqa: E402
from train_tcnn import reference_predictors  # noqa: E402  (score_users put jacob/training on sys.path)


class Statistics(unittest.TestCase):
    def test_bootstrap_of_a_constant_is_the_constant(self):
        low, high = report.bootstrap_ci(np.full(10, 4.2), np.arange(1, 11, dtype=float))
        self.assertAlmostEqual(low, 4.2)
        self.assertAlmostEqual(high, 4.2)

    def test_bootstrap_covers_the_weighted_mean(self):
        values, weights = np.random.default_rng(1).normal(4, 0.5, 40), np.random.default_rng(2).uniform(1, 9, 40)
        low, high = report.bootstrap_ci(values, weights)
        self.assertLess(low, report.weighted_mean(values, weights))
        self.assertGreater(high, report.weighted_mean(values, weights))

    def test_two_groups_finds_a_shift_and_not_a_copy(self):
        a = np.random.default_rng(3).normal(0.80, 0.05, 50)
        shifted = report.two_groups(a, a + 0.1, draws=2000)
        self.assertAlmostEqual(shifted["diff"], 0.1)
        self.assertGreater(shifted["ci95"][0], 0)
        self.assertLess(shifted["permutation_p"], 0.01)
        same = report.two_groups(a, a.copy(), draws=2000)
        self.assertAlmostEqual(same["diff"], 0)
        self.assertGreater(same["permutation_p"], 0.9)

    def test_paired_is_the_window_weighted_difference(self):
        ids = np.array(["u1", "u2", "u3"])
        weight = np.array([100.0, 300.0, 600.0])
        a = report.Users(ids, np.array([4.0, 5.0, 3.0]), weight)
        b = report.Users(ids, np.array([4.5, 4.0, 3.5]), weight)
        p = report.paired(a, b, draws=500)
        self.assertAlmostEqual(p["mean_diff_deg"], (-0.5 * 100 + 1.0 * 300 - 0.5 * 600) / 1000)
        self.assertEqual(p["players_better"], 2)
        with self.assertRaises(ValueError):
            report.paired(a, report.Users(ids[::-1], b.mean, weight))


class Roles(unittest.TestCase):
    old = {"train": ["a", "b"], "validation": ["c"], "test": ["d"]}
    merged = {"train": ["a", "b", "e", "f"], "validation": ["c", "g"], "test": ["d", "h"]}

    def test_old_model_sees_every_new_user_as_unseen(self):
        r = report.roles("old", self.old, self.merged)
        self.assertEqual(r, {"trained": {"a", "b"}, "selection": {"c"}, "unseen": {"d", "e", "f", "g", "h"}})

    def test_merged_model_only_has_test_as_unseen(self):
        r = report.roles("merged", self.old, self.merged)
        self.assertEqual(r["unseen"], {"d", "h"})

    def test_overlapping_splits_are_refused(self):
        with self.assertRaises(ValueError):
            report.roles("old", {"train": ["a"], "validation": ["a"], "test": []}, self.merged)

    @unittest.skipUnless(report.MERGED_MANIFEST.exists() and report.OLD_MANIFEST.exists(), "manifests not present")
    def test_real_role_sizes(self):
        merged = json.loads(report.MERGED_MANIFEST.read_text(encoding="utf-8"))["splits"]
        old = json.loads(report.OLD_MANIFEST.read_text(encoding="utf-8"))["splits"]
        sizes = {m: {k: len(v) for k, v in report.roles(m, old, merged).items()} for m in ("old", "merged")}
        self.assertEqual(sizes["old"], {"trained": 100, "selection": 15, "unseen": 965})
        self.assertEqual(sizes["merged"], {"trained": 800, "selection": 120, "unseen": 160})


class Headsets(unittest.TestCase):
    def test_no_string_is_in_two_families(self):
        names = [n for family in report.HEADSET_FAMILIES.values() for n in family]
        self.assertEqual(len(names), len(set(names)))

    @unittest.skipUnless(report.MERGED_MANIFEST.exists(), "merged manifest not present")
    def test_every_manifest_headset_has_a_family(self):
        manifest = json.loads(report.MERGED_MANIFEST.read_text(encoding="utf-8"))
        headsets = {r["headset"] for r in manifest["recordings"] if r["status"] == "qualified"}
        self.assertEqual(headsets - set(report.FAMILY_OF), set())


class Scoring(unittest.TestCase):
    def test_strided_windows_split_back_into_their_segments(self):
        lengths = [420, 421, 457, 1000]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.h5"
            rows = []
            with h5py.File(path, "w") as h5:
                for i, n in enumerate(lengths):
                    q = random_motion(n, seed=i)
                    h5.create_dataset(f"u/r{i}/s0/quaternion_wxyz", data=q.astype(np.float32))
                    rows.append({"key": f"u/r{i}/s0", "path": f"u/r{i}/s0", "user": "u", "recording": f"r{i}"})
            for stride in (1, 25):
                predictors = reference_predictors(20)
                errors, counts = score_users.score_user(predictors, path, rows, "cpu", stride, batch=64)
                self.assertEqual(counts, [(n - 420) // stride + 1 for n in lengths])
                # The last segment's windows, scored on their own, give the same errors as its slice of the run.
                alone, _ = score_users.score_user(predictors, path, rows[-1:], "cpu", stride, batch=64)
                np.testing.assert_allclose(errors["hold_last"][-counts[-1]:], alone["hold_last"])

    def test_constant_and_damped_velocity_match_the_bench(self):
        q = random_motion(440, seed=7)
        windows = np.lib.stride_tricks.sliding_window_view(q, 420, axis=0).transpose(0, 2, 1)
        history = torch.from_numpy(windows[:, :400].astype(np.float32))
        mine = {"cv": score_users.constant_velocity(20)(history).double().numpy(),
                "damped": reference_predictors(20)["damped_velocity"](history).double().numpy()}
        theirs = {"cv": ConstantVelocity(20).predict_batch(windows[:, :400]),
                  "damped": DampedVelocity(20).predict_batch(windows[:, :400])}
        for name in mine:
            mine[name] /= np.linalg.norm(mine[name], axis=-1, keepdims=True)
            self.assertLess(geodesic_degrees(mine[name], theirs[name]).max(), 1e-3, name)


if __name__ == "__main__":
    unittest.main()
