"""Run from jacob/training: python -m unittest experimental.v10_weight_edit.test_metrics."""
import unittest

import numpy as np

from experimental.v10_weight_edit.metrics import ErrorAccumulator, paired_comparison, promising_gate


def summary(user_rows):
    result = ErrorAccumulator()
    for user, values in user_rows.items():
        endpoints = np.asarray(values, dtype=np.float64)
        result.update(user, np.repeat(endpoints[:, None], 20, axis=1))
    return result.summary()


class MetricsTests(unittest.TestCase):
    def test_window_weighting_and_exact_percentiles(self):
        result = summary({"a": [1, 2, 3], "b": [10]})
        self.assertEqual(result["windows"], 4)
        self.assertEqual(result["mean_endpoint_deg"], 4)
        self.assertEqual(result["equal_user_mean_endpoint_deg"], 6)
        self.assertEqual(result["mean_deg_by_horizon"], [4] * 20)
        self.assertEqual(result["mean_all_horizons_deg"], 4)
        self.assertEqual(result["per_user"]["a"]["sum_deg_by_horizon"], [6] * 20)
        self.assertEqual(result["median_endpoint_deg"], 2.5)
        self.assertAlmostEqual(result["p95_endpoint_deg"], 8.95)
        self.assertAlmostEqual(result["p99_endpoint_deg"], 9.79)

    def test_multiple_updates_copy_endpoint_buffer(self):
        accumulator = ErrorAccumulator()
        batch = np.full((2, 20), 2.0)
        accumulator.update("a", batch)
        batch.fill(100)
        accumulator.update("a", np.full((1, 20), 5.0))
        result = accumulator.summary()
        self.assertEqual(result["mean_endpoint_deg"], 3)
        self.assertEqual(result["median_endpoint_deg"], 2)
        self.assertEqual(result["per_user"]["a"]["windows"], 3)

    def test_bootstrap_uses_ratio_of_resampled_sums_and_counts(self):
        baseline = summary({"a": [5, 5, 5], "b": [5]})
        candidate = summary({"a": [4, 4, 4], "b": [7]})
        result = paired_comparison(candidate, baseline)
        self.assertEqual(result["mean_diff_deg"], -0.25)
        self.assertEqual(result["equal_user_mean_diff_deg"], 0.5)
        self.assertEqual(result["users_better"], 1)
        self.assertEqual(result["ci95"], [-1.0, 2.0])
        self.assertEqual(result, paired_comparison(candidate, baseline))
        picks = np.random.default_rng(0).integers(0, 2, (10000, 2))
        counts, differences = np.array([3, 1]), np.array([-1.0, 2.0])
        boots = (differences[picks] * counts[picks]).sum(1) / counts[picks].sum(1)
        np.testing.assert_array_equal(result["ci95"], np.percentile(boots, [2.5, 97.5]))

    def test_pairing_rejects_different_users_counts_and_horizons(self):
        baseline = summary({"a": [5], "b": [5]})
        for candidate in (summary({"a": [4], "c": [4]}), summary({"a": [4, 4], "b": [4]})):
            with self.assertRaises(ValueError):
                paired_comparison(candidate, baseline)
        candidate = summary({"a": [4], "b": [4]})
        candidate["per_user"]["a"]["sum_deg_by_horizon"] = [4] * 19
        candidate["per_user"]["b"]["sum_deg_by_horizon"] = [4] * 19
        with self.assertRaises(ValueError):
            paired_comparison(candidate, baseline)

    def test_gate_requires_strict_improvement_and_40_users(self):
        baseline = summary({str(i): [5] for i in range(40)})
        candidate = summary({str(i): [4] for i in range(40)})
        self.assertTrue(promising_gate(candidate, baseline)["passed"])
        tied = promising_gate(baseline, baseline)
        self.assertFalse(tied["passed"])
        self.assertFalse(tied["checks"]["mean_endpoint_lower"])
        self.assertFalse(tied["checks"]["ci95_below_zero"])
        small = promising_gate(summary({"a": [4], "b": [4]}), summary({"a": [5], "b": [5]}))
        self.assertFalse(small["checks"]["expected_confirmation_users"])

    def test_gate_requires_majority_and_no_tail_or_horizon_regression(self):
        baseline = summary({str(i): [5] for i in range(40)})
        half_better = summary({str(i): [4 if i < 20 else 5] for i in range(40)})
        self.assertFalse(promising_gate(half_better, baseline)["checks"]["majority_users_better"])
        accumulator = ErrorAccumulator()
        for i in range(40):
            errors = np.full((1, 20), 4.0)
            errors[:, 0] = 6
            accumulator.update(str(i), errors)
        result = promising_gate(accumulator.summary(), baseline)
        # One horizon may regress when the average over all 20 still improves.
        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"]["mean_all_horizons_no_worse"])
        accumulator = ErrorAccumulator()
        for i in range(40):
            errors = np.full((1, 20), 6.0)
            errors[:, -1] = 4
            accumulator.update(str(i), errors)
        result = promising_gate(accumulator.summary(), baseline)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["mean_all_horizons_no_worse"])
        tail_worse = summary({str(i): [4, 4, 4, 8] for i in range(40)})
        reference = summary({str(i): [5, 5, 5, 5] for i in range(40)})
        self.assertFalse(promising_gate(tail_worse, reference)["checks"]["p95_no_worse"])

    def test_invalid_errors_and_empty_accumulator(self):
        accumulator = ErrorAccumulator()
        with self.assertRaises(ValueError):
            accumulator.summary()
        for errors in (np.zeros((2, 19)), np.full((1, 20), np.nan),
                       np.full((1, 20), -1), np.full((1, 20), 181)):
            with self.assertRaises(ValueError):
                accumulator.update("a", errors)


if __name__ == "__main__":
    unittest.main()
