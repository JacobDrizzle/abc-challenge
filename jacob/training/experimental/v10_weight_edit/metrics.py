"""Window-weighted errors and paired user comparisons for the weight-edit experiment.

Inputs are already-computed sign-invariant geodesic errors in degrees. Resampling
users, rather than overlapping windows, matches the team's benchmark statistic.
"""
from __future__ import annotations

import numpy as np


class ErrorAccumulator:
    """Accumulate batches for one user at a time, retaining endpoint quantiles."""

    def __init__(self, horizon=20):
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        self.horizon = horizon
        self._users = {}
        self._endpoints = []

    def update(self, user_id, errors_deg):
        if not isinstance(user_id, str) or not user_id:
            raise ValueError("user_id must be a nonempty string")
        errors = np.asarray(errors_deg, dtype=np.float64)
        if errors.ndim != 2 or errors.shape[1] != self.horizon:
            raise ValueError(f"Expected errors shaped [batch, {self.horizon}]")
        if not np.isfinite(errors).all() or np.any(errors < 0) or np.any(errors > 180):
            raise ValueError("Geodesic errors must be finite and between 0 and 180 degrees")
        if not len(errors):
            return
        row = self._users.setdefault(user_id, {
            "windows": 0, "sum_deg_by_horizon": np.zeros(self.horizon, dtype=np.float64)})
        row["windows"] += len(errors)
        row["sum_deg_by_horizon"] += errors.sum(axis=0, dtype=np.float64)
        # The caller may reuse its batch buffer after this update.
        self._endpoints.append(errors[:, -1].copy())

    def summary(self):
        if not self._users:
            raise ValueError("No windows have been accumulated")
        users = {user: {"windows": row["windows"],
                        "sum_deg_by_horizon": row["sum_deg_by_horizon"].tolist()}
                 for user, row in sorted(self._users.items())}
        count = sum(row["windows"] for row in users.values())
        sums = np.asarray([row["sum_deg_by_horizon"] for row in users.values()]).sum(axis=0)
        means = sums / count
        endpoints = np.concatenate(self._endpoints)
        median, p95, p99 = np.percentile(endpoints, [50, 95, 99])
        return {
            "windows": count,
            "mean_endpoint_deg": float(means[-1]),
            "mean_deg_by_horizon": means.tolist(),
            "mean_all_horizons_deg": float(means.mean()),
            "equal_user_mean_endpoint_deg": float(np.mean([
                row["sum_deg_by_horizon"][-1] / row["windows"] for row in users.values()])),
            "median_endpoint_deg": float(median),
            "p95_endpoint_deg": float(p95),
            "p99_endpoint_deg": float(p99),
            "per_user": users,
        }


def _paired_arrays(candidate, baseline):
    users = sorted(candidate["per_user"])
    if not users or users != sorted(baseline["per_user"]):
        raise ValueError("Comparisons require the same nonempty user cohort")
    counts = np.asarray([candidate["per_user"][u]["windows"] for u in users], dtype=np.float64)
    reference_counts = np.asarray([baseline["per_user"][u]["windows"] for u in users], dtype=np.float64)
    if not np.array_equal(counts, reference_counts) or np.any(counts <= 0):
        raise ValueError("Comparisons require identical positive window counts per user")
    if counts.sum() != candidate["windows"] or counts.sum() != baseline["windows"]:
        raise ValueError("Total window counts must match the per-user counts")
    sums = np.asarray([candidate["per_user"][u]["sum_deg_by_horizon"] for u in users], dtype=np.float64)
    reference_sums = np.asarray([baseline["per_user"][u]["sum_deg_by_horizon"] for u in users], dtype=np.float64)
    if sums.ndim != 2 or sums.shape != reference_sums.shape or sums.shape[1] == 0:
        raise ValueError("Comparisons require the same prediction horizons")
    if not np.isfinite(sums).all() or not np.isfinite(reference_sums).all():
        raise ValueError("Error sums must be finite")
    return users, counts, sums[:, -1] / counts - reference_sums[:, -1] / counts


def paired_comparison(candidate, baseline, draws=10000, seed=0):
    """Candidate minus baseline, with a window-weighted user-bootstrap interval.

    Identical counts alone cannot prove identical windows: callers must also use
    the same frozen window inventory and record its fingerprint.
    """
    if not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    users, counts, differences = _paired_arrays(candidate, baseline)
    picks = np.random.default_rng(seed).integers(0, len(users), (draws, len(users)))
    bootstrap = (differences[picks] * counts[picks]).sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return {
        "mean_diff_deg": float(np.sum(differences * counts) / counts.sum()),
        "mean_diff_deg_by_horizon": (
            np.asarray(candidate["mean_deg_by_horizon"]) - np.asarray(baseline["mean_deg_by_horizon"])
        ).tolist(),
        "mean_all_horizons_diff_deg": candidate["mean_all_horizons_deg"] - baseline["mean_all_horizons_deg"],
        "ci95": [float(low), float(high)],
        "equal_user_mean_diff_deg": float(differences.mean()),
        "users_better": int(np.count_nonzero(differences < 0)),
        "users": len(users),
        "draws": draws,
        "seed": seed,
    }


def promising_gate(candidate, baseline, paired=None, expected_users=40):
    """The predeclared seed-44 confirmation gate; this does not promote a model."""
    users, _, _ = _paired_arrays(candidate, baseline)
    paired = paired_comparison(candidate, baseline) if paired is None else paired
    if paired["users"] != len(users):
        raise ValueError("Paired result has a different number of users")
    candidate_horizons = np.asarray(candidate["mean_deg_by_horizon"], dtype=np.float64)
    baseline_horizons = np.asarray(baseline["mean_deg_by_horizon"], dtype=np.float64)
    if candidate_horizons.shape != baseline_horizons.shape:
        raise ValueError("Comparisons require the same prediction horizons")
    checks = {
        "expected_confirmation_users": len(users) == expected_users,
        "mean_endpoint_lower": paired["mean_diff_deg"] < 0,
        "equal_user_mean_lower": paired["equal_user_mean_diff_deg"] < 0,
        "majority_users_better": paired["users_better"] > len(users) / 2,
        "ci95_below_zero": paired["ci95"][1] < 0,
        "p95_no_worse": candidate["p95_endpoint_deg"] <= baseline["p95_endpoint_deg"],
        "mean_all_horizons_no_worse": bool(candidate_horizons.mean() <= baseline_horizons.mean()),
    }
    return {"passed": all(checks.values()), "checks": checks}
