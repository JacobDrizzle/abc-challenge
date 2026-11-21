"""Paired per-player bootstrap of the difference in mean step-20 error between two models.

Port of paired_users in ABC_local_test/experimental/team_comparison/compare_nic.py. Players, not windows,
are resampled: windows of one player are strongly correlated, so a window bootstrap would be far too
narrow. Each player counts with its number of windows, so the resampled statistic is the same
window-weighted mean the leaderboard reports. On identical windows, a player's mean difference equals
the difference of the two models' per-player means, so the per-player sums stored in each result JSON
are all that is needed: no per-window files.
"""
import numpy as np


def has_per_player(accuracy):
    """True if the result keeps its per-player sums. The public copy of the results leaves them out (they are
    per-user scores), and without them there is no paired bootstrap, only the difference of the two means."""
    return bool(accuracy) and "per_player" in accuracy


def per_player_step20(accuracy):
    players = sorted(accuracy["per_player"])
    windows = np.array([accuracy["per_player"][u]["windows"] for u in players], np.float64)
    sums = np.array([accuracy["per_player"][u]["sum_deg_by_horizon"][-1] for u in players], np.float64)
    return players, windows, sums


def paired(a, b, draws=10000, seed=0):
    """Mean of a - b over windows, with a 95% interval from resampling players. Negative = a is better."""
    players, weight, sum_a = per_player_step20(a)
    players_b, weight_b, sum_b = per_player_step20(b)
    if players != players_b or not np.array_equal(weight, weight_b):
        raise ValueError("The two results are not on the same players and windows")
    per_player = sum_a / weight - sum_b / weight
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(players), (draws, len(players)))
    boots = (per_player[picks] * weight[picks]).sum(1) / weight[picks].sum(1)
    low, high = np.percentile(boots, [2.5, 97.5])
    return {"mean_diff_deg": float((per_player * weight).sum() / weight.sum()), "ci95": [float(low), float(high)],
            "equal_player_mean_diff_deg": float(per_player.mean()),
            "players_better": int((per_player < 0).sum()), "players": len(players), "draws": draws, "seed": seed}
