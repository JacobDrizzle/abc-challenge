"""v10_users: users_100_half against users_50 (diversity or volume), and the per-doubling scaling of the user curve.

Read-only: reads v10_users_paired.json next to this script and k8_combined seed 42's full validation for the window
weights, and writes v10_users_scaling.json. Run from jacob/training: python <this script>.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONTROL = Path("outputs/v9_confirm/k8_combined/seed_42/full_validation/results.json")

rows = json.loads((HERE/"v10_users_paired.json").read_text(encoding="utf-8"))["v10_users"]
by_user = json.loads(CONTROL.read_text(encoding="utf-8"))["models"]["tiny_tcnn"]["by_user"]
users = sorted(by_user)
w = np.array([by_user[u]["windows"] for u in users], dtype=float)
delta = {name: np.array([rows[name]["paired"]["per_user_delta"][u] for u in users])
         for name in ("users_25", "users_50", "users_100_half")}
d = delta["users_100_half"] - delta["users_50"]
picks = np.random.default_rng(0).integers(0, len(users), size=(20000, len(users)))
boot = (d[picks] * w[picks]).sum(1) / w[picks].sum(1)
gain_25_50 = rows["users_25"]["mean"] - rows["users_50"]["mean"]
gain_50_100 = rows["users_50"]["mean"] - rows["control"]["mean"]
ratio = gain_50_100 / gain_25_50  # each doubling's gain over the one before
report = {"half_minus_users_50": {"window_weighted_delta": float((d * w).sum() / w.sum()),
                                  "users_better": int((d < 0).sum()), "users": len(users),
                                  "user_bootstrap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]},
          "gain_25_to_50": gain_25_50, "gain_50_to_100": gain_50_100, "doubling_ratio": ratio,
          "power_law_exponent": float(-np.log2(ratio)),
          "projected_gain_100_to_200": gain_50_100 * ratio, "projected_gain_200_to_400": gain_50_100 * ratio ** 2,
          "projected_gain_all_further_doublings": gain_50_100 * ratio / (1 - ratio)}
(HERE/"v10_users_scaling.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
print(json.dumps(report, indent=1))
