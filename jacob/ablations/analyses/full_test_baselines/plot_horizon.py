"""Redraw the paper's per-step error figure from full_test_summary.json, with the damped prior added.

Same curves and styles as Authors/figures/horizon_errors_final.png (Nic's PC run), plus one line for the damped
prior alone. Usage: python plot_horizon.py OUTPUT.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

summary = json.loads((Path(__file__).parent / "full_test_summary.json").read_text(encoding="utf-8"))["predictors"]
CURVES = [  # key in the summary, legend label, colour, line style, line width
    ("model", "k8-combined, final (7,555 parameters)", "#c0392b", "-", 2.6),
    ("damped_velocity", "Damped prior alone", "#6c3483", (0, (5, 1.5, 1, 1.5, 1, 1.5)), 1.8),
    ("hold_last", "Hold last", "#1f4e9c", "--", 1.8),
    ("constant_velocity_1step", "Constant velocity, 1 step", "#e08214", "-.", 1.8),
    ("constant_velocity_5step", "Constant velocity, 5 steps", "#2e8b57", ":", 2.2),
]
steps_ms = [10 * k for k in range(1, 21)]

fig, ax = plt.subplots(figsize=(6.3, 3.0))
for key, label, colour, style, width in CURVES:
    ax.plot(steps_ms, summary[key]["mean_deg_by_step"], color=colour, linestyle=style, linewidth=width, label=label)
ax.axvline(200, color="0.6", linewidth=1)
ax.text(197, 0.25, "scored step", ha="right", va="bottom", fontsize=8, color="0.4")
ax.set_xlim(0, 205)
ax.set_ylim(0, 8)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_xlabel("Time ahead (ms)")
ax.set_ylabel("Mean error (degrees)")
ax.grid(color="0.92")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc="upper left", handlelength=3.2)
fig.tight_layout(pad=0.4)
fig.savefig(sys.argv[1], dpi=300)
print("wrote", sys.argv[1])
