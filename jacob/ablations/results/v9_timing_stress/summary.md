# Timing jitter and rate offsets: summary for the team (29 Oct 2026)

**Why we ran this.** On 28 Oct the organisers said the test stream is "roughly ~100 Hz (not exactly 100 Hz)" and has
"some jitter". Our data is an exact 100 Hz grid. These runs measure what uneven timing costs, and whether training
with jitter on top of the k8_combined recipe pays off.

**Setup**
- All numbers are validation results, seed 42, step-20 mean error in degrees.
- The stress rows come from `timing_stress.py` at stride 5: `step1_existing.md` and `step3_screen.md`.
- The paired rows use full validation and are paired per user against `v9_confirm/k8_combined/seed_42`.

## 1. Rate offsets need nothing

- At 90–110 Hz, 20 samples span 182–222 ms, so the raw deltas mostly reflect a longer or shorter horizon. The damped
  baseline moves the same way: +0.54 at 90 Hz and −0.48 at 110 Hz.
- k8_combined's error divided by the damped baseline's is flat:
  - 0.837–0.839 at every rate from 90 to 110 Hz
  - 0.841 on the exact grid
- The model ranking is the same at every rate. A constant rate offset doesn't hurt the learned model beyond the change
  in the task itself.
- **No time-scale augmentation is needed.**
- On the plan's literal threshold: 98 Hz shows +0.047 for k8_combined, which is above 0.03. The baseline, though, gets
  +0.078 worse on the same stream.

## 2. Jitter is the real risk, and jitter training on k8_combined works

| | k8_combined | + jitter 1.5 ms | + jitter 3 ms |
| --- | ---: | ---: | ---: |
| Exact grid, full validation, paired vs k8_combined | 4.2471 | +0.0090 (4/15 users better, CI [−0.008, +0.024], p 0.29) | +0.0320 (3/15, CI [+0.007, +0.058], p 0.04) |
| Equal-user delta | — | +0.015 | +0.046 |
| Cost at ±0.5 ms jitter | +0.012 | +0.002 | −0.001 |
| Cost at ±1 ms | +0.048 | +0.017 | +0.004 |
| Cost at ±1.5 ms | +0.099 | +0.043 | +0.013 |
| Cost at ±2 ms | +0.157 | +0.078 | +0.028 |
| Cost at ±3 ms | +0.283 | +0.171 | +0.073 |
| Cost at 1% repeats / 1% drops | +0.043 / +0.063 | +0.040 / +0.057 | +0.032 / +0.054 |
| **Break-even vs k8_combined** (uniform jitter) | — | **±0.5 ms** (std 0.3 ms) | **about ±0.9 ms** (std 0.5 ms) |
| 28 Oct rule (grid Δ ≤ +0.01 and ±2 ms cost ≤ half of k8_combined's 0.157) | — | passes, narrowly on both (0.0090; 0.078 vs 0.079) | fails on the grid |

- How to read the variant columns: each cost row is that model's change from its own exact-grid score.
- Each model minus k8_combined on the same stream:
  - 1.5 ms: −0.023 at ±1 ms, −0.071 at ±2 ms, −0.103 at ±3 ms
  - 3 ms: −0.013, −0.098 and −0.179 at the same levels
- On the grid, the combined recipe and jitter stack: combined + 3 ms is 0.043 better than bare k8 + 3 ms, with the
  same jitter robustness.
- Two users drive most of the grid cost in both variants:
  - daa4c4dd, the 12° outlier: +0.20 (1.5 ms) and +0.37 (3 ms)
  - 4b681b4c: −0.14 and −0.19, the other way
- The typical user is about +0.01–0.03 worse with 1.5 ms.

## 3. The decision for the team

- **The trade-off is lopsided.** If the test stream is a perfect grid, combined + 1.5 ms loses 0.009°. If the
  jitter is ±1 ms it gains 0.023°, and at ±2 ms it gains 0.071°. The organisers said there is "some jitter", so a
  perfect grid is the least likely case.
- **The 3 ms variant is the bigger bet.** It loses 0.032° on a perfect grid and wins from about ±0.9 ms upwards. It
  gains the most at large jitter.
- **Caveats:**
  - Each variant is one seed.
  - The 1.5 ms variant passes both parts of the rule by under 0.001.
  - We don't know the real jitter size or its shape. We modelled it as uniform per-sample timing noise.
- **If the team picks a variant, the next step is plan Step 4:** confirm it on seeds 43 and 44 (about 2 h), then pair
  it against k8_combined's 3-seed mean of 4.2501.
