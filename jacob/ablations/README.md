# Ablation record: Jacob's TCNN, v5 to k8_combined and after (26–31 Oct 2026)

> Working log, kept as written. "Pending" and "team call" describe the state at the time. The submitted model and its recipe are in [`model_card.md`](../../Submission/Motion_Pending_headpose/model_card.md).

This is every experiment run on Jacob's TCNN line, with the evidence behind each one. The paper's ablation section draws on it. Nic's GRU and GADS-T work is logged separately (see [Nic's work](#nics-work)).

| Path | What |
| --- | --- |
| `runs.csv` | One row per suite, variant and seed (97 runs). Written by `harvest.py`. |
| `results/<suite>/` | Each suite's reports, configs and per-seed training histories, copied from the gitignored `jacob/training/outputs/` by `harvest.py` |
| `results/logs/` | The run logs |
| `analyses/` | Analyses run outside the trainer: paired per-user tests, robustness, latency and graph work. Rescued from temporary session folders; see `analyses/README.md`. |
| `harvest.py` | Re-run after every new suite: `python jacob/ablations/harvest.py` (training venv) |

In the public copy, machine-specific paths in scripts, logs and records are replaced by placeholders: `<repo>` (the repository root), `<scratch>` (a temporary session folder), `<local_test>` (the earlier local workspace) and `<venv>` (the training environment). Per-user scores are removed.

## How to read it

- **Metric.** The error is the mean geodesic angle at step 20, which is 200 ms ahead and the only step the challenge scores. It is measured on the 15 validation users, at stride 1 (657,189 windows), in degrees, and lower is better.
- **Splits.** Every number here is validation unless it says test.
  - Test is the team bench `group-a-v2-test20`: 20 unseen users, 931,051 windows. It is reported, never used to choose.
  - The split is group-a-v2: 100 train, 15 validation and 20 test users, split by user.
- **Seeds.** A *screen* trains seed 42 only. A *confirm* trains seeds 42, 43 and 44 and reports mean ± sample SD. A screen's control always reproduced the previous confirm's seed-42 number exactly, so screen deltas are paired against the same seed.
- **Seed noise.**
  - The three v6 control seeds span 0.023° (SD 0.012), so a single-seed Δ under about 0.02° is not resolved.
  - The SDs of later confirms are 0.004–0.014.
- **Paired per-user tests.** "Users better" counts the validation users whose own mean error improved. CIs are user-level bootstrap 95% intervals. Scripts and outputs are in `analyses/paired_tests/`.
- **Rules in force.**
  - v7 tolerance: a 3-seed mean ≤ 4.382°, i.e. v6 control + 0.04°.
  - v9 screen rule: Δ < 0 with most users better.
  - Jitter rule: an exact-grid Δ ≤ +0.01°, and its own ±2 ms cost no more than half of k8's.
- **Baselines** (validation):
  - Hold-last: 6.0826°.
  - Damped velocity (lag 1, τ 100 ms, tuned on validation): 5.0492°.
  - On test: 6.037° and 5.114°.
- **Code.**
  - Every run to 29 Oct used the code at git tag `v9-ablations` (b796dc4); `runs.csv` names the package each run hashed.
  - The hashes in each `experiment.json` are of LF files.
  - The packages are now in `jacob/training/archive/`, and their code is `jacob/training/best/`.
  - The config paths below are relative to `jacob/training/`.

## Lineage

| Step | Model | Validation, 3 seeds | Test (selected seed) | Params | What changed |
| --- | --- | ---: | ---: | ---: | --- |
| v5 | control | 4.4049 ± 0.0042 | 4.4534 | 51,900 | Dilated causal TCNN (7 blocks, 48 channels) on body-frame velocity, predicting a residual on the damped-velocity prior |
| v6 | control | 4.3423 ± 0.0121 | 4.3584 | 26,012 | + gravity in the head frame, + exact left-right mirror equivariance (tied weights halve the parameters) |
| v7 | separable_base3_addons | 4.3829 ± 0.0119 | 4.4186 | 6,651 | Compression: separable blocks, base-3 dilations (5 blocks), L1 speed, 3-term polynomial head |
| v8 | sep_base3_addons + angular | 4.3297 ± 0.0142 | 4.3289 | 6,651 | Angular (geodesic) loss in place of the quaternion loss |
| v9 | **k8_combined** | **4.2501 ± 0.0041** | **4.2666** | 7,555 | 3 blocks of kernel 8 (receptive field 393), window sampling, native maps, 15° tilt augmentation |

From v5 to k8_combined:
- −0.155° on validation and −0.187° on test.
- 85% fewer parameters.
- The lean graph runs at 55 µs, against v5's 92 µs. Both are ORT CPU, 1 thread, on jacob-amd, timed in different sessions, so they are relative only.

## Experiments

### 1. v5 suite (26 Oct)
- **Question:** does the v5 TCNN, moved onto group-a-v2, beat the baselines, and do a narrower width or a shorter context cost accuracy?
- **Change:** `width_32` (32 channels) and `short_128` (effective context 128, 6 blocks), against the control.
- **Seeds:** screen on 42; the control was confirmed on 42–44.
- **Result:**

  | Variant | Error (seed 42) | Δ |
  | --- | ---: | ---: |
  | control | 4.4057 | 0 |
  | short_128 | 4.4345 | +0.029 |
  | width_32 | 4.4354 | +0.030 |

  The control's 3-seed mean is 4.4049. On test it scores 4.4534, against damped velocity's 5.114.
- **Verdict:** control kept. The 26 Oct gate ("beats constant velocity") was passed.
- Config: `configs/v5_run.json` on `configs/tcnn_v5.json`. Results: `results/v5_group_a_v2/`.

### 2. v6 suite: gravity and mirror equivariance (26 Oct)
- **Question:** do world-up gravity in the head frame and exact mirror equivariance help?
- **Seeds:** screen on 42; the control was confirmed on 42–44.
- **Result** (seed 42):

  | Variant | Error | Δ | Params |
  | --- | ---: | ---: | ---: |
  | control | 4.3511 | 0 | 26,012 |
  | no_gravity | 4.3685 | +0.017 | 25,940 |
  | narrow (32 ch) | 4.3795 | +0.028 | 11,972 |
  | no_mirror | 4.3803 | +0.029 | 52,044 |
  | plain (both off) | 4.4057 | +0.055 | 51,900 |

  - `plain` reproduces v5's control exactly.
  - The control's 3-seed mean is 4.3423. On test it scores 4.3584, and beats v5 on 18/20 users.
  - `narrow` stopped early (best epoch 31), partly from undertraining.
- **Verdict:** both adopted. Mirror equivariance also halves the parameter count.
- Config: `archive/v6_gravity_mirror/configs/v6_run.json`. Results: `results/v6_group_a_v2/`.

### 3. M1, M2, E1: latency and head-basis measurements, no training (26 Oct)
- **M1:** on ORT CPU, latency follows the node count (about 0.75 µs a node), not MACs. v5 has 133 nodes and v6 174, at the same ~52K MACs.
- **M2:** an SVD of the trained head shows that a polynomial basis t, t², t³ (P = 3) costs +0.001°.
- **E1:** v6 control through the operator-reduced v7 graph has parity 7e-5° and is 13% faster. Only `Sqrt` is left off the safe list.
- **Verdict:** this led to the v7 plan: cut nodes, not only MACs.
- Results: `analyses/v7_compression/`.

### 4. v7 compression screen (26–27 Oct)
- **Question:** which compression options keep accuracy, under matched budgets with patience ≥ epochs?
- **Seeds:** screen on 42. The control reproduces v6 seed 42 (4.3511).
- **Result:**

  | Variant | Error | Δ | Params |
  | --- | ---: | ---: | ---: |
  | speed_l1 | 4.3378 | −0.013 | 26,012 |
  | basis_head | 4.3407 | −0.010 | 24,771 |
  | base3_5 | 4.3433 | −0.008 | 19,052 |
  | separable_24 | 4.3550 | +0.004 | 10,388 |
  | width_20 (40 ch) | 4.3573 | +0.006 | 18,320 |
  | narrow_full (32 ch) | 4.3777 | +0.027 | 11,972 |
  | narrow_base3 | 4.4119 | +0.061 | 8,868 |

  - narrow_base3 fails the 4.402 screen bar.
  - The full metrics are in `results/v7_metrics/metrics.md`: horizons, percentiles, equal-user, worst user, nodes and latency.
  - The basis head costs the short horizons, +0.017° at step 5.
- **Verdict:** the add-ons (L1 speed + basis head) were combined with base3_5 and with separable + base 3.
- Config: `archive/v7_compact/configs/v7_run.json`. Results: `results/v7_group_a_v2/`, `results/v7_metrics/`.

### 5. v7 extra screen and v7 confirm (27 Oct)
- **Question:** do the combined compact models stay within the 4.382° tolerance?
- **Result** (3 seeds):

  | Candidate | Params | Seeds 42 / 43 / 44 | Mean ± SD |
  | --- | ---: | --- | ---: |
  | base3_5_addons | 17,811 | 4.3802 / 4.3561 / 4.3655 | 4.3673 ± 0.0121 |
  | separable_base3_addons | 6,651 | 4.3855 / 4.3699 / 4.3933 | 4.3829 ± 0.0119 |

  - base3_5_addons passes. separable_base3_addons misses by 0.0009°.
  - The add-ons don't stack: base3_5_addons scores 4.380 at seed 42, against 4.343 for plain base3_5.
- **Verdict:** base3_5_addons is the plan-rule finalist. Both went on to the v8 recipe work.
- Config: `archive/v7_compact/configs/{v7_extra,v7_confirm}.json`. Results: `results/v7_extra/`, `results/v7_confirm/`.

### 6. E3: int8 post-training quantisation (26–27 Oct)
- **Question:** what does int8 cost?
- **Setup:** v6 control seed 44 through the v7 graph. The float stream reproduces 4.3285. Each scheme's cost on all 657K validation windows:

  | Scheme | Cost |
  | --- | ---: |
  | int8 minmax | +0.021° |
  | int8 percentile | +0.012° |
  | w8a16 | +0.006° |

  The latency in that run is unreliable, because a screen was training at the same time.
- **Verdict:** QAT isn't needed under a 0.02° rule. **Not yet run on k8_combined** (`python -m best.quantize`, see `best/README.md`).
- Results: `results/v7_e3_int8/`.

### 7. Mirror-split export (27 Oct)
- **Question:** can the mirror block's [[A, B], [B, A]] Conv run at half the MACs?
- **Change:** (A+B)/2 on h0 + h1, and (A−B)/2 on h0 − h1.
- **Result** on sep+angular:
  - MACs fall from 13,863 to 8,103, with identical accuracy (Δ 1e-9 on test).
  - It adds 20 nodes (130 → 150), and ORT CPU latency rises from 87.0 to 98.5 µs (+13%).
- **Verdict:** pending the ST hardware timings. It is ported to the lean graph as `split_mirror` in `best/export.py`, but untimed on k8.
- Results: `results/v8_bench/split_latency.json`, `analyses/bench/split_latency.py`.

### 8. A1: generalisation gap (27 Oct)
- **Question:** is the model capacity-limited, or memorising?
- **Setup:** v6 control seed 44, compared as model/damped error per user.
- **Result:**
  - Train users score 0.805 against validation's 0.846: +0.041, CI [+0.013, +0.069], permutation p 0.09.
  - The gap sits in the small train users, which user-balanced sampling re-draws about 21 times a window. They score 0.750.
  - Big users, at 1.2 draws a window, score 0.839.
- **Verdict:** this reads as memorisation of oversampled users. It led to the sampling options tested in v9.
- Results: `analyses/a1_generalisation_gap/`.

### 9. v8 recipe screen (27 Oct)
- **Question:** which config-only recipe changes help?
- **Setup:** run on both v7 finalists, seed 42, paired against each one's control.
- **Result:**

  | Variant | On separable (control 4.3855) | On base3_5 (control 4.3802) |
  | --- | ---: | ---: |
  | angular_loss | **−0.047** (13/15 users) | **−0.046** (10/15) |
  | epochs_80 | −0.019 | −0.003 |
  | lr_2e3 | −0.011 | +0.034 (0/15) |

  - The angular loss leads at every epoch and improves the median and 100 ms error.
  - It costs about +0.14° at p99.
- **Verdict:** the angular loss went to confirm. 80 epochs is within seed noise, and LR 2e-3 was dropped.
- Config: `archive/v7_compact/configs/v8_recipe_screen.json` on `tcnn_v7_{sep_base3,base3_5}_addons.json`. Results: `results/v8_recipe_*`, `analyses/paired_tests/v8_recipe_screen_paired.*`.

### 10. v8 angular-loss confirm (27 Oct)
- **Result:**

  | Model | Before | With angular loss (3 seeds) |
  | --- | ---: | ---: |
  | separable | 4.3829 | 4.3297 ± 0.0142 |
  | base3_5 | 4.3673 | 4.3247 ± 0.0104 |

  - All 6 of 6 seeds improve: mean −0.048°, p = 0.001.
  - The 100 ms error improves by 0.031°, and p99 gets about 0.09° worse.
- **Verdict:** adopted. The plan rule picked sep+angular, which is within tolerance and has 6,651 params. It was the best model until v9.
- Config: `archive/v7_compact/configs/v8_angular_confirm.json`. Results: `results/v8_angular_confirm/`, `analyses/paired_tests/angular_confirm_paired.*`.

### 11. Frame robustness (27–28 Oct)
- **Question:** what if the test stream's world frame differs from BOXRR's?
- **Setup:** each convention applied to whole validation segments, at stride 5.
- **Result:**
  - Nothing changes, for any model, under any heading, heading-only re-centring, or (for mirror models) a mirror.
  - Gravity models: +0.06 to +0.08 at 15° tilt, and +0.02 to +0.04 under full first-pose re-centring. A z-up world costs +0.9 to +1.7, worse than damped velocity.
  - A handedness flip costs every learned model +0.12 to +0.18.
  - Estimating gravity from the stream can't recover posture: a 4 s mean head-up is still 8.4° off.
- **Verdict:** on 28 Oct the organisers said they will feed test data in our documented frame, so gravity is kept. The frame spec must be exact.
- Results: `analyses/frame_robustness/`.

### 12. v8 no-gravity hedge (28 Oct)
- **Result:** sep+angular without gravity scores 4.3705 ± 0.0086 (3 seeds), against 4.3297 with it.
  - So gravity is worth **0.041°**: every seed and 14/15 users are worse without it.
  - The exception is user daa4c4dd. Their posture sits 26° back, and gravity costs them 0.94°.
- **Verdict:** gravity is kept. The v6 single-seed estimate (+0.017) was low.
- Config: `archive/v7_compact/configs/v8_no_gravity_confirm.json`. Results: `results/v8_no_gravity/`.

### 13. v8 bench and Nic comparison (27 Oct, test)
- **Result:**
  - sep+angular scores 4.3289 on test: −0.030 against v6 (CI excludes 0, 16/20 users).
  - Against Nic's 5-seed means: −0.138° against the GRU (19/20) and −0.177° against GADS-T.
  - The GRU is about 4× faster on ORT CPU (23 against 89 µs) at 6,384 MACs.
- Results: `results/v8_bench/` (`benchmark_comparison.md`, `nic_comparison.md`).

### 14. Track A: the lean streaming graph (28 Oct)
- **Question:** can the same weights run as a cheaper graph?
- **Result** (sep+angular weights):
  - The lean graph has 84 nodes against the v7 graph's 130, and runs at 70.5 against 87.6 µs, with the same 6,648 stored weights. Accuracy changes by about 1e-6°.
  - The padded-rows option is 15.5% faster again, measured on a contended CPU.
  - A4 (fused head): 81 nodes, 68.6 µs.
- **Verdict:** the lean graph was adopted as the default export. Padded rows and A4 wait on the ST timings.
- Results: `analyses/track_a_graph/`.

### 15. v9 screen: layouts and recipe options (28 Oct)
- **Setup:** seed 42 on sep+angular. The control reproduces v8 exactly (4.3386).
- **Result:**

  | Variant | Error | Δ | Users better | Params |
  | --- | ---: | ---: | :---: | ---: |
  | k5_4blocks | 4.2898 | −0.049 | 15/15 | 5,595 |
  | k8_3blocks_w32 | 4.2911 | −0.048 | 15/15 | 7,555 |
  | sampling_window | 4.3094 | −0.029 | 14/15 | 6,651 |
  | rf_397 | 4.3098 | −0.029 | 13/15 | 6,699 |
  | sampling_sqrt | 4.3109 | −0.028 | 13/15 | 6,651 |
  | tilt_15 | 4.3289 | −0.010 | 8/15 | 6,651 |
  | rf_324 | 4.3304 | −0.008 | 7/15 | 6,675 |
  | native_maps | 4.3314 | −0.007 | 8/15 | 6,651 |
  | endpoint_1 | 4.3315 | −0.007 | 12/15 | 6,651 |

  - A longer receptive field is the big gain, and it costs fewer graph nodes: 60 for k8 against 84.
  - Most of sqrt sampling's gain comes from one outlier user.
- **Verdict:**
  - k8 and k5 were both taken forward (a size trade-off).
  - Window sampling, native maps (a latency gain) and tilt 15 were combined with them.
  - endpoint_1 and rf_324 were dropped.
- Config: `archive/v9_lean/configs/v9_screen.json` on `tcnn_v9.json`. Results: `results/v9_screen/`. The paired JSON for this screen was overwritten by a later run; the numbers above are from the session notes.

### 16. C0 and C2: ensemble teacher and distillation (28 Oct)
- **Question:** can distillation from an ensemble beat the k8 layout?
- **Setup:** the teacher (C0) is the sign-aligned mean of 9 checkpoints. It scores 4.2725, only about 0.02° better than k8's 4.2911.
- **Result:**

  | β | Error | Δ against k8 |
  | ---: | ---: | ---: |
  | 0.5 | 4.3123 | +0.021 (3/15 users) |
  | 1.0 | 4.3240 | +0.033 |

- **Verdict:** failed. A next try needs a stronger teacher.
- Config: `archive/v9_lean/configs/v9_distill.json`. Results: `results/v9_distill/`, `results/v9_teacher/`, `analyses/distillation/`.

### 17. Jitter robustness and the jitter screen (28 Oct)
- **Question:** the organisers said the test stream has "some jitter". What does it cost, and does augmentation fix it?
- **Cost on validation resampled at jittered times:**

  | Jitter | Learned models | Damped baseline |
  | ---: | ---: | ---: |
  | ±1 ms | +0.05 | +0.002 |
  | ±2 ms | +0.16 to +0.18 | +0.019 |
  | ±3 ms | +0.29 to +0.33 | +0.053 |

- **Screen:** k8 with 3 ms jitter augmentation scores 4.3212: +0.030 against k8 (4/15 users, CI [+0.002, +0.057]). Its own ±2 ms cost is +0.028, against k8's +0.158.
- **Verdict:** it fails the exact-grid rule, so jitter was kept out of the D confirm.
- Config: `archive/v9_lean/configs/v9_jitter.json`. Results: `results/v9_jitter/`, `analyses/jitter_robustness/`.

### 18. D confirm: the combined candidates (28–29 Oct)
- **Setup:** 3 seeds each, against sep+angular (4.3297).
- **Result:**

  | Candidate | Params | Seeds 42 / 43 / 44 | Mean ± SD | Δ | Users better | CI |
  | --- | ---: | --- | ---: | ---: | :---: | --- |
  | **k8_combined** | 7,555 | 4.2471 / 4.2547 / 4.2485 | **4.2501 ± 0.0041** | −0.0796 | 15/15 | [−0.124, −0.047] |
  | k5_combined | 5,595 | 4.2652 / 4.2714 / 4.2785 | 4.2717 ± 0.0067 | −0.0580 | 15/15 | [−0.094, −0.032] |

  - k8 against k5: −0.0216 (14/15 users, CI [−0.032, −0.013]).
  - Seed 42 was selected for both.
  - Frame: 15° tilt now costs +0.028 (k8) and +0.031 (k5), against the reference's +0.059.
  - Jitter: ±2 ms still costs +0.157 (k8) and +0.176 (k5).
- **Verdict:** k8_combined is the best model, and was **promoted to `best/` on 30 Oct**. k8 against k5 is still a team call.
- Config: `archive/v9_lean/configs/v9_confirm.json`. Results: `results/v9_confirm/`, `analyses/paired_tests/d_confirm_paired*`.

### 19. Final latency and v9 bench (29 Oct)
- **Latency** (ORT CPU, 1 thread, jacob-amd, idle machine):

  | Graph | Nodes | Latency |
  | --- | ---: | ---: |
  | v7 graph (reference) | 130 | 90.3 µs |
  | lean (reference) | 84 | 72.0 µs |
  | **k8 lean** | 60 | **55.2 µs** |
  | k8 padded | 54 executed | 48.3 µs |
  | k8 padded + A4 | 51 | 46.9 µs |
  | k5 lean | 69 | 61.9 µs |
  | k5 padded | — | 53.5 µs |

  - k8 has 16.2K MACs and 25,088 state floats. k5 has 12.3K and 19,008.
  - Every graph has parity ≤ 1e-4° and no off-list operators.
- **Test** (bench `group-a-v2-test20`, reporting only):

  | Model | Test | Δ | Users better |
  | --- | ---: | ---: | :---: |
  | k8_combined | 4.2666 | −0.062 against sep+angular | 18/20 |
  | k5_combined | 4.2871 | | |
  | Nic GRU, 5-seed mean | 4.4670 | | |
  | Nic GADS-T, 5-seed mean | 4.5056 | | |

  k8_combined is −0.200 against Nic's GRU, on 20/20 users.
- Results: `results/v9_bench/full_comparison.md`, `analyses/track_a_graph/final_latency*`, `analyses/bench/`.

### 20. Timing stress test and jitter on k8_combined (29 Oct)
- **Question:** are rate offsets, drops and jitter a risk, and is a small jitter augmentation worth it?
- **Result:**
  - **Rate offsets of 90–110 Hz are harmless.** The model-to-damped error ratio stays at 0.837–0.839, against 0.841 on the exact grid.
  - k8_combined's jitter cost:

    | Jitter | ±0.5 ms | ±1 ms | ±1.5 ms | ±2 ms | ±3 ms |
    | --- | ---: | ---: | ---: | ---: | ---: |
    | Cost | +0.012 | +0.048 | +0.099 | +0.157 | +0.283 |

  - Dropping 1% of samples costs +0.063.
  - Jitter augmentation, both at seed 42 against k8_combined's 4.2471:

    | Variant | Error | Δ | Users better | CI | Rule | Break-even |
    | --- | ---: | ---: | :---: | --- | --- | ---: |
    | 1.5 ms | 4.2561 | +0.0090 | 4/15 | [−0.008, +0.024] | passes narrowly | ±0.5 ms |
    | 3 ms | 4.2791 | +0.032 | 3/15 | | fails | ±0.9 ms |

- **Verdict:** pending a team decision. A picked variant needs seeds 43 and 44.
- Config: `archive/v9_lean/configs/v9_jitter_combined.json`. Results: `results/v9_timing_stress/` (`summary.md`), `results/v9_jitter_combined/`, `analyses/paired_tests/v9_jitter_combined_paired.*`, `analyses/timing_stress/`.

### 21. Promotion to `best/` (30 Oct)
- The v6, v7 and v9 code was merged into `jacob/training/best/`, and the packages were archived.
- Every check against v9 came out exact:
  - 288 config combinations normalise identically.
  - The k8, k5 and sep+angular checkpoints give identical outputs.
  - The k8 ONNX exports are byte-identical to `jacob/models/`.
  - A training run is bit-identical.
  - Full validation reproduces 4.247104990427709.
- Results: `analyses/promotion/`.

### 22. v10_users: training-user scaling (31 Oct)
- **Question:** k8_combined's training and validation error fall together, so it is limited by data, not memorising. Would more training users help, and is it the number of users (diversity) or the number of windows (volume) that counts?
  - Grokking and Grokfast were judged unfit for the same reason: there is no memorisation phase.
- **Change:** `experimental/v10_users/` adds two keys.
  - `train_users`: a nested subset, ordered by SHA-256 of each user name.
  - `user_window_fraction`: the leading fraction of each user's windows.
  - Validation keeps all 15 users.
  - The 1-epoch parity run with every user reproduced k8_combined's seed-42 epoch 1 exactly.
- **Setup:** seed 42, the same 40k steps for every variant. The control is k8_combined seed 42 (4.2471, 100 users, 3,844,021 windows).
- **Result:**

  | Variant | Users | Windows | Error | Δ | Users better | CI |
  | --- | ---: | ---: | ---: | ---: | :---: | --- |
  | users_25 | 25 | 20.8% | 4.3272 | +0.080 | 0/15 | [+0.039, +0.137] |
  | users_50 | 50 | 44.3% | 4.2763 | +0.029 | 2/15 | [+0.009, +0.051] |
  | users_100_half | 100 | 50.0% | 4.2683 | +0.021 | 4/15 | [+0.006, +0.040] |

  - **Scaling:** doubling the users gained 0.051 (25 → 50) and then 0.029 (50 → 100). That is a ratio of 0.57 per doubling, a power-law exponent of about 0.8.
    - Extrapolated from one seed, a further doubling to 200 users would gain about 0.017.
    - All further doublings together would gain about 0.039.
  - **Diversity or volume:** users_100_half against users_50 is −0.008, with 10/15 users better and CI [−0.023, +0.009]. It also has 13% more windows, so the two aren't resolved.
    - Halving the windows of every user costs about as much as halving the users.
    - The leading half also drops each user's later sessions, so users_100_half loses session variety too.
  - Median, p90, p99 and the 100 ms error all move the same way.
- **Verdict:** k8 is still data-limited at 100 users, with diminishing returns. More BOXRR users would add both users and windows. Expect roughly −0.015 to −0.02 for 200 users: about the size of the other remaining levers, and within two seed SDs of noise per doubling. Whether to download is a team call.
  - Training on more users would also take k8 off the leaderboard's same-data comparison with Nic's models.
- Config: `experimental/v10_users/configs/screen.json`. Results: `results/v10_users/`, `analyses/paired_tests/v10_users_paired.*` and `v10_users_scaling.*`.

### 23. v10_loo: leave-one-out on k8_combined (6–8 Nov)
- **Question:** what does each component add to the final model, measured on that model and on the same seeds? The component effects below were each measured on a different base, mostly on one seed.
- **Change:** none. It is configs only, on `best/` unchanged, in `experimental/v10_loo/`, with the 100-user file.
  - It runs the suite suggested under [Gaps for the paper](#gaps-for-the-paper), plus **no_prior**.
  - no_prior sets `damped_prior_tau_ms: null`, so the residual corrects hold last instead of the damped prior. No earlier run had ablated the prior.
- **Setup:**
  - Screen: seed 42 for the control and 8 variants. Confirm: seeds 43 and 44.
  - Each seed is paired with k8_combined's own seed from exp. 18: 4.2471 / 4.2547 / 4.2485.
  - The control reproduced seed 42 exactly: 4.247104990427709, all 40 epochs.
- **Result** (Δ = variant − k8_combined, the mean of the three seed-paired differences; positive means removing the component hurts):

  | Removed | Override | Params | Seeds 42 / 43 / 44 | Mean ± SD | Δ | SD of Δ | Seeds worse |
  | --- | --- | ---: | --- | ---: | ---: | ---: | :---: |
  | Gravity | `gravity_features: false` | 7,459 | 4.2942 / 4.3067 / 4.3009 | 4.3006 ± 0.0063 | **+0.0505** | 0.0030 | 3/3 |
  | Angular loss | `loss_kind: quaternion` | 7,555 | 4.3058 / 4.2855 / 4.2980 | 4.2964 ± 0.0103 | **+0.0463** | 0.0143 | 3/3 |
  | Mirror equivariance | `mirror_equivariant: false` | 15,113 | 4.2740 / 4.2817 / 4.2692 | 4.2750 ± 0.0063 | **+0.0249** | 0.0035 | 3/3 |
  | Window sampling | `sampling: user` | 7,555 | 4.2790 / 4.2763 / 4.2692 | 4.2748 ± 0.0051 | **+0.0247** | 0.0062 | 3/3 |
  | k8 layout (5 blocks, k = 3, field 243 instead) | `rf_243` | 11,427 | 4.2625 / 4.2765 / 4.2694 | 4.2695 ± 0.0070 | **+0.0194** | 0.0035 | 3/3 |
  | Tilt augmentation | `tilt_augmentation_deg: 0` | 7,555 | 4.2469 / 4.2503 / 4.2511 | 4.2494 ± 0.0022 | −0.0007 | 0.0036 | 1/3 |
  | Damped prior | `damped_prior_tau_ms: null` | 7,555 | 4.2619 / 4.2405 / 4.2466 | 4.2497 ± 0.0110 | −0.0004 | 0.0145 | 1/3 |
  | Native maps (exact instead) | `maps: exact` | 7,555 | 4.2419 / 4.2455 / 4.2493 | 4.2456 ± 0.0037 | −0.0045 | 0.0051 | 1/3 |

  - **Paired per-user tests have not been run**, so there are no users-better counts or CIs yet.
  - no_mirror and rf_243 are not size-matched. Each needs more parameters to do worse.
- **Verdict:**
  - Five components help on every seed, by 0.019–0.050:
    - gravity
    - the angular loss
    - mirror equivariance
    - window sampling
    - the k8 layout
  - Three are within noise on validation accuracy:
    - **Tilt:** expected. It is there for frame robustness (exp. 11, 18), which this suite doesn't measure.
    - **The damped prior:** it gains nothing on average and has the largest seed-to-seed spread (paired SD 0.015). The Method section presents it as a design element.
    - **Native maps:** exact maps are slightly better, if anything. Native maps were kept for the graph: no trigonometric operators, 6 fewer nodes (exp. 15).
  - Whether to keep the prior and the native maps, and how the paper words them, is a team call.
  - The paper's Table "Effect of Each Component" was switched to this suite on 8 Nov. The mixed-base version is kept in `ABC_Challenge_paper/ablations_ledger.tex`.
- **Run notes:** about 20 GPU-hours on the 3080. Each interruption was resumed from finished seeds or epoch checkpoints, and the trainer's resume is exact (`tests/test_tcnn.py`).
  1. On 6 Nov at 23:19, something outside the run deleted most of the training venv. A fresh venv was built from the root `requirements.txt` and reproduced the control bit for bit (`results/v10_loo_envcheck/`).
  2. On 7 Nov at 02:44, the PC was put to sleep manually.
  3. On 7 Nov at 21:20, a `CUDA error: unknown error` in rf_243 seed 43, followed by an unclean reboot with no bugcheck.
     - The partial seed is kept as `seed_43_crashed_*` and was retrained fresh.
     - Its epochs 1–2 matched the crashed copy bit for bit.
     - GPU at most 80 °C and 312 W afterwards.
  - Details: `experimental/v10_loo/README.md`.
- Config: `experimental/v10_loo/configs/screen.json`, `confirm.json`. Results: `results/v10_loo/`, `results/v10_loo_confirm/` (`confirmation.md`), logs `results/logs/v10_loo_*`.

### 24. v10_fast: a forward that skips the steps nobody reads (9 Nov)
- **Question:** a code review of `best/` found that `LeanTCNN.forward` runs every block at all 400 time steps and reads only the last. Can training and validation skip the rest without changing the model?
- **Change:** `experimental/v10_fast/`. `FastLeanTCNN` runs block 3 at 1 step, block 2 at its 8 taps and block 1 at their 64: 73 block evaluations a window in place of 1,200.
  - The parameters, checkpoints and dropout masks are LeanTCNN's.
  - `best/` is unchanged, and nothing is promoted.
- **Same function:**
  - k8_combined on its full validation: 4.247104990° with both forwards. The largest difference in any forecast component is 1.8e-7.
  - The 800-user model (seed 44) on its 4,755,761 validation windows: 4.403628580° with both, as recorded on 2 Nov. The largest difference is 1.1e-6.
  - One training step on the GPU with dropout on and the same seed: the losses differ by 9e-8 of their value.
- **Same training, within seed noise.** Training is not bit-identical, because the sums are rounded in another order. k8_combined was retrained on the 100-user file with the new forward:

  | Forward | Seeds 42 / 43 / 44 | Mean ± SD |
  | --- | --- | ---: |
  | `best/` (exp. 18) | 4.2471 / 4.2547 / 4.2485 | 4.2501 ± 0.0041 |
  | v10_fast | 4.2390 / 4.2394 / 4.2513 | 4.2432 ± 0.0070 |

  - The difference is −0.0069°, with two seeds lower and one higher. A Welch test gives p = 0.23, so three seeds show no difference. They can't rule out one of a few thousandths of a degree.
  - Paired per-user tests were not run.
- **Speed** (3080, same deterministic settings):
  - A training epoch on the 100-user file takes 15.0 s in place of 69.5 s (4.6×).
  - A full validation pass of the 800-user model takes 23 s in place of 348 s.
  - Peak GPU memory is 0.8 GB in place of 4.8 GB.
  - Projected for a full-pass seed on 800 users: about 35 minutes of training in place of 2.8 hours. Not yet run.
- **Verdict:** usable for screens, with one rule: compare a v10_fast run with a v10_fast control, not with an older run of the same seed. Whether the submitted model may be trained with it is a team call.
- **Also from the review** (all in `best/`, none in the hashed files):
  - `export.py` has a new `st_layout` option, off by default. It is the LayerNormalization layout that ST Edge AI Core 4.1 compiles correctly. ONNX Runtime's output is bit-identical with it, and the default export is still byte-identical to `jacob/models/`.
  - `confirm.py` no longer crashes when the reference has no `selection.json`.
  - `configs/screen.json` now carries the damped-baseline grid that a screen needs.
- **Not run:** `experimental/v10_fast/configs/screen_800.json`, a seed-42 screen on the 800-user file: dropout 0, 2 passes, 1.5 ms jitter and 96 channels. Dropout has been 0.1 since v3 and was never ablated.
- Config: `experimental/v10_fast/configs/check100_confirm.json`. Results: `results/v10_fast_check100/`, `results/v10_fast_check100_confirm/` (`confirmation.md`), logs `results/logs/v10_fast_check100_*`. Details: `experimental/v10_fast/README.md`.

### 25. Sensor-noise stress test (9 Nov)
- **Question:** jitter was measured in exp. 17 and 20. Per-sample noise in the poses never was, and it corrupts the same thing: the differences between neighbouring samples. What does it cost?
- **Setup:** a `noise_deg` scenario in `best/timing_stress.py`. Every sample is turned by its own random rotation, whose angle has the given RMS, independent from sample to sample.
  - The 15 validation users at stride 5 (131,459 windows). The target is the noisy stream's 20th next sample, so the damped baseline's rise is what any model would pay.
  - The exact-grid and jitter rows reproduce the 29 Oct numbers to four decimals.
- **Result** (each model's change from its own exact-grid error):

  | Stream | Damped baseline | k8_combined | k8, 800 users | Jitter 1.5 ms | Jitter 3 ms |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Exact grid (error) | 5.0497 | 4.2478 | 4.1812 | 4.2566 | 4.2793 |
  | Noise 0.005° | +0.0005 | +0.0031 | +0.0041 | +0.0021 | +0.0014 |
  | Noise 0.01° | +0.0019 | +0.0123 | +0.0158 | +0.0083 | +0.0056 |
  | Noise 0.02° | +0.0075 | +0.0458 | +0.0545 | +0.0318 | +0.0218 |
  | Noise 0.05° | +0.0463 | +0.2095 | +0.2121 | +0.1636 | +0.1211 |
  | Jitter ±1 ms | +0.0010 | +0.0480 | +0.0532 | +0.0168 | +0.0038 |
  | Jitter ±2 ms | +0.0183 | +0.1573 | +0.1551 | +0.0778 | +0.0277 |

  - Up to 0.02° the cost grows about fourfold for each doubling of the noise. 0.02° costs k8 about what ±1 ms of jitter does, and 0.05° more than ±2 ms.
  - The 800-user model is no more robust to noise or jitter than k8_combined. It keeps its 0.06° lead on every stream.
  - The jitter-trained models are also more robust to noise. At 0.02° the 1.5 ms variant loses 31% less than k8 and the 3 ms variant 52% less. Counting its exact-grid cost, the 1.5 ms variant overtakes k8 at about 0.015° of noise.
- **Verdict:** noise is a second exposure of the same kind as jitter, and jitter augmentation covers part of it. That adds to the case for a jitter variant in decision D4. How noisy the organisers' stream is isn't known.
- Results: `results/timing_stress/v10_noise_100.md`.

## Component effects, for the Ablations section

Each Δ is validation step-20 error with the component added (or removed, as marked), on the base where it was measured. The bases differ, so the effects don't add up exactly.

**For the final model, use exp. 23 instead.** It removes each component from k8_combined on seeds 42–44. The paper's table has used it since 8 Nov. The rows below are the history of how each component was chosen.

| Component | Measured by | Base | Seeds | Δ |
| --- | --- | --- | :---: | ---: |
| Learned TCNN over the damped-velocity prior | v5 against the damped baseline | — | 3 | −0.644 (4.4049 against 5.0492) |
| Gravity (removed) | v6 no_gravity | v6 control | 1 | +0.017 |
| Gravity (removed) | v8 no-gravity confirm | sep+angular | 3 | **+0.041** |
| Mirror equivariance (removed) | v6 no_mirror | v6 control | 1 | +0.029, at double the params |
| Gravity and mirror (both removed) | v6 plain | v6 control | 1 | +0.055 |
| L1 speed channel | v7 speed_l1 | v6 control | 1 | −0.013 |
| Polynomial head, P = 3 | v7 basis_head | v6 control | 1 | −0.010 (−1,241 params) |
| Base-3 dilations, 5 blocks | v7 base3_5 | v6 control | 1 | −0.008 (−6,960 params) |
| Separable blocks | v7 separable_24 | v6 control | 1 | +0.004 (−15,624 params) |
| All four compressions | v7 confirm | v6 control | 3 | +0.041 (−19,361 params) |
| Angular loss | v8 confirm | sep / base3_5 | 3 + 3 | **−0.053 / −0.043** |
| Receptive field 243 → 393 (k8, 3 blocks) | v9 k8_3blocks_w32 | sep+angular | 1 | −0.048 |
| Receptive field 243 → 397 (k5, 4 blocks) | v9 k5_4blocks | sep+angular | 1 | −0.049 |
| Receptive field 243 → 397 (kernel 3, 5 blocks) | v9 rf_397 | sep+angular | 1 | −0.029 |
| Window sampling | v9 sampling_window | sep+angular | 1 | −0.029 |
| Native maps | v9 native_maps | sep+angular | 1 | −0.007 (−6 graph nodes) |
| 15° tilt augmentation | v9 tilt_15 | sep+angular | 1 | −0.010. With it, k8_combined's 15° tilt cost is +0.028, against +0.059 for sep+angular. |
| k8 + window + native + tilt | D confirm | sep+angular | 3 | **−0.080** |
| Distillation, β = 0.5 | v9 distill | k8 layout | 1 | +0.021 |
| Jitter augmentation, 1.5 ms | v9 jitter_combined | k8_combined | 1 | +0.009 on the grid, −0.071 at ±2 ms |
| 80 epochs | v8 epochs_80 | sep / base3_5 | 1 | −0.019 / −0.003 |
| Narrower width | v5 width_32, v6 narrow, v7 narrow_full | each control | 1 | +0.027 to +0.030 |
| Shorter context, 128 | v5 short_128 | v5 control | 1 | +0.029 |
| int8, minmax / percentile / w8a16 | E3 | v6 control seed 44 | — | +0.021 / +0.012 / +0.006 |
| Training users 100 → 50 / 25 | v10 users_50 / users_25 | k8_combined | 1 | +0.029 / +0.080 |
| Half of each user's windows (100 users) | v10 users_100_half | k8_combined | 1 | +0.021 |

## Coverage against the action plan (weeks 2–3)

| Action plan item | Status |
| --- | --- |
| Predict the change from the last pose | Done since v5: a residual rotation on the damped-velocity prior, in the body frame |
| Rotate each window to face forward | Done since v5: body-frame velocity input; heading-invariant by construction (exp. 11) |
| Shorter context windows | Tested: short_128 costs +0.029. The opposite helped: k8 sees 393 of the 400 samples. |
| n-futures output | **Not tried.** The submission format takes one `[1, 20, 4]` forecast, scored at step 20, so it has no slot for alternatives. |
| Innovation: distillation | Tried, failed (exp. 16) |
| Innovation: 8-bit quantisation | Measured on v6 only (exp. 6); not yet on k8_combined |
| Innovation: recurrent model that keeps state | Done: the stateful streaming graph keeps the dilated-convolution history as `state_in_i` tensors. Nic's GRU is the recurrent line. |
| Test unusual cases (fast flicks, noisy data, odd sampling rates) | Partly: frame conventions (exp. 11), jitter, rate offsets and drops (exp. 17, 20), sensor noise (exp. 25) |
| Export to ONNX with the same outputs | Done: every export is checked by stream parity (≤ 1e-4°) |

## Gaps for the paper

- **Leave-one-out on the final model: done in exp. 23 (6–8 Nov), with no_prior added.** Still missing: paired per-user tests (users better, CIs) on it. The original note, kept for reference:
- *No leave-one-out on the final model.* Each component was measured on a different base (v6 control, sep+angular, k8), mostly on one seed. A clean ablation table would remove one component at a time from k8_combined on seeds 42–44. That is about 0.8 h a seed on the 3080, so about 2.4 h a variant. A suggested suite:

  | Variant | Overrides of `best/configs/base.json` | Params |
  | --- | --- | ---: |
  | control | `{}`: k8_combined itself. A new suite retrains it (about 2.4 h), and it should reproduce 4.2471 / 4.2547 / 4.2485. | 7,555 |
  | no_gravity | `{"gravity_features": false}` | 7,459 |
  | no_mirror | `{"mirror_equivariant": false}` | 15,113 |
  | quaternion_loss | `{"loss_kind": "quaternion"}` | 7,555 |
  | user_sampling | `{"sampling": "user"}` | 7,555 |
  | no_tilt | `{"tilt_augmentation_deg": 0}` | 7,555 |
  | exact_maps | `{"maps": "exact"}` | 7,555 |
  | rf_243 | `{"num_blocks": 5, "kernel_sizes": [3, 3, 3, 3, 3], "dilations": [1, 3, 9, 27, 81]}` | 11,427 |

  - On 30 Oct, every override was checked to normalise and build under `best/`.
  - `no_mirror` and `rf_243` change the parameter count, so they are not size-matched.
  - The seven new variants come to about 17 h of GPU time. Whether they are worth it is a team call.
- **Dropout was never ablated.** It has been 0.1 since v3. The suite for it is written but not run (exp. 24).
- **Single seeds.** The v9 screen and every recipe screen use seed 42 only. Only the D confirm and exp. 23 have 3 seeds.
- **Test numbers** exist only for the selected seed of each model.
- **Latency** is ORT CPU on jacob-amd, relative only. The challenge measures on an undisclosed XR device, and the ST timings are still pending.

## Nic's work

Nic's GRU and GADS-T experiments are logged in `nic/FINDINGS.md`, with reports in `nic/results/`. On the same group-a-v2 data, his models are compared with these in exp. 13 and 19, and in `results/v8_bench/nic_comparison.md` and `results/v9_bench/full_comparison.md`.
