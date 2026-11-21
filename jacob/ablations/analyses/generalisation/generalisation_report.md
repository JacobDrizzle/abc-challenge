# Generalisation report: k8 trained on 100 and on 800 users

Made by `report.py` from `scores/*.json` (2026-11-05T21:14:07). Every number is at step 20 (200 ms) in degrees, window-weighted unless it says equal-user. CIs are 95%, from resampling users.

- **k8_100**: k8_combined trained on the 100-user file (v9_confirm, selected seed 42)
- **k8_800**: k8_combined trained on the 800-user file, one full pass (v10_k8_800users_full_pass, selected seed 44)

## A. Unseen-user accuracy

### k8_100

| Cohort | Users | Windows | Mean (CI) | Equal-user | 100 ms | Median / p95 / p99 | Between-user SD | Seeds mean ± SD | Δ vs CV (CI), users better | Δ vs damped (CI), users better |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| test20 | 20 | 931,051 | 4.267 [3.747, 4.751] | 4.210 | 2.189 | 3.55 / 9.90 / 15.01 | 1.210 | 4.2745 ± 0.0076 | -2.677 [-3.115, -2.239], 20/20 | -0.847 [-0.986, -0.743], 20/20 |
| test160 | 160 | 6,359,967 | 4.584 [4.377, 4.789] | 4.514 | 2.293 | 3.68 / 11.08 / 17.65 | 1.188 | 4.5901 ± 0.0066 | -2.888 [-3.114, -2.674], 159/160 | -1.003 [-1.097, -0.922], 159/160 |
| unseen pool (965) | 965 | 38,992,694 | 4.616 [4.526, 4.708] | 4.537 | 2.306 | – | 1.419 | 4.6207 ± 0.0051 | -2.808 [-2.905, -2.716], 963/965 | -0.976 [-1.008, -0.945], 963/965 |

Baselines on the same windows, and the 5 worst users by model/damped ratio:

- test20: hold 6.037, CV 6.944, damped 5.114. Worst: aecf2eb7 (Quest 2) 4.00 vs 4.47, ratio 0.894; 318e521e (Quest 2) 3.54 vs 3.99, ratio 0.887; a3fcf0fa (Unknown) 5.25 vs 5.97, ratio 0.880; 6338df1d (Quest 2) 4.87 vs 5.55, ratio 0.878; 6924807d (Vive family) 4.88 vs 5.57, ratio 0.877
- test160: hold 6.645, CV 7.472, damped 5.587. Worst: 62c2d6ab (Quest 2) 2.82 vs 2.48, ratio 1.136; 3c147cb2 (Quest 2) 5.83 vs 6.21, ratio 0.940; 643ea9bf (Pico) 4.39 vs 4.81, ratio 0.911; f5bd1d5b (Index) 2.95 vs 3.24, ratio 0.910; c2d4b427 (Quest 2) 4.80 vs 5.30, ratio 0.907
- unseen pool (965): hold 6.698, CV 7.424, damped 5.592. Worst: 62c2d6ab (Quest 2) 2.82 vs 2.48, ratio 1.136; 8124c348 (Quest 2) 14.17 vs 14.14, ratio 1.002; 4f7198bd (Quest 2) 5.22 vs 5.31, ratio 0.983; f105e445 (Quest 2) 6.55 vs 6.78, ratio 0.967; c5dd668c (Rift CV1) 3.51 vs 3.67, ratio 0.956

### k8_800

| Cohort | Users | Windows | Mean (CI) | Equal-user | 100 ms | Median / p95 / p99 | Between-user SD | Seeds mean ± SD | Δ vs CV (CI), users better | Δ vs damped (CI), users better |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| test20 | 20 | 931,051 | 4.214 [3.697, 4.697] | 4.152 | 2.163 | 3.51 / 9.77 / 14.87 | 1.200 | 4.2169 ± 0.0032 | -2.730 [-3.168, -2.290], 20/20 | -0.900 [-1.038, -0.796], 20/20 |
| test160 | 160 | 6,359,967 | 4.520 [4.317, 4.723] | 4.448 | 2.259 | 3.63 / 10.93 / 17.43 | 1.174 | 4.5216 ± 0.0028 | -2.952 [-3.180, -2.735], 160/160 | -1.067 [-1.165, -0.983], 159/160 |

Baselines on the same windows, and the 5 worst users by model/damped ratio:

- test20: hold 6.037, CV 6.944, damped 5.114. Worst: aecf2eb7 (Quest 2) 3.92 vs 4.47, ratio 0.878; a3fcf0fa (Unknown) 5.21 vs 5.97, ratio 0.873; 6924807d (Vive family) 4.85 vs 5.57, ratio 0.870; 6338df1d (Quest 2) 4.82 vs 5.55, ratio 0.869; 318e521e (Quest 2) 3.46 vs 3.99, ratio 0.867
- test160: hold 6.645, CV 7.472, damped 5.587. Worst: 62c2d6ab (Quest 2) 2.74 vs 2.48, ratio 1.101; 3c147cb2 (Quest 2) 5.72 vs 6.21, ratio 0.922; 643ea9bf (Pico) 4.33 vs 4.81, ratio 0.899; d9f4fda6 (Quest 2) 5.27 vs 5.88, ratio 0.897; f5bd1d5b (Index) 2.89 vs 3.24, ratio 0.891

## B. Seen against unseen users

Mean of the per-user model/damped ratio (equal-user). Lower is better. The ratio corrects for users whose motion is simply harder to predict. A positive unseen − trained difference whose CI excludes 0 means the model does better on the users it trained on.

### k8_100 (selected seed)

| Role | Users | Model | Damped | Mean ratio (CI) |
| --- | ---: | ---: | ---: | --- |
| trained | 100 | 4.778 | 5.946 | 0.8069 [0.7948, 0.8180] |
| selection | 15 | 4.247 | 5.049 | 0.8372 [0.8186, 0.8548] |
| unseen | 965 | 4.616 | 5.592 | 0.8289 [0.8253, 0.8324] |
| unseen_train_split | 700 | 4.637 | 5.618 | 0.8284 [0.8242, 0.8326] |

| Difference | Δ ratio | CI | Permutation p |
| --- | ---: | --- | ---: |
| unseen - trained | +0.0220 | [+0.0104, +0.0344] | 0.0003 |
| selection - trained | +0.0303 | [+0.0083, +0.0514] | 0.0578 |
| unseen - selection | -0.0083 | [-0.0256, +0.0105] | 0.5646 |
| unseen_train_split - trained | +0.0216 | [+0.0094, +0.0346] | 0.0004 |

unseen − trained by seed: 42: +0.0220, 43: +0.0211, 44: +0.0216

### k8_800 (selected seed)

| Role | Users | Model | Damped | Mean ratio (CI) |
| --- | ---: | ---: | ---: | --- |
| trained | 800 | 4.586 | 5.657 | 0.8128 [0.8088, 0.8167] |
| selection | 120 | 4.404 | 5.370 | 0.8197 [0.8096, 0.8296] |
| unseen | 160 | 4.520 | 5.587 | 0.8156 [0.8077, 0.8237] |

| Difference | Δ ratio | CI | Permutation p |
| --- | ---: | --- | ---: |
| unseen - trained | +0.0028 | [-0.0058, +0.0117] | 0.5603 |
| selection - trained | +0.0069 | [-0.0040, +0.0174] | 0.2114 |
| unseen - selection | -0.0041 | [-0.0165, +0.0088] | 0.5269 |

unseen − trained by seed: 42: +0.0038, 43: +0.0048, 44: +0.0028

## C. Unseen users by headset, source rate and motion

Per-user model/damped ratio inside each group. CIs only for groups with ≥ 5 users. Groups are of recordings, so a user with recordings in two groups counts in both. 'Trained' counts the model's training users who have a recording in the group. Small groups are anecdotes, not findings.

### k8_100 (motion tercile cut points 22.7 / 30.6 °/s, set on its training users)

**test160**

| headset family | Users | Recordings | Windows | Model | Damped | Mean ratio (CI) | Trained |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Index | 12 | 34 | 513,411 | 3.979 | 4.780 | 0.845 [0.820, 0.867] | 12 |
| Pico | 3 | 9 | 161,656 | 5.373 | 6.078 | 0.880 | 2 |
| Quest 1 | 4 | 13 | 107,594 | 4.321 | 5.057 | 0.862 | 1 |
| Quest 2 | 86 | 235 | 3,182,344 | 4.917 | 6.037 | 0.828 [0.816, 0.840] | 56 |
| Rift CV1 | 15 | 38 | 548,778 | 4.083 | 5.034 | 0.826 [0.804, 0.847] | 7 |
| Rift S | 1 | 4 | 24,946 | 4.419 | 5.301 | 0.834 | 2 |
| Unknown | 19 | 41 | 522,767 | 3.266 | 3.954 | 0.821 [0.793, 0.848] | 10 |
| Vive family | 25 | 62 | 1,077,453 | 4.651 | 5.713 | 0.811 [0.794, 0.827] | 12 |
| Windows MR | 4 | 12 | 221,018 | 4.797 | 5.536 | 0.849 | 1 |

| source rate (Hz, nearest 5) | Users | Recordings | Windows | Model | Damped | Mean ratio (CI) | Trained |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 85 | 4 | 5 | 120,011 | 4.535 | 5.261 | 0.863 | 4 |
| 90 | 156 | 419 | 5,958,799 | 4.586 | 5.600 | 0.827 [0.818, 0.835] | 97 |
| 95 | 8 | 14 | 172,708 | 4.139 | 4.930 | 0.844 [0.820, 0.872] | 7 |
| 100 | 2 | 4 | 44,735 | 4.762 | 5.779 | 0.821 | 2 |
| 105 | 2 | 2 | 12,768 | 8.179 | 9.891 | 0.796 | 0 |
| 110 | 3 | 4 | 50,946 | 4.928 | 5.839 | 0.849 | 0 |

| motion intensity (velocity_deg_s tercile) | Users | Recordings | Windows | Model | Damped | Mean ratio (CI) | Trained |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| low | 95 | 171 | 2,412,255 | 3.592 | 4.249 | 0.850 [0.839, 0.860] | 52 |
| mid | 96 | 156 | 2,340,324 | 4.475 | 5.409 | 0.827 [0.817, 0.836] | 50 |
| high | 73 | 121 | 1,607,388 | 6.232 | 7.854 | 0.800 [0.784, 0.815] | 45 |

Headsets absent from the training users: DELL VISOR VR1180 (1 user); HP Reverb VR Headset VR1000-2xxx0 (1 user); HP Reverb Virtual Reality Headset G20 (1 user); Rift_S (1 user); Samsung Windows Mixed Reality 800ZBA0 (1 user); Vive MV (4 users); Vive MV. (1 user)


**unseen pool (965)**

| headset family | Users | Recordings | Windows | Model | Damped | Mean ratio (CI) | Trained |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Index | 95 | 270 | 4,393,932 | 4.727 | 5.759 | 0.828 [0.817, 0.839] | 12 |
| Other | 4 | 7 | 100,401 | 3.518 | 4.284 | 0.826 | 0 |
| Pico | 27 | 61 | 994,654 | 5.206 | 6.045 | 0.864 [0.844, 0.883] | 2 |
| Pimax | 2 | 5 | 102,442 | 4.253 | 5.009 | 0.845 | 0 |
| Quest 1 | 17 | 47 | 401,634 | 5.286 | 6.450 | 0.834 [0.803, 0.865] | 1 |
| Quest 2 | 552 | 1481 | 21,035,000 | 4.746 | 5.753 | 0.827 [0.822, 0.832] | 56 |
| Rift CV1 | 79 | 193 | 3,036,122 | 4.270 | 5.221 | 0.828 [0.814, 0.842] | 7 |
| Rift S | 2 | 6 | 47,553 | 4.407 | 5.181 | 0.852 | 2 |
| Unknown | 85 | 181 | 2,311,523 | 3.535 | 4.281 | 0.833 [0.822, 0.844] | 10 |
| Vive family | 121 | 325 | 5,578,612 | 4.627 | 5.607 | 0.822 [0.812, 0.830] | 12 |
| Windows MR | 28 | 64 | 990,821 | 4.166 | 4.960 | 0.844 [0.831, 0.857] | 1 |

| source rate (Hz, nearest 5) | Users | Recordings | Windows | Model | Damped | Mean ratio (CI) | Trained |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 85 | 46 | 53 | 838,494 | 4.802 | 5.674 | 0.858 [0.842, 0.874] | 4 |
| 90 | 930 | 2489 | 36,744,082 | 4.611 | 5.594 | 0.828 [0.825, 0.832] | 97 |
| 95 | 46 | 58 | 866,315 | 4.584 | 5.474 | 0.834 [0.809, 0.855] | 7 |
| 100 | 12 | 17 | 247,567 | 4.385 | 5.243 | 0.815 [0.760, 0.858] | 2 |
| 105 | 11 | 15 | 196,820 | 4.855 | 5.722 | 0.849 [0.812, 0.881] | 0 |
| 110 | 7 | 8 | 99,416 | 4.984 | 5.808 | 0.865 [0.844, 0.889] | 0 |

| motion intensity (velocity_deg_s tercile) | Users | Recordings | Windows | Model | Damped | Mean ratio (CI) | Trained |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| low | 556 | 959 | 14,350,664 | 3.542 | 4.203 | 0.846 [0.842, 0.850] | 52 |
| mid | 583 | 988 | 14,957,177 | 4.636 | 5.567 | 0.832 [0.828, 0.836] | 50 |
| high | 401 | 693 | 9,684,853 | 6.176 | 7.689 | 0.805 [0.797, 0.812] | 45 |

Headsets absent from the training users: 3GLASSES S20 (1 user); Acer AH1010 (1 user); DELL VISOR VR1180 (1 user); HP Reverb VR Headset VR1000-2xxx0 (1 user); HP Reverb Virtual Reality Headset G20 (18 users); Lenovo Explorer0 (1 user); Miramar (2 users); PICO 4 (11 users); Pico Neo 2 (1 user); Pimax 5K Plus (1 user); Pimax 5K Super (1 user); Rift_S (1 user); Samsung Windows Mixed Reality 800ZAA0 (2 users); Samsung Windows Mixed Reality 800ZBA0 (4 users); Vive (2 users); Vive MV (20 users); Vive MV. (4 users); hedy (1 user); vive_cosmos (3 users)

### k8_800 (motion tercile cut points 22.3 / 29.0 °/s, set on its training users)

**test160**

| headset family | Users | Recordings | Windows | Model | Damped | Mean ratio (CI) | Trained |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Index | 12 | 34 | 513,411 | 3.915 | 4.780 | 0.830 [0.806, 0.852] | 89 |
| Pico | 3 | 9 | 161,656 | 5.310 | 6.078 | 0.868 | 24 |
| Quest 1 | 4 | 13 | 107,594 | 4.234 | 5.057 | 0.835 | 13 |
| Quest 2 | 86 | 235 | 3,182,344 | 4.846 | 6.037 | 0.815 [0.804, 0.828] | 463 |
| Rift CV1 | 15 | 38 | 548,778 | 4.019 | 5.034 | 0.814 [0.793, 0.835] | 62 |
| Rift S | 1 | 4 | 24,946 | 4.313 | 5.301 | 0.814 | 2 |
| Unknown | 19 | 41 | 522,767 | 3.221 | 3.954 | 0.809 [0.781, 0.837] | 67 |
| Vive family | 25 | 62 | 1,077,453 | 4.598 | 5.713 | 0.802 [0.784, 0.818] | 91 |
| Windows MR | 4 | 12 | 221,018 | 4.748 | 5.536 | 0.839 | 23 |

| source rate (Hz, nearest 5) | Users | Recordings | Windows | Model | Damped | Mean ratio (CI) | Trained |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 85 | 4 | 5 | 120,011 | 4.494 | 5.261 | 0.855 | 43 |
| 90 | 156 | 419 | 5,958,799 | 4.522 | 5.600 | 0.814 [0.806, 0.823] | 773 |
| 95 | 8 | 14 | 172,708 | 4.077 | 4.930 | 0.830 [0.806, 0.858] | 40 |
| 100 | 2 | 4 | 44,735 | 4.654 | 5.779 | 0.801 | 10 |
| 105 | 2 | 2 | 12,768 | 8.032 | 9.891 | 0.780 | 8 |
| 110 | 3 | 4 | 50,946 | 4.854 | 5.839 | 0.833 | 4 |

| motion intensity (velocity_deg_s tercile) | Users | Recordings | Windows | Model | Damped | Mean ratio (CI) | Trained |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| low | 90 | 162 | 2,273,293 | 3.509 | 4.216 | 0.835 [0.824, 0.846] | 428 |
| mid | 93 | 147 | 2,239,128 | 4.384 | 5.340 | 0.819 [0.809, 0.828] | 467 |
| high | 77 | 139 | 1,847,546 | 5.929 | 7.574 | 0.792 [0.778, 0.805] | 386 |

Headsets absent from the training users: DELL VISOR VR1180 (1 user); HP Reverb VR Headset VR1000-2xxx0 (1 user); Rift_S (1 user)

## D. Validation to test

Ledger models (selected checkpoint, val15 from `runs.csv`, test20 from the bench): Spearman 0.983, Kendall 0.944, Pearson 0.992; test − val +0.0301 ± 0.0135. With 9 models that share one cohort, this is weak evidence.

| Bench model | Ledger | Seed | val15 | test20 | test − val |
| --- | --- | ---: | ---: | ---: | ---: |
| v5_tcnn_ga2 | v5_group_a_v2/control | 44 | 4.4004 | 4.4534 | +0.0530 |
| v6_tcnn_ga2 | v6_group_a_v2/control | 44 | 4.3285 | 4.3584 | +0.0298 |
| v7_sep_base3_addons_ga2 | v7_extra/separable_base3_addons | 43 | 4.3699 | 4.4186 | +0.0487 |
| v7_base3_5_addons_ga2 | v7_confirm/base3_5_addons | 43 | 4.3561 | 4.3923 | +0.0362 |
| v7_sep_base3_addons_angular_ga2 | v8_recipe_sep_base3_addons/angular_loss | 44 | 4.3134 | 4.3289 | +0.0156 |
| v7_base3_5_addons_angular_ga2 | v8_recipe_base3_5_addons/angular_loss | 44 | 4.3135 | 4.3420 | +0.0285 |
| v9_k5_combined_ga2 | v9_confirm/k5_combined | 42 | 4.2652 | 4.2871 | +0.0220 |
| v9_k8_combined_ga2 | v9_confirm/k8_combined | 42 | 4.2471 | 4.2666 | +0.0195 |
| v9_k8_jitter3_s42_ga2 | v9_jitter/jitter_3ms | 42 | 4.3212 | 4.3388 | +0.0176 |

k8-800 − k8-100, paired per user (negative = the 800-user model is better). val15 is k8-100's selection set and val120 is k8-800's; val105_new and the test sets are unseen by both.

| Cohort | Selected seeds: Δ (CI), users better | 3-seed average: Δ (CI), users better |
| --- | --- | --- |
| val15 | -0.0668 [-0.082, -0.055], 15/15 | -0.0665 [-0.084, -0.052], 15/15 |
| val105_new | -0.0750 [-0.085, -0.065], 103/105 | -0.0733 [-0.083, -0.065], 105/105 |
| val120 | -0.0739 [-0.083, -0.066], 118/120 | -0.0724 [-0.081, -0.065], 120/120 |
| test20 | -0.0524 [-0.061, -0.045], 20/20 | -0.0575 [-0.067, -0.048], 20/20 |
| test160 | -0.0639 [-0.073, -0.056], 151/160 | -0.0684 [-0.077, -0.061], 157/160 |

## Verification

- OK: scored the merged file (data sha256 ['11b53c681e51cc1c03d5ba98b34aeb0683a17133bafff1edce0bc763c4cacf60'])
- OK: train windows match check_corpus (32,378,176 vs 32,378,176)
- OK: validation windows match check_corpus (4,755,761 vs 4,755,761)
- OK: validation scored at stride 1 (every window scored)
- OK: test windows match check_corpus (6,359,967 vs 6,359,967)
- OK: test scored at stride 1 (every window scored)
- OK: test20 windows (931,051)
- OK: hold_last on test20 = bench hold_last (max per-user difference 2.45e-10 deg)
- OK: constant_velocity on test20 = bench constant_velocity_1step (max per-user difference 2.08e-07 deg)
- OK: damped_velocity on test20 = bench damped_velocity (max per-user difference 1.04e-07 deg)
- OK: k8_100_s42 on test20 = bench v9_k8_combined_ga2 (max per-user difference 2.01e-07 deg)
- OK: k8_100_s42 on val15 = trainer (4.247105 vs 4.247105)
- OK: k8_100_s43 on val15 = trainer (4.254723 vs 4.254723)
- OK: k8_100_s44 on val15 = trainer (4.248472 vs 4.248472)
- OK: k8_100_s42 is the selected checkpoint (1857d399d6d31dc2)
- OK: k8_100 roles are disjoint and cover every user ({'trained': 100, 'selection': 15, 'unseen': 965})
- OK: k8_800_s42 on val120 = trainer (4.411101 vs 4.411101)
- OK: k8_800_s43 on val120 = trainer (4.405637 vs 4.405637)
- OK: k8_800_s44 on val120 = trainer (4.403629 vs 4.403629)
- OK: k8_800_s44 is the selected checkpoint (c70aa91599322467)
- OK: k8_800 roles are disjoint and cover every user ({'trained': 800, 'selection': 120, 'unseen': 160})
