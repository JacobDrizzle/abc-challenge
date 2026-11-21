# Model comparison on group-a-v2-test20 (jacob-amd, 29 Oct 2026)

Every model of Jacob's from v7 on and all ten of Nic's group-a-v2 models, scored with `python -m bench score` on jacob-amd at one bench commit, plus one latency session (`jacob-amd_20261029T125659.json`). v5, v6 and the baselines are earlier jacob-amd scores, shown for context. All rows share one cohort fingerprint (20 test players, 931,051 windows). Test is for reporting; choose between models on validation.

## Accuracy (degrees, lower is better)

Test = step 20 (200 ms), streamed from segment start. vs current best and vs Nic GRU: paired per-player difference [95% CI], players better (negative = this model better). Nic GRU = the 5-seed mean per player.

| Model | Graph | Validation (3 seeds) | Test s20 | vs current best | vs Nic GRU | Cold start s20 | Test s10 | Median s20 | p99 s20 | Equal-player s20 |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| k8_combined | v9 padded rows | 4.2501 ± 0.0041 | **4.2666** | -0.0623 [-0.0817, -0.0449], 18/20 | -0.2004 [-0.2340, -0.1692], 20/20 | 4.2713 | 2.1887 | 3.5545 | 15.008 | 4.2103 |
| k8_combined | v9 lean | 4.2501 ± 0.0041 | **4.2666** | -0.0623 [-0.0817, -0.0449], 18/20 | -0.2004 [-0.2340, -0.1692], 20/20 | 4.2713 | 2.1887 | 3.5545 | 15.008 | 4.2103 |
| k8_combined | v9 padded + A4 | 4.2501 ± 0.0041 | **4.2666** | -0.0623 [-0.0817, -0.0449], 18/20 | -0.2004 [-0.2340, -0.1692], 20/20 | 4.2713 | 2.1887 | 3.5545 | 15.008 | 4.2103 |
| k5_combined | v9 lean | 4.2717 ± 0.0067 | **4.2871** | -0.0418 [-0.0576, -0.0273], 17/20 | -0.1798 [-0.2075, -0.1536], 20/20 | 4.2937 | 2.1982 | 3.5717 | 15.102 | 4.2289 |
| k5_combined | v9 padded rows | 4.2717 ± 0.0067 | **4.2871** | -0.0418 [-0.0576, -0.0273], 17/20 | -0.1798 [-0.2075, -0.1536], 20/20 | 4.2937 | 2.1982 | 3.5717 | 15.102 | 4.2289 |
| k5_combined | v9 padded + A4 | 4.2717 ± 0.0067 | **4.2871** | -0.0418 [-0.0576, -0.0273], 17/20 | -0.1798 [-0.2075, -0.1536], 20/20 | 4.2937 | 2.1982 | 3.5717 | 15.102 | 4.2289 |
| sep + angular | v9 lean | 4.3297 ± 0.0142 | **4.3289** | -0.0000 [-0.0000, -0.0000], 16/20 | -0.1380 [-0.1706, -0.1048], 19/20 | 4.3315 | 2.2097 | 3.6095 | 15.221 | 4.2778 |
| sep + angular | v7 mirror-split | 4.3297 ± 0.0142 | **4.3289** | -0.0000 [-0.0000, +0.0000], 12/20 | -0.1380 [-0.1706, -0.1048], 19/20 | 4.3315 | 2.2097 | 3.6095 | 15.221 | 4.2778 |
| sep + angular (current best) | v7 | 4.3297 ± 0.0142 | **4.3289** | reference | -0.1380 [-0.1706, -0.1048], 19/20 | 4.3315 | 2.2097 | 3.6095 | 15.221 | 4.2778 |
| k8 + jitter 3 ms (screen) | v9 lean | 4.3212 (seed 42 only) | **4.3388** | +0.0099 [-0.0016, +0.0232], 5/20 | -0.1282 [-0.1573, -0.0978], 19/20 | 4.3429 | 2.2388 | 3.6125 | 15.262 | 4.2944 |
| base3_5 + angular | v7 | 4.3247 ± 0.0104 | **4.3420** | +0.0131 [+0.0017, +0.0255], 7/20 | -0.1250 [-0.1530, -0.0940], 18/20 | 4.3441 | 2.2143 | 3.6163 | 15.253 | 4.2879 |
| v6 control | v7 | 4.3423 ± 0.0121 | **4.3584** | +0.0295 [+0.0048, +0.0510], 4/20 | -0.1086 [-0.1442, -0.0766], 18/20 | 4.3624 | 2.2305 | 3.6451 | 15.138 | 4.2979 |
| base3_5 (quaternion loss) | v7 | 4.3673 ± 0.0121 | **4.3923** | +0.0634 [+0.0472, +0.0781], 1/20 | -0.0747 [-0.1066, -0.0431], 17/20 | 4.3946 | 2.2484 | 3.6764 | 15.212 | 4.3370 |
| sep (quaternion loss) | v7 | 4.3829 ± 0.0119 | **4.4186** | +0.0897 [+0.0729, +0.1062], 0/20 | -0.0483 [-0.0774, -0.0171], 14/20 | 4.4225 | 2.2504 | 3.7020 | 15.253 | 4.3645 |
| Nic GADS-T s42 | GADS-T | – | **4.4459** | +0.1170 [+0.0839, +0.1535], 1/20 | -0.0210 [-0.0514, +0.0124], 14/20 | – | 2.2958 | 3.7002 | 15.672 | 4.3980 |
| Nic stateful GRU s43 | GRU, 32 units | – | **4.4533** | +0.1244 [+0.0798, +0.1714], 1/20 | – | 4.4562 | 2.2574 | 3.7167 | 15.639 | 4.3969 |
| Nic stateful GRU s45 | GRU, 32 units | – | **4.4574** | +0.1284 [+0.0963, +0.1596], 1/20 | – | 4.4599 | 2.2597 | 3.7255 | 15.525 | 4.4077 |
| Nic stateful GRU s44 | GRU, 32 units | – | **4.4661** | +0.1372 [+0.1062, +0.1659], 1/20 | – | 4.4678 | 2.2600 | 3.7230 | 15.708 | 4.4146 |
| Nic stateful GRU s46 | GRU, 32 units | – | **4.4759** | +0.1470 [+0.1138, +0.1787], 1/20 | – | 4.4775 | 2.2687 | 3.7371 | 15.745 | 4.4241 |
| Nic stateful GRU s42 | GRU, 32 units | – | **4.4821** | +0.1532 [+0.1183, +0.1888], 1/20 | – | 4.4841 | 2.2562 | 3.7340 | 15.816 | 4.4323 |
| Nic GADS-T s44 | GADS-T | – | **4.4935** | +0.1645 [+0.1281, +0.2066], 0/20 | +0.0265 [-0.0028, +0.0638], 6/20 | – | 2.3319 | 3.7352 | 15.919 | 4.4525 |
| Nic GADS-T s45 | GADS-T | – | **4.5231** | +0.1942 [+0.1615, +0.2316], 1/20 | +0.0562 [+0.0231, +0.0926], 4/20 | – | 2.3143 | 3.7667 | 15.981 | 4.4796 |
| Nic GADS-T s43 | GADS-T | – | **4.5253** | +0.1964 [+0.1605, +0.2397], 1/20 | +0.0584 [+0.0219, +0.1010], 5/20 | – | 2.3169 | 3.7692 | 15.988 | 4.4801 |
| Nic GADS-T s46 | GADS-T | – | **4.5404** | +0.2114 [+0.1715, +0.2567], 0/20 | +0.0734 [+0.0383, +0.1133], 1/20 | – | 2.3284 | 3.7874 | 15.974 | 4.4993 |
| v6 control (registered) | v6 | 4.3423 ± 0.0121 | **4.3584** | +0.0295 [+0.0048, +0.0510], 4/20 | -0.1086 [-0.1442, -0.0766], 18/20 | 4.3624 | 2.2305 | 3.6451 | 15.138 | 4.2979 |
| v5 control | v5 | 4.4049 ± 0.0042 | **4.4534** | +0.1245 [+0.1037, +0.1456], 1/20 | -0.0136 [-0.0507, +0.0217], 13/20 | 4.4560 | 2.2595 | 3.7182 | 15.539 | 4.3929 |
| damped velocity | baseline | – | **5.1137** | +0.7848 [+0.6852, +0.9137], 0/20 | +0.6468 [+0.5405, +0.7761], 0/20 | – | 2.5903 | 4.2985 | 17.677 | 5.0729 |
| hold last | baseline | – | **6.0373** | +1.7083 [+1.5216, +1.9166], 0/20 | +1.5703 [+1.3687, +1.7912], 0/20 | – | 3.4733 | 5.0155 | 21.381 | 5.9381 |

## Size, compute and latency

Latency: ONNX Runtime CPU on jacob-amd, one call = one new sample, median / p99 in µs, alternating rounds in one session. Relative only, not ST's hardware. Nodes: non-Constant nodes in the file, then after ONNX Runtime folds constants at load.

| Model | Graph | Declared params | ONNX weight elements | MACs / call | State floats | Nodes (file / run) | Off safe list | 1 thread median / p99 | 4 threads median | Deterministic, 1 = 4 threads |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| k8_combined | v9 padded rows | 7,555 | 8,088 | 22,366 | 25,088 | 57 / 54 | none | 48.8 / 99.2 | 75.7 | yes |
| k8_combined | v9 lean | 7,555 | 8,100 | 16,222 | 25,088 | 60 / 60 | none | 56.1 / 112.8 | 91.4 | yes |
| k8_combined | v9 padded + A4 | 7,555 | 8,016 | 22,168 | 25,088 | 58 / 51 | none | 47.2 / 96.8 | 77.7 | yes |
| k5_combined | v9 lean | 5,595 | 6,148 | 12,318 | 19,008 | 69 / 69 | none | 61.9 / 123.9 | 97.3 | yes |
| k5_combined | v9 padded rows | 5,595 | 6,132 | 16,926 | 19,008 | 65 / 61 | none | 53.6 / 106.1 | 85.6 | yes |
| k5_combined | v9 padded + A4 | 5,595 | 6,060 | 16,728 | 19,008 | 66 / 58 | none | 52.2 / 111.9 | 83.8 | yes |
| sep + angular | v9 lean | 6,651 | 7,200 | 14,446 | 11,616 | 84 / 84 | none | 72.0 / 147.8 | 113.1 | yes |
| sep + angular | v7 mirror-split | 6,651 | 7,127 | 8,103 | 11,616 | 150 / 150 | none | 103.9 / 204.9 | 145.7 | yes |
| sep + angular (current best) | v7 | 6,651 | 7,112 | 13,863 | 11,616 | 130 / 130 | none | 91.0 / 175.3 | 112.1 | yes |
| k8 + jitter 3 ms (screen) | v9 lean | 7,555 | 8,088 | 16,286 | 25,088 | 66 / 66 | none | 58.5 / 151.3 | 95.2 | yes |
| base3_5 + angular | v7 | 17,811 | 18,272 | 36,183 | 11,616 | 125 / 125 | none | 91.4 / 174.7 | 94.8 | yes |
| v6 control | v7 | 26,012 | 26,501 | 52,194 | 12,192 | 145 / 145 | Sqrt | 108.1 / 216.3 | 111.1 | yes |
| base3_5 (quaternion loss) | v7 | 17,811 | 18,272 | 36,183 | 11,616 | 125 / 125 | none | 91.6 / 171.9 | 110.5 | yes |
| sep (quaternion loss) | v7 | 6,651 | 7,112 | 13,863 | 11,616 | 130 / 130 | none | 91.4 / 178.6 | 128.4 | yes |
| Nic GADS-T s42 | GADS-T | 10,352 | 11,128 | 606,320 | 0 | 102 / 102 | Softmax (conditional) | 174.3 / 305.9 | 273.5 | yes |
| Nic stateful GRU s43 | GRU, 32 units | 6,192 | 6,284 | 6,384 | 36 | 35 / 35 | none | 23.6 / 48.6 | 24.6 | yes |
| Nic stateful GRU s45 | GRU, 32 units | 6,192 | 6,284 | 6,384 | 36 | 35 / 35 | none | 23.6 / 53.0 | 24.5 | yes |
| Nic stateful GRU s44 | GRU, 32 units | 6,192 | 6,284 | 6,384 | 36 | 35 / 35 | none | 23.7 / 47.7 | 24.5 | yes |
| Nic stateful GRU s46 | GRU, 32 units | 6,192 | 6,284 | 6,384 | 36 | 35 / 35 | none | 23.7 / 47.8 | 24.3 | yes |
| Nic stateful GRU s42 | GRU, 32 units | 6,192 | 6,284 | 6,384 | 36 | 35 / 35 | none | 23.7 / 48.2 | 24.5 | yes |
| Nic GADS-T s44 | GADS-T | 10,352 | 11,128 | 606,320 | 0 | 102 / 102 | Softmax (conditional) | 172.8 / 311.0 | 257.6 | yes |
| Nic GADS-T s45 | GADS-T | 10,352 | 11,128 | 606,320 | 0 | 102 / 102 | Softmax (conditional) | 172.6 / 326.1 | 274.0 | yes |
| Nic GADS-T s43 | GADS-T | 10,352 | 11,128 | 606,320 | 0 | 102 / 102 | Softmax (conditional) | 172.5 / 305.4 | 258.3 | yes |
| Nic GADS-T s46 | GADS-T | 10,352 | 11,128 | 606,320 | 0 | 102 / 102 | Softmax (conditional) | 171.9 / 291.1 | 292.3 | yes |
| v6 control (registered) | v6 | 26,012 | 26,528 | 52,144 | 12,192 | 174 / 174 | Div, Less, LessOrEqual, ReduceSum, Sqrt, Sub, Where | – / – | – | yes |
| v5 control | v5 | 51,900 | 52,160 | 51,920 | 12,192 | 133 / 133 | Div, Less, LessOrEqual, ReduceSum, Sqrt, Sub, Where | – / – | – | yes |

## Nic's seed groups (5 seeds)

| Group | Test s20 mean ± sd | Test s10 mean ± sd | Cold-start s20 mean ± sd |
|---|---:|---:|---:|
| Nic stateful GRU | 4.4670 ± 0.0121 | 2.2604 ± 0.0049 | 4.4691 ± 0.0117 |
| Nic GADS-T | 4.5056 ± 0.0375 | 2.3175 ± 0.0142 | 4.5056 ± 0.0375 |

## Key paired comparisons (A − B, negative = A better)

Each cell: mean difference [95% CI], players where A is better. 20 players, 10,000 draws, seed 0.

| A | B | From start s20 | Cold start s20 | From start s10 |
|---|---|---|---|---|
| v9_k8_combined_ga2 | v7_sep_base3_addons_angular_ga2 | -0.0623 [-0.0817, -0.0449], 18/20 | -0.0602 [-0.0794, -0.0428], 17/20 | -0.0210 [-0.0279, -0.0145], 18/20 |
| v9_k5_combined_ga2 | v7_sep_base3_addons_angular_ga2 | -0.0418 [-0.0576, -0.0273], 17/20 | -0.0378 [-0.0530, -0.0245], 16/20 | -0.0114 [-0.0194, -0.0046], 15/20 |
| v9_k8_combined_ga2 | v9_k5_combined_ga2 | -0.0205 [-0.0300, -0.0108], 16/20 | -0.0224 [-0.0310, -0.0134], 15/20 | -0.0095 [-0.0130, -0.0060], 17/20 |
| v9_k8_jitter3_s42_ga2 | v9_k8_combined_ga2 | +0.0722 [+0.0557, +0.0930], 0/20 | +0.0716 [+0.0549, +0.0923], 0/20 | +0.0501 [+0.0378, +0.0639], 1/20 |
| v9_k8_combined_ga2 | nic_ga2_gru (5-seed mean) | -0.2004 [-0.2340, -0.1692], 20/20 | -0.1978 [-0.2313, -0.1665], 19/20 | -0.0717 [-0.0845, -0.0601], 20/20 |
| v9_k8_combined_ga2 | nic_ga2_gadst (5-seed mean) | -0.2391 [-0.2914, -0.1989], 20/20 | -0.2343 [-0.2863, -0.1941], 20/20 | -0.1288 [-0.1442, -0.1153], 20/20 |
| v9_k5_combined_ga2 | nic_ga2_gru (5-seed mean) | -0.1798 [-0.2075, -0.1536], 20/20 | -0.1754 [-0.2039, -0.1476], 20/20 | -0.0622 [-0.0744, -0.0514], 20/20 |
| v7_sep_base3_addons_angular_ga2 | nic_ga2_gru (5-seed mean) | -0.1380 [-0.1706, -0.1048], 19/20 | -0.1376 [-0.1701, -0.1043], 19/20 | -0.0507 [-0.0654, -0.0360], 18/20 |
| v7_sep_base3_addons_angular_ga2 | nic_ga2_gadst (5-seed mean) | -0.1767 [-0.2160, -0.1430], 19/20 | -0.1742 [-0.2138, -0.1398], 19/20 | -0.1078 [-0.1220, -0.0952], 20/20 |
| v9_k8_combined_padded_ga2 | v9_k8_combined_ga2 | -0.0000 [-0.0000, +0.0000], 12/20 | -0.0000 [-0.0000, +0.0000], 11/20 | +0.0000 [-0.0000, +0.0000], 7/20 |
| v9_k8_combined_padded_a4_ga2 | v9_k8_combined_ga2 | +0.0000 [-0.0000, +0.0000], 7/20 | +0.0000 [-0.0000, +0.0000], 10/20 | -0.0000 [-0.0000, +0.0000], 11/20 |
| v9_sep_angular_lean_ga2 | v7_sep_base3_addons_angular_ga2 | -0.0000 [-0.0000, -0.0000], 16/20 | -0.0000 [-0.0000, -0.0000], 16/20 | -0.0000 [-0.0000, +0.0000], 11/20 |
