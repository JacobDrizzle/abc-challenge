# Benchmark: v5 / v6 / v7 and angular-loss models

Bench `group-a-v2-test20` (931,051 windows, 20 unseen players, cohort `6303a55cc75316cc`), machine `jacob-amd`, latency session `jacob-amd_20261027T195719.json`.

## Size, compute and latency

Latency: ONNX Runtime CPU, one call = one new sample, median / p99 in microseconds; alternating rounds in one session. Relative only, not the challenge hardware.

| Model | Graph | Declared params | ONNX weight elements | MACs / call | State floats | Nodes | Off safe list | 1 thread median / p99 (µs) | 4 threads median (µs) | Deterministic, 1 = 4 threads |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| v5 control | v5 | 51,900 | 52,160 | 51,920 | 12,192 | 226 | Div, Less, LessOrEqual, ReduceSum, Sqrt, Sub, Where | 92 / 137 | 95 | yes |
| v6 control (registered) | v6 | 26,012 | 26,528 | 52,144 | 12,192 | 345 | Div, Less, LessOrEqual, ReduceSum, Sqrt, Sub, Where | 119 / 166 | 177 | yes |
| v6 control, v7 graph | v7 | 26,012 | 26,501 | 52,194 | 12,192 | 299 | Sqrt | 104 / 169 | 109 | yes |
| base3_5_addons | v7 | 17,811 | 18,272 | 36,183 | 11,616 | 256 | none | 88 / 122 | 94 | yes |
| base3_5_addons + angular | v7 | 17,811 | 18,272 | 36,183 | 11,616 | 256 | none | 89 / 106 | 93 | yes |
| separable_base3_addons | v7 | 6,651 | 7,112 | 13,863 | 11,616 | 261 | none | 86 / 100 | 111 | yes |
| separable_base3_addons + angular | v7 | 6,651 | 7,112 | 13,863 | 11,616 | 261 | none | 88 / 118 | 130 | yes |

## Accuracy (degrees, lower is better)

Validation: 15 users, 3-seed mean ± SD; the selected seed is the ONNX file scored here (chosen on validation). Test: the selected seed only, step 20 = 200 ms, streamed from segment start; cold start = zero states at the first full window. vs v6_tcnn_ga2: paired per-player difference, bootstrap 95% CI, players better.

| Model | Loss | Validation 3-seed | Seed | Test step 20 | vs v6 (test) | Test cold start | Test step 10 (100 ms) | Test median | Test p95 | Test p99 | Test equal-player |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v5 control | quaternion | 4.4049 ± 0.0042 | 44 | **4.4534** | +0.0950 [+0.0641, +0.1317], 2/20 | 4.4560 | 2.2595 | 3.7182 | 10.284 | 15.539 | 4.3929 |
| v6 control (registered) | quaternion | 4.3423 ± 0.0121 | 44 | **4.3584** | reference | 4.3624 | 2.2305 | 3.6451 | 10.053 | 15.138 | 4.2979 |
| v6 control, v7 graph | quaternion | 4.3423 ± 0.0121 | 44 | **4.3584** | +0.0000 [-0.0000, +0.0000], 7/20 | 4.3624 | 2.2305 | 3.6451 | 10.053 | 15.138 | 4.2979 |
| base3_5_addons | quaternion | 4.3673 ± 0.0121 | 43 | **4.3923** | +0.0339 [+0.0249, +0.0446], 0/20 | 4.3946 | 2.2484 | 3.6764 | 10.117 | 15.212 | 4.3370 |
| base3_5_addons + angular | angular | 4.3247 ± 0.0104 | 44 | **4.3420** | -0.0164 [-0.0342, +0.0055], 16/20 | 4.3441 | 2.2143 | 3.6163 | 10.074 | 15.253 | 4.2879 |
| separable_base3_addons | quaternion | 4.3829 ± 0.0119 | 43 | **4.4186** | +0.0603 [+0.0440, +0.0807], 1/20 | 4.4225 | 2.2504 | 3.7020 | 10.183 | 15.253 | 4.3645 |
| separable_base3_addons + angular | angular | 4.3297 ± 0.0142 | 44 | **4.3289** | -0.0295 [-0.0510, -0.0048], 16/20 | 4.3315 | 2.2097 | 3.6095 | 10.050 | 15.221 | 4.2778 |
| damped velocity (baseline) | – | – | – | 5.1137 | – | – | 2.5903 | 4.2985 | 11.782 | 17.677 | 5.0729 |
| hold last (baseline) | – | – | – | 6.0373 | – | – | 3.4733 | 5.0155 | 14.198 | 21.381 | 5.9381 |
