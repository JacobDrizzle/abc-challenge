# Head-orientation baseline evaluation

**Filter-qualified local evaluation — not an official challenge score.**

Split: **test** · 20 users · 72 segments · **931,051 windows**.

History: 400 samples. Forecast: 20 samples. Clock: 100 Hz. Stride: 1.

Every predictor uses the same windows and targets. Lower angular error is better.

| Predictor | Endpoint mean (°) | Endpoint median (°) | Endpoint p95 (°) | All-horizon mean (°) | Local median call (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| hold_last | 6.037257 | 5.015505 | 14.197877 | 3.478687 | 0.0013 |
| constant_velocity_1step | 6.943825 | 5.662354 | 16.618377 | 3.319682 | 0.0422 |
| constant_velocity_5step | 7.047833 | 5.808559 | 16.588100 | 3.535619 | 0.0421 |
| tiny_tcnn | 4.453404 | 3.718204 | 10.283792 | 2.318566 | 2.8581 |
| damped_velocity | 5.113740 | 4.298485 | 11.781809 | 2.660125 | 0.0565 |

The primary number is the **window-weighted mean error at step 20**. Median, p95, all-horizon mean, and equal-user mean are diagnostics; they are not interchangeable with it.

## Selected horizons

| Predictor | Step | Actual time ahead (ms) | Mean error (°) |
| --- | ---: | ---: | ---: |
| hold_last | 1 | 10 | 0.412106 |
| hold_last | 5 | 50 | 1.926884 |
| hold_last | 10 | 100 | 3.473258 |
| hold_last | 20 | 200 | 6.037257 |
| constant_velocity_1step | 1 | 10 | 0.116496 |
| constant_velocity_1step | 5 | 50 | 1.250760 |
| constant_velocity_1step | 10 | 100 | 3.054225 |
| constant_velocity_1step | 20 | 200 | 6.943825 |
| constant_velocity_5step | 1 | 10 | 0.249911 |
| constant_velocity_5step | 5 | 50 | 1.560733 |
| constant_velocity_5step | 10 | 100 | 3.298758 |
| constant_velocity_5step | 20 | 200 | 7.047833 |
| tiny_tcnn | 1 | 10 | 0.101040 |
| tiny_tcnn | 5 | 50 | 1.023341 |
| tiny_tcnn | 10 | 100 | 2.259544 |
| tiny_tcnn | 20 | 200 | 4.453404 |
| damped_velocity | 1 | 10 | 0.115474 |
| damped_velocity | 5 | 50 | 1.160412 |
| damped_velocity | 10 | 100 | 2.590293 |
| damped_velocity | 20 | 200 | 5.113740 |

![Mean error versus forecast horizon](horizon_errors.png)

## Interpretation

- At 100 Hz, step 10 is 100 ms and step 20 is 200 ms. Challenge prose calls step 20 100 ms; unresolved.
- CPU/Python call timings are a local diagnostic, not ST’s XR-device measurements. GPU adapters must synchronise before timing is meaningful.
- Overlapping windows are correlated. Their count is not the number of independent test examples.
- Input is an offline-prepared uniform stream. Causality of raw-data interpolation is not certified by this evaluator.
- Qualification is taken from the input segment metadata; the evaluator does not rerun or certify the upstream filter.
- All three built-in predictors have zero trainable parameters. They use only past observations.

## Reusing these results

`results.json` contains full horizon curves and per-user, per-recording, and per-segment summaries. Each `endpoint_*.npy` stores one endpoint error per row of `window_index.npy`; its segment IDs refer to the ordered `segments` list in the JSON. Anchor indices are zero based within each segment.

Cohort fingerprint: `015851ffd032833b294db919929d2ad4b141df1794c0f3e4ceaf4b9ac73a51d5`. Compare runs only when the evaluation cohort and protocol match. A future model can be evaluated alongside the baselines using the same command’s `--model` option.
