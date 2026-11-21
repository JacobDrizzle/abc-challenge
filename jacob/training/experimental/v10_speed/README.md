# v10_speed: faster training and validation, same numbers

Written by Nic with AI help on 9 Nov 2026, for review. It overlaps with Jacob's unpushed `v10_fast` (same idea for
the training forward); the two should be reconciled before either is promoted.

## Question

Can k8_combined be trained and validated fast enough for the 1,600-user and 4,000-user sets, without changing what
it learns?

## What changes

| File | Change | Exact? |
| --- | --- | --- |
| `model.py` | `LastPositionTCNN.forward` computes only the positions the last position reads: 73 block positions a window instead of 1,200 | Same forecast to float32 rounding (about 1e-7). Training draws fewer dropout numbers, so runs are not bit-identical to `LeanTCNN` runs |
| `validation.py` | Validation runs the network once along each recording and reads the forecast at every anchor | Same score to about 1e-9 degrees |
| `training.py` | The window-start list stays in host memory | Same integers, same batches |

Checkpoints are unchanged and load with `best.model.load_checkpoint` as `LeanTCNN`. Nothing in `best/` is edited.

Two config keys, both `true` by default: `last_position_forward` and `dense_validation`. With both `false`, a run is
`best`'s run exactly (tested).

## Why it is the same number

- **Last-position forward.** A block's output at time t reads its input at t, t - d, ..., t - (k - 1)d. Working back
  from the last position with kernel 8 and dilations 47, 8, 1: block 2 is needed at 1 position, block 1 at 8,
  block 0 at 64, and the projection at 393 (the receptive field).
- **Dense validation.** The receptive field (393) is shorter than a window (400). So the forecast for the window
  ending at sample a depends only on samples a - 393 ... a, wherever the window starts, and one pass along the whole
  recording gives every anchor's forecast.

## Measured (RTX 4060 Ti, 1,600 train / 240 validation users, torch 2.7.1, deterministic float32)

| | `best` | `v10_speed` | Factor |
| --- | ---: | ---: | ---: |
| Validation, stride 5 (1,838,498 windows) | 409 s | 5.2 s | 78x |
| Validation, stride 1 (9,191,096 windows) | about 34 min | 6.5 s | about 300x |
| Training step, batch 256 | 143.7 ms | 17.9 ms | 8.0x |
| 40 x 1,000-step epoch | 550 s | about 23 s | about 24x |
| Peak GPU memory in a training step | 2.45 GB | 1.32 GB | |

Real-data check: the dense score of `moredata_1600_k8_s42` is 4.614193412 deg at stride 5 and 4.615883569 deg at
stride 1. The stored values are 4.614193412 and 4.615883568. The windowed time is extrapolated from 100 batches.

## Safety

- Every run's first validation compares the dense forecasts with windowed ones on its first 8,192 windows and stops
  the run if any quaternion component differs by more than 1e-4.
- `test_v10.py` (19 tests): equality with `LeanTCNN` for forecasts and gradients on eight layouts, dense against
  windowed forecasts with another recording in front, the anchor list, chunk sizes, the host-memory starts, and
  tiny training runs with each switch.

## Run

```
python -m unittest experimental.v10_speed.test_v10 -v
python -m experimental.v10_speed.run --stage screen --output outputs/v10_speed
```

The frozen runner still scores each seed with the numpy evaluator at stride 1 and tunes the damped baselines once
per suite folder. Both are now the slow parts of a screen.

## Open

- Control against `best`: the 1,600-user seed-42 control scored 4.6159 deg on full validation with `LeanTCNN.forward`.
  The matching `v10_speed` run is in progress (9 Nov).
- Not yet timed on the 3080 or Colab.
