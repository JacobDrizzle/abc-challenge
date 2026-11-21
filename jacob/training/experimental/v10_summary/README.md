# v10_summary: running averages instead of a long history buffer

Written by Jacob with AI help on 13 Nov 2026, for review. Built on `best/`; nothing in `best/` is edited.

## Question

In the streaming graph, k8_combined's last block (kernel 8, dilation 47) keeps its last 329 inputs for all 64 channels
as state. That is 21,056 of the model's 25,088 state floats (84 %), and every call passes the whole state in and out
and shifts it by one sample. On the STM32N6 (ST Edge AI Developer Cloud, 13 Nov) that copying was the largest single
cost, and the challenge warns that memory-bound models time worse than their MACs suggest. Can the last block keep a
few running averages per channel instead, at the same parameter count, without losing accuracy?

## Change

One config key, `summary_taus`: `null` (best's model exactly), or J time constants in samples. The last block becomes a
`SummaryBlock`. For each channel it reads its input now and J exponential moving averages of its inputs,
`s_j(t) = a_j s_j(t-1) + (1 - a_j) x(t)` with `a_j = exp(-1/tau_j)`, in place of 8 taps spaced 47 apart. The rest is
the KernelBlock arithmetic: depthwise weights over the J + 1 taps, pointwise conv, ReLU, residual, ReLU. Mirror
equivariance is kept (the averages are the same on both streams).

| | k8_combined | `summary_tau4_64` |
| --- | ---: | ---: |
| Last block reads | 8 taps, 0 to 329 samples back | input now + 7 averages, tau 4 to 64 samples |
| Parameters | 7,555 | 7,555 |
| State floats (blocks 0, 1, 2) | 448 + 3,584 + 21,056 = 25,088 | 448 + 3,584 + 448 = 4,480 (−82 %) |
| Streaming update of block 2 | shift a 329-sample buffer | one Mul and one Add per average |

The time constants are geometric, 4 · 16^(i/6) for i = 0 … 6: 4, 6.35, 10.08, 16, 25.4, 40.32 and 64 samples.

### Training windows against streaming

Training sees 400-sample windows; the harness streams whole sessions. In a window, the block's input is valid from
position 64 on (blocks 0 and 1 reach 63 samples back, and the oldest feature row has no rate), so the averages are
taken over those 336 positions with weights `(1 - a) a^k` renormalised to sum to 1. A stream gives its older inputs
a total weight of `a^336`: 0.5 % for tau 64. That is why the longest time constant is 64. With random weights, the
windowed and streamed forecasts differ by at most 0.0011° at step 20 (tau up to 64) but 0.027° with tau 256
(`test_windowed_forecasts_are_the_streamed_ones`, and a check on 13 Nov). The trained model is checked under Results. `SummaryTCNN.streamed(q)` computes the
forecasts exactly as a streaming graph would, from zero state at the session start.

The averages also start from zero in a stream, so warm-up is about 5 × 64 = 320 samples after the 63 samples of
blocks 0 and 1. That is shorter than k8_combined's 393.

## Streaming ONNX export

`export.py` (13 Nov): `StreamingSummary` is `best.export.StreamingLean` with the summary block's state holding its
J averages, [1, K, 2, J], slowest first. Each call computes `averages = state * a + x * (1 - a)`, and the block
reads `Concat(averages, x)` with a (J + 1)-tap depthwise Conv and no dilation. `state_out` is `averages`. Blocks
0 and 1, the features, the head and the output are StreamingLean's default graph, and `st_layout` is kept.
fused_head, padded_rows and split_mirror are not supported. Against k8_combined's graph this is two more Muls and
one more Add, one Slice fewer, and `state_in_2` [1, 32, 2, 7] instead of [1, 32, 2, 329]. All the ops are on the
challenge's safe list. `test_onnx_stream_is_the_streamed_model` checks that the ONNX Runtime stream equals
`SummaryTCNN.streamed`. ST files: `outputs/st_dc_upload/v10_summary/` (see its README; int8 +0.022°).

## Not done

- Learned time constants, other tau sets, or applying summaries to block 1 as well.
- No latency measurement yet (ST boards: 13 Nov, by Jacob).

## Run

```
python -m unittest experimental.v10_summary.test_v10 -v
python -m experimental.v10_summary.run --stage screen --output outputs/v10_summary_screen100
```

## Results

Screen on the 100-user file (`data/h5/qualified_head_100hz_20261026_131953.h5`), seed 42, 40 epochs × 1,000 steps,
full stride-1 validation (15 users, 657,189 windows), 13 Nov. Output: `outputs/v10_summary_screen100/` (log
`outputs/v10_summary_screen100_run.log`). The control reproduces k8_combined seed 42 exactly
(`outputs/v9_confirm/k8_combined/seed_42`, 4.247105).

| Validation, degrees | control (k8_combined) | `summary_tau4_64` | change |
| --- | ---: | ---: | ---: |
| Mean step-20 error (selection metric) | 4.2471 | 4.2354 | −0.0117 |
| Same, streamed from each session's start | 4.2471 | 4.2350 | −0.0121 |
| Median | 3.4114 | 3.4056 | −0.0058 |
| p95 / p99 | 10.142 / 16.944 | 10.112 / 16.785 | −0.030 / −0.159 |
| Step 10 (100 ms) | 2.1159 | 2.1072 | −0.0087 |
| Mean over all 20 steps | 2.1826 | 2.1756 | −0.0070 |
| Equal-weight user mean | 4.6634 | 4.6197 | −0.0437 |
| Users better | | 9 of 15 | Wilcoxon p = 0.36 |
| Training time per epoch (RTX 3080) | 67–70 s | 46 s | |

- **One user carries most of the gain.** User `daa4c4dd` (9,975 windows, the hardest: 10.69° → 10.11°) gives
  −0.0089° of the −0.0117°. Without that user the change is −0.003°. On the other 14 users it is neutral: 8 better
  and 6 worse, each by at most 0.06°.
- **Streaming matches training.** Scored as a stream (the averages run from zero state at each session's start, by
  `SummaryTCNN.streamed`'s recursion), the trained model's errors differ from the windowed ones by −0.0004° on
  average. Per window, the median difference is 0.004°, p99 0.038° and max 0.23°. k8_combined's streaming graph equals its
  windowed forward at these anchors (its receptive field, 393, fits in the window), so the control needs no rerun.
- **The training speed-up is not a deployment number.** In training, the summary block forms only the last
  position's averages (one matrix product), where k8's last block convolves every position in the window.
- Single seed, 100 users. The k8 controls for seeds 43 and 44 already exist (4.2547, 4.2485 in `v9_confirm`), so a
  3-seed check needs two more summary runs. Exported 13 Nov (below); not timed yet.

Scripts used for the streamed check and the per-user table (13 Nov, kept outside the repo): `summary_streamed.py`,
`summary_compare.py`.
