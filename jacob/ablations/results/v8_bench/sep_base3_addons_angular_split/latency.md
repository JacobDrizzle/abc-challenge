# Latency benchmark: outputs\v8_recipe_sep_base3_addons\angular_loss\selected_model.pt

2026-10-27T23:38:32 UTC · AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD · ONNX Runtime 1.31.0 (CPU) · batch 1 · 3000 timed calls per setting in 3 alternating rounds, 200 warm-up calls before each · stream: validation users in <repo>\data\h5\qualified_head_100hz_20261026_131953.h5

Local CPU timings compare models on this PC. ST's measurement on the XR target is the official figure.

| Model | Threads | Median (ms) | p90 (ms) | p99 (ms) | Max (ms) | Calls/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stream | 1 | 0.102 | 0.115 | 0.182 | 0.426 | 9,814 |
| PyTorch eager (reference) | 1 | 2.418 | 2.844 | 3.558 | 4.282 | 414 |

## Models

All files hold the same 6,651 trained parameters. Stateless graphs process the 256-sample window on every call; the stream graph carries 5 state tensors (11,616 float32 values) and processes one new sample per call.

| Model | What it computes | MACs per step | ONNX (KiB) | Max diff vs PyTorch (°) | Error at 200 ms (°) |
| --- | --- | ---: | ---: | ---: | ---: |
| stream (`model_fp32_stream.onnx`) | stateful: one new sample per call, per-layer inputs cached as state (no trig ops) | 8,103 | 53 | 7.5e-05 | 3.2453 |

Stream graph: one validation session streamed from its first sample, compared on its 1,000 full windows (different windows, so its error is not comparable with the row above); PyTorch's error there is 3.2453°.

## Operators: stream

- On the safe list: Add, Clip, Concat, Conv, MatMul, Mul, Relu, Reshape, Slice
- On the avoid list: none
- On neither list (elementwise; check with organisers if unsure): none
- Often slow on NPUs: none
- float64 tensors: 0
- Counts: Constant 141, Add 36, Mul 33, Slice 23, Conv 16, MatMul 12, Relu 12, Reshape 9, Concat 8, Clip 1
