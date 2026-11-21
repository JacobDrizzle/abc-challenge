# Latency benchmark: outputs\v6_group_a_v2\selected_model.pt

2026-10-26T20:32:53 UTC · AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD · ONNX Runtime 1.31.0 (CPU) · batch 1 · 3000 timed calls per setting in 3 alternating rounds, 200 warm-up calls before each · stream: validation users in <repo>\data\h5\qualified_head_100hz_20261026_131953.h5

Local CPU timings compare models on this PC. ST's measurement on the XR target is the official figure.

| Model | Threads | Median (ms) | p90 (ms) | p99 (ms) | Max (ms) | Calls/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stream | 1 | 0.127 | 0.216 | 0.245 | 0.499 | 7,886 |
| stream | 4 | 0.130 | 0.185 | 0.241 | 0.261 | 7,680 |
| PyTorch eager (reference) | 1 | 1.138 | 1.644 | 1.990 | 2.354 | 879 |

## Models

All files hold the same 26,012 trained parameters. Stateless graphs process the 256-sample window on every call; the stream graph carries 7 state tensors (12,192 float32 values) and processes one new sample per call.

| Model | What it computes | MACs per step | ONNX (KiB) | Max diff vs PyTorch (°) | Error at 200 ms (°) |
| --- | --- | ---: | ---: | ---: | ---: |
| stream (`model_fp32_stream.onnx`) | stateful: one new sample per call, per-layer inputs cached as state (no trig ops) | 52,144 | 133 | 7.0e-05 | 3.2917 |

Stream graph: one validation session streamed from its first sample, compared on its 1,000 full windows (different windows, so its error is not comparable with the row above); PyTorch's error there is 3.2917°.

## Operators: stream

- On the safe list: Add, Clip, Concat, Conv, MatMul, Mul, Relu, Reshape, Slice
- On the avoid list: none
- On neither list (elementwise; check with organisers if unsure): Div, Less, LessOrEqual, ReduceSum, Sqrt, Sub, Where
- Often slow on NPUs: none
- float64 tensors: 0
- Counts: Constant 171, Mul 32, Slice 31, Add 24, Concat 23, Relu 15, Reshape 9, Conv 8, MatMul 7, Div 7, Clip 5, ReduceSum 4, Where 3, Sqrt 3, Less 1, LessOrEqual 1, Sub 1
