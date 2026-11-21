# Latency benchmark: outputs\v5_group_a_v2\selected_model.pt

2026-10-26T20:32:44 UTC · AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD · ONNX Runtime 1.31.0 (CPU) · batch 1 · 3000 timed calls per setting in 3 alternating rounds, 200 warm-up calls before each · stream: validation users in <repo>\data\h5\qualified_head_100hz_20261026_131953.h5

Local CPU timings compare models on this PC. ST's measurement on the XR target is the official figure.

| Model | Threads | Median (ms) | p90 (ms) | p99 (ms) | Max (ms) | Calls/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stream | 1 | 0.094 | 0.096 | 0.113 | 0.187 | 10,638 |
| full | 1 | 0.363 | 0.369 | 0.397 | 0.529 | 2,754 |
| strided | 1 | 0.207 | 0.212 | 0.270 | 0.513 | 4,829 |
| fast | 1 | 0.172 | 0.176 | 0.236 | 0.310 | 5,814 |
| fast_notrig | 1 | 0.203 | 0.208 | 0.323 | 0.671 | 4,926 |
| stream | 4 | 0.096 | 0.100 | 0.146 | 0.289 | 10,384 |
| full | 4 | 0.254 | 0.383 | 0.496 | 0.538 | 3,937 |
| strided | 4 | 0.198 | 0.347 | 0.360 | 0.405 | 5,038 |
| fast | 4 | 0.158 | 0.276 | 0.286 | 0.594 | 6,329 |
| fast_notrig | 4 | 0.192 | 0.290 | 0.302 | 0.425 | 5,222 |
| PyTorch eager (reference) | 1 | 0.870 | 0.981 | 1.532 | 2.364 | 1,150 |

- 1 thread(s), speed-up over full: stream 3.9x, strided 1.8x, fast 2.1x, fast_notrig 1.8x.
- 4 thread(s), speed-up over full: stream 2.6x, strided 1.3x, fast 1.6x, fast_notrig 1.3x.

## Models

All files hold the same 51,900 trained parameters. Stateless graphs process the 256-sample window on every call; the stream graph carries 7 state tensors (12,192 float32 values) and processes one new sample per call.

| Model | What it computes | MACs per step | ONNX (KiB) | Max diff vs PyTorch (°) | Error at 200 ms (°) |
| --- | --- | ---: | ---: | ---: | ---: |
| stream (`model_fp32_stream.onnx`) | stateful: one new sample per call, per-layer inputs cached as state (no trig ops) | 51,920 | 223 | 1.1e-04 | 3.3059 |
| full (`model_fp32_full.onnx`) | every layer computes all time steps (the trained graph as written) | 12,438,336 | 278 | 1.3e-04 | 4.3974 |
| strided (`model_fp32_strided.onnx`) | each layer computes only the time steps the prediction depends on (stride-2 Conv) | 1,807,680 | 235 | 1.3e-04 | 4.3974 |
| fast (`model_fp32_fast.onnx`) | strided, plus quaternion products as MatMul and zero padding inside Conv | 1,828,464 | 221 | 2.1e-04 | 4.3974 |
| fast_notrig (`model_fp32_fast_notrig.onnx`) | fast, with atan/sin/cos as Mul/Add polynomials: no trigonometric operators | 1,828,464 | 224 | 2.1e-04 | 4.3974 |

Stateless graphs: checked on 1,000 random validation windows; PyTorch's own error at 200 ms on them is 4.3974°.

Stream graph: one validation session streamed from its first sample, compared on its 1,000 full windows (different windows, so its error is not comparable with the row above); PyTorch's error there is 3.3059°.

## Operators: stream

- On the safe list: Add, Clip, Concat, Conv, Gemm, MatMul, Mul, Relu, Reshape, Slice
- On the avoid list: none
- On neither list (elementwise; check with organisers if unsure): Div, Less, LessOrEqual, ReduceSum, Sqrt, Sub, Where
- Often slow on NPUs: none
- float64 tensors: 0
- Counts: Constant 93, Mul 30, Add 22, Relu 15, Slice 13, Concat 9, Conv 8, Reshape 7, Div 6, Clip 5, MatMul 4, ReduceSum 4, Where 3, Sqrt 3, Less 1, LessOrEqual 1, Sub 1, Gemm 1

## Operators: full

- On the safe list: Add, Clip, Concat, Conv, Gemm, Mul, Relu, Reshape, Slice, Transpose
- On the avoid list: none
- On neither list (elementwise; check with organisers if unsure): Atan, Cos, Div, Greater, Less, Neg, ReduceSum, Sin, Sqrt, Sub, Where
- Often slow on NPUs: Atan, Cos, Sin
- float64 tensors: 0
- Counts: Constant 140, Mul 47, Slice 27, Add 21, Relu 15, Concat 13, Sub 13, Div 11, Conv 8, ReduceSum 7, Sqrt 7, Clip 7, Where 3, Greater 2, Reshape 2, Neg 1, Less 1, Atan 1, Transpose 1, Gemm 1, Sin 1, Cos 1

## Operators: strided

- On the safe list: Add, Clip, Concat, Conv, Gemm, Mul, Relu, Reshape, Slice, Transpose
- On the avoid list: none
- On neither list (elementwise; check with organisers if unsure): Atan, Cos, Div, Greater, Less, Neg, ReduceSum, Sin, Sqrt, Sub, Where
- Often slow on NPUs: Atan, Cos, Sin
- float64 tensors: 0
- Counts: Constant 168, Mul 47, Slice 34, Add 21, Relu 15, Concat 13, Sub 13, Div 11, Conv 8, ReduceSum 7, Sqrt 7, Clip 7, Where 3, Greater 2, Reshape 2, Neg 1, Less 1, Atan 1, Transpose 1, Gemm 1, Sin 1, Cos 1

## Operators: fast

- On the safe list: Add, Clip, Concat, Conv, Gemm, MatMul, Mul, Relu, Reshape, Slice, Transpose
- On the avoid list: none
- On neither list (elementwise; check with organisers if unsure): Atan, Cos, Div, Greater, Less, ReduceSum, Sin, Sqrt, Sub, Where
- Often slow on NPUs: Atan, Cos, Sin
- float64 tensors: 0
- Counts: Constant 94, Slice 16, Relu 15, Mul 14, Div 10, Add 9, Conv 8, Reshape 6, ReduceSum 6, Sqrt 6, Clip 6, MatMul 4, Where 3, Concat 3, Greater 2, Less 1, Atan 1, Transpose 1, Gemm 1, Sin 1, Sub 1, Cos 1

## Operators: fast_notrig

- On the safe list: Add, Clip, Concat, Conv, Gemm, MatMul, Mul, Relu, Reshape, Slice, Transpose
- On the avoid list: none
- On neither list (elementwise; check with organisers if unsure): Div, Less, LessOrEqual, ReduceSum, Sqrt, Sub, Where
- Often slow on NPUs: none
- float64 tensors: 0
- Counts: Constant 110, Mul 30, Add 22, Slice 17, Relu 15, Conv 8, Div 7, Reshape 6, Clip 5, MatMul 4, ReduceSum 4, Where 3, Sqrt 3, Concat 3, Less 1, LessOrEqual 1, Sub 1, Transpose 1, Gemm 1
