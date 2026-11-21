# Model card: Motion_Pending head-pose predictor

Final model: **`final_k8_combined_v10_s42`** (k8_combined architecture, v10 training recipe, seed 42).
File: `model_fp32.onnx`, sha256 `28221e1e0b559e8d4d900f8b448acd22a0fbe0da38f904cb12d4e268aca53cad`.

## What it does

Input: the last 400 head orientations (4 s at 100 Hz) as unit quaternions (w, x, y, z), head → world, in the frame
given in [`README.md`](README.md). Output: the next 20 orientations (10 ms apart, 200 ms in total). Scored at the 20th.

## Architecture

A causal temporal convolutional network (TCN) that runs as a stream: one new sample in per call, with the history
kept in state buffers.

1. **Front end.** From the last two quaternions: the rotation between them (angular velocity in the body frame),
   gravity in the head frame (`R(q)ᵀ ŷ`) and a speed feature. 7 features per stream.
2. **Mirror equivariance.** Every feature is computed for the head and for its left-right mirror (y and z flipped),
   so the network has 2 streams that share all weights. The output is put back together with the mirror signs, so a
   mirrored input gives an exactly mirrored forecast. This halves what the network has to learn.
3. **Projection** to 32 channels per stream, then **3 separable causal blocks** (kernel 8, dilations 1, 8, 47). Each
   block is a depthwise convolution over time, then a pointwise convolution that mixes the 2 streams and 32 channels,
   with LayerNorm and ReLU and a residual connection. The receptive field is 1 + 7 × (1 + 8 + 47) = 393 samples.
4. **Head.** A linear readout to 3 coefficients per rotation axis on a polynomial time basis (t, t², t³ over the 20
   steps), added to a damped constant-velocity prior (τ = 100 ms). The forecast is a rotation applied to the last
   input quaternion, so the output is always unit length.

## Size and compute

| | Value |
| --- | ---: |
| Trainable parameters | **7,555** |
| Stored in the ONNX as initializers | 7,552 |
| MACs per streaming call | **16,222** |
| ONNX file | opset 17, float32, 124 nodes (including Constants) |
| Operators | Add, Concat, Constant, Conv, LayerNormalization, MatMul, Mul, Relu, Reshape, Slice (no Clip) |
| State | 3 tensors, 25,088 floats (98 KB) |
| Context actually used | 394 samples (3.93 s). Each call reads the newest 2 rows of `context`; older history is held in state |
| Quantisation, pruning, distillation | None. `model_fp32.onnx` (float32) is the only ONNX file and the primary submission |

**Why 7,555 and 7,552.** The 3 trainable x-axis biases of the head are written into a `Constant` node at export
(`/Constant_53`, a [1, 3, 3] tensor with the 3 biases in its first column), not into an initializer. 7,552 + 3 = 7,555.

**MACs, worked.** We count multiply-adds in MatMul, Gemm and Conv only, the usual convention. Element-wise ops,
LayerNorm and the quaternion arithmetic are not counted. Shapes are measured by running the graph once.

| Part | Nodes | Shape | MACs |
| --- | --- | --- | ---: |
| Front end (quaternion delta, gravity, speed) | 6 MatMul | small fixed products | 168 |
| Projection | 1 Conv | 32 out × 7 in × 2 streams | 448 |
| Block 1, 2, 3 depthwise | 3 Conv | 32 ch × 2 streams × 8 taps, each | 3 × 512 = 1,536 |
| Block 1, 2, 3 pointwise | 3 Conv | 64 out × 64 in (32 ch × 2 streams), each | 3 × 4,096 = 12,288 |
| Head readout | 2 MatMul | [9, 32] × [32, 2], then [9, 2] × [2, 1] | 594 |
| Time basis and prior | 1 MatMul | [20, 7] × [7, 3] | 420 |
| Forecast rotation onto the last input | 2 MatMul | [1, 4] × [4, 32], [20, 4] × [4, 8] | 768 |
| **Total** | | Conv 14,272 + MatMul 1,950 | **16,222** |

## Streaming state and warm-up

Inputs: `context` [1, 400, 4], `state_in_0` [1, 32, 2, 7], `state_in_1` [1, 32, 2, 56], `state_in_2`
[1, 32, 2, 329]. Outputs: `prediction` [1, 20, 4] and `state_out_0..2`. Each state is one block's past inputs
(kernel − 1 = 7 taps × dilation). States start at zero and depend only on earlier samples of the same session.

**Warm-up: 392 calls. Please start scoring at call 393 of each session**, counting the first call with a full
400-sample window as call 1. After a reset the state holds zeros, not history. From call 393 (the receptive field)
the zero start no longer affects the output: the same stream started from zero states and from random states gives
bit-identical predictions from call 393 on. `run_inference.py` handles this: on any window that is not the previous
one shifted by one sample, it resets and replays the window from its first sample, so the state is always warm by the
last sample and the answer does not depend on what came before. On the bench, streaming from segment start scores
4.1836° and a cold start at the first full window 4.1946°.

## Training

### Data preparation

BOXRR-23 only, no other data. The cleaning and the filter (steps 2 and 3) follow the challenge's Data page.

1. **Users.** Users (one archive each) are drawn at random from the Hugging Face mirror of BOXRR-23
   (`cschell/boxrr-23`) and decoded with XROR. Tilt Brush recordings and recordings without a head device are
   skipped. Only the head rotation is read, in the recorded frame (no axis remapping), and reordered to (w, x, y, z).
2. **Cleaning.** Samples with a non-finite time or quaternion, or a quaternion of length ≤ 1e-8, are dropped, and the
   rest are scaled to unit length. The first 5 s are trimmed, counting forward time steps only, so a later clock
   reset does not trim again. Of adjacent samples with the same timestamp, the first is kept. Samples are never
   re-sorted.
3. **Challenge filter.** A recording is kept only if its rate, from the median positive time step, is 85–110 Hz,
   and its head moves enough. For each axis of the xyz Euler angles, over non-overlapping 5 s windows, we take the
   sweep (90th minus 10th percentile) and the angular velocity (summed angle changes, wrapped, ÷ 5 s). The median
   over windows, then the largest axis, must exceed 17° for the sweep and 15°/s for the velocity (strict >).
4. **Resampling.** Recordings are split at backward clock steps and at gaps over 100 ms. Each piece is resampled to
   an exact 100 Hz grid by SLERP between recorded samples, never extrapolating past its ends, with signs aligned so
   consecutive quaternions lie in the same hemisphere. Pieces under 1,000 samples (10 s) are dropped.
5. **Per user.** Up to 5 qualifying recordings per user, taken in a seeded order.
6. **Split.** Users are shuffled with a fixed seed (42) and assigned to training, validation and test in the ratio
   20:3:4, so no user is in two splits. Recordings were also checked by SHA-256, so none appears twice.

| Split | Users | Recordings | Segments | Samples (100 Hz) |
| --- | ---: | ---: | ---: | ---: |
| Training | 5,243 | 13,814 | 14,482 | 210.7 M |
| Validation | 786 | 2,072 | 2,154 | 31.3 M |
| Test | 1,049 | 2,724 | 2,884 | 41.6 M |

The only augmentation is the tilt in the table below, applied during training. Mirror equivariance is built into the
architecture, not added as augmentation.

### Training run

| | |
| --- | --- |
| Data | `moredata_final.h5`, prepared as above |
| Split, by user | 5,243 train (210.7 M samples) / 786 validation / 1,049 test. No user in more than one split |
| Steps | 40 epochs × 3,162 = 126,480 steps, batch 256, on a Colab A100 (deterministic float32) |
| Optimiser | lr 1e-3, weight decay 1e-4, gradient clip 1.0, EMA of weights 0.999, dropout 0 |
| Loss | geodesic angle: 0.5 × mean over the 20 steps + 0.5 × the 20th step |
| Augmentation | half the windows tilted (history and future alike) by 0–15° about a random horizontal world axis |
| Seeds | 42, 43, 44 |
| Selection | lowest step-20 error on the full validation split (stride 1), seed breaks ties |

The full config is `weights/config.json`, and the code and data hashes are in `weights/provenance.json`.

## Results

| Set | Users | Error at 200 ms (°) |
| --- | ---: | ---: |
| Validation (stride 5, training run) | 786 | 4.542 (s42), 4.553 (s43), 4.546 (s44) |
| Validation (stride 1, selection) | 786 | **4.5428** (s42, picked), 4.5533 (s43), 4.5470 (s44); hold-last 6.753 |
| Test, scored once after selection (40.4 M windows) | 1,049 | **4.4979** (median 3.60, p95 10.90); hold-last 6.700, constant velocity 7.427 |
| Team bench `group-a-v2-test20` (20 of the test users) | 20 | 4.1836 (median 3.49, p95 9.68) |

## Latency

| Machine | Runtime | Per call |
| --- | --- | ---: |
| AMD Ryzen 7 5800X3D (desktop PC) | ONNX Runtime CPU, 1 thread | **78 µs median (p99 88 µs)** |
| AMD Ryzen 7 5800X3D (desktop PC) | ONNX Runtime CPU, 4 threads | 92 µs median (p99 111 µs) |
| AMD Ryzen 9 7900X (desktop PC) | ONNX Runtime CPU, 1 thread | 54 µs median (p99 89 µs) |
| AMD Ryzen 9 7900X (desktop PC) | ONNX Runtime CPU, 4 threads | 63 µs median (p99 75 µs) |
| Apple M5 Pro | ONNX Runtime CPU, 1 thread | 35 µs median (p99 55 µs) |
| Samsung Galaxy S23 (Qualcomm AI Hub) | ONNX Runtime CPU | 77 µs |
| Samsung Galaxy S23 (Qualcomm AI Hub) | QNN on the Hexagon NPU, float16 | 117 µs |
| Samsung Galaxy S22 (Qualcomm AI Hub) | ONNX Runtime CPU | 66 µs |
| Samsung Galaxy S22 (Qualcomm AI Hub) | QNN on the Hexagon NPU, float16 | 144 µs |

AI Hub reports "estimated inference time" for the runtime call only. On the phone CPU the outputs match the Mac to
3e-6°. **The float16 NPU paths do not:** with warm states they are up to 2.0° off (the CPU path matches, so the
most likely cause is float16 precision, not the graph). Run the model in float32 on the CPU, as `run_inference.py` does.

## Limits and failure modes

- **Frame convention.** The model expects BOXRR-23's frame (y up, left-handed, head → world). A wrong up axis or a
  handedness flip feeds it the wrong gravity and mirror features and makes it worse (on this model, +0.83° for
  z-up and +0.16° for a handedness flip, on the 20-user bench). See the README's conversion section.
- **Sudden, large movements.** The worst 5 % of bench windows are above 9.7° at 200 ms. A turn that starts inside
  the 200 ms has no sign in the past 4 s, so no causal model can predict it.
- **One game dominates the data.** Most of BOXRR-23 is Beat Saber, so motion from other kinds of apps may be predicted
  less well.
- **Sample rate.** Trained at 100 Hz. A different rate changes the angular velocities the model sees.
- **Uneven timing.** The organisers say the test stream is about 100 Hz with some jitter, but every training window
  was resampled to an exact 100 Hz grid. On validation users, this architecture (trained on 800 users) loses 0.05°
  when each sample is taken up to ±1 ms off its tick and 0.16° at ±2 ms; damped velocity loses 0.00° and 0.02°.
  With 1% of samples repeated, the 100-user model lost 0.04° (damped velocity 0.02°). Training with ±1.5 ms jitter
  halved the ±2 ms cost but was 0.01° worse on an exact grid; the submitted model was trained without it.
- **Sensor noise.** The model reads velocity from consecutive samples, so noise in each sample is amplified. Random
  noise of 0.02° RMS per sample costs 0.05°, and 0.05° RMS costs 0.21° (damped velocity 0.01° and 0.05°; same
  800-user model). Keep the input in float32: rounding the sample input to float16 moved its 200 ms forecast by 0.24°.
- **Invalid input.** The model does not normalise its input. On the sample input, a 1% error in quaternion length
  moved the forecast by 0.4–0.5°, and 0.1% by up to 0.04°. The sign of each quaternion does not matter (q and −q,
  even mixed in one window, give the same output). In streaming mode one NaN sample makes the next 394 outputs
  non-finite, until it has passed through all three state buffers. A zero quaternion as the newest sample gives an
  all-zero forecast.
- **Weakest users.** On the 1,049 test users the model beats damped velocity for 1,047 and hold-last for 1,048. Its
  gain over damped velocity is smallest for slow movers (error ratio 0.846 for the slowest third of recordings,
  0.805 for the fastest), on Pico and Windows Mixed Reality headsets (0.864 and 0.844, against 0.827 on Quest 2) and
  for recordings captured at 85 or 110 Hz (0.858 and 0.865, against 0.828 at 90 Hz). These ratios are for unseen
  users of the 100-user model, not the submitted one.
- **float16 and the NPU.** See Latency.
- **ST Edge AI Core 4.1.** Compiled with ST Edge AI Core 4.1 (STM32N6, H7, SR6), graphs of this architecture give
  wrong predictions (cosine 0.80 against ONNX Runtime). The compiler stores the LayerNormalization input [1, 20, 8]
  channel-last and then normalises the wrong groups of 8. Running the same LayerNormalization on [1, 1, 20, 8],
  between two Reshapes, gives correct output on all three boards and leaves ONNX Runtime's output unchanged. We can
  supply that graph.

## Not submitted: a reduced-state variant

Most of this model's state is block 3's history: 21,056 of its 25,088 floats (84%), passed in and out on every
call. On ST boards, moving that state cost more time than the arithmetic. We built a variant in which block 3 keeps
7 exponential moving averages per channel instead, `s ← a·s + (1 − a)·x` with `a = exp(−1/τ)` and τ from 4 to 64
samples. The block reads the current input plus the 7 averages with the same depthwise and pointwise arithmetic.

- Same 7,555 parameters. State 4,480 floats instead of 25,088 (−82%). The only added ops are Mul and Add.
- Trained only on a 100-user subset (seed 42), against this architecture trained the same way: validation error
  4.235° vs 4.247° (15 users, 657,189 windows), so no loss of accuracy. Streamed from session start, it gives the
  same result.
- There was not time to retrain it on the 5,243 users before the deadline, so it is not submitted.

Latency on ST Edge AI Developer Cloud (ST Edge AI Core 4.1, batch 1). These boards are not the challenge's XR target:

| Board | This model, fp32 | This model, int8 | Variant, fp32 | Variant, int8 |
| --- | ---: | ---: | ---: | ---: |
| STM32N6570-DK | 2.63 ms | 1.58 ms | 1.71 ms | 1.38 ms |
| STM32H7P5I-DK | 2.25 ms | 2.12 ms | 1.09 ms | 0.93 ms |
| SR6P3E-EVBC400P | 1.67 ms | 1.79 ms | 0.88 ms | 0.82 ms |

- **"This model", fp32:** measured on graphs of the same architecture. The H7 and SR6 runs were made without the
  LayerNorm fix above, which adds about 0.05 ms.
- **int8:** every int8 column uses the fixed graph. No int8 file is submitted.
- **Correctness:** the variant's fp32 output on all three boards matches ONNX Runtime (cosine 1.000000).

The variant keeps the same interface: `state_in_2` becomes [1, 32, 2, 7]. Once retrained, it can replace this file
directly.
