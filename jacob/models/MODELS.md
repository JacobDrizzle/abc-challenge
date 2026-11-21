# Jacob's ONNX models

Every file here was trained with the `jacob/training/` code and exported as a **stateful streaming graph**: one new
sample per call. Their SHA-256s are in `bench/models.json`, and their scores are in `LEADERBOARD.md` (benchmark
`group-a-v2-test20`). Every choice between models was made on validation; the test numbers are for reporting only.

## The submitted model and the larger-data runs

The architecture is v9 k8_combined throughout. Only the training data and the recipe differ. Nic trained these four
models on more users.

| File | Bench name | Training data (train / validation users) | Recipe | Seed | Test 200 ms (°) |
| --- | --- | --- | --- | ---: | ---: |
| `tcnn_v9_k8_combined_moredata_final_v10_s42_stream.onnx` | `final_k8_combined_v10_s42` | `moredata_final.h5` (5,243 / 786) | `experimental/v10_speed`, 40 × 3,162 steps, Colab A100 | 42 | **4.1836** |
| `tcnn_v9_k8_combined_moredata_ga2_stream.onnx` | `moredata_k8_combined_s42` | `moredata_1600.h5` (1,600 / 240) | `best/` | 42 | 4.2365 |
| `tcnn_v9_k8_combined_moredata_s43_ga2_stream.onnx` | `moredata_k8_combined_s43` | `moredata_1600.h5` (1,600 / 240) | `best/` | 43 | 4.2419 |
| `tcnn_v9_k8_combined_moredata_s44_ga2_stream.onnx` | `moredata_k8_combined_s44` | `moredata_1600.h5` (1,600 / 240) | `best/` | 44 | 4.2475 |

- **The submission.** `final_k8_combined_v10_s42` is the submitted model, byte-identical to
  `Submission/Motion_Pending_headpose/model_fp32.onnx` (SHA-256 `28221e1e…`). Seed 42 was picked on validation over
  seeds 42–44 (4.5428° vs 4.5533° and 4.5470°, all 786 validation users).
  - Its model card (`Submission/Motion_Pending_headpose/model_card.md`) gives the full recipe, the data preparation,
    the test result on all 1,049 test users (4.4979°) and latency.
  - The PyTorch checkpoint and its training record are in `Submission/Motion_Pending_headpose/weights/`.
- **Comparing the scores.** All four are scored on the 20 `group-a-v2-test20` users, and none of them trained on
  those users. The validation sets differ, though, so compare these four with the 100-user models below on test
  only.
- Both larger files are described in `ray/README.md`. Their train and validation users are listed in
  `bench/trained_on/moredata_1600_trainval.txt` and `moredata_final_trainval.txt`.

## The 100-user models

| File | Bench name | Model | Graph | Declared params | ONNX initializer elements | MACs per call | State floats | Validation, 3-seed mean ± sd (°) | Test 200 ms (°) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tcnn_v9_k8_combined_ga2_stream.onnx` | `v9_k8_combined_ga2` | v9 k8_combined | lean | 7,555 | 7,552 | 16,222 | 25,088 | 4.2501 ± 0.0041 | 4.2666 |
| `tcnn_v9_k8_combined_padded_ga2_stream.onnx` | `v9_k8_combined_padded_ga2` | v9 k8_combined | padded rows | 7,555 | 7,552 | 22,366 | 25,088 | (same weights) | 4.2666 |
| `tcnn_v9_k8_combined_padded_a4_ga2_stream.onnx` | `v9_k8_combined_padded_a4_ga2` | v9 k8_combined | padded rows + A4 | 7,555 | 7,555 | 22,168 | 25,088 | (same weights) | 4.2666 |
| `tcnn_v9_k5_combined_ga2_stream.onnx` | `v9_k5_combined_ga2` | v9 k5_combined | lean | 5,595 | 5,592 | 12,318 | 19,008 | 4.2717 ± 0.0067 | 4.2871 |
| `tcnn_v9_k5_combined_padded_ga2_stream.onnx` | `v9_k5_combined_padded_ga2` | v9 k5_combined | padded rows | 5,595 | 5,592 | 16,926 | 19,008 | (same weights) | 4.2871 |
| `tcnn_v9_k5_combined_padded_a4_ga2_stream.onnx` | `v9_k5_combined_padded_a4_ga2` | v9 k5_combined | padded rows + A4 | 5,595 | 5,595 | 16,728 | 19,008 | (same weights) | 4.2871 |
| `tcnn_v7_sep_base3_addons_angular_ga2_stream.onnx` | `v7_sep_base3_addons_angular_ga2` | v7 sep + angular | v7 | 6,651 | 6,648 | 13,863 | 11,616 | 4.3297 ± 0.0142 | 4.3289 |
| `tcnn_v6_ga2_stream.onnx` | `v6_tcnn_ga2` | v6 control | v6 | 26,012 | 25,992 (head bias folded into a Constant) | 52,144 | 12,192 | 4.3423 ± 0.0121 | 4.3584 |
| `tcnn_v5_ga2_stream.onnx` | `v5_tcnn_ga2` | v5 control | v5 | 51,900 | 51,900 | 51,920 | 12,192 | 4.4049 ± 0.0042 | 4.4534 |

- The validation column is the recipe's 3-seed mean. Each file holds one seed, the one validation selected (see
  "How they were made").
- The three graphs of one model hold the same weights and give the same predictions: on test, their paired
  difference is 0.0000° and every export matches its PyTorch model to within 1e-4°.

Latency, from one session (jacob-amd, 29 Oct 2026, ONNX Runtime 1.31.0 CPU, `bench/results/latency/jacob-amd_20261029T125659.json`).
It is relative only: the organisers time models on their own XR hardware.

| Bench name | Nodes (in the file / after ORT folds constants) | 1 thread median / p99 (µs) | 4 threads median (µs) |
| --- | ---: | ---: | ---: |
| `v9_k8_combined_ga2` | 60 / 60 | 56.1 / 112.8 | 91.4 |
| `v9_k8_combined_padded_ga2` | 57 / 54 | 48.8 / 99.2 | 75.7 |
| `v9_k8_combined_padded_a4_ga2` | 58 / 51 | 47.2 / 96.8 | 77.7 |
| `v9_k5_combined_ga2` | 69 / 69 | 61.9 / 123.9 | 97.3 |
| `v9_k5_combined_padded_ga2` | 65 / 61 | 53.6 / 106.1 | 85.6 |
| `v9_k5_combined_padded_a4_ga2` | 66 / 58 | 52.2 / 111.9 | 83.8 |
| `v7_sep_base3_addons_angular_ga2` | 130 / 130 | 91.0 / 175.3 | 112.1 |
| `nic_ga2_gru_s43`, for scale | 35 / 35 | 23.6 / 48.6 | 24.6 |

v5 and v6 were timed in an earlier session (`LEADERBOARD.md`, 26 Oct), so their times don't compare with these.

## The models

**v9 k8_combined and k5_combined** (confirmed 29 Oct; k8_combined was promoted to `jacob/training/best/` on 30 Oct, and
the v9 code is archived in `jacob/training/archive/v9_lean/`):
- Inputs per step: the body-frame angular velocity, an L1 speed channel and gravity in the head frame.
- Network: separable dilated causal blocks with exact left-right mirror equivariance (two weight-tied streams), and a
  3-term polynomial head basis. It predicts a residual on a damped-velocity prior (τ = 100 ms).
- Blocks:
  - k8: 3 blocks, kernel 8, dilations 1 / 8 / 47, 64 channels (32 per mirror stream). Receptive field 393 samples.
  - k5: 4 blocks, kernel 5, dilations 1 / 5 / 25 / 68, 48 channels (24 per stream). Receptive field 397 samples.
- Recipe: angular loss, windows sampled uniformly (not user-balanced), 15° tilt augmentation, and "native" maps
  (rate in as sin θ · n, forecast out through the Cayley map).
- k8 vs k5 on validation: −0.0216°, with 14 of 15 users better. k5 has 1,960 fewer parameters. The team submitted
  k8, retrained on more users (see above).

**The three v9 graphs** (`best/export.py`, formerly `v9_lean/export.py`):
- **lean**: the default v9 graph.
- **padded rows**: each block's mirror conv is one (3, 1) Conv over zero-padded rows, instead of a Slice and a
  Concat. That means fewer nodes but 2K² more MACs a block.
- **padded rows + A4**: the same, plus the head fused into one Conv.
- Both options store each tied weight once. Concat nodes build the copies, and ONNX Runtime folds those when the
  session is created. A runtime that doesn't fold constants would run them on every call.
- The submission uses the lean graph.

**v7 sep + angular** (`jacob/training/archive/v7_compact/`, 27 Oct):
- The v7 `separable_base3_addons` model: 5 separable mirror blocks, dilation base 3 (receptive field 243), basis head
  P = 3 and L1 speed. Trained with the angular loss.
- It was the best model before v9, and the v9 candidates were confirmed against it.
- Its graph contains one `Clip`. The organisers asked us to avoid Clip, or to disclose it; the v9 graphs have none.

**v6 and v5** (26 Oct):
- v5: a TCN on body-frame angular velocity and speed, 7 dilated blocks (receptive field 2.56 s), with a residual on
  a damped-velocity prior.
- v6: v5 plus gravity channels and exact left-right mirror equivariance.

## Interface (challenge stateful option)

- Inputs:
  - `context`: float32 `[1, 400, 4]`, unit quaternions (w, x, y, z), with the newest sample last. Only the last two
    samples are read; the history lives in the states.
  - `state_in_0` … `state_in_N`: float32.

    | Model | States | Shapes | Floats |
    | --- | ---: | --- | ---: |
    | v9 k8 (all three graphs) | 3 | `[1, 32, 2, 7]`, `[1, 32, 2, 56]`, `[1, 32, 2, 329]` | 25,088 |
    | v9 k5 (all three graphs) | 4 | `[1, 24, 2, 4]`, `[1, 24, 2, 20]`, `[1, 24, 2, 100]`, `[1, 24, 2, 272]` | 19,008 |
    | v7 sep + angular | 5 | `[2, 24, 2]`, `[2, 24, 6]`, `[2, 24, 18]`, `[2, 24, 54]`, `[2, 24, 162]` | 11,616 |
    | v6 | 7 | `[2, 24, 2·2^i]`, the two mirror branches | 12,192 |
    | v5 | 7 | `[1, 48, 2·2^i]` | 12,192 |

- Outputs:
  - `prediction`: float32 `[1, 20, 4]`, the next 20 orientations (w, x, y, z).
    - v9 normalises its output in the graph, with a LayerNormalization.
    - v7's output is not normalised in the graph, but it is unit length to about 1e-6.
    - v5 and v6 output unit length.
  - `state_out_0` … `state_out_N`: the same shapes as the state inputs.
- Zero the states at session start, call once per new sample, and feed `state_out_k` back into `state_in_k` on
  every call.
- Opset 17, float32 only. No trigonometric operators: polynomials replace them.
  - v9: Add, Concat, Constant, Conv, LayerNormalization, MatMul, Mul, Relu, Reshape and Slice, all on the safe list.
  - v7: the same set without LayerNormalization, plus one Clip.
  - v5 and v6: off the safe list but elementwise: Div, Less, LessOrEqual, ReduceSum, Sqrt, Sub and Where.

## Warm-up

- A prediction depends on the last *receptive field* steps of motion: 393 samples for k8, 397 for k5, 243 for v7, and
  2.56 s for v5 and v6.
- The bench scores two protocols:
  - **from segment start**: states zeroed at the first sample, with the context left-padded. This is the headline.
  - **cold start**: states zeroed at the first full window.
- The organisers' harness gives the first call a full 400-sample window. These graphs read only its last two
  samples, so under that harness the states start empty at the first window. That is the bench's cold-start case
  (k8: 4.2713, against 4.2666 from segment start).
- The submitted model card asks the organisers to start scoring at call 393, after 392 warm-up calls (the k8
  receptive field).

## World frame

- The models expect quaternions as `ray/` exports them from BOXRR-23: `[HEAD.1, HEAD.i, HEAD.j, HEAD.k]`, read as
  (w, x, y, z), with the source frame unchanged.
- The gravity channels (v6, v7, v9) take world **+y** as up, which is gravity-aligned in BOXRR/Unity
  (`gravity_in_head_frame` in `jacob/training/best/model.py`). v5 has no gravity channel.
- The organisers convert their test stream to the frame we document. The exact spec, with handedness, head axes and
  worked examples, is in `Submission/Motion_Pending_headpose/README.md`. A wrong frame is costly: on the submitted
  model, a handedness flip adds 0.16° and a z-up world adds 0.83°.

## Training data (the 100-user models)

- The `/train` (100) and `/validation` (15) users of `data/h5/qualified_head_100hz_20261026_131953.h5`
  (SHA-256 `a56c3a60…`), the ray/ export. Its `/test` users are the benchmark `group-a-v2-test20`.
- The player list is in `bench/trained_on/group_a_v2_20261026_131953_trainval.txt`.
- Every suite screened its variants on seed 42, then confirmed on seeds 42–44, on validation. Test users were
  never read during training or selection.
- The earlier v5/v6, trained on `ABC_local_test/data/tcnn_v4`, were removed on 26 Oct; they are in git history.

## How they were made

The checkpoints are in `jacob/training/outputs/`, which is gitignored (they live on Jacob's PC).

| Bench name | Checkpoint | Seed | Export |
| --- | --- | ---: | --- |
| `v9_k8_combined_ga2` | `v9_confirm/k8_combined/selected_model.pt` | 42 | `best/export.py` |
| `v9_k8_combined_padded_ga2` | the same | 42 | `padded_rows=True` |
| `v9_k8_combined_padded_a4_ga2` | the same | 42 | `padded_rows=True, fused_head=True` |
| `v9_k5_combined_ga2` | `v9_confirm/k5_combined/selected_model.pt` | 42 | `best/export.py` |
| `v9_k5_combined_padded_ga2` | the same | 42 | `padded_rows=True` |
| `v9_k5_combined_padded_a4_ga2` | the same | 42 | `padded_rows=True, fused_head=True` |
| `v7_sep_base3_addons_angular_ga2` | `v8_recipe_sep_base3_addons/angular_loss/selected_model.pt` | 44 | `archive/v7_compact/export.py` |
| `v6_tcnn_ga2` | `v6_group_a_v2/selected_model.pt` | 44 | `archive/v6_gravity_mirror/export.py` |
| `v5_tcnn_ga2` | `v5_group_a_v2/selected_model.pt` | 44 | `benchmark_latency.py` |

To export a v9 graph again, run this from `jacob/training/`. It also checks parity against PyTorch and times the graph:

```bash
python -m best.benchmark --checkpoint outputs/v9_confirm/k8_combined/selected_model.pt [--padded-rows] [--fused-head]
```

- On 30 Oct, `best/` re-exported the three k8 graphs byte-identical to the files here
  (`jacob/ablations/analyses/promotion/`).
- The v7 and v6 graphs need their archived exporters. From a checkout of tag `v9-ablations` (see
  `jacob/training/archive/README.md`), run
  `python -m experimental.v7_compact.benchmark --checkpoint outputs/v8_recipe_sep_base3_addons/angular_loss/selected_model.pt`.
- `best.benchmark` exports the same weights as a lean graph instead.

## Testing them

From the repo root, in the bench environment (`bench/README.md`):

```bash
python -m bench audit jacob/models/tcnn_v9_k8_combined_ga2_stream.onnx
python -m bench score jacob/models/tcnn_v9_k8_combined_ga2_stream.onnx --name v9_k8_combined_ga2 \
    --h5 data/h5/qualified_head_100hz_20261026_131953.h5 --machine nic-mac
python -m bench latency \
    v9_k8_combined_ga2=jacob/models/tcnn_v9_k8_combined_ga2_stream.onnx \
    v9_k8_combined_padded_ga2=jacob/models/tcnn_v9_k8_combined_padded_ga2_stream.onnx \
    v9_k8_combined_padded_a4_ga2=jacob/models/tcnn_v9_k8_combined_padded_a4_ga2_stream.onnx \
    v9_k5_combined_ga2=jacob/models/tcnn_v9_k5_combined_ga2_stream.onnx \
    v9_k5_combined_padded_ga2=jacob/models/tcnn_v9_k5_combined_padded_ga2_stream.onnx \
    v9_k5_combined_padded_a4_ga2=jacob/models/tcnn_v9_k5_combined_padded_a4_ga2_stream.onnx \
    v7_sep_base3_addons_angular_ga2=jacob/models/tcnn_v7_sep_base3_addons_angular_ga2_stream.onnx \
    nic_ga2_gru_s43=nic/models/group_a_v2/s43/gru_stateful_window.onnx \
    --machine nic-mac --threads 1 4
python -m bench compare v9_k8_combined_ga2 v9_k5_combined_ga2
```

- `score` checks the file against its SHA-256 in `bench/models.json` and takes the trained-on list from there. It
  defaults to `--bench group-a-v2-test20`.
- The jacob-amd result JSONs record where each file sat when it was scored (`jacob/training/outputs/v8_bench/` or
  `v9_bench/`). Their SHA-256s are the same as the files here.

## Benchmarked graphs not in this folder

These were scored on jacob-amd and have rows on `LEADERBOARD.md`. Their files stay in `jacob/training/outputs/` on
Jacob's PC, and their SHA-256s are in their result JSONs.

- `v9_sep_angular_lean_ga2`: the sep + angular weights on the v9 lean graph.
- `v7_sep_base3_addons_angular_split_ga2`: the sep + angular weights on the v7 mirror-split graph.
- `v7_base3_5_addons_angular_ga2`, `v7_base3_5_addons_ga2`, `v7_sep_base3_addons_ga2`: the other v7 finalists, with
  and without the angular loss.
- `v9_k8_jitter3_s42_ga2`: the k8 + 3 ms jitter screen, seed 42.
- `v6_control_v7graph_ga2`: the v6 control weights on the v7 graph.
