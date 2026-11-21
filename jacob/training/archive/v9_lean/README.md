# v9: lean graph, longer receptive fields, recipe options

The v9 plan of 28 Oct 2026 was "more accuracy, less latency, safe ops only". It has four tracks:
- **A**: a cheaper streaming graph for the same weights.
- **B**: longer receptive fields and recipe options, screened on seed 42.
- **C**: distillation from an ensemble.
- **D**: confirming the combined candidates on seeds 42–44.

On top of these, `timing_stress.py` (29 Oct) measures what uneven sample timing costs, after the organisers said the
test stream has "some jitter".

v9 runs the unchanged v5 training, validation and benchmark scripts through `hooks.installed()`, as v6 and v7 do. No
v5, v6 or v7 file is edited. With every v9 option at its default, training is bit-identical to v7 (a test checks this).

Only the v9 files that can change a training or validation number are hashed into runs: `__init__`, `configuration`,
`hooks`, `model` and `training`. Don't edit those while a run is using them. `export.py` is not hashed; instead,
every export is checked against its PyTorch model by stream parity.

## Model options (`model.py`)

| Key | What it does |
| --- | --- |
| `kernel_sizes`, `dilations` | One per block, for separable blocks. The receptive field is 1 + Σ (k − 1) d. v7's best sees 243 of the 400 samples allowed. Fewer, wider blocks reach further for fewer graph nodes. |
| `maps` | How per-step rates come in and forecast rotations go out. For a step d = conj(q[t−1]) q[t] = (w, v) of angle θ and a forecast rotation vector r: |
| | `"exact"` (v7): rate = log(d) = θ n; quaternion = exp(r). |
| | `"lean"`: rate = 2 w v (1 + 2/3 \|v\|²) and quaternion along (2 − \|r\|²/6, r), second-order series of the exact maps. The streaming graph computes these for exact-map checkpoints. |
| | `"native"`: rate = 2 w v = sin θ n; quaternion along (2, r), the Cayley map. A model trained with these needs neither series. |

## Recipe options (`training.py`)

All of them are off by default.

| Key | What it does |
| --- | --- |
| `sampling` | `"user"` (v5–v8): a uniform user, then a uniform window of that user. `"sqrt"`: users in proportion to the square root of their window count. `"window"`: uniform over windows. On 27 Oct, A1 found that user balance re-draws the small users about 21× a window, and the model memorises them. |
| `tilt_augmentation_deg` | Half the windows are turned by a uniform 0..max angle about a random horizontal world axis (+y is up). Only gravity in the head frame moves. This models a tilted world frame or an unusual posture (the worst validation user sits 26° back). It uses its own generator. |
| `jitter_augmentation_ms` | Each window is sampled at times k + e_k: a per-window half-width uniform in 0..max ms, and e_k uniform within it, by SLERP. The target is the jittered window's 20th next sample, as the organisers score it. It uses its own generator. |
| `distill_beta`, `teacher` | The loss becomes (1 − β) loss(truth) + β loss(teacher). `distill.py` caches the teacher's forecasts. |

## Export (Track A, `export.py`)

The interface is v7's: `context [1, 400, 4]` (only the last two samples are read) and `state_in_i` in;
`prediction [1, 20, 4]` and `state_out_i` out. On the local CPU, latency follows the node count (about 0.7 µs a
node), so the graph cuts nodes:

- **A1, rate**: the lean rate is one MatMul of d ⊙ d, using w² + \|v\|² = 1 for unit inputs. With native maps, w v is
  already the rate.
- **A2, output**: (2 − \|r\|²/6, r), or (2, r) with native maps, replaces the exp map's 18 nodes.
  - The output is normalised in the graph: one MatMul gives [p, −p], and a LayerNormalization with scale ½ turns that
    into p / \|p\|.
  - This matters because `baseline_pipeline` refuses predictions whose norm is off by more than 1e-3, as a strict
    harness might.
- **A3, mirror layout**: the two streams are the rows of `[1, K, 2, T]`. A block is a depthwise (1, k) Conv, then a
  (2, 1) Conv over the rows [h0, h1, h0]. That gives [[A, B], [B, A]] in one Conv, without v7's four swap nodes.
- **Head**: one MatMul, a sign mask and a ones MatMul. The biases and the damped prior's rate are extra rows of the
  basis coefficients.
- **Options, off by default**: adopting either is a team decision.
  - `padded_rows`: a (3, 1) Conv with kernel rows [B, A, B] over [0, h0, h1, 0] drops the Slice and Concat, for 2K²
    more MACs a block.
  - `fused_head` (A4): the head becomes one (2, 1) Conv.
  - Both store each tied weight once and build the copies with Concat nodes. ONNX Runtime folds those when the
    session is created; a runtime without constant folding would run them on every call.
- The v9 graphs contain no Clip, and only operators from the safe list.

## Files

| File | Role |
| --- | --- |
| `model.py` | `LeanTCNN`: block layouts and maps on top of v7's `CompactTCNN`. `deployed(model)` gives the lean-map model a graph reproduces. |
| `configuration.py` | Schema for the v9 keys; checks the real receptive field |
| `training.py` | The recipe options: `TrainingWindows` (sampling, tilt, jitter, teacher) and the distillation loss |
| `hooks.py` | Swaps the v9 model, schema, trainer pieces and exporter into the frozen v5 scripts |
| `run.py` | Screen a suite (`--stage plan` or `screen` only) |
| `confirm.py` | Confirm candidates on seeds 42–44 against the reference, from a `--plan` file |
| `distill.py` | C1: cache the teacher's forecasts for every training window, in `outputs/v9_teacher/` |
| `export.py` | The lean streaming graph |
| `benchmark.py` | `benchmark_latency.py` with the v9 graph; `--padded-rows`, `--fused-head` |
| `timing_stress.py` | Validation streams with jitter, rate offsets, drops and repeats |
| `test_v9.py` | 28 tests |

| Config | What |
| --- | --- |
| `tcnn_v9.json` | Base config: v7 sep_base3_addons with the angular loss (the 28 Oct reference) |
| `v9_screen.json` | B screen: control, rf_397, rf_324, sampling_sqrt, sampling_window, tilt_15, endpoint_1, native_maps, k5_4blocks, k8_3blocks_w32 |
| `v9_distill.json` | C2: k8 with β = 0.5 and β = 1.0 |
| `v9_jitter.json` | Jitter screen: k8 + 3 ms |
| `v9_confirm.json` | D: k8_combined and k5_combined (window sampling, native maps, tilt 15°) against sep + angular |
| `v9_jitter_combined.json` | k8_combined + 1.5 ms and + 3 ms jitter |

## Run

From `jacob/training/`, one GPU job at a time:

```bash
python -m experimental.v9_lean.run --stage screen        # defaults: v9_screen.json on tcnn_v9.json, into outputs/v9_screen
python -m experimental.v9_lean.run --stage screen --suite experimental/v9_lean/configs/v9_jitter_combined.json --output outputs/v9_jitter_combined
python -m experimental.v9_lean.distill                   # once, before v9_distill.json
python -m experimental.v9_lean.confirm --plan experimental/v9_lean/configs/v9_confirm.json [--dry-run]
python -m experimental.v9_lean.benchmark --checkpoint outputs/v9_confirm/k8_combined/selected_model.pt [--padded-rows] [--fused-head]
python -m experimental.v9_lean.timing_stress --name <run> k8=v9_confirm/k8_combined/selected_model.pt [--reference k8]
python -m unittest experimental.v9_lean.test_v9 -v
```

- `timing_stress.py` takes `label=checkpoint` pairs, with each checkpoint relative to `outputs/`. It writes
  `outputs/v9_timing_stress/<run>.{json,md}` and reads only validation users.
- `confirm.py` only reports. The submission choice stays with the team.

## Results (validation, step-20 mean, degrees)

These come from the reports in `outputs/` (gitignored).

**Track A** (sep + angular weights, seed 44):
- The v7 graph has 130 nodes and the lean graph 84, with the same 6,648 stored weights. Accuracy changes by about
  1e-6°.
- The 1-thread medians on jacob-amd (29 Oct) are 91.0 and 72.0 µs.

**B screen** (seed 42, against the control at 4.3386, which reproduces v8 exactly):

| Variant | Error | Δ |
| --- | ---: | ---: |
| k5_4blocks | 4.2898 | −0.049 |
| k8_3blocks_w32 | 4.2911 | −0.048 |
| sampling_window | 4.3094 | −0.029 |
| rf_397 | 4.3098 | −0.029 |
| sampling_sqrt | 4.3109 | −0.028 |
| tilt_15 | 4.3289 | −0.010 |
| rf_324 | 4.3304 | −0.008 |
| native_maps | 4.3314 | −0.007 |
| endpoint_1 | 4.3315 | −0.007 |

**C, distillation, failed**:
- The teacher is a 9-checkpoint ensemble (4.2725), only about 0.02° better than the new k5 and k8 layouts (4.2898 and 4.2911).
- β = 0.5 scored 4.3123 and β = 1.0 scored 4.3240, against k8's 4.2911.

**Jitter screen, failed the rule**: k8 + 3 ms scored 4.3212 (+0.030 on the exact grid).

**D confirm** (3 seeds, against sep + angular at 4.3297):

| Candidate | Params | Seeds 42 / 43 / 44 | Mean ± sd | Δ |
| --- | ---: | --- | ---: | ---: |
| k8_combined | 7,555 | 4.2471 / 4.2547 / 4.2485 | **4.2501 ± 0.0041** | −0.0796 |
| k5_combined | 5,595 | 4.2652 / 4.2714 / 4.2785 | 4.2717 ± 0.0067 | −0.0580 |

Seed 42 was selected for both. They are in `jacob/models/` as lean, padded-rows and padded-rows + A4 graphs. On
test (bench, for reporting), k8_combined scores 4.2666 and k5_combined 4.2871.

**Timing stress** (29 Oct, `outputs/v9_timing_stress/summary.md`):
- Constant rate offsets of 90–110 Hz need no augmentation: the ratio of k8_combined's error to the damped
  baseline's stays at 0.837–0.839, against 0.841 on the exact grid.
- Jitter is the real risk. Each row is the cost against that model's own exact-grid score:

  | Model | Exact grid (paired vs k8_combined) | ±1 ms | ±2 ms | ±3 ms |
  | --- | ---: | ---: | ---: | ---: |
  | k8_combined | 4.2471 | +0.048 | +0.157 | +0.283 |
  | + jitter 1.5 ms | +0.0090 | +0.017 | +0.078 | +0.171 |
  | + jitter 3 ms | +0.0320 | +0.004 | +0.028 | +0.073 |

- Against k8_combined, the 1.5 ms variant breaks even at about ±0.5 ms of uniform jitter, and the 3 ms variant at
  about ±0.9 ms. Each variant has one seed only.

## Open team decisions

- k8_combined or k5_combined.
- The lean, padded-rows or padded-rows + A4 graph (this waits for ST hardware timings).
- Whether to train a jitter variant. If one is picked, confirm it on seeds 43 and 44 (about 2 h) and pair it
  against k8_combined's 4.2501.
- The warm-up to declare in the model card: at least the receptive field, which is 393 samples for k8 and 397 for k5.
