# best: the current best model

> Working log, kept as written. "Pending" and "team call" describe the state at the time. The submitted model and its recipe are in [`model_card.md`](../../../Submission/Motion_Pending_headpose/model_card.md).

`best/` always holds the best confirmed model and the code to train, confirm, export and stress-test it. It was
promoted on 30 Oct 2026 from v9 (`archive/v9_lean`). Every experiment behind it is logged in
[`jacob/ablations/README.md`](../../ablations/README.md).

## k8_combined

| | |
| --- | --- |
| Validation, step 20 (seeds 42 / 43 / 44) | 4.2471 / 4.2547 / 4.2485, mean **4.2501 ± 0.0041°** |
| Test, bench (reporting only) | 4.2666° |
| Parameters | 7,555 (7,552 stored in the lean graph) |
| Latency, lean graph | 55.2 µs (ORT CPU, 1 thread, jacob-amd, 29 Oct) |
| Receptive field | 393 samples |
| Checkpoint | `outputs/v9_confirm/k8_combined/selected_model.pt` (seed 42; `outputs/` is gitignored) |
| Graphs | `jacob/models/tcnn_v9_k8_combined_ga2_stream.onnx`, `_padded_`, `_padded_a4_` |

The network:
- 3 separable blocks with kernel 8 and dilations 1 / 8 / 47.
- 64 channels, 32 per mirror stream.
- Inputs are body-frame velocity, gravity in the head frame, and an L1 speed channel.
- It is mirror-equivariant, with native maps.
- The head predicts a 3-term polynomial residual on the damped-velocity prior (τ 100 ms).

The recipe:
- Angular loss with endpoint weight 0.5.
- Sampling uniform over windows, with tilt augmentation up to 15°.
- 40 epochs × 1,000 steps × 256 windows, EMA 0.999.

## Where the code came from

The v6, v7 and v9 experiment code was moved into one package, with its behaviour unchanged:
- `model.py` holds the three model classes verbatim, so parameters are created in the same order: GravityMirrorTCNN, then CompactTCNN, then LeanTCNN.
- `configuration.py` keeps the three chained schema functions.

On 30 Oct, the promotion was checked against `archive/v9_lean` while both still existed, and every check was exact:
- The configs.
- The checkpoint outputs.
- The ONNX files.
- A training run.
- The full validation of k8_combined (4.247105).

The script and its log are in `jacob/ablations/analyses/promotion/`.

`configs/base.json` is k8_combined's recorded config with every key written out, so a screen's control reproduces k8_combined seed 42.

Runs hash `run_desktop`'s files plus `best/{__init__,configuration,hooks,model,training}.py`. Don't edit those while a run is using them. A new run's `code_sha256` differs from the v9 runs' only because these paths changed.

## Files

| File | Role |
| --- | --- |
| `model.py` | `LeanTCNN` and the classes it builds on. `deployed(model)` gives the lean-map model a graph reproduces. `load_checkpoint` loads every group-a-v2 checkpoint since v5. |
| `configuration.py` | The schema for every model and recipe key. It checks the real receptive field. |
| `training.py` | The recipe options: `TrainingWindows` (sampling, tilt, jitter, teacher) and the distillation loss |
| `hooks.py` | Swaps this package into the frozen v5 scripts (`train_tcnn`, `run_desktop`, `run_next`, `benchmark_latency`) |
| `run.py` | Screen a suite (`--stage plan` or `screen` only) |
| `confirm.py` | Confirm candidates on seeds 42–44 against k8_combined, from a `--plan` file |
| `export.py` | The lean streaming graph, with the `padded_rows`, `fused_head`, `split_mirror` and `st_layout` options |
| `benchmark.py` | `benchmark_latency.py` with this graph |
| `quantize.py` | Int8 post-training quantisation of the graph's network, scored on validation |
| `distill.py` | Cache a teacher's forecasts for distillation |
| `timing_stress.py` | Validation streams with jitter, rate offsets, drops, repeats and sensor noise |
| `test_best.py` | 40 tests |

| Config | What |
| --- | --- |
| `configs/base.json` | k8_combined |
| `configs/screen.json` | Template suite: the control only, seeds 42–44, with the damped-baseline grid a screen tunes first |
| `configs/confirm.json` | Template plan: reference k8_combined, no candidates yet |

k5_combined is the smaller confirmed alternative: 4.2717 ± 0.0067°, test 4.2871°, 5,595 parameters, 61.9 µs, receptive field 397. As overrides of `base.json` it is:

```json
{"num_blocks": 4, "kernel_sizes": [5, 5, 5, 5], "dilations": [1, 5, 25, 68], "channels": 48}
```

## Run

From `jacob/training/`, one GPU job at a time. `--output` is always a new folder: the frozen runner's own default is the v5 folder.

```bash
python -m best.run --stage plan --output outputs/<suite>                  # print the configs, train nothing
python -m best.run --stage screen --suite <suite>.json --output outputs/<suite>
python -m best.confirm --plan <plan>.json --output outputs/<report folder> [--dry-run]
python -m best.benchmark --checkpoint outputs/v9_confirm/k8_combined/selected_model.pt [--padded-rows] [--fused-head] [--split-mirror] [--st-layout]
python -m best.quantize --checkpoint outputs/v9_confirm/k8_combined/selected_model.pt [--output DIR]
python -m best.timing_stress --name <run> k8=v9_confirm/k8_combined/selected_model.pt [--reference k8] [--data <h5>]
python -m best.distill --output outputs/<teacher folder>                  # then distill_beta and teacher in a suite
python -m unittest best.test_best -v
```

- A suite is `{"version": 1, "seeds": [...], "variants": {"control": {}, "<name>": {<overrides of base.json>}}}`, plus `baseline_lags` and `baseline_tau_ms`. A screen stops without those two, so copy them from `configs/screen.json`.
- A confirm plan names the reference folder and each candidate's `output` and `overrides`.
- `confirm.py` only reports. The submission choice stays with the team.

## Options

Every key below is in `base.json`. A variant overrides it.

**Model**

| Key | k8_combined | What it does |
| --- | --- | --- |
| `num_blocks`, `kernel_sizes`, `dilations` | 3, [8, 8, 8], [1, 8, 47] | One kernel and one dilation per separable block. The receptive field is 1 + Σ (k − 1) d, and must not exceed `effective_context`. |
| `channels` | 64 | Split between the two mirror streams |
| `maps` | `"native"` | How a step d = (w, v) becomes a rate, and a forecast r becomes a quaternion. `"exact"`: log and exp. `"lean"`: second-order series of them. `"native"`: rate 2wv = sin θ n, and output along (2, r) (Cayley). |
| `gravity_features`, `mirror_equivariant` | true, true | Gravity in the head frame as an input. Left-right mirror equivariance by tied weights. |
| `separable`, `dilation_base` | true, 3 | Depthwise + pointwise blocks. `dilation_base` applies only without explicit `dilations`. |
| `head_basis`, `speed_feature` | 3, `"l1"` | Polynomial residual of degree 1..p. The speed channel as an L1 norm. |
| `damped_prior_tau_ms` | 100 | The damped-velocity prior that the residual corrects |

**Recipe**

| Key | k8_combined | What it does |
| --- | --- | --- |
| `loss_kind`, `endpoint_weight` | `"angular"`, 0.5 | Geodesic-angle loss, with extra weight on step 20 |
| `sampling` | `"window"` | `"user"`: a uniform user, then a uniform window. `"sqrt"`: users weighted by the square root of their window count. `"window"`: uniform over windows. A1 (27 Oct) found the model memorises the small users that user balance re-draws. |
| `tilt_augmentation_deg` | 15 | Half the windows are turned by up to this angle about a random horizontal axis. Only gravity in the head frame moves. |
| `jitter_augmentation_ms` | 0 | Each window is resampled at times k + e_k, uniform within a per-window half-width of up to this many ms. |
| `distill_beta`, `teacher` | 0, null | Loss = (1 − β) loss(truth) + β loss(teacher), with a cache from `distill.py` |

**Graph** (`export.py`; every option is off by default, and adopting one is a team call)
- The lean graph:
  - Interface: `context [1, 400, 4]` and `state_in_i` in; `prediction [1, 20, 4]` and `state_out_i` out.
  - The output is normalised in-graph.
  - No Clip, and only safe-list operators.
- `padded_rows`: a (3, 1) pointwise Conv with no row Slice or Concat. 48.3 µs on k8, with +2K² MACs a block.
- `fused_head` (A4): the head as one Conv. With padded rows, 46.9 µs on k8.
  - Both options store each tied weight once and rely on ORT folding constants at load.
- `split_mirror` (new on 30 Oct): v7's mirror split for this layout.
  - P = ½(A + B) on h0 + h1 and Q = ½(A − B) on h0 − h1; the rows are P ± Q.
  - Half the pointwise MACs, 6 more nodes a block, and the same stored weights.
  - It can't be combined with `padded_rows`. Not yet timed.
- `st_layout` (new on 9 Nov): the output LayerNormalization on `[1, 1, 20, 8]`, between two Reshapes.
  - ST Edge AI Core 4.1 miscompiles the default `[1, 20, 8]` layout. On 6 Nov the STM32N6, H7 and SR6 boards gave a wrong prediction (cosine 0.796 with ST's reference). With this layout the N6 gave cosine 1.000000, for +0.05 ms.
  - ONNX Runtime's output is bit-identical with and without it. It adds 2 nodes (62 on k8).
  - The challenge's test hardware is not known, so whether it needs this is not known either.

## Int8 on k8_combined (measured 6 Nov)

`python -m best.quantize --checkpoint outputs/v9_confirm/k8_combined/selected_model.pt`, on all 657,189 validation windows:

| Graph | Step 20 | Against float |
| --- | ---: | ---: |
| float32 | 4.2471° | |
| int8, percentile calibration | 4.2720° | +0.0249° |
| int8, min-max calibration | 4.3488° | +0.1017° |

- Both int8 graphs ran about 60% slower than float on the PC's CPU (0.086 against 0.053 ms). Only a device that runs int8 natively could make them worth it.
- w8a16 was not run on k8 here. E3 (26 Oct) measured it at +0.006° on v6 control.
- The report is `jacob/ablations/results/v9_confirm/k8_combined/int8/quantization.md`.

## The 800-user retrain (2 Nov)

The same config with `steps_per_epoch: 3162` (one pass), trained on `data/h5/qualified_head_100hz_merged_20261101_140915.h5`:

- 120 validation users: 4.4068 ± 0.0039° (seeds 42–44). Seed 44 was selected.
- The 15 validation users above: 4.1836 ± 0.0047°, against k8_combined's 4.2501°. That is −0.0665°, with 15/15 users better.
- Checkpoint: `outputs/v10_k8_800users_full_pass/full_pass/selected_model.pt`.
- It was not promoted. The submitted model is k8_combined retrained on 5,243 users with the `experimental/v10_speed`
  recipe (`final_k8_combined_v10_s42`, see `jacob/models/MODELS.md`).

`experimental/v10_fast/` trains and validates this model about 4 times faster, with the same outputs. See its README.

## Not yet measured on k8_combined

- **Mirror split latency**: `python -m best.benchmark --checkpoint outputs/v9_confirm/k8_combined/selected_model.pt --split-mirror`.
  - Time it only on an idle machine.
  - The deciding numbers are the ST hardware timings.

## Team decisions open after v9 (29 Oct)

All were settled for the submission. The submitted model and its warm-up are described in its model card.

- k8_combined or k5_combined.
- The lean, padded-rows or padded-rows + A4 graph. This waits for the ST hardware timings.
- Whether to train a jitter variant:
  - The 1.5 ms variant passed the screen rule narrowly: +0.009° on the exact grid, breaking even at about ±0.5 ms of jitter.
  - If it is picked, confirm it on seeds 43 and 44.
- The warm-up to declare in the model card: at least the receptive field, 393 samples for k8.

## Next round

New experiments go in `experimental/v10_<name>/` and hook onto this package. See [`../experimental/README.md`](../experimental/README.md). A confirmed winner is promoted into `best/` the same way, and the code it replaces moves to `archive/`.
