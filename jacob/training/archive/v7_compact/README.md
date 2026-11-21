# v7: compact TCNN (and the v8 recipe runs)

The v7 plan of 26 Oct 2026 aimed to cut latency, parameters and MACs at the same accuracy as v6. Validation accuracy
had plateaued at a seed spread of about 0.02°, so the remaining gains were in the size and latency criteria. The v8
recipe runs (angular loss, no-gravity hedge) reuse this package, because they only change config keys.

v7 runs the unchanged v5 training, validation, test and benchmark scripts through `hooks.installed()`, as v6 does.
No v5 or v6 file is edited, so their manifests still verify. The v6 and v7 source files are hashed into every run's
`experiment.json` / `suite.json`, so don't edit them while a run is using them.

## Model options (`model.py`, `configuration.py`)

With every option at its default, the model is v6's `GravityMirrorTCNN`.

| Key | Values | What it does |
| --- | --- | --- |
| `dilation_base` | 2 (v6), 3 | Block i has dilation base^i. Base 3 with kernel 3 tiles every lag once, so 5 blocks reach a receptive field of 243 samples, where base 2 needs 7 blocks for 255. |
| `separable` | false (v6), true | Each block becomes a depthwise dilated kernel-3 conv (one filter per channel, shared by both mirror streams), then a pointwise conv. On twin streams the pointwise weight `[[A, B], [B, A]]` is stored once. |
| `head_basis` | null (v6), P | The head emits P coefficients per axis, and step k's rotation vector is Σ c_j (k / horizon)^(j+1). P = 3 corrects the damped prior's rate, acceleration and jerk. |
| `speed_feature` | `"norm"` (v6), `"l1"` | The L1 speed \|wx\| + \|wy\| + \|wz\| can be built from Relu and MatMul alone, so no Sqrt is needed. |
| `loss_kind` | `"quaternion"`, `"angular"` | Defined in `tcnn/configuration.py`. The angular loss is the v8 recipe change. |
| `gravity_features` | true, false | A v6 key. False is the no-gravity hedge. |

`configuration.py` replaces two v5 rules rather than inheriting them: v5 allows only 32 or 48 channels, and it
checks the receptive field for base-2 dilations only.

## Export

- **`export.py`**: the v7 streaming graph. It has the same interface as v6's: `context [1, 400, 4]` (only the last two
  samples are read) and `state_in_i` in; `prediction [1, 20, 4]` and `state_out_i` out. The maths relies on unit-norm
  inputs, in three ways:
  - **Rate**: the rotation vector of each step is a degree-8 polynomial in 1 − cos θ. Its input is clipped at 1 (the
    graph's one `Clip`), so steps up to 90° are exact to float32. The largest step in the data is 66°.
  - **Norms**: squared norms are MatMuls with a ones vector instead of ReduceSum, and gravity skips the division by
    \|q\|².
  - **Output**: not normalised in the graph; it is unit length to about 1e-6.
  - With `speed_feature="l1"`, every operator is on the safe list. `"norm"` keeps one Sqrt.
- **`mirror_split.py`**: an alternative graph that splits each mirror conv into a sum part and a difference part,
  (A + B)/2 on h0 + h1 and (A − B)/2 on h0 − h1. That halves the block conv MACs but adds 20 nodes.
  - On sep + angular, MACs drop from 13,863 to 8,103 with the same test score.
  - The 1-thread CPU median rises from 91 to 104 µs. Whether that pays off depends on ST hardware.
- **`quantize.py`** (E3): post-training int8 quantisation, in ONNX Runtime QDQ format.
  - Only the block convs and the head are quantised. The feature maths, the output maths and the states stay float.
  - Calibration streams train users; the error is measured on every validation window.

## Files

| File | Role |
| --- | --- |
| `model.py` | `CompactTCNN`: the four compression options on top of v6 |
| `configuration.py` | Schema for the v7 keys |
| `hooks.py` | Swaps the v7 model, schema and exporter into the frozen v5 scripts, inside this process only |
| `run.py` | Screen a suite (`--stage plan` or `screen` only) |
| `confirm.py` | Confirm candidates on seeds 42–44 and report them against the tolerance |
| `evaluate.py` | `evaluate_tcnn.py` for a v7 checkpoint |
| `export.py`, `mirror_split.py` | Streaming ONNX graphs |
| `benchmark.py` | `benchmark_latency.py` with the v7 graph; `--mirror-split` exports the split graph |
| `quantize.py` | E3 int8 PTQ, with its validation error |
| `test_v7.py` | 20 tests |
| `configs/` | Base configs, suites and confirm plans (below) |

| Config | What |
| --- | --- |
| `tcnn_v7.json` | Base config (the v6 control settings) |
| `tcnn_v7_base3_5_addons.json`, `tcnn_v7_sep_base3_addons.json` | The two finalists as full configs |
| `v7_run.json` | Screen of 8 variants: control, speed_l1, basis_head, base3_5, width_20, narrow_full, separable_24, narrow_base3 |
| `v7_extra.json` | separable_base3_addons, unscreened in v7_run |
| `v7_confirm.json` | Confirm plan, tolerance 4.382° |
| `v8_recipe_screen.json` | Recipe screen: angular_loss, lr_2e3, epochs_80 |
| `v8_angular_confirm.json` | Both finalists with the angular loss, on 3 seeds |
| `v8_no_gravity_confirm.json` | sep + angular with and without gravity, on 3 seeds |

## Run

From `jacob/training/`, one GPU job at a time:

```bash
python -m experimental.v7_compact.run --stage screen        # defaults: v7_run.json on tcnn_v7.json, into outputs/v7_group_a_v2
python -m experimental.v7_compact.run --stage screen --suite experimental/v7_compact/configs/v8_recipe_screen.json \
    --base-config experimental/v7_compact/configs/tcnn_v7_sep_base3_addons.json --output outputs/v8_recipe_sep_base3_addons
python -m experimental.v7_compact.confirm [--plan experimental/v7_compact/configs/v8_angular_confirm.json] [--dry-run]
python -m experimental.v7_compact.benchmark --checkpoint outputs/v8_recipe_sep_base3_addons/angular_loss/selected_model.pt [--mirror-split]
python -m experimental.v7_compact.quantize --checkpoint outputs/v6_group_a_v2/selected_model.pt --output outputs/v7_e3_int8/v6_control_seed44
python -m unittest experimental.v7_compact.test_v7 -v
```

- The confirm stage of `desktop_next.py` would confirm the lowest-error variant. v7's rule picks the smallest
  variant within tolerance instead, so `run.py` refuses `--stage confirm`, and `confirm.py` does the confirming.
- `confirm.py` reuses finished seeds and locks each candidate's best seed in `selected_model.pt`. It only reports:
  the submission choice stays with the team.

## Results (validation, step-20 mean, degrees)

These come from the reports in `outputs/` (gitignored).

**Screen** (seed 42, 26–27 Oct; v6 control reproduced exactly at 4.3511):

| Variant | Error |
| --- | ---: |
| speed_l1 | 4.3378 |
| basis_head | 4.3407 |
| base3_5 | 4.3433 |
| control | 4.3511 |
| separable_24 | 4.3550 |
| width_20 | 4.3573 |
| narrow_full | 4.3777 |
| separable_base3_addons (`v7_extra`) | 4.3855 |
| narrow_base3 | 4.4119 (fails the 4.402 screen bar) |

**Confirm and v8 recipe** (3 seeds; the tolerance is v6 control's 4.342 + 0.04 = 4.382):

| Candidate | Params | Seeds 42 / 43 / 44 | Mean ± sd |
| --- | ---: | --- | ---: |
| v6 control (reference) | 26,012 | 4.3511 / 4.3472 / 4.3285 | 4.3423 ± 0.0121 |
| separable_base3_addons | 6,651 | 4.3855 / 4.3699 / 4.3933 | 4.3829 ± 0.0119 (misses the tolerance by 0.0009) |
| base3_5_addons | 17,811 | 4.3802 / 4.3561 / 4.3655 | 4.3673 ± 0.0121 |
| separable_base3_addons + angular | 6,651 | 4.3386 / 4.3370 / 4.3134 | **4.3297 ± 0.0142** |
| base3_5_addons + angular | 17,811 | 4.3340 / 4.3265 / 4.3135 | 4.3247 ± 0.0104 |
| separable_base3_addons + angular, no gravity | 6,579 | 4.3630 / 4.3686 / 4.3799 | 4.3705 ± 0.0086 |

- **Angular loss**: −0.047° (separable) and −0.046° (base3_5) at seed 42, and −0.048° on average over 6 of 6 seeds.
  p99 gets about 0.09° worse. On the same seed-42 screen, `lr_2e3` and `epochs_80` didn't help.
- **Gravity** is worth 0.041°. It is also the only part of the model that depends on the world frame. The organisers
  will feed our documented frame, so the no-gravity hedge was dropped (28 Oct).
- **Test** (bench, for reporting): sep + angular scores 4.3289, against 4.3584 for v6. It is in
  `jacob/models/tcnn_v7_sep_base3_addons_angular_ga2_stream.onnx`.

**E3 int8** (v6 control seed 44, 657,189 validation windows, float 4.3285):

| Scheme | Change vs float |
| --- | ---: |
| int8 minmax | +0.0213 |
| int8 percentile | +0.0124 |
| w8a16 minmax | +0.0064 |

The quantised graphs add QuantizeLinear and DequantizeLinear, which are off the safe list.
