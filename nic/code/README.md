# nic/code: how Nic's two models were trained and exported

The code behind every file in `../models/group_a_v2/`, copied unchanged from Nic's scratch space (`nic_scratch/pc_train/`)
by `../publish.sh`. Check (29 Oct): running `export_main.py` here on the saved checkpoints gives ONNX files byte-identical
to the published GADS-T s42 and GRU s43 / s45.

## The two best models

One seed of each, chosen on **validation** (the h5's 15 `/validation` users), like Jacob's models. Test numbers are for
reporting only.

| Model | File | Bench name | Validation (°) | Test 200 ms (°) | 5-seed test mean ± sd | MACs per call | Params | Latency, 1 thread median |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stateful GRU | `../models/group_a_v2/s45/gru_stateful_window.onnx` | `nic_ga2_gru_s45` | 4.4110 | 4.4574 | 4.4670 ± 0.0114 | 6,384 | 6,192 | 0.024 ms |
| GADS-T | `../models/group_a_v2/s42/gadst.onnx` | `nic_ga2_gadst_s42` | 4.4016 | 4.4459 | 4.5056 ± 0.0353 | 606,320 | 10,352 | 0.174 ms |

- Test: `group-a-v2-test20` (the h5's 20 `/test` users), headline protocol, from `LEADERBOARD.md`. The best GRU seed on
  test is s43 (4.4533), but validation picks s45, so s45 is the one we report.
- Latency: jacob-amd, 29 Oct, ONNX Runtime 1.31.0 CPU. Relative only: the organisers time on their XR hardware.
- Both use only chip-safe operators (no Clip, Sub, Div or Sqrt). Both predict in the body frame (relative to the newest
  reading), so a fixed world rotation of the input does not change the prediction. Quaternions are (w, x, y, z).
- **GRU**: stateful. Input is angular velocity only (`vel+noq`), 32 hidden units. Graph: `context [1,400,4]` (only the
  newest sample is used), `state_in_0 [1,32]`, `state_in_1 [1,4]` (previous quaternion) -> `prediction [1,20,4]` and the
  two `state_out_k`. States start at zero. Warm-up from zero state (`gru_warmup.py`, result `../results/gru_s45_warmup.json`):
  +1.5° worse on the first call, within 0.1° of the warm stream from call 144, within 0.01° from call 403.
- **GADS-T**: stateless. Input is rotation + velocity (`vel`). The 400 samples are cut into 20 slices; a Deep Set summarises
  each slice, then attention runs across the slices. `context [1,400,4]` -> `prediction [1,20,4]`.

## Files

| File | Role |
| --- | --- |
| `h5_to_npz.py` | Team h5 (Ray's group-a-v2 export) -> one `.npz` per player plus the train / val / test player lists |
| `train.py` | Models (GRU, GADS-T, TCN), features, body-frame maths, training loop, scoring. `--selftest` checks the maths without data |
| `windows.py` | Window indexing and the hash-based player split used by `train.py` |
| `grid.py` | Runs `train.py` for several input sets and seeds in a row |
| `ga2.sh` | The exact training run (PC container): both models, seeds 42 to 46 |
| `stateful.py` | Rewrites the trained GRU as a one-sample-per-call step and checks it matches window mode |
| `onnx_check.py` | ONNX export checks: PyTorch parity, MACs, operator audit, latency |
| `export_main.py` | Exports `gadst.onnx`, `gru_stateful_window.onnx`, `gru_stateful_step.onnx` and `export_check.json` |
| `ga2_export.sh` | The exact export run (Mac): export all seeds, register them on the bench, score them |
| `gru_warmup.py` | How many calls the GRU needs after zero state before it scores like the warm stream |
| `requirements.txt` | Pinned versions |

## Training settings (both models)

`train.py` defaults except the flags below: batch 128, Adam lr 1e-3, gradient clip 1.0, loss on all 20 steps, body
frame, 400-sample context, validation every 100 steps, at most 20,000 steps (`grid.py` sets that), stop after
`--patience` checks without a validation improvement.

| Model | Flags set by `ga2.sh` | Defaults that matter | Steps run |
| --- | --- | --- | --- |
| GRU | `--features vel+noq --patience 30` | 32 hidden units | 20,000 (never early-stopped, so it may be under-trained) |
| GADS-T | `--features vel --model gads` | width 32, 20 slices, 4 heads, max pooling, patience 10 | 10,100 (s42) |

## Reproduce

Paths in the two shell scripts are the ones they ran with: `/data` and `/runs` inside the PC container, and
`nic_scratch/pc_train/` on the Mac. Adjust them to run elsewhere.

```bash
python h5_to_npz.py ../../data/qualified_head_100hz_20261026_131953.h5 DATA_DIR   # 1. h5 -> per-player .npz
sh ga2.sh                                                                          # 2. train (edit D and OUT first)
python export_main.py --gads RUN/gads_vel_s42/model.pt \
    --stateful RUN/gruvp30_vel+noq_s45/model.pt --out OUT_DIR                      # 3. export the best two
python train.py --selftest                                                         # maths checks, no data
```

From the repo root, the warm-up: `.venv-bench/bin/python nic/code/gru_warmup.py nic/models/group_a_v2/s45/gru_stateful_window.onnx`.

Not in git: the data, run outputs and the PyTorch checkpoints (`model.pt`, gitignored). The checkpoints of the best two
are 28 KB (GRU) and 48 KB (GADS-T) on Nic's Mac; ask if you need them.
