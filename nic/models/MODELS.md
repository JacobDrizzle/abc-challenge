# ONNX models

Two **main** models, both exported with only operators from the challenge's safe list. Older exports are in `archived/` (not chip-safe; kept for reference).

| File | Model | Input -> output | Weights | MACs per prediction | Error at step 20 (our 47 test players, all windows) |
|---|---|---|---|---|---|
| `gru_stateful_window.onnx` | Stateful GRU (velocity-only input, 32 memory units) | `context` [1,400,4] + `state_in_0` [1,32] + `state_in_1` [1,4] -> `prediction` [1,20,4] + `state_out_0` [1,32] + `state_out_1` [1,4] | 6,192 | 6,384 | 5.172 deg |
| `gru_stateful_step.onnx` | Same model | as above, but `context` is [1,1,4] (just the newest sample) | 6,192 | 6,384 | 5.172 deg |
| `gadst.onnx` | GADS-T (rotation + velocity input; our own parallel model) | `context` [1,400,4] -> `prediction` [1,20,4] | 10,352 | 606,320 | 5.200 deg |

All trained on 1,229 training players ("big data"), seed 42 (one training run each; the 5-run means are 5.159 for both). Reference on the same windows: stateless GRU 32 5.085 (not chip-safe), hold-last 6.963, damped velocity 5.982. Full table: `../results/eval_table.md`.

## Interface (all files)
- Quaternions are **(w, x, y, z)**, float32, unit length. Opset 17, no float64.
- `prediction` is the next 20 absolute head rotations. It is not normalised (the harness re-normalises); normalise it yourself before measuring angles.
- The challenge scores only the 20th row (0.2 s ahead).

## How to run the stateful GRU (important)
It is a **streaming** model: call it once per new sample, in order, and feed each call's `state_out_0`/`state_out_1` back as the next call's `state_in_0`/`state_in_1`. At the start of each recording set both states to zeros (exactly what the challenge harness does). It only reads the newest sample of `context`, so the window file accepts the harness's 400-sample window and the step file takes just one sample; use whichever matches how the organisers feed it (we still need to ask them).

To score it on a window that ends at sample `e`, stream it over the recording from the start (or from at least 4 s before `e`) and take the prediction at `e`. Starting 4 s before `e` changes the step-20 error by about 0.003 deg (we measured how far back it looks: its influence roughly halves every second; FINDINGS N09).

`run_example.py` shows both models on a recording:
```bash
python nic/models/run_example.py path/to/recording.npy        # [N,4] quaternions (w,x,y,z) at 100 Hz
```

## Chip-safety check (ONNX Runtime + our operator audit)
| File | Operators outside the safe list | ONNX vs PyTorch (max) | Time per prediction, 1 core: Mac M5 Pro / PC Ryzen 7 5800X3D |
|---|---|---|---|
| `gru_stateful_*.onnx` | none | 0.044 deg | 0.012 / 0.026 ms |
| `gadst.onnx` | none | 0.034 deg | 0.188 / 0.234 ms |

The small ONNX-vs-PyTorch difference comes from computing velocity with a small-angle formula in the export (no Atan, which is slow on NPUs).
