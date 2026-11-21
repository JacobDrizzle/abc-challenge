# v10_loo: leave-one-out ablation of k8_combined

Started 6 Nov 2026 at Jacob's request. It feeds the Ablations section of the paper.

**Question.** What does each component of the final model contribute when it is removed from that model, on the
same seeds? The ledger's component-effects table (`jacob/ablations/README.md`) measured each component on a different
base, mostly on one seed.

**No new code.** It is configs only and runs on `best/` unchanged:
- `configs/screen.json`: a suite for `best.run`.
- `configs/confirm.json`: a plan for `best.confirm`.

The data is the default 100-user file (`qualified_head_100hz_20261026_131953.h5`), the same as k8_combined's.

| Variant | Override of `best/configs/base.json` | Params | Tests |
| --- | --- | ---: | --- |
| control | `{}` (k8_combined itself) | 7,555 | Must reproduce seed 42 exactly: 4.247104990427709 |
| no_prior | `{"damped_prior_tau_ms": null}` | 7,555 | The residual corrects hold last instead of the damped prior. **New**: not in the ledger's suggested suite. |
| no_gravity | `{"gravity_features": false}` | 7,459 | Gravity in the head frame |
| no_mirror | `{"mirror_equivariant": false}` | 15,113 | Mirror equivariance. Not size-matched. |
| quaternion_loss | `{"loss_kind": "quaternion"}` | 7,555 | Angular loss |
| user_sampling | `{"sampling": "user"}` | 7,555 | Window sampling |
| rf_243 | `{"num_blocks": 5, "kernel_sizes": [3, 3, 3, 3, 3], "dilations": [1, 3, 9, 27, 81]}` | 11,427 | The kernel-8 layout (receptive field 393). Not size-matched. |
| exact_maps | `{"maps": "exact"}` | 7,555 | Native maps |
| no_tilt | `{"tilt_augmentation_deg": 0}` | 7,555 | Tilt augmentation. Its main purpose is frame robustness, which this suite doesn't measure. |

Pre-flight checks, 6 Nov:
- `--stage plan`: each variant differs from the control in its own keys only.
- `best.confirm --dry-run`: the confirm configs are identical to the screen configs, so seed 42 is reused.
- Every variant builds, runs a forward pass, and gives finite unit quaternions.

## Run

From `jacob/training/`:

```
powershell -NoProfile -ExecutionPolicy Bypass -File experimental\v10_loo\keep_awake.ps1
```

This runs `run_loo.cmd` and stops Windows from sleeping until it finishes. The machine sleeps after 10 min of no
input, which would pause training. No power settings are changed. `run_loo.cmd` runs two stages, one after the other:
1. **Screen**, seed 42, for the control and the 8 variants: 9 runs, about 7.5 h.
   - Output: `outputs/v10_loo/`. Log: `outputs/v10_loo_screen_run.log`.
2. **Confirm**, seeds 43 and 44 for the 8 variants, reusing seed 42 from the screen: 16 runs, about 13 h.
   - Reference: `outputs/v9_confirm/k8_combined`. Report: `outputs/v10_loo_confirm/confirmation.md`.
   - Log: `outputs/v10_loo_confirm_run.log`.

The run times assume about 0.8 h a run on the 3080. Finished seeds are reused, so re-running the command above after an
interruption resumes it.

The 6 Nov launch:
- `run_loo.cmd` was started at 22:37 as a detached process (PID 6964).
- `keep_awake.ps1 -WatchPid 6964` holds the no-sleep request for it (PID 22016).
- **The control finished and reproduced k8_combined exactly**: all 40 epochs, and validation 4.247104990427709.
- At 23:19, something outside this run deleted most of the `ABC_local_test/ABC_local_test/.venv` it ran in:
  - Gone: `pyvenv.cfg` and every file in `Scripts/` except the locked `python.exe`.
  - `Lib/` was untouched.
  - Then no_prior failed at its start-up `pip freeze`, at 23:27.
- Jacob chose a fresh venv: `<venv>`, built from the root `requirements.txt` (torch
  2.7.1+cu118, Python 3.12.10). `run_loo.cmd` now uses it. The old venv was left as it was.
- The new venv reproduces the control bit for bit: epochs 1 and 2 match k8_combined. The check run is
  `outputs/v10_loo_envcheck/`, stopped after epoch 2.
- Resumed at 23:44 through `keep_awake.ps1`, with no arguments.
  - The control was reused.
  - no_prior restarted from scratch. Its partial folder was kept as `seed_42_interrupted_*`.
- Stopped at 02:44 on 7 Nov, during quaternion_loss epoch 32:
  - First the run aborted with `forrtl: error (200): program aborting due to window-CLOSE event`.
  - 17 s later a sleep command put the PC to sleep. Event log: "Sleep Reason: Application API" (Start-menu Sleep or an app).
  - `keep_awake.ps1` blocks only idle sleep. It can't stop a manual sleep, a sign-out or a shutdown.
  - By then no_prior, no_gravity and no_mirror had finished.
- Resumed at 10:23 on 7 Nov the same way (PID 41820). quaternion_loss continued from its epoch-31 `last.pt`: the
  resume restores the optimizer and random-number state, which `tests/test_tcnn.py` checks matches an uninterrupted run.
- The screen finished at 13:18 on 7 Nov.
- Crashed at 21:20 on 7 Nov, in confirm rf_243 seed 43, epoch 3:
  - `RuntimeError: CUDA error: unknown error`, then the PC rebooted uncleanly.
  - Event log: Kernel-Power 41, no bugcheck, no power-button press. The cause is unknown.
  - 10 of the 16 confirm seeds had finished. All their checkpoints were checked afterwards and load and are finite.
- Resumed at about 11:10 on 8 Nov:
  - The partial seed was kept as `rf_243/seed_43_crashed_20261107T212050` and retrained fresh.
  - Its new epochs 1 and 2 are bit-identical to the crashed copy, so the GPU still computes the same.
  - GPU temperature and power were logged every 30 s to `outputs/v10_loo_gpu_log.csv`: at most 80 °C and 312 W
    (limit 320 W).
- **Confirm finished at 15:26 on 8 Nov.** Report: `outputs/v10_loo_confirm/confirmation.md`.

Don't edit `best/*.py`, `run_desktop.py`, `run_next.py` or `train_tcnn.py` while it runs:
their hashes are recorded. Don't time latency on this machine while it trains.

## Afterwards

1. Run `python jacob/ablations/harvest.py`. **Done 8 Nov.**
2. Run the paired per-user tests against k8_combined's seeds 42–44. **Not done:** it needs a new script, adapted from
   `jacob/ablations/analyses/paired_tests/d_confirm_paired.py`.
3. Add a ledger entry: Δ for each removed component, users better, CI. **Done 8 Nov as ledger exp. 23,** without users
   or CIs until step 2 is done.
4. Replace the paper's Table "Effect of Each Component" with one row per removed component. **Done 8 Nov** in
   `ABC_Challenge_paper/ablations.tex`. The old table is kept in `ablations_ledger.tex`.
