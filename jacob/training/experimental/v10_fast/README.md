# v10_fast: the same model, trained 4.6 times faster (9 Nov 2026)

**Question.** A code review of `best/` on 9 Nov found that `LeanTCNN.forward` runs all three blocks at all 400 time steps and then reads only the last one. Can training and validation skip the steps nobody reads, without changing the model?

**Answer.** Yes. `FastLeanTCNN` gives the same forecasts to float32 rounding. On the 3080 a training epoch takes 15 s in place of 70 s, and a full validation pass is about 15 times faster. It is not promoted: `best/` is unchanged, and using it for the submitted model is a team call.

## What changes

A block's output at step t depends on its input at that step's taps only: t, t − d, …, t − (k − 1)d. Working back from the last step, k8_combined (kernels 8, dilations 1, 8, 47) needs:

| | LeanTCNN | FastLeanTCNN |
| --- | ---: | ---: |
| Block 3 | 400 steps | 1 |
| Block 2 | 400 | 8 (block 3's taps) |
| Block 1 | 400 | 64 (their taps) |
| Projection | 400 | 400 (it is cheap, and 393 of them are read) |

That is 73 block evaluations a window in place of 1,200. `model.py` gathers the projected features at the 512 tap positions as an `[8, 8, 8]` grid, and each block consumes one axis of it. Nothing else changes:

- The parameters, their order and `specification()` are LeanTCNN's, so a checkpoint from either class loads in the other. Export, the benchmark and `best.model.load_checkpoint` take v10_fast checkpoints as they are.
- Dropout draws LeanTCNN's `[B, C, T]` mask from the same generator state and reads it at the steps each block outputs, so a training step drops the same activations.
- Models without separable blocks, or with a window shorter than the receptive field, run `LeanTCNN.forward` unchanged.

## How it was checked

| Check | Result |
| --- | --- |
| Eval forecasts against LeanTCNN: 4 layouts × 3 maps × gravity and mirror on/off, random weights | Largest difference under 1e-5° (`test_v10_fast.py`) |
| The same comparison with one block's taps reversed | Fails by over 0.01°, so the test can see a wrong index |
| One training step with dropout on, same seed | The loss is identical, and every gradient is within 1.4e-7 of its largest entry |
| A tiny training run through the hooks, against the same run with `best.hooks` | Training loss and validation error agree at every epoch |
| k8_combined on its full validation (657,189 windows), `rescore.py` | LeanTCNN 4.247104990°, fast 4.247104990°. The means differ by 6e-10°, and the largest difference in any forecast component is 1.8e-7. |
| The 800-user model (seed 44) on its full validation (4,755,761 windows, 120 users), `rescore.py` | LeanTCNN 4.403628580°, fast 4.403628580°, and 4.403628580° recorded on 2 Nov. The means differ by 5e-10°, and the largest difference in any forecast component is 1.1e-6 (about 1e-4°, in one window of 4.76M). |
| One training step on the GPU, k8_combined's trained weights, dropout on, same seed | The losses differ by 9e-8 of their value, and every gradient is within 3e-7 of its largest entry. Another mask seed moves the loss by 2%. |
| k8_combined retrained on the 100-user file, seeds 42–44 (`configs/check100_confirm.json`) | 4.2390 / 4.2394 / 4.2513, mean 4.2432 ± 0.0070°, against 4.2501 ± 0.0041° with `best/`. See below. |

**The retraining check.** The three seeds average 0.0069° below k8_combined's, with two seeds lower and one higher. That is within what seeds do: the per-seed differences are −0.0081, −0.0153 and +0.0029 (SD 0.0092), and a Welch test of the two sets of three gives p = 0.23. So three seeds show no difference between the two forwards. They can't rule out one of a few thousandths of a degree either. The runs drift apart gradually, as rounding does: seed 42's validation error is 0.001° from the old run after epoch 1, level at epoch 3, and 0.006–0.008° away from epoch 20 on.

**What is not the same.** The sums run in another order, so a training run is not bit-identical to one made with `best/`. The rounding differences grow over a run the way they do between two machines. So:

- Compare a v10_fast run with a v10_fast control, not with an older run of the same seed.
- The exact reproduction checks in the ledger ("the control reproduced 4.247104990427709") don't carry over to this package.

## Speed

Measured on the 3080 with the trainer's deterministic settings.

| | LeanTCNN | FastLeanTCNN |
| --- | ---: | ---: |
| Training step, batch 256, synthetic windows (first prototype, without the dropout-mask step) | 57.3 ms | 13.4 ms |
| Peak GPU memory in that step | 1,430 MB | 295 MB |
| Full validation forward, 657,189 windows (k8_combined, 15 users) | 47.4 s | 3.2 s |
| Full validation forward, 4,755,761 windows (800-user model, 120 users) | 347.5 s | 23.3 s |
| A whole epoch on the 100-user file: 1,000 steps and the stride-5 validation, mean of 3 seeds | 69.5 s | 15.0 s |
| Training throughput in that run (seed 42, last epoch) | 4,244 windows/s | 18,023 windows/s |
| Peak GPU memory in that run | 4,819 MB | 813 MB |

So a 40-epoch seed on the 100-user file takes 10 minutes of training in place of 46. On the 800-user file a full-pass epoch (3,162 steps) took about 250 s on 1–2 Nov. From the numbers above it should take about 50 s, or about 35 minutes of training a seed in place of 2.8 hours. That 800-user figure is a projection until the screen below has run.

## Run

From `jacob/training/`, one GPU job at a time.

```bash
python -m unittest experimental.v10_fast.test_v10_fast -v
python -m experimental.v10_fast.rescore --checkpoint outputs/<folder>/selected_model.pt [--data <h5>]
python -m experimental.v10_fast.run --stage plan --output outputs/<suite>                 # print the configs
python -m experimental.v10_fast.run --stage screen --data <h5> --output outputs/<suite> [--suite ...] [--variants ...]
python -m experimental.v10_fast.confirm --plan <plan>.json --data <h5> --output outputs/<report folder>
```

`run.py` and `confirm.py` are `best/run.py` and `best/confirm.py` with this package's hooks. Runs hash `experimental/v10_fast/{__init__,hooks,model}.py` on top of best's files, so don't edit those while a run is using them.

## The 800-user screen (`configs/screen_800.json`)

The time saved pays for a screen that was too expensive before. Each variant overrides `best/configs/base.json`, and every one uses the 800-user file.

| Variant | Override | Why |
| --- | --- | --- |
| control | `steps_per_epoch: 3162` | The full pass confirmed on 2 Nov (4.4068 ± 0.0039° on 120 validation users with `best/`) |
| dropout_0 | + `dropout: 0` | Dropout has been 0.1 since v3 and was never ablated. At 800 users the seen-unseen gap is +0.003, so there is no memorisation left for it to fight. |
| passes_2 | `steps_per_epoch: 6324` | Validation was still falling at the end of the full pass (4.4209 at epoch 25, 4.4124 at epoch 38) |
| jitter_1_5 | + `jitter_augmentation_ms: 1.5` | Decision D4. It was measured only at 100 users: +0.009° on the grid, −0.071° at ±2 ms. |
| width_96 | + `channels: 96` | Width gained nothing at 100 users, when the model was data-limited. It costs size and latency: 15,939 parameters against 7,555, and 50% more state. |

**Rules, fixed before the run.**
- The v9 screen rule: a variant goes on to seeds 43 and 44 if its Δ against this suite's control is below 0 with most of the 120 validation users better, paired per user (`jacob/ablations/analyses/paired_tests/v10_fast_800_paired.py`).
- jitter_1_5 keeps the 28 Oct jitter rule instead: Δ on the grid at most +0.01°, and its own ±2 ms cost at most half of the control's (`best.timing_stress --data <merged h5>`).
- width_96 changes size and latency, so passing the rule only makes it a candidate for a team decision.
- Test users are not read.

**Launch** (not started; Jacob's call):

```
powershell -NoProfile -ExecutionPolicy Bypass -File experimental\v10_fast\keep_awake.ps1
```

It runs `run_screen.cmd` and keeps the PC awake. Output goes to `outputs/v10_fast_800/`, the log to `outputs/v10_fast_800_run.log`. Running it again resumes. Expect roughly 4.5 to 5 hours for the five variants: about 35 minutes of training each (70 for passes_2, more for width_96), plus the data audit and full validation of each run and one 10-minute baseline tune. That is an estimate from the 100-user timings.

**Result.** Not run yet.
