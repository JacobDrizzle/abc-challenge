> Copy of nic_scratch/FINDINGS.md, refreshed by publish.sh. Paths such as pc_train/outputs/... refer to Nic's local scratch space (not in git).

# Findings log (Nic, ABC Challenge model side)

The single place for what we have learned: good, bad and dead ends. Gitignored with the rest of `nic_scratch/`.

Status: **SOLID** = repeated across seeds / checked, **TENTATIVE** = one run or an interpretation, **DEAD END** = tried, not worth repeating, **OPEN** = still to do or to ask.
All errors are mean geodesic angle in degrees at output step 20 (0.2 s ahead, the only step the challenge scores), smaller is better. "Fri" = 409 training players, "Sat" = 1,229, always scored on the **same 47 unseen test players** (`pc_train/fixed_test_players.txt`) and seeds 42-46.

## 0. Model status (25 Oct): two main models, the rest archived
| Model | Status | Error at step 20 (big data) | Work per prediction | Chip rules | Why |
|---|---|---|---|---|---|
| **Stateful GRU** (velocity-only input, 3 numbers; 32 memory units; patience 30) | **MAIN** | 5.159 (mean of 5 runs); 5.172 for the seed-42 model on all 1.75M test windows | 6,384 MACs; 0.012 ms Mac / 0.026 ms PC (1 core) | no off-list operators | Far cheapest per prediction, which the latency score (30%) rewards. Open: organisers must confirm a memory-carrying model fits the 400-sample rule (memory use measured in N09). |
| **GADS-T** (rotation + velocity input, 7 numbers; chip-safe export) | **MAIN** | 5.159 (mean of 5 runs); 5.200 for the seed-42 model | 606,320 MACs; 0.188 ms Mac / 0.234 ms PC | no off-list operators | Stateless, so no memory question; our own design (innovation score); faster than the stateless GRU. |
| Stateless GRU 32 / 42 (rotation + velocity + sizes) | archived (accuracy reference) | 5.084 | 1.62 M MACs; 0.34 / 0.30 ms | GRU operator (400-step loop) + ReduceL2: off-list, likely CPU fallback | Most accurate, but breaks the chip rules. Kept as the yardstick. |
| GADS-T with speed/size inputs (vel+mag), first exports | archived | 5.210 | 0.63-0.70 M | ReduceL2 (and more in the first export) | Worse on big data and not chip-safe. |
| TCN (dilated, strided) | archived (Jacob's area) | 5.180 (small data) / strided failed | 2.74 M | ReduceL2 | Jacob is working on convolution models. |
| Input variants (acc, speed, mag), speed augmentation, history cut-offs | archived | - | - | - | No gain over velocity input (F02-F05, N05). |

## 1. Challenge rules (checked 2026-10-24 on the site)
| # | Finding | Status |
|---|---|---|
| R1 | Scoring: accuracy 40%, latency on XR platform 30%, innovation 20%, parameter count 10%. How each becomes points is not published. [Home](https://sites.google.com/view/9th-abc-challenge/home) | SOLID |
| R2 | The 500 MB limit is the submission archive only. No parameter cap (count is reported in `model_card.md`). | SOLID |
| R3 | Accuracy is measured on the **20th** output sample only (0.2 s at 100 Hz; the pages call it "100 ms", inconsistent). Output `[1,20,4]`, input `[400,4]`, harness always feeds 400. | SOLID. OPEN: a teammate is asking organisers about the 100 ms wording |
| R4 | ONNX is the deployed format. **Operator rules published (site text Nic pasted 25 Oct):** SAFE = Conv1d/2d (incl. depthwise, dilated), MatMul, Gemm, LayerNorm, BatchNorm, ReLU, ReLU6, HardSwish, GELU, Sigmoid, Tanh, Add, Mul, Concat, Slice, Reshape, Transpose, GlobalAveragePool. AVOID = Loop/If/Scan, NonZero, TopK, custom ops, big einsum, **float64 anywhere**, sequence types, **unrolled recurrences deeper than ~64 steps**. Attention only as plain MatMul-Softmax-MatMul (export unfused). Trig / arccos ops are supported but slow on NPUs. Off-list operators fall back to CPU (about 10x slower). Target class: e.g. Qualcomm NPU. **Stateful option:** recurrent models may expose hidden state as extra inputs `state_in_k` / outputs `state_out_k` (static float32, declared in model_card.md); the harness zero-inits them and feeds one sample at a time; state must use only past samples. A model with no declared state is re-run on each shifted window. | SOLID. OPEN: latency formula and exact hardware still unpublished |
| R5 | Test set is unreleased and from **outside BOXRR-23**. Extra data allowed if disclosed. Deliverables: model, paper (4-6 pp IEEE), model_card.md. | SOLID |
| R6 | **Submission format (Submitting artefacts page, pasted 25 Oct):** `TEAMNAME_headpose.zip` (Drive link, <=500 MB) with `model_fp32.onnx`, optional `model_int8.onnx`, `run_inference.py` (exactly `load_model(model_dir)` and `predict(model, context[400,4]) -> [20,4]`, float32, quaternion order **(w,x,y,z)**, must also run as `python run_inference.py --model_dir . --input sample.npy --output pred.npy`), pinned `requirements.txt`, `model_card.md`, `README.md`, optional `weights/`. No network at inference. Output tensor must be `[1,20,4]` and finite; it is re-normalised before scoring, so **we do not need to normalise output**. Rejected if: non-finite, non-causal, >400 samples of context, wrong shape, fails clean install, no paper / model_card. | SOLID |
| R7 | **MACs per streaming step** must be reported (model_card + paper, with working). Stateless model = cost of a full 400-sample pass. Stateful model = cost of one new sample, "typically far lower; that difference is precisely what the latency criterion rewards". Shorter context is allowed and "doing less work per step is rewarded". model_card also needs: params, context used, state tensors, quantisation, training data, known failure modes. | SOLID |

## 2. Results (what worked, with numbers)
| # | Finding | Evidence | Status |
|---|---|---|---|
| F01 | First GRU (hidden 32, 4 inputs): 5.57 deg vs Hold 6.96; better than Hold for 45/47 players; test minus train gap -0.13 (no memorising). | `pc_train/outputs/` (early runs) | SOLID |
| F02 | **Velocity as extra input is the key gain.** Base 5.586 +-0.067 (Fri). Best sets: `vel+acc+mag` 5.114, `vel+mag` 5.115 (about -8.5%, 5/5 seeds). Speed alone ~0. | `pc_train/outputs/fri_grid/grid_report.html`, `docs/results/friday_grid_results.pdf` | SOLID |
| F03 | Same grid on 3x data: ranking holds (all `vel` sets clear). Base only -0.05 (5.536), best `vel+mag` 5.084. **More data barely helps; it steadies training** (`vel` alone: spread 0.29 -> 0.04, two stuck runs on Fri vanish). | `outputs/sat_grid/`, `outputs/compare/`, `docs/results/saturday_grid_results.pdf` | SOLID |
| F04 | Extras on top of velocity add very little (`vel` 5.148 vs best combo 5.084 on Sat). Plain `vel` (7 columns) is almost as good. | same | SOLID |
| F05 | More history helps slowly, no plateau at the 4 s maximum: last 0.5 s 5.387, 1 s 5.283, 2 s 5.181, 4 s 5.115 (GRU, vel+mag, Fri). | `outputs/fri_history/`, `docs/results/where_error_comes_from.pdf` | SOLID (5 seeds). Pattern/rhythm explanation is TENTATIVE |
| F06 | **Where the error is** (1.75M test windows, seed 42): mid-swing decisions. Head already moving and moves >=10 deg next = 19% of windows, 36% of all error (model err 9.4). Onset from rest = only 2.7% of error. Model is *worse than hold* (2.2 vs 1.4) when the head ends up still. | `outputs/fri_history/breakdown_vel+mag_s42/per_window.npz` | TENTATIVE (one seed) |
| F07 | Error grows ~linearly with horizon: step 1 0.11, step 10 (0.1 s) 2.6, step 20 5.1 deg. So ~2 deg would be reached at ~0.08 s ahead; 1.5-2 deg at 0.2 s is not realistic. | `fri_grid/vel+mag_s42/metrics.json` | SOLID |
| F08 | Baselines on the same 47 players: Hold 6.965, DampedVel 5.977 (its k,d were tuned on an early 11-player pilot, so slightly generous). | grid reports | SOLID |

## 3. GADS-style parallel model ("GADS-T", our own adaptation)
| # | Finding | Evidence | Status |
|---|---|---|---|
| G01 | The GADS paper (arXiv 2504.15751) is face-landmark pose *estimation* from one frame, no time axis, not peer reviewed, no code. GADS-T (20 slices of 20 readings, Deep Set per slice + position codes, 4-head attention, hold-initialised head) is **our design**, not a replication. | `research/modelling/gads_verified.md` | SOLID |
| G02 | Accuracy: GADS-T 5.171 vs GRU32 5.115 vs GRU42 (same size) 5.094 on vel+mag. Not a win (rule: 5/5 seeds, >2x spread, >=0.10 deg). Not run on big data. | `outputs/fri_gads/models_report.md`, `docs/results/gads_vs_gru_results.pdf` | SOLID |
| G03 | Ablations: no position codes 5.287 (worse), mean-pool 5.432 (worse), 10 or 40 slices 5.196 / 5.184 (no better). On plain `base` input 2 of 5 seeds stalled at the hold level (6.97) - early stopping fired too soon; a warm-up might fix it (untested). | same | SOLID / OPEN |
| G04 | Speed: ONNX Runtime, Mac CPU, 1 thread, whole pipeline: GRU32 0.33 ms, GRU42 0.35 ms, **GADS-T 0.096 ms (3.4x faster)**. Plain PyTorch gap is 10x (ONNX fuses the GRU). Files 0.06 / 0.07 / 0.12 MB. ONNX matches PyTorch to ~1e-7. | `pc_train/onnx_speed.py` | SOLID for the ratio; ms are Mac, not the XR target |
| G05 | Literature: no clear evidence that attention/TCN beats GRU/LSTM at horizons <= 0.5 s (Illahi: transformers/TCN no better than hold-last at 100 ms, from earlier note); DLinear shows simple linear models beat transformers at long horizons. | `research/modelling/gads_verified.md` | TENTATIVE (some sources unverified) |
| G06 | **Operator audit of our ONNX graphs vs the safe list (25 Oct).** No float64, no Loop/If/Scan in either. GRU32: uses the `GRU` operator (not on the safe list; it is a 400-step recurrence, over the ~64-step guideline, so likely CPU fallback on the target), plus `Atan` from the velocity maths. GADS-T: attention is plain MatMul/Softmax/MatMul (good) but has off-list `ReduceMax` (the max-pool), `Atan`, `Sin`/`Cos`/`Range`/`Mod`/`ScatterND` (position codes built in-graph, fixable as constants). Our CPU timings therefore probably *understate* GADS-T's advantage on the real target. | `pc_train/onnx_speed.py` + audit | TENTATIVE (audit is certain; the NPU consequence is inference) |
| G07 | **Final GADS-T (main model)**: trained on rotation + velocity input (7 numbers) on big data: 5.159 +-0.046 (5 runs), seed-42 model 5.200 on all 1.75M test windows (+0.115 [+0.084, +0.142] vs GRU 32, beats it for 7/47 players). 10,352 weights, 606,320 MACs, 0.188 ms Mac / 0.234 ms PC (1 core). Chip-safe export: max-pool as a ReLU tree, attention as plain MatMul/Softmax/MatMul for the newest slice only, position codes as constants, small-angle velocity, no output normalisation: **no off-list operators**. With the extra speed/size inputs it got worse (5.210). | `outputs/sat_round4/gads_vel_*`, `outputs/eval_table/`, `nic/models/` | SOLID |

## 3b. Round 3 (25 Oct): NPU fit, TCN, stateful GRU, speed changes
| # | Finding | Evidence | Status |
|---|---|---|---|
| N01 | Export clean-up: quaternion product as MatMul, small-angle velocity (no Atan), Concat/Slice instead of Pad/Gather, no output normalisation, onnxsim. ONNX matches PyTorch to 0.04 deg. Remaining off-list: ReduceL2 (speed/mag inputs), the GRU operator, GADS-T ReduceMax + attention Gather. | `pc_train/onnx_check.py`, `outputs/onnx_round3/` | SOLID |
| N02 | **Stateful GRU (velocity-only input, `vel+noq`)**: 6,384 MACs per step (vs 1.62M stateless GRU), 0.011 ms, no off-list operators. Good seeds 5.18-5.23 deg; 2 of 5 seeds stopped early at 5.67 (same stall as Friday `vel`). Memory run from the recording start = fresh 4 s window (within 0.003 deg). | `outputs/fri_round3/gruv_*`, `stateful_stream.json`, `stateful.py` | SOLID (accuracy needs big-data rerun) |
| N03 | Dilated TCN: 5.180 +-0.028, 12,304 weights, all safe ops except ReduceL2, but 2.74M MACs (more than GRU). | `outputs/fri_round3/tcnd_*` | SOLID |
| N04 | Strided TCN as built did not train (all seeds stuck at hold 6.97). | `outputs/fri_round3/tcns_*` | DEAD END (as built) |
| N05 | Speed-change test 0.67x-1.5x: GRU beats hold at every speed; error grows with speed for all methods. Time-warp augmentation: no effect (within 0.02 deg); normal-speed 5.133 vs 5.115. | `outputs/fri_round3/warp_eval.json` | SOLID |
| N06 | Control: fresh GRU vel+mag with current code = 5.115 +-0.031, identical to Friday. | `outputs/fri_round3/grunew_*` | SOLID |
| N07 | **Round 4, big data, chip-compliant models.** Stateful GRU (vel+noq, patience 30): 5.159 +-0.009, all 5 runs trained to 20,000 steps (default patience: 5.185 +-0.018, no stalls on big data). GADS-T on velocity input: 5.159 +-0.046, 10,352 weights, 0.61M MACs; GADS-T vel+mag got worse (5.210). Reference GRU vel+mag (big data) 5.084. Both candidates export with **no off-list operators**. Streaming from recording start = fresh 4 s window (within 0.006 deg). | `outputs/sat_round4/`, `outputs/onnx_round4/`, `docs/results/round4_compliant_models.pdf` | SOLID |
| N08 | **Jacob-style table** (47 test players, all 1.75M windows, step 20, seed-42 models, big data): GRU 32 5.085; stateful GRU 5.172 (+0.087 [+0.069, +0.105], beats GRU for 5/47 players, 6.4K MACs, 0.012 ms Mac / 0.028 ms PC); GADS-T 5.200 (+0.115 [+0.084, +0.142], 7/47, 0.61M MACs, 0.189 / 0.237 ms). Not comparable row-by-row with Jacob's table (different test players: his 12 vs our 47). | `pc_train/outputs/eval_table/table.md`, `eval_table.py` | SOLID |
| N09 | **How far back the stateful GRU looks** (cut-off test, 35K test windows): wiping memory T s before a prediction changes it on average by 2.22 deg (0.1 s), 0.93 (1 s), 0.43 (2 s), 0.21 (3 s), 0.11 deg (4 s; 99th pct 0.50): influence roughly halves every second. Error: 5.566 (0.1 s) ... 5.154 (4 s) vs 5.151 full history: history older than 4 s still nudges predictions slightly but does not improve them (0.003 deg). Per-neuron update-gate spans are short (median 0.015 s, max 0.42 s): the network's memory is longer than any single unit's because units pass information to each other. | `pc_train/memory_probe.py`, `outputs/eval_table/memory_probe.json` | SOLID |
| N10 | **What the stateful GRU's '32' means and how it uses the 4 s** (seed 42 = the exported model; 5,000 test windows, small-data snapshot; hold 6.922). '32' is the memory size (32 numbers), not 32 samples (Jacob read it as 32 samples / 0.32 s): it reads every sample, one per step, and its memory is a lossy summary of all of them. Newest k samples only (memory from zero) - error / mean change vs all 400: k=32 5.391 / 1.76 deg, 100 5.239 / 0.92, 200 5.120 / 0.45, 300 5.093 / 0.24, 400 5.089. History swap (all 400 fed, but everything older than the newest k taken from another window): k=32 5.785 / 2.62, 100 5.369 / 1.40, 200 5.162 / 0.69, 300 5.099 / 0.37, so the CONTENT of old samples matters, not just warm-up. Gradient sensitivity: 50% of influence in the newest 5 samples, 61% in 32, 90% in 193 (~1.9 s), 99% in 361; a sample 4 s back has ~0.1% of the newest one's influence. Summary: a summary of the whole 4 s, weighted toward the recent past; older than 4 s does not help (N09). | `pc_train/memory_test.py` (run: `abc_torch/bin/python memory_test.py <model.pt>`) | SOLID (seed 42; two methods agree with N09) |

## 3c. Team benchmark (26 Oct): both main models retrained on the team data only
Data: `qualified_head_100hz_20261026_131953.h5` (group-a-v2; 100 train / 15 val / 20 test users, 11.0 h of training data vs 154 h before). Same settings as round 4, seeds 42-46. Scored with Jacob's `python -m bench` on `group-a-v2-test20` (931,051 windows, Mac). **These numbers are not comparable with the 47-player numbers above**: this test set is easier (hold-last 6.04 vs 6.96).
| # | Finding | Evidence | Status |
|---|---|---|---|
| B01 | Our training script and Jacob's bench agree: hold-last 6.036 vs 6.037, damped velocity 5.114 vs 5.114 on the same 20 players. | `outputs/ga2_round/*/metrics.json`, `bench/results/group-a-v2-test20/` | SOLID |
| B02 | Stateful GRU 4.467 +-0.012 (5 seeds, best 4.453), no stalled seed; every seed used all 20,000 steps without early stopping, so it may be under-trained. GADS-T 4.506 +-0.038 (best 4.446, seed 42). Both chip-safe (no audit issues). | `LEADERBOARD.md` | SOLID |
| B03 | Jacob's v6 TCN 4.378 beats our best (GADS-T s42) by 0.068 [0.024, 0.122]; v5 4.457 ties ours. v5/v6 saw 1 of the 20 test players in training, but it is only 0.6% of windows: without it v6 is still 4.383 vs 4.451. v6 was picked from 3 seeds on validation and trained on 136 users. | `LEADERBOARD.md`, per-player sums | SOLID |
| B04 | Latency (Mac, 1 thread, bench session): stateful GRU 0.012 ms, v5 0.052, v6 0.074, GADS-T 0.188. MACs per call: GRU 6,384, v5 51,920, v6 52,144, GADS-T 606,320. | `bench/results/latency/nic-mac_20261026T173931.json` | SOLID for the ratios (Mac, not the XR target) |
| B05 | The bench counts the GRU's ONNX parameters as 2,663 because the export stores most weights as Constant nodes (+3,621); the declared count 6,192 is the real one. | leaderboard | SOLID |

## 4. Dead ends and mistakes (so we do not repeat them)
| # | What | Why |
|---|---|---|
| D01 | Hoping more data lowers the floor | 3x data: base -0.05 deg, best set -0.03 deg (F03) |
| D02 | Hoping a different architecture breaks 5 deg | GRU, size-matched GRU and GADS-T all land at 5.1-5.2 (G02) |
| D03 | Speed (alone) as input | +0.04 deg only (F02) |
| D04 | Time-warp augmentation | no gain (N05) |
| D05 | Strided TCN (as built) | did not train (N04) |
| M01 | I guessed the error was mostly movement onsets from rest | Breakdown: onset is 2.7% of error (F06) |
| M02 | I said size was a reason against a transformer | Parameters are only 10% of the score; latency is 30% (R1) |
| M03 | ONNX export failed: ONNX Runtime has no double-precision `Atan` | Feature maths now runs in float32 only for export (`train.FEAT_DTYPE`) |
| M04 | Reading the challenge pages: horizon "100 ms" | 20 readings at 100 Hz = 0.2 s; scored step is the 20th (R3) |

## 5. Open questions / next steps (the team decides)
- O1 Ask organisers: latency formula and exact target hardware (operator rules are now known, see R4), and the "100 ms" wording (a teammate is doing the horizon question).
- O8 **Ask organisers about stateful models:** (a) what shape is `context` when fed one sample at a time; (b) a GRU's hidden state carries the whole session, not just the last 400 samples - is that allowed given "uses more than 400 samples of context" is a rejection reason, or must state be limited to a 4 s window?
- O9 Output normalisation can be dropped from the exported graph (harness re-normalises). Hand estimate of full-pass MACs (to verify with a profiler): GRU32 about 1.6M, GADS-T about 0.7M; a stateful GRU step would be about 0.01M.
- O7 Make the model NPU-friendly: remove `Atan` (small-angle velocity), position codes as constants, decide on `ReduceMax`; consider a stateful one-sample-at-a-time GRU (needs a different, increment-based input, so a retrain) and a Conv1d/dilated TCN (first-class safe op, untested).
- O2 GADS-T worth finishing? Tidy its ONNX graph (precompute position codes as constants), per-operator profile, int8, big-data run. See `docs/results/gads_vs_gru_results.pdf`.
- O3 Cheap check from research: a DLinear-style linear baseline. If it matches the GRU, neither architecture adds much.
- O4 Does GADS-T/GRU use rhythm? Longer-history test on GADS-T; try the history result on big data.
- O5 Generalisation to the hidden, outside-BOXRR-23 test set is untested (we only have within-BOXRR-23 unseen players).
- O6 Jacob's WhatsApp comparison: compare on a shared split (offered, not done).
- O11 The retrained GRU never early-stopped (all 5 seeds hit 20,000 steps): try more steps on the team data. Team to decide whether to retrain on more users than the h5's 100 (F03 said more data barely helps).
- O10 (DONE 26 Oct, see B01-B05) **Our main models can't be ranked on the team bench `group-a-v2-test20` as they are:** 19 of its 20 test users are in our sha1 train/val buckets (18 train, 1 val), so they were seen in Saturday training. To compare fairly, retrain on the file's `/train` (100 users, about 11 h, vs our 154 h) and tune on `/validation`. The bench's pinned scipy 1.15.3 fails to import on macOS 27 (dyld error); scipy 1.16.3 works.

## 6. Where things are
| What | Where |
|---|---|
| Code | `pc_train/` (`train.py` model + flags, `grid.py`, `compare_grid.py`, `onnx_speed.py`, `latency.py`, `*.sh` run scripts) |
| Result folders (Mac) | `pc_train/outputs/{fri_grid,sat_grid,fri_gads,fri_history,compare}/` (same on PC under `/srv/abc/runs/`) |
| Data snapshots | `data/snapshot_fri/` (504 players, Mac); PC `/srv/data/abc/snapshot_{fri,sat}/` (1,417 players, 154 h) |
| Team data (26 Oct) | `data/qualified_head_100hz_20261026_131953.h5` at the repo root (gitignored): Jacob/Ray's group-a-v2 export, 100 train / 15 val / 20 test users split inside the file; the `group-a-v2-test20` bench scores its `/test` users. Verified on the Mac (sha `a56c3a60…`, fingerprint OK, 931,051 windows) |
| Plain-language docs | `docs/results/*.pdf` (list in `docs/README.md`), sources `docs/src/`, rebuild with `docs/src/build.sh NAME` |
| Research notes | `research/*/ *_verified.md` |
| Cross-session memory | alcedo, project `abc-challenge` (short facts; this file has the detail) |

## How to add
Add a row in the right table with the next number, the numbers, the evidence path and a status; change a status when it is confirmed or disproved; never delete a dead end. Keep one line per finding. Also file the one-sentence version with alcedo `save_fact` (project `abc-challenge`).
