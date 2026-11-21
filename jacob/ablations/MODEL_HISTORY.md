# Model history: why each version exists

> Working log, kept as written. "Pending" and "team call" describe the state at the time. The submitted model and its recipe are in [`model_card.md`](../../Submission/Motion_Pending_headpose/model_card.md).

This file gives the reasoning behind every model version and experiment on Jacob's TCNN line, from the first test run on 23 Oct to the endpoint refit on 6 Nov. For each one it says why it was run, what changed, how it performed and what was decided. It was written on 6 Nov 2026 from the team's development sessions with AI assistants (Claude Code and Codex) and the plans approved in them. Those sessions are not public.

It complements [`README.md`](README.md), the ablation ledger. The ledger holds the full result tables for entries 1–22. This file holds the motivation and the decision trail, and points to ledger entries rather than repeating their tables. It also covers what the ledger doesn't: the runs before the team data (23–26 Oct) and the work after 31 Oct.

Nic's GRU and GADS-T appear only as comparison points. Their design history is in `nic/FINDINGS.md`.

## How to read this

- **Metric.** Mean geodesic error at step 20 (200 ms), in degrees; lower is better. "Val" means validation and "test" means the test users. Selection was always on validation.
- **Two data eras. Their numbers can't be compared.**
  - **Local (23–25 Oct):** Jacob's own downloads in `ABC_local_test`. The first run had 7 test users, `tcnn_next` and `tcnn_expanded` had 6–7, and v3–v6 had 13. Validation strides also differed (20, then 5).
  - **group-a-v2 (from 26 Oct):** 100 train / 15 validation / 20 test users. The test set is the bench `group-a-v2-test20`. From 1 Nov there is also the merged 800 / 120 / 160 file.
- **Who decided.** "Decision" lines name the source: Jacob's choice, a rule fixed before the run, or an AI assistant's suggestion. The team rule was that humans decide, so AI suggestions are labelled as such.

## Timeline at a glance

| # | Date | Version or experiment | Why it was run | Main result | Outcome |
| ---: | --- | --- | --- | --- | --- |
| 1 | 23 Oct | First TCNN (`tcnn_3080`) | First test of a learned predictor | Test 5.593° against hold-last 6.741° (7 users) | Five follow-up experiments proposed |
| 2 | 23–24 Oct | `tcnn_next` (8 variants) | Test those five | Velocity features −7.9% on val | Velocity input kept |
| 3 | 24 Oct | `tcnn_expanded` | More users (66) | Only 3% better than damped velocity at epoch 18 | Retired; its diagnosis led to v3 |
| 4 | 24 Oct | v3 | Fix slow training, undertraining and the short-horizon weakness | Test 5.10° against damped 6.13° (13/13 users) | Damped prior and EMA adopted |
| 5 | 24 Oct | v4 | More training users (54 → 125) | Test −0.035° against v3 (11/13) | Work on the model, not more downloads |
| 6 | 24–25 Oct | Local v5 | Latency is 30% of the score, so stream | 60× fewer MACs and 3× faster, but +0.160° | Streaming kept; accuracy to win back |
| 7 | 25 Oct | Local v6 | Innovation (20%), and win back accuracy | −0.124° against v5 (13/13), half the parameters | Gravity and mirror adopted |
| 8 | 25–26 Oct | Shared bench and group-a-v2 data | Team numbers were not comparable | One benchmark, one dataset | All later work on group-a-v2 |
| 9 | 26 Oct | v5 and v6 retrained (ledger 1–2) | Re-baseline on team data | v6 val 4.342°, test 4.358° | v6 is the team-data reference |
| 10 | 26–27 Oct | v7 compression (ledger 3–7) | Accuracy plateau, so cut latency and size | sep_base3_addons: 6,651 params, misses tolerance by 0.0009° | Both finalists go to v8 |
| 11 | 27 Oct | v8 recipe and angular loss (ledger 8–10) | The recipe had never been tuned | Angular loss −0.048°, 6/6 seeds | sep + angular is best (4.330°) |
| 12 | 27–28 Oct | Frame robustness and no-gravity hedge (ledger 11–12) | Does gravity break on the test frame? | Gravity worth 0.041° but frame-fragile | Kept after the organisers' answer |
| 13 | 28 Oct | Lean graph (ledger 14) | Half the graph nodes are maths, not network | 130 → 84 nodes, 87.6 → 70.5 µs | Default export |
| 14 | 28–29 Oct | v9 → k8_combined (ledger 15–19, 21) | Literature review; the receptive field was only 243 samples | −0.080° (15/15), 55 µs | Promoted to `best/` |
| 15 | 28–29 Oct | Distillation and jitter augmentation (ledger 16, 17, 20) | Innovation, and the organisers' "some jitter" | Distillation +0.021°; jitter 3 ms +0.030°; 1.5 ms +0.009° | Not adopted |
| 16 | 31 Oct | Grokking question → v10_users (ledger 22) | Can accuracy go further? | Data-limited: 50 → 100 users gave −0.029° | Download more users |
| 17 | 31 Oct–2 Nov | 800-user dataset and k8 retrain | Test the scaling projection | −0.0665° on the old 15 users (3 seeds, 15/15) | Adopted; the final model went on to 5,243 users |
| 18 | 2 Nov | Nic's compression study | Interpret his results | w8a16 near-free; shrinks cost accuracy | Nothing adopted |
| 19 | 4–5 Nov | Weight-edit pilot | A blog post on weight-space programming | −0.002°, CI crosses 0 | Failed its gate |
| 20 | 5–6 Nov | Endpoint refit | Go further with weight edits | −0.006° on 40 reused users (28/40) | Not adopted |
| 21 | 5 Nov | Generalisation report | How well do we generalise to unseen users? | Seen-unseen gap +0.022 at 100 users → +0.003 at 800 | Evidence for training on more users |

---

## 1. Origins: local data (23–25 Oct)

These runs used Jacob's own BOXRR downloads, in a local workspace (`ABC_local_test`) that isn't in the repo. The ideas behind the final model come from here: the damped-velocity prior, EMA, streaming body-frame inputs, gravity and the mirror. The numbers can't be compared with anything after section 2.

### 1.1 First TCNN run (`outputs/tcnn_3080_1`), 23 Oct
- **Why:** this was the first check that a small TCNN could beat simple predictors on unseen users. Jacob asked Codex to review the run and its test results.
- **What it was:** a TCNN with 26,972 parameters on anchor-relative inputs (rotations relative to the newest sample), trained on 27 users (about 3.15 h of Beat Saber) over seeds 42–44.
- **How it performed** (7 test users, 200 ms):

  | Predictor | Test error |
  | --- | ---: |
  | First TCNN | 5.593° |
  | Hold-last | 6.741° |
  | Constant velocity | 8.806° |

  - The TCNN beat hold-last for all 7 users. Seed 44 was selected (val 5.630°).
  - Constant velocity won at 10–50 ms, and the TCNN won only from 60 ms on.
- **Decision:** Codex suggested five experiments, in order, and Jacob had all five implemented:
  1. a tuned damped-velocity baseline
  2. angular-velocity input features
  3. shorter contexts
  4. an angular loss and endpoint weighting
  5. more users and headsets

  Codex also suggested correcting a damped-velocity forecast, citing Martinez et al. (CVPR 2017). That is where the prior-residual design starts.

### 1.2 `tcnn_next`: the five recommendations, 23–24 Oct
- **Why:** to test the five suggestions.
- **What changed:** 8 variants on seed 42, plus a damped-velocity baseline tuned on validation.
- **How it performed** (`outputs/tcnn_next/comparison.md`, val):

  | Variant | Val error |
  | --- | ---: |
  | velocity | **5.196°** |
  | angular_loss | 5.628° |
  | control | 5.642° |
  | context_200 | 5.676° |
  | context_100 | 5.777° |
  | context_50 | 5.886° |

  Endpoint weighting made it worse.
  - The velocity input was the only clear winner, about 7.9% better.
  - **Every shorter context hurt.** This is the first evidence for using the whole 4 s, which v9 came back to (section 5.3).
  - The angular loss gained only 0.014° here. It was treated as noise and dropped from the local v5 suite, then turned out to be the strongest recipe change in v8 (section 4.2).
- **Decision:** the velocity input won, and its seeds 43 and 44 were confirmed. The download stopped at the 15 GB cap with 34 new users. Jacob chose a smaller expansion from the saved data, with the cleaning rules unchanged.

### 1.3 `tcnn_expanded`, 24 Oct (retired)
- **Why:** more users, 66 in total (50 train / 10 val / 6 test).
- **How it performed:** it stopped at epoch 18 of 50, at val 5.468° against damped velocity's 5.616°. That is only about 3% better than a predictor with no parameters.
- **The diagnosis, which led to v3:**
  - **Undertrained.** Every `tcnn_next` run peaked at epochs 42–50 of 50.
  - **Slow.** In deterministic mode PyTorch runs dilated Conv1d on a fallback path: 283 ms a step instead of 7 ms.
  - **Weak at short horizons.** At step 1 it scored 0.157° against damped velocity's 0.108°, because an untrained model starts as hold-last.
  - **Noisy validation.** ±0.1° from epoch to epoch.
- **Decision:** Jacob chose to improve the model for v3 and retire `tcnn_expanded`, with a short screen followed by a confirm (24 Oct).

### 1.4 v3: damped prior, EMA and faster training, 24 Oct
- **Why:** to fix each item in the `tcnn_expanded` diagnosis.
- **Data:** a new local dataset of 78 users (54 / 11 / 13). It was built after Jacob had the cleaning brought in line with the challenge's official filter.
- **What changed:**
  - The dilated convolution was rewritten as shifted taps plus a matmul: 27 s an epoch instead of 120 s, with identical outputs.
  - Data was kept on the GPU, and 8× more windows were drawn per epoch.
  - **Damped-velocity prior:** the network predicts a correction to the damped forecast, so an untrained model equals the baseline.
  - **EMA weights (0.999)** to smooth out the validation noise.
  - Validation at stride 5.
- **How it performed:**
  - Seed-42 screen: damped_prior_48 4.825°, control 4.833°, damped_prior (32 channels) 4.868°.
  - On test, the selected model scored **5.10°** against damped velocity's 6.13° (−16.8%, CI 0.75–1.32°, 13/13 users).
  - The step-1 weakness was gone: 0.134° against 0.136°.
- **Decision:** the prior and EMA stayed in every later version.
  - **Caveat for the paper:** the prior's own accuracy effect was small. In v4 at 48 channels it was 4.7746° with the prior against 4.7772° without (−0.003°, seed 42). It has never been ablated on the final architecture (see [Threads for the paper](#threads-for-the-paper)).

### 1.5 v4: more training users, 24 Oct
- **Why:** to test whether more users would help. Jacob had more data downloaded, and chose to make all the new users training-only, so validation and test stayed comparable with v3.
- **What changed:** training users went from 54 to 125 (4.27M windows). There were three variants: control, control_48 and damped_prior_48.
- **How it performed:**
  - Val 4.7746° against v3's 4.8226°.
  - Test 5.0595° against 5.0950°: −0.035°, CI [−0.061, −0.020], 11/13 users better.
  - The biggest gain was at 10 ms (−9.4%).
- **Decision:** the AI assistant concluded that more data gave little, and suggested working on the model instead (24 Oct). This reading changed in v10 (section 7).
- **Side work the same night:**
  - Jacob asked for a latency script because latency is graded. v4 ran at 0.63 ms with 22.4M MACs.
  - He then asked for a faster TCNN: a strided export brought it to 0.34 ms with 3.07M MACs.
  - Steps 1 and 2 removed Atan and simplified the input preparation.

### 1.6 Local v5: the streaming design, 24–25 Oct
- **Why:** Nic's message (24 Oct) pointed out that latency counts 30% of the score.
  - v4's anchor-relative inputs change on every call, so each call recomputes the whole receptive field (3.1M MACs). Source: `ABC_local_test/NEXT_RUN.md`, "v5 run".
  - The challenge allows a stateful graph.
- **What changed:**
  - **Body-frame velocity inputs:** the per-step rotation in the head frame × 100, plus its magnitude. Old inputs never change, so each layer's history can be cached as state, and each call computes only the newest step.
  - 7 blocks with a 256-sample context.
  - A stateful streaming ONNX export.
  - No trig operators.
  - A world-up gravity input was considered and **rejected**, because the up axis seemed impossible to identify from the data.
- **How it performed:**
  - The selected model (seed 44) scored val 4.898° and test **5.219°**: +0.160° against v4, 0/13 users better.
  - The stream graph ran at 0.092 ms with 51,920 MACs, against v4's 0.276 ms and 3.11M.
  - Screen: width_32 cost only +0.025° at half the parameters. short_128 cost +0.026°.
  - So the gap came from the input design, not from model size.
- **Decision:** the streaming design was kept for its cost. The open problem was winning back the accuracy.

### 1.7 Local v6: gravity and mirror equivariance, 25 Oct
- **Why:** Jacob asked for an innovative feature grounded in the literature.
  - The FAQ says a standard architecture done well "will rank mid-field" on innovation (20%).
  - Every part of v5 was an established technique (the v6 plan).
- **Evidence before building:** a read-only MLP probe on v4 data.
  - Gravity in the head frame gained 1.5%, and mirrored copies of the data gained more.
  - The world frame turned out to be gravity-aligned with +y up (median head tilt 10°), which reversed v5's rejection of gravity.
  - Yaw and roll rates are left-right symmetric.
- **What changed:**
  - **Gravity (1a):** three channels giving world-up in the head frame. Inputs go from 4 to 7.
  - **Exact mirror equivariance (1b):** twin streams with tied [[A, B], [B, A]] weights, adapted from Chirality Nets and EqNIO-style subequivariance. This halves the parameters (26,012 against 51,900) at the same MACs.
  - Jacob's instruction was to build it in `experimental/` and keep v5 stable.
- **How it performed:**

  | Variant (val, seed 42) | Error |
  | --- | ---: |
  | control (gravity + mirror) | 4.7997° |
  | no_mirror | 4.8359° |
  | no_gravity | 4.8459° |
  | plain | 4.9052° |

  - `plain` reproduced v5 exactly.
  - Test 5.0949°: −0.124° against v5, CI [−0.18, −0.07], 13/13 users better. That ties with v4 (+0.035°, CI touching 0).
  - The spread over seeds fell from 0.055° to 0.007°.
  - The cost: the CPU latency rose from 0.092 to 0.120 ms, because the swap adds Slice and Concat nodes.
- **Decision:** both features adopted. Two plan items, the fully safe-op graph and a cold-start check, were carried into later work.

## 2. Team data and the shared benchmark (25–26 Oct)

### 2.1 Why the move
- On 25 Oct Jacob asked to compare v5 and v6 with Nic's models.
- The AI assistant's answer was that the numbers couldn't be compared as they stood: different test users, training data and machines.
- A like-for-like re-score on 12 users that both sides had held out ranked them v6 4.858°, GRU-42 4.943°, GRU-32 4.962°, v5 4.981° and GADS-T 5.078°. It still used Jacob's preprocessing.
- Jacob then asked for a single benchmarking framework for Jacob's and Nic's models.

### 2.2 What was built
- **Shared bench.** Jacob's choices:
  - which users form the benchmark: configurable, decided with Nic
  - which preprocessing: Group A's, with a shared bench HDF5
  - how to stream stateful models: both ways, with from-start as the headline
  - seed means or single models: both
- **group-a-v2 data.** Group A's pipeline was compared with Jacob's and then adapted. Jacob's rule was that the loader reads a single HDF5 file, with the train, validation and test users split inside it as the action plan set out. The result is 100 / 15 / 20 users.
- **Same-data leaderboard.** With the training setups aligned, every model is trained and tested on the same data, for fairer comparisons. The old v5 and v6 graphs and the `shared12-v0` scores were removed.

### 2.3 v5 and v6 retrained on group-a-v2 (ledger 1–2), 26 Oct
- **Why:** to move the trainer into the repo and re-baseline on the team data, with the code unchanged.
- **Result:**
  - v6: val 4.3423° (3 seeds), test 4.358°. v5: test 4.453°.
  - v6 was better on 18/20 users, and both models beat damped velocity for 20/20.
  - The v6 suite's `plain` variant reproduced v5 exactly. Removing gravity cost +0.017° and removing the mirror +0.029°, each on one seed.
- **Decision:** v6 control became the team-data reference.

## 3. Compression: v7 (26–27 Oct; ledger 3–7)

### 3.1 Why
- Accuracy had reached a plateau, so Jacob judged that the biggest remaining gains were in MACs and parameter count, while keeping the current accuracy.
- v6's three seeds spanned 0.02° (the v7 plan).
- **Jacob's decisions** (26 Oct 23:17):
  - go after latency and parameters in one suite
  - try real hardware: ST Edge AI and Qualcomm AI Hub (the uploads are his)
  - accept a smaller model if it is at most **+0.04°** worse (3-seed mean ≤ 4.382°)
  - extras: the fully safe-op retrain and int8
  - out of scope: distillation and the mirror export
- **The measurements that steered the suite** (Jacob had M2 and M1 run before v7):
  - **M1:** on ORT CPU, latency follows the **node count**, about 0.75 µs a node, not MACs. v5 has 133 nodes and v6 174, at the same MACs.
  - **M2:** a polynomial head with t, t² and t³ costs only +0.001°.
  - **E1:** v6 through a reduced graph is 13% faster, with only Sqrt left off the safe list.
  - This is why v7 cut nodes (fewer blocks through base-3 dilations) as well as MACs.

### 3.2 Screen and confirm
- **Screen** (ledger 4), seed 42; the control reproduced v6 exactly:
  - speed_l1: −0.013°
  - basis_head: −0.010°
  - base3_5: −0.008°, at 27% fewer parameters and a 27% faster CPU graph
  - narrow_base3 failed the screen bar
- **Separable + base 3** was timed and screened at Jacob's request: 6,651 params at the same CPU speed as base3_5, but +0.034° on seed 42: the cuts compounded instead of adding up.
- **Confirm** (ledger 5): the AI assistant recommended two finalists, following the plan's rule (the smallest model that passes, plus a backup), and Jacob had both fully trained.

  | Finalist | 3-seed mean | Against the 4.382° limit |
  | --- | ---: | --- |
  | base3_5_addons | 4.3673° | Passes |
  | separable_base3_addons | 4.3829° | Misses by 0.0009° |

  The tolerance had been fixed in advance, so the miss counted as a miss.
- **Side measurements:**
  - **int8 (E3, ledger 6):** written while the screen ran. Percentile int8 costs +0.012° and w8a16 +0.006°, so QAT isn't needed.
  - **Mirror split (ledger 7):** Jacob asked whether the MACs of v7 sep + angular could be cut. The split cuts MACs by 42% at identical accuracy, but is 13% slower on the CPU. It waits on ST timings.
- **Decision:** both finalists went on to the recipe work.

## 4. Recipe: v8 (27–28 Oct; ledger 8–13)

### 4.1 Why
- Jacob asked how the training pipeline could be improved.
- The plan found the recipe had never been tuned on v6 or v7:
  - The loss is a **squared** angle, but the score is the **mean** angle, and errors are skewed (median 3.51°, mean 4.35°).
  - User-balanced sampling draws the smallest user's windows about 74 times each.
  - The learning rate, dropout and weight decay were unchanged since v3.
- Gains here cost nothing in latency, and would buy headroom for separable_base3_addons, which sat at the tolerance limit.
- **A1, read-only diagnostic first (ledger 8):**
  - Small train users scored a 0.750 model/damped ratio, against 0.846 on validation.
  - The model **memorises the users it oversamples**. This led to the sampling options in v9.

### 4.2 Recipe screen and angular-loss confirm (ledger 9–10)
- **Screen:** Jacob chose to run it on both finalists.
  - The angular loss gained −0.047° and −0.046°.
  - 80 epochs was within noise, and LR 2e-3 failed.
- **Confirm:** Jacob had the angular loss run on 3 seeds on both architectures.
  - All 6 of 6 seeds improved: −0.048°, p 0.001.
  - sep + angular scored **4.3297°** with 6,651 params, and was chosen by the plan rule (the smallest model within tolerance).
- **Bench (ledger 13):**
  - Test 4.3289°, −0.030° against v6.
  - −0.138° against Nic's GRU (19/20 users), but the GRU is about 4× faster on the CPU (23 against 89 µs).

### 4.3 Frame robustness and the no-gravity hedge (ledger 11–12)
- **Why:** Jacob asked whether the gravity model would work on the challenge's test input. The test set "comes from people and hardware absent from BOXRR-23", and its world frame wasn't documented.
- **Result:**
  - Gravity models break in a z-up world (+0.9 to +1.7°, worse than damped velocity).
  - A handedness flip costs every learned model +0.12 to +0.18°.
- **Follow-up:**
  - Jacob asked to estimate gravity from the stream (option 3). Analysis showed a 4 s mean head-up is still 8.4° off, so it can't recover posture.
  - Jacob chose to run the no-gravity model first. Removing gravity cost **+0.041°** on 3 seeds, so gravity is real and worth more than v6's single-seed estimate.
- **Decision:**
  - Jacob emailed the organisers (13 questions, drafted with an AI assistant).
  - Their 28 Oct answers: the horizon is 200 ms; they will feed test data in the frame we document; the stream is roughly 100 Hz with "some jitter".
  - So gravity was kept. The frame specification became part of the submission.

## 5. Lean graph and v9 → k8_combined (28–30 Oct; ledger 14–21)

### 5.1 Why
- Jacob asked for a literature review on improving accuracy, within the challenge's safe operator list and without increasing latency (ideally reducing it).
- The plan read the evidence so far:

  | Possible limit | Verdict | Evidence |
  | --- | --- | --- |
  | Optimisation | No | 80 epochs and LR 2e-3 didn't help |
  | Width | No | 3× the parameters gained 0.005° |
  | Memorisation | Partly | A1 |
  | **Short history** | **Yes** | The receptive field was **243 samples**, so the model never saw 39% of the 4 s window |

- **What the literature added:**
  - Mao et al. (ECCV 2020): motion repeats, so long history helps. Nic's ablations agreed.
  - ModernTCN (ICLR 2024): large depthwise kernels are cheap.
  - Idrissi et al. (2022): subsampling as a balancing baseline.
  - The challenge page's hint about where an exponential map belongs.
- **Jacob's decisions:**
  - include distillation, gated on an ensemble check
  - use the CPU node count as the latency proxy until hardware numbers exist

### 5.2 Track A: same weights, cheaper graph (ledger 14)
- **Why:** half of the 130 nodes were feature and output maths, not network.
- **What changed:**
  - a degree-1 rate
  - a Cayley output in place of the exponential map
  - a Conv2d mirror layout
- **Result:** 84 nodes and 70.5 µs against 87.6 µs, with no retraining. It became the default export.

### 5.3 v9 screen, distillation and the D confirm (ledger 15–18)
- **Screen:** a longer receptive field gave the big gain, with fewer nodes:
  - k8 with 3 blocks: −0.048°
  - k5 with 4 blocks: −0.049°
  - Window sampling (−0.029°), native maps (fewer nodes) and 15° tilt augmentation (robustness) were taken forward.
- **Distillation** (C0 / C2): the 9-checkpoint teacher was only 0.02° better than k8. β 0.5 cost +0.021°, so it failed.
- **Jacob's decisions before the confirm** (28 Oct 19:04):
  - add jitter augmentation (it then failed its exact-grid rule, ledger 17)
  - drop the no-gravity twin, since the organisers will use our frame
  - confirm both k8 and k5, because they tied and size is scored
- **D confirm:**

  | Candidate | Val, 3 seeds | Against sep + angular | Users better |
  | --- | ---: | ---: | :---: |
  | **k8_combined** | **4.2501°** | −0.080°, CI [−0.124, −0.047] | 15/15 |
  | k5_combined | 4.2717° | −0.058° | 15/15 |

- **Latency and test (ledger 19):**
  - k8 lean: 60 nodes, 55.2 µs.
  - Test 4.2666°: −0.200° against Nic's GRU, 20/20 users.
- **Promotion (ledger 21):** Jacob chose a single package, archived the old packages, and set up `jacob/ablations/`. Every check against v9 came out exact.

### 5.4 Timing robustness after the organisers' answer (ledger 17, 20)
- **Why:** the organisers said "some jitter, but mostly 100 Hz". Jacob asked whether to add jitter and a non-uniform rate to the data pipeline.
- **Advice and decisions:**
  - The AI assistant advised against changing the shared `ray/` pipeline, and suggested an on-the-fly training option instead.
  - Jacob chose to measure first and then screen, without emailing the organisers again.
- **Result:**
  - Rate offsets of 90–110 Hz are harmless (the model/damped ratio stays 0.837–0.839).
  - ±2 ms of jitter costs k8 +0.157°.
  - 1.5 ms jitter augmentation costs +0.009° on the grid and passes the rule narrowly.
- **Decision:** not adopted. The submitted model was trained on the exact 100 Hz grid, and its model card lists uneven timing as a known failure mode.

## 6. Comparison points: Nic's models

| Date | Comparison | Result |
| --- | --- | --- |
| 25 Oct | 12 shared local users | v6 led the GRUs by 0.09–0.10° |
| 27 Oct | sep + angular on group-a-v2 | −0.138° against the GRU's 5-seed mean, which is 4× faster on CPU (ledger 13) |
| 29 Oct | k8_combined | −0.200° against the GRU (20/20 users) and −0.239° against GADS-T (ledger 19) |
| 2 Nov | Nic's compression study | See below |

On the compression study:
- Graph-only changes are free.
- w8a16 is near-free on every model.
- w8a8 is fine only on k8 (+0.019°).
- Retrained shrinks cost accuracy for little CPU latency. k8 at 48 channels has 4,515 params but tests at 4.310°.
- The study's selection used test users, so it can't be reported as clean.

## 7. Data scaling: v10 (31 Oct–2 Nov)

### 7.1 Grokking ruled out → v10_users (ledger 22)
- **Why:** Jacob asked whether grokking could improve accuracy.
- **Reasoning:** grokking needs a memorisation phase. k8's training and validation errors fall together over about 2.7 passes, so there is none. Grokfast is close to an LR change, and LR 2e-3 had already failed (the v10_users plan).
- Jacob picked the user-count check instead, so the value of more data was measured before days went into downloading.
- **Result** at the same 40k steps:

  | Variant | Δ against k8 |
  | --- | ---: |
  | 25 users | +0.080° |
  | 50 users | +0.029° |
  | 100 users, half their windows | +0.021° |

  - Each doubling gives about 0.57× the previous gain.
  - So k8 is still data-limited at 100 users. This reverses the v4-era reading that "more data gave little".

### 7.2 800-user dataset and k8 retrain
- **Why:** Jacob asked for a dataset with 800 training users and the same validation and test proportions.
- **Dataset:**
  - The merged file is a nested superset: 800 / 120 / 160. All 135 old users keep their splits, so old models stay leak-free on the new validation and test sets.
  - Two tars with malformed recordings crashed the decoder. They were marked as processed and skipped.
- **Retraining**, each step at Jacob's request:

  | Step | Jacob's request | Result on the old 15 validation users |
  | --- | --- | --- |
  | 1. Same recipe, 1,000 steps an epoch (about 0.32 passes) | Retrain the best model on the new dataset | 4.2200° against 4.2471°: −0.027°, 13/15 users |
  | 2. Full pass, 3,162 steps an epoch | Train longer, using all training windows | 4.1889°: −0.058°, 14/15 users |
  | 3. Confirm on seeds 43 and 44 | Confirm on seeds 43 and 44 | **4.1836 ± 0.0047°** against 4.2501 ± 0.0041°: −0.0665°, CI [−0.084, −0.052], 15/15 users, p 0.0001 |

  - On all 120 validation users, the confirmed model scores 4.4068 ± 0.0039°.
  - Parameters and MACs are unchanged at 7,555 and 16,222.
- **Decision:** adopted. The team then went further: the submitted model was trained on 5,243 users (see its model card). Training on more users also takes k8 out of the same-data comparison with Nic's models.
- **Results:** `results/v10_k8_800users/`, `results/v10_k8_800users_full_pass/` and the run logs in `results/logs/`.

## 8. Editing the trained model, and a generalisation check (4–6 Nov)

### 8.1 Weight-edit pilot (Codex, 4–5 Nov)
- **Why:** Jacob asked Codex to read "Sparks of Weight-Space Programming" and see whether an experiment could tune the model's weights.
- **Hypothesis:** the learned correction systematically over- or under-corrects. Jacob asked for a fresh branch and for the original model to be kept.
- **What changed:** gains of 0.90–1.10 on the head weights, applied globally or to one axis: 16 edits.
  - They were selected on 80 validation users and confirmed on 40, both groups drawn from the 800-user model's 120 validation users.
  - There was no gradient training.
- **Result:** the 1.05 global gain gave −0.0023° on seed 44, CI [−0.0064, +0.0018], 22/40 users. All three seeds' intervals cross zero.
- **Decision:** it failed its pre-declared gate, so the original model is kept (`results/v10_weight_edit_20261104/README.md`).

### 8.2 Endpoint refit (Codex, 5–6 Nov)
- **Why:** Jacob asked whether further weight edits could improve accuracy.
- **What changed:**
  - The 97 coefficients that set the 200 ms output were refitted by regularised least squares. The fit used 1.62M windows from the 800 training users (every 20th window), with the network frozen.
  - The fit keeps the mirror symmetry and uses a sign-invariant objective.
- **Result:**
  - Seed 44 went from 4.47235° to 4.46649° (−0.0059°) on the 40 users, 28/40 better, gain CI [0.0017, 0.0102].
  - All three seeds passed the exploratory checks.
  - Size and latency are unchanged.
  - The README itself notes the caveat that **these 40 users were already used in the earlier pilot** (`results/v10_endpoint_refit_20261105/README.md`).
- **Status on 6 Nov:**
  - Codex *recommended* adopting the refit. That is an AI suggestion, not a team decision.
  - It then ran test20: 4.21084° against the original 800-user model's 4.21414°, 15/20 users better, CI includes zero.
  - A 160-user run was still going. Results will be in `results/v10_endpoint_bench_20261106/`, which has no report yet.
  - Selection should stay on validation users that haven't been used before.
- **Outcome:** not adopted. The submitted model has no weight edits. A later endpoint fit on the final model also fell short of its gain threshold (`results/final_v10_s42_weightspace_20261112/README.md`).

### 8.3 Generalisation report (5 Nov)
- **Why:** Jacob asked how to measure how well the model generalises to unseen users. This is evidence for the training-data choice and for the paper's limitations.
- **Jacob's choices:**
  - both k8 models
  - test20 and test160
  - all four metric groups
- **Result** (`analyses/generalisation/generalisation_report.md`):

  | Measure | k8_100 | k8_800 |
  | --- | ---: | ---: |
  | test160 | 4.584° | 4.520° |
  | Seen-unseen gap (model/damped ratio) | +0.022, p 0.0003 | +0.003, CI includes 0 |

  - **test20 is an easy cohort**, about 0.3° below larger populations.
  - Validation predicts test well: Spearman 0.983 across 9 ledger models.

## Threads for the paper

These connect the versions into a story. Each is sourced above.

1. **Prior-residual output.**
   - Suggested on 23 Oct (section 1.1) and adopted in v3. Its value is the starting point (untrained model = damped velocity) and the short horizons.
   - On the anchor models its own effect at 200 ms was ≤ 0.003° (section 1.4).
   - The ledger's "−0.644°" compares k8 with the prior *alone*. It is not an ablation of the prior. If the paper claims the prior as a contribution, a `no_prior` run on k8 is the missing evidence.
2. **Streaming forced the input change, and the cost was won back with symmetry.**
   - Body-frame velocity made caching possible (v5: 60× fewer MACs) but cost +0.160°.
   - Gravity and the mirror (v6) recovered that loss at half the parameters.
3. **The gravity reversal.**
   - Rejected on 24 Oct because the up axis seemed unidentifiable.
   - Adopted on 25 Oct after a probe found a y-up, gravity-aligned frame.
   - Shown to be fragile to the frame on 27 Oct.
   - Kept on 28 Oct after the organisers agreed to use our documented frame.
   - Worth 0.041° on 3 seeds.
4. **Latency follows nodes, not MACs, on our CPU.** This was measured on 26 Oct (M1) and drove v7's base-3 dilations, the lean graph and k8's 3 blocks. The target device is undisclosed, so the paper should state this as a CPU result.
5. **Recipe changes need re-testing after architecture changes.** The angular loss was noise on the anchor model (−0.014°, `tcnn_next`) and the strongest recipe change on v7 (−0.048°, 6/6 seeds).
6. **Sampling and memorisation.**
   - A1 found oversampled users memorised, which led to window sampling (−0.029°).
   - A gap remains at 100 users (+0.022) and closes at 800 (+0.003).
7. **Long history matters.**
   - Shorter contexts hurt from the first suite on: `tcnn_next`, v5 short_128 (+0.029°).
   - The jump came when the receptive field went from 243 to 393 samples (v9, −0.048°).
8. **Data.**
   - The 24 Oct reading ("more data gave little", v4 −0.035°) was overturned. v10 measured diminishing but real returns, and the 800-user model gained −0.0665° on 3 seeds.
9. **Rules fixed before each run kept the choices honest:** the v7 tolerance (separable missed by 0.0009° and was treated as a miss), the jitter rule, the weight-edit gate, and selection on validation only.

## Discrepancies found while writing this

| Item | What differs | Which to use |
| --- | --- | --- |
| Gravity's value | v6 one seed: +0.017°. v8 three seeds: +0.041° | The 3-seed value (the ledger already notes this) |
| "k8 is not overfitting" (v10 plan, 31 Oct) | The 5 Nov report finds a seen-unseen gap of +0.022, p 0.0003, for k8_100 | Both hold. There is no memorisation *phase* in the training curves, but a mild per-user gap exists at 100 users and closes at 800. Say so in the limitations. |
| The prior as a contribution (paper plan) | The evidence row is k8 against the prior alone, not an ablation | Reword, or run `no_prior` (done later, ledger exp. 23) |
| Endpoint refit "use this model" (Codex, 6 Nov) | The confirmation users had been used before, and the test20 CI includes zero | Team decision. Don't choose on test. |
| Local gravity verdicts | Rejected 24 Oct, adopted 25 Oct (the v6 plan) | Story only. Not a data conflict. |
| `tcnn_next` angular loss | It was dropped from local v5 as "no reason to expect a big gain" | It was later the biggest recipe gain. Story only. |

## Sources

This history was compiled from the team's development sessions with AI assistants (Claude Code and Codex, 23 Oct–6 Nov 2026), the plans approved in them, and the result files of the first local runs (`ABC_local_test`). None of these are public. The public results are cited in each section.
