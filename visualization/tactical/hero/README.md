# hero/clip.js: the hero's motion clip

`clip.js` holds 12 s (1,200 consecutive 10 ms samples) of one QuestSet player's recorded head orientation, plus, for
every sample, the model's predicted orientation 200 ms ahead, that prediction's error and the "hold last pose" error.
The hero replays it in a loop (SPEC B1 to B3). It is generated data: do not edit it by hand.

**Why QuestSet (20 Nov 2026).** Until then the clip was 12 s of a BOXRR-23 test player. BOXRR-23's data use agreement
(clause 4) does not allow passing its recordings on without Berkeley's written consent, so a public site cannot carry
them. QuestSet (Baldoni et al., 2024, <https://researchdata.cab.unipd.it/1239/>) is CC BY 4.0: Meta Quest players in four
games, none of them in the model's training data. The credit is on the page (Credits, "Replay data").

- Player: QuestSet `group1_order1_user0`, the Beat Saber session (`group1_order1_user0_fast_movement.csv`, 20 min). It is
  the one Beat Saber session of the 8-player sample that passes the challenge's motion filter, the game BOXRR-23's
  players play. Chosen without looking at the model's error.
- Window: sample indices 71878..73077 of the resampled session, the last 12 s of the viewer trace window
  (`traces/test50.js`, "QuestSet player · Beat Saber"). Its last 300 samples are the 300 the viewer trace scores. The
  trace window is the 300-anchor stretch whose mean head speed (25.5 deg/s) is the median of the session's windows:
  chosen from motion only, never from error.
- Preparation: the frame converted from QuestSet's (right-handed, -z forward) with `q_ours = (w, -x, -y, z)`, then
  `ray/code/cleaning.py` (5 s trim, invalid samples, duplicate timestamps, SLERP onto an exact 100 Hz grid). QuestSet logs
  at about 59 Hz, so the data is upsampled.
- Model: the final model, `Submission/Motion_Pending_headpose/model_fp32.onnx` (final_k8_combined_v10_s42, sha256
  `28221e1e…`), run with the bench driver (`bench.drivers.OnnxModel`, protocol `from-start`) from the session's first
  sample, so it is fully warmed up at every clip sample.
- On the clip the model's error at 200 ms averages 3.17°; on the 300 shared samples, 2.29° against 4.42° for holding the
  last pose. These are one stretch of one player, not a benchmark score.

## Regenerate

From the repo root, with the bench's environment (`bench/requirements.txt`) plus scipy, and the QuestSet session's CSV
(from `Questset.zip`, under `Complete data/group1_order1_user0/`):

```
python visualization/tactical/tools/make_questset_data.py PATH/group1_order1_user0_fast_movement.csv
```

It writes this file, `traces/test50.js` and `figs/challenge.svg` (the story's input-and-answer figure: the scored call
whose 200 ms turn is the trace's median), and puts the trace's entry in `traces/index.js`, QuestSet entries first. The
output is byte-identical between runs on the same day (only `generated` changes with the date). Runtime is under a
minute. The script's docstring has the steps, and how its output format was checked against the BOXRR trace and clip of
13 Nov.

## What the generator asserts before writing

It stops with an `AssertionError` and writes nothing on the first failure.

1. Every model call in the trace returns bitwise the same prediction and states from the instrumented graph (the one
   that exposes the activations) as from the bench driver.
2. The clip's last 300 predictions, streamed from the session's first sample, are bitwise equal to the trace's, which
   start from zero states 792 calls earlier.
3. In the output: no sign flip between neighbouring quaternions in `q` or in `pred`, `pred[i]` in the same hemisphere as
   `q[i]`, every error finite and within 0..180°.

## Fields of `window.K8_HERO_CLIP`

| Field | Meaning |
|---|---|
| `rate_hz` | 100: samples are 10 ms apart. |
| `horizon` | 20: the prediction is for 20 samples (200 ms) ahead. |
| `q[i]` | True head orientation at sample `first_t + i`, quaternion (w, x, y, z), Unity axes, 6 decimals. |
| `pred[i]` | The model's prediction, made at sample `first_t + i`, of the orientation at `first_t + i + 20`. Normalised, same hemisphere as `q[i]`. |
| `err[i]` | Geodesic angle in degrees between `pred[i]` and the true orientation at `first_t + i + 20` (the challenge metric), 3 decimals. |
| `hold[i]` | Same angle for the "hold last pose" baseline: `q[i]` against the true orientation at `first_t + i + 20`. |
| `meta` | Provenance: model and data paths and sha256, user, recording, segment, `first_t`/`last_t`, the selection rule, the clip's mean head speed (deg/s), the seam (angle between last and first sample, speeds at both ends and their difference), the reproduced means over the 300 shared samples, the prediction note, generator and date. |

Note for the page: the true orientation at `first_t + i + 20` is `q[i + 20]` only for `i < 1180`; for the last 20 samples
it lies beyond the clip (it is in the data file, and `err`/`hold` already use it). The shared-sample means in `meta` are
reproduced figures for checking, not something to show on the page (SPEC B2).

## Pointing it at another player

Run the script on another QuestSet `*_movement.csv`. The window rule, the trace id (`test50`) and the clip length are
constants at its top (`WARMUP`, `RECORDED`, `CLIP`). Update the player sentence above, the Data side's stage 9 (rerun
`tools/make_synthetic_data.py examples`, which copies its window from `traces/test50.js`), and the `?r=` version tags that load `hero/clip.js` and the traces (`stage_task.js`, `journey/journey.html`,
`viewer.mjs`, `journey/modelviews.js`, `index.html`, `story.js`), so browsers fetch the new files.
