# Site data tools

These scripts make the data files the page shows, so that it shows no BOXRR-23 recording. BOXRR-23's data use agreement
(clause 4) does not allow us to pass its recordings on without Berkeley's written consent. The challenge also asks
teams not to share raw or derived data outside the team.

What the page shows now:

| On the page | Made from | Script |
|---|---|---|
| Hero clip, the explorer's first player, journey replays, the story's input figure (`hero/clip.js`, `traces/test50.js`, `figs/challenge.svg`) | QuestSet (Baldoni et al., 2024), CC BY 4.0, real head motion of one player the model never trained on | `make_questset_data.py` |
| The explorer's four faster players (`traces/syn25.js` … `syn97.js`) | synthetic head motion, at the speeds of the 25th, 75th, 90th and 97th percentile BOXRR test windows | `make_synthetic_data.py traces` |
| The Data side's pictures (`data/pipeline/examples.js`: `examples`, `wall`) | a synthetic pile of 766 recordings, run through the team's pipeline (`ray/code`) | `make_synthetic_data.py examples` |
| The Data side's stage 9 window | the first scored call of `traces/test50.js` (QuestSet) | `make_synthetic_data.py examples` |
| Every count, rate and percentage about BOXRR-23 (`config`, `pull`, `final`, `local`), the results bars | the team's real runs (aggregate figures, not recordings) | kept as they were |

`synthetic_motion.py` is the motion model: quick minimum-jerk head turns with a pull back to the front, now-and-then
sideways dodges, slow drift and tracking noise. It is not fitted to anyone's recordings. One setting, how long a head
turn lasts, was set from a statistic of QuestSet's public Beat Saber sessions: how long the head keeps turning the same
way. The comment above `TURN_S` gives the figures.

## Running them

From the repo root, with the training venv (numpy, scipy, onnxruntime, xror):

```
python visualization/tactical/tools/make_questset_data.py PATH/group1_order1_user0_fast_movement.csv   # under a minute
python visualization/tactical/tools/make_synthetic_data.py traces                                      # a few seconds
python visualization/tactical/tools/make_synthetic_data.py examples                                    # about a minute
```

Run them in that order. `examples` copies its stage-9 window from `traces/test50.js`, and `traces` keeps the QuestSet
entries of `traces/index.js` and drops every other non-synthetic entry. Both use fixed seeds, so a rerun gives the same
files apart from the date.

## What `make_synthetic_data.py examples` checks

- Every synthetic recording is packed as a real `.xror` file with the xror library, read back, and run through
  `ray/code/head.py` and `ray/code/cleaning.py prepare_recording`. Its fate is whatever the pipeline returns. If that
  fate is not the planned one, the recording is remade with a new seed and a livelier or stiller head.
- `local` is recomputed from the pile: the fates, sampling rates, empty first rows, rows at 0 s, repeated timestamps,
  backward clocks, holes over 0.1 s, sign flips, and the shortest, median and longest lengths and longest hole. The
  script stops if any of these differs from the real folder's counts.
- Each example is picked by the same rule as before, for example the qualified recording closest to 170 s, or the
  recording with the most repeated timestamps. Asserts check that each rule lands on the recording that was planted
  for it.
- The Data side's own `check()` (in `dataview.js`) must return an empty list in the browser.

The defects in the examples were planted on purpose, at the sizes the real folder had them: a stuck clock of about
3,600 rows, a 0.6 s clock reset in the first 5 s, one sign flip, a 3.2 s hole, a 0.09 s hole.
