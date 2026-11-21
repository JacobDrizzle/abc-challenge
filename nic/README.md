# nic/ : Nic's model-side results (curated copy)

A small, reviewed copy of results from my scratch space, so the team can see and reuse them. Updated by `publish.sh`.

## The two best models (team data, group-a-v2)

| Model | File | Test 200 ms (°) | MACs per call | Params |
| --- | --- | ---: | ---: | ---: |
| Stateful GRU, seed 45 | `models/group_a_v2/s45/gru_stateful_window.onnx` | 4.4574 | 6,384 | 6,192 |
| GADS-T, seed 42 | `models/group_a_v2/s42/gadst.onnx` | 4.4459 | 606,320 | 10,352 |

Seeds chosen on validation; test is `group-a-v2-test20` from `LEADERBOARD.md`. Both chip-safe. The code that trained and
exported them, their settings, interface, latency and the GRU's warm-up: `code/README.md`.

## Folders

| Folder / file | What |
|---|---|
| `FINDINGS.md` | Everything we have learned (good and bad) with evidence and status. Start here. Section 3c is the team-data retrain. |
| `code/` | Training and export code behind `models/group_a_v2/`, with pinned requirements and how to reproduce. |
| `models/group_a_v2/` | Both models retrained on the team h5 only, five seeds each (`s42` to `s46`). |
| `models/` (top level) | The earlier two main models, trained on our own 47-player split, with how to run them (`MODELS.md`, `run_example.py`); older exports in `models/archived/`. |
| `results/` | `eval_table.md` (main models vs the GRU on our 47 test players), `gru_s45_warmup.json` (GRU warm-up from zero state) and plain-language PDFs: input test (Fri, Sat), GADS-T vs GRU, where the error comes from. |
| `splits/` | Train / val / test player ids behind the 47-player numbers (same 47 test players throughout). |

Earlier headline, on our own 47 test players (not comparable with the team bench): **stateful GRU** 5.159 deg and **GADS-T** 5.159 deg. The stateless GRU is slightly more accurate (5.084 deg) but breaks the chip rules, so it is archived as the reference. Hold-last scores 6.97 deg. Details: `FINDINGS.md` section 0 and `results/eval_table.md`.

Not included on purpose: data, run outputs, trained weights (`.pt`).
