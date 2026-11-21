# Archive: the experiment code behind every run to 29 Oct 2026

> Working log, kept as written. "Pending" and "team call" describe the state at the time. The submitted model and its recipe are in [`model_card.md`](../../../Submission/Motion_Pending_headpose/model_card.md).

> The `v9-ablations` tag cited below is in the team's private history. The public repository starts from a single snapshot, so `git worktree add … v9-ablations` won't work there.

This is frozen, read-only code. On 30 Oct 2026, v9 k8_combined was promoted into `best/`, and the experiment
packages it was built from moved here unchanged. `best/` holds their code in one package, and
`jacob/ablations/analyses/promotion/` shows that it behaves exactly as v9 did (configs, outputs, ONNX files, training
and full validation).

| Folder | Dates | What | Outputs |
| --- | --- | --- | --- |
| `v6_gravity_mirror/` | 26 Oct | Gravity channels and exact left-right mirror equivariance | `v6_group_a_v2` |
| `v7_compact/` | 26–28 Oct | Compression options, the operator-reduced graph, the mirror split, int8 (E3), and the v8 recipe configs (angular loss, no gravity). See its README. | `v7_*`, `v8_*` |
| `v9_lean/` | 28–29 Oct | The lean graph, longer receptive fields, sampling, tilt, jitter, distillation and the timing stress test. See its README. | `v9_*` |
| `run_v5_v6_windows.cmd` | 26 Oct | The one-click v5 then v6 run (train, confirm, test) | `v5_group_a_v2`, `v6_group_a_v2` |
| `configs/v4_run.json` | before 26 Oct | The v4 suite (control_48, damped_prior_48). Nothing references it since v5. | — |

The results of every run, and what each experiment found, are in [`jacob/ablations/`](../../ablations/README.md).

## It doesn't run from here

The imports still say `experimental.v6_gravity_mirror`, `experimental.v7_compact` and `experimental.v9_lean`, as
they did when the runs hashed these files. There is no `archive/__init__.py`, so `python -m unittest discover` skips
this folder.

To rerun or inspect an archived experiment, check out the code it ran with. Every run up to 29 Oct used the code at
tag `v9-ablations` (commit b796dc4). Line endings aside, the files there match the `code_sha256` in each run's
`experiment.json`. That hash is of the LF file, and a Windows checkout has CRLF.

```bash
git worktree add ../abc-v9 v9-ablations
cd ../abc-v9/jacob/training
python -m experimental.v9_lean.run --stage plan --output outputs/_check
```

The worktree has no `data/` or `outputs/` (both are gitignored). Pass `--data` with the H5 in this checkout's
`data/h5/`. Point `--output` at a new folder, so no finished run is touched.

On a CRLF checkout, the three tests that compare the frozen v5 files with their suite manifest fail
(`test_v6`, `test_v7` and `test_v9`, "frozen ... code still matches its suite manifest"): `tcnn/model.py` hashes
differently with CRLF. That was already the case before the archive.
