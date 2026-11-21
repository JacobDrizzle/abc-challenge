# Jacob's TCNN training

The training, validation, test and ONNX-export code behind Jacob's models in `jacob/models/` (see `MODELS.md`). It
was moved from `ABC_local_test/ABC_local_test` on 26 Oct 2026 and now reads ray/'s group-a-v2 export directly. The v5
model code, training loop and hyper-parameters are unchanged; only the data layer changed (see "What changed").

## Layout

| Folder | What | Docs |
| --- | --- | --- |
| `best/` | The k8_combined architecture and recipe, promoted 30 Oct: validation 4.2501 ± 0.0041°, test 4.2666°, 7,555 params on the 100-user data. Train, confirm, export, int8, distillation and the timing stress test. | `best/README.md` |
| `experimental/` | Next-round experiments, built on `best/` | `experimental/README.md` |
| `archive/` | The v6, v7 and v9 packages behind every run to 29 Oct. Frozen; their code is now `best/`. | `archive/README.md` |
| Top level, `tcnn/`, `configs/` | The frozen v5 trainer, runner, evaluation and export. They are never edited: `best/` and each experiment swap names into them through `hooks.installed()`, so every earlier manifest still verifies. | below |
| `outputs/` | Every run (gitignored, on Jacob's PC). Its record, with the results of every experiment, is `jacob/ablations/`. | `jacob/ablations/README.md` |

**The submitted model** is k8_combined, `final_k8_combined_v10_s42`. It is the `best/` architecture, retrained by
Nic on `moredata_final.h5`: 5,243 train / 786 validation users (`ray/README.md`).
- **Code.** `experimental/v10_speed/` is the faster trainer: same forecasts, last-position forward and dense
  validation.
- **Run.** The suite was `experimental/v10_speed/configs/moredata_final_v10.json`: 40 × 3,162 steps, seeds 42–44,
  on a Colab A100. Seed 42 was picked on validation.
- **Records.** The resolved config, checkpoint, training history and code hashes are in
  `Submission/Motion_Pending_headpose/weights/`. The model card next to it has the full recipe and results.

- Run one GPU job at a time. Two runs in parallel are each about half as fast, because both are bound by kernel
  launch overhead. Force-killing one CUDA run once crashed the other.
- To pause a run, stop it when nothing else is on the GPU. Rerunning the same command resumes from `last.pt`.

## Data

Input is the HDF5 that `ray/main.py` writes (menu option 13), with its manifest next to it:

```
<repo>/data/h5/qualified_head_100hz_20261026_131953.h5     (default: baseline_pipeline.DEFAULT_H5)
<repo>/data/h5/manifest_20261026_131953.json
```

- **Layout**: `/train | /validation | /test / <user> / <recording> / segment_NNN / {time_s, quaternion_wxyz}`.
  The root attribute `splits` names the split groups, and `run_id` names the manifest.
- **Split**: the file's own split groups (100 / 15 / 20 users), which the manifest calls authoritative.
  `--split-file` defaults to `manifest_<run_id>.json` next to `--data`. `tcnn.data.check_corpus` refuses to train
  unless the manifest's `data_sha256` matches the file and its `splits` lists equal the file's groups.
- **Segment keys**: `<user>/<recording>/<segment>`, the same form as the old root layout. Each inventory row also
  carries `path` (`<split>/<user>/...`), which is what gets read.
- **Users**: training uses `/train`, checkpoint and variant selection use `/validation`, and `/test` is scored
  once, after the selection is locked. `/test` is the team benchmark `group-a-v2-test20`, with the same 20
  players and the same 931,051 windows.

## Run

From this folder, with a Python 3.12 environment built from `requirements-training.txt` (Torch 2.7.1+cu118).

**The best model** (`best/README.md` has every flag):

```bash
python -m best.run --stage plan --output outputs/<suite>                  # print the configs, train nothing
python -m best.run --stage screen --suite <suite>.json --output outputs/<suite>
python -m best.confirm --plan <plan>.json --output outputs/<report folder>
python -m best.benchmark --checkpoint outputs/v9_confirm/k8_combined/selected_model.pt   # stream ONNX, parity, latency
python ../ablations/harvest.py                                            # after each suite: update the record
```

- A screen trains seed 42 of each variant. `confirm.py` trains seeds 42–44 of the candidates the team picks, and
  compares them with k8_combined. Both use `/validation` only.
- `--output` is required, and must be a new folder. Interrupted runs resume at the last saved epoch. Changed code,
  data or settings are refused, because the files are hashed into `suite.json` and `experiment.json`.
- Test scores come from the team bench, after export: `python -m bench score ... --bench group-a-v2-test20`,
  registered with `--trained-on bench/trained_on/group_a_v2_20261026_131953_trainval.txt` (see `bench/README.md`).

**The v5 trainer on its own** (how v5 was run on 26 Oct; v6 then ran from `archive/run_v5_v6_windows.cmd`):

```bash
python desktop_next.py --stage all --output outputs/v5_group_a_v2 --suite configs/v5_run.json --base-config configs/tcnn_v5.json
python evaluate_tcnn.py --checkpoint outputs/v5_group_a_v2/selected_model.pt --selection outputs/v5_group_a_v2/selection.json \
    --damped-selection outputs/v5_group_a_v2/damped_baseline/selection.json --split test --device cuda --output outputs/v5_group_a_v2/test
python benchmark_latency.py --checkpoint outputs/v5_group_a_v2/selected_model.pt            # all graphs incl. stream
```

Runs from v6 to v9 used the archived packages. To rerun one, see `archive/README.md`.

Outputs go to `outputs/`, which is gitignored, like `*.pt`.

## Tests

```bash
python -m unittest discover -s tests                         # 65 tests, ~30 s, synthetic data
python -m unittest best.test_best                            # 36 tests, ~25 s; a few also load outputs/ when it exists
python smoke_next.py --device cuda                           # synthetic v5 suite end to end
```

The archived v6, v7 and v9 tests run only from a checkout of tag `v9-ablations` (`archive/README.md`).

`smoke_tcnn.make_fixture` writes a small synthetic export in the group-a-v2 layout, so tests and smoke runs go
through the same data path as the real file.

## What changed from ABC_local_test

- `baseline_pipeline.py`:
  - `load_split` reads the manifest's `splits`.
  - `inventory` finds users inside the split groups (same rule as `bench/data.py:player_paths`) and returns `path`.
  - `manifest_for` and `DEFAULT_H5` are new.
- `tcnn/data.py`: `check_corpus` checks the manifest against the file. The loaders read `row["path"]`.
- `run_next.py`: `--data <h5>` replaces `--data-dir`.
- CLI defaults point at the group-a-v2 file and the v5 configs.
- `jsonio.py` holds `save_json`, which came from `prepare_tcnn_data.py`.
- Left behind:
  - `prepare_tcnn_data.py`, `tcnn/preparation.py`, `tcnn/coverage.py`, `expand_*`, `group_a_data_pipeline/`.
    ray/ prepares the data now.
  - `experimental/team_comparison/`, which `bench/` replaced.
  - The tests of those modules.
- Results from these runs don't compare directly with the old v5/v6 numbers, which came from different users
  (`data/tcnn_v4`, 125 / 11 / 13 users).
