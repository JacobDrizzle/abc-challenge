# `ray/`: the BOXRR-23 data pipeline (group-a-v2)

It downloads BOXRR-23 users from Hugging Face, applies the challenge's data filter, resamples head orientation onto
an exact 100 Hz grid, and exports one HDF5 file, with the train / validation / test users split inside it. Every
model in this repo trains on that file, and `bench/` scores on its test users.

`ray/` is a port of Group B's reference pipeline (`ABC_local_test/group_a_data_pipeline/abc_data/`, a team folder
that isn't in this repo). Every output is stamped `preparation_version = "group-a-v2"`. The stage-by-stage write-up,
with the numbers of the 26 Oct run, was `PAPER_OUTLINE.md` Part 2 (team-internal, not in the public repo).

## Setup (Python 3.12)

```bash
python -m venv ray/.venv
ray/.venv/Scripts/pip install numpy==2.2.6 scipy==1.15.3 h5py==3.13.0 fpzip==1.2.5 pymongo==4.13.2 \
    huggingface_hub==2.3.0 pytest "xror @ git+https://github.com/MetaGuard/xror.git"
```

- xror isn't on PyPI, so it comes from GitHub. pymongo provides `bson`.
- These are the versions in Jacob's `ray/.venv`. There is no requirements file yet.
- The dataset is public (`cschell/boxrr-23`), so no Hugging Face token is needed.
- Download and export both refuse to run without a reserved-user list. By default they read
  `<repo>/data/reserved/frozen_user_split.json`. Set the `ABC_RESERVED_SPLIT` environment variable to read it from
  somewhere else.
  - The team used Group B's `frozen_user_split.json` (not in the repo). Its test users may only go into the test
    split.
  - To reserve nobody, create the file with `{"test_users": []}`.

## Run

From `ray/`:

```bash
python main.py download --train 100 --val 15 --test 20 [--threads 4] [--no-export]
python main.py export --train 100 --val 15 --test 20
python main.py merge h5/manifest_<run A>.json h5/manifest_<run B>.json [...]
python main.py                                  # interactive menu
```

- **`download`** picks random unseen tars, downloads them one at a time, checks every recording, and keeps only the
  passing ones. It stops when every split can be filled, then exports. It also stops after 10 failed tars in a row.
- **`export`** writes the users already in `saved/` to one HDF5 file. It takes only users that no earlier export
  holds.
- **`merge`** (`code/h5_merge.py`) joins exports into `qualified_head_100hz_merged_<time>.h5` plus its manifest.
  Each user stays in the split its own export gave it, so a bigger cohort keeps the earlier train, validation and
  test users where they were.
  - It refuses overlapping users, differing settings, policies or reserved lists, and files changed since their
    manifest.
  - Keep the source files: `h5_exported.json` still points at them. If they are deleted, their users become
    available to a new export again.
- **Malformed recordings crash `xror`.** `XROR.unpack` (`xror.py:389`) dies with an access violation on a few
  replays, and Python can't catch that, so the whole download stops.
  - Rerunning carries on from where it stopped.
  - The two tars found so far are recorded in `cache/processed.json` with a `crash` note, so they are skipped:
    `users/6/607b28cc-….tar` and `users/9/92731c5d-….tar`.
- The **menu** has 14 options:
  - processing tars: 1, 4, 8 and 14
  - re-checking: 2 and 9
  - inspecting and reports: 3, 5, 6 and 7
  - maintenance: 10–12
  - export: 13

Everything goes into `<repo>/data/` (gitignored), wherever `main.py` is run from: `main.py` changes into it, and
the paths in `config.py` are relative.

```
data/
  cache/   tars.json (the repo listing), processed.json (one record per tar), results.sqlite (one row per recording,
           pass or fail), h5_exported.json (users already exported), hf_tmp/ (each tar while it is read)
  saved/   the passing recordings (.xror)
  h5/      qualified_head_100hz_<run_id>.h5 + manifest_<run_id>.json
```

## What it does

| Stage | Code | Rule |
| --- | --- | --- |
| Pre-screen | `head.py` | Metadata only: skip Tilt Brush (no head tracking) and recordings without a HEAD device with rotation axes |
| Cleaning | `cleaning.clean` | Drop non-finite or near-zero quaternions and renormalise; trim the first 5 s; drop duplicate timestamps; keep only recordings at 85–110 Hz (median step) |
| Motion filter | `cleaning.motion_filter` | 5 s windows, `xyz` extrinsic Euler angles. Per axis, take the median over windows of the sweep (P90 − P10) and of the velocity; pass if the largest sweep > 17° **and** the largest velocity > 15°/s |
| Audit | `results_db.py`, `state.py` | Every recording is logged with its reason, rate, sweep and velocity |
| Per-user selection | `h5_export.prepare_user` | Recordings in a seeded order, `sha256("42:<user>/<recording>.xror")`; keep the first 5 that give a usable segment |
| Resampling | `cleaning.resample` | Split at backward clock jumps and gaps > 100 ms; align quaternion signs; SLERP onto `t0 + k / 100`; drop segments < 1,000 samples |
| Split and export | `h5_export.py` | Users sorted, shuffled with seed 42, and filled into train, validation, test. Written under a temporary name, then renamed |

All settings are in `code/config.py`, and all rules are in `code/cleaning.py`. The filter and the export share them.
Bump `PREPARATION_VERSION` whenever a rule changes.

## The HDF5 file

```
/train | /validation | /test / <user_id> / <recording_id> / segment_NNN /
    time_s            float64 [N]     uniform 0.01 s
    quaternion_wxyz   float32 [N, 4]  [HEAD.1, HEAD.i, HEAD.j, HEAD.k], unit length, signs continuous
```

- The source rotation frame is kept as it is: no axis remapping.
- A user sits under exactly one split group, so users can't leak between splits.
- Root attributes: `quaternion_order`, `rate_hz`, `preparation_version`, `run_id`, `source`, `trim_s`, `max_gap_s`,
  `min_segment_samples`, `split_seed`, `splits`.
- Segment attributes: `challenge_filter_pass`, `sweep_deg`, `velocity_deg_s`, `source_rate_hz`, `app`, `headset`.
- `manifest_<run_id>.json` records the settings, the file's SHA-256, the users of each split, and an audit row for
  every recording inspected. The training code refuses a file that doesn't match its manifest.

The HDF5 files themselves aren't distributed: BOXRR-23's data use agreement doesn't allow sharing the data outside
the team. Their manifests record each file's SHA-256 and users.

**The team's file** is run `20261026_131953`: `qualified_head_100hz_20261026_131953.h5`, 68.5 MB, SHA-256 `a56c3a60…`.
- 135 users (100 / 15 / 20), 367 recordings, 405 segments, 15.6 h at 100 Hz.
- Every recording is Beat Saber.
- The source rates are 85–100 Hz (median 90), so the resampling is mostly upsampling.

**The 800-user file** (1 Nov 2026) is `qualified_head_100hz_merged_20261101_140915.h5` (545.5 MB). It merges run
`20261026_131953` with run `20261101_140548`, which holds 945 new users (700 / 105 / 140).
- 1,080 users: 800 / 120 / 160, the same ratio as before.
- 92.6 / 13.6 / 18.2 h, with 32.4M / 4.76M / 6.36M windows of 400 + 20 samples.
- The 135 earlier users keep their splits. Models trained on the earlier file have never seen the merged validation
  or test users.

**The larger files** were built by `export_moredata.py` and `export_final.py`. Both scripts keep every earlier user
in its split, add newly pulled users in the ratio 20:3:4 (seeded shuffle), and drop any recording whose SHA-256 is
already in the file.
- `moredata_1600.h5` (6 Nov): 1,600 / 240 / 320 users. Its manifest is `manifests/manifest_moredata_1600.json`.
- `moredata_final.h5`: 5,243 / 786 / 1,049 users (210.7 M / 31.3 M / 41.6 M samples). It adds 3,643 / 546 / 729
  new users to `moredata_1600.h5`. The submitted model was trained on it. Its manifest is
  `../Submission/Motion_Pending_headpose/weights/manifest_moredata_final.json`.

The tar shuffle is not seeded, so a new download gives different users. Reproduce from the archived file and its
manifest, not by downloading again.

## Tests

```bash
ray/.venv/Scripts/python -m pytest ray/tests -p no:cacheprovider     # 51 tests, a few seconds
```

- The package is called `code`, which shadows the standard library's `code` module. `tests/conftest.py` evicts the
  standard one, and scripts must put `ray/` first on `sys.path`.
- The tests change into a temporary folder, which relies on the relative paths in `config.py`. So don't make them
  absolute.

## Files

| File | Role |
| --- | --- |
| `main.py` | Entry point: the `download`, `export` and `merge` commands, and the menu |
| `code/config.py` | Constants and paths: change settings here |
| `code/cleaning.py` | The rules: head channel, cleaning, motion filter, 100 Hz resample |
| `code/head.py` | Finds the head device and the game in the metadata |
| `code/processing.py` | The filter: runs `cleaning.py`, logs sweep and velocity |
| `code/pipeline.py` | Download tars, check recordings, keep passes, download until the splits can be filled |
| `code/h5_export.py` | The HDF5 export and its manifest |
| `code/h5_merge.py` | Joins exports into one file, each user in its own split |
| `code/hf.py` | Hugging Face: the tar list, and reading replays from a tar |
| `code/results_db.py`, `code/state.py` | `cache/results.sqlite` and `cache/processed.json` |
| `code/storage.py` | Folder layout of `saved/` |
| `code/extracted.py`, `code/reports.py`, `code/viewer.py`, `code/maintenance.py` | Menu options for review, reports, inspection and repair |
| `code/timefmt.py`, `code/cli.py` | Timestamp-unit detection, and prompts |
| `code/legacy.py` | Deprecated: the old `cache/manifest.json` and extraction flow |
