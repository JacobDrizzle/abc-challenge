# `bench/`: the team benchmark

One evaluator for everyone's head-pose models, so numbers from Jacob's PC, Nic's Mac and the Linux box can
be compared. The unit under test is the **ONNX file** with the challenge interface. Results are small JSON
files that are committed; `LEADERBOARD.md` is generated from them.

Ported from Jacob's local evaluator (`ABC_local_test`: `baseline_pipeline.py`, `tcnn/damped.py`,
`benchmark_latency.py`, `experimental/team_comparison/compare_nic.py`). It reproduces that evaluator's test
scores of v5/v6 (now `jacob/training/`) to within 1e-4° (see "Regression" below).

## Install (Windows, macOS, Linux; Python 3.12)

```bash
python -m venv .venv-bench
.venv-bench/bin/pip install -r bench/requirements.txt      # Windows: .venv-bench\Scripts\pip
python -m unittest discover -s bench/tests -t .             # ~1 s; the regression test skips without data
```

Only numpy, scipy, h5py, onnx and onnxruntime are needed (no torch). Wheels for every pin exist for Python
3.12 on Windows x64, macOS arm64 and Linux x86-64 (glibc 2.28 or newer).

## Data

A benchmark is a frozen file in `bench/definitions/<name>.json`. It holds the player list, the protocol,
the SHA-256 of the HDF5 it was built on, and a **cohort fingerprint**: a SHA-256 of the protocol, the players
and the exact samples of every scored segment. Scoring refuses an HDF5 whose content gives a different
fingerprint, so equal fingerprints mean the same windows and the same values on every machine. The
fingerprint does not depend on the file itself: the full HDF5 and a benchmark-only extract of it give the
same value.

Agreed (25 Oct): group_a_data_pipeline is the canonical preprocessing, and a benchmark-only HDF5 is shared
on the team Drive. The data is gitignored; BOXRR-23's terms allow sharing it within the team.

```bash
python -m bench data-extract --from qualified_head_100hz.h5 --players players.txt --out bench_v1.h5 --name team-v1
python -m bench data-verify --bench team-v1 --h5 bench_v1.h5
```

Both HDF5 layouts work: group_a_data_pipeline's, with players at the root, and the ray/ export's (group-a-v2),
with each player under `/train`, `/validation` or `/test` (listed in the root attribute `splits`). Segments
are keyed `<user>/<recording>/<segment>` either way, so a layout doesn't change the fingerprint, and
`data-extract` always writes the players at the root.

| Benchmark | Players | Windows | Data | Notes |
| --- | ---: | ---: | --- | --- |
| `group-a-v2-test20` | 20 | 931,051 | `qualified_head_100hz_20261026_131953.h5` (`a56c3a60…`), the ray/ group-a-v2 export on the team Drive | The `/test` users of that file (100 train / 15 validation / 20 test inside it). Train on its `/train`, tune on `/validation`, register with `--trained-on bench/trained_on/group_a_v2_20261026_131953_trainval.txt`. Nic's archived Friday-snapshot models are contaminated (1 player each). |

The leaderboard compares only models trained on the same data: everyone trains on the group-a-v2 `/train`
users and tunes on `/validation`, and `group-a-v2-test20` is the default benchmark. `shared12-v0`, built on
Jacob's `tcnn_v4` data, was retired on 26 Oct, together with its scores and the models trained on that data.
It is in git history.

**Windows**: every stride-1 window of every `challenge_filter_pass` segment with at least 420 samples.
The anchor (last context sample) runs from 399 to N-21; history is the 400 samples up to the anchor,
targets are the next 20.

## Commands

```bash
python -m bench audit model.onnx [--stateless-reference stateless.onnx]   # no data needed
python -m bench register NAME model.onnx --trained-on SPEC [--declared-params N] [--group G] [--owner O]
python -m bench score model.onnx --name NAME --h5 DATA.h5 [--bench group-a-v2-test20] [--machine LABEL]
python -m bench score baseline:damped_velocity --h5 DATA.h5               # also hold_last, constant_velocity
python -m bench score-submission TEAM_headpose/                           # run_inference.py vs its ONNX
python -m bench compare v6_tcnn_ga2 v5_tcnn_ga2                            # paired per-player 95% CI
python -m bench latency v5=v5.onnx v6=v6.onnx gru32=gru32.onnx --machine jacob-amd --threads 1 4
python -m bench leaderboard                                                # rewrites LEADERBOARD.md
```

Set `BENCH_MACHINE` (e.g. `jacob-amd`, `nic-mac`, `nic-linux`) once, instead of passing `--machine` each time.

## What a score contains

- **Accuracy** (`accuracy`): the primary number is the window-weighted mean geodesic angle at step 20
  (200 ms ahead), in float64, after re-normalising the output as the harness does. Also reported: per-horizon
  means (step 10 = 100 ms), median, p95, p99, an equal-weight per-player mean, and per-player window counts
  and error sums.
- **Stateful graphs** (agreed 25 Oct: report both): the headline is **from segment start**: zero states at
  the segment's first sample, one sample per call, with the context left-padded with the first sample until
  400 samples exist. `accuracy_cold_start` has zero states at the first full window. Both score the same
  windows. What the real harness does is still open (nic/FINDINGS.md O8).
- **Efficiency** (`efficiency`), all on generated motion:
  - ONNX initializer elements and Constant-node elements, plus the declared trainable count, kept separately.
    For example, v6 declares 26,012 parameters but its ONNX initializers hold 25,992, because the export
    folded the head bias into a Constant.
  - MACs per call, from run-time shapes (MatMul, Gemm, Conv, GRU, LSTM, RNN, Einsum).
  - Operator audit against the safe and avoid lists in nic/FINDINGS.md R4: float64 anywhere, custom domains,
    sequence types, and recurrences longer than 64 steps.
  - Output shape and finiteness; bitwise repeatability; drift between 1 and 4 threads; stream parity against a
    stateless twin (`--stateless-reference`).
- **Contamination** (`contamination`): the benchmark players found in the model's training or validation data.
  Scoring **refuses** unless `--allow-contaminated` is given; such rows are marked ⚠ on the leaderboard and not
  ranked. Specs (repeatable, combined):
  - `none`
  - a text file with one player id per line
  - `csv:nic/splits/splits_fri.csv[#train,val]`
  - `split-json:frozen_user_split.json`
  - `rule:sha1:0:0.9` (Nic's hash rule, train and val buckets)
- **Provenance**: ONNX SHA-256, data SHA-256, definition SHA-256, cohort fingerprint (plus the old evaluator's
  `legacy_cohort_fingerprint`), git commit and dirty flag, and machine (CPU, OS, cores, Python and ORT versions).

Files:
- `bench/results/<benchmark>/<model>@<machine>.json`: committed.
- `bench/results/latency/<machine>_<utc>.json`: committed; one file per latency session.
- `bench_runs/`: local, gitignored; per-window step-20 errors as float32, with their SHA-256 in the JSON.

`compare` needs only the JSONs. On identical windows, each player's mean difference equals the difference of
the per-player means.

The published copy of `bench/results/` leaves out the per-player sums, because they are per-user scores. Without
them, `compare` and `leaderboard` still give the difference of the two means but mark it `(no CI)`. Re-score both
models with `python -m bench score` to get the interval back.

**Seeds** (agreed: both). Every ONNX file is its own row, because the challenge scores one artefact. Give
models a `--group` (e.g. `gru32_velmag`) and the leaderboard adds a mean ± sd row per group.

**Latency** is local ONNX Runtime CPU time, not ST's target. Each session times all its models on one machine,
alternating over rounds, after warm-up. The leaderboard prints one table per session, never merged.

## Regression

`bench/tests/test_regression.py` reproduces the test evaluation that `jacob/training/` wrote for v5/v6
(`outputs/v{5,6}_group_a_v2/test/results.json`) on `group-a-v2-test20`. It checks:
- 931,051 windows in 72 segments, and the legacy fingerprint `015851ff…` (the trainer's cohort fingerprint)
- every model's mean within 1e-4° (`v5_tcnn_ga2` / `v6_tcnn_ga2` from `jacob/models/`, and the three baselines)
- MACs, parameter counts, registry SHA-256s, and that the models are clean on the benchmark
- per-window errors against the trainer's run, when `BENCH_REF_DIR` is set

```bash
set BENCH_H5=...\abc-challenge\data\h5\qualified_head_100hz_20261026_131953.h5
set BENCH_REF_DIR=...\abc-challenge\jacob\training\outputs
python -m unittest -v bench.tests.test_regression        # ~5 min
```

Note: the trainer scored v5/v6 from their PyTorch checkpoints on stateless windows. Here their stream ONNX
graphs are scored, which agree with those numbers to float32 rounding (max ~3e-4° on a single window). Nic's
group-a-v2 models join the test once they are registered.

## Layout

| File | Role |
| --- | --- |
| `geometry.py` | unit checks, harness re-normalisation, float64 geodesic, generated motion |
| `data.py` | definitions, HDF5 inventory, segment checks, cohort and legacy fingerprints, extract |
| `baselines.py` | hold-last, constant velocity (lag 1), damped velocity (lag 1, tau 100 ms, fixed) |
| `drivers.py` | stateless and stateful ONNX (any tensor names, state paired by name), submission folders |
| `scoring.py` | every window, per-player sums, quantiles |
| `efficiency.py` | parameters, MACs, operator audit, output checks, stream parity |
| `contamination.py` | trained-on specs and overlap |
| `latency.py` | one alternating timing session, machine info |
| `stats.py` | paired per-player bootstrap |
| `results.py` | JSON writes (with retries for Windows file locks), registry, leaderboard |
| `models.json`, `trained_on/` | registered models and the players each was fitted on |
