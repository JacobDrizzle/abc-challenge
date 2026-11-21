# Runs on Nic's PC server

Every score here was produced on Nic's PC server, so the paper can cite one machine for its results.
Latency is not in this file: the paper's phone numbers come from Qualcomm AI Hub (`bench/results/aihub/`).

## Machine

- AMD Ryzen 7 5800X3D 8-Core Processor, 16 logical cores, Linux-6.8.0-147-generic-x86_64-with-glibc2.39
- Run with `prun` (container image `ml-base:cu130`, CPU only for ONNX Runtime): Python 3.12.3, ONNX Runtime 1.31.1, numpy 2.5.4, h5py 3.13.0
- These are newer than the bench's pinned versions (ONNX Runtime 1.31.0, numpy 2.2.6). The scores match the earlier runs anyway (see below).
- Run 13 Nov 2026 (UTC), from a copy of `bench/` outside the repo (`~/Dev/experiments/abc-bench-pc`), so no git commit is recorded in the JSONs.

## Benchmark `group-a-v2-test20`

20 test users, 931,051 windows, data `qualified_head_100hz_20261026_131953.h5` (sha256 `a56c3a607dc53f91…`).
Error is the window-weighted mean geodesic angle at step 20 (200 ms). Learned models are streamed from segment start (`--protocol from-start`).
Every learned model passed the contamination check: none of the 20 users is in its training list.

| Paper row (Table 3) | Bench name | Params | PC error 200 ms (°) | PC error 100 ms (°) | Earlier run (°) | Difference | ONNX sha256 |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| v5 | `v5_tcnn_ga2` | 51,900 | 4.4534 | 2.2595 | 4.4534 (jacob-amd) | 0e+00 | `c746ce1611bb…` |
| v6 | `v6_tcnn_ga2` | 26,012 | 4.3584 | 2.2305 | 4.3584 (jacob-amd) | 0e+00 | `cf065dd37603…` |
| v7 | `v7_sep_base3_addons_ga2` | 6,651 | 4.4186 | 2.2504 | 4.4186 (jacob-amd) | 0e+00 | `f475f2f9b271…` |
| v8 | `v7_sep_base3_addons_angular_ga2` | 6,651 | 4.3289 | 2.2097 | 4.3289 (jacob-amd) | 0e+00 | `d9f48aa1f25e…` |
| k8-combined | `v9_k8_combined_ga2` | 7,555 | 4.2666 | 2.1887 | 4.2666 (jacob-amd) | 4e-10 | `fa69c3612716…` |
| k8-combined, final | `final_k8_combined_v10_s42` | 7,555 | 4.1836 | 2.1469 | 4.1836 (nic-mac) | 2e-09 | `28221e1e0b55…` |
| Damped prior alone | `damped_velocity` | 0 | 5.1137 | 2.5903 | 5.1137 (jacob-amd), 5.1137 (nic-mac) | 2e-15 | - |
| Hold last | `hold_last` | 0 | 6.0373 | 3.4733 | 6.0373 (jacob-amd), 6.0373 (nic-mac) | 0e+00 | - |
| Const. velocity, 1 step | `constant_velocity_1step` | 0 | 6.9438 | 3.0542 | 6.9438 (jacob-amd) | 0e+00 | - |

All nine match the earlier runs on Jacob's AMD and Nic's Mac to within 2e-9°, so the scores do not depend on the machine.

Notes:
- `v7_sep_base3_addons_angular_ga2` is v8 in the paper (v7 plus the angular loss); `v9_k8_combined_ga2` is k8-combined trained on 100 users.
- v7 (`v7_sep_base3_addons_ga2`) is not in `bench/models.json`: Jacob scored it from a file path. Its ONNX came from Jacob (sha256 matches his result) and was registered at score time with the group-a-v2 training list and 6,651 declared parameters (`scripts/run_v7.py`).

## Reproduce

```bash
# on the Mac, from ~/Dev/experiments/abc-bench-pc (bench/ copy, models/, run_bench.py, run_v7.py, extras.txt = h5py==3.13.0)
ssh pc mkdir -p /srv/data && rsync -ah data/qualified_head_100hz_20261026_131953.h5 pc:/srv/data/
prun --name abc-bench-pc run_bench.py   # eight rows, ~25 min
prun --name abc-bench-pc run_v7.py      # v7, ~3 min
```

## Other PC results used in the paper

- Full test, 1,049 users (Tables 2 and the horizon figure): `~/Dev/experiments/abc-testeval-results/test/`, PC run of 11 Nov 2026 (`report.md`, `figures.md`, `test_receipt.json`). Not copied here: `results.json` is 27 MB and lists user IDs.
