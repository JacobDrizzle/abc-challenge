# Docker clean-room check

The Submitting Artifacts page says `run_inference.py` must run "on a clean Linux machine with CUDA" after only
`pip install -r requirements.txt`, with no network at inference. This folder rebuilds that machine in Docker and runs
the submission the way the organisers will. **It is not part of the submission: never put it in the zip.**

| File | What it does |
| --- | --- |
| `Dockerfile` | Fresh Linux → empty venv → copy the submission folder → `pip install -r requirements.txt`, nothing else |
| `check_submission.py` | Runs inside the container: the organisers' command, the API, the ONNX interface, the rejection checks |
| `run_checks.sh` | Builds and runs the check on six Linux images, network off, GPU attached where possible |

## Run it

From the repo root, in Git Bash (Windows), macOS or Linux, with Docker running:

```bash
bash Submission/docker_check/run_checks.sh                                   # the Motion_Pending_headpose/ folder
bash Submission/docker_check/run_checks.sh Submission/Motion_Pending_headpose.zip   # the zip you will upload
TARGETS="cuda py3.12" bash Submission/docker_check/run_checks.sh             # a subset
```

Run it on the zip once more just before uploading. That catches a zip with the wrong folder layout or a missing file.

One image by hand (PowerShell or bash, from `Submission/`):

```bash
docker build --build-context checks=docker_check -f docker_check/Dockerfile -t abc-check Motion_Pending_headpose
docker run --rm --network none --gpus all abc-check     # drop --gpus all on a machine without an NVIDIA GPU
docker run --rm -it --network none abc-check bash       # a shell in the clean machine, to poke around
```

## Targets

| Target | Image | Why |
| --- | --- | --- |
| `cuda` | `nvidia/cuda:12.6.3-base-ubuntu24.04` + Ubuntu's Python 3.12, `--gpus all` | Closest to "clean Linux with CUDA" |
| `py3.11`, `py3.12`, `py3.13` | `python:3.x-slim` (Debian) | The versions our README says we support |
| `py3.10`, `py3.14` | `python:3.x-slim` | Probes: Ubuntu 22.04's and the newest Python. A failure here does not fail the run |

The build context is the submission folder only, so nothing else from the repo can leak in. `pip install` runs at
build time (it needs the network). The check then runs with `--network none`.

## What it checks

`[FAIL]` = breaks an organiser rule. `[WARN]` = a probe beyond the rules, worth a look. `[INFO]` = facts.

1. **Environment:** Python, glibc, installed packages, GPU visible, network really off.
2. **Archive:** folder named `*_headpose`, the five required files, every requirement pinned `==`, no stray files
   (`__pycache__`, `pred.npy`), under 500 MB.
3. **Command line:** exactly `python run_inference.py --model_dir . --input sample.npy --output pred.npy`. The output
   must be float32 `[20, 4]`, finite and equal to `sample_expected_output.npy` within 1e-4 (float rounding).
4. **Python API:** `load_model(model_dir)` and `predict(model, context)` exist. `predict` returns a finite float32
   `[20, 4]` on synthetic edge cases: still head, sensor noise, steady turn, 1000°/s spin, w crossing 0 near the end,
   random motion. The error at +200 ms is printed for each as a sanity check.
5. **ONNX graph:** `context [1, 400, 4]` → `prediction [1, 20, 4]`, `state_in_i`/`state_out_i` naming and shapes,
   float32, static shapes. The raw graph is then driven the harness way (zero state, one sample per call, state fed
   back) for 101 calls.
6. **Streaming:** the shifted-window fast path gives the same answer as a cold start, and a new session doesn't
   inherit the previous session's state.
7. **Probes:** `w ≥ 0` input instead of hemisphere-aligned, and zero rows at the start of the window.

All test motion is synthetic. No BOXRR data is used (DUA clause 4).
