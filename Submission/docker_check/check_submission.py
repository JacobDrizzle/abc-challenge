"""Clean-room check of a 9th ABC Challenge submission, run inside the Docker image (see README.md).

The image is a fresh Linux install that has only run `pip install -r requirements.txt` in the submission folder, and
run_checks.sh starts it with the network switched off. This script runs from that folder and does what the organisers
will do, plus the rejection checks on the Submitting Artifacts page. It imports only numpy and onnxruntime, the two
packages the submission installs, so it can't hide a missing dependency.

PASS / FAIL = an organiser rule (any FAIL gives exit code 1). WARN = a probe beyond the rules. INFO = facts only.
"""
import inspect
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import onnxruntime as ort

SUBMISSION = Path.cwd()
REQUIRED_FILES = ["model_fp32.onnx", "run_inference.py", "requirements.txt", "model_card.md", "README.md"]
STRAY = re.compile(r"^(__pycache__|.*\.pyc|\.DS_Store|Thumbs\.db|desktop\.ini|\.ipynb_checkpoints|pred.*\.npy)$")
CONTEXT, HORIZON, HZ = 400, 20, 100.0
ROUNDING = 1e-4  # largest |difference| we accept as float rounding between machines

results = []


def report(status, name, detail=""):
    results.append(status)
    print(f"[{status}] {name}" + (f": {detail}" if detail else ""), flush=True)


def check(ok, name, detail="", probe=False):
    report("PASS" if ok else ("WARN" if probe else "FAIL"), name, detail)
    return bool(ok)


def section(title, fn):
    """Run one group of checks. A crash inside it is a FAIL, and the other groups still run."""
    print(f"\n== {title} ==", flush=True)
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - we want every crash in the report
        traceback.print_exc()
        report("FAIL", f"{title} crashed", f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- quaternions (w, x, y, z), Hamilton, float64

def qmul(a, b):
    w1, x1, y1, z1 = np.moveaxis(np.asarray(a, float), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(b, float), -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def rotation(axis, angle_rad):
    """Rotation(s) by angle_rad (scalar or [n]) about one axis."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    half = np.asarray(angle_rad, float)[..., None] / 2
    return np.concatenate([np.cos(half), np.sin(half) * axis], -1)


def small_rotations(rotvecs):
    """Rotations for small rotation vectors [n, 3] (first-order, then normalised)."""
    q = np.concatenate([np.ones((len(rotvecs), 1)), rotvecs / 2], -1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def angle_deg(a, b):
    """Geodesic angle between rotations in degrees (q and -q are the same rotation)."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(a * b, -1)), 0, 1)))


def harness_ready(q):
    """What the harness promises: unit norm, hemisphere-aligned (same sign side as the sample before), float32."""
    q = np.array(q, float)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q.astype(np.float32)


START = qmul(rotation([0, 1, 0], np.radians(40)), rotation([1, 0, 0], np.radians(11)))  # turned 40°, looking down 11°
T = np.arange(CONTEXT + HORIZON) / HZ  # 400 samples of context, then the 20 the model should predict


def steady_turn(deg_per_s, axis):
    return qmul(rotation(axis, np.radians(deg_per_s) * T), START)


def random_head_motion(n, seed):
    """Smooth made-up motion: the body-frame angular velocity drifts as a random walk (no BOXRR data)."""
    rng = np.random.default_rng(seed)
    omega = np.cumsum(rng.normal(0, 0.05, (n, 3)), 0)  # rad/s
    q = [START]
    for w in omega[1:]:
        angle = np.linalg.norm(w) / HZ
        q.append(qmul(q[-1], rotation(w, angle)) if angle > 0 else q[-1])
    return np.array(q)


def synthetic_cases():
    rng = np.random.default_rng(0)
    n = CONTEXT + HORIZON
    # 5°/s about a tilted axis, passing 180° (w = 0) 10 samples before the end of the context, so w changes sign
    # inside the window.
    through_180 = rotation([0.2, 1, 0.1], np.radians(180 + 5 * (T - T[CONTEXT - 11])))
    return {
        "still head, identity": np.tile([1.0, 0, 0, 0], (n, 1)),
        "still head, turned and tilted": np.tile(START, (n, 1)),
        "still head + 0.02° sensor noise": qmul(START, small_rotations(rng.normal(0, np.radians(0.02), (n, 3)))),
        "steady turn 45°/s": steady_turn(45, [0, 1, 0]),
        "fast spin 1000°/s, tilted axis": steady_turn(1000, [0.3, 1, 0.2]),
        "rotation through 180° (w crosses 0)": through_180,
        "random head motion": random_head_motion(n, 1),
    }


# ---------------------------------------------------------------- checks

def environment():
    print(f"[INFO] python {platform.python_version()} on {platform.platform()} ({' '.join(platform.libc_ver())})")
    print(f"[INFO] numpy {np.__version__}, onnxruntime {ort.__version__}, providers {ort.get_available_providers()}")
    if shutil.which("nvidia-smi"):
        gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
        print(f"[INFO] GPU visible in the container: {gpus or 'nvidia-smi found but listed nothing'}")
    else:
        print("[INFO] no GPU visible in the container")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.split()
    print(f"[INFO] installed by requirements.txt (with dependencies): {', '.join(freeze)}")
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=3).close()
        network_off = False
    except OSError:
        network_off = True
    check(network_off, "network is off, so the no-network rule is being tested",
          "" if network_off else "start the container with --network none", probe=True)


def archive():
    files = sorted(p.relative_to(SUBMISSION).as_posix() for p in SUBMISSION.rglob("*") if p.is_file())
    check(SUBMISSION.name.endswith("_headpose"), "folder is named TEAMNAME_headpose", SUBMISSION.name)
    for name in REQUIRED_FILES:
        check((SUBMISSION / name).is_file(), f"required file {name}")
    optional = [n for n in ("model_int8.onnx", "weights") if (SUBMISSION / n).exists()]
    print(f"[INFO] optional parts present: {', '.join(optional) or 'none'}; {len(files)} files in total")
    lines = (SUBMISSION / "requirements.txt").read_text().splitlines()
    reqs = [line.split("#")[0].strip() for line in lines]
    reqs = [r for r in reqs if r]
    # name==version, optionally with extras and an environment marker such as `; python_version < "3.11"`.
    unpinned = [r for r in reqs if not re.fullmatch(r"[A-Za-z0-9_.\-]+(\[[^\]]*\])?==[^=\s;]+(\s*;.+)?", r)]
    check(reqs and not unpinned, "every requirement is pinned with ==", ", ".join(unpinned or reqs))
    stray = [f for f in files if any(STRAY.match(part) for part in f.split("/"))]
    check(not stray, "no stray files (__pycache__, .pyc, .DS_Store, pred*.npy)", ", ".join(stray), probe=True)
    size = sum((SUBMISSION / f).stat().st_size for f in files)
    check(size <= 500e6, "folder is under the 500 MB archive limit", f"{size / 1e6:.2f} MB")


def command_line():
    sample, expected = SUBMISSION / "sample_input.npy", SUBMISSION / "sample_expected_output.npy"
    if not sample.is_file():
        report("WARN", "no sample_input.npy, so the command-line check was skipped")
        return
    shutil.copy(sample, SUBMISSION / "sample.npy")  # the organisers' command uses this name
    cmd = ["python", "run_inference.py", "--model_dir", ".", "--input", "sample.npy", "--output", "pred.npy"]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=SUBMISSION, capture_output=True, text=True)
    wall = time.perf_counter() - start
    if not check(proc.returncode == 0, "the organisers' command runs: " + " ".join(cmd),
                 f"{wall:.2f} s" if proc.returncode == 0 else proc.stderr.strip()[-800:]):
        return
    pred = np.load(SUBMISSION / "pred.npy")
    check(pred.shape == (HORIZON, 4) and pred.dtype == np.float32, "pred.npy is float32 [20, 4]",
          f"{pred.dtype} {list(pred.shape)}")
    check(np.isfinite(pred).all(), "pred.npy is finite")
    if expected.is_file():
        ref = np.load(expected)
        diff = float(np.abs(pred - ref).max())
        check(diff <= ROUNDING, "pred.npy matches sample_expected_output.npy",
              f"max |diff| {diff:.1e}, max angle {angle_deg(pred, ref).max():.1e}°")

    np.save(SUBMISSION / "sample_batched.npy", np.load(sample)[None])  # [1, 400, 4], which the README also accepts
    proc = subprocess.run(cmd[:5] + ["sample_batched.npy", "--output", "pred_batched.npy"], cwd=SUBMISSION,
                          capture_output=True, text=True)
    check(proc.returncode == 0 and np.array_equal(np.load(SUBMISSION / "pred_batched.npy"), pred),
          "a [1, 400, 4] input gives the same prediction", proc.stderr.strip()[-300:])


def python_api():
    global run_inference
    sys.path.insert(0, str(SUBMISSION))
    import run_inference  # noqa: F401 - the submission's own file

    load_model, predict = getattr(run_inference, "load_model", None), getattr(run_inference, "predict", None)
    check(callable(load_model) and list(inspect.signature(load_model).parameters) == ["model_dir"],
          "run_inference.load_model(model_dir) exists")
    check(callable(predict) and list(inspect.signature(predict).parameters) == ["model", "context"],
          "run_inference.predict(model, context) exists")
    model = load_model(str(SUBMISSION))
    for name, stream in synthetic_cases().items():
        q = harness_ready(stream)
        pred = predict(model, q[:CONTEXT])
        ok = isinstance(pred, np.ndarray) and pred.shape == (HORIZON, 4) and pred.dtype == np.float32
        ok = ok and bool(np.isfinite(pred).all())
        detail = (f"error at +200 ms {angle_deg(pred[-1], q[-1]):.2f}°, "
                  f"|norm - 1| ≤ {np.abs(np.linalg.norm(pred, axis=-1) - 1).max():.0e}") if ok else repr(pred)[:200]
        check(ok, f"predict() gives a finite float32 [20, 4]: {name}", detail)


def onnx_graph():
    session = ort.InferenceSession(str(SUBMISSION / "model_fp32.onnx"), providers=["CPUExecutionProvider"])
    ins, outs = session.get_inputs(), session.get_outputs()
    check(ins[0].name == "context" and list(ins[0].shape) == [1, CONTEXT, 4],
          "graph input 0 is context [1, 400, 4]", f"{ins[0].name} {ins[0].shape}")
    check(outs[0].name == "prediction" and list(outs[0].shape) == [1, HORIZON, 4],
          "graph output 0 is prediction [1, 20, 4]", f"{outs[0].name} {outs[0].shape}")
    k = len(ins) - 1
    check([i.name for i in ins[1:]] == [f"state_in_{j}" for j in range(k)]
          and [o.name for o in outs[1:]] == [f"state_out_{j}" for j in range(k)]
          and all(i.shape == o.shape for i, o in zip(ins[1:], outs[1:])),
          "state tensors are state_in_i / state_out_i with matching shapes",
          ", ".join(f"{i.name} {i.shape}" for i in ins[1:]) or "stateless")
    check(all(x.type == "tensor(float)" for x in ins + outs), "every graph input and output is float32",
          ", ".join(sorted({x.type for x in ins + outs})))
    check(all(isinstance(d, int) for x in ins + outs for d in x.shape), "every graph shape is static")

    # Drive the graph the way the harness does: zero state at session start, one new sample per call, the full
    # 400-sample window as context, and each state_out_i fed back as state_in_i.
    stream = harness_ready(random_head_motion(CONTEXT + 100, 2))
    state = {i.name: np.zeros(i.shape, np.float32) for i in ins[1:]}
    names = [o.name for o in outs]
    bad = []
    for step in range(101):
        out = session.run(names, {"context": stream[step:step + CONTEXT][None], **state})
        if out[0].shape != (1, HORIZON, 4) or not np.isfinite(out[0]).all():
            bad.append(step)
        state = {i.name: v for i, v in zip(ins[1:], out[1:])}
    check(not bad, "harness-style streaming of the raw graph: 101 calls, all finite [1, 20, 4]",
          f"bad at calls {bad[:10]}" if bad else "")


def streaming():
    load_model, predict = run_inference.load_model, run_inference.predict
    stream = harness_ready(random_head_motion(CONTEXT + 100, 3))
    model = load_model(str(SUBMISSION))
    start = time.perf_counter()
    predict(model, stream[:CONTEXT])
    cold_ms = (time.perf_counter() - start) * 1e3
    times = []
    for i in range(1, 101):  # the next 100 windows of the same stream, each shifted on by one sample
        start = time.perf_counter()
        streamed = predict(model, stream[i:i + CONTEXT])
        times.append((time.perf_counter() - start) * 1e3)
    fast_ms = float(np.median(times))
    print(f"[INFO] predict() on this CPU: new window {cold_ms:.1f} ms, next shifted window {fast_ms:.3f} ms (median)")
    check(fast_ms < cold_ms / 10, "shifted windows take the one-step fast path", probe=True)
    cold = predict(load_model(str(SUBMISSION)), stream[100:100 + CONTEXT])
    diff = float(np.abs(streamed - cold).max())
    check(diff <= ROUNDING, "after 100 streamed steps, the prediction equals a cold start on the same window",
          f"max |diff| {diff:.1e}")

    other = harness_ready(random_head_motion(CONTEXT, 4))
    check(np.array_equal(predict(model, other), predict(load_model(str(SUBMISSION)), other)),
          "a new session's window ignores the previous session's state (no leak across sessions)")


def probes():
    predict, model = run_inference.predict, run_inference.load_model(str(SUBMISSION))
    # The README asks for hemisphere-aligned input. If the harness instead forces w >= 0, the sign jumps where w
    # crosses 0. A sign-blind model gives the same rotation either way.
    stream = synthetic_cases()["rotation through 180° (w crosses 0)"]
    aligned = harness_ready(stream)[:CONTEXT]
    w_positive = (aligned * np.where(aligned[:, :1] < 0, -1, 1)).astype(np.float32)
    flips = int((np.sum(w_positive[1:] * w_positive[:-1], -1) < 0).sum())
    a, b = predict(model, aligned), predict(model, w_positive)
    gap = float(angle_deg(a[-1], b[-1])) if np.isfinite(b).all() else float("nan")
    check(np.isfinite(b).all() and gap < 0.5, "w >= 0 input instead of hemisphere-aligned gives the same forecast",
          f"{flips} sign flip(s) in the window, step-20 gap {gap:.3f}°", probe=True)

    # If the harness ever padded a short history with zeros (the page says it always sends 400 real samples).
    padded = harness_ready(random_head_motion(CONTEXT, 5))
    padded[:100] = 0
    out = predict(model, padded)
    check(np.isfinite(out).all(), "zero rows at the start of the window still give a finite forecast", probe=True)


def main():
    print(f"Checking {SUBMISSION}")
    section("Environment", environment)
    section("Archive contents", archive)
    section("Command line (what the organisers run)", command_line)
    section("Python API: load_model / predict", python_api)
    section("ONNX graph interface", onnx_graph)
    section("Streaming state in run_inference.py", streaming)
    section("Probes beyond the rules", probes)
    n_fail, n_warn, n_pass = results.count("FAIL"), results.count("WARN"), results.count("PASS")
    print(f"\nRESULT: {'OK' if not n_fail else 'FAILED'} ({n_pass} passed, {n_warn} warnings, {n_fail} failed)")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
