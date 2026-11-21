"""Local CPU latency, as one session on one machine. Not ST's XR target; only numbers measured in the same
session are comparable, so each session is stored (and shown) on its own.

Protocol, ported from ABC_local_test/benchmark_latency.py and compare_nic.py: ONNX Runtime CPU, batch 1;
every call is one harness step (the window slides one sample; a stateful graph gets its states back);
per round each model/thread setting gets `warmup` untimed calls, then runs/rounds timed calls; models
alternate between rounds so drift (thermal, background work) hits all of them alike. Stateful models are
reset at the start of every round. The stream is generated motion, so it is identical on every machine.
"""
import os
import platform
import subprocess
import time
from datetime import datetime, timezone

import numpy as np

from bench.data import sha256_file
from bench.drivers import OnnxModel
from bench.geometry import synthetic_stream


def cpu_name():
    system = platform.system()
    try:
        if system == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if system == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  check=True).stdout.strip()
        if system == "Linux":
            for line in open("/proc/cpuinfo", encoding="utf-8"):
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def machine_info():
    import onnxruntime
    return {"cpu": cpu_name(), "processor": platform.processor(), "machine": platform.machine(),
            "os": platform.platform(), "logical_cores": os.cpu_count(), "python": platform.python_version(),
            "onnxruntime": onnxruntime.__version__, "numpy": np.__version__}


def summarise(seconds):
    ms = np.asarray(seconds) * 1e3
    return {"calls": int(len(ms)), "median_ms": float(np.median(ms)), "mean_ms": float(ms.mean()),
            "p90_ms": float(np.percentile(ms, 90)), "p99_ms": float(np.percentile(ms, 99)),
            "min_ms": float(ms.min()), "max_ms": float(ms.max())}


def time_stream(model, stream, context, warmup, runs):
    """Slide one sample per call; warm-up calls are excluded. Returns per-call seconds."""
    timings = []
    for step in range(warmup + runs):
        window = stream[step:step + context][None]
        began = time.perf_counter()
        model.step(window)
        elapsed = time.perf_counter() - began
        if step >= warmup:
            timings.append(elapsed)
    return timings


def session(models, label, threads=(1,), warmup=200, runs=3000, rounds=3, progress=print):
    """models: {name: onnx path}. Returns one session record (machine + every model's timings)."""
    per_round = -(-runs // rounds)
    stream = synthetic_stream(400 + warmup + per_round)
    calls = {(name, t): OnnxModel(path, name, threads=t) for t in threads for name, path in models.items()}
    timings = {key: [] for key in calls}
    for r in range(rounds):
        for key, model in calls.items():
            model.reset()
            timings[key] += time_stream(model, stream, 400, warmup, per_round)
        if progress:
            progress(f"  latency round {r + 1}/{rounds} done")
    result = {"schema": "bench-latency-v1", "label": label, "created_utc": datetime.now(timezone.utc).isoformat(),
              "machine": machine_info(),
              "protocol": {"runtime": "ONNX Runtime CPUExecutionProvider", "batch_size": 1, "warmup_per_round": warmup,
                           "timed_calls": per_round * rounds, "rounds": rounds, "threads": list(threads),
                           "graph_optimization": "ORT_ENABLE_ALL", "stream": "bench.geometry.synthetic_stream(seed 0)",
                           "timed": "session.run of one harness step; stateful graphs get their states back"},
              "models": {}}
    for (name, t), seconds in timings.items():
        entry = result["models"].setdefault(name, {"onnx": str(models[name]), "onnx_sha256": sha256_file(models[name]),
                                                   "stateful": calls[(name, t)].stateful, "threads": {}})
        entry["threads"][str(t)] = summarise(seconds)
    return result
