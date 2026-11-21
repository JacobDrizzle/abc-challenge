"""Shared, causal-window evaluation of head-orientation forecasts (wxyz).

The evaluator consumes the existing cleaned HDF5; it does not resample raw data.
Predictions receive history only. All scores use float64 quaternion geodesics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
from contextlib import ExitStack
import os
from pathlib import Path
import platform
import tempfile
import time

import h5py
import numpy as np
import scipy
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
PROTOCOL_VERSION = "head-orientation-geodesic-v1"
SPLITS = ("train", "validation", "test")
# ray/'s group-a-v2 export (main.py option 13), in the repo's gitignored data/ folder.
DEFAULT_H5 = ROOT.parents[1] / "data" / "h5" / "qualified_head_100hz_20261026_131953.h5"


@dataclass(frozen=True)
class EvaluationConfig:
    context: int = 400
    horizon: int = 20
    rate_hz: float = 100.0
    stride: int = 1
    batch_size: int = 1024

    def __post_init__(self):
        for field in ("context", "horizon", "stride", "batch_size"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.context < 6:
            raise ValueError("At least 6 history samples are needed for the five-step baseline")
        if not np.isfinite(self.rate_hz) or self.rate_hz <= 0:
            raise ValueError("rate_hz must be positive and finite")


def unit_quaternions(values, *, source="quaternions", tolerance=1e-3):
    """Reject invalid rotations; normalise only small numerical norm deviations."""
    q = np.asarray(values, dtype=np.float64)
    if q.ndim < 1 or q.shape[-1] != 4 or q.size == 0:
        raise ValueError(f"{source}: expected a nonempty array ending in 4, got {q.shape}")
    if not np.isfinite(q).all():
        raise ValueError(f"{source}: NaN or infinity")
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError(f"{source}: zero/near-zero quaternion")
    deviation = float(np.max(np.abs(norms - 1)))
    if deviation > tolerance:
        raise ValueError(f"{source}: quaternion norm deviation {deviation:.6g} exceeds {tolerance}")
    return q / norms


def geodesic_degrees(predicted, target):
    """Sign-invariant SO(3) angle, not quaternion MSE or RMS aggregation."""
    a = unit_quaternions(predicted, source="prediction")
    b = unit_quaternions(target, source="target")
    if a.shape != b.shape:
        raise ValueError(f"Prediction/target shapes differ: {a.shape} vs {b.shape}")
    dot = np.abs(np.sum(a * b, axis=-1))
    return np.rad2deg(2 * np.arccos(np.clip(dot, 0, 1)))


class HoldLast:
    name = "hold_last"
    stateful = False
    metadata = {"trainable_parameters": 0, "method": "Repeat the latest orientation"}

    def __init__(self, config):
        self.config = config

    def predict_batch(self, history):
        return np.repeat(history[:, -1:, :], self.config.horizon, axis=1)

    def predict(self, history):
        return self.predict_batch(history[None])[0]


class ConstantVelocity(HoldLast):
    """Extrapolate the shortest relative rotation over the last lag intervals.

    D = R_now R_past^-1; prediction(h) = Exp((h/lag) Log(D)) R_now.
    Both h and lag are in samples on the same uniform clock.
    """
    def __init__(self, config, lag):
        super().__init__(config)
        if lag < 1 or lag >= config.context:
            raise ValueError("Velocity lag must be positive and smaller than context")
        self.lag = lag
        self.name = f"constant_velocity_{lag}step"
        self.metadata = {"trainable_parameters": 0, "lag_samples": lag,
                         "method": "Shortest relative-rotation extrapolation on SO(3)"}

    def predict_batch(self, history):
        now = Rotation.from_quat(history[:, -1, [1, 2, 3, 0]])
        past = Rotation.from_quat(history[:, -1 - self.lag, [1, 2, 3, 0]])
        delta = (now * past.inv()).as_rotvec()
        scale = np.arange(1, self.config.horizon + 1) / self.lag
        step = Rotation.from_rotvec((delta[:, None, :] * scale[None, :, None]).reshape(-1, 3))
        repeated_now = Rotation.from_quat(np.repeat(now.as_quat(), self.config.horizon, axis=0))
        xyzw = (step * repeated_now).as_quat().reshape(len(history), self.config.horizon, 4)
        return xyzw[..., [3, 0, 1, 2]]


def make_predictors(config, model_specs=()):
    """Additional local adapters are module:factory; factory(config) returns a model."""
    models = [HoldLast(config), ConstantVelocity(config, 1), ConstantVelocity(config, 5)]
    for spec in model_specs:
        module_name, separator, factory_name = spec.partition(":")
        if not separator or not module_name or not factory_name:
            raise ValueError("Model adapter must be module:factory")
        module = importlib.import_module(module_name)
        model = getattr(module, factory_name)(config)
        model.adapter_provenance = {"entry_point": spec}
        source = getattr(module, "__file__", None)
        if source and Path(source).is_file():
            model.adapter_provenance.update({"source_path": str(Path(source).resolve()),
                                             "source_sha256": sha256_file(source)})
        models.append(model)
    names = []
    for model in models:
        name = getattr(model, "name", "")
        if not name or len(name) > 64 or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
            raise ValueError("Predictor names must contain 1–64 lowercase letters, digits, '-' or '_'")
        if name in names:
            raise ValueError(f"Duplicate predictor name: {name}")
        if not callable(getattr(model, "predict", None)):
            raise ValueError(f"{name}: predict(history_wxyz) is required")
        if not isinstance(getattr(model, "stateful", False), bool):
            raise ValueError(f"{name}: stateful must be a bool")
        if getattr(model, "stateful", False):
            if not callable(getattr(model, "reset", None)):
                raise ValueError(f"{name}: stateful predictors must implement reset()")
            if config.stride != 1:
                raise ValueError("Stateful evaluation requires stride=1; samples must not be skipped")
        json.dumps(getattr(model, "metadata", {}), allow_nan=False)
        names.append(name)
    return models


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_for(data_path):
    """The manifest_<run_id>.json that ray/'s export writes next to its HDF5."""
    data_path = Path(data_path)
    with h5py.File(data_path, "r") as h5:
        run_id = h5.attrs.get("run_id")
    if not run_id:
        raise ValueError(f"{data_path.name} has no run_id attribute; expected a ray/ (group-a-v2) export")
    return data_path.parent / f"manifest_{run_id}.json"


def load_split(path, split_name):
    """Users per split from a ray/ export manifest ("splits": {train, validation, test}).
    The HDF5's own /<split> groups are authoritative; tcnn.data.check_corpus checks that they agree."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest.get("splits"), dict):
        raise ValueError(f"{Path(path).name}: expected a ray/ export manifest with a 'splits' object")
    groups = {}
    for key in SPLITS:
        values = manifest["splits"].get(key, [])
        if not isinstance(values, list) or any(not isinstance(x, str) or not x for x in values):
            raise ValueError(f"{key} must be a list of nonempty user-ID strings")
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate user in {key}")
        groups[key] = set(values)
    seen = set()
    for key, users in groups.items():
        if seen.intersection(users):
            raise ValueError(f"User leakage: {key} overlaps another split")
        seen.update(users)
    if split_name == "all":
        selected = seen
    elif split_name in SPLITS:
        selected = groups[split_name]
    else:
        raise ValueError(f"Unknown split: {split_name}")
    if not selected:
        raise ValueError(f"The '{split_name}' split has no users")
    # The split alone, not the whole manifest (that also holds one audit row per inspected recording).
    split = {"run_id": manifest.get("run_id"), "data_sha256": manifest.get("data_sha256"),
             "splits": {key: sorted(users) for key, users in groups.items()}}
    return split, sorted(selected)


def user_paths(h5):
    """{user: path of the user's group}. The ray/ export puts each user under exactly one of the split
    groups listed in the root attribute splits; the same rule as bench/data.py:player_paths."""
    splits = [str(s) for s in h5.attrs.get("splits", [])]
    if not splits or set(h5.keys()) != set(splits):
        raise ValueError("HDF5 must hold only the split groups named by its 'splits' attribute (ray/ group-a-v2 export)")
    paths = {}
    for split in splits:
        for user in h5[split]:
            if user in paths:
                raise ValueError(f"{user} is in two splits: {paths[user].split('/')[0]} and {split}")
            paths[user] = f"{split}/{user}"
    return paths


def inventory(h5, users, config, allow_exploratory):
    """Segments of the selected users. "key" is <user>/<recording>/<segment>, as in the old root layout, so
    cohort fingerprints keep their form; "path" is where the segment sits in the file."""
    if h5.attrs.get("quaternion_order") != "wxyz":
        raise ValueError("HDF5 must declare quaternion_order='wxyz'")
    if not np.isclose(float(h5.attrs.get("rate_hz", np.nan)), config.rate_hz, rtol=0, atol=1e-9):
        raise ValueError("HDF5 rate_hz does not match the evaluation clock")
    paths = user_paths(h5)
    missing = set(users) - set(paths)
    if missing:
        raise ValueError(f"Selected users absent from HDF5: {sorted(missing)}")
    included, skipped = [], []
    for user in users:
        for recording in sorted(h5[paths[user]]):
            for segment in sorted(h5[paths[user]][recording]):
                key = f"{user}/{recording}/{segment}"
                path = f"{paths[user]}/{recording}/{segment}"
                group = h5[path]
                if not isinstance(group, h5py.Group) or not all(k in group for k in ("time_s", "quaternion_wxyz")):
                    raise ValueError(f"{key}: expected time_s and quaternion_wxyz datasets")
                q, t = group["quaternion_wxyz"], group["time_s"]
                if q.ndim != 2 or q.shape[1] != 4 or t.shape != (len(q),):
                    raise ValueError(f"{key}: invalid dataset shapes")
                flag = group.attrs.get("challenge_filter_pass")
                if flag is not None and not (isinstance(flag, (bool, np.bool_, int, np.integer)) and flag in (0, 1)):
                    raise ValueError(f"{key}: challenge_filter_pass must be a boolean or 0/1")
                qualified = flag is not None and bool(flag)
                reason = None
                if not qualified and not allow_exploratory:
                    reason = "motion_filter_failed" if flag is not None else "qualification_missing"
                elif len(q) < config.context + config.horizon:
                    reason = "too_short"
                if reason:
                    skipped.append({"key": key, "reason": reason})
                    continue
                windows = (len(q) - config.context - config.horizon) // config.stride + 1
                included.append({"key": key, "path": path, "user": user, "recording": recording,
                                 "samples": len(q), "windows": windows,
                                 "challenge_filter_pass": qualified})
    if not included:
        raise ValueError(f"No eligible windows in {len(users)} selected user(s); skipped {len(skipped)} segments. "
                         "For the existing non-qualifying sample only, explicitly use --allow-exploratory.")
    return included, skipped


def window_batches(group, config):
    """Read bounded blocks. First anchor is C-1; targets are anchor+1 … anchor+H."""
    count = (len(group["time_s"]) - config.context - config.horizon) // config.stride + 1
    for start in range(0, count, config.batch_size):
        n = min(config.batch_size, count - start)
        anchors = config.context - 1 + (start + np.arange(n, dtype=np.int64)) * config.stride
        begin, end = int(anchors[0] - config.context + 1), int(anchors[-1] + config.horizon + 1)
        times = np.asarray(group["time_s"][begin:end], dtype=np.float64)
        if not np.isfinite(times).all() or not np.allclose(np.diff(times), 1 / config.rate_hz, rtol=0, atol=1e-6):
            raise ValueError(f"{group.name}: timestamps are not a finite uniform {config.rate_hz:g} Hz grid; split gaps upstream")
        q = unit_quaternions(group["quaternion_wxyz"][begin:end], source=group.name)
        if np.any(np.sum(q[:-1] * q[1:], axis=-1) < -1e-8):
            raise ValueError(f"{group.name}: quaternion signs are discontinuous; align them in the data preparation step")
        local = anchors - begin
        view = np.lib.stride_tricks.sliding_window_view(q, config.context, axis=0).transpose(0, 2, 1)
        history = np.array(view[local - config.context + 1], dtype=np.float32, copy=True)
        target = q[local[:, None] + np.arange(1, config.horizon + 1)[None, :]]
        yield anchors, history, target


def predict_batch(model, history, config):
    # A private copy prevents one predictor from corrupting another predictor's inputs.
    if not getattr(model, "stateful", False) and callable(getattr(model, "predict_batch", None)):
        result = np.asarray(model.predict_batch(history.copy()))
    else:
        result = np.stack([np.asarray(model.predict(item.copy())) for item in history])
    expected = (len(history), config.horizon, 4)
    if result.shape != expected:
        raise ValueError(f"{model.name}: expected {expected}, got {result.shape}; return one trajectory per input")
    return unit_quaternions(result, source=f"{model.name} output")


def _reset(model):
    if callable(getattr(model, "reset", None)):
        model.reset()


def measure_latency(model, group, config, steps):
    """Separate sequential replay, after scoring. Warm-up is excluded; state is reset."""
    available = len(group["time_s"]) - config.context - config.horizon + 1
    warmup = min(20, max(0, available - 1))
    measured = min(steps, available - warmup)
    if measured < 1:
        return None
    q = unit_quaternions(group["quaternion_wxyz"][:config.context + warmup + measured - 1]).astype(np.float32)
    synchronize = getattr(model, "synchronize", lambda: None)
    durations = []
    _reset(model)
    try:
        for i in range(warmup + measured):
            history = q[i:i + config.context].copy()
            synchronize()
            start = time.perf_counter_ns()
            result = model.predict(history)
            synchronize()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if np.asarray(result).shape != (config.horizon, 4):
                raise ValueError(f"{model.name}: invalid output shape during latency replay")
            unit_quaternions(result, source=f"{model.name} latency output")
            if i >= warmup:
                durations.append(elapsed)
    finally:
        _reset(model)
    return {"median_ms": float(np.median(durations)), "p95_ms": float(np.percentile(durations, 95)),
            "measured_steps": measured, "warmup_steps": warmup, "batch_size": 1,
            "synchronization_hook": callable(getattr(model, "synchronize", None)),
            "scope": "Local Python predict() call including adapter work; excludes data loading, scorer and input copy. Not ST hardware latency."}


def _summarize(sums, count):
    means = sums / count
    return {"windows": int(count), "mean_deg_by_horizon": means.tolist(),
            "mean_endpoint_deg": float(means[-1]), "mean_all_horizons_deg": float(means.mean())}


def _group_summaries(rows, by, horizon):
    groups = {}
    for row in rows:
        key = row[by] if by == "user" else f"{row['user']}/{row['recording']}"
        entry = groups.setdefault(key, {"count": 0, "sums": np.zeros(horizon)})
        entry["count"] += row["windows"]
        entry["sums"] += np.array(row["mean_deg_by_horizon"]) * row["windows"]
    return {key: _summarize(value["sums"], value["count"]) for key, value in sorted(groups.items())}


def evaluate(data_path, split_path, output_path, *, config=None, split_name="validation",
             allow_exploratory=False, model_specs=(), latency_steps=0, progress=print):
    config = config or EvaluationConfig()
    if latency_steps < 0:
        raise ValueError("latency_steps cannot be negative")
    data_path, split_path, output_path = (Path(p).resolve() for p in (data_path, split_path, output_path))
    if output_path.exists():
        raise ValueError(f"Output already exists: {output_path}. Choose a new run directory; previous results are preserved.")
    split_document, users = load_split(split_path, split_name)
    models = make_predictors(config, model_specs)
    initial_stat = data_path.stat()
    with h5py.File(data_path, "r") as h5:
        segments, skipped = inventory(h5, users, config, allow_exploratory)
        total = sum(s["windows"] for s in segments)
        exploratory = any(not s["challenge_filter_pass"] for s in segments)
        if progress:
            progress(f"Scoring {total:,} windows from {len(segments)} segments / {len(set(s['user'] for s in segments))} users; "
                     f"{'EXPLORATORY: includes filter failures' if exploratory else 'filter-qualified according to input metadata'}.")
        source_sha = sha256_file(data_path)
        cohort = {"data_sha256": source_sha, "users": users, "segments": [s["key"] for s in segments],
                  "context": config.context, "horizon": config.horizon, "rate_hz": config.rate_hz,
                  "stride": config.stride, "protocol": PROTOCOL_VERSION}
        fingerprint = hashlib.sha256(json.dumps(cohort, sort_keys=True).encode()).hexdigest()
        report = {"protocol_version": PROTOCOL_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "cohort_fingerprint": fingerprint, "config": asdict(config), "split": split_name,
                  "split_document": split_document, "selected_users": users,
                  "evaluated_users": sorted({s["user"] for s in segments}),
                  "selected_users_without_windows": sorted(set(users) - {s["user"] for s in segments}),
                  "exploratory": exploratory, "allow_exploratory": allow_exploratory,
                  "synthetic": bool(h5.attrs.get("synthetic", False)),
                  "windows": total, "segments": segments, "skipped_segments": skipped,
                  "metric": "Window-weighted mean sign-invariant geodesic angle in degrees at the final predicted sample",
                  "horizons_ms": (np.arange(1, config.horizon + 1) * 1000 / config.rate_hz).tolist(),
                  "timing_note": "At 100 Hz, step 10 is 100 ms and step 20 is 200 ms. Challenge prose calls step 20 100 ms; unresolved.",
                  "preprocessing_note": "Input is an offline-prepared uniform stream. Causality of raw-data interpolation is not certified by this evaluator.",
                  "input": {"path": str(data_path), "bytes": initial_stat.st_size, "sha256": source_sha,
                            "split_path": str(split_path), "split_sha256": sha256_file(split_path)},
                  "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                  "scipy": scipy.__version__, "h5py": h5py.__version__,
                                  "platform": platform.platform(), "processor": platform.processor(),
                                  "pipeline_sha256": sha256_file(__file__)}, "models": {}}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".baseline-run-", dir=output_path.parent) as temporary, ExitStack() as maps:
            out = Path(temporary)
            index = np.lib.format.open_memmap(out / "window_index.npy", mode="w+",
                                              dtype=[("segment_id", "<i4"), ("anchor", "<i8")], shape=(total,))
            maps.callback(index._mmap.close)
            endpoints = {}
            for model in models:
                values = np.lib.format.open_memmap(out / f"endpoint_{model.name}.npy", mode="w+",
                                                   dtype="<f8", shape=(total,))
                maps.callback(values._mmap.close)
                endpoints[model.name] = values
            sums = {m.name: np.zeros(config.horizon, dtype=np.float64) for m in models}
            segment_scores = {m.name: [] for m in models}
            cursor = 0
            for segment_id, segment in enumerate(segments):
                segment_sums = {m.name: np.zeros(config.horizon) for m in models}
                for model in models:
                    _reset(model)
                for anchors, history, targets in window_batches(h5[segment["path"]], config):
                    rows = slice(cursor, cursor + len(anchors))
                    index[rows] = np.array([(segment_id, int(a)) for a in anchors], dtype=index.dtype)
                    for model in models:
                        predictions = predict_batch(model, history, config)
                        errors = geodesic_degrees(predictions, targets)
                        endpoints[model.name][rows] = errors[:, -1]
                        block_sum = errors.sum(axis=0, dtype=np.float64)
                        sums[model.name] += block_sum
                        segment_sums[model.name] += block_sum
                    cursor += len(anchors)
                for model in models:
                    segment_scores[model.name].append({"segment": segment["key"], "user": segment["user"],
                                                       "recording": segment["recording"],
                                                       **_summarize(segment_sums[model.name], segment["windows"])})
                if progress:
                    progress(f"  Segment {segment_id + 1}/{len(segments)}: {cursor:,}/{total:,} windows")
            if cursor != total:
                raise RuntimeError("Internal window count mismatch")
            index.flush()
            for model in models:
                values = endpoints[model.name]
                values.flush()
                result = _summarize(sums[model.name], total)
                result.update({"median_endpoint_deg": float(np.median(values)),
                               "p95_endpoint_deg": float(np.percentile(values, 95)),
                               "p99_endpoint_deg": float(np.percentile(values, 99)),
                               "by_segment": segment_scores[model.name],
                               "by_user": _group_summaries(segment_scores[model.name], "user", config.horizon),
                               "by_recording": _group_summaries(segment_scores[model.name], "recording", config.horizon),
                               "stateful": getattr(model, "stateful", False),
                               "metadata": getattr(model, "metadata", {}),
                               "adapter_provenance": getattr(model, "adapter_provenance", None),
                               "endpoint_errors_file": f"endpoint_{model.name}.npy"})
                result["equal_user_mean_endpoint_deg"] = float(np.mean([r["mean_endpoint_deg"] for r in result["by_user"].values()]))
                result["local_latency"] = measure_latency(model, h5[segments[0]["path"]], config, latency_steps) if latency_steps else None
                report["models"][model.name] = result
            hold = report["models"]["hold_last"]["mean_endpoint_deg"]
            for result in report["models"].values():
                result["endpoint_change_vs_hold_deg"] = result["mean_endpoint_deg"] - hold
                result["endpoint_improvement_vs_hold_pct"] = 100 * (hold - result["mean_endpoint_deg"]) / hold if hold > 1e-8 else None
            final_stat = data_path.stat()
            if (initial_stat.st_mtime_ns, initial_stat.st_size) != (final_stat.st_mtime_ns, final_stat.st_size):
                raise ValueError("Input HDF5 changed during evaluation; discard this run and retry on a fixed file")
            (out / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            write_report(report, out)
            # Close disk-backed files before the successful run is published.
            maps.close()
            out.rename(output_path)
    return report


def write_report(report, out):
    config = report["config"]
    endpoint = config["horizon"]
    status = "EXPLORATORY — includes sessions that fail the challenge filter" if report["exploratory"] else "Filter-qualified local evaluation — not an official challenge score"
    if report.get("synthetic", False):
        status = "SYNTHETIC SOFTWARE CHECK — artificial data; not a research result"
    lines = ["# Head-orientation baseline evaluation", "", f"**{status}.**", "",
             f"Split: **{report['split']}** · {len(report['evaluated_users'])} users · {len(report['segments'])} segments · **{report['windows']:,} windows**.", "",
             f"History: {config['context']} samples. Forecast: {endpoint} samples. Clock: {config['rate_hz']:g} Hz. Stride: {config['stride']}.", "",
             "Every predictor uses the same windows and targets. Lower angular error is better.", "",
             "| Predictor | Endpoint mean (°) | Endpoint median (°) | Endpoint p95 (°) | All-horizon mean (°) | Local median call (ms) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in report["models"].items():
        latency = result["local_latency"]
        ms = f"{latency['median_ms']:.4f}" if latency else "Not measured"
        lines.append(f"| {name} | {result['mean_endpoint_deg']:.6f} | {result['median_endpoint_deg']:.6f} | {result['p95_endpoint_deg']:.6f} | {result['mean_all_horizons_deg']:.6f} | {ms} |")
    lines += ["", f"The primary number is the **window-weighted mean error at step {endpoint}**. Median, p95, all-horizon mean, and equal-user mean are diagnostics; they are not interchangeable with it.", "",
              "## Selected horizons", "", "| Predictor | Step | Actual time ahead (ms) | Mean error (°) |", "| --- | ---: | ---: | ---: |"]
    for name, result in report["models"].items():
        for h in sorted({h for h in (1, 5, 10, endpoint) if h <= endpoint}):
            lines.append(f"| {name} | {h} | {h * 1000 / config['rate_hz']:g} | {result['mean_deg_by_horizon'][h - 1]:.6f} |")
    lines += ["", "![Mean error versus forecast horizon](horizon_errors.png)", "", "## Interpretation", "",
              "- " + report["timing_note"],
              "- CPU/Python call timings are a local diagnostic, not ST’s XR-device measurements. GPU adapters must synchronise before timing is meaningful.",
              "- Overlapping windows are correlated. Their count is not the number of independent test examples.",
              "- " + report["preprocessing_note"],
              "- Qualification is taken from the input segment metadata; the evaluator does not rerun or certify the upstream filter.",
              "- All three built-in predictors have zero trainable parameters. They use only past observations.", "",
              "## Reusing these results", "",
              "`results.json` contains full horizon curves and per-user, per-recording, and per-segment summaries. Each `endpoint_*.npy` stores one endpoint error per row of `window_index.npy`; its segment IDs refer to the ordered `segments` list in the JSON. Anchor indices are zero based within each segment.", "",
              f"Cohort fingerprint: `{report['cohort_fingerprint']}`. Compare runs only when the evaluation cohort and protocol match. A future model can be evaluated alongside the baselines using the same command’s `--model` option.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    for name, result in report["models"].items():
        ax.plot(report["horizons_ms"], result["mean_deg_by_horizon"], label=name.replace("_", " "), linewidth=2)
    data_label = "synthetic software check" if report.get("synthetic", False) else (
        "exploratory data" if report["exploratory"] else "qualified local data")
    ax.set(xlabel=f"Time ahead at {config['rate_hz']:g} Hz (ms)", ylabel="Mean geodesic error (degrees)",
           title=f"{report['split'].title()} users · {data_label}\nPrimary endpoint: step {endpoint} / {report['horizons_ms'][-1]:g} ms")
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(out / "horizon_errors.png", dpi=160)
    plt.close(fig)
