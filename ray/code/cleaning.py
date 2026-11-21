"""One recording through the challenge rules: head channel -> cleaning -> motion filter -> 100 Hz.

process_func (what goes into saved/) and the HDF5 export both call this module, so they can't
disagree. It is a line-for-line port of Group B's reference pipeline (abc_data/xror.py read_head,
cleaning.py, motion_filter.py, resample.py and preparation.py), with the settings in config.py.

Input is a recording as XROR.unpack(raw).data returns it. Quaternions are xyzw (scipy order)
inside this module and come out of resample() as wxyz."""

import warnings

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .config import (EULER_ORDER, H5_MAX_GAP_S, H5_MIN_SAMPLES, H5_RATE_HZ, MIN_ASSESSED_SAMPLES,
                     MIN_QUAT_NORM, MIN_WINDOW_SAMPLES, PREPARATION_VERSION, RATE_BAND_HZ,
                     SWEEP_MIN_DEG, SWEEP_PERCENTILES, TRIM_S, VELOCITY_MIN_DEG_S, WINDOW_S)


POLICIES = {  # written once into every export manifest
    "duplicates": "Keep the first valid sample of adjacent equal timestamps; never sort",
    "startup_trim": "First 5 seconds of accumulated positive timestamp intervals; clock resets add zero elapsed time",
    "backwards_jumps": "Split; never sort pieces together",
    "motion_windows_split_on": "backward clock jumps only (tracking gaps stay inside their window)",
    "velocity_denominator": "nominal 5 s window",
    "sweep_unwrapped": False,
    "coordinate_convention": "Source rotation frame preserved; export scalar-first wxyz = "
                             "[HEAD.1, HEAD.i, HEAD.j, HEAD.k]; no physical-axis remapping",
}


def read_head(data: dict, user: str | None = None, recording: str | None = None):
    """Return (metadata, t, q_xyzw). t and q are None when the recording has no HEAD device.
    The head columns come from the device/axis metadata, never fixed positions (XROR's
    HEAD.i/j/k/1 are x, y, z, w; HEAD.x/y/z are positions). Column 0 is the time in seconds.
    When user is given, the recording's own user and recording IDs must match."""
    info = data["info"]
    if user is not None and (str(info["user"]["id"]) != user or str(data["$id"]) != recording):
        raise ValueError("Archive IDs do not match recording metadata")
    metadata = {"app": info.get("software", {}).get("app", {}).get("name", "unknown")}
    offset, columns, head = 1, 1, None  # column 0 is the timestamp
    for device in info["hardware"]["devices"]:
        axes = device["axes"]
        if device.get("joint") == "HEAD":
            if head is not None:
                raise ValueError("Multiple HEAD channels")
            head = [offset + axes.index(a) for a in ("i", "j", "k", "1")]
            metadata["headset"] = device.get("name", "unknown")
        offset += len(axes)
        columns += len(axes)
    if head is None:
        return metadata, None, None
    frames = np.asarray(data["frames"], dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] != columns:
        raise ValueError("Unexpected frame shape")
    return metadata, frames[:, 0], frames[:, head]


def clean(t, q, row):
    """Return (t, q, status). status is None when the session continues to the motion filter.
    Each step records its counts in row."""
    # Reject non-finite samples and near-zero quaternions, then scale to unit length.
    norm = np.linalg.norm(q, axis=1)
    valid = np.isfinite(t) & np.isfinite(q).all(axis=1) & (norm > MIN_QUAT_NORM)
    row["raw_samples"] = len(t)
    row["invalid_samples"] = int((~valid).sum())
    t, q = t[valid], q[valid] / norm[valid, None]
    if len(t) < 2:
        return t, q, "too_short"
    row["backwards_steps"] = int((np.diff(t) < 0).sum())
    row["gaps_over_100ms"] = int((np.diff(t) > H5_MAX_GAP_S).sum())

    # Step 1: trim the first 5 s. Elapsed time accumulates positive intervals only,
    # so a later clock reset to zero does not trigger a second startup trim.
    elapsed = np.r_[0., np.cumsum(np.maximum(np.diff(t), 0.))]
    keep = elapsed >= TRIM_S
    row["trimmed_samples"] = int((~keep).sum())
    t, q = t[keep], q[keep]
    # Step 2: a session with fewer than 20 samples after trimming is not assessed.
    if len(t) < MIN_ASSESSED_SAMPLES:
        return t, q, "too_short_after_trim"

    # Keep the first valid sample of adjacent equal timestamps; never sort.
    unique = np.r_[True, np.diff(t) != 0]
    row["duplicate_samples"] = int((~unique).sum())
    t, q = t[unique], q[unique]

    # Step 8: rate from the median positive interval; out-of-band sessions are discarded.
    dt = np.diff(t)
    positive = dt[dt > 0]
    if not len(positive):
        return t, q, "no_positive_intervals"
    row["source_rate_hz"] = float(1 / np.median(positive))
    low, high = RATE_BAND_HZ
    if not low <= row["source_rate_hz"] <= high:
        return t, q, "rate_reject"
    return t, q, None


def motion_filter(t, q):
    """Data page steps 2-7 on the cleaned source samples: median-over-windows sweep and
    velocity per axis, then the max across axes."""
    sweeps, velocities = [], []
    wrap_steps = warning_count = 0
    # Windows only need a monotonic clock, so split at backward jumps alone. Tracking gaps
    # stay inside their window, as in the official procedure.
    cuts = np.r_[0, np.flatnonzero(np.diff(t) <= 0) + 1, len(t)]
    low, high = SWEEP_PERCENTILES
    for begin, end in zip(cuts[:-1], cuts[1:]):
        tt, qq = t[begin:end], q[begin:end]
        if len(tt) < MIN_WINDOW_SAMPLES:
            continue
        # Step 3: Euler angles in degrees, computed per axis. No unwrap.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            angles = Rotation.from_quat(qq).as_euler(EULER_ORDER, degrees=True)
        warning_count += len(caught)
        wrap_steps += int(np.any(np.abs(np.diff(angles, axis=0)) > 180, axis=1).sum())
        # Step 2: non-overlapping 5 s windows on one grid anchored at the trimmed start.
        # A trailing partial window is not assessed.
        first = t[0] + np.ceil((tt[0] - t[0]) / WINDOW_S) * WINDOW_S
        for start in np.arange(first, tt[-1] - WINDOW_S + 1e-9, WINDOW_S):
            keep = (tt >= start) & (tt < start + WINDOW_S)
            times, a = tt[keep], angles[keep]
            if len(times) < MIN_WINDOW_SAMPLES or times[-1] <= times[0]:
                continue
            # Step 4: 10th-90th percentile range.
            sweeps.append(np.percentile(a, high, axis=0) - np.percentile(a, low, axis=0))
            # Step 5: wrapped absolute differences over the window duration (5 s).
            differences = np.abs(np.diff(a, axis=0))
            velocities.append(np.minimum(differences, 360 - differences).sum(axis=0) / WINDOW_S)
    if not sweeps:
        return {"sweep_deg": None, "velocity_deg_s": None, "motion_windows": 0}
    # Step 6: medians across windows, then the maximum across axes.
    s, v = np.median(sweeps, axis=0), np.median(velocities, axis=0)
    return {"sweep_deg": float(s.max()), "velocity_deg_s": float(v.max()),
            "sweep_xyz_deg": s.tolist(), "velocity_xyz_deg_s": v.tolist(),
            "motion_windows": len(sweeps), "euler_wrap_steps": wrap_steps,
            "euler_warning_count": warning_count}


def passes(measures) -> bool:
    """Step 7: strict thresholds, following the detailed algorithm."""
    return (measures["sweep_deg"] is not None and measures["sweep_deg"] > SWEEP_MIN_DEG
            and measures["velocity_deg_s"] > VELOCITY_MIN_DEG_S)


def align(q):
    """Normalise, then negate any sample whose dot product with its predecessor is negative."""
    q = np.asarray(q, dtype=np.float64).copy()
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    flips = np.r_[1, np.where(np.sum(q[1:] * q[:-1], axis=1) < 0, -1, 1)]
    return q * np.cumprod(flips)[:, None]


def resample(t, q, row):
    """Return [(time_s, quaternion_wxyz float32)] for each kept segment. q is xyzw.
    Splits at backward jumps and gaps over H5_MAX_GAP_S; never extrapolates past endpoints."""
    segments = []
    row["segments"] = []
    row["short_segments_dropped"] = 0
    dt = np.diff(t)
    cuts = np.r_[0, np.flatnonzero((dt <= 0) | (dt > H5_MAX_GAP_S)) + 1, len(t)]
    for begin, end in zip(cuts[:-1], cuts[1:]):
        tt, qq = t[begin:end], q[begin:end]
        if len(tt) < 2:
            continue
        grid = tt[0] + np.arange(int(np.floor((tt[-1] - tt[0]) * H5_RATE_HZ)) + 1) / H5_RATE_HZ
        grid = grid[grid <= tt[-1]]
        if len(grid) < H5_MIN_SAMPLES:
            row["short_segments_dropped"] += 1
            continue
        # Align before SLERP so it takes the short path, and after so the output is continuous.
        aligned = align(Slerp(tt, Rotation.from_quat(align(qq)))(grid).as_quat())
        segments.append((grid, aligned[:, [3, 0, 1, 2]].astype(np.float32)))
        row["segments"].append({"samples": len(grid), "start_time_s": float(grid[0]),
                                "end_time_s": float(grid[-1]), "duration_s": float(grid[-1] - grid[0]),
                                "source_start_time_s": float(tt[0]), "source_end_time_s": float(tt[-1])})
    return segments


def filter_recording(data: dict, user: str | None = None, recording: str | None = None):
    """Head -> clean -> motion filter. Returns (row, t, q). row["status"] is None when the
    recording passes the challenge filter (t, q are then the cleaned source samples), otherwise
    it names the rejection. t and q are None when there is no head."""
    metadata, t, q = read_head(data, user, recording)
    row = {"user": user, "recording": recording, **metadata, "preparation_version": PREPARATION_VERSION}
    if t is None:
        return dict(row, status="no_head"), None, None
    t, q, status = clean(t, q, row)
    if status:
        return dict(row, status=status), t, q
    row.update(motion_filter(t, q))
    if not passes(row):
        return dict(row, status="motion_reject"), t, q
    return dict(row, status=None), t, q


def prepare_recording(data: dict, user: str, recording: str):
    """Return (audit row, qualified segments). Segments are empty unless status is qualified."""
    row, t, q = filter_recording(data, user, recording)
    if row["status"]:
        return row, []
    segments = resample(t, q, row)
    return dict(row, status="qualified" if segments else "no_long_segment",
                retained_samples=sum(len(t) for t, _ in segments)), segments
