"""The filter itself: process_func measures head sweep (S) and velocity (V).

The rules live in cleaning.py, which the HDF5 export uses too. process_func adds the bookkeeping
for the download pipeline: it reports its numbers through the module globals LAST_METRICS /
LAST_FAIL. Those are shared between threads, so EVAL_LOCK must be held from the process_func call
until the numbers have been read (results_db.evaluate_replay does exactly that)."""

import threading

import numpy as np

from .cleaning import filter_recording
from .head import unusable_reason
from .storage import save_passed
from .timefmt import detect_time_format


TIME_UNIT_COUNTS: dict[str, int] = {}  # passes per non-second time format (filled by process_func)


EVAL_LOCK = threading.Lock()        # guards process_func's LAST_METRICS / LAST_FAIL globals


LAST_METRICS: dict = {}  # numbers measured during the last process_func call


LAST_FAIL: list[str] = ["", ""]  # [category, detail] of the last rejected recording.
# It is a list that is changed in place (never re-assigned) so that other modules that
# imported it by name always see the current value.


def _fail(category: str, detail: str = ""):
    """Remember why process_func rejected a recording, then return None."""
    LAST_FAIL[:] = [category, detail]
    return None


def process_func(data: dict, src_path: str | None = None, quiet: bool = False):
    """
    Runs the challenge filter (cleaning.filter_recording): 5 s trim, 85-110 Hz rate gate,
    per-axis sweep and velocity over non-overlapping 5 s windows, the median over windows,
    the max across axes, and keeps the session only if S > 17 deg and V > 15 deg/s.
    The time column is read as seconds, like Group B's reference.
    On PASS, copies the source file to saved/<src_path> (if src_path is given).
    Returns the per-axis results if the session passes, otherwise None.
    """
    LAST_METRICS.clear()
    if "frames" not in data:
        if not quiet:
            print("Error: 'frames' key not found in data.")
        return _fail("no frames key")

    # Head check: skip recordings without a head device (e.g. Tilt Brush only records the brush)
    # and excluded games from the metadata alone.
    info = data.get("info")
    if not info:
        return _fail("no head device", "no info block")
    why = unusable_reason(info)
    if why:
        return _fail(*why)

    try:
        row, t, _q = filter_recording(data)
    except (KeyError, TypeError, ValueError) as e:
        return _fail("bad frame array", str(e)[:100])

    # Logged for information only (a millisecond clock shows up here and fails the rate gate).
    fmt = detect_time_format(np.array([frame[0] for frame in data["frames"]], dtype=float))
    label = (fmt["unit"] + (" (epoch)" if fmt["kind"] == "epoch" else "")) if fmt.get("unit") else "unknown"
    LAST_METRICS["time_unit"] = label

    status = row["status"]
    if status == "no_head":
        return _fail("no head device", "no device with joint HEAD")
    LAST_METRICS["n_frames"] = int(len(t))
    if status in ("too_short", "too_short_after_trim"):
        return _fail("too few valid frames", status)
    if status == "no_positive_intervals":
        return _fail("no positive time steps")

    hz = row["source_rate_hz"]
    LAST_METRICS["hz"] = hz
    LAST_METRICS["duration_s"] = float(np.maximum(np.diff(t), 0).sum())
    if status == "rate_reject":
        return _fail("sampling rate outside 85-110 Hz", f"{hz:.1f} Hz")

    LAST_METRICS["n_windows"] = row["motion_windows"]
    if not row["motion_windows"]:
        return _fail("no usable 5 s windows")

    session_results = {
        axis_name: {"S": row["sweep_xyz_deg"][i], "V": row["velocity_xyz_deg_s"][i]}
        for i, axis_name in enumerate(["Pitch", "Yaw", "Roll"])  # x, y, z
    }
    max_S, max_V = row["sweep_deg"], row["velocity_deg_s"]
    for axis_name, m in session_results.items():
        LAST_METRICS[f"{axis_name.lower()}_S"] = m["S"]
        LAST_METRICS[f"{axis_name.lower()}_V"] = m["V"]
    LAST_METRICS["max_S"] = max_S
    LAST_METRICS["max_V"] = max_V

    if status == "motion_reject":
        return _fail("below S/V threshold", f"S={max_S:.1f}, V={max_V:.1f}")
    if label != "seconds":
        TIME_UNIT_COUNTS[label] = TIME_UNIT_COUNTS.get(label, 0) + 1
    if not quiet:
        print(f"PASS: S={max_S:.2f}°, V={max_V:.2f}°/s" + ("" if label == "seconds" else f"  [time: {label}]"))
    if src_path:
        dest = save_passed(src_path)
        if not quiet:
            print(f"  saved -> {dest}")
    return session_results
