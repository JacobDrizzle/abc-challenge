"""Work out whether a recording's timestamps are seconds, milliseconds, microseconds
or nanoseconds, and whether they are relative or Unix/epoch time."""

from datetime import datetime, timezone

import numpy as np


TIME_UNITS = [("seconds", 1.0), ("milliseconds", 1e-3), ("microseconds", 1e-6), ("nanoseconds", 1e-9)]


TIME_EXAMPLES = {
    "seconds": "0.1 = 100 ms",
    "milliseconds": "100 = 100 ms",
    "microseconds": "100000 = 100 ms",
    "nanoseconds": "100000000 = 100 ms",
}


def detect_time_format(times: np.ndarray) -> dict:
    """Work out the unit (s / ms / us / ns) and whether the times are relative or Unix epoch.
    The unit is the one that makes the median step a plausible 10-1000 Hz sampling rate."""
    t = times[np.isfinite(times)]
    if len(t) < 2:
        return {"ok": False}

    diffs = np.diff(t)
    dt = float(np.median(diffs))
    info = {
        "ok": True,
        "first": float(t[0]),
        "last": float(t[-1]),
        "dt": dt,
        "monotonic": bool(np.all(diffs >= 0)),
        "unit": None,
    }
    if dt > 0:
        for unit, scale in TIME_UNITS:
            hz = 1.0 / (dt * scale)
            if 10.0 <= hz <= 1000.0:
                info.update(unit=unit, scale=scale, hz=hz)
                break

    if info["unit"]:
        first_s = info["first"] * info["scale"]
        info["duration_s"] = (info["last"] - info["first"]) * info["scale"]
        info["kind"] = "epoch" if first_s > 1e8 else "relative"
        if info["kind"] == "epoch":
            try:
                info["date"] = datetime.fromtimestamp(first_s, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S UTC")
            except (OverflowError, OSError, ValueError):
                info["date"] = None
    return info


def print_time_format(info: dict) -> None:
    print("Timestamp format (frame column 0)")
    if not info.get("ok"):
        print("  not enough frames to tell")
        return

    print(f"  raw values : first {info['first']:.6g}, last {info['last']:.6g}, "
          f"median step {info['dt']:.6g}")
    if not info["unit"]:
        print("  unit       : could not tell (the step doesn't give a 10-1000 Hz rate "
              "in s / ms / us / ns)")
        return

    unit = info["unit"]
    print(f"  unit       : {unit}  (so {TIME_EXAMPLES[unit]})")
    print(f"  sampling   : {info['hz']:.1f} Hz (median step = {info['dt'] * info['scale'] * 1000:.2f} ms)")
    if info["kind"] == "epoch":
        print("  type       : absolute Unix / epoch time"
              + (f", starts {info['date']}" if info.get("date") else ""))
    else:
        print("  type       : relative, time elapsed since the recording started")
    print(f"  duration   : {info['duration_s']:.1f} s")
    if not info["monotonic"]:
        print("  warning    : timestamps go backwards in places")
