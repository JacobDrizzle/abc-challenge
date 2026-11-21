"""Menu option 3: step-by-step inspector for one saved recording
(game, timestamp format, keys, frame statistics, filters)."""

import os
import re
from datetime import datetime, timezone

import numpy as np
from xror import XROR

from .cli import ask_float, ask_int
from .config import SAVED_ROOT
from .head import describe_game
from .storage import list_saved
from .timefmt import detect_time_format, print_time_format


def frame_columns(data: dict, width: int) -> list[str]:
    """Column names for the frames array: time, then <device>.<axis> for every device axis."""
    devices = ((data.get("info") or {}).get("hardware") or {}).get("devices") or []
    cols = ["time"]
    seen: dict[str, int] = {}
    for i, dev in enumerate(devices):
        base = str(dev.get("name") or dev.get("type") or f"device{i}")
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            base = f"{base}#{seen[base]}"
        for ax in dev.get("axes", []):
            cols.append(f"{base}.{ax}")
    if len(cols) != width:
        print(f"  (note: device metadata gives {len(cols)} columns but frames have {width}; "
              f"missing ones are named colN)")
        cols = cols[:width] + [f"col{i}" for i in range(len(cols), width)]
    return cols


def print_overview(path: str, data: dict, frames: np.ndarray) -> None:
    info = data.get("info") or {}
    software = info.get("software") or {}

    print("\n" + "=" * 60)
    print(f"{'File':<10}: {path}")
    print(f"{'Game/app':<10}: {describe_game(data)}")
    for label, key in (("Runtime", "runtime"), ("API", "api")):
        if software.get(key):
            print(f"{label:<10}: {software[key]}")
    for label, key in (("Env", "environment"), ("Activity", "activity")):
        sub = software.get(key)
        if isinstance(sub, dict) and sub.get("name"):
            print(f"{label:<10}: {sub['name']}")

    ts = info.get("timestamp")
    if isinstance(ts, (int, float)):
        try:
            when = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"{'Recorded':<10}: {when}  (info.timestamp = {ts}, Unix seconds)")
        except (OverflowError, OSError, ValueError):
            print(f"{'Recorded':<10}: info.timestamp = {ts}")

    devices = (info.get("hardware") or {}).get("devices") or []
    if devices:
        print("Devices   :")
        for dev in devices:
            label = dev.get("name") or dev.get("type") or "?"
            extra = f", {dev['type']}" if dev.get("type") and dev.get("name") else ""
            print(f"  - {label}{extra}: {len(dev.get('axes', []))} axes")

    print(f"{'Events':<10}: {len(data.get('events') or [])} event types")
    if frames.size:
        print(f"{'Frames':<10}: {frames.shape[0]} rows x {frames.shape[1]} columns")
        print()
        print_time_format(detect_time_format(frames[:, 0]))
    else:
        print(f"{'Frames':<10}: none")
    print("=" * 60)


def print_columns(columns: list[str]) -> None:
    print("Columns:")
    for i in range(0, len(columns), 3):
        print("  " + "".join(f"{j:>3} {columns[j]:<20}" for j in range(i, min(i + 3, len(columns)))))


def pick_columns(columns: list[str], prompt: str, default) -> list[int]:
    """Parse '1,4,7', '3-9' or name text like 'Left' (matches every column containing it)."""
    raw = input(prompt).strip()
    if not raw:
        return list(default)

    chosen: list[int] = []
    for tok in (t.strip() for t in raw.split(",")):
        if not tok:
            continue
        if re.fullmatch(r"\d+-\d+", tok):
            a, b = (int(x) for x in tok.split("-"))
            idxs = list(range(a, b + 1))
        elif tok.isdigit():
            idxs = [int(tok)]
        else:
            idxs = [i for i, c in enumerate(columns) if tok.lower() in c.lower()]
            if not idxs:
                print(f"  no column matches '{tok}'")
        for i in idxs:
            if 0 <= i < len(columns) and i not in chosen:
                chosen.append(i)

    if not chosen:
        print("  nothing selected")
    return chosen or list(default)


def print_rows(frames: np.ndarray, columns: list[str], idxs: list[int], rows) -> None:
    print("  " + f"{'row':>7}" + "".join(f"{columns[i][:12]:>14}" for i in idxs))
    for r in rows:
        print("  " + f"{r:>7}" + "".join(f"{frames[r, i]:>14.4f}" for i in idxs))


def frame_viewer(frames: np.ndarray, columns: list[str]) -> None:
    if frames.size == 0:
        print("  no frame data")
        return

    n_rows, n_cols = frames.shape
    print(f"{n_rows} frames x {n_cols} columns")
    print_columns(columns)

    while True:
        print("\n  1) Column statistics")
        print("  2) Show rows")
        print("  3) Filter rows by a value range")
        print("  4) Show column list")
        print("  b) Back")
        cmd = input("  Pick: ").strip().lower()

        if cmd in ("b", "q", ".."):
            return

        if cmd == "1":
            idxs = pick_columns(columns,
                                "  Columns (numbers, 1-3, or name text like 'Left'; Enter = all): ",
                                range(n_cols))
            print(f"\n  {'idx':>3}  {'column':<20}{'min':>12}{'max':>12}{'mean':>12}{'std':>12}{'NaN':>6}")
            for i in idxs:
                col = frames[:, i]
                finite = col[np.isfinite(col)]
                nans = int(len(col) - len(finite))
                if len(finite) == 0:
                    print(f"  {i:>3}  {columns[i]:<20}  no finite values")
                    continue
                print(f"  {i:>3}  {columns[i]:<20}{finite.min():>12.4f}{finite.max():>12.4f}"
                      f"{finite.mean():>12.4f}{finite.std():>12.4f}{nans:>6}")

        elif cmd == "2":
            idxs = pick_columns(columns, "  Columns (Enter = first 8): ", range(min(8, n_cols)))
            start = max(0, ask_int("  Start row [0]: ", 0))
            count = max(1, ask_int("  How many rows [10]: ", 10))
            print()
            print_rows(frames, columns, idxs, range(start, min(start + count, n_rows)))

        elif cmd == "3":
            sel = pick_columns(columns, "  Column to filter on (number or name): ", [])
            if not sel:
                continue
            ci = sel[0]
            lo = ask_float(f"  Min value for {columns[ci]} (blank = no limit): ")
            hi = ask_float(f"  Max value for {columns[ci]} (blank = no limit): ")
            col = frames[:, ci]
            mask = np.isfinite(col)
            if lo is not None:
                mask &= col >= lo
            if hi is not None:
                mask &= col <= hi
            rows = np.flatnonzero(mask)
            print(f"\n  {len(rows)}/{n_rows} frames match ({100.0 * len(rows) / n_rows:.1f}%)")
            if len(rows):
                print(f"  first match at time {frames[rows[0], 0]:.6g}, last at {frames[rows[-1], 0]:.6g}")
                show = pick_columns(columns, "  Columns to show [Enter = time + that column]: ", [0, ci])
                count = max(1, ask_int("  How many rows [10]: ", 10))
                print()
                print_rows(frames, columns, show, rows[:count])

        elif cmd == "4":
            print_columns(columns)


def preview(value, width: int = 70) -> str:
    if isinstance(value, dict):
        return f"dict ({len(value)} keys)"
    if isinstance(value, (list, tuple)):
        if len(value) <= 6 and not any(isinstance(x, (dict, list, tuple)) for x in value):
            text = repr(value)
            return text if len(text) <= width else text[:width - 3] + "..."
        return f"list ({len(value)} items)"
    if isinstance(value, (bytes, bytearray)):
        return f"bytes ({len(value)})"
    text = repr(value)
    return text if len(text) <= width else text[:width - 3] + "..."


def browse_data(data: dict, frames: np.ndarray, columns: list[str]) -> None:
    """Walk the recording's keys step by step. 'frames' opens the frame viewer."""
    stack: list[tuple[str, object]] = [("data", data)]

    while True:
        name, node = stack[-1]
        print("\n=== " + " > ".join(n for n, _ in stack) + " ===")

        if len(stack) == 2 and name == "frames":
            frame_viewer(frames, columns)
            stack.pop()
            continue

        options: dict[str, object] = {}
        if isinstance(node, dict):
            if not node:
                print("  (empty)")
                stack.pop()
                continue
            for i, (k, v) in enumerate(node.items(), 1):
                shown = preview(v)
                if len(stack) == 1 and k == "frames" and frames.size:
                    shown = f"{frames.shape[0]} frames x {frames.shape[1]} columns (opens the frame viewer)"
                print(f"  {i:>3}) {k}: {shown}")
                options[str(i)] = k
                options[str(k).lower()] = k
        elif isinstance(node, (list, tuple)):
            if not node:
                print("  (empty list)")
                stack.pop()
                continue
            print(f"  list with {len(node)} items" + (" (showing first 25)" if len(node) > 25 else ""))
            for i, v in enumerate(node[:25]):
                print(f"  [{i}] {preview(v)}")
            for i in range(len(node)):
                options[str(i)] = i
        else:
            print(f"  {preview(node, 2000)}")
            stack.pop()
            continue

        choice = input("\nOpen key/number  ('..' = back, 'q' = file menu): ").strip()
        if choice.lower() in ("q", "quit"):
            return
        if choice in ("..", "b", "back"):
            if len(stack) == 1:
                return
            stack.pop()
            continue

        key = None
        if choice in options:
            key = options[choice]
        elif choice.lower() in options:
            key = options[choice.lower()]
        else:
            print(f"  '{choice}' not found")
            continue

        label = f"[{key}]" if isinstance(node, (list, tuple)) else str(key)
        stack.append((label, node[key])) #type:ignore


def show_list(view: list[str], limit: int = 20) -> None:
    for i, p in enumerate(view[:limit], 1):
        print(f"  {i:>4}) {os.path.relpath(p, SAVED_ROOT)}")
    if len(view) > limit:
        print(f"  ... {len(view) - limit} more (type some text to narrow the list)")


def choose_saved_file(paths: list[str]) -> str | None:
    print(f"\n{len(paths)} saved recordings.")
    print("Enter = first file, a number, 'l' = list, text = search by path, 'all' = reset, 'q' = quit")
    view = paths
    while True:
        raw = input("File: ").strip()
        low = raw.lower()
        if low in ("q", "quit"):
            return None
        if raw == "":
            return view[0]
        if low == "l":
            show_list(view)
        elif low == "all":
            view = paths
            show_list(view)
        elif raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(view):
                return view[n - 1]
            print(f"  pick a number between 1 and {len(view)}")
        else:
            matches = [p for p in paths if low in p.lower()]
            if not matches:
                print("  no saved file matches that text")
                continue
            view = matches
            print(f"  {len(view)} match")
            show_list(view)


def inspect_saved() -> None:
    """Pick a saved recording, see which game it came from and its timestamp format,
    then explore its keys and frame data step by step."""
    paths = list_saved()
    if not paths:
        print(f"No saved recordings found under '{SAVED_ROOT}/'.")
        return

    while True:
        path = choose_saved_file(paths)
        if path is None:
            return
        try:
            with open(path, "rb") as f:
                data = XROR.unpack(f.read()).data
        except Exception as e:
            print(f"Could not read {path}: {e}")
            continue

        raw_frames = data.get("frames")
        try:
            frames = np.array(raw_frames, dtype=float) if raw_frames is not None else np.empty((0, 0))
        except (ValueError, TypeError):
            frames = np.empty((0, 0))
        if frames.ndim != 2:
            frames = np.empty((0, 0))
        columns = frame_columns(data, frames.shape[1]) if frames.size else []

        print_overview(path, data, frames)
        browse_data(data, frames, columns)
