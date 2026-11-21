"""Read-only reports: startup status (indexes), saved-folder summary (option 5), results log (option 7)."""

import contextlib
import csv
import io
import os
import sqlite3

from xror import XROR

from .config import CACHED_DIR, PROCESSED_PATH, RESULTS_DB, SAVED_ROOT
from .head import describe_game
from .hf import get_tar_names
from .legacy import MANIFEST_PATH
from .processing import process_func
from .results_db import RESULT_COLUMNS
from .state import load_processed
from .storage import list_saved, scan_extracted_tars


def check_saved_indexes() -> None:
    """Print which tar indexes are already done: tars still extracted under users/
    (found by scanning the folders) and tars processed by process_tar
    (processed.json + results.sqlite)."""
    tars = get_tar_names()
    index_of = {t: i for i, t in enumerate(tars)}
    extracted = scan_extracted_tars()
    processed = load_processed()

    print("=" * 50)
    print("STATUS: Checking locally saved tar indexes...")
    if os.path.isfile(MANIFEST_PATH):
        print(f"Note: {MANIFEST_PATH} is DEPRECATED and ignored. The users/ folders are scanned instead,"
              f" so the file can be deleted.")

    # 1) tars whose replays are still extracted under users/<shard>/<tar>/
    for tar_name, files in sorted(extracted.items(), key=lambda kv: index_of.get(kv[0], -1)):
        print(f"  [Index {index_of.get(tar_name, '?')}] -> EXTRACTED ({tar_name}, {len(files)} files left)")
    print(f"Tars still extracted under users/: {len(extracted)}")

    # 2) tars that were checked and logged (includes random tars)
    rows = sorted((index_of.get(t, -1), t, rec) for t, rec in processed.items())
    print(f"\nProcessed and logged tars: {len(rows)}/{len(tars)}")
    limit = 30
    for idx, tar_name, rec in rows[:limit]:
        label = idx if idx >= 0 else "?"
        print(f"  [Index {label}] -> DONE ({tar_name}, {rec['passed']}/{rec['total']} passed)")
    if len(rows) > limit:
        print(f"  ... and {len(rows) - limit} more (all listed in {PROCESSED_PATH})")
    print("=" * 50 + "\n")


def read_saved() -> None:
    """Read every file in saved/ and print its per-axis S/V results.
    Nothing is copied or written (src_path is not passed)."""
    paths = list_saved()
    if not paths:
        print(f"No saved recordings found under '{SAVED_ROOT}/'.")
        return

    print(f"Reading {len(paths)} saved recordings from '{SAVED_ROOT}/'...\n")
    ok = 0
    errors = 0

    for path in paths:
        try:
            with open(path, "rb") as f:
                data = XROR.unpack(f.read()).data
            with contextlib.redirect_stdout(io.StringIO()):
                result = process_func(data)
        except Exception as e:
            print(f"ERROR  {path}: {e}")
            errors += 1
            continue

        rel = os.path.relpath(path, SAVED_ROOT)
        if result is None:
            print(f"FAIL   {rel} (no longer passes the thresholds)")
            continue

        ok += 1
        per_axis = "  ".join(
            f"{axis}: S={m['S']:.2f}° V={m['V']:.2f}°/s" for axis, m in result.items()
        )
        print(f"PASS   {rel}  [{describe_game(data)}]\n       {per_axis}")

    print("\n" + "=" * 50)
    print(f"SAVED: {ok}/{len(paths)} still pass" + (f", {errors} unreadable" if errors else ""))
    print("=" * 50)


def show_results_log() -> None:
    """Summarise cache/results.sqlite (every recording, pass or fail) and optionally export CSV."""
    if not os.path.isfile(RESULTS_DB):
        print(f"No results logged yet ({RESULTS_DB} doesn't exist).")
        return

    con = sqlite3.connect(RESULTS_DB)
    total, passed = con.execute("SELECT COUNT(*), COALESCE(SUM(passed), 0) FROM results").fetchone()
    if not total:
        print("The results log is empty.")
        con.close()
        return

    print("\n" + "=" * 60)
    print(f"LOGGED: {total} recordings, {passed} passed, {total - passed} failed")

    print("\nBy game (passed / total):")
    for game, n, p in con.execute(
            "SELECT game, COUNT(*), COALESCE(SUM(passed), 0) FROM results "
            "GROUP BY game ORDER BY COUNT(*) DESC"):
        print(f"  {game:<30}{p:>8} / {n}")

    print("\nWhy recordings failed (avg of the max sweep S / velocity V where measured):")
    for reason, n, s, v in con.execute(
            "SELECT reason, COUNT(*), AVG(max_S), AVG(max_V) FROM results "
            "WHERE passed = 0 GROUP BY reason ORDER BY COUNT(*) DESC"):
        extra = f"   avg S={s:.1f}  V={v:.1f}" if s is not None and v is not None else ""
        print(f"  {reason:<34}{n:>8}{extra}")
    print("=" * 60)

    ans = input("\nExport to CSV?  [a] all  [f] failed only  [N] no: ").strip().lower()
    if ans in ("a", "f"):
        where = " WHERE passed = 0" if ans == "f" else ""
        path = os.path.join(CACHED_DIR, "results_failed.csv" if ans == "f" else "results.csv")
        names = [c for c, _ in RESULT_COLUMNS]
        cur = con.execute(f"SELECT {', '.join(names)} FROM results{where}")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(names)
            w.writerows(cur)
        print(f"Wrote {path}")
    con.close()
