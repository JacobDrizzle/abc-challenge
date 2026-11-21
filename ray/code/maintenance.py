"""Menu options 9-12: re-check saved/, restore missing files, repair the results log,
remove unusable recordings."""

import os
import shutil
import sqlite3
import tempfile

from huggingface_hub import hf_hub_download

from .cli import human_size
from .config import CACHED_DIR, EXCLUDED_GAMES, REPO, RESULTS_DB, SAVED_ROOT
from .head import peek_info, unusable_reason
from .hf import get_tar_names, iter_tar_replays
from .pipeline import process_tar
from .results_db import METRIC_KEYS, RESULT_COLUMNS, evaluate_replay, log_results, make_result_row, make_skip_row, open_results_db
from .state import PROCESSED_LOCK, load_processed, save_processed
from .storage import list_saved, saved_path_to_key, tar_folder


def evaluate_saved_files() -> None:
    """Re-run process_func on every recording in saved/ (one at a time, so the numbers are
    clean) and log the result under the SAME (tar, file) key process_tar uses."""
    paths = list_saved()
    if not paths:
        print(f"No saved recordings found under '{SAVED_ROOT}/'.")
        return

    by_base = {os.path.basename(t): t for t in get_tar_names()}
    print(f"Re-checking {len(paths)} files under '{SAVED_ROOT}/'...\n")
    ok = 0
    errors = 0
    rows = []

    for path in paths:
        tar_name, filename = saved_path_to_key(path, by_base)
        try:
            with open(path, "rb") as f:
                raw = f.read()
        except OSError as e:
            rows.append(make_result_row(tar_name, filename, None, 0, None, str(e)))
            errors += 1
            continue

        row, passed = evaluate_replay(tar_name, filename, raw)
        rows.append(row)
        if row["reason"] == "unreadable file":
            errors += 1
        elif passed:
            ok += 1

    log_results(rows)
    still_bad = len(paths) - ok - errors
    print("\n" + "=" * 50)
    print(f"SAVED FOLDERS RE-CHECK: {ok}/{len(paths)} still pass"
          + (f", {still_bad} no longer pass (kept on disk, logged as failed)" if still_bad else "")
          + (f", {errors} unreadable" if errors else ""))
    print(f"Results logged to {RESULTS_DB} (existing rows were replaced, not duplicated)")
    print("=" * 50)


def restore_missing_saved_files() -> None:
    """Find passing recordings whose file is missing from saved/, re-download their tars,
    re-check each one, write it only if it still passes, and update its log row."""
    if not os.path.isfile(RESULTS_DB):
        print(f"Results database not found at {RESULTS_DB}.")
        return

    con = sqlite3.connect(RESULTS_DB)
    passed_rows = con.execute("SELECT tar, file FROM results WHERE passed = 1 AND tar LIKE '%.tar'").fetchall()
    bare = con.execute("SELECT COUNT(*) FROM results WHERE tar NOT LIKE '%.tar'").fetchone()[0]
    con.close()
    if bare:
        print(f"Note: {bare} log rows use the old bare-tar key. Run option 11 first to clean them up.\n")

    # Same layout process_tar writes: saved/users/<shard>/<tarname>/<file>.xror
    missing_by_tar: dict[str, list[str]] = {}
    for tar_name, filename in passed_rows:
        if not os.path.isfile(os.path.join(SAVED_ROOT, tar_folder(tar_name), filename)):
            missing_by_tar.setdefault(tar_name, []).append(filename)

    total_missing = sum(len(v) for v in missing_by_tar.values())
    if total_missing == 0:
        print("All passing database records already have files on disk in saved/!")
        return

    print(f"Found {total_missing} passing files missing from disk across {len(missing_by_tar)} tars.\n")
    os.makedirs(CACHED_DIR, exist_ok=True)
    restored = 0
    no_longer = 0

    for tar_name, missing_files in missing_by_tar.items():
        print(f"Checking and restoring files from {tar_name}...")
        target_set = set(missing_files)
        dest_folder = os.path.join(SAVED_ROOT, tar_folder(tar_name))
        tmp_dir = tempfile.mkdtemp(dir=CACHED_DIR)  # one temp dir per tar, deleted right after
        rows = []
        try:
            downloaded = hf_hub_download(REPO, tar_name, repo_type="dataset", cache_dir=tmp_dir)
            for name, raw in iter_tar_replays(downloaded):
                if name not in target_set:
                    continue
                row, passed = evaluate_replay(tar_name, name, raw)
                rows.append(row)  # keep the log truthful either way
                if passed:
                    os.makedirs(dest_folder, exist_ok=True)
                    with open(os.path.join(dest_folder, name), "wb") as out_f:
                        out_f.write(raw)
                    restored += 1
                    print(f"  [Pass & Restored] {name}")
                else:
                    no_longer += 1
                    print(f"  [Failed Spec - Skipped] {name}")
                target_set.discard(name)
                if not target_set:
                    break
            if target_set:
                print(f"  [Warning] {len(target_set)} files not found inside the tar")
        except Exception as e:
            print(f"  [Error] Failed processing tar {tar_name}: {e}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        log_results(rows)

    print("\n" + "=" * 50)
    print(f"RESTORATION COMPLETE: restored {restored} files to '{SAVED_ROOT}/'"
          + (f", {no_longer} no longer pass (logged as failed)" if no_longer else ""))
    print("=" * 50)


BAD_PASSED_SQL = """passed = 1 AND (
    hz IS NULL OR duration_s IS NULL OR n_frames IS NULL OR n_windows IS NULL OR time_unit IS NULL
    OR pitch_S IS NULL OR pitch_V IS NULL OR yaw_S IS NULL OR yaw_V IS NULL
    OR roll_S IS NULL OR roll_V IS NULL OR max_S IS NULL OR max_V IS NULL
    OR hz < 85 OR hz > 110 OR max_S <= 17 OR max_V <= 15)"""


SUSPECT_FAILED_SQL = """passed = 0 AND (
    (reason = 'below S/V threshold' AND (hz IS NULL OR max_S IS NULL OR max_V IS NULL
        OR (max_S > 17 AND max_V > 15)))
 OR (reason LIKE 'sampling rate%' AND (hz IS NULL OR hz BETWEEN 85 AND 110 OR max_S IS NOT NULL))
 OR (reason = 'no usable 5 s windows' AND (hz IS NULL OR n_windows IS NULL OR n_windows <> 0
        OR max_S IS NOT NULL))
 OR (reason = 'too few valid frames' AND (n_frames IS NULL OR n_frames >= 20 OR hz IS NOT NULL))
 OR (reason IN ('time unit unknown', 'bad frame array', 'no frames key')
        AND (hz IS NOT NULL OR max_S IS NOT NULL)))"""


def repair_results_log() -> None:
    """Option 11: clean up cache/results.sqlite.
    1) rows logged under a bare tar id (old option 9) are moved to the real
       users/<shard>/<tar>.tar key, replacing the duplicate row with the clean re-check,
    2) processed.json is synced with the log (it can lose entries when threads race),
    3) the log is audited against the spec and against what is really in saved/,
    4) tars with contradictory failed rows can be re-downloaded and re-checked."""
    if not os.path.isfile(RESULTS_DB):
        print(f"Results database not found at {RESULTS_DB}.")
        return

    by_base = {os.path.basename(t): t for t in get_tar_names()}
    names = [c for c, _ in RESULT_COLUMNS]
    insert_sql = (f"INSERT OR REPLACE INTO results ({', '.join(names)}) "
                  f"VALUES ({', '.join('?' for _ in names)})")
    con = open_results_db()
    con.row_factory = sqlite3.Row

    n0, p0 = con.execute("SELECT COUNT(*), COALESCE(SUM(passed), 0) FROM results").fetchone()
    print(f"Before: {n0} rows, {p0} marked passed\n")

    # 1) bare-key rows -> real key
    moved = unmapped = 0
    with con:
        for r in con.execute("SELECT * FROM results WHERE tar NOT LIKE '%.tar'").fetchall():
            proper = by_base.get(r["tar"] + ".tar")
            if proper is None:
                unmapped += 1
                continue
            con.execute(insert_sql, [proper if c == "tar" else r[c] for c in names])
            con.execute("DELETE FROM results WHERE tar = ? AND file = ?", (r["tar"], r["file"]))
            moved += 1
    print(f"1) Moved {moved} bare-key rows onto their real tar key"
          + (f" ({unmapped} could not be matched to a tar and were left alone)" if unmapped else ""))

    # 2) processed.json vs the log
    with PROCESSED_LOCK:
        processed = load_processed()
        added = changed = 0
        for r in con.execute(
                "SELECT tar, COUNT(*) n, COALESCE(SUM(passed), 0) p, "
                "COALESCE(SUM(reason = 'unreadable file'), 0) e, "
                "COALESCE(SUM(passed = 0), 0) f FROM results GROUP BY tar"):
            rec = {"total": r["n"], "passed": int(r["p"]), "errors": int(r["e"])}
            if r["tar"] not in processed:
                if r["f"] > 0:  # a fully checked tar nearly always has failures; skip stray rows
                    processed[r["tar"]] = rec
                    added += 1
            elif (processed[r["tar"]].get("total"), processed[r["tar"]].get("passed")) != (rec["total"], rec["passed"]):
                processed[r["tar"]] = rec
                changed += 1
        save_processed(processed)
    print(f"2) processed.json: {added} tars added back, {changed} counts corrected")

    # 3) audit
    bad_passed = con.execute(f"SELECT tar, file FROM results WHERE {BAD_PASSED_SQL}").fetchall()
    suspect = con.execute(f"SELECT tar, file, reason FROM results WHERE {SUSPECT_FAILED_SQL}").fetchall()
    passed_keys = {(r["tar"], r["file"]) for r in con.execute("SELECT tar, file FROM results WHERE passed = 1")}
    missing = [k for k in passed_keys
               if not os.path.isfile(os.path.join(SAVED_ROOT, tar_folder(k[0]), k[1]))]
    on_disk = {saved_path_to_key(p, by_base) for p in list_saved()}
    orphans = on_disk - passed_keys

    print("\n3) Audit")
    print(f"   passing rows with missing or out-of-spec numbers : {len(bad_passed)}"
          + ("   <- run option 9 (file on disk) or 10 (file missing)" if bad_passed else ""))
    print(f"   passing rows whose file is not in saved/         : {len(missing)}"
          + ("   <- run option 10" if missing else ""))
    print(f"   files in saved/ with no passing row              : {len(orphans)}"
          + ("   <- run option 9" if orphans else ""))
    print(f"   failed rows with contradictory reason/numbers    : {len(suspect)}")

    n1, p1 = con.execute("SELECT COUNT(*), COALESCE(SUM(passed), 0) FROM results").fetchone()
    print(f"\nAfter: {n1} rows, {p1} marked passed, {len(on_disk)} files in saved/")

    # 4) re-check tars with contradictory failed rows
    bad_tars = sorted({r["tar"] for r in suspect})
    con.close()
    if bad_tars:
        print(f"\nThe {len(suspect)} contradictory rows come from {len(bad_tars)} tars. Their pass/fail "
              f"verdicts are fine, but the logged reason and numbers can't be trusted.")
        ans = input(f"Re-download and re-check those {len(bad_tars)} tars now? [y/N]: ").strip().lower()
        if ans in ("y", "yes"):
            for i, tar_name in enumerate(bad_tars, 1):
                print(f"[{i}/{len(bad_tars)}]", end=" ")
                try:
                    process_tar(tar_name, force=True)
                except Exception as e:
                    print(f"ERROR on {tar_name}: {e}")
            print("\nRun option 11 again to confirm everything is clean.")


def purge_unusable_saved() -> None:
    """Option 12: delete recordings in saved/ that have no head device or belong to an
    excluded game (Tilt Brush), and relabel their log rows so they're never counted as passing."""
    by_base = {os.path.basename(t): t for t in get_tar_names()}
    paths = list_saved()
    bad = []
    unreadable = 0
    for p in paths:
        try:
            with open(p, "rb") as f:
                info = peek_info(f.read())
        except OSError:
            info = None
        if info is None:
            unreadable += 1
            continue
        why = unusable_reason(info)
        if why:
            tar_name, fname = saved_path_to_key(p, by_base)
            bad.append({"path": p, "tar": tar_name, "file": fname, "info": info,
                        "why": why, "size": os.path.getsize(p)})

    stale = 0
    placeholders = ", ".join("?" for _ in EXCLUDED_GAMES)
    if os.path.isfile(RESULTS_DB) and EXCLUDED_GAMES:
        con = sqlite3.connect(RESULTS_DB)
        stale = con.execute(
            f"SELECT COUNT(*) FROM results WHERE lower(game) IN ({placeholders}) AND reason <> 'excluded game'",
            tuple(EXCLUDED_GAMES)).fetchone()[0]
        con.close()

    print(f"Checked {len(paths)} files in '{SAVED_ROOT}/'"
          + (f" ({unreadable} unreadable, left alone)" if unreadable else ""))
    if not bad and not stale:
        print("Nothing to remove: every saved recording has a head device and none are from an excluded game.")
        return

    by_kind: dict[str, list[int]] = {}
    for b in bad:
        game = (b["info"].get("software") or {}).get("app", {}).get("name") or "unknown"
        by_kind.setdefault(f"{game}: {b['why'][0]}", []).append(b["size"])
    for kind, sizes in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
        print(f"  {kind:<45}{len(sizes):>6} files{human_size(sum(sizes)):>12}")
    print(f"  log rows from excluded games still showing a failure reason or numbers: {stale}")

    ans = input(f"\nDelete {len(bad)} files ({human_size(sum(b['size'] for b in bad))}) and relabel "
                f"the log rows? [y/N]: ").strip().lower()
    if ans not in ("y", "yes"):
        print("Nothing changed.")
        return

    rows = []
    per_tar: dict[str, int] = {}
    for b in bad:
        try:
            os.remove(b["path"])
        except OSError as e:
            print(f"  could not delete {b['path']}: {e}")
            continue
        rows.append(make_skip_row(b["tar"], b["file"], b["info"], b["size"], *b["why"]))
        per_tar[b["tar"]] = per_tar.get(b["tar"], 0) + 1
        folder = os.path.dirname(b["path"])
        for _ in range(2):  # remove the tar folder, then its shard folder, if they are now empty
            try:
                os.rmdir(folder)
            except OSError:
                break
            folder = os.path.dirname(folder)
    log_results(rows)

    if os.path.isfile(RESULTS_DB) and EXCLUDED_GAMES:
        wipe = ", ".join(f"{k} = NULL" for k in METRIC_KEYS)
        con = open_results_db()
        with con:
            con.execute(
                f"UPDATE results SET passed = 0, reason = 'excluded game', detail = game, {wipe} "
                f"WHERE lower(game) IN ({placeholders})", tuple(EXCLUDED_GAMES))
        con.close()

    # recount the affected tars from the log instead of subtracting, so the numbers can't drift
    with PROCESSED_LOCK:
        processed = load_processed()
        con = sqlite3.connect(RESULTS_DB)
        for tar_name in per_tar:
            rec = processed.get(tar_name)
            if rec:
                p, s = con.execute(
                    "SELECT COALESCE(SUM(passed), 0), "
                    "COALESCE(SUM(reason IN ('excluded game', 'no head device', 'head has no rotation')), 0) "
                    "FROM results WHERE tar = ?", (tar_name,)).fetchone()
                rec["passed"], rec["skipped"] = int(p), int(s)
        con.close()
        save_processed(processed)
    print(f"Deleted {len(rows)} files. Log rows updated.")
