"""Menu options 1, 4, 8 and 14: download tars, check every recording, keep only the passes."""

import os
import random
import shutil
import sqlite3
import tempfile
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait

from huggingface_hub import hf_hub_download

from .config import (CACHED_DIR, DOWNLOAD_MAX_ERRORS_IN_ROW, PREPARATION_VERSION, REPO, RESERVED_SPLIT_PATH,
                     RESULTS_DB, SAVED_ROOT)
from .h5_export import (current_entries, export_hdf5, load_h5_cache, load_reserved_users, prepare_user,
                        saved_users, user_id_of, users_needed)
from .head import UNUSABLE_REASONS
from .hf import get_tar_names, iter_tar_replays
from .results_db import evaluate_replay, log_results
from .state import load_processed, record_processed
from .storage import scan_extracted_tars, tar_folder


def process_tar(tar_name: str, force: bool = False) -> dict:
    processed = load_processed()
    if tar_name in processed and not force:
        p = processed[tar_name]
        print(f"{tar_name} already processed ({p['passed']}/{p['total']} saved), skipping")
        return p

    # Stagger thread requests slightly to avoid triggering Hugging Face's DDoS protection / connection resets
    time.sleep(random.uniform(0.1, 0.6))

    print(f"{tar_name}: downloading...")
    rows: list[dict] = []
    staged: list[str] = []
    dest_folder = os.path.join(SAVED_ROOT, tar_folder(tar_name))

    os.makedirs(CACHED_DIR, exist_ok=True)
    unique_tmp_dir = tempfile.mkdtemp(dir=CACHED_DIR)
    staging = os.path.join(unique_tmp_dir, "passed")  # passes wait here until the whole tar has been read

    try:
        tar_path = hf_hub_download(REPO, tar_name, repo_type="dataset", cache_dir=unique_tmp_dir)
        for name, raw in iter_tar_replays(tar_path):
            row, ok = evaluate_replay(tar_name, name, raw)
            rows.append(row)
            if ok:
                os.makedirs(staging, exist_ok=True)
                path = os.path.join(staging, name)
                with open(path, "wb") as f:
                    f.write(raw)
                staged.append(path)
        # How many of the passes the HDF5 export would take, using the export's own check
        # (a 100 Hz segment of at least H5_MIN_SAMPLES). It stops counting at H5_MAX_RECORDINGS_PER_USER.
        qualified = len(prepare_user(user_id_of(tar_name), staged)[1]) if staged else 0
        if staged:
            move_into_saved(staging, dest_folder)
    finally:
        shutil.rmtree(unique_tmp_dir, ignore_errors=True)  # after a failure this drops the staged passes too

    log_results(rows)
    passed = len(staged)
    errors = sum(1 for r in rows if r["reason"] == "unreadable file")
    skipped = sum(1 for r in rows if r["reason"] in UNUSABLE_REASONS)
    record = {"total": len(rows), "passed": passed, "qualified": qualified, "errors": errors, "skipped": skipped,
              "preparation_version": PREPARATION_VERSION}  # which filter picked the files in saved/
    record_processed(tar_name, record)

    pct = 100.0 * passed / len(rows) if rows else 0.0
    err_note = f", {errors} unreadable" if errors else ""
    skip_note = f", {skipped} skipped (no head / excluded game)" if skipped else ""
    print(f"{tar_name}: {passed}/{len(rows)} passed and saved ({pct:.1f}%), "
          f"{qualified} usable in the export{err_note}{skip_note}")
    return record


def move_into_saved(staging: str, dest_folder: str, attempts: int = 8) -> None:
    """Move a finished tar's passes into saved/ with one folder rename, so saved/ never holds part of a tar.
    Retried, because on Windows Defender or the indexer briefly locks new files (WinError 5).
    A tar processed again (force=True) already has a folder, so its files are replaced one by one."""
    if os.path.isdir(dest_folder):
        for name in os.listdir(staging):
            os.replace(os.path.join(staging, name), os.path.join(dest_folder, name))
        return
    os.makedirs(os.path.dirname(dest_folder), exist_ok=True)
    for attempt in range(attempts):
        try:
            os.rename(staging, dest_folder)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.1 * 2 ** attempt)


def tars_already_done() -> dict[str, set[str]]:
    """Every tar we already have in some form, grouped by where the evidence is."""
    found: dict[str, set[str]] = {
        "processed.json": set(load_processed()),
        "users/ (extracted)": set(scan_extracted_tars()),  # folders on disk, not the deprecated manifest.json
        "results log": set(),
        "saved/": set(),
    }
    if os.path.isfile(RESULTS_DB):
        con = sqlite3.connect(RESULTS_DB)
        found["results log"] = {r[0] for r in con.execute("SELECT DISTINCT tar FROM results WHERE tar LIKE '%.tar'")}
        con.close()
    # saved/users/<shard>/<tarname>/ holding .xror files  ->  users/<shard>/<tarname>.tar
    for root, _dirs, files in os.walk(SAVED_ROOT):
        if any(f.endswith(".xror") for f in files):
            found["saved/"].add(os.path.relpath(root, SAVED_ROOT).replace(os.sep, "/") + ".tar")
    return found


def pick_new_tars() -> list[str]:
    """All tars we have no trace of (not processed, not in the log, not in saved/, not extracted), shuffled."""
    tars = get_tar_names()
    found = tars_already_done()
    skip = set().union(*found.values())
    candidates = [t for t in tars if t not in skip]
    random.shuffle(candidates)
    where = ", ".join(f"{len(v)} in {k}" for k, v in found.items())
    print(f"{len(candidates)} of {len(tars)} tars are new ({len(skip & set(tars))} already have: {where}).\n")
    return candidates


def is_cached(tar_name: str) -> bool:
    """True if the tar's recordings are already on disk: extracted under users/ or kept under saved/."""
    for base in ("", SAVED_ROOT):
        folder = os.path.join(base, tar_folder(tar_name)) if base else tar_folder(tar_name)
        if os.path.isdir(folder) and any(f.endswith(".xror") for f in os.listdir(folder)):
            return True
    return False


def process_random_tars(count: int) -> None:
    """Pick random tars that are NOT cached or processed yet, download them,
    and check/save every replay inside each one."""
    candidates = pick_new_tars()

    done = 0
    total_files = 0
    total_passed = 0

    for tar_name in candidates:
        if done >= count:
            break
        if is_cached(tar_name):  # a folder exists under users/ or saved/
            continue
        try:
            record = process_tar(tar_name)
        except Exception as e:
            print(f"ERROR on {tar_name}: {e}")
            continue
        done += 1
        total_files += record["total"]
        total_passed += record["passed"]

    print("\n" + "=" * 50)
    print(f"RANDOM: processed {done}/{count} new tars, "
          f"{total_passed}/{total_files} recordings saved")
    if done < count:
        print("Ran out of uncached tars before reaching the requested count.")
    print("=" * 50)


def quiet_hf_warnings() -> None:
    """Hide Hugging Face's symlink warnings, which repeat for every tar when several threads download."""
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    warnings.filterwarnings("ignore", category=UserWarning, message=".*huggingface_hub.*")


def process_random_tars_threaded(count: int, max_workers: int = 4) -> None:
    """Pick random tars and process them concurrently using a ThreadPoolExecutor."""
    quiet_hf_warnings()
    candidates = pick_new_tars()

    targets = candidates[:count]
    if not targets:
        print("No uncached tars found to process.")
        return

    print(f"Processing {len(targets)} tars concurrently using {max_workers} threads...\n")
    
    total_files = 0
    total_passed = 0
    done = 0

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tar = {
                executor.submit(process_tar, tar_name): tar_name 
                for tar_name in targets
            }

            for future in as_completed(future_to_tar):
                tar_name = future_to_tar[future]
                try:
                    record = future.result()
                    done += 1
                    total_files += record["total"]
                    total_passed += record["passed"]
                    print(f"[{done}/{len(targets)}] Finished {tar_name}: {record['passed']}/{record['total']} passed")
                except Exception as exc:
                    print(f"[Error] Tar {tar_name} generated an exception: {exc}")
    except Exception as e:
        print(f"[Fatal ThreadPool Error]: {e}")

    print("\n" + "=" * 50)
    print(f"THREADED BATCH: processed {done}/{len(targets)} tars, "
          f"{total_passed}/{total_files} recordings saved")
    print("=" * 50)


def usable_saved_users(reserved: set[str]) -> tuple[int, int]:
    """(other users, reserved users) in saved/ that the HDF5 export can still use: not exported under
    this preparation version yet, and with at least one recording the export would take."""
    done, vanished = current_entries(load_h5_cache())
    exported = set(done) - set(vanished)
    processed = load_processed()
    found = {False: 0, True: 0}
    for tar_name, paths in saved_users().items():
        if tar_name in exported:
            continue
        record = processed.get(tar_name, {})
        if record.get("preparation_version") == PREPARATION_VERSION and "qualified" in record:
            qualified = record["qualified"]
        else:  # saved by an older version, before process_tar counted this: check the files themselves
            qualified = len(prepare_user(user_id_of(tar_name), paths)[1])
        if qualified:
            found[user_id_of(tar_name) in reserved] += 1
    return found[False], found[True]


def download_users(counts: tuple[int, int, int], max_workers: int = 4, export: bool = True) -> str | None:
    """Option 14 and `python main.py download`: download random new tars until saved/ holds enough users
    to fill counts = (train, validation, test), then run the HDF5 export with those counts.
    A user counts once process_tar finds a recording the export would take. Group B's test users only
    count towards test. Returns the new .h5 path, or None."""
    if min(counts) < 1:
        print("Every split needs at least one user, so the file is a complete train / validation / test set.")
        return None
    try:
        reserved = load_reserved_users(RESERVED_SPLIT_PATH)
    except (OSError, ValueError, KeyError) as e:
        print(f"Can't read Group B's reserved test users from {RESERVED_SPLIT_PATH} ({e}).\n"
              f"Nothing downloaded: the export refuses to run without that list.")
        return None
    max_workers = max(1, max_workers)

    other, res = usable_saved_users(reserved)
    need = users_needed(counts, other, res)
    print(f"Asked for {counts[0]} train + {counts[1]} validation + {counts[2]} test users. saved/ has "
          f"{other + res} usable users that aren't exported yet, so {need} more are needed.\n")

    tried = errors_in_row = 0
    interrupted = False
    if need:
        quiet_hf_warnings()
        candidates = iter(pick_new_tars())
        running: dict = {}  # future -> tar name
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            try:
                while True:
                    # Keep every thread busy until enough users are in. Downloads still running then are
                    # finished, so a run can end with up to (threads - 1) more tars than it needed.
                    while need and errors_in_row < DOWNLOAD_MAX_ERRORS_IN_ROW and len(running) < max_workers:
                        tar_name = next(candidates, None)
                        if tar_name is None:
                            break
                        running[pool.submit(process_tar, tar_name)] = tar_name
                    if not running:
                        break
                    # The timeout lets Ctrl+C through on Windows, where an endless wait would block it.
                    finished, _ = wait(running, timeout=1, return_when=FIRST_COMPLETED)
                    for future in finished:
                        tar_name = running.pop(future)
                        tried += 1
                        try:
                            record = future.result()
                        except Exception as e:
                            errors_in_row += 1
                            print(f"[Error] {tar_name}: {e}")
                            continue
                        errors_in_row = 0
                        if record.get("qualified"):
                            if user_id_of(tar_name) in reserved:
                                res += 1
                            else:
                                other += 1
                            need = users_needed(counts, other, res)
                        print(f"  [{tried} tars checked, {other + res} usable users, {need} more needed]")
            except KeyboardInterrupt:
                interrupted = True
                print(f"\nStopping. Letting the {len(running)} downloads in progress finish...")

    if interrupted or need:
        if interrupted:
            why = "stopped with Ctrl+C"
        elif errors_in_row >= DOWNLOAD_MAX_ERRORS_IN_ROW:
            why = f"{errors_in_row} tars failed in a row (Hugging Face unreachable or rate limiting?)"
        else:
            why = "ran out of new tars"
        print(f"\nStopped after {tried} tars: {why}. Nothing exported. Everything downloaded stays in saved/, "
              f"so running this again carries on from there.")
        return None

    print(f"\n{other + res} usable users after checking {tried} new tars.")
    if not export:
        print(f"Export them with:  python main.py export --train {counts[0]} --val {counts[1]} --test {counts[2]}")
        return None
    return export_hdf5(counts=counts)
