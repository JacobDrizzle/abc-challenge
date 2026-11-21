"""Menu options 2 and 6: re-check or review tars that are still extracted under users/
(left over from the old extraction flow). Files are found by scanning the folders."""

import contextlib
import io
import os

from xror import XROR

from .cli import human_size
from .config import RESULTS_DB, SAVED_ROOT
from .head import describe_game, game_name
from .processing import LAST_FAIL, process_func
from .results_db import evaluate_replay, log_results
from .state import load_processed, save_processed
from .storage import save_passed, scan_extracted_tars, tar_folder


def process_tar_files(tar_name: str, files: list[str]) -> dict:
    """Legacy tars already extracted under users/: evaluate every file, log every result,
    copy passing files to saved/. Failed files are NOT deleted here."""
    folder = tar_folder(tar_name)
    out = {"passed": [], "failed": [], "errors": 0, "missing": 0, "total": len(files)}
    rows = []

    for name in files:
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            out["missing"] += 1
            continue
        with open(path, "rb") as f:
            raw = f.read()
        row, ok = evaluate_replay(tar_name, name, raw)
        rows.append(row)
        if row["reason"] == "unreadable file":
            out["errors"] += 1
        if ok:
            save_passed(path)
            out["passed"].append(name)
        else:
            out["failed"].append(name)

    log_results(rows)
    return out


def delete_cached_files(results: dict, key: str) -> None:
    """Delete files from users/<shard>/<tar>/. key is 'failed' or 'passed'.
    Passing originals are only deleted if an identical copy already exists under saved/."""
    freed = 0
    count = 0
    for tar_name, r in results.items():
        folder = tar_folder(tar_name)
        for name in r[key]:
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            if key == "passed":
                copy = os.path.join(SAVED_ROOT, path)
                if not (os.path.isfile(copy) and os.path.getsize(copy) == os.path.getsize(path)):
                    continue  # no safe copy, keep the original
            try:
                size = os.path.getsize(path)
                os.remove(path)
            except OSError:
                continue
            freed += size
            count += 1
        try:
            os.rmdir(folder)  # only succeeds when the folder is empty
        except OSError:
            pass
    print(f"Deleted {count} files, freed {human_size(freed)}.")


def evaluate_cached_tars() -> None:
    """Re-run process_func on every recording already extracted under users/, log the
    result of each (pass or fail), copy passing files to saved/, then optionally clean up."""
    cached = sorted(scan_extracted_tars().items())  # folders under users/, not the deprecated manifest.json

    if not cached:
        print("No extracted tars found under users/.")
        return

    print(f"Checking {len(cached)} cached tars...\n")
    total_recordings = 0
    total_passed = 0
    processed = load_processed()
    results: dict[str, dict] = {}

    for tar_name, files in cached:
        r = process_tar_files(tar_name, files)
        results[tar_name] = r
        passed = len(r["passed"])
        processed[tar_name] = {"total": len(files), "passed": passed, "errors": r["errors"]}

        n = len(files)
        pct = 100.0 * passed / n
        notes = (f", {r['errors']} unreadable" if r["errors"] else "") + \
                (f", {r['missing']} missing" if r["missing"] else "")
        print(f"{tar_name}: {passed}/{n} passed ({pct:.1f}%){notes}")

        total_recordings += n
        total_passed += passed

    save_processed(processed)

    overall = 100.0 * total_passed / total_recordings if total_recordings else 0.0
    print("\n" + "=" * 50)
    print(f"OVERALL: {total_passed}/{total_recordings} passed ({overall:.1f}%) "
          f"across {len(cached)} tars")
    print(f"Every result (pass and fail, with game, reason and S/V) is logged in {RESULTS_DB}")
    print("=" * 50)

    # Optional cleanup: failures are now only kept as log rows
    for key, what in (("failed", "failed recordings"),
                      ("passed", "originals of passing recordings (already copied to saved/)")):
        names = [(t, n) for t, r in results.items() for n in r[key]]
        if not names:
            continue
        size = sum(os.path.getsize(os.path.join(tar_folder(t), n)) for t, n in names
                   if os.path.isfile(os.path.join(tar_folder(t), n)))
        ans = input(f"\nDelete {len(names)} {what} from users/ ({human_size(size)})? [y/N]: ")
        if ans.strip().lower() in ("y", "yes"):
            delete_cached_files(results, key)


def review_failed() -> None:
    """Re-check every cached recording and list the ones that do NOT pass, with the game
    each came from, why it failed and how much space it takes. Read-only: nothing is
    copied, moved or deleted."""
    cached = sorted(scan_extracted_tars().items())  # folders under users/, not the deprecated manifest.json
    if not cached:
        print("No extracted tars found under users/.")
        return

    total = 0
    passed = 0
    missing = 0
    failed: list[dict] = []

    print(f"Re-checking {len(cached)} cached tars (read-only)...\n")
    for ti, (tar_name, files) in enumerate(cached, 1):
        folder = tar_folder(tar_name)
        for name in files:
            total += 1
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                missing += 1
                continue
            size = os.path.getsize(path)
            try:
                with open(path, "rb") as f:
                    data = XROR.unpack(f.read()).data
                with contextlib.redirect_stdout(io.StringIO()):
                    result = process_func(data)
            except Exception as e:
                failed.append({"path": path, "game": "unreadable", "name": "unreadable",
                               "category": "unreadable file", "detail": str(e)[:60], "size": size})
                continue

            if result is not None:
                passed += 1
            else:
                category, detail = LAST_FAIL
                failed.append({"path": path, "game": describe_game(data), "name": game_name(data),
                               "category": category, "detail": detail, "size": size})
        if ti % 25 == 0 or ti == len(cached):
            print(f"  ...{ti}/{len(cached)} tars checked, {len(failed)} failed so far")

    if not failed:
        print("\nEvery readable recording passes, nothing to review.")
        return

    freed = sum(x["size"] for x in failed)
    print("\n" + "=" * 60)
    print(f"FAILED: {len(failed)} of {total} recordings "
          f"({100.0 * len(failed) / total:.1f}%), {human_size(freed)} on disk")
    print(f"passed: {passed}" + (f", already missing from disk: {missing}" if missing else ""))

    print("\nBy game:")
    by_game: dict[str, list[int]] = {}
    for x in failed:
        by_game.setdefault(x["name"], []).append(x["size"])
    for g, sizes in sorted(by_game.items(), key=lambda kv: -len(kv[1])):
        print(f"  {g:<30}{len(sizes):>8} files{human_size(sum(sizes)):>12}")

    print("\nBy reason:")
    by_reason: dict[str, list[int]] = {}
    for x in failed:
        by_reason.setdefault(x["category"], []).append(x["size"])
    for r, sizes in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        print(f"  {r:<30}{len(sizes):>8} files{human_size(sum(sizes)):>12}")
    print("=" * 60)

    if input("\nList every failed file with its game? [y/N]: ").strip().lower() in ("y", "yes"):
        for x in failed:
            rel = x["path"]
            extra = f" ({x['detail']})" if x["detail"] else ""
            print(f"FAIL   {rel}  [{x['game']}]\n       {x['category']}{extra}, {human_size(x['size'])}")
