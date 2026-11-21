"""Entry point. Run:  python main.py   (the menu)

Or without the menu. The counts are users per split, and one user is one tar:
  python main.py download --train 100 --val 15 --test 20 [--threads 8] [--no-export]
      download new users until every split can be filled, then export them to one HDF5 file
  python main.py export --train 100 --val 15 --test 20
      export the users already in saved/ (option 13 without the questions)
  python main.py merge h5/manifest_<run A>.json h5/manifest_<run B>.json [...]
      join exports into one file, each user staying in its own split (code/h5_merge.py)

All data (cache/, saved/, h5/) goes into <project root>/data, wherever this is run from.

All the real work lives in the code/ package, see code/__init__.py for a map of the files.
Note: cache/manifest.json is DEPRECATED and no longer used (see code/legacy.py).
"""
import argparse
import os

from code.cli import ask_int
from code.config import DATA_DIR, SAVED_ROOT, TAR_INDEX
from code.extracted import evaluate_cached_tars, review_failed
from code.h5_export import ask_split_counts, export_hdf5
from code.h5_merge import merge_exports
from code.hf import get_tar_names
from code.maintenance import (evaluate_saved_files, purge_unusable_saved, repair_results_log,
                               restore_missing_saved_files)
from code.pipeline import download_users, process_random_tars, process_random_tars_threaded, process_tar
from code.reports import check_saved_indexes, read_saved, show_results_log
from code.viewer import inspect_saved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BOXRR-23 head-motion pipeline. With no command it shows the menu.")
    commands = parser.add_subparsers(dest="command")
    download_cmd = commands.add_parser("download", help="download new users until every split can be filled, then export")
    export_cmd = commands.add_parser("export", help="export the users already in saved/ to one HDF5 file")
    for cmd in (download_cmd, export_cmd):
        cmd.add_argument("--train", type=int, required=True, help="training users")
        cmd.add_argument("--val", type=int, required=True, help="validation users")
        cmd.add_argument("--test", type=int, required=True, help="test users")
    download_cmd.add_argument("--threads", type=int, default=4, help="tars downloaded at the same time (default 4)")
    download_cmd.add_argument("--no-export", action="store_true", help="only download, run the export command later")
    merge_cmd = commands.add_parser("merge", help="join exports into one file, each user staying in its own split")
    merge_cmd.add_argument("manifests", nargs="+", help="the exports' manifest_<run_id>.json files, under data/")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "download":
        download_users((args.train, args.val, args.test), max_workers=args.threads, export=not args.no_export)
    elif args.command == "export":
        export_hdf5(counts=(args.train, args.val, args.test))
    elif args.command == "merge":
        merge_exports(args.manifests)
    else:
        menu()


def menu() -> None:
    check_saved_indexes()

    print("What do you want to do?")
    print(f"  1) Process tar index {TAR_INDEX} (skipped if already processed; only passes are stored)")
    print("  2) Re-check tars already extracted under users/ (logs results, optional cleanup)")
    print("  3) Inspect a saved recording step by step (game, timestamp format, keys, frames)")
    print("  4) Process random new (uncached) tars")
    print(f"  5) Summary of all recordings in '{SAVED_ROOT}/' (pass/fail + S/V per file)")
    print("  6) Review recordings that FAILED (game, reason, disk space) before deleting anything")
    print("  7) Results log: every recording with game / reason / S / V (summary + CSV export)")
    print("  8) Process random new tars concurrently (Multithreaded)")
    print("  9) Re-check and re-log all files in saved/ folder")
    print("  10) Restore missing passing files from Hugging Face into saved/")
    print("  11) Audit and repair the results log (duplicates, NULL metrics, processed.json)")
    print("  12) Remove unusable recordings from saved/ (Tilt Brush / no head device)")
    print("  13) Export saved recordings to one HDF5 file (train / validation / test users inside it)")
    print("  14) Download new users until train / validation / test can be filled, then export them")
    choice = input("Choice [1/2/3/4/5/6/7/8/9/10/11/12/13/14]: ").strip()

    if choice == "2":
        evaluate_cached_tars()
    elif choice == "3":
        inspect_saved()
    elif choice == "4":
        raw = input("How many random tars? [1]: ").strip()
        n = int(raw) if raw.isdigit() and int(raw) > 0 else 1
        process_random_tars(n)
    elif choice == "5":
        read_saved()
    elif choice == "6":
        review_failed()
    elif choice == "7":
        show_results_log()
    elif choice == "8":
        raw = input("How many random tars? [3]: ").strip()
        n = int(raw) if raw.isdigit() and int(raw) > 0 else 3
        workers = ask_int("How many threads? [4]: ", 4)
        process_random_tars_threaded(n, max_workers=workers)
    elif choice == "9":
        evaluate_saved_files()
    elif choice == "10":
        restore_missing_saved_files()
    elif choice == "11":
        repair_results_log()
    elif choice == "12":
        purge_unusable_saved()
    elif choice == "13":
        export_hdf5()
    elif choice == "14":
        counts = ask_split_counts()
        workers = ask_int("How many threads? [4]: ", 4)
        download_users(counts, max_workers=workers)
    else:
        process_tar(get_tar_names()[TAR_INDEX])


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.chdir(DATA_DIR)  # the data paths in config.py are relative, so this puts all data in <project root>/data
    main()
