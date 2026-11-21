"""Build the 'more data' HDF5: Jacob's 800/120/160 users kept exactly where they are, plus every newly pulled
user assigned to train/validation/test in the ratio 20:3:4 (seeded shuffle).

Runs on the PC (needs ray's venv). Reads a frozen snapshot of saved/, so the pull can keep running.

  python export_moredata.py --jacob-h5 JACOB.h5 --snapshot /srv/data/abc_boxrr_data/snapshot_20261106 \
      --exclude exclude_test_users.txt --have have_users.txt --hashes test_recording_sha256.txt \
      --out /srv/data/abc_boxrr_data/export/moredata_20261106.h5 [--dry-run]

--dry-run does the counting pass only and writes nothing.
"""
import argparse
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # ray/ first: its package is called `code` and shadows the standard library's

import h5py  # noqa: E402

from code.config import H5_COMPRESSION, H5_MAX_GAP_S, H5_MIN_SAMPLES, H5_RATE_HZ, H5_SEED, PREPARATION_VERSION, TRIM_S  # noqa: E402
from code.h5_export import prepare_user, sha256_file, write_user  # noqa: E402

SPLITS = ("train", "validation", "test")
RATIO = {"train": 20, "validation": 3, "test": 4}


def read_set(path):
    with open(path, encoding="utf-8") as f:
        return {l.strip() for l in f if l.strip()}


def snapshot_users(snapshot, skip, test_hashes):
    """{user_id: [.xror paths]} for users that finished before the snapshot, minus skipped users and minus any
    recording whose bytes equal a test recording."""
    with open(os.path.join(snapshot, "cache", "processed.json"), encoding="utf-8") as f:
        done = json.load(f)
    qualified = {os.path.splitext(os.path.basename(t))[0] for t, v in done.items() if v.get("qualified")}
    users, dropped_hash, dropped_skip = {}, 0, 0
    root = os.path.join(snapshot, "saved", "users")
    for shard in sorted(os.listdir(root)):
        for uid in sorted(os.listdir(os.path.join(root, shard))):
            if uid not in qualified:
                continue  # tar was still being written when the snapshot was taken
            if uid in skip:
                dropped_skip += 1
                continue
            folder = os.path.join(root, shard, uid)
            keep = []
            for name in sorted(os.listdir(folder)):
                p = os.path.join(folder, name)
                with open(p, "rb") as f:
                    if hashlib.sha256(f.read()).hexdigest() in test_hashes:
                        dropped_hash += 1
                        continue
                keep.append(p)
            if keep:
                users[uid] = keep
    return users, dropped_hash, dropped_skip


def split_counts(n):
    """n new users -> (train, validation, test) in 20:3:4. Validation and test are rounded, train takes the rest."""
    val = round(n * RATIO["validation"] / 27)
    test = round(n * RATIO["test"] / 27)
    return n - val - test, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jacob-h5", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--exclude", required=True, help="test users (Jacob's 160 + the older 47)")
    ap.add_argument("--have", required=True, help="train/validation users already in Jacob's file")
    ap.add_argument("--hashes", required=True, help="SHA-256 of every test recording")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-new", type=int, help="use exactly this many new users (1080 doubles Jacob's 800/120/160)")
    args = ap.parse_args()

    skip = read_set(args.exclude) | read_set(args.have)
    test_hashes = read_set(args.hashes)
    with h5py.File(args.jacob_h5, "r") as j:
        jacob = {s: sorted(j[s].keys()) for s in SPLITS}
    jacob_all = set().union(*jacob.values())
    print("Jacob:", {s: len(v) for s, v in jacob.items()}, flush=True)

    users, dropped_hash, dropped_skip = snapshot_users(args.snapshot, skip | jacob_all, test_hashes)
    print(f"snapshot: {len(users)} candidate new users ({dropped_skip} skipped as already known/test, "
          f"{dropped_hash} test-identical recordings dropped)", flush=True)

    # Pass 1: which candidates have at least one recording that survives ray's cleaning + 5-recording cap?
    usable = sorted(u for u in users if prepare_user(u, users[u])[1])
    print(f"{len(usable)} of {len(users)} candidates are usable", flush=True)
    random.Random(H5_SEED).shuffle(usable)
    if args.n_new is not None:
        if len(usable) < args.n_new:
            raise SystemExit(f"only {len(usable)} usable new users, need {args.n_new}; wait for the pull")
        usable = usable[:args.n_new]  # first n of the seeded shuffle, so the choice is not hand-picked
    n_tr, n_va, n_te = split_counts(len(usable))
    assign = {"train": usable[:n_tr], "validation": usable[n_tr:n_tr + n_va], "test": usable[n_tr + n_va:]}
    print(f"new users -> train {n_tr}, validation {n_va}, test {n_te}", flush=True)
    print("final totals:", {s: len(jacob[s]) + len(assign[s]) for s in SPLITS}, flush=True)
    if args.dry_run:
        return

    # Pass 2: write. Jacob's users are copied as they are; new users go through write_user.
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    # Jacob's trainer finds the split with manifest_for(): the h5's run_id attribute names the manifest next to it.
    run_id = os.path.splitext(os.path.basename(args.out))[0]
    manifest_path = os.path.join(os.path.dirname(args.out), f"manifest_{run_id}.json")
    stats = {s: {"users": 0, "recordings": 0, "segments": 0, "samples": 0} for s in SPLITS}
    with h5py.File(tmp, "w") as out, h5py.File(args.jacob_h5, "r") as j:
        out.attrs.update({"run_id": run_id, "quaternion_order": "wxyz", "rate_hz": H5_RATE_HZ, "preparation_version": PREPARATION_VERSION,
                          "source": "BOXRR-23", "trim_s": TRIM_S, "max_gap_s": H5_MAX_GAP_S,
                          "min_segment_samples": H5_MIN_SAMPLES, "split_seed": H5_SEED})
        out.attrs.create("splits", list(SPLITS), dtype=h5py.string_dtype())
        for s in SPLITS:
            g = out.create_group(s)
            for uid in jacob[s]:
                j.copy(j[s][uid], g, name=uid)
        for s in SPLITS:
            for uid in assign[s]:
                _, accepted = prepare_user(uid, users[uid])
                write_user(out[s], uid, accepted)
        for s in SPLITS:
            for uid in out[s]:
                stats[s]["users"] += 1
                for rec in out[s][uid].values():
                    stats[s]["recordings"] += 1
                    for seg in rec.values():
                        stats[s]["segments"] += 1
                        stats[s]["samples"] += seg["time_s"].shape[0]
            out[s].attrs.update({f"n_{k}": v for k, v in stats[s].items()})

    # Checks before the file gets its real name
    with h5py.File(tmp, "r") as f:
        members = {s: set(f[s].keys()) for s in SPLITS}
        assert not (members["train"] & members["validation"] or members["train"] & members["test"]
                    or members["validation"] & members["test"]), "a user is in two splits"
        for s in SPLITS:
            assert set(jacob[s]) <= members[s], f"a Jacob {s} user moved"
        # The current test sets are Jacob's 160 plus the new test users; the disjointness check above covers both.
        # The older 47-user list is not scored any more; 5 of its users sit in Jacob's train/val, so only record them.
        assert not ((members["train"] | members["validation"]) & set(jacob["test"])), "a test user is in train/val"
        old47_in_trainval = sorted((members["train"] | members["validation"]) & (read_set(args.exclude) - set(jacob["test"])))
        print(f"old-47 users sitting in train/validation (from Jacob's file, kept): {len(old47_in_trainval)}", flush=True)
    os.replace(tmp, args.out)

    manifest = {"run_id": run_id, "file": os.path.basename(args.out), "data_sha256": sha256_file(args.out),
                "preparation_version": PREPARATION_VERSION, "snapshot": os.path.basename(args.snapshot),
                "base": {"file": os.path.basename(args.jacob_h5), "users": {s: len(jacob[s]) for s in SPLITS}},
                "new_users": {s: len(assign[s]) for s in SPLITS}, "ratio": RATIO, "seed": H5_SEED,
                "splits": {s: sorted(members[s]) for s in SPLITS}, "totals": stats,
                "dropped_test_identical_recordings": dropped_hash, "compression": H5_COMPRESSION,
                "old47_users_in_train_validation": old47_in_trainval,
                "old47_note": "Older 47-user test list is no longer scored; these users come from Jacob's file and were kept."}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("done:", args.out, {s: stats[s]["users"] for s in SPLITS}, flush=True)


if __name__ == "__main__":
    main()
