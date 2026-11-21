"""Build the final 'more data' HDF5: moredata_1600.h5 kept exactly as it is (same users, same splits), plus every
newly pulled user from one or more pull folders, assigned to train/validation/test in the ratio 20:3:4 (seeded
shuffle, as export_moredata.py).

A pull folder ("source") has the Colab layout: processed.json (ray's list of finished tars) and
saved/users/<shard>/<user>/*.xror. The PC's own data folder (cache/processed.json) works too.

  python export_final.py --base moredata_1600.h5 --source DIR [--source DIR ...] \
      --exclude exclude_test_users.txt --hashes base_recording_sha256.txt --out OUT.h5 [--workers N] [--dry-run]

--hashes: SHA-256 of every recording already in the base file (and of every test recording). A new recording
with one of these hashes is dropped before cleaning, so a recording can't sit in two splits.
--dry-run lists and counts the candidates only, without cleaning or writing anything.

Pass 1 cleans each candidate once (in --workers processes) and writes the usable ones to a pool file next to the
output. Pass 2 copies the base users and then the pooled users into their splits, checks, and writes the manifest.
"""
import argparse
import hashlib
import json
import multiprocessing
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # ray/ first: its package is called `code` and shadows the standard library's

import h5py  # noqa: E402

from code.config import H5_COMPRESSION, H5_MAX_GAP_S, H5_MIN_SAMPLES, H5_RATE_HZ, H5_SEED, PREPARATION_VERSION, TRIM_S  # noqa: E402
from code.h5_export import prepare_user, sha256_file, write_user  # noqa: E402
from export_moredata import RATIO, SPLITS, read_set, split_counts  # noqa: E402


def processed_path(source):
    for p in (os.path.join(source, "processed.json"), os.path.join(source, "cache", "processed.json")):
        if os.path.isfile(p):
            return p
    raise SystemExit(f"{source}: no processed.json")


def source_users(sources, skip):
    """{user: {file name: path}} over all sources, for users whose tar is recorded as qualified there.
    A user found in two sources keeps the union of its files; the same file name must have the same size."""
    users, origin, skipped, overlap = {}, {}, 0, []
    for source in sources:
        with open(processed_path(source), encoding="utf-8") as f:
            done = json.load(f)
        qualified = {os.path.splitext(os.path.basename(t))[0] for t, v in done.items() if v.get("qualified")}
        root, found = os.path.join(source, "saved", "users"), 0
        for shard in sorted(os.listdir(root)):
            for uid in sorted(os.listdir(os.path.join(root, shard))):
                if uid not in qualified:
                    continue  # tar not finished (or its files not confirmed) in this source
                if uid in skip:
                    skipped += 1
                    continue
                folder = os.path.join(root, shard, uid)
                files = {n: os.path.join(folder, n) for n in sorted(os.listdir(folder)) if n.endswith(".xror")}
                if not files:
                    continue
                found += 1
                if uid in users:
                    overlap.append(uid)
                    for n, p in files.items():
                        if n in users[uid] and os.path.getsize(users[uid][n]) != os.path.getsize(p):
                            raise SystemExit(f"{uid}/{n}: different sizes in {origin[uid]} and {source}")
                        users[uid].setdefault(n, p)
                else:
                    users[uid], origin[uid] = dict(files), source
        print(f"{source}: {found} candidate users", flush=True)
    return users, skipped, overlap


def clean(job):
    """Worker: hash a user's recordings, drop the ones already in the base file, clean the rest."""
    uid, paths, base_hashes = job
    keep, dropped = [], 0
    for p in paths:
        with open(p, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        if h in base_hashes:
            dropped += 1
        else:
            keep.append(p)
    accepted = prepare_user(uid, keep)[1] if keep else []
    used = {row["source_sha256"] for _, row, _ in accepted}
    return uid, dropped, sorted(used), accepted


_BASE_HASHES = None


def _init(hashes):
    global _BASE_HASHES
    _BASE_HASHES = hashes


def _clean(item):
    uid, paths = item
    return clean((uid, paths, _BASE_HASHES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="moredata_1600.h5: kept as it is")
    ap.add_argument("--source", action="append", required=True, help="pull folder; give it once per folder")
    ap.add_argument("--exclude", required=True, help="test users (Jacob's 160 + the older 47)")
    ap.add_argument("--hashes", required=True, help="SHA-256 of every recording already in the base file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    excluded, base_hashes = read_set(args.exclude), read_set(args.hashes)
    with h5py.File(args.base, "r") as b:
        base = {s: sorted(b[s].keys()) for s in SPLITS}
    base_all = set().union(*base.values())
    print("base:", {s: len(v) for s, v in base.items()}, f"| {len(base_hashes)} base recording hashes", flush=True)

    users, skipped, overlap = source_users(args.source, excluded | base_all)
    print(f"{len(users)} candidate new users ({skipped} skipped as base or test users, "
          f"{len(overlap)} found in more than one source)", flush=True)
    if args.dry_run:
        return

    # Pass 1: clean every candidate once; usable users go to the pool file.
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    pool_path = os.path.join(out_dir, "pool.tmp.h5")
    seen, dupes, usable, dropped_hash = {}, [], [], 0
    jobs = [(u, list(users[u].values())) for u in sorted(users)]
    ctx = multiprocessing.get_context("fork")
    with h5py.File(pool_path, "w") as pool, ctx.Pool(args.workers, _init, (base_hashes,)) as workers:
        for i, (uid, dropped, used, accepted) in enumerate(workers.imap(_clean, jobs, chunksize=4), 1):
            dropped_hash += dropped
            for h in used:
                if h in seen:
                    dupes.append((seen[h], uid, h))
                seen.setdefault(h, uid)
            if accepted:
                write_user(pool, uid, accepted)
                usable.append(uid)
            if i % 200 == 0 or i == len(jobs):
                rate = i / (time.time() - t0)
                print(f"  cleaned {i}/{len(jobs)} users, {len(usable)} usable | {rate:.1f} users/s, "
                      f"~{(len(jobs) - i) / rate / 60:.0f} min left", flush=True)
    if dupes:
        for a, b_, h in dupes[:20]:
            print(f"  same recording in {a} and {b_}: {h[:16]}", flush=True)
        raise SystemExit(f"{len(dupes)} recordings appear under two new users: decide what to do before building")
    print(f"{len(usable)} of {len(users)} candidates are usable; {dropped_hash} recordings dropped as identical "
          f"to a base recording", flush=True)

    usable.sort()
    random.Random(H5_SEED).shuffle(usable)
    n_tr, n_va, n_te = split_counts(len(usable))
    assign = {"train": usable[:n_tr], "validation": usable[n_tr:n_tr + n_va], "test": usable[n_tr + n_va:]}
    print(f"new users -> train {n_tr}, validation {n_va}, test {n_te}", flush=True)
    print("final totals:", {s: len(base[s]) + len(assign[s]) for s in SPLITS}, flush=True)

    # Pass 2: base users copied as they are, then the pooled users into their splits.
    tmp = args.out + ".tmp"
    run_id = os.path.splitext(os.path.basename(args.out))[0]  # Jacob's trainer finds manifest_<run_id>.json next to it
    manifest_path = os.path.join(out_dir, f"manifest_{run_id}.json")
    stats = {s: {"users": 0, "recordings": 0, "segments": 0, "samples": 0} for s in SPLITS}
    with h5py.File(tmp, "w") as out, h5py.File(args.base, "r") as b, h5py.File(pool_path, "r") as pool:
        out.attrs.update({"run_id": run_id, "quaternion_order": "wxyz", "rate_hz": H5_RATE_HZ,
                          "preparation_version": PREPARATION_VERSION, "source": "BOXRR-23", "trim_s": TRIM_S,
                          "max_gap_s": H5_MAX_GAP_S, "min_segment_samples": H5_MIN_SAMPLES, "split_seed": H5_SEED})
        out.attrs.create("splits", list(SPLITS), dtype=h5py.string_dtype())
        for s in SPLITS:
            g = out.create_group(s)
            for uid in base[s]:
                b.copy(b[s][uid], g, name=uid)
            for uid in assign[s]:
                pool.copy(pool[uid], g, name=uid)
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
        assert set(base[s]) <= members[s], f"a base {s} user moved"
        assert members[s] == set(base[s]) | set(assign[s]), f"{s} has users it should not"
    new_all = set(usable)
    assert not (new_all & excluded), "an excluded test user got in"
    assert not (new_all & base_all), "a base user was added twice"
    os.replace(tmp, args.out)
    os.remove(pool_path)

    manifest = {"run_id": run_id, "file": os.path.basename(args.out), "data_sha256": sha256_file(args.out),
                "preparation_version": PREPARATION_VERSION,
                "sources": [os.path.basename(os.path.normpath(s)) for s in args.source],
                "base": {"file": os.path.basename(args.base), "users": {s: len(base[s]) for s in SPLITS}},
                "new_users": {s: len(assign[s]) for s in SPLITS}, "ratio": RATIO, "seed": H5_SEED,
                "candidates": len(users), "usable": len(usable), "users_in_two_sources": len(overlap),
                "dropped_base_identical_recordings": dropped_hash, "compression": H5_COMPRESSION,
                "splits": {s: sorted(members[s]) for s in SPLITS}, "totals": stats}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"done in {(time.time() - t0) / 60:.0f} min:", args.out, {s: stats[s]["users"] for s in SPLITS}, flush=True)


if __name__ == "__main__":
    main()
