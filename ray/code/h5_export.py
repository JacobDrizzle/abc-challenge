"""Menu option 13: write the saved recordings to ONE HDF5 file, split by user inside it.

Layout (Group B's data spec):
  /train | /validation | /test / <user_id> / <recording_id> / segment_NNN /
      time_s            float64 [N]     seconds, uniform 0.01 s spacing
      quaternion_wxyz   float32 [N, 4]  [HEAD.1, HEAD.i, HEAD.j, HEAD.k], unit length, no sign flips
A user's group sits under exactly one split, so the splits can't leak into each other.
manifest_<run_id>.json next to it records the settings, every recording's status and the totals."""

import hashlib
import json
import os
import random
from datetime import datetime

from xror import XROR

from .cleaning import POLICIES, prepare_recording
from .cli import ask_int, human_size
from .config import (EULER_ORDER, H5_CACHE_PATH, H5_COMPRESSION, H5_DEFAULT_USERS, H5_DIR, H5_MAX_GAP_S,
                     H5_MAX_RECORDINGS_PER_USER, H5_MIN_SAMPLES, H5_RATE_HZ, H5_SEED, H5_SPLITS,
                     MIN_ASSESSED_SAMPLES, MIN_QUAT_NORM, MIN_WINDOW_SAMPLES, PREPARATION_VERSION,
                     RATE_BAND_HZ, RESERVED_SPLIT_PATH, SWEEP_MIN_DEG, SWEEP_PERCENTILES, TRIM_S,
                     VELOCITY_MIN_DEG_S, WINDOW_S)
from .hf import get_tar_names
from .state import load_processed
from .storage import list_saved, saved_path_to_key


def load_h5_cache(path: str = H5_CACHE_PATH) -> dict:
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"users": {}, "runs": []}


def current_entries(cache: dict) -> tuple[dict, list[str]]:
    """Cache entries from this preparation version, and the tars among them whose .h5 file no longer exists.
    Entries from any other version don't count, so a version bump makes every user available again."""
    done = {t: r for t, r in cache["users"].items() if r.get("preparation_version") == PREPARATION_VERSION}
    vanished = [t for t, r in done.items() if r.get("file") and not os.path.isfile(r["file"])]
    return done, vanished


def save_h5_cache(cache: dict, path: str = H5_CACHE_PATH) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp, path)  # atomic, so a crash can't leave half a file


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish(tmp: str, final: str) -> None:
    """Flush tmp to disk, then give it its final name in one step."""
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    os.replace(tmp, final)


def write_json(path: str, value) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
    publish(tmp, path)


def load_reserved_users(path: str) -> set[str]:
    """Group B's test_users and retired_test_users. They may only go into the test split."""
    with open(path, encoding="utf-8") as f:
        split = json.load(f)
    reserved: set[str] = set()
    for key in ("test_users", "retired_test_users"):
        users = split[key] if key == "test_users" else split.get(key, [])
        if not isinstance(users, list) or not all(isinstance(u, str) for u in users):
            raise ValueError(f"'{key}' is not a list of user IDs")
        reserved.update(users)
    return reserved


def saved_users() -> dict[str, list[str]]:
    """{tar name: [saved .xror paths]}. One tar = one user."""
    by_base = {os.path.basename(t): t for t in get_tar_names()}
    users: dict[str, list[str]] = {}
    for p in list_saved():
        users.setdefault(saved_path_to_key(p, by_base)[0], []).append(p)
    return users


def user_id_of(tar_name: str) -> str:
    """users/<shard>/<uuid>.tar -> <uuid>"""
    return os.path.splitext(os.path.basename(tar_name))[0]


def recording_order(user_id: str, paths: list[str]) -> list[str]:
    """Seeded order, so the per-user cap doesn't favour the archive's own ordering. It hashes the
    tar member name (<user>/<recording>.xror) exactly as Group B's reference does."""
    def key(path):
        member = f"{user_id}/{os.path.splitext(os.path.basename(path))[0]}.xror"
        return hashlib.sha256(f"{H5_SEED}:{member}".encode()).hexdigest()
    return sorted(paths, key=key)


def prepare_user(user_id: str, paths: list[str]) -> tuple[list[dict], list[tuple]]:
    """Run one user's recordings through cleaning.prepare_recording in the seeded order until
    H5_MAX_RECORDINGS_PER_USER qualify. Returns (one audit row per recording,
    [(recording_id, row, segments) for each qualified recording])."""
    rows: list[dict] = []
    accepted: list[tuple] = []
    for path in recording_order(user_id, paths):
        recording = os.path.splitext(os.path.basename(path))[0]
        if len(accepted) >= H5_MAX_RECORDINGS_PER_USER:
            rows.append({"user": user_id, "recording": recording, "status": "not_inspected_user_cap"})
            continue
        try:
            with open(path, "rb") as f:
                raw = f.read()
            row, segments = prepare_recording(XROR.unpack(raw).data, user_id, recording)
            row["source_sha256"] = hashlib.sha256(raw).hexdigest()
        except Exception as e:
            row, segments = {"user": user_id, "recording": recording, "status": "decode_error",
                             "error": f"{type(e).__name__}: {e}"}, []
        rows.append(row)
        if segments:
            accepted.append((recording, row, segments))
    return rows, accepted


def pick_split(user_id: str, reserved: set[str], filled: dict[str, int], need: dict[str, int]) -> str | None:
    """First split (train, then validation, then test) that still needs users and may take this one.
    Reserved users (Group B's test users) may only go into test."""
    allowed = ("test",) if user_id in reserved else H5_SPLITS
    return next((s for s in H5_SPLITS if s in allowed and filled[s] < need[s]), None)


def users_needed(counts: tuple[int, int, int], other: int, reserved: int) -> int:
    """How many more usable users the export needs to fill every split, given `other` usable users and
    `reserved` usable Group B test users. Reserved users only go into test (pick_split), so train and
    validation must come from other users, and all three splits together from other and reserved users."""
    train, validation, test = counts
    return max(train + validation - other, train + validation + test - other - reserved, 0)


def write_user(split_group, user_id: str, accepted: list[tuple]) -> tuple[int, int, int]:
    """Write one user's qualified recordings under its split. Returns (recordings, segments, samples)."""
    kw = {"compression": H5_COMPRESSION, "shuffle": True} if H5_COMPRESSION else {}
    user_group = split_group.create_group(user_id)
    n_segments = n_samples = 0
    for recording, row, segments in accepted:
        rec_group = user_group.create_group(recording)
        for i, (t, q) in enumerate(segments):
            g = rec_group.create_group(f"segment_{i:03d}")
            g.create_dataset("time_s", data=t, **kw)
            g.create_dataset("quaternion_wxyz", data=q, **kw)
            g.attrs.update(challenge_filter_pass=True, sweep_deg=row["sweep_deg"],
                           velocity_deg_s=row["velocity_deg_s"], source_rate_hz=row["source_rate_hz"],
                           app=row.get("app", "unknown"), headset=row.get("headset", "unknown"),
                           preparation_version=PREPARATION_VERSION)
            n_segments += 1
            n_samples += len(t)
    return len(accepted), n_segments, n_samples


def settings() -> dict:
    """Every setting that shapes the export, for the manifest."""
    return {"rate_hz": H5_RATE_HZ, "trim_s": TRIM_S, "window_s": WINDOW_S,
            "min_window_samples": MIN_WINDOW_SAMPLES, "min_assessed_samples": MIN_ASSESSED_SAMPLES,
            "euler_order": EULER_ORDER + " (extrinsic, degrees)", "sweep_percentiles": list(SWEEP_PERCENTILES),
            "sweep_deg_strict_gt": SWEEP_MIN_DEG, "velocity_deg_s_strict_gt": VELOCITY_MIN_DEG_S,
            "rate_band_hz_inclusive": list(RATE_BAND_HZ), "rate_estimate": "1 / median positive interval, after trim and de-duplication",
            "min_quat_norm": MIN_QUAT_NORM, "max_gap_s": H5_MAX_GAP_S, "min_segment_samples": H5_MIN_SAMPLES,
            "max_recordings_per_user": H5_MAX_RECORDINGS_PER_USER,
            "recording_order": f"sha256('{H5_SEED}:<user>/<recording>.xror')",
            "split_seed": H5_SEED, "split_order": "sorted users, random.Random(seed).shuffle, filled train -> validation -> test",
            "compression": H5_COMPRESSION}


def scale_split(counts: tuple[int, int, int], available: int) -> tuple[int, int, int]:
    """Largest split with the same train:val:test ratio that fits in `available` users."""
    total = sum(counts)
    exact = [c * available / total for c in counts]
    out = [int(x) for x in exact]
    for i in sorted(range(3), key=lambda i: out[i] - exact[i])[: available - sum(out)]:
        out[i] += 1                                           # hand leftovers to the biggest remainders
    for i in range(3):                                        # don't silently drop a split you asked for
        if counts[i] > 0 and out[i] == 0 and max(out) > 1:
            out[out.index(max(out))] -= 1
            out[i] = 1
    return out[0], out[1], out[2]


def ask_split_counts() -> tuple[int, int, int]:
    """Ask how many training, validation and test users to put into the h5 file."""
    tr, va, te = H5_DEFAULT_USERS
    print("\nHow many users per split? (Enter = default. A user is one tar file, and a user never appears in two splits.)")
    train = max(0, ask_int(f"  Training users [{tr}]: ", tr))
    val = max(0, ask_int(f"  Validation users [{va}]: ", va))
    test = max(0, ask_int(f"  Test users [{te}]: ", te))
    return train, val, test


def export_hdf5(counts: tuple[int, int, int] | None = None, out_dir: str = H5_DIR,
                cache_path: str = H5_CACHE_PATH, reserved_path: str = RESERVED_SPLIT_PATH) -> str | None:
    """Option 13: write the saved recordings to h5/qualified_head_100hz_<run_id>.h5 with the
    train / validation / test users inside it, plus manifest_<run_id>.json next to it.
    The file is written under a temporary name and renamed only once it is complete.
    counts=(train, validation, test) skips the questions. Returns the new .h5 path, or None."""
    try:
        import h5py
    except ImportError:
        print("h5py is not installed. Run:  pip install h5py")
        return None
    try:
        reserved = load_reserved_users(reserved_path)
    except (OSError, ValueError, KeyError) as e:
        print(f"Can't read Group B's reserved test users from {reserved_path} ({e}).\n"
              f"Nothing exported: without that list a reserved user could end up in train or validation.")
        return None

    cache = load_h5_cache(cache_path)
    done, vanished = current_entries(cache)
    for t in vanished:
        del done[t]
        del cache["users"][t]
    if vanished:
        print(f"{len(vanished)} users were written to h5 files that no longer exist, so they are available again.")
        save_h5_cache(cache, cache_path)
    n_older = sum(1 for r in cache["users"].values() if r.get("preparation_version") != PREPARATION_VERSION)

    users = saved_users()
    n_rec = sum(len(v) for v in users.values())
    candidates = sorted(t for t in users if t not in done)
    n_exp = sum(1 for t in users if t in done and done[t].get("status") != "empty")
    n_empty = sum(1 for t in users if t in done and done[t].get("status") == "empty")
    n_reserved = sum(1 for t in candidates if user_id_of(t) in reserved)
    print(f"\n{len(users)} users with saved recordings ({n_rec} recordings): {n_exp} already exported, "
          f"{n_empty} skipped as unusable, {len(candidates)} available ({PREPARATION_VERSION}).")
    if n_older:
        print(f"{n_older} users in the cache were exported under an older preparation version; they are available again.")
    if n_reserved:
        print(f"{n_reserved} available users are Group B test users, so they can only go into the test split.")
    if not candidates:
        print("Nothing to export. Download more users (option 14, or python main.py download) first.")
        return None

    interactive = counts is None
    counts = ask_split_counts() if interactive else tuple(counts)
    if min(counts) < 1:
        print("Every split needs at least one user, so the file is a complete train / validation / test set.")
        return None
    asked = sum(counts)
    if asked > len(candidates):
        scaled = scale_split(counts, len(candidates))
        print("\nNot enough users for that.")
        print(f"  You asked for : {counts[0]} train + {counts[1]} validation + {counts[2]} test = {asked} users")
        print(f"  Available     : {len(candidates)} users that are not exported yet")
        print(f"  Missing       : {asked - len(candidates)} users (option 14, or python main.py download, gets more)")
        print(f"  Max with the same ratio : {scaled[0]} train + {scaled[1]} validation + {scaled[2]} test = {sum(scaled)}")
        if min(scaled) < 1:
            print("That leaves a split empty. Download more tars first.")
            return None
        if interactive and input(f"Run with {scaled[0]} / {scaled[1]} / {scaled[2]} instead? [y/N]: ").strip().lower() not in ("y", "yes"):
            print("Cancelled.")
            return None
        counts = scaled

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    created = datetime.now().isoformat(timespec="seconds")
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, f"qualified_head_100hz_{run_id}.h5")
    tmp = final + ".tmp"
    manifest_path = os.path.join(out_dir, f"manifest_{run_id}.json")
    random.Random(H5_SEED).shuffle(candidates)
    need: dict[str, int] = dict(zip(H5_SPLITS, counts))
    stats = {s: {"users": 0, "recordings": 0, "segments": 0, "samples": 0} for s in H5_SPLITS}
    members: dict[str, list[str]] = {s: [] for s in H5_SPLITS}
    per_user: dict[str, dict] = {}       # tar name -> what was written for that user
    empty: list[str] = []                # tars with no qualified recording
    rows: list[dict] = []                # one audit row per inspected recording

    try:
        with h5py.File(tmp, "w") as h5:
            h5.attrs.update({"quaternion_order": "wxyz", "rate_hz": H5_RATE_HZ,
                             "preparation_version": PREPARATION_VERSION, "run_id": run_id, "created": created,
                             "source": "BOXRR-23", "trim_s": TRIM_S, "max_gap_s": H5_MAX_GAP_S,
                             "min_segment_samples": H5_MIN_SAMPLES, "split_seed": H5_SEED})
            h5.attrs.create("splits", list(H5_SPLITS), dtype=h5py.string_dtype())
            for s in H5_SPLITS:
                h5.create_group(s)

            for tar_name in candidates:
                if all(stats[s]["users"] >= need[s] for s in H5_SPLITS):
                    break
                uid = user_id_of(tar_name)
                split = pick_split(uid, reserved, {s: stats[s]["users"] for s in H5_SPLITS}, need)
                if split is None:
                    continue  # a reserved user while test is full: left for a later run

                user_rows, accepted = prepare_user(uid, users[tar_name])
                rows.extend(dict(r, tar=tar_name, split=split if accepted else None) for r in user_rows)
                if not accepted:
                    empty.append(tar_name)
                    print(f"  {uid[:12]}...: no qualified recording, skipped and remembered")
                    continue

                n_r, n_seg, n_samp = write_user(h5[split], uid, accepted)
                st = stats[split]
                st["users"] += 1
                st["recordings"] += n_r
                st["segments"] += n_seg
                st["samples"] += n_samp
                members[split].append(uid)
                per_user[tar_name] = {"split": split, "recordings": n_r, "segments": n_seg, "samples": n_samp}
                print(f"  [{split} {st['users']}/{need[split]}] {uid[:12]}...  "
                      f"{len(users[tar_name])} saved -> {n_r} recordings, {n_seg} segments, {n_samp:,} samples "
                      f"({n_samp / H5_RATE_HZ / 60:.1f} min)")

            for s in H5_SPLITS:
                h5[s].attrs.update({"n_users": stats[s]["users"], "n_recordings": stats[s]["recordings"],
                                    "n_segments": stats[s]["segments"], "n_samples": stats[s]["samples"]})

        missing = [s for s in H5_SPLITS if stats[s]["users"] == 0]
        if missing:
            os.remove(tmp)
            print(f"\nNo file written: no usable user for {', '.join(missing)}. "
                  f"Download more users (option 14, or python main.py download) and run this again.")
            return None
        publish(tmp, final)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)  # never leave a half-written file behind
        raise

    # Which filter version picked each source tar's files in saved/ (recorded by process_tar).
    processed = load_processed()
    filtered_with = {t: processed.get(t, {}).get("preparation_version", "unknown") for t in per_user}
    all_current = all(v == PREPARATION_VERSION for v in filtered_with.values())
    totals = {s: dict(stats[s], hours=stats[s]["samples"] / H5_RATE_HZ / 3600) for s in H5_SPLITS}
    manifest = {
        "dataset": "BOXRR-23 (cschell/boxrr-23)",
        "preparation_version": PREPARATION_VERSION,
        "run_id": run_id,
        "created": created,
        "file": os.path.basename(final),
        "data_sha256": sha256_file(final),
        "settings": settings(),
        "policies": POLICIES,
        "quaternion_order": "wxyz",
        "reserved_users": {"path": reserved_path, "sha256": sha256_file(reserved_path), "count": len(reserved),
                           "placed_in_test": sorted(set(members["test"]) & reserved),
                           "rule": "test_users and retired_test_users never go into train or validation"},
        "requested_users": dict(zip(H5_SPLITS, counts)),
        "splits": {s: sorted(members[s]) for s in H5_SPLITS},
        "splits_note": "A record only. The top-level groups of the .h5 file are the authoritative split.",
        "totals": totals,
        "source_tars": {t: {"user": user_id_of(t), "split": per_user[t]["split"], "filtered_with": filtered_with[t]}
                        for t in sorted(per_user)},
        "inspected_tars_without_qualified_recordings": sorted(empty),
        "selection": {
            "source": "saved/, which keeps only the recordings that passed process_func when their tar was "
                      "downloaded. Recordings rejected at download time are not re-examined here.",
            "all_sources_filtered_with_current_version": all_current,
            "limitation": None if all_current else
                "Some source tars were filtered at download time by another (or an unknown) preparation "
                "version, see source_tars[].filtered_with. Recordings that version wrongly rejected are "
                "missing from saved/, so the cohort and the per-user picks can differ from a fresh "
                f"{PREPARATION_VERSION} download. Nothing was re-downloaded.",
        },
        "notes": ["100 Hz / 20 steps = 200 ms; the challenge FAQ says 100 ms. Open question for the organisers.",
                  "Offline SLERP can use later samples; it is not certified causal raw-sensor preprocessing.",
                  "Strict > thresholds follow the detailed algorithm; the Data page summary uses >=."],
        "recordings": rows,
    }
    write_json(manifest_path, manifest)

    # Remember the exported users only now that the file and manifest exist under their final names.
    for t, info in per_user.items():
        cache["users"][t] = dict(info, file=final, manifest=manifest_path, run=run_id,
                                 preparation_version=PREPARATION_VERSION)
    for t in empty:
        cache["users"][t] = {"status": "empty", "run": run_id, "preparation_version": PREPARATION_VERSION,
                             "reason": f"no qualified recording (challenge filter, then segments of at least {H5_MIN_SAMPLES} samples)"}
    cache["runs"].append({"run": run_id, "preparation_version": PREPARATION_VERSION, "requested": list(counts),
                          "written": [stats[s]["users"] for s in H5_SPLITS], "file": final, "manifest": manifest_path})
    save_h5_cache(cache, cache_path)

    print("\n" + "=" * 60)
    print(f"HDF5 EXPORT {run_id}  ({PREPARATION_VERSION})")
    for s in H5_SPLITS:
        st = stats[s]
        print(f"  {s:<11}{st['users']:>4}/{need[s]:<4} users {st['recordings']:>5} recordings "
              f"{st['segments']:>6} segments {totals[s]['hours']:>7.2f} h")
    print(f"  file     {os.path.abspath(final)}  ({human_size(os.path.getsize(final))})")
    print(f"  manifest {os.path.abspath(manifest_path)}")
    short = [s for s in H5_SPLITS if stats[s]["users"] < need[s]]
    if short:
        print(f"\nRan out of usable users before filling: {', '.join(short)}."
              f" {len(empty)} users had no qualified recording. Download more users (option 14, or python main.py download) and run this again.")
    if not all_current:
        print(f"\nWarning: some source tars were filtered at download time by an older or unknown version, "
              f"so recordings that version rejected are missing. See 'selection' in the manifest.")
    print("=" * 60)
    return final
