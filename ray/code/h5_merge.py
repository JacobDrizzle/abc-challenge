"""`python main.py merge`: join HDF5 exports into one file, each user staying in the split it was exported to.

An export only takes users that no earlier export holds, so a bigger cohort is a new export of new users plus a
merge with the earlier file: its train, validation and test users stay train, validation and test. Models trained
on the earlier file then never see the merged validation or test users, and the earlier validation and test users
are subsets of the merged ones.

The sources must share the preparation version, settings, policies and reserved-user list, and no user may be in
two of them. The user groups are copied unchanged (h5py copy keeps the datasets, compression and attributes). The
merged manifest is the export's, with lists joined, totals summed, and the sources under merged_from. The export
cache (h5_exported.json) is left alone: it still names the source files, so keep them, or their users become
"available again" and a later export could take them a second time.
"""
import json
import os
from datetime import datetime

from .cli import human_size
from .config import H5_DIR, H5_SPLITS
from .h5_export import publish, sha256_file, write_json

SAME = ("dataset", "preparation_version", "settings", "policies", "quaternion_order")
TOTALS = ("users", "recordings", "segments", "samples", "hours")


def load_source(manifest_path: str) -> dict:
    """A source export's manifest, refused unless its HDF5 file matches the manifest's checksum."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    path = os.path.join(os.path.dirname(manifest_path), manifest["file"])
    if sha256_file(path) != manifest["data_sha256"]:
        raise ValueError(f"{path}: the file differs from its manifest's data_sha256")
    return dict(manifest, _path=path, _manifest=manifest_path)


def check_sources(sources: list[dict]) -> None:
    if len(sources) < 2:
        raise ValueError("Name at least two exports to merge")
    first = sources[0]
    for source in sources[1:]:
        for key in SAME:
            if source[key] != first[key]:
                raise ValueError(f"{source['run_id']} and {first['run_id']} differ in {key}")
        if source["reserved_users"]["sha256"] != first["reserved_users"]["sha256"]:
            raise ValueError(f"{source['run_id']} and {first['run_id']} used different reserved-user lists")
    seen: dict[str, str] = {}
    for source in sources:
        for split in H5_SPLITS:
            for user in source["splits"][split]:
                if user in seen:
                    raise ValueError(f"user {user} is in both {seen[user]} and {source['run_id']}")
                seen[user] = source["run_id"]


def merge_exports(manifest_paths: list[str], out_dir: str = H5_DIR) -> str:
    """Write h5/qualified_head_100hz_merged_<time>.h5 and its manifest from the named exports. Returns the path."""
    import h5py

    sources = [load_source(p) for p in manifest_paths]
    check_sources(sources)
    run_id = "merged_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    created = datetime.now().isoformat(timespec="seconds")
    os.makedirs(out_dir, exist_ok=True)
    final = os.path.join(out_dir, f"qualified_head_100hz_{run_id}.h5")
    manifest_path = os.path.join(out_dir, f"manifest_{run_id}.json")
    if os.path.exists(final) or os.path.exists(manifest_path):
        raise FileExistsError(f"{final} already exists")
    tmp = final + ".tmp"
    totals = {s: {k: 0 for k in TOTALS} for s in H5_SPLITS}
    try:
        with h5py.File(tmp, "w") as out:
            with h5py.File(sources[0]["_path"], "r") as h5:
                out.attrs.update(dict(h5.attrs))  # the export's attributes; run_id and created are this merge's
            out.attrs.update({"run_id": run_id, "created": created})
            out.attrs.create("merged_from", [s["run_id"] for s in sources], dtype=h5py.string_dtype())
            for split in H5_SPLITS:
                out.create_group(split)
            for source in sources:
                with h5py.File(source["_path"], "r") as h5:
                    for split in H5_SPLITS:
                        if set(h5[split]) != set(source["splits"][split]):
                            raise ValueError(f"{source['run_id']}: /{split} differs from its manifest's users")
                        for user in h5[split]:
                            h5.copy(h5[split][user], out[split], name=user)
                for split in H5_SPLITS:
                    for key in TOTALS:
                        totals[split][key] += source["totals"][split][key]
            for split in H5_SPLITS:
                out[split].attrs.update({f"n_{k}": totals[split][k] for k in TOTALS if k != "hours"})
        publish(tmp, final)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)  # never leave a half-written file behind
        raise

    first = sources[0]
    limitations = [s["selection"]["limitation"] for s in sources if s["selection"]["limitation"]]
    manifest = {
        **{key: first[key] for key in SAME},
        "run_id": run_id,
        "created": created,
        "file": os.path.basename(final),
        "data_sha256": sha256_file(final),
        "merged_from": [{"run_id": s["run_id"], "file": s["file"], "data_sha256": s["data_sha256"],
                         "manifest": os.path.basename(s["_manifest"]), "manifest_sha256": sha256_file(s["_manifest"]),
                         "users": {split: len(s["splits"][split]) for split in H5_SPLITS}} for s in sources],
        "reserved_users": dict(first["reserved_users"], placed_in_test=sorted(
            u for s in sources for u in s["reserved_users"]["placed_in_test"])),
        "requested_users": {split: sum(s["requested_users"][split] for s in sources) for split in H5_SPLITS},
        "splits": {split: sorted(u for s in sources for u in s["splits"][split]) for split in H5_SPLITS},
        "splits_note": first["splits_note"],
        "totals": totals,
        "source_tars": {t: v for s in sources for t, v in s["source_tars"].items()},
        "inspected_tars_without_qualified_recordings": sorted(
            t for s in sources for t in s["inspected_tars_without_qualified_recordings"]),
        "selection": dict(first["selection"], all_sources_filtered_with_current_version=all(
            s["selection"]["all_sources_filtered_with_current_version"] for s in sources),
            limitation=" | ".join(limitations) or None),
        "notes": first["notes"] + ["A merge of the exports under merged_from: each user is in the split its own "
                                   "export put it in."],
        "recordings": [row for s in sources for row in s["recordings"]],
    }
    write_json(manifest_path, manifest)

    print("\n" + "=" * 60)
    print(f"HDF5 MERGE {run_id}  from {', '.join(s['run_id'] for s in sources)}")
    for split in H5_SPLITS:
        t = totals[split]
        print(f"  {split:<11}{t['users']:>5} users {t['recordings']:>6} recordings {t['segments']:>7} segments "
              f"{t['hours']:>8.2f} h")
    print(f"  file     {os.path.abspath(final)}  ({human_size(os.path.getsize(final))})")
    print(f"  manifest {os.path.abspath(manifest_path)}")
    print("=" * 60)
    return final
