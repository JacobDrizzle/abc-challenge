"""The merge (code/h5_merge.py) on two real tiny exports: each user stays in its own split, the data is copied
unchanged, the manifest adds up, and overlapping, mismatched or changed sources are refused."""
import hashlib
import json
import os
import shutil

import h5py
import numpy as np
import pytest

from code import h5_export
from code.h5_merge import merge_exports
from test_h5_export import RESERVED, corpus, segments  # noqa: F401 (corpus is a fixture)


def manifest_of(path):
    return path.replace("qualified_head_100hz_", "manifest_").replace(".h5", ".json")


@pytest.fixture
def two_exports(corpus, tmp_path):
    """Two exports of disjoint users, as two download rounds give: (2, 1, 2) users, then (3, 1, 2) more."""
    corpus({**{f"user-{c}": [True] for c in "abcdefghijk"}, RESERVED: [True]})
    paths = dict(corpus.paths)
    first = h5_export.export_hdf5(counts=(2, 1, 2), **dict(paths, out_dir=str(tmp_path / "h5a")))
    second = h5_export.export_hdf5(counts=(3, 1, 2), **dict(paths, out_dir=str(tmp_path / "h5b")))
    return [manifest_of(first), manifest_of(second)], str(tmp_path / "h5")


def groups(path):
    with h5py.File(path, "r") as h5:
        return {s: set(h5[s]) for s in h5}


def test_each_user_keeps_its_split_and_its_data(two_exports):
    manifests, out_dir = two_exports
    out = merge_exports(manifests, out_dir)
    sources = [json.loads(open(m).read()) for m in manifests]
    files = [os.path.join(os.path.dirname(m), s["file"]) for m, s in zip(manifests, sources)]

    merged = groups(out)
    for split in ("train", "validation", "test"):
        assert merged[split] == groups(files[0])[split] | groups(files[1])[split]
    assert [len(merged[s]) for s in ("train", "validation", "test")] == [5, 2, 4]
    with h5py.File(out, "r") as h5:
        for path in files:
            with h5py.File(path, "r") as src:
                for split, user, rec, name, g in segments(src):
                    copy = h5[split][user][rec][name]
                    assert np.array_equal(copy["time_s"][:], g["time_s"][:])
                    assert np.array_equal(copy["quaternion_wxyz"][:], g["quaternion_wxyz"][:])
                    assert copy["quaternion_wxyz"].compression == g["quaternion_wxyz"].compression
                    assert dict(copy.attrs) == dict(g.attrs)
        manifest = json.loads(open(manifest_of(out)).read())
        assert h5.attrs["run_id"] == manifest["run_id"] and manifest["run_id"].startswith("merged_")
        assert list(h5.attrs["merged_from"]) == [s["run_id"] for s in sources]
        assert h5.attrs["preparation_version"] == sources[0]["preparation_version"]
        assert [h5[s].attrs["n_users"] for s in ("train", "validation", "test")] == [5, 2, 4]

    assert manifest["data_sha256"] == hashlib.sha256(open(out, "rb").read()).hexdigest()
    assert {s: set(u) for s, u in manifest["splits"].items()} == merged
    assert manifest["requested_users"] == {"train": 5, "validation": 2, "test": 4}
    for split in ("train", "validation", "test"):
        for key in ("users", "recordings", "segments", "samples"):
            assert manifest["totals"][split][key] == sum(s["totals"][split][key] for s in sources)
    assert manifest["settings"] == sources[0]["settings"] and manifest["policies"] == sources[0]["policies"]
    assert manifest["reserved_users"]["placed_in_test"] == [RESERVED]
    assert len(manifest["recordings"]) == sum(len(s["recordings"]) for s in sources)
    assert set(manifest["source_tars"]) == set(sources[0]["source_tars"]) | set(sources[1]["source_tars"])
    assert [m["data_sha256"] for m in manifest["merged_from"]] == [s["data_sha256"] for s in sources]


def test_overlapping_mismatched_or_changed_sources_are_refused(two_exports, tmp_path):
    manifests, out_dir = two_exports
    with pytest.raises(ValueError, match="at least two"):
        merge_exports(manifests[:1], out_dir)
    with pytest.raises(ValueError, match="is in both"):
        merge_exports([manifests[0], manifests[0]], out_dir)

    changed = json.loads(open(manifests[1]).read())
    changed["settings"]["max_gap_s"] = 0.2
    other = tmp_path / "h5b" / "manifest_changed.json"
    other.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="differ in settings"):
        merge_exports([manifests[0], str(other)], out_dir)

    data = os.path.join(os.path.dirname(manifests[1]), changed["file"])
    shutil.copyfile(data, data + ".bak")
    with h5py.File(data, "a") as h5:
        h5.attrs["note"] = "edited"
    with pytest.raises(ValueError, match="differs from its manifest"):
        merge_exports(manifests, out_dir)
    assert not os.path.exists(out_dir) or not os.listdir(out_dir)  # nothing written by a refused merge
