"""The HDF5 export (code/h5_export.py) on tiny real .xror files: one file with the three user
splits inside it, the spec's attributes, reserved users, the per-user cap, the cache and crash safety."""
import hashlib
import json
import os

import bson
import fpzip
import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from code import h5_export
from code.config import PREPARATION_VERSION


RESERVED = "user-r"


def write_xror(path, user, rec, motion=True, seconds=45, rate=100):
    t = np.arange(int(seconds * rate)) / rate
    angles = 30 * np.sin(1.5 * t) if motion else np.zeros(len(t))
    q = Rotation.from_euler("z", angles, degrees=True).as_quat()  # xyzw = HEAD.i/j/k/1
    frames = np.column_stack((t, q)).astype(np.float64)
    payload = bson.encode({
        "$id": rec, "events": [], "frames": fpzip.compress(frames[None, None]),
        "info": {"user": {"id": user}, "software": {"app": {"name": "Beat Saber"}},
                 "hardware": {"devices": [{"joint": "HEAD", "type": "HMD", "name": "Quest 2",
                                           "axes": ["i", "j", "k", "1"]}]}}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Returns make(spec): spec is {user: [motion, ...]}, one entry per recording. Writes the
    files under tmp_path/saved and points the export at them. Paths for the export go in `paths`."""
    monkeypatch.chdir(tmp_path)  # cache/processed.json is read relative to the working directory
    (tmp_path / "split.json").write_text(json.dumps({"train_users": [], "validation_users": [],
                                                     "test_users": [RESERVED], "retired_test_users": []}))

    def make(spec):
        users = {}
        for user, motions in spec.items():
            paths = []
            for i, motion in enumerate(motions):
                p = tmp_path / "saved" / "users" / "0" / user / f"rec-{i}.xror"
                write_xror(p, user, f"rec-{i}", motion)
                paths.append(str(p))
            users[f"users/0/{user}.tar"] = paths
        monkeypatch.setattr(h5_export, "saved_users", lambda: users)
        return users

    make.paths = {"out_dir": str(tmp_path / "h5"), "cache_path": str(tmp_path / "cache.json"),
                  "reserved_path": str(tmp_path / "split.json")}
    return make


def segments(h5):
    for split in h5:
        for user, user_group in h5[split].items():
            for rec, rec_group in user_group.items():
                for name, g in rec_group.items():
                    yield split, user, rec, name, g


def test_one_file_with_three_user_splits_and_the_spec_layout(corpus):
    corpus({**{f"user-{c}": [True] for c in "abcde"}, RESERVED: [True], "user-s": [False]})
    out = h5_export.export_hdf5(counts=(2, 1, 2), **corpus.paths)

    run_id = os.path.basename(out)[len("qualified_head_100hz_"):-len(".h5")]
    assert sorted(os.listdir(corpus.paths["out_dir"])) == sorted([os.path.basename(out), f"manifest_{run_id}.json"])
    with h5py.File(out, "r") as h5:
        assert set(h5) == {"train", "validation", "test"}
        a = h5.attrs
        assert (a["quaternion_order"], a["rate_hz"], a["preparation_version"], a["source"]) == \
               ("wxyz", 100, PREPARATION_VERSION, "BOXRR-23")
        assert (a["trim_s"], a["max_gap_s"], a["min_segment_samples"], a["split_seed"]) == (5.0, 0.1, 1000, 42)
        assert a["run_id"] == run_id and a["created"]
        assert list(a["splits"]) == ["train", "validation", "test"]

        groups = {s: set(h5[s]) for s in h5}
        assert [len(groups[s]) for s in ("train", "validation", "test")] == [2, 1, 2]
        assert not (groups["train"] & groups["validation"] or groups["train"] & groups["test"]
                    or groups["validation"] & groups["test"])
        assert RESERVED not in groups["train"] | groups["validation"]
        for s in h5:
            n_seg = sum(1 for _ in segments({s: h5[s]}))
            assert h5[s].attrs["n_users"] == len(groups[s]) and h5[s].attrs["n_segments"] == n_seg

        for split, user, rec, name, g in segments(h5):
            t, q = g["time_s"][:], g["quaternion_wxyz"][:]
            assert name == "segment_000" and rec == "rec-0"
            assert t.dtype == np.float64 and q.dtype == np.float32 and q.shape == (len(t), 4)
            assert np.allclose(np.diff(t), .01, atol=1e-6, rtol=0) and len(t) >= 1000
            assert np.all(q[:, 0] > .96) and np.all(np.abs(q[:, 1:3]) < 1e-6)   # w first, rotation about z
            assert bool(g.attrs["challenge_filter_pass"]) is True
            assert g.attrs["sweep_deg"] > 17 and g.attrs["velocity_deg_s"] > 15
            assert g.attrs["source_rate_hz"] == pytest.approx(100)
            assert (g.attrs["app"], g.attrs["headset"], g.attrs["preparation_version"]) == \
                   ("Beat Saber", "Quest 2", PREPARATION_VERSION)

    manifest = json.loads(open(os.path.join(corpus.paths["out_dir"], f"manifest_{run_id}.json")).read())
    assert manifest["data_sha256"] == hashlib.sha256(open(out, "rb").read()).hexdigest()
    assert {s: set(u) for s, u in manifest["splits"].items()} == groups
    assert manifest["preparation_version"] == PREPARATION_VERSION
    assert manifest["settings"]["max_gap_s"] == 0.1 and manifest["policies"]["sweep_unwrapped"] is False
    assert manifest["selection"]["all_sources_filtered_with_current_version"] is False  # no processed.json here
    assert {r["status"] for r in manifest["recordings"]} <= {"qualified", "motion_reject"}
    cache = json.loads(open(corpus.paths["cache_path"]).read())
    assert all(e["preparation_version"] == PREPARATION_VERSION for e in cache["users"].values())


def test_per_user_cap_takes_five_recordings_in_the_seeded_order(corpus):
    users = corpus({"user-a": [True] * 7, "user-b": [True], "user-c": [True]})
    out = h5_export.export_hdf5(counts=(1, 1, 1), **corpus.paths)

    order = [os.path.basename(p)[:-5] for p in h5_export.recording_order("user-a", users["users/0/user-a.tar"])]
    with h5py.File(out, "r") as h5:
        split = next(s for s in h5 if "user-a" in h5[s])
        assert sorted(h5[split]["user-a"]) == sorted(order[:5])
    manifest_path = out.replace("qualified_head_100hz_", "manifest_").replace(".h5", ".json")
    rows = [r for r in json.loads(open(manifest_path).read())["recordings"] if r["user"] == "user-a"]
    assert [r["recording"] for r in rows] == order
    assert [r["status"] for r in rows] == ["qualified"] * 5 + ["not_inspected_user_cap"] * 2


def test_reserved_users_may_only_go_into_test():
    need = {"train": 1, "validation": 1, "test": 1}
    empty = {"train": 0, "validation": 0, "test": 0}
    assert h5_export.pick_split(RESERVED, {RESERVED}, empty, need) == "test"
    assert h5_export.pick_split(RESERVED, {RESERVED}, dict(empty, test=1), need) is None
    assert h5_export.pick_split("user-a", {RESERVED}, empty, need) == "train"
    assert h5_export.pick_split("user-a", {RESERVED}, dict(empty, train=1), need) == "validation"


def test_users_from_older_versions_are_available_again(corpus, tmp_path):
    corpus({c: [True] for c in ("user-a", "user-b", "user-c", "user-d")})
    still_there = tmp_path / "current.h5"
    still_there.write_bytes(b"")
    old = {"users": {"users/0/user-a.tar": {"split": "train", "file": "h5/train_old.h5", "run": "old"},
                     "users/0/user-d.tar": {"split": "train", "file": str(still_there), "run": "now",
                                            "preparation_version": PREPARATION_VERSION}},
           "runs": [{"run": "old"}]}
    with open(corpus.paths["cache_path"], "w") as f:
        json.dump(old, f)

    out = h5_export.export_hdf5(counts=(1, 1, 1), **corpus.paths)
    with h5py.File(out, "r") as h5:
        exported = {u for s in h5 for u in h5[s]}
    assert exported == {"user-a", "user-b", "user-c"}   # the old-format entry doesn't block user-a
    cache = json.loads(open(corpus.paths["cache_path"]).read())
    assert cache["users"]["users/0/user-a.tar"]["preparation_version"] == PREPARATION_VERSION
    assert cache["users"]["users/0/user-d.tar"] == old["users"]["users/0/user-d.tar"]
    assert cache["runs"][0] == {"run": "old"} and cache["runs"][-1]["file"] == out


def test_a_crash_leaves_no_file_and_no_cache_entry(corpus, monkeypatch):
    corpus({c: [True] for c in ("user-a", "user-b", "user-c")})
    real, calls = h5_export.prepare_user, []

    def crash_on_second_user(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("simulated crash")
        return real(*args)

    monkeypatch.setattr(h5_export, "prepare_user", crash_on_second_user)
    with pytest.raises(RuntimeError):
        h5_export.export_hdf5(counts=(1, 1, 1), **corpus.paths)
    assert os.listdir(corpus.paths["out_dir"]) == []
    assert not os.path.exists(corpus.paths["cache_path"])


def test_an_empty_split_writes_nothing(corpus):
    corpus({"user-a": [True], "user-b": [True], "user-s": [False]})
    assert h5_export.export_hdf5(counts=(1, 1, 1), **corpus.paths) is None
    assert os.listdir(corpus.paths["out_dir"]) == []
    assert not os.path.exists(corpus.paths["cache_path"])


def test_no_reserved_list_no_export(corpus):
    corpus({c: [True] for c in ("user-a", "user-b", "user-c")})
    paths = dict(corpus.paths, reserved_path=corpus.paths["reserved_path"] + ".missing")
    assert h5_export.export_hdf5(counts=(1, 1, 1), **paths) is None
    assert not os.path.exists(corpus.paths["out_dir"])
