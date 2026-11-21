"""Downloading until every split can be filled (pipeline.download_users), the number of users it
still needs (h5_export.users_needed), the qualified count process_tar records, and main.py's commands."""
import json
import os
import tarfile

import pytest

import main
from code import h5_export, pipeline
from code.config import DATA_DIR, PREPARATION_VERSION
from test_h5_export import write_xror


@pytest.mark.parametrize("other, reserved, needed", [(0, 0, 4), (3, 0, 1), (4, 0, 0), (9, 0, 0),
                                                     (2, 2, 1), (3, 1, 0), (0, 5, 3)])
def test_users_needed_counts_reserved_users_towards_test_only(other, reserved, needed):
    assert h5_export.users_needed((2, 1, 1), other, reserved) == needed


@pytest.fixture
def run(monkeypatch):
    """run(counts, outcomes) calls download_users with process_tar faked. outcomes is
    [(tar name, qualified recordings or an exception), ...] in download order.
    Returns (download_users' result, tars downloaded, counts passed to the export)."""
    def run(counts, outcomes, reserved=(), existing=(0, 0), **kwargs):
        outcome, downloaded, exported = dict(outcomes), [], []

        def fake_process_tar(tar_name):
            downloaded.append(tar_name)
            if isinstance(outcome[tar_name], Exception):
                raise outcome[tar_name]
            return {"total": 10, "passed": 1, "qualified": outcome[tar_name]}

        monkeypatch.setattr(pipeline, "process_tar", fake_process_tar)
        monkeypatch.setattr(pipeline, "pick_new_tars", lambda: [t for t, _ in outcomes])
        monkeypatch.setattr(pipeline, "load_reserved_users", lambda path: set(reserved))
        monkeypatch.setattr(pipeline, "usable_saved_users", lambda reserved_users: existing)
        monkeypatch.setattr(pipeline, "export_hdf5", lambda counts: exported.append(counts) or "out.h5")
        kwargs.setdefault("max_workers", 1)
        return pipeline.download_users(counts, **kwargs), downloaded, exported
    return run


def tars(*qualified):
    return [(f"users/0/user-{i}.tar", q) for i, q in enumerate(qualified)]


def test_downloads_until_every_split_can_be_filled_then_exports(run):
    result, downloaded, exported = run((2, 1, 1), tars(*[0, 0, 1] * 6))
    assert len(downloaded) == 12          # the 4th usable user is the 12th tar
    assert exported == [(2, 1, 1)] and result == "out.h5"


def test_reserved_users_only_count_towards_test(run):
    outcomes = [("users/0/r0.tar", 1), ("users/0/r1.tar", 2)] + tars(1, 1, 1, 1)
    _, downloaded, exported = run((1, 1, 1), outcomes, reserved={"r0", "r1"})
    # r0 fills test and r1 is a spare, so train and validation still need two other users.
    assert downloaded == ["users/0/r0.tar", "users/0/r1.tar", "users/0/user-0.tar", "users/0/user-1.tar"]
    assert exported == [(1, 1, 1)]


def test_users_already_in_saved_count_first(run):
    _, downloaded, exported = run((2, 1, 1), tars(1, 1), existing=(4, 0))
    assert downloaded == [] and exported == [(2, 1, 1)]
    _, downloaded, _ = run((2, 1, 1), tars(0, 1, 1), existing=(3, 0))
    assert downloaded == ["users/0/user-0.tar", "users/0/user-1.tar"]


def test_stops_after_failures_in_a_row_and_exports_nothing(run, monkeypatch):
    monkeypatch.setattr(pipeline, "DOWNLOAD_MAX_ERRORS_IN_ROW", 3)
    fail = OSError("connection reset")
    result, downloaded, exported = run((1, 1, 1), tars(fail, fail, 0, fail, fail, fail, 1, 1, 1))
    assert len(downloaded) == 6           # the success after two failures restarts the count
    assert result is None and exported == []


def test_running_out_of_tars_or_no_export_exports_nothing(run):
    result, downloaded, exported = run((1, 1, 1), tars(1, 0, 1))
    assert len(downloaded) == 3 and result is None and exported == []
    result, downloaded, exported = run((1, 1, 1), tars(1, 1, 1), export=False)
    assert len(downloaded) == 3 and result is None and exported == []


def test_threads_finish_the_downloads_already_running(run):
    _, downloaded, exported = run((1, 1, 1), tars(*[1] * 10), max_workers=4)
    assert len(downloaded) == 4           # 4 started together, 3 were needed
    assert exported == [(1, 1, 1)]


def test_bad_counts_or_no_reserved_list_download_nothing(run, monkeypatch):
    assert run((1, 0, 1), tars(1, 1, 1))[1] == []

    def missing(path):
        raise OSError("no such file")
    monkeypatch.setattr(pipeline, "load_reserved_users", missing)
    monkeypatch.setattr(pipeline, "pick_new_tars", lambda: pytest.fail("downloaded without a reserved list"))
    assert pipeline.download_users((1, 1, 1)) is None


@pytest.fixture
def user_a_tar(tmp_path, monkeypatch):
    """users/0/user-a.tar, "downloaded" from tmp_path. Works in tmp_path, since cache/ and saved/ are
    relative to the working directory. Returns the tar name."""
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "src"
    write_xror(src / "rec-0.xror", "user-a", "rec-0")                 # qualifies
    write_xror(src / "rec-1.xror", "user-a", "rec-1", seconds=12)     # passes the filter, but only 7 s after the trim
    write_xror(src / "rec-2.xror", "user-a", "rec-2", motion=False)   # fails the filter
    with tarfile.open(tmp_path / "user-a.tar", "w") as tar:
        for p in sorted(src.iterdir()):
            tar.add(p, arcname=f"user-a/{p.name}")
    monkeypatch.setattr(pipeline, "hf_hub_download", lambda *a, **k: str(tmp_path / "user-a.tar"))
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    return "users/0/user-a.tar"


def test_process_tar_records_how_many_recordings_the_export_would_take(user_a_tar, monkeypatch):
    tar_name = user_a_tar
    record = pipeline.process_tar(tar_name)
    assert (record["total"], record["passed"], record["qualified"]) == (3, 2, 1)
    assert sorted(os.listdir("saved/users/0/user-a")) == ["rec-0.xror", "rec-1.xror"]

    # usable_saved_users trusts processed.json for this version, and checks the files otherwise.
    monkeypatch.setattr(h5_export, "get_tar_names", lambda: [tar_name])
    assert pipeline.usable_saved_users(set()) == (1, 0)
    assert pipeline.usable_saved_users({"user-a"}) == (0, 1)
    processed = json.loads(open("cache/processed.json").read())
    processed[tar_name]["qualified"] = 0
    with open("cache/processed.json", "w") as f:
        json.dump(processed, f)
    assert pipeline.usable_saved_users(set()) == (0, 0)
    processed[tar_name]["preparation_version"] = "older"
    with open("cache/processed.json", "w") as f:
        json.dump(processed, f)
    assert pipeline.usable_saved_users(set()) == (1, 0)

    # A user already exported under this version isn't usable again.
    open("old.h5", "wb").close()
    with open("cache/h5_exported.json", "w") as f:
        json.dump({"users": {tar_name: {"split": "train", "file": "old.h5",
                                        "preparation_version": PREPARATION_VERSION}}, "runs": []}, f)
    assert pipeline.usable_saved_users(set()) == (0, 0)


def test_a_tar_that_fails_part_way_leaves_nothing_behind(user_a_tar, monkeypatch):
    real = pipeline.iter_tar_replays

    def truncated(path):
        replays = real(path)
        yield next(replays)                    # rec-0 passes and is staged...
        raise OSError("unexpected end of data")  # ...then the download turns out to be cut short

    monkeypatch.setattr(pipeline, "iter_tar_replays", truncated)
    with pytest.raises(OSError):
        pipeline.process_tar(user_a_tar)
    assert not os.path.exists("saved")        # no half-downloaded user
    assert os.listdir("cache") == []          # no record, no log, no temporary folder
    assert not set().union(*pipeline.tars_already_done().values())   # so the tar counts as new again

    monkeypatch.setattr(pipeline, "iter_tar_replays", real)
    assert pipeline.process_tar(user_a_tar)["passed"] == 2


def test_saved_folder_arrives_whole_even_if_windows_locks_it_briefly(user_a_tar, monkeypatch):
    real_rename, renames = os.rename, []

    def locked_twice(src, dst):
        renames.append(dst)
        if len(renames) <= 2:
            raise PermissionError(13, "Access is denied")
        real_rename(src, dst)

    monkeypatch.setattr(pipeline.os, "rename", locked_twice)
    pipeline.process_tar(user_a_tar)
    assert len(renames) == 3
    assert sorted(os.listdir("saved/users/0/user-a")) == ["rec-0.xror", "rec-1.xror"]

    record = pipeline.process_tar(user_a_tar, force=True)   # the folder exists now: its files are replaced
    assert record["passed"] == 2 and len(renames) == 3
    assert sorted(os.listdir("saved/users/0/user-a")) == ["rec-0.xror", "rec-1.xror"]


def test_data_goes_into_the_project_data_folder():
    project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert DATA_DIR == os.path.join(project, "data")


def test_command_line(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "download_users", lambda counts, **kw: calls.append(("download", counts, kw)))
    monkeypatch.setattr(main, "export_hdf5", lambda counts: calls.append(("export", counts)))
    monkeypatch.setattr(main, "menu", lambda: calls.append(("menu",)))

    main.main(["download", "--train", "100", "--val", "15", "--test", "20"])
    main.main(["download", "--train", "5", "--val", "2", "--test", "2", "--threads", "8", "--no-export"])
    main.main(["export", "--train", "5", "--val", "2", "--test", "2"])
    main.main([])
    assert calls == [("download", (100, 15, 20), {"max_workers": 4, "export": True}),
                     ("download", (5, 2, 2), {"max_workers": 8, "export": False}),
                     ("export", (5, 2, 2)),
                     ("menu",)]
    with pytest.raises(SystemExit):
        main.main(["download", "--train", "100"])       # every count is required
