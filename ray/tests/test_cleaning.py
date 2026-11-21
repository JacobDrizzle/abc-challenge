"""Cleaning, motion filter and resample rules (code/cleaning.py) on synthetic recordings.
Several cases are ported from Group B's reference tests (group_a_data_pipeline/tests/test_cleaning.py)."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from code.cleaning import align, clean, motion_filter, passes, prepare_recording, resample
from code.config import H5_MAX_GAP_S
from code.processing import LAST_FAIL, LAST_METRICS, process_func


HEAD = {"joint": "HEAD", "type": "HMD", "name": "Test headset", "axes": ["i", "j", "k", "1"]}


def recording(t, q=None, frames=None, devices=(HEAD,), user="user", rec="recording"):
    """A recording shaped like XROR.unpack(raw).data. q is xyzw, stored as HEAD.i/j/k/1."""
    if frames is None:
        frames = np.column_stack((t, q))
    return {"$id": rec, "info": {"user": {"id": user}, "software": {"app": {"name": "Test app"}},
                                 "hardware": {"devices": list(devices)}},
            "frames": np.asarray(frames, dtype=np.float64).tolist()}


def swing(t, degrees=True):
    """A head turning +-30 deg about z, fast enough to pass the motion filter."""
    return Rotation.from_euler("z", 30 * np.sin(1.5 * t), degrees=degrees).as_quat()


def wave(rate=100, motion=True, gap=False, invalid=False, seconds=45):
    """Port of the reference recording_payload: every 7th quaternion has its sign flipped."""
    t = np.arange(int(seconds * rate)) / rate
    angles = 30 * np.sin(1.5 * t) if motion else np.zeros(len(t))
    q = Rotation.from_euler("z", angles[:, None], degrees=True).as_quat()
    q[::7] *= -1
    if gap:
        t[int(22 * rate):] += .5
    if invalid:
        q[:] = 0
    return recording(t, q)


def no_negative_dots(q):
    return np.all(np.sum(q[1:] * q[:-1], axis=1) >= 0)


# Trim -------------------------------------------------------------------------------------

def test_trim_counts_elapsed_time_when_clock_does_not_start_at_zero():
    # The old code kept t > 5 on absolute time, so a clock starting at 1000 s was not trimmed at all.
    t = 1000 + np.arange(4500) / 100
    row, segments = prepare_recording(recording(t, swing(t)), "user", "recording")
    assert row["status"] == "qualified"
    assert row["trimmed_samples"] == 500
    assert segments[0][0][0] == pytest.approx(1005.0)


def test_later_clock_reset_splits_without_sorting_or_a_second_trim():
    t = np.arange(4500) / 100
    q = Rotation.from_euler("z", 30 * np.sin(1.5 * t)).as_quat()
    t[2500:] -= 25
    t[1001] = t[1000]
    row, segments = prepare_recording(recording(t, q), "user", "recording")
    assert row["status"] == "qualified"
    assert row["duplicate_samples"] == 1
    assert row["backwards_steps"] == 1
    assert len(segments) == 2
    assert segments[0][0][0] == pytest.approx(5.)
    assert segments[1][0][0] == pytest.approx(0.)          # after the reset: not trimmed again
    assert segments[1][0][0] < segments[0][0][-1]          # kept in recording order, not sorted
    assert all(np.all(np.diff(s) > 0) for s, _ in segments)


def test_fewer_than_20_samples_after_trim_is_not_assessed():
    row, segments = prepare_recording(wave(seconds=5.15), "user", "recording")
    assert row["status"] == "too_short_after_trim"
    assert segments == []


# Duplicates and rate ----------------------------------------------------------------------

def test_duplicate_timestamp_keeps_the_first_sample():
    t = np.arange(3000) / 100
    q = swing(t)
    t[1001] = t[1000]
    q[1001] = Rotation.from_euler("x", 90, degrees=True).as_quat()  # a different rotation at the same time
    row = {}
    t2, q2, status = clean(t, q, row)
    assert status is None
    assert row["duplicate_samples"] == 1
    kept = np.flatnonzero(t2 == t[1000])
    assert len(kept) == 1
    np.testing.assert_allclose(q2[kept[0]], q[1000])
    assert row["source_rate_hz"] == pytest.approx(100, rel=1e-6)   # median of positive steps only


@pytest.mark.parametrize("rate, status", [(84, "rate_reject"), (86, "qualified"),
                                          (109, "qualified"), (111, "rate_reject"), (80, "rate_reject")])
def test_rate_gate(rate, status):
    row, _ = prepare_recording(wave(rate=rate), "user", "recording")
    assert row["status"] == status
    assert row["source_rate_hz"] == pytest.approx(rate, rel=1e-6)


# Motion filter ----------------------------------------------------------------------------

def test_thresholds_are_strict():
    assert not passes({"sweep_deg": 17.0, "velocity_deg_s": 30.0})
    assert not passes({"sweep_deg": 30.0, "velocity_deg_s": 15.0})
    assert passes({"sweep_deg": 17.000001, "velocity_deg_s": 15.000001})
    assert not passes({"sweep_deg": None, "velocity_deg_s": None})


def test_motion_windows_span_tracking_gaps():
    # 3 s of tracking then a 0.2 s dropout, repeated: no gap-free 5 s window exists.
    t = np.arange(6000) / 100
    t += .2 * (t // 3)
    angles = np.column_stack((np.zeros(len(t)), np.zeros(len(t)), 30 * np.sin(1.5 * t)))
    q = Rotation.from_euler("xyz", angles, degrees=True).as_quat()
    measures = motion_filter(t, q)
    assert measures["motion_windows"] > 5
    assert measures["sweep_deg"] > 17


def test_velocity_uses_nominal_window_duration():
    t = np.arange(2000) / 100
    angles = np.column_stack((np.zeros(len(t)), np.zeros(len(t)), 30 * t))
    q = Rotation.from_euler("xyz", angles, degrees=True).as_quat()
    # Each window holds 500 samples = 499 steps of 0.3 deg, divided by 5 s.
    assert motion_filter(t, q)["velocity_deg_s"] == pytest.approx(499 * .3 / 5, abs=1e-6)


def test_sweep_uses_the_angles_as_computed_without_unwrap():
    # Yaw wobbling around 180 deg: the Euler angle jumps between +180 and -180.
    # Unwrapped, the sweep would be about 20 deg; the spec's angles give about 350 deg.
    t = np.arange(3000) / 100
    angles = np.column_stack((np.zeros(len(t)), np.zeros(len(t)), 180 + 10 * np.sin(1.5 * t)))
    q = Rotation.from_euler("xyz", angles, degrees=True).as_quat()
    assert motion_filter(t, q)["sweep_deg"] > 300


@pytest.mark.parametrize("kwargs, status", [({"motion": False}, "motion_reject"),
                                            ({"rate": 80}, "rate_reject"),
                                            ({"invalid": True}, "too_short")])
def test_rejects(kwargs, status):
    row, segments = prepare_recording(wave(**kwargs), "user", "recording")
    assert row["status"] == status
    assert segments == []


def test_mismatched_ids_are_refused():
    with pytest.raises(ValueError, match="IDs"):
        prepare_recording(wave(), "other", "recording")


# Gaps, signs and output order -------------------------------------------------------------

def two_pieces(first_after_gap):
    """-20 s .. 0 s, a gap, then 20 s more. The gap ends at first_after_gap / 100 s, so a gap of
    exactly 0.1 s is exact in floating point (0.1 - 0.0)."""
    t = np.r_[np.arange(-2000, 1), np.arange(first_after_gap, first_after_gap + 2000)] / 100
    return t, swing(t)


def test_gap_of_100ms_is_filled_and_a_longer_gap_splits():
    t, q = two_pieces(10)
    assert np.diff(t).max() == H5_MAX_GAP_S
    segments = resample(t, q, {})
    assert len(segments) == 1
    assert np.any(np.isclose(segments[0][0], 0.05))       # filled by SLERP

    t, q = two_pieces(11)
    segments = resample(t, q, {})
    assert len(segments) == 2
    assert segments[0][0][-1] == pytest.approx(0.0)
    assert segments[1][0][0] == pytest.approx(0.11)


def test_qualified_recording_with_a_gap_has_uniform_aligned_segments():
    row, segments = prepare_recording(wave(gap=True), "user", "recording")
    assert row["status"] == "qualified"
    assert len(segments) == 2
    for t, q in segments:
        assert np.allclose(np.diff(t), .01, atol=1e-6, rtol=0)
        assert no_negative_dots(q)
        assert np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-5)
        assert len(t) >= 1000


def test_signs_are_aligned_before_and_after_slerp():
    # A full turn about z: the quaternions cross hemispheres, and every 3rd input is sign-flipped.
    t = np.arange(4000) / 100
    q = Rotation.from_euler("z", 9 * t, degrees=True).as_quat()
    q[::3] *= -1
    _, out = resample(t, q, {})[0]
    assert no_negative_dots(out)
    assert np.allclose(np.linalg.norm(out, axis=1), 1, atol=1e-6)

    aligned = align(q)
    assert no_negative_dots(aligned)
    assert np.allclose(np.abs(np.sum(aligned * q, axis=1)), 1)   # same rotations, only signs changed


def test_output_is_wxyz_from_head_1_i_j_k():
    t = np.arange(1500) / 100
    q = np.tile(Rotation.from_euler("z", 30, degrees=True).as_quat(), (len(t), 1))  # xyzw
    _, out = resample(t, q, {})[0]
    expected = [np.cos(np.radians(15)), 0, 0, np.sin(np.radians(15))]
    np.testing.assert_allclose(out[0], expected, atol=1e-6)
    assert out.dtype == np.float32


def test_head_columns_come_from_axis_metadata():
    t = np.arange(4500) / 100
    q = swing(t)
    position = np.random.default_rng(0).normal(size=(len(t), 3))
    # The rotation axes come in a shuffled order after a position device.
    frames = np.column_stack((t, position, q[:, 3], q[:, 2], q[:, 0], q[:, 1]))
    devices = ({"joint": "HAND", "axes": ["x", "y", "z"]}, dict(HEAD, axes=["1", "k", "i", "j"]))
    row, segments = prepare_recording(recording(t, frames=frames, devices=devices), "user", "recording")
    assert row["status"] == "qualified"
    assert row["headset"] == "Test headset" and row["app"] == "Test app"
    expected = swing(segments[0][0])[:, [3, 0, 1, 2]]
    assert np.all(np.abs(np.sum(segments[0][1] * expected, axis=1)) > 1 - 1e-5)


# process_func uses the same rules ---------------------------------------------------------

def test_process_func_runs_the_shared_filter():
    data = wave()
    row, _ = prepare_recording(data, "user", "recording")
    result = process_func(data, quiet=True)
    assert result is not None
    assert LAST_METRICS["max_S"] == row["sweep_deg"]
    assert LAST_METRICS["max_V"] == row["velocity_deg_s"]
    assert LAST_METRICS["hz"] == row["source_rate_hz"]
    assert [result[a]["S"] for a in ("Pitch", "Yaw", "Roll")] == row["sweep_xyz_deg"]

    assert process_func(wave(rate=80), quiet=True) is None
    assert LAST_FAIL[0] == "sampling rate outside 85-110 Hz"
    assert process_func(wave(motion=False), quiet=True) is None
    assert LAST_FAIL[0] == "below S/V threshold"
    assert process_func(wave(seconds=5.15), quiet=True) is None
    assert LAST_FAIL == ["too few valid frames", "too_short_after_trim"]
