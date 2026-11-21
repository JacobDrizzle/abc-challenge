"""make_synthetic_data.py: synthetic stand-ins for the BOXRR-23 recordings the site used to show (20 Nov 2026).

Why: BOXRR-23's data use agreement (clause 4) does not allow passing its recordings on without Berkeley's written
consent, and the challenge asks teams not to share raw or derived data, so a public site cannot show them. Counts and
other statistics about the data stay (like the paper's); every picture of a recording is replaced by synthetic head
motion from tools/synthetic_motion.py, which is not a recording of anyone and is not fitted to anyone's recordings.

  python visualization/tactical/tools/make_synthetic_data.py traces     (a few seconds)
  python visualization/tactical/tools/make_synthetic_data.py examples   (about a minute)

traces: four synthetic players for the model explorer, traces/syn25.js, syn75.js, syn90.js and syn97.js, in the format
  of traces/test50.js (the model functions are tools/make_questset_data.py's: same model, same checks). Each replaces the
  BOXRR test player of the same speed: its 300 scored calls are the stretch whose mean head speed is closest to that
  player's target, the 25th, 75th, 90th or 97th percentile of the BOXRR test windows' speeds (31.2, 53.7, 69.7 and
  92.2 deg/s, aggregate figures from Nic's 13 Nov traces). traces/index.js keeps its QuestSet entries, first, and gets
  these four after them; every other entry (the BOXRR players) is dropped.

examples: data/pipeline/examples.js, the Data side's file. Its `config`, `pull`, `final` and `local` blocks are counts
  and settings and are kept as they are (made by Nic's make_data_examples.py and make_data_final.py). `examples` and
  `wall` are rebuilt:
  1. A synthetic pile of 766 raw recordings from 17 synthetic players, made to the counts in `local` (the team's local
     folder of 766 real recordings): the same number of each fate, the same sampling rates, the same number of
     recordings with an empty first row, rows stamped 0 s, repeated timestamps, a clock that steps back, holes over
     0.1 s and sign flips, the same shortest, median and longest length and the same longest hole. The generator
     recomputes `local` from the pile and stops if any of those counts differs.
  2. Every recording is a real .xror file (packed with the xror library, as BOXRR-23's are), read back and run through
     the team's pipeline (ray/code/head.py, ray/code/cleaning.py prepare_recording). Its fate is what the pipeline
     says, and every number shown about it comes from the pipeline or from the file.
  3. The examples are picked from the pile by the same rules as before (closest to 170 s, most repeated timestamps,
     stillest, liveliest, ...). The defects the examples show were planted in the pile on purpose, at the sizes the
     real folder had them (a stuck clock of about 3,600 rows, a 0.6 s clock reset in the first 5 s, one sign flip, ...).
  4. Stage 9's window is not synthetic: it is the input and answer of the first call that traces/test50.js (the QuestSet
     player, CC BY 4.0) scores, so the Data side and the Model side show the same window.
  The unused `timestamps` example is left out.
"""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
SITE = TOOLS.parent
REPO = SITE.parents[1]
sys.path[:0] = [str(REPO), str(TOOLS)]
from scipy.spatial.transform import Rotation  # noqa: E402
from xror import XROR  # noqa: E402

import make_questset_data as mq  # noqa: E402
from bench.geometry import geodesic_degrees  # noqa: E402
from ray.code.cleaning import align, clean, motion_filter, passes, prepare_recording  # noqa: E402
from ray.code.config import (EULER_ORDER, H5_MAX_GAP_S, H5_MAX_RECORDINGS_PER_USER, H5_MIN_SAMPLES,  # noqa: E402
                             H5_RATE_HZ, H5_SEED, MIN_ASSESSED_SAMPLES, MIN_WINDOW_SAMPLES, SWEEP_PERCENTILES,
                             TRIM_S, WINDOW_S)
from ray.code.head import unusable_reason  # noqa: E402
from synthetic_motion import HeadMotion, quaternions, ttl  # noqa: E402

EXAMPLES = SITE / "data/pipeline/examples.js"
EXAMPLES_PREFIX = "window.DATA_EXAMPLES = "
TODAY = datetime.date.today().isoformat()

# ---------------------------------------------------------------- traces

# id, label, BOXRR test window-speed percentile, its speed (deg/s, window_speed_percentile_target_deg_s of the 13 Nov
# traces), the activity to start from (tools/synthetic_motion.py), seed
TRACES = (("syn25", "Synthetic · calm", 25, 31.208772601540986, 2.0, 25),
          ("syn75", "Synthetic · turning", 75, 53.65382188667495, 5.0, 75),
          ("syn90", "Synthetic · fast turning", 90, 69.71085109620624, 8.0, 90),
          ("syn97", "Synthetic · very fast turning", 97, 92.2280057388669, 12.0, 97))
TRACE_SECONDS, TRACE_NOISE_DEG = 180, 0.015


def trace_session(activity, seed):
    """A synthetic session on an exact 100 Hz grid: (times, unit float64 quaternions wxyz)."""
    times = np.arange(TRACE_SECONDS * H5_RATE_HZ) / H5_RATE_HZ
    motion = HeadMotion(TRACE_SECONDS, activity, np.random.default_rng(seed), noise_deg=TRACE_NOISE_DEG)
    q = quaternions(motion.angles(times))
    return times, q / np.linalg.norm(q, axis=1, keepdims=True)


def make_traces():
    entries = [e for e in mq.read_index() if e["source"] == "questset"]
    for trace_id, label, percentile, target, activity, seed in TRACES:
        for _ in range(6):   # nudge the activity until the session's median window speed is near the target
            times, q = trace_session(activity, seed)
            median = mq.select_window(q)[2]
            if abs(median / target - 1) < 0.08:
                break
            activity *= (target / median) ** 1.3
        first, speed, _, candidates = mq.select_window(q, target)
        assert abs(speed - target) < 0.5, f"{trace_id}: no window within 0.5 deg/s of {target:.1f} ({speed:.2f})"
        record = motion_filter(times, q[:, [1, 2, 3, 0]])
        steps, _ = mq.trace_steps(q, times, first)
        about = {
            "dataset": "synthetic head motion: not a recording of any person",
            "credit": "Synthetic: made by tools/make_synthetic_data.py with tools/synthetic_motion.py.",
            "synthesis": {"module": "tools/synthetic_motion.py", "seed": seed, "activity": activity,
                          "seconds": TRACE_SECONDS, "rate_hz": H5_RATE_HZ, "noise_deg": TRACE_NOISE_DEG},
            "user": f"synthetic-{percentile}", "recording": "synthetic", "segment": "segment_000",
            "split": "synthetic (no person; not used for training)",
            "frame_conversion": "none: made in the model's frame (Unity: +x right, +y up, +z forward), head -> world",
            "preparation": "made directly on an exact 100 Hz grid; nothing to clean",
            "challenge_filter_pass": bool(passes(record)), "segment_samples": len(q),
            "preparation_record": {"source_rate_hz": float(H5_RATE_HZ), "sweep_deg": record["sweep_deg"],
                                   "velocity_deg_s": record["velocity_deg_s"], "motion_windows": record["motion_windows"]},
            "segment_start_time_s": 0.0,
            "selection": "Motion only, never error: of the 300-anchor windows with 792 calls of history, the one whose "
                         f"mean head speed is closest to {target:.2f} deg/s, the {percentile}th percentile of the BOXRR-23 "
                         "test windows' speeds (an aggregate figure from the 13 Nov traces)",
            "selection_mean_speed_deg_s": speed, "target_window_speed_deg_s": target, "candidate_windows": candidates}
        trace, mean_err, mean_hold = mq.trace_document(q, first, steps, about, trace_id, label, "synthetic", percentile,
                                                       "tools/make_synthetic_data.py traces")
        mq.write_trace(trace)
        entries.append(mq.index_entry(trace))
        print(f"{trace_id}: activity {activity:.3f}, window t = {first}..{first + mq.RECORDED - 1} at {speed:.2f} deg/s "
              f"(target {target:.2f}); error {mean_err:.3f} deg, hold-last {mean_hold:.3f}", flush=True)
    mq.write_index(entries)


# ---------------------------------------------------------------- the synthetic pile: plan

PLAYER_SIZES = (406, 83, 20, 180, 1, 9, 4, 8, 7, 1, 5, 10, 3, 15, 12, 1, 1)   # recordings per player in the local folder
KEEN = {"qualified": 15, "no_long_segment": 4, "too_short_after_trim": 8, "motion_reject": 200, "rate_reject": 179}
FATES = ("excluded game", "too_short_after_trim", "rate_reject", "motion_reject", "no_long_segment", "qualified")
FATE_NAMES = ["no head in the file", "too short", "wrong sampling rate", "too little motion",
              "passes, but no 10 s piece", "kept"]
TILE_FIELDS = ["fate", "seconds", "rate_hz", "rows_at_time_zero", "repeated_timestamps", "backward_steps",
               "holes_over_100ms", "longest_hole_s", "turn_envelope_deg"]
DEVICES = [("HMD", "HEAD"), ("CONTROLLER", "HAND_LEFT"), ("CONTROLLER", "HAND_RIGHT")]
AXES = ["x", "y", "z", "i", "j", "k", "1"]
SEED = 2027


def truncated_lognormal(rng, n, low, high, median=155.4, sigma=0.55):
    out = []
    while len(out) < n:
        v = median * np.exp(rng.normal(0, sigma, 4 * n))
        out.extend(v[(v >= low) & (v <= high)].tolist())
    return np.round(np.array(out[:n]), 2)


def plan_pile(local):
    """One spec per recording, in path order (player, then recording number). The counts in `local` decide how many
    recordings get each fate, rate and defect; which recordings get them is random (seed SEED), except for the planted
    examples (`role`)."""
    rng = np.random.default_rng(SEED)
    specs = []
    for p, size in enumerate(PLAYER_SIZES):
        for _ in range(size):
            specs.append({"player": f"player-{p + 1:04d}", "recording": f"rec-{len(specs) + 1:04d}", "role": None,
                          "n0": None, "zero_first": False, "dups": 0, "freezes": [], "backs": [], "holes": [],
                          "flips": 0, "seed": SEED * 1000 + len(specs)})
    keen, others = specs[:PLAYER_SIZES[0]], specs[PLAYER_SIZES[0]:]
    fates = [f for f, n in KEEN.items() for _ in range(n)]
    rng.shuffle(fates)
    for s, f in zip(keen, fates):
        s["fate"] = f
    status = local["status"]
    rest = (["motion_reject"] * (status["motion_reject"] - KEEN["motion_reject"])
            + ["rate_reject"] * (status["rate_reject"] - KEEN["rate_reject"]))
    rng.shuffle(rest)
    rest.insert(int(rng.integers(0, PLAYER_SIZES[1])), "excluded game")          # the Tilt Brush file: player 2
    for s, f in zip(others, rest):
        s["fate"] = f
    for f, n in status.items():
        assert sum(s["fate"] == f for s in specs) == n, f

    # sampling rates: the folder's measured rates, in band for those that reach the motion filter
    rates = [int(k) for k, n in local["rates_hz_rounded"].items() for _ in range(n)]
    in_band = [r for r in rates if 85 <= r <= 110]
    out_band = [r for r in rates if not 85 <= r <= 110]
    rng.shuffle(in_band)
    rng.shuffle(out_band)
    for s in specs:
        s["game"] = "Tilt Brush" if s["fate"] == "excluded game" else "Beat Saber"
        if s["fate"] in ("qualified", "no_long_segment", "motion_reject"):
            s["rate"] = in_band.pop()
        elif s["fate"] == "rate_reject":
            s["rate"] = out_band.pop()
        else:
            s["rate"] = 90
        s["headset"] = "Oculus Quest2" if 87 <= s["rate"] <= 91 else "unknown"
    assert not in_band and not out_band

    def pick(pool, role, **need):
        choice = [s for s in pool if s["role"] is None and all(s[k] == v for k, v in need.items())]
        s = choice[int(rng.integers(len(choice)))]
        s["role"] = role
        return s

    decoded = [s for s in specs if s["fate"] != "excluded game"]
    raw = pick(keen, "raw", fate="qualified", rate=90)
    split = pick(keen, "split", fate="qualified")
    fill = pick(keen, "slerp_fill", fate="qualified")
    back = pick(keen, "backward", fate="qualified")
    freeze = pick(others, "freeze", fate="rate_reject", rate=80)
    flip = pick(others, "flip", fate="rate_reject", rate=72)
    longest = pick(others, "longest", fate="rate_reject", rate=72)

    # lengths (s, the last timestamp): the folder's shortest, median and longest exactly
    fixed = {raw["recording"]: 168.4, longest["recording"]: float(local["duration_s"]["longest"]), freeze["recording"]: 158.0}
    for s, d in zip([s for s in keen if s["fate"] == "too_short_after_trim"],
                    [local["duration_s"]["shortest"], 4.6, 4.2, 3.9, 3.5, 3.2, 2.9, 2.6]):
        fixed[s["recording"]] = d
    for s, d in zip([s for s in keen if s["fate"] == "no_long_segment"], [14.98, 14.4, 13.7, 12.9]):
        fixed[s["recording"]] = d
    median = float(local["duration_s"]["median"])
    free = [s for s in decoded if s["recording"] not in fixed]
    below = sum(v < median for v in fixed.values())
    need_below = (len(decoded) - 1) // 2 - below
    durations = np.r_[truncated_lognormal(rng, need_below, 20.0, median - 0.1), [median],
                      truncated_lognormal(rng, len(free) - need_below - 1, median + 0.1, 420.0)]
    rng.shuffle(durations)
    for s, d in zip(free, durations):
        if s["fate"] == "qualified" and abs(d - 170) < 4:      # keep the stage-1 recording the closest to 170 s
            d = d - 6 if d - 6 > median else d + 6
        s["duration"] = float(d)
    for s in decoded:
        s["duration"] = fixed.get(s["recording"], s.get("duration"))
    for s in specs:
        if s["fate"] == "excluded game":
            s["duration"] = 190.0

    # rows stamped 0 s at the start: median 19, as in the folder
    n_low, n_mid = (len(decoded) - 1) // 2 - 22, 45
    zeros = np.r_[rng.integers(2, 19, n_low), np.full(n_mid, 19), rng.integers(20, 90, len(decoded) - n_low - n_mid - 1)]
    rng.shuffle(zeros)
    raw["n0"] = 41
    for s, z in zip([s for s in decoded if s["n0"] is None], zeros):
        s["n0"] = int(z)
    # an empty first row
    raw["zero_first"] = True
    for s in rng.choice([s for s in decoded if s is not raw], local["first_quaternion_is_zero"] - 1, replace=False):
        s["zero_first"] = True

    plain = [s for s in decoded if s["role"] is None and s["fate"] in ("rate_reject", "motion_reject")]
    # repeated timestamps
    raw["dups"] = 1
    freeze["freezes"] = [3597, 2950, 2140, 1117]
    for s in rng.choice(plain, local["with_duplicate_timestamps"] - 2, replace=False):
        s["dups"] = int(1 + rng.geometric(0.3))
    # a clock that steps back
    back["backs"] = [(0.6, 0.597)]               # (at this timestamp, jump back this far): inside the 5 s cut
    for s in rng.choice(plain, local["with_backward_steps"] - 1, replace=False):
        b = float(rng.uniform(0.05, 1.5))
        s["backs"] = [(float(rng.uniform(b + 0.5, s["duration"] - 1.0)), b)]   # never below 0 s
    # holes over 0.1 s (the 0.092 s one of the SLERP example does not count)
    split["holes"] = [(split["duration"] - 4.6, 3.2)]
    fill["holes"] = [(fill["duration"] * 0.6, 0.0915 - 1.0 / fill["rate"])]           # the gap stays under 0.1 s
    longest["holes"] = [(200.0, float(local["longest_hole_s"]) - 1.0 / longest["rate"])]
    for s in rng.choice([s for s in plain if s is not longest], local["with_gap_over_100ms"] - 2, replace=False):
        k = int(rng.integers(1, 4))
        s["holes"] = [(float(rng.uniform(1.0, s["duration"] - 1.0)), float(np.clip(np.exp(rng.normal(-1.0, 1.0)), 0.12, 30.0)))
                      for _ in range(k)]
    # sign flips: one recording with a single flip after 5 s, the rest share the folder's total
    flip["flips"] = 1
    many = rng.choice([s for s in plain if s["duration"] > 60 and not s["backs"]], local["with_raw_sign_flips"] - 1,
                      replace=False)
    counts = np.maximum(2, np.round(np.exp(rng.normal(2.5, 1.4, len(many))))).astype(int)
    counts = np.maximum(2, np.round(counts * (local["raw_sign_flips"] - 1) / counts.sum())).astype(int)
    counts[np.argmax(counts)] += local["raw_sign_flips"] - 1 - counts.sum()
    for s, n in zip(many, counts):
        s["flips"] = int(n)

    # how lively the head is
    for s in specs:
        r = np.random.default_rng(s["seed"])
        if s["fate"] in ("qualified", "no_long_segment"):
            s["activity"] = float(r.uniform(1.5, 3.5))
        elif s["fate"] == "motion_reject":
            s["activity"] = float(np.exp(r.uniform(np.log(0.004), np.log(0.5))))
        else:
            s["activity"] = float(np.exp(r.uniform(np.log(0.02), np.log(4.0))))
    return specs


# ---------------------------------------------------------------- the synthetic pile: one recording

def timeline(spec, rng):
    """(true sample times, stamps) for the rows after the start-up rows. The head moves in true time; the stamps are what
    the file says: holes are missing time, a stuck clock repeats one stamp for many rows, a reset steps it back."""
    rate, duration = spec["rate"], spec["duration"]
    holes = sum(h for _, h in spec["holes"])
    backs = sum(b for _, b in spec["backs"])
    frozen = sum(r - 1 for r in spec["freezes"])
    steps = max(int(round((duration - holes + backs) * rate)), 3)
    n = steps + frozen
    dt = np.clip(rng.normal(1.0, 0.02, n), 0.3, 1.7) / rate
    true = np.cumsum(dt)
    for at, length in spec["holes"]:                         # a hole: time passes, no rows (`at` as a share of the rows)
        true[int(n * at / duration):] += length
    stamp = true.copy()
    lost = 0.0
    if spec["freezes"]:
        slot = int(0.8 * n) // len(spec["freezes"])
        for k, run in enumerate(spec["freezes"]):             # a stuck clock: `run` rows share one stamp
            assert run < slot - 50
            start = int(0.1 * n) + k * slot + int(rng.integers(0, slot - run - 50))
            stamp[start:start + run] = stamp[start]
            gap = true[start + run - 1] - true[start]
            stamp[start + run:] -= gap
            lost += gap
    for at, back in spec["backs"]:                            # a clock reset: later stamps step back
        i = max(int(np.searchsorted(stamp, at)), 1)
        stamp[i:] -= back
    if spec["dups"]:                                          # single repeated stamps
        for i in rng.choice(np.arange(int(5.5 * rate), n), min(spec["dups"], n - int(5.5 * rate)), replace=False):
            stamp[i] = stamp[i - 1]
    scale = duration / stamp.max()
    assert abs(scale - 1) < 0.01, (spec["recording"], scale)
    return true, stamp * scale


def frames_for(spec):
    """The recording's frame table [rows, 22]: time, then head, left hand, right hand (x, y, z, i, j, k, 1)."""
    rng = np.random.default_rng(spec["seed"])
    true, stamp = timeline(spec, rng)
    motion = HeadMotion(true[-1] + 1.0, spec["activity"], rng, noise_deg=0.04,
                        start=(rng.normal(0, 8), rng.normal(-6, 4), 0.0))
    angles = motion.angles(true)
    n0 = spec["n0"]
    start_up = angles[0] + rng.normal(0, (25.0, 12.0, 8.0), (n0, 3))   # the head jumps while the clock is at 0
    angles = np.r_[start_up, angles]
    q = quaternions(angles)
    if spec["flips"]:                                         # the file switches between q and -q
        if spec["role"] == "flip":
            rows = [n0 + int(16.0 * spec["rate"])]                # one flip, about 16 s in
        else:
            rows = np.sort(rng.choice(np.arange(n0 + 1, len(q)), spec["flips"], replace=False))
        sign = np.ones(len(q))
        for r in rows:
            sign[r:] *= -1
        q *= sign[:, None]
    t = np.r_[np.zeros(n0), stamp]
    tt = np.r_[true[0] - np.arange(n0, 0, -1) / spec["rate"], true]
    # positions: the head sways and moves with a lean; the hands hang in front of a body that turns more slowly
    head = np.c_[0.01 * np.sin(0.7 * tt) + 0.012 * angles[:, 2] * -1, 1.62 + 0.01 * np.sin(0.4 * tt), 0.02 * np.sin(0.3 * tt)]
    width = max(1, int(spec["rate"]))
    body = np.convolve(np.pad(angles[:, 0], width, mode="edge"), np.ones(2 * width + 1) / (2 * width + 1), "valid")
    swing = min(1.0, spec["activity"]) * 0.12
    columns = [t[:, None], head, q[:, [1, 2, 3, 0]]]
    for side, base, phase in (("left", (-0.22, -0.5, 0.32), 0.0), ("right", (0.24, -0.45, 0.3), 1.7)):
        local = np.c_[base[0] + swing * np.sin(1.3 * tt + phase), base[1] + swing * np.sin(1.9 * tt + phase),
                      base[2] + 0.5 * swing * np.cos(1.1 * tt + phase)]
        yaw = np.radians(body)
        world = head + np.c_[local[:, 0] * np.cos(yaw) + local[:, 2] * np.sin(yaw), local[:, 1],
                             -local[:, 0] * np.sin(yaw) + local[:, 2] * np.cos(yaw)]
        hand = quaternions(np.c_[body + 20 * np.sin(1.3 * tt + phase), -20 + 15 * np.sin(1.9 * tt), 10 * np.sin(tt + phase)])
        columns += [world, hand[:, [1, 2, 3, 0]]]
    frames = np.hstack(columns)
    if spec["zero_first"]:
        frames[0] = 0.0
    return frames


def pack(spec, frames):
    """The recording as an .xror file (bytes), as BOXRR-23 stores them."""
    x = XROR(id=spec["recording"])
    x.setApp(name=spec["game"])
    x.setUser(id=spec["player"])
    if spec["game"] == "Tilt Brush":
        x.addDevice(name="BRUSH", type="OTHER", axes=AXES + ["p"])
    else:
        x.addDevice(name=spec["headset"], type="HMD", joint="HEAD", axes=AXES)
        x.addDevice(name="Left Controller", type="CONTROLLER", joint="HAND_LEFT", axes=AXES)
        x.addDevice(name="Right Controller", type="CONTROLLER", joint="HAND_RIGHT", axes=AXES)
    x.data["frames"] = frames.astype(np.float32)
    return x.pack()


def brush_frames(spec):
    rng = np.random.default_rng(spec["seed"])
    n = int(spec["duration"] * spec["rate"])
    t = np.arange(n) / spec["rate"]
    pos = np.c_[0.3 * np.sin(0.9 * t), 1.2 + 0.2 * np.sin(1.4 * t), 0.4 + 0.1 * np.cos(0.6 * t)] + rng.normal(0, 0.002, (n, 3))
    rot = quaternions(np.c_[30 * np.sin(0.5 * t), 20 * np.sin(0.8 * t), 10 * np.sin(0.3 * t)])
    return np.c_[t, pos, rot[:, [1, 2, 3, 0]], 0.5 + 0.5 * np.sin(2.0 * t)]


def run(spec):
    """Make the recording, pack it, read it back and run the pipeline. Retries with a livelier or stiller head (new seed)
    until the pipeline's fate is the planned one."""
    for attempt in range(12):
        if spec["game"] == "Tilt Brush":
            frames = brush_frames(spec)
        else:
            frames = frames_for(spec)
        raw = pack(spec, frames)
        data = XROR.unpack(raw).data
        reason = unusable_reason(data["info"])
        if reason:
            row, segments, status = {"status": reason[0], "reason": list(reason)}, [], reason[0]
        else:
            row, segments = prepare_recording(data, spec["player"], spec["recording"])
            status = row["status"]
        if status == spec["fate"]:
            return {"spec": spec, "bytes": len(raw), "data": data, "row": row, "segments": segments, "status": status,
                    "frames": np.asarray(data["frames"], np.float64), "devices": devices_of(data)}
        spec["seed"] += 100_000
        if spec["fate"] in ("qualified", "no_long_segment"):
            spec["activity"] = min(spec["activity"] * 1.3, 6.0)
        elif spec["fate"] == "motion_reject":
            spec["activity"] *= 0.6
    raise AssertionError(f"{spec['recording']}: planned {spec['fate']}, the pipeline says {status} ({row})")


# ---------------------------------------------------------------- measuring a recording

def head_columns(data):
    offset = 1
    for d in data["info"]["hardware"]["devices"]:
        if d.get("joint") == "HEAD":
            return [offset + d["axes"].index(a) for a in ("i", "j", "k", "1")]
        offset += len(d["axes"])
    return None


def measure(rec):
    """The per-recording figures of the wall and of `local`, read from the file as the pipeline read it."""
    if rec["status"] == "excluded game":
        rec.update(tile=[0, None, None, 0, 0, 0, 0, 0.0, None])
        return rec
    f = rec["frames"]
    t = f[:, 0]
    cols = head_columns(rec["data"])
    q = f[:, cols][:, [3, 0, 1, 2]]
    norm = np.linalg.norm(q, axis=1)
    valid = np.isfinite(q).all(1) & (norm > 1e-8)
    step = np.diff(t)
    qv = q[valid]
    flips = int((np.sum(qv[1:] * qv[:-1], axis=1) < 0).sum())
    rows_zero = int((t == 0).sum())
    repeated = int(((step == 0) & (t[1:] > 0)).sum())
    holes = step[step > H5_MAX_GAP_S]
    turn = np.unwrap(np.radians(ttl(q[valid & (t > 0)])[:, 0]))
    turn = np.degrees(turn - np.median(turn))
    env = []
    for chunk in np.array_split(turn, 16):
        env += [int(round(chunk.min())), int(round(chunk.max()))] if len(chunk) else [0, 0]
    row = rec["row"]
    rate = row.get("source_rate_hz")
    fate = FATES.index(rec["status"])
    duration = float(t.max() - t.min())
    after = [int(np.flatnonzero(valid)[i]) for i in np.flatnonzero(np.sum(qv[1:] * qv[:-1], axis=1) < 0) + 1
             if t[valid][i] >= TRIM_S]
    filled = 0.0
    if rec["status"] == "qualified":
        d = np.diff(cleaned(rec)[0])
        filled = float(d[d <= H5_MAX_GAP_S].max())
    rec.update(t=t, q=q, valid=valid, flips=flips, flip_rows_after_cut=after, filled_step=filled,
               first_valid=bool(valid[0]), rows_zero=rows_zero, repeated=repeated, duration=duration,
               backward=int((step < 0).sum()), holes=len(holes), longest_hole=float(holes.max()) if len(holes) else 0.0,
               max_step=float(step.max()),
               tile=[fate, int(round(duration)), None if rate is None or rec["status"] == "too_short_after_trim" else round(rate * 2) / 2,
                     rows_zero, repeated, int((step < 0).sum()), len(holes), round(float(holes.max()), 1) if len(holes) else 0.0, env])
    return rec


HEAVY = ("data", "frames", "segments", "t", "q", "valid")


def light(rec):
    """The record without its arrays (the pile is too big to keep whole); full() makes them again."""
    return {k: v for k, v in rec.items() if k not in HEAVY}


def full(rec):
    """The same recording again, with its arrays: the spec holds the seed and activity that gave the planned fate."""
    again = measure(run(rec["spec"]))
    assert again["tile"] == rec["tile"] and again["bytes"] == rec["bytes"], rec["spec"]["recording"]
    return again


def local_counts(recs, recordings, users):
    """`local` recomputed from the pile."""
    decoded = [r for r in recs if r["status"] != "excluded game"]
    status = {}
    for r in recs:
        status[r["status"]] = status.get(r["status"], 0) + 1
    rates = {}
    for r in decoded:
        if "source_rate_hz" in r["row"]:
            k = str(int(round(r["row"]["source_rate_hz"])))
            rates[k] = rates.get(k, 0) + 1
    d = sorted(r["duration"] for r in decoded)
    return {"recordings": recordings, "users": users, "status": status, "decoded": len(decoded),
            "first_quaternion_is_zero": sum(not r["first_valid"] for r in decoded),
            "with_invalid_samples": sum(r["row"]["invalid_samples"] > 0 for r in decoded),
            "with_duplicate_timestamps": sum(r["repeated"] > 0 for r in decoded),
            "with_backward_steps": sum(r["backward"] > 0 for r in decoded),
            "with_gap_over_100ms": sum(r["holes"] > 0 for r in decoded),
            "with_raw_sign_flips": sum(r["flips"] > 0 for r in decoded),
            "raw_sign_flips": sum(r["flips"] for r in decoded),
            "with_several_rows_at_time_zero": sum(r["rows_zero"] > 1 for r in decoded),
            "median_rows_at_time_zero": float(np.median([r["rows_zero"] for r in decoded])),
            "longest_hole_s": round(max(r["max_step"] for r in decoded), 1),
            "duration_s": {"shortest": round(d[0], 1), "median": round(float(np.median(d)), 1), "longest": round(d[-1], 1)},
            "rates_hz_rounded": dict(sorted(rates.items(), key=lambda kv: int(kv[0])))}


# ---------------------------------------------------------------- the examples

def r4(v, digits=4):
    return np.round(np.asarray(v, np.float64), digits).tolist()


def summary(rec, chosen):
    s, row = rec["spec"], rec["row"]
    out = {"user": s["player"], "recording": s["recording"], "chosen": chosen, "status": rec["status"]}
    for k in ("raw_samples", "invalid_samples", "backwards_steps", "gaps_over_100ms", "trimmed_samples",
              "duplicate_samples", "source_rate_hz", "sweep_deg", "velocity_deg_s", "motion_windows"):
        if row.get(k) is not None:
            out[k] = row[k]
    out["headset"] = row.get("headset")
    out["duration_s"] = rec["duration"]
    if "segments" in row:
        out["segments"] = [g["samples"] for g in row["segments"]]
        out["short_segments_dropped"] = row["short_segments_dropped"]
    return out


def cleaned(rec):
    """The pipeline's cleaned samples (clean() on the head columns), xyzw."""
    f = rec["frames"]
    t, q, _ = clean(f[:, 0], f[:, head_columns(rec["data"])], {})
    return t, q


def pieces(t):
    """resample()'s pieces, kept or not: (first, last source index, grid samples)."""
    dt = np.diff(t)
    cuts = np.r_[0, np.flatnonzero((dt <= 0) | (dt > H5_MAX_GAP_S)) + 1, len(t)]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 2:
            continue
        tt = t[a:b]
        grid = tt[0] + np.arange(int(np.floor((tt[-1] - tt[0]) * H5_RATE_HZ)) + 1) / H5_RATE_HZ
        out.append((a, b, int((grid <= tt[-1]).sum())))
    return out


def resample_view(rec, centre, half, chosen):
    t, q = cleaned(rec)
    lo, hi = centre - half, centre + half
    inside = (t >= lo) & (t <= hi)
    after = []
    for grid, qw in [(g, s) for g, s in rec["segments"]]:
        k = (grid >= lo) & (grid <= hi)
        if k.any():
            after.append({"t": r4(grid[k]), "q_wxyz": r4(qw[k])})
    parts = [{"t0": round(float(t[a]), 3), "t1": round(float(t[b - 1]), 3), "source_samples": int(b - a), "samples": n,
              "kept": n >= H5_MIN_SAMPLES} for a, b, n in pieces(t)]
    assert [p["samples"] for p in parts if p["kept"]] == [len(g) for g, _ in rec["segments"]]
    return {**summary(rec, chosen), "centre_s": float(centre),
            "before": {"t": r4(t[inside]), "q_wxyz": r4(q[inside][:, [3, 0, 1, 2]])}, "after": after, "pieces": parts,
            "cleaned_span_s": [round(float(t[0]), 3), round(float(t[-1]), 3)],
            "source_samples_in_view": int(inside.sum()), "output_samples_in_view": int(sum(len(a["t"]) for a in after))}


def windows_of(t, q):
    """motion_filter's 5 s windows one by one (its code, unchanged, but keeping each window): [(start, sweep, velocity)]."""
    out = []
    cuts = np.r_[0, np.flatnonzero(np.diff(t) <= 0) + 1, len(t)]
    low, high = SWEEP_PERCENTILES
    for begin, end in zip(cuts[:-1], cuts[1:]):
        tt, qq = t[begin:end], q[begin:end]
        if len(tt) < MIN_WINDOW_SAMPLES:
            continue
        angles = Rotation.from_quat(qq).as_euler(EULER_ORDER, degrees=True)
        first = t[0] + np.ceil((tt[0] - t[0]) / WINDOW_S) * WINDOW_S
        for start in np.arange(first, tt[-1] - WINDOW_S + 1e-9, WINDOW_S):
            keep = (tt >= start) & (tt < start + WINDOW_S)
            times, a = tt[keep], angles[keep]
            if len(times) < MIN_WINDOW_SAMPLES or times[-1] <= times[0]:
                continue
            d = np.abs(np.diff(a, axis=0))
            out.append((float(start), np.percentile(a, high, axis=0) - np.percentile(a, low, axis=0),
                        np.minimum(d, 360 - d).sum(axis=0) / WINDOW_S))
    return out


def motion_example(rec, chosen):
    t, q = cleaned(rec)
    w = windows_of(t, q)
    assert np.allclose(np.median([x[1] for x in w], 0), rec["row"]["sweep_xyz_deg"])
    end = w[0][0] + 3 * WINDOW_S
    rows = np.flatnonzero((t >= w[0][0]) & (t < end))[::3]
    return {**summary(rec, chosen), "t": r4(t[rows]),
            "euler_xyz_deg": r4(Rotation.from_quat(q[rows]).as_euler(EULER_ORDER, degrees=True), 2), "drawn_every": 3,
            "q_wxyz": r4(q[rows][:, [3, 0, 1, 2]], 3),
            "windows": [{"start_s": round(s, 3), "sweep_xyz_deg": r4(sw, 2), "velocity_xyz_deg_s": r4(v, 2)} for s, sw, v in w[:3]],
            "sweep_xyz_deg": r4(rec["row"]["sweep_xyz_deg"], 2), "velocity_xyz_deg_s": r4(rec["row"]["velocity_xyz_deg_s"], 2)}


def devices_of(data):
    return [{"name": d.get("name"), "type": d.get("type"), "joint": d.get("joint"), "axes": d["axes"]}
            for d in data["info"]["hardware"]["devices"]]


def hash_order(player, recordings):
    """ray/code/h5_export.py recording_order: sha256('<seed>:<player>/<recording>.xror')."""
    return sorted(recordings, key=lambda r: hashlib.sha256(f"{H5_SEED}:{player}/{r}.xror".encode()).hexdigest())


def build_examples(recs, by_id):
    def first(pool, key=None, reverse=False):
        pool = list(pool)
        return sorted(pool, key=key, reverse=reverse)[0] if key else pool[0]

    E = {}
    qualified = [r for r in recs if r["status"] == "qualified"]
    raw = full(first(qualified, key=lambda r: abs(r["duration"] - 170)))
    assert raw["spec"]["role"] == "raw"
    chosen_raw = f"qualified recording closest to 170 s long (of {len(qualified)} synthetic recordings with status 'qualified')"
    t, q, f = raw["t"], raw["q"], raw["frames"]
    n8 = int(np.argmax(t >= 8.0))
    E["raw"] = {**summary(raw, chosen_raw), "file_bytes": raw["bytes"], "game": raw["spec"]["game"], "devices": raw["devices"],
                "columns": int(f.shape[1]), "frames": int(f.shape[0]), "head_columns": head_columns(raw["data"]),
                "first_rows": r4(f[:3]), "t": r4(t[:n8]), "q_wxyz": r4(q[:n8]),
                "quat_norm_first": r4(np.linalg.norm(q[:3], axis=1)), "trim_s": TRIM_S}

    ct, cq = cleaned(raw)
    start = float(round((ct[0] + ct[-1]) / 2) - 6)
    rows = np.flatnonzero((t >= start) & (t < start + 12))
    one_q = np.round(q[rows], 4)
    rot = Rotation.from_quat(one_q[:5][:, [1, 2, 3, 0]])
    fwd, right, up = (rot.apply(np.tile(v, (5, 1))) for v in ((0, 0, 1), (1, 0, 0), (0, 1, 0)))
    check = np.degrees(np.c_[np.arctan2(fwd[:, 0], fwd[:, 2]), np.arcsin(fwd[:, 1]), np.arctan2(right[:, 1], up[:, 1])])
    assert np.allclose(check, ttl(one_q[:5]), atol=1e-6)
    E["one"] = {**summary(raw, chosen_raw), "start_s": start, "seconds": 12.0, "t": r4(t[rows]), "q_wxyz": one_q.tolist(),
                "check_turn_tilt_lean_deg": r4(check, 3)}

    keep = t >= ct[0]
    hq = Rotation.from_quat(q[keep & raw["valid"]][:, [1, 2, 3, 0]]).inv()
    head = f[keep & raw["valid"], 1:4]
    hands = [np.median(hq.apply(f[keep & raw["valid"], c:c + 3] - head), 0) for c in (8, 15)]
    E["frame_check"] = {"user": raw["spec"]["player"], "recording": raw["spec"]["recording"],
                        "left_hand_xyz_m": r4(hands[0], 3), "right_hand_xyz_m": r4(hands[1], 3),
                        "meaning": "median hand position in the head's frame: x right, y up, z forward"}
    E["grid"] = resample_view(raw, ct[0] + 10.0, 0.3, chosen_raw)

    excluded = [r for r in recs if r["status"] == "excluded game"]
    x = excluded[0]
    E["excluded"] = {"user": x["spec"]["player"], "recording": x["spec"]["recording"],
                     "chosen": f"first by path (of {len(excluded)} synthetic recordings with status 'excluded game')",
                     "status": "excluded game", "file_bytes": x["bytes"], "game": x["spec"]["game"],
                     "reason": x["row"]["reason"], "devices": x["devices"]}

    short = [r for r in recs if r["status"] == "too_short_after_trim"]
    s = full(first(short, key=lambda r: r["duration"], reverse=True))
    E["too_short"] = {**summary(s, f"the longest one (of {len(short)} synthetic recordings with status 'too_short_after_trim')"),
                      "t": r4(s["t"]), "q_wxyz": r4(s["q"]), "trim_s": TRIM_S, "min_assessed_samples": MIN_ASSESSED_SAMPLES}

    decoded = [r for r in recs if r["status"] != "excluded game"]
    fz = full(first(decoded, key=lambda r: r["repeated"], reverse=True))
    assert fz["spec"]["role"] == "freeze"
    t, q = fz["t"], fz["q"]
    same = np.r_[False, (np.diff(t) == 0) & (t[1:] > 0)]
    runs, i = [], 1
    while i < len(t):
        if same[i]:
            j = i
            while j + 1 < len(t) and same[j + 1]:
                j += 1
            runs.append((i - 1, j))
            i = j + 1
        else:
            i += 1
    a, b = max(runs, key=lambda r: r[1] - r[0])
    view = np.arange(max(0, a - 400), min(len(t), b + 400), 8)
    turn = np.degrees(np.unwrap(np.radians(ttl(q[a:b + 1])[:, 0])))
    positive = np.diff(t)[np.diff(t) > 0]
    E["freeze"] = {**summary(fz, f"most duplicate timestamps of the {len(decoded)} decoded synthetic recordings"),
                   "first_row": int(a), "last_row": int(b), "rows_in_run": int(b - a + 1), "timestamp_s": round(float(t[a]), 4),
                   "next_timestamp_s": round(float(t[b + 1]), 4), "turn_span_deg": round(float(turn.max() - turn.min()), 1),
                   "runs_over_100_rows": sum(e - s0 + 1 > 100 for s0, e in runs),
                   "usual_step_ms": round(float(np.median(positive)) * 1000, 1), "drawn_every": 8, "row": view.tolist(),
                   "t": r4(t[view]), "q_wxyz": r4(q[view], 3), "kept": [int(not same[i]) for i in view]}

    E["rate"] = {}
    for key, lo, hi, words in ((72, 71, 73, "71-73"), (80, 79, 81, "79-81"), (90, 89.5, 90.5, "89.5-90.5"), (144, 143, 147, "143-147")):
        pool = [r for r in decoded if lo <= r["row"].get("source_rate_hz", 0) <= hi and r["row"]["raw_samples"] > 2000]
        r = full(pool[0])
        ct, _ = cleaned(r)
        E["rate"][str(key)] = {**summary(r, f"first by path of {len(pool)} synthetic recordings with a median rate of {words} Hz and over 2,000 samples"),
                               "dt_ms": r4(np.diff(ct[:301]) * 1000, 3)}

    rejects = [r for r in recs if r["status"] == "motion_reject" and r["row"]["motion_windows"] >= 3]
    E["still"] = motion_example(full(first(rejects, key=lambda r: r["row"]["sweep_deg"])),
                                f"the smallest sweep among those with at least 3 windows (of {len(rejects)} synthetic recordings with status 'motion_reject')")
    E["moving"] = motion_example(full(first(qualified, key=lambda r: r["row"]["sweep_deg"], reverse=True)),
                                 f"the largest sweep (of {len(qualified)} synthetic recordings with status 'qualified')")

    fill = full(first(qualified, key=lambda r: r["filled_step"], reverse=True))
    assert fill["spec"]["role"] == "slerp_fill"
    ct, _ = cleaned(fill)
    j = int(np.argmax(np.diff(ct)))
    E["slerp_fill"] = resample_view(fill, (ct[j] + ct[j + 1]) / 2, 0.25,
                                    f"the largest gap that is still SLERP-filled (at most 0.10 s) (of {len(qualified)} synthetic recordings with status 'qualified')")
    splits = [r for r in qualified if r["row"]["gaps_over_100ms"] and r["row"]["short_segments_dropped"]]
    sp = full(splits[0])
    assert sp["spec"]["role"] == "split"
    ct, _ = cleaned(sp)
    j = int(np.argmax(np.diff(ct)))
    E["split"] = resample_view(sp, (ct[j] + ct[j + 1]) / 2, 2.5,
                               f"qualified, with a gap over 0.10 s and a piece dropped as too short (of {len(splits)} synthetic recordings like that)")
    nls = [r for r in recs if r["status"] == "no_long_segment"]
    nl = full(first(nls, key=lambda r: r["duration"], reverse=True))
    ct, _ = cleaned(nl)
    E["no_long_segment"] = resample_view(nl, ct[0] + 10.0, 0.3, f"the longest one (of {len(nls)} synthetic recordings with status 'no_long_segment')")

    flipped = [r for r in decoded if r["flip_rows_after_cut"]]
    fl = full(first(flipped, key=lambda r: r["flips"]))
    assert fl["spec"]["role"] == "flip"
    i = fl["flip_rows_after_cut"][0]
    view = np.arange(i - 42, i + 44)
    before = fl["q"][view]
    inview = [int(k) for k in np.flatnonzero(np.sum(before[1:] * before[:-1], axis=1) < 0) + 1]
    E["flip"] = {**summary(fl, f"fewest sign flips of the {len(flipped)} synthetic recordings that have one after the first 5 s; its first such flip"),
                 "flips_in_recording": fl["flips"], "flip_time_s": round(float(fl["t"][i]), 4), "flips_in_view": len(inview),
                 "flip_rows_in_view": inview, "head_turned_across_flip_deg": round(float(geodesic_degrees(fl["q"][i - 1], fl["q"][i])), 3),
                 "t": r4(fl["t"][view]), "before_wxyz": r4(before), "after_wxyz": r4(align(before))}

    backs = sorted([r for r in decoded if r["backward"]], key=lambda r: (r["status"] != "qualified", r["spec"]["recording"]))
    bk = full(backs[0])
    assert bk["spec"]["role"] == "backward"
    t, step = bk["t"], np.diff(bk["t"])
    at = int(np.argmax(step < 0)) + 1
    lo = max(0, at - 61)
    view = np.arange(lo, min(len(t), lo + 160))
    E["backward"] = {**summary(bk, f"first by path, qualified ones first, of the {len(backs)} synthetic recordings whose clock steps backwards"),
                     "steps": {"zero": int((step == 0).sum()), "negative": int((step < 0).sum()),
                               "over_100ms": int((step > H5_MAX_GAP_S).sum()), "all": int(len(step))},
                     "backward": {"first_index": int(lo), "t": r4(t[view], 5), "step_s": round(float(step[at - 1]), 4),
                                  "at_row": at, "q_wxyz": r4(bk["q"][view])}}

    players = sorted({r["spec"]["player"] for r in recs})
    passed = {p: [r for r in recs if r["spec"]["player"] == p and r["status"] in ("qualified", "no_long_segment")] for p in players}
    player = max(players, key=lambda p: len(passed[p]))
    mine = [r for r in recs if r["spec"]["player"] == player]
    counts = {k: sum(r["status"] == k for r in mine) for k in ("motion_reject", "rate_reject", "no_long_segment", "too_short_after_trim", "qualified")}
    order, accepted = [], 0
    for rid in hash_order(player, [r["spec"]["recording"] for r in passed[player]]):
        r = by_id[rid]
        if accepted >= H5_MAX_RECORDINGS_PER_USER:
            order.append({"recording": rid, "status": "not_inspected_user_cap", "kept": False, "samples": None, "source_rate_hz": None})
            continue
        kept = r["status"] == "qualified"
        accepted += kept
        order.append({"recording": rid, "status": r["status"], "kept": kept,
                      "samples": r["row"]["retained_samples"] if kept else None, "source_rate_hz": r["row"]["source_rate_hz"]})
    E["user"] = {"user": player, "chosen": "the synthetic player with the most passing recordings", "recordings_in_tar": len(mine),
                 "status_counts": counts, "passed_filter": len(passed[player]), "cap": H5_MAX_RECORDINGS_PER_USER, "order": order}
    E["window"] = questset_window()
    return E, raw


def questset_window():
    """Stage 9: the input and answer of the first call traces/test50.js scores (the QuestSet player)."""
    text = (SITE / "traces/test50.js").read_text(encoding="utf-8")
    trace = json.loads(text[text.index("= {") + 2:].rstrip(";"))
    m = trace["metadata"]
    assert m["source"] == "questset", "traces/test50.js is not the QuestSet player: run make_questset_data.py first"
    first = m["first_recorded_t"]
    by_t = {s["t"]: s for s in trace["steps"]}
    q = [by_t[t]["input_quaternion"] for t in range(first - mq.C + 1, first + 1)] + by_t[first]["ground_truth"]
    start = first - mq.C + 1
    prep = m["preparation_record"]
    return {"h5_path": None, "user": m["user"], "recording": m["recording"], "segment": m["segment"],
            "start_sample": start, "segment_samples": m["segment_samples"],
            "chosen": "not synthetic: the input and answer of the first call the Model side's test50 trace scores "
                      f"({m['label']}, {m['dataset']}, {m['licence']}), so both sides show the same window",
            "t0_s": round(m["segment_start_time_s"] + start / H5_RATE_HZ, 6), "dt_s": 1 / H5_RATE_HZ,
            "attrs": {"app": m["game"], "dataset": m["dataset"], "challenge_filter_pass": m["challenge_filter_pass"],
                      "source_rate_hz": prep["source_rate_hz"], "sweep_deg": prep["sweep_deg"],
                      "velocity_deg_s": prep["velocity_deg_s"], "preparation": m["preparation"]},
            "q_wxyz": r4(q)}


def make_examples():
    text = EXAMPLES.read_text(encoding="utf-8")
    header, body = text[:text.index(EXAMPLES_PREFIX)], text[text.index(EXAMPLES_PREFIX) + len(EXAMPLES_PREFIX):]
    old = json.loads(body.rstrip().rstrip(";"))
    local = old["local"]
    specs = plan_pile(local)
    recs = []
    for i, spec in enumerate(specs):
        recs.append(light(measure(run(spec))))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} of {len(specs)} recordings", flush=True)
    by_id = {r["spec"]["recording"]: r for r in recs}

    again = local_counts(recs, len(specs), len(PLAYER_SIZES))
    differ = {k: (v, again.get(k)) for k, v in local.items() if k != "source" and again.get(k) != v}
    assert not differ, f"the synthetic pile does not match the folder's counts: {differ}"

    examples, raw = build_examples(recs, by_id)
    tiles = [r["tile"] for r in recs]
    starts = np.r_[0, np.cumsum(PLAYER_SIZES)[:-1]].tolist()
    wall = {"fates": FATE_NAMES, "fields": TILE_FIELDS, "tiles": tiles, "user_starts": starts,
            "example": list(by_id).index(raw["spec"]["recording"]), "fate_counts": [sum(t[0] == f for t in tiles) for f in range(6)]}
    assert wall["fate_counts"] == [local["status"].get(f, 0) for f in FATES]

    # where the kept counts came from: written once, from the last file Nic's scripts made, and carried over on reruns
    counts_from = old.get("counts_from") or f"the file of {old['made']}, made by {old['made_by']}"
    out = {"made": TODAY, "made_by": "tools/make_synthetic_data.py examples (examples and wall: synthetic recordings; "
                                     "examples.window: the QuestSet player of traces/test50.js); config, pull, final and local: "
                                     "kept as they were (see counts_from)",
           "counts_from": counts_from,
           "synthetic": {"generator": "tools/make_synthetic_data.py", "motion": "tools/synthetic_motion.py", "seed": SEED,
                         "note": "The recordings in `examples` and `wall` are synthetic: made to the counts in `local` "
                                 "and run through the team's pipeline (ray/code). No BOXRR-23 recording is in this file."},
           **{k: old[k] for k in ("config", "pull", "final", "local")}, "wall": wall, "examples": examples}
    header = ("// Generated by tools/make_synthetic_data.py examples: do not edit. Synthetic recordings (examples, wall), "
              "real counts (config, pull, final, local).\n")
    EXAMPLES.write_text(header + EXAMPLES_PREFIX + json.dumps(out, allow_nan=False) + ";", encoding="utf-8")
    print(f"examples.js: {len(recs)} synthetic recordings, fates {wall['fate_counts']}; stage-1 recording "
          f"{raw['spec']['recording']} ({raw['duration']:.1f} s); every count in `local` reproduced")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=("traces", "examples"))
    args = parser.parse_args(argv)
    (make_traces if args.what == "traces" else make_examples)()


if __name__ == "__main__":
    main()
