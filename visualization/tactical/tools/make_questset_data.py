"""make_questset_data.py: the site's replay data from a QuestSet session instead of BOXRR-23 (20 Nov 2026).

Why: BOXRR-23's data use agreement (clause 4) does not allow passing its recordings on without Berkeley's written
consent, so a public site cannot replay BOXRR players. This script makes the same files from QuestSet (Baldoni et al.,
2024, CC BY 4.0, https://researchdata.cab.unipd.it/1239/): Meta Quest players the model has never seen.

It writes, in the formats of Nic's tools/make_trace_final.py, tools/make_hero_clip.py and tools/make_story_figs.mjs
(kept in his working repo):
  traces/test50.js     the model explorer's trace: 792 warm-up calls, then 300 recorded calls with every activation
  hero/clip.js         the hero's 12 s clip: the last 1,200 samples of the trace window
  figs/challenge.svg   the story's "4 seconds in, 200 ms out" figure: one recorded call's input and its answer
and puts the trace's entry first in traces/index.js (replacing the old test50 entry; the others are left alone).
The trace and clip formats were checked on 20 Nov by regenerating the 13 Nov BOXRR trace and clip from the final HDF5
with these functions. Keys, order, shapes, inputs, ground truth, hold errors, weights, the clip's poses and every clip
statistic came out identical. Model outputs differed only in float32 last bits (127 of 1.2 M activations and some
predictions moved by one unit in the last stored decimal), as two different CPUs give.

Run from the repo root, in the bench's environment (bench/requirements.txt) plus scipy:
  python visualization/tactical/tools/make_questset_data.py PATH/group1_order1_user0_fast_movement.csv
tools/make_synthetic_data.py imports the model functions below for its synthetic traces.

Steps
1. Frame. QuestSet's head pose is head -> world in a right-handed frame (+x right, +y up, -z forward): in the head
   frame the right controller is at +x and both controllers at -z. Converted with the submission README's rule for
   that frame, q_ours = (w, -x, -y, z).
2. Preparation: ray/code/cleaning.py's clean (5 s trim, invalid samples, duplicate timestamps), motion filter and
   resample (SLERP onto an exact 100 Hz grid). QuestSet logs at about 59 Hz, below the pipeline's 85-110 Hz band:
   that band check is the one step skipped.
3. Selection, from motion only and never from error (the rule of the BOXRR traces): of all 300-anchor windows with
   792 calls of history before them, the one whose mean head speed is closest to the median over those windows.
   The figure's call: of the 300 recorded calls, the one whose head turn over the next 200 ms is closest to the
   median of those turns.
4. Model: Submission/Motion_Pending_headpose/model_fp32.onnx, one call per sample. The trace starts from zero states
   792 calls before its first recorded anchor; the clip streams from the segment's first sample (bench protocol
   from-start). Every prediction and state is checked bitwise against bench.drivers.OnnxModel.
5. Checks before writing (it stops with an AssertionError and writes nothing): the clip's 300 last predictions, streamed
   from the segment start, are bitwise equal to the trace's, which start cold 792 calls earlier; no sign flip between
   neighbouring samples in q or pred; pred in the same hemisphere as q; every error finite and within 0..180 deg.
"""
import argparse
import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

SITE = Path(__file__).resolve().parents[1]  # visualization/tactical
REPO = SITE.parents[1]
sys.path.insert(0, str(REPO))
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from onnx import numpy_helper  # noqa: E402

from bench.drivers import OnnxModel  # noqa: E402
from bench.geometry import geodesic_degrees, renormalise  # noqa: E402
from ray.code.cleaning import clean, motion_filter, passes, resample  # noqa: E402

MODEL = REPO / "Submission/Motion_Pending_headpose/model_fp32.onnx"
C, H = 400, 20                         # context samples, forecast steps
WARMUP, RECORDED, CLIP = 792, 300, 1200
TENSORS = {"features": "/Concat_output_0", "projection_relu": "/Relu_1_output_0", "block_0": "/Relu_3_output_0",
           "block_1": "/Relu_5_output_0", "block_2": "/Relu_7_output_0", "head_coefficients": "/Reshape_5_output_0",
           "trajectory_r": "/MatMul_8_output_0"}
ACTIVATIONS = ("features", "projection_relu", "block_0", "block_1", "block_2")
HEAD_BIAS = "/Constant_53_output_0"    # learned bias of the x coefficients, concatenated after /Reshape_5
GAMES = {("1", "fast"): "Beat Saber", ("1", "slow"): "Cooking Simulator",
         ("2", "fast"): "Medal of Honor", ("2", "slow"): "Forklift Simulator"}
CREDIT = ("QuestSet: Baldoni, S. et al. (2024), Research Data Unipd, https://researchdata.cab.unipd.it/1239/, "
          "licensed CC BY 4.0. Changed: head orientation only, converted to the model's frame and resampled to 100 Hz.")
INDEX = SITE / "traces/index.js"
INDEX_PREFIX = "window.TRACE_INDEX = "


def rounded(values, digits):
    return np.round(np.asarray(values, dtype=np.float64), digits).tolist()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def speeds(q):
    """Head speed in deg/s between neighbouring samples: element i is the step from sample i to i + 1."""
    return geodesic_degrees(q[1:], q[:-1]) * 100.0


# ---------------------------------------------------------------- data

def load_session(path):
    """One QuestSet movement CSV -> (times, unit float64 quaternions wxyz in our frame, preparation record)."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    column = {name: data[:, i] for i, name in enumerate(header)}
    w, x, y, z = (column[f"HeadOrientation{k}"] for k in "WXYZ")
    record = {}
    t, q_xyzw, status = clean(column["time"], np.stack([-x, -y, z, w], 1), record)  # ours, as xyzw
    if status not in (None, "rate_reject"):
        raise ValueError(f"{path.name}: {status}")
    record.update(motion_filter(t, q_xyzw))
    record["challenge_filter_pass"] = bool(passes(record))
    segments = resample(t, q_xyzw, record)
    times, q32 = max(segments, key=lambda s: len(s[1]))
    q = q32.astype(np.float64)
    return times, q / np.linalg.norm(q, axis=1, keepdims=True), record


def select_window(q, target=None):
    """First recorded anchor of the 300-anchor window whose mean speed is closest to `target` (deg/s), by default the
    median window speed. Returns (first anchor, its mean speed, the target, number of candidate windows)."""
    steps = np.r_[0.0, np.cumsum(speeds(q))]
    starts = np.arange(C - 1 + WARMUP, len(q) - H - RECORDED + 1)
    means = (steps[starts + RECORDED - 1] - steps[starts]) / (RECORDED - 1)
    target = float(np.percentile(means, 50)) if target is None else float(target)
    pick = int(np.argmin(np.abs(means - target)))
    return int(starts[pick]), float(means[pick]), target, len(starts)


# ---------------------------------------------------------------- model

def session_with(names):
    model = onnx.shape_inference.infer_shapes(onnx.load(str(MODEL)))
    infos = {v.name: v for v in model.graph.value_info}
    model.graph.output.extend(infos[n] for n in names)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def constant(name):
    for node in onnx.load(str(MODEL)).graph.node:
        if node.op_type == "Constant" and node.output[0] == name:
            return numpy_helper.to_array(node.attribute[0].t)
    raise KeyError(name)


def weights():
    """Every initializer, exact float32, with size-1 axes dropped; plus the head's x bias (a Constant in the graph)."""
    out = {}
    for init in onnx.load(str(MODEL)).graph.initializer:
        value = np.squeeze(numpy_helper.to_array(init))
        out[init.name] = {"shape": list(value.shape), "values": value.tolist()}
    bias = constant(HEAD_BIAS)[0, :, 0]
    out["head_bias_x"] = {"shape": [3], "values": bias.tolist(), "source": "ONNX /Constant_53; only x coefficients learned"}
    return out


def trace_steps(q, times, first_recorded):
    """The trace's steps (zero states at the first call, 792 warm-up calls, then 300 recorded calls) and the recorded
    calls' raw predictions [300, 20, 4]."""
    q32 = q.astype(np.float32)
    plain, session = OnnxModel(MODEL, threads=1), session_with(list(TENSORS.values()))
    outputs, bias = [o.name for o in session.get_outputs()], constant(HEAD_BIAS)
    state = {i: np.zeros(shape, np.float32) for i, _, shape in plain.states}
    steps, raws = [], []
    for t in range(first_recorded - WARMUP, first_recorded + RECORDED):
        window = q32[t - C + 1:t + 1][None]
        raw = plain.step(window)
        result = dict(zip(outputs, session.run(outputs, {plain.context_name: window, **state})))
        if not np.array_equal(result[plain.prediction_name], raw):
            raise AssertionError(f"t={t}: the instrumented graph's prediction differs from the bench driver's")
        for i, o, _ in plain.states:
            if not np.array_equal(result[o], plain.state[i]):
                raise AssertionError(f"t={t}: the instrumented graph's state {o} differs from the bench driver's")
            state[i] = result[o]
        step = {"t": t, "warmup": t < first_recorded, "time_s": float(times[t]), "input_quaternion": rounded(q32[t], 7)}
        step.update({k: rounded(result[TENSORS[k]], 4) for k in ACTIVATIONS})
        if t >= first_recorded:
            raws.append(raw[0])
            truth = q[t + 1:t + H + 1]
            step.update({
                "head_coefficients": rounded(result[TENSORS["head_coefficients"]] + bias, 4),
                "trajectory_r": rounded(result[TENSORS["trajectory_r"]], 4),
                "prediction": rounded(raw, 7),
                "ground_truth": rounded(truth, 7),
                "error_step20_deg": float(geodesic_degrees(renormalise(raw[0, -1]), truth[-1])),
                "hold_error_step20_deg": float(geodesic_degrees(q[t], truth[-1]))})
        steps.append(step)
    return steps, np.stack(raws)


def trace_document(q, first, steps, about, trace_id, label, source, percentile, generator):
    """The viewer trace for the recorded anchors first..first + 299: steps from trace_steps(), `about` = the fields
    that describe the data (they go right after the model's). Returns (trace, mean error, mean hold-last error)."""
    last = first + RECORDED - 1
    recorded = [s for s in steps if not s["warmup"]]
    mean_err = float(np.mean([s["error_step20_deg"] for s in recorded]))
    mean_hold = float(np.mean([s["hold_error_step20_deg"] for s in recorded]))
    for s in recorded:  # rounded only after the means, which use the unrounded errors
        s["error_step20_deg"], s["hold_error_step20_deg"] = round(s["error_step20_deg"], 4), round(s["hold_error_step20_deg"], 4)
    metadata = {
        "model": "Submission/Motion_Pending_headpose/model_fp32.onnx", "model_sha256": sha256(MODEL), **about,
        "protocol": "cold start 792 calls before the first scored anchor; scored outputs identical to bench from-start",
        "input_normalisation": "float64 divide by norm, then float32 for graph input; ground truth stays unit float64 "
                               "like bench/scoring.py",
        "warm_columns_from": 791,
        "warm_columns_from_definition": "Zero-based adapter frame index, including 399 initial_context pre-call frames; "
                                        "all block outputs from this frame have fully warm causal input history",
        "fully_warm_call_from": 392, "fully_warm_anchor_from": first - WARMUP + 392,
        "warmup_calls": WARMUP, "recorded_calls": RECORDED, "total_calls": WARMUP + RECORDED, "steps_includes_warmup": True,
        "first_call_t": first - WARMUP, "first_recorded_t": first, "last_recorded_t": last,
        "t_definition": "Zero-based anchor index in the resampled segment; truth rows are q[t+1] through q[t+20]",
        "sample_rate_hz": 100, "quaternion_order": "wxyz", "context_samples": C, "horizon_samples": H,
        "activation_decimal_places": 4, "weights_precision": "Exact float32 initializer values serialized as JSON "
                                                             "numbers; not rounded",
        "mean_step20_error_deg": mean_err,
        "error_note": "Computed on unrounded predictions and truth in float64; this selected stretch is not a benchmark score",
        "verification": {"prediction_bitwise_equal_all_calls": True, "state_bitwise_equal_all_calls": True,
                         "compared_calls": WARMUP + RECORDED, "graph_optimization": "ORT_ENABLE_ALL",
                         "provider": "CPUExecutionProvider", "onnxruntime_version": ort.__version__},
        "tensor_sources": {**TENSORS, "head_coefficients": "/Reshape_5_output_0 + /Constant_53 (x bias)"},
        "tensor_shapes": {"features": [1, 1, 7], "projection_relu": [1, 32, 2, 1], "block_0": [1, 32, 2, 1],
                          "block_1": [1, 32, 2, 1], "block_2": [1, 32, 2, 1], "head_coefficients": [1, 3, 3],
                          "trajectory_r": [1, 20, 3]},
        "feature_order": ["vx", "vy", "vz", "L1_speed", "gx", "gy", "gz"],
        "head_note": "head_coefficients includes learned x bias; trajectory graph itself is unchanged",
        "state_note": "Caches are past block inputs, are fed back each call, and are not dumped",
        "trace_id": trace_id, "label": label, "source": source, "percentile": percentile,
        "hold_last_err20_deg": mean_hold,
        "hold_last_definition": "Mean geodesic error of q[t] vs q[t+20] on the same 300 scored anchors, computed before rounding",
        "generator": generator, "generated": datetime.date.today().isoformat()}
    trace = {"metadata": metadata, "initial_context": rounded(q[first - WARMUP - C + 1:first - WARMUP + 1].astype(np.float32), 7),
             "weights": weights(), "steps": steps}
    return trace, mean_err, mean_hold


def index_entry(trace):
    """The trace's line in traces/index.js (the viewer's player picker)."""
    m = trace["metadata"]
    return {"id": m["trace_id"], "label": m["label"], "user": m["user"], "recording": m["recording"],
            "segment": m["segment"], "percentile": m["percentile"], "mean_speed_deg_s": m["selection_mean_speed_deg_s"],
            "mean_err20_deg": m["mean_step20_error_deg"], "hold_last_err20_deg": m["hold_last_err20_deg"],
            "source": m["source"]}


def read_index():
    text = INDEX.read_text(encoding="utf-8")
    return json.loads(text[len(INDEX_PREFIX):].rstrip().rstrip(";"))


def write_index(entries):
    INDEX.write_text(INDEX_PREFIX + json.dumps(entries, separators=(",", ":")) + ";", encoding="utf-8")


def clip_arrays(q, last_t):
    """The hero clip: 1,200 samples ending at last_t, with the 200 ms prediction streamed from the segment start.
    Returns q, pred, err, hold and the raw outputs [1200, 20, 4]."""
    first_t = last_t - CLIP + 1
    raw = OnnxModel(MODEL, threads=1).stream(q.astype(np.float32), C, H)  # row i is the call at anchor 399 + i
    raw = raw[first_t - (C - 1):last_t - (C - 1) + 1]
    pred = renormalise(raw[:, -1])
    now, truth = q[first_t:last_t + 1], q[first_t + H:last_t + H + 1]
    pred *= np.where(np.sum(pred * now, axis=1) < 0, -1.0, 1.0)[:, None]  # same hemisphere as the current pose
    return now, pred, geodesic_degrees(pred, truth), geodesic_degrees(now, truth), raw


def check(trace_raw, clip):
    now, pred, err, hold, raw = clip
    assert np.array_equal(raw[-RECORDED:], trace_raw), "clip and trace predictions differ on the shared 300 samples"
    for name, values in (("q", now), ("pred", pred)):
        assert (np.sum(values[1:] * values[:-1], axis=1) > 0).all(), f"sign flip between neighbouring samples in {name}"
    assert (np.sum(pred * now, axis=1) > 0).all(), "pred not in the same hemisphere as q"
    for name, values in (("err", err), ("hold", hold)):
        assert np.isfinite(values).all() and (values >= 0).all() and (values <= 180).all(), f"{name} out of range"


# ---------------------------------------------------------------- the story's figure

def figure_anchor(q, first):
    """Of the recorded anchors, the one whose head turn over the next 200 ms is closest to the median turn."""
    anchors = np.arange(first, first + RECORDED)
    turns = geodesic_degrees(q[anchors], q[anchors + H])
    return int(anchors[np.argmin(np.abs(turns - np.median(turns)))]), float(np.median(turns))


def challenge_svg(q, anchor, note):
    """figs/challenge.svg in the layout of Nic's make_story_figs.mjs (840 x 378): four lanes, w x y z from the top,
    each 64 px tall and scaled to its own range over input and answer (higher value = higher on the page). Grey: the 400
    input samples, 2 px apart; blue: the answer (the anchor, then its 20 targets) in the shaded band; a dot on the 20th."""
    window = q[anchor - C + 1:anchor + H + 1]                     # 420 samples: input, then the 20 targets
    fmt = lambda v: f"{round(float(v), 1):g}"                     # noqa: E731
    grey, blue, dots = [], [], []
    for lane in range(4):
        top, bottom = 10 + 98 * lane, 74 + 98 * lane
        v = window[:, lane]
        lo, hi = float(v.min()), float(v.max())
        y = bottom - (v - lo) / max(hi - lo, 1e-12) * (bottom - top)
        grey.append(" ".join(f"{1 + 2 * k},{fmt(y[k])}" for k in range(C)))
        blue.append(" ".join(f"{799 + 2 * k},{fmt(y[C - 1 + k])}" for k in range(H + 1)))
        dots.append(f'<circle cx="839" cy="{fmt(y[-1])}" r="3.2"/>')
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 378" width="840" height="378">',
             f"<!-- {note} -->",
             '<rect x="800" y="0" width="40" height="378" fill="#6fa8ff" fill-opacity=".13"/>',
             '<line x1="800" y1="0" x2="800" y2="378" stroke="#8f9cb0" stroke-width="1" stroke-dasharray="3 4"/>',
             '<g fill="none" stroke="#c3ccd9" stroke-width="1.4" stroke-linejoin="round">',
             *[f'<polyline points="{p}"/>' for p in grey], "</g>",
             '<g fill="none" stroke="#6fa8ff" stroke-width="1.8" stroke-linejoin="round">',
             *[f'<polyline points="{p}"/>' for p in blue], "</g>",
             '<g fill="#f0b24a">', *dots, "</g>", "</svg>"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- files

def write_js(path, prefix, value):
    path.write_text(prefix + json.dumps(value, separators=(",", ":"), allow_nan=False) + ";", encoding="utf-8")


def write_trace(trace):
    trace_id = trace["metadata"]["trace_id"]
    write_js(SITE / f"traces/{trace_id}.js", f"window.TRACES = window.TRACES || {{}}; window.TRACES['{trace_id}'] = ", trace)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv", help="a QuestSet *_movement.csv")
    args = parser.parse_args(argv)
    csv = Path(args.csv)
    group, order, user, speed = re.search(r"group(\d+)_order(\d+)_user(\d+)_(fast|slow)", csv.name).groups()
    game, player = GAMES[(group, speed)], f"questset-g{group}o{order}u{user}"
    today = datetime.date.today().isoformat()

    times, q, prep = load_session(csv)
    first, window_speed, target, candidates = select_window(q)
    last = first + RECORDED - 1
    steps, trace_raw = trace_steps(q, times, first)
    clip_data = clip_arrays(q, last)
    check(trace_raw, clip_data)
    label = f"QuestSet player · {game}"
    source = {"dataset": "QuestSet (Baldoni et al., 2024)", "dataset_url": "https://researchdata.cab.unipd.it/1239/",
              "licence": "CC BY 4.0", "credit": CREDIT, "file": csv.name, "file_sha256": sha256(csv),
              "game": game, "player": player, "session": f"{speed} game, group {group}, order {order}, user {user}"}
    about = {
        **source, "user": player, "recording": f"{game.lower().replace(' ', '-')}-{speed}", "segment": "segment_000",
        "split": "not used for training (external data)",
        "frame_conversion": "QuestSet right-handed (+x right, +y up, -z forward), head -> world; q_ours = (w, -x, -y, z)",
        "preparation": "ray/code/cleaning.py clean (5 s trim), motion_filter, resample to an exact 100 Hz grid; source "
                       f"rate {prep['source_rate_hz']:.1f} Hz (outside the 85-110 Hz band, the one check skipped)",
        "challenge_filter_pass": prep["challenge_filter_pass"], "segment_samples": len(q),
        "preparation_record": {k: prep[k] for k in ("source_rate_hz", "sweep_deg", "velocity_deg_s", "motion_windows")},
        "segment_start_time_s": float(times[0]),
        "selection": "Motion only, never error: of the 300-anchor windows with 792 calls of history, the one whose mean "
                     "head speed is closest to the median over those windows",
        "selection_mean_speed_deg_s": window_speed, "median_window_speed_deg_s": target, "candidate_windows": candidates}
    trace, mean_err, mean_hold = trace_document(q, first, steps, about, "test50", label, "questset", 50,
                                                "tools/make_questset_data.py")
    write_trace(trace)

    now, pred, err, hold, _ = clip_data
    clip_speeds = speeds(now)
    clip = {"meta": {
        "model": trace["metadata"]["model"], "model_sha256": trace["metadata"]["model_sha256"], **source,
        "user": player, "recording": about["recording"], "segment": "segment_000",
        "viewer_trace": "traces/test50.js", "viewer_label": label, "first_t": last - CLIP + 1, "last_t": last,
        "samples": CLIP, "t_definition": "zero-based sample index in the resampled segment; err/hold at t compare "
                                         "against sample t+20",
        "selection": "the viewer trace's player and stretch, chosen by head speed; the last 12 s of the viewer trace "
                     "window; chosen from motion only, never from error",
        "quaternion_order": "wxyz", "clip_mean_speed_deg_s": round(float(clip_speeds.mean()), 4),
        "seam_angle_deg": round(float(geodesic_degrees(now[-1], now[0])), 4),
        "seam_speed_start_deg_s": round(float(clip_speeds[0]), 4), "seam_speed_end_deg_s": round(float(clip_speeds[-1]), 4),
        "seam_speed_diff_deg_s": round(float(abs(clip_speeds[0] - clip_speeds[-1])), 4),
        "shared_samples": f"the last {RECORDED} (t = {first}..{last}), also scored in the viewer trace",
        "shared_mean_err_deg": round(float(err[-RECORDED:].mean()), 6),
        "shared_mean_hold_deg": round(float(hold[-RECORDED:].mean()), 6),
        "prediction_note": "predictions come from the streaming ONNX model run with bench.drivers.OnnxModel from the "
                           "segment's first sample (protocol from-start), so it is fully warmed up at every clip sample; "
                           "pred is the 20th output (200 ms ahead), normalised",
        "rounding": "quaternions 6 decimals, errors 3 decimals; errors computed before rounding in float64",
        "generator": "tools/make_questset_data.py", "generated": today},
        "rate_hz": 100, "horizon": H, "q": rounded(now, 6), "pred": rounded(pred, 6),
        "err": rounded(err, 3), "hold": rounded(hold, 3)}
    write_js(SITE / "hero/clip.js", "// Generated by tools/make_questset_data.py; do not edit. See hero/README.md.\n"
                                    "window.K8_HERO_CLIP = ", clip)

    anchor, median_turn = figure_anchor(q, first)
    (SITE / "figs/challenge.svg").write_text(challenge_svg(q, anchor, (
        f"Generated by tools/make_questset_data.py ({today}); do not edit. {label} ({player}, CC BY 4.0), the call at "
        f"t = {anchor} of traces/test50.js: its 400 input samples and 20 targets. Chosen as the recorded call whose "
        f"200 ms turn ({median_turn:.2f} deg) is the median of the 300.")), encoding="utf-8", newline="\n")

    entry = index_entry(trace)
    entries = read_index()
    at = next((i for i, e in enumerate(entries) if e["id"] == entry["id"]), None)
    if at is None:
        entries.insert(0, entry)
    else:
        entries[at] = entry
    entries.sort(key=lambda e: e["source"] != "questset")   # the QuestSet player first: the viewer opens on it
    write_index(entries)
    print(f"{csv.name}: {game}, {len(q):,} samples at 100 Hz, window t = {first}..{last} "
          f"({window_speed:.1f} deg/s; median {target:.1f}); trace error {mean_err:.4f} deg, hold-last {mean_hold:.4f}; "
          f"clip error {err.mean():.3f} deg; figure call t = {anchor}")
    return entry


if __name__ == "__main__":
    main()
