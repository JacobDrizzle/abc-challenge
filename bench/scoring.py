"""Score one predictor on every window of a benchmark.

A predictor has `name`, `stateful` and predict_segment(q32, context, horizon, protocol), a generator
that yields prediction blocks [n, H, 4] for the segment's anchors C-1 ... N-H-1, in order. Outputs are
re-normalised (as the harness does) and scored with the float64 geodesic of bench.geometry.

Kept per player: window count and per-horizon error sums. Those are enough to recompute every mean
and to run the paired per-player bootstrap later (bench.stats), so they go into the committed JSON.
The per-window step-20 errors (for medians, quantiles and window-level analysis) go to a local file.
"""
import numpy as np

from bench.geometry import geodesic_degrees, renormalise

PROTOCOLS = {
    "stateless": "Stateless: each window is a fresh call on the 400-sample context.",
    "from-start": "Stateful, from segment start: zero states at the segment's first sample, one sample per call; "
                  "before 400 samples exist the context is left-padded with the first sample. Scored from the "
                  "first full window.",
    "cold-start": "Stateful, cold start: zero states at the first full window (anchor 399), then one sample per call.",
}


class BaselinePredictor:
    """Adapts a bench.baselines model (vectorised predict_batch) to the predictor interface."""
    stateful = False

    def __init__(self, model, block=1024):
        self.model, self.name, self.block = model, model.name, block

    def predict_segment(self, q32, context, horizon, protocol="stateless"):
        windows = np.lib.stride_tricks.sliding_window_view(q32, context, axis=0).transpose(0, 2, 1)
        count = len(q32) - context - horizon + 1
        for start in range(0, count, self.block):
            yield self.model.predict_batch(np.array(windows[start:min(count, start + self.block)]))


def score(predictor, bench, protocol="stateless", progress=print):
    """Returns (accuracy summary, per-window step-20 errors float64, per-window segment ids)."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}")
    if (protocol == "stateless") == bool(predictor.stateful):
        raise ValueError(f"{predictor.name}: protocol {protocol!r} does not fit a "
                         f"{'stateful' if predictor.stateful else 'stateless'} model")
    C, H = bench.context, bench.horizon
    endpoint = np.empty(bench.windows, np.float64)
    segment_id = np.empty(bench.windows, np.int32)
    players = {}
    cursor = 0
    for sid, s in enumerate(bench.segments):
        q = bench.segment(s)
        q32 = q.astype(np.float32)
        anchors = np.arange(C - 1, len(q) - H)
        done = 0
        entry = players.setdefault(s["user"], {"windows": 0, "sum_deg_by_horizon": np.zeros(H)})
        for block in predictor.predict_segment(q32, C, H, protocol):
            block = np.asarray(block)
            n = len(block)
            if block.shape != (n, H, 4) or done + n > len(anchors):
                raise ValueError(f"{predictor.name}: bad prediction block {block.shape} at window {done}")
            targets = q[anchors[done:done + n, None] + np.arange(1, H + 1)[None, :]]
            errors = geodesic_degrees(renormalise(block, source=f"{predictor.name} output"), targets)
            entry["sum_deg_by_horizon"] += errors.sum(axis=0, dtype=np.float64)
            endpoint[cursor + done:cursor + done + n] = errors[:, -1]
            done += n
        if done != s["windows"]:
            raise RuntimeError(f"{predictor.name}: {done} predictions for {s['windows']} windows in {s['key']}")
        entry["windows"] += done
        segment_id[cursor:cursor + done] = sid
        cursor += done
        if progress:
            progress(f"  {predictor.name} [{protocol}] segment {sid + 1}/{len(bench.segments)}: "
                     f"{cursor:,}/{bench.windows:,} windows")
    return summarise(players, endpoint, protocol, bench.rate_hz), endpoint, segment_id


def summarise(players, endpoint, protocol, rate_hz=100.0):
    windows = sum(p["windows"] for p in players.values())
    sums = sum(p["sum_deg_by_horizon"] for p in players.values())
    means = sums / windows
    per_player_means = [p["sum_deg_by_horizon"][-1] / p["windows"] for p in players.values() if p["windows"]]
    horizon = len(means)
    return {
        "protocol": protocol, "protocol_text": PROTOCOLS[protocol], "windows": int(windows),
        "mean_deg_step20": float(means[-1]),
        "mean_deg_step10": float(means[9]) if horizon >= 10 else None,
        "mean_deg_by_horizon": [float(x) for x in means],
        "horizon_ms": [float(h * 1000 / rate_hz) for h in range(1, horizon + 1)],
        "median_deg_step20": float(np.median(endpoint)),
        "p95_deg_step20": float(np.percentile(endpoint, 95)),
        "p99_deg_step20": float(np.percentile(endpoint, 99)),
        "equal_player_mean_deg_step20": float(np.mean(per_player_means)),
        "per_player": {u: {"windows": int(p["windows"]), "sum_deg_by_horizon": [float(x) for x in p["sum_deg_by_horizon"]]}
                       for u, p in sorted(players.items())},
    }

