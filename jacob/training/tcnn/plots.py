"""Figures for training histories and saved evaluations; reads outputs only, never rescores.

Colours follow the entity (a predictor keeps its colour in every figure). Values are
the reference categorical palette, light mode, in fixed slot order.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FIXED = {"tiny_tcnn": SLOTS[0], "damped_velocity": SLOTS[1], "hold_last": SLOTS[2],
         "constant_velocity_1step": SLOTS[3], "constant_velocity_5step": SLOTS[4]}
LABELS = {"tiny_tcnn": "TCNN", "damped_velocity": "Damped velocity", "hold_last": "Hold last",
          "constant_velocity_1step": "Constant velocity (1 step)",
          "constant_velocity_5step": "Constant velocity (5 steps)"}
BASELINES = set(FIXED) - {"tiny_tcnn"}


def _pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    import warnings
    with warnings.catch_warnings():
        # The installed pyparsing warns about matplotlib's own parser calls.
        warnings.simplefilter("ignore", DeprecationWarning)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure().canvas.draw()
        plt.close("all")
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK,
        "lines.linewidth": 2, "font.size": 9, "text.color": INK})
    return plt


def colour(name, order=()):
    if name in FIXED:
        return FIXED[name]
    others = [n for n in order if n not in FIXED]
    return SLOTS[(len(FIXED) + (others.index(name) if name in others else 0)) % len(SLOTS)]


def label(name):
    return LABELS.get(name, name.replace("_", " "))


def _plain_log_labels(axis):
    """Plain numbers on log axes (avoids mathtext powers of ten)."""
    from matplotlib.ticker import FuncFormatter
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def _save(fig, path):
    fig.savefig(path, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return Path(path)


# ---------------------------------------------------------------- training

def plot_training(history_paths, out_png, labels=None):
    """Loss, validation endpoint and learning rate per epoch; one line per run."""
    runs = []
    for i, path in enumerate(map(Path, history_paths)):
        history = json.loads(path.read_text(encoding="utf-8"))
        if history:
            reference_path = path.parent / "validation_reference.json"
            reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.exists() else None
            runs.append((labels[i] if labels else path.parent.name.replace("_", " "), history, reference))
    if not runs:
        raise ValueError("No training history to plot")
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    for i, (name, history, _) in enumerate(runs):
        c = SLOTS[i % len(SLOTS)]
        epochs = [r["epoch"] for r in history]
        scores = [r["validation_endpoint_deg"] for r in history]
        best = int(np.argmin(scores))
        axes[0].plot(epochs, [r["training_loss"] for r in history], color=c, label=name)
        axes[1].plot(epochs, scores, color=c, label=f"{name}: best {scores[best]:.3f}° @ epoch {epochs[best]}")
        axes[1].plot(epochs[best], scores[best], "o", ms=8, color=c, mec=SURFACE, mew=2)
        axes[2].plot(epochs, [r["learning_rate"] for r in history], color=c, label=name)
    scores = np.concatenate([[r["validation_endpoint_deg"] for r in h] for _, h, _ in runs])
    low, high = float(scores.min()), float(scores.max())
    reference = next((r for _, _, r in runs if r), None)
    if reference:
        high = max(high, reference["damped_velocity"]["mean_endpoint_deg"])
    pad = .08 * (high - low) or .05
    axes[1].set_ylim(low - pad, high + 2 * pad)
    if reference:
        off_scale = []
        for key, style in (("damped_velocity", "--"), ("hold_last", ":")):
            value = reference[key]["mean_endpoint_deg"]
            if value > high + 2 * pad:
                # Drawing it would squash the model curves into a thin band.
                off_scale.append(f"{label(key)} {value:.3f}° (above this range)")
                continue
            axes[1].axhline(value, color=colour(key), linestyle=style, linewidth=1.5)
            axes[1].annotate(f"{label(key)} {value:.3f}°", (1, value), xycoords=("axes fraction", "data"),
                             xytext=(-4, 3), textcoords="offset points", ha="right", fontsize=8, color=INK_SECONDARY)
        if off_scale:
            axes[1].annotate("\n".join(off_scale), (0, 1), xycoords="axes fraction", xytext=(4, -4),
                             textcoords="offset points", ha="left", va="top", fontsize=8, color=INK_SECONDARY)
    losses = np.concatenate([[r["training_loss"] for r in h] for _, h, _ in runs])
    if losses.max() / losses.min() >= 10:
        axes[0].set_yscale("log")
        _plain_log_labels(axes[0].yaxis)
        axes[0].set(ylabel="Loss (log scale)")
    else:
        # Less than one decade: a log axis would show no labelled ticks.
        axes[0].set(ylabel="Loss")
    axes[0].set(title="Training loss", xlabel="Epoch")
    axes[1].set(title="Validation error at step 20 (200 ms)", xlabel="Epoch", ylabel="Mean geodesic error (°)")
    axes[2].set(title="Learning rate", xlabel="Epoch", ylabel="Learning rate")
    # One legend for all panels (a run keeps its colour everywhere), outside the plots.
    handles, labels_ = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="outside lower center", ncol=min(len(labels_), 4))
    return _save(fig, out_png)


def plot_variants(comparison_path, out_png, reference_deg=None):
    """Dot plot: the axis need not start at zero, unlike bars."""
    comparison_path = Path(comparison_path)
    rows = json.loads(comparison_path.read_text(encoding="utf-8"))["rows"]
    rows = sorted(rows, key=lambda r: r["validation_endpoint_deg"])
    damped = comparison_path.parent / "damped_baseline" / "full_validation" / "results.json"
    if reference_deg is None and damped.exists():
        reference_deg = json.loads(damped.read_text(encoding="utf-8"))["models"]["damped_velocity"]["mean_endpoint_deg"]
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, .45 * len(rows) + 1.6), layout="constrained")
    values = [r["validation_endpoint_deg"] for r in rows]
    ax.plot(values, range(len(rows)), "o", ms=9, color=FIXED["tiny_tcnn"], mec=SURFACE, mew=1.5, zorder=3)
    for i, v in enumerate(values):
        ax.annotate(f"{v:.3f}°", (v, i), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)), [f"{r['variant']} (seed {r['seed']})" for r in rows])
    ax.invert_yaxis()
    if reference_deg is not None:
        ax.axvline(reference_deg, color=FIXED["damped_velocity"], linestyle="--", linewidth=1.5)
        ax.annotate(f"Damped velocity {reference_deg:.3f}°", (reference_deg, 1), xycoords=("data", "axes fraction"),
                    xytext=(-4, -12), textcoords="offset points", ha="right", fontsize=8, color=INK_SECONDARY)
    ax.margins(x=.15)
    ax.grid(axis="y", visible=False)
    ax.set(title="Screened variants: full validation error at step 20 (lower is better)", xlabel="Mean geodesic error (°)")
    return _save(fig, out_png)


# ---------------------------------------------------------------- evaluation

def _choose(models, primary, reference):
    names = list(models)
    if primary is None:
        primary = "tiny_tcnn" if "tiny_tcnn" in models else min(names, key=lambda n: models[n]["mean_endpoint_deg"])
    if reference is None:
        if "damped_velocity" in models and primary != "damped_velocity":
            reference = "damped_velocity"
        else:
            others = [n for n in names if n != primary]
            reference = min(others, key=lambda n: models[n]["mean_endpoint_deg"]) if others else None
    return primary, reference


def user_bootstrap(differences, users, rounds=2000, seed=0):
    """Window-weighted mean difference with a 95% interval that resamples users, not windows."""
    keys, inverse = np.unique(users, return_inverse=True)
    sums = np.bincount(inverse, weights=differences, minlength=len(keys))
    counts = np.bincount(inverse, minlength=len(keys)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(rounds, len(keys)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"mean": float(sums.sum() / counts.sum()), "ci95": [float(low), float(high)], "users": len(keys),
            "per_user": dict(zip(keys.tolist(), (sums / counts).tolist()))}


def _anchor_speeds(data_path, segments, index, rate_hz, lag=5):
    """Head angular speed (°/s) over the last `lag` samples before each anchor."""
    import h5py
    speeds = np.empty(len(index))
    with h5py.File(data_path, "r") as h5:
        for segment_id, row in enumerate(segments):
            rows = np.flatnonzero(index["segment_id"] == segment_id)
            if not len(rows):
                continue
            q = np.asarray(h5[row["path"]]["quaternion_wxyz"][:], dtype=np.float64)
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            anchors = index["anchor"][rows]
            dot = np.abs(np.sum(q[anchors] * q[anchors - lag], axis=1))
            speeds[rows] = np.rad2deg(2 * np.arccos(np.clip(dot, 0, 1))) * rate_hz / lag
    return speeds


def render_evaluation(eval_dir, data_path=None, *, primary=None, reference=None, histories=(),
                      comparison=None, examples=None, out_dir=None):
    """Write figures and figures.md for an evaluation folder produced by baseline_pipeline.evaluate.

    examples: optional {predictor name: object with predict_batch(history_wxyz)} for forecast traces.
    """
    eval_dir = Path(eval_dir)
    out = Path(out_dir) if out_dir else eval_dir
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((eval_dir / "results.json").read_text(encoding="utf-8"))
    models = report["models"]
    primary, reference = _choose(models, primary, reference)
    names = list(models)
    ms = np.asarray(report["horizons_ms"])
    rate = report["config"]["rate_hz"]
    plt = _pyplot()
    figures, notes = [], []

    def add(path, caption):
        figures.append((Path(path).name, caption))

    # Horizon profile relative to hold-last and to the reference.
    hold = np.asarray(models["hold_last"]["mean_deg_by_horizon"]) if "hold_last" in models else None
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout="constrained")
    if hold is not None:
        for name in names:
            if name == "hold_last":
                continue
            curve = 100 * (hold - np.asarray(models[name]["mean_deg_by_horizon"])) / hold
            axes[0].plot(ms, curve, color=colour(name, names), label=label(name))
        axes[0].axhline(0, color=INK_SECONDARY, linewidth=1)
        axes[0].set(title="Improvement over hold-last by horizon", xlabel="Time ahead (ms)",
                    ylabel="Error reduction vs hold-last (%)")
        axes[0].legend(loc="upper right")
    if reference:
        gap = np.asarray(models[primary]["mean_deg_by_horizon"]) - np.asarray(models[reference]["mean_deg_by_horizon"])
        axes[1].bar(ms, gap, width=(ms[1] - ms[0]) * .7 if len(ms) > 1 else 5,
                    color=[colour(primary, names) if g <= 0 else colour(reference, names) for g in gap])
        axes[1].axhline(0, color=INK_SECONDARY, linewidth=1)
        axes[1].set(title=f"{label(primary)} minus {label(reference)} (below 0: {label(primary)} better)",
                    xlabel="Time ahead (ms)", ylabel="Difference in mean error (°)")
    _save(fig, out / "horizon_relative.png")
    add("horizon_relative.png", "Left: percentage error reduction against hold-last at every forecast step. "
        f"Right: {label(primary)} minus {label(reference)} in degrees at each step; bars above zero are steps where {label(reference)} is better.")

    # Endpoint error distributions.
    endpoints = {n: np.load(eval_dir / models[n]["endpoint_errors_file"]) for n in names}
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    levels = np.linspace(0, 1, 801)[1:-1]
    for name in names:
        quantiles = np.quantile(endpoints[name], levels)
        ax.plot(np.maximum(quantiles, 1e-3), levels, color=colour(name, names), label=label(name),
                linewidth=2.5 if name == primary else 1.6)
        if name in (primary, reference):
            for q in (.5, .95):
                x = float(np.quantile(endpoints[name], q))
                ax.plot(x, q, "o", ms=8, color=colour(name, names), mec=SURFACE, mew=2)
    ax.set_xscale("log")
    _plain_log_labels(ax.xaxis)
    ax.set(title="Distribution of step-20 errors (further left is better)", xlabel="Endpoint geodesic error (°, log scale)",
           ylabel="Share of windows at or below")
    ax.axhline(.5, color=GRID, linewidth=1); ax.axhline(.95, color=GRID, linewidth=1)
    ax.legend(loc="lower right")
    _save(fig, out / "endpoint_cdf.png")
    add("endpoint_cdf.png", f"Cumulative distribution of endpoint error per predictor. Dots mark the median and 95th percentile of {label(primary)} and {label(reference)}.")

    # Per-user comparison.
    users = sorted(models[primary]["by_user"], key=lambda u: models[primary]["by_user"][u]["mean_endpoint_deg"])
    shown = [n for n in (primary, reference, "hold_last") if n and n in models and n in names]
    shown = list(dict.fromkeys(shown))
    fig, ax = plt.subplots(figsize=(9, .32 * len(users) + 1.8), layout="constrained")
    y = np.arange(len(users))
    for k, name in enumerate(shown):
        values = [models[name]["by_user"][u]["mean_endpoint_deg"] for u in users]
        # Small vertical dodge so predictors with near-equal errors stay visible.
        dodge = (k - (len(shown) - 1) / 2) * .18
        ax.plot(values, y + dodge, "o", ms=8, color=colour(name, names), mec=SURFACE, mew=1.5, label=label(name), zorder=3)
    ax.set_yticks(y, [u[:8] for u in users])
    ax.invert_yaxis()
    wins = None
    if reference:
        wins = sum(models[primary]["by_user"][u]["mean_endpoint_deg"] < models[reference]["by_user"][u]["mean_endpoint_deg"] for u in users)
        notes.append(f"{label(primary)} has lower mean endpoint error than {label(reference)} for **{wins} of {len(users)}** users.")
    ax.set(title=f"Per-user mean step-20 error" + (f": {label(primary)} better than {label(reference)} for {wins}/{len(users)} users" if reference else ""),
           xlabel="Mean geodesic error (°)", ylabel="User (ID prefix)")
    ax.grid(axis="y", visible=False)
    fig.legend(loc="outside lower center", ncol=len(shown))
    _save(fig, out / "per_user_endpoint.png")
    add("per_user_endpoint.png", "Mean endpoint error for each user, sorted by the primary model. Users differ far more than predictors do, which is why user-level uncertainty matters.")

    # Paired differences with a user-level bootstrap.
    index = np.load(eval_dir / "window_index.npy")
    segment_users = np.array([s["user"] for s in report["segments"]])
    window_users = segment_users[np.asarray(index["segment_id"])]
    summary = None
    if reference:
        diff = np.asarray(endpoints[primary]) - np.asarray(endpoints[reference])
        summary = user_bootstrap(diff, window_users)
        share = float(np.mean(diff < 0))
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout="constrained")
        limit = float(np.percentile(np.abs(diff), 99.5)) or 1.
        axes[0].hist(diff, bins=120, range=(-limit, limit), color=colour(primary, names))
        axes[0].axvline(0, color=INK_SECONDARY, linewidth=1)
        axes[0].set(title=f"Per-window difference: {label(primary)} better in {100*share:.1f}% of windows",
                    xlabel=f"{label(primary)} minus {label(reference)} endpoint error (°)", ylabel="Windows")
        per_user = summary["per_user"]
        ordered = sorted(per_user, key=per_user.get)
        axes[1].axvspan(*summary["ci95"], color=colour(primary, names), alpha=.15, linewidth=0)
        axes[1].axvline(summary["mean"], color=colour(primary, names), linewidth=2)
        axes[1].axvline(0, color=INK_SECONDARY, linewidth=1)
        axes[1].plot([per_user[u] for u in ordered], range(len(ordered)), "o", ms=8,
                     color=colour(primary, names), mec=SURFACE, mew=1.5)
        axes[1].set_yticks(range(len(ordered)), [u[:8] for u in ordered])
        axes[1].invert_yaxis()
        axes[1].grid(axis="y", visible=False)
        axes[1].set(title=f"Per-user mean difference\noverall {summary['mean']:+.3f}°, "
                          f"user-bootstrap 95% CI {summary['ci95'][0]:+.3f} to {summary['ci95'][1]:+.3f}",
                    xlabel=f"Mean difference (°, below 0: {label(primary)} better)", ylabel="User (ID prefix)")
        _save(fig, out / "paired_difference.png")
        add("paired_difference.png", "Left: per-window endpoint difference (the 0.5% largest absolute differences are outside the plotted range). "
            "Right: per-user mean difference; the line and band are the window-weighted mean and a 95% interval from resampling users, "
            "since windows from one user are correlated.")
        notes.append(f"{label(primary)} minus {label(reference)}: **{summary['mean']:+.4f}°** endpoint error, "
                     f"user-bootstrap 95% CI {summary['ci95'][0]:+.4f}° to {summary['ci95'][1]:+.4f}° "
                     f"({summary['users']} users). {label(primary)} is better in {100*share:.1f}% of windows.")

    # Error against head speed.
    if data_path is not None and Path(data_path).exists():
        from baseline_pipeline import sha256_file
        if sha256_file(data_path) != report["input"]["sha256"]:
            notes.append("Speed and example figures skipped: the data file differs from the evaluated file.")
        else:
            speeds = _anchor_speeds(data_path, report["segments"], index, rate)
            edges = np.array([0, 5, 10, 20, 40, 80, 160, 320, np.inf])
            bins = np.digitize(speeds, edges[1:-1])
            centres = [f"{edges[i]:g}–{edges[i+1]:g}" if np.isfinite(edges[i+1]) else f"≥{edges[i]:g}" for i in range(len(edges)-1)]
            counts = np.bincount(bins, minlength=len(centres))
            present = np.flatnonzero(counts)
            fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6.4), height_ratios=(3, 1), sharex=True, layout="constrained")
            for name in names:
                means = np.bincount(bins, weights=np.asarray(endpoints[name]), minlength=len(centres))[present] / counts[present]
                top.plot(present, means, "-o", ms=8, color=colour(name, names), mec=SURFACE, mew=1.5, label=label(name))
            top.set(title="Step-20 error by head angular speed at the forecast start", ylabel="Mean geodesic error (°)")
            top.legend(loc="upper left")
            bottom.bar(present, counts[present], width=.6, color=INK_SECONDARY)
            for i in present:
                bottom.annotate(f"{counts[i]:,}", (i, counts[i]), xytext=(0, 2), textcoords="offset points",
                                ha="center", fontsize=7, color=INK_SECONDARY)
            bottom.set(ylabel="Windows", xlabel="Head angular speed over the last 50 ms (°/s)")
            bottom.set_xticks(present, [centres[i] for i in present])
            _save(fig, out / "error_vs_speed.png")
            add("error_vs_speed.png", "Endpoint error grouped by how fast the head was turning just before the forecast. "
                "The lower panel shows how many windows fall in each speed band.")
            if examples:
                _plot_examples(data_path, report, index, endpoints, primary, examples, names, out, plt)
                add("example_forecasts.png", f"Angle from the last observed orientation for real windows at the median, 90th and 99th percentile of {label(primary)} error. "
                    "Grey is the observed past; black is what actually happened.")

    if histories:
        plot_training(histories, out / "training_curves.png")
        add("training_curves.png", "Training loss, validation endpoint error and learning rate per epoch for each seed, "
            "with the damped-velocity and hold-last references measured on the same validation windows.")
    if comparison and Path(comparison).exists():
        damped = models.get("damped_velocity", {}).get("mean_endpoint_deg") if report["split"] == "validation" else None
        plot_variants(comparison, out / "variant_comparison.png", damped)
        add("variant_comparison.png", "Validation error of each screened configuration (seed 42), from comparison.json.")

    _write_markdown(out, report, primary, reference, summary, notes, figures)
    return [out / name for name, _ in figures]


def _plot_examples(data_path, report, index, endpoints, primary, predictors, names, out, plt):
    import h5py
    from baseline_pipeline import unit_quaternions
    context, horizon = report["config"]["context"], report["config"]["horizon"]
    errors = np.asarray(endpoints[primary])
    picks = [(q, int(np.argmin(np.abs(errors - np.quantile(errors, q))))) for q in (.5, .9, .99)]
    fig, axes = plt.subplots(1, len(picks), figsize=(14, 4.2), layout="constrained", sharey=False)
    past = 50
    with h5py.File(data_path, "r") as h5:
        for ax, (quantile, row) in zip(axes, picks):
            segment = report["segments"][int(index["segment_id"][row])]
            anchor = int(index["anchor"][row])
            q = unit_quaternions(h5[segment["path"]]["quaternion_wxyz"][anchor - context + 1:anchor + horizon + 1])
            history, future, last = q[:context], q[context:], q[context - 1]

            def angle(x):
                return np.rad2deg(2 * np.arccos(np.clip(np.abs(x @ last), 0, 1)))
            t_past = (np.arange(-past, 1)) * 1000 / report["config"]["rate_hz"]
            t_future = np.arange(1, horizon + 1) * 1000 / report["config"]["rate_hz"]
            ax.plot(t_past, angle(history[-past - 1:]), color=INK_SECONDARY, linewidth=1.5)
            ax.plot(t_future, angle(future), color=INK, linewidth=2, label="Actual")
            for name, model in predictors.items():
                prediction = np.asarray(model.predict_batch(history[None].astype(np.float32)))[0]
                ax.plot(t_future, angle(prediction), color=colour(name, names), linestyle="--", label=label(name))
            ax.axvline(0, color=GRID, linewidth=1)
            ax.set(title=f"{int(quantile*100)}th percentile window ({errors[row]:.2f}° error)",
                   xlabel="Time relative to last input (ms)", ylabel="Angle from last input (°)")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="outside upper center", ncol=len(labels_))
    _save(fig, out / "example_forecasts.png")


def _write_markdown(out, report, primary, reference, summary, notes, figures):
    models = report["models"]
    lines = ["# Model performance figures", "",
             f"Split: **{report['split']}** · {len(report['evaluated_users'])} users · {report['windows']:,} windows. "
             "Lower error is better. Figures are drawn from `results.json` and the saved per-window errors.", "",
             "| Predictor | Endpoint mean (°) | Median (°) | p95 (°) | 100 ms (°) | Equal-user mean (°) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in sorted(models.items(), key=lambda item: item[1]["mean_endpoint_deg"]):
        step10 = result["mean_deg_by_horizon"][9] if len(result["mean_deg_by_horizon"]) >= 10 else float("nan")
        lines.append(f"| {label(name)} | {result['mean_endpoint_deg']:.4f} | {result['median_endpoint_deg']:.4f} | "
                     f"{result['p95_endpoint_deg']:.4f} | {step10:.4f} | {result['equal_user_mean_endpoint_deg']:.4f} |")
    lines += [""] + [f"- {note}" for note in notes] + [""]
    for name, caption in figures:
        lines += [f"## {name.removesuffix('.png').replace('_', ' ').capitalize()}", "", caption, "", f"![{name}]({name})", ""]
    (Path(out) / "figures.md").write_text("\n".join(lines), encoding="utf-8")


def experiment_files(selection_path):
    """Seed histories and the screening comparison behind a locked selection.json."""
    selection_path = Path(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    base = selection_path.parent / selection["selected_variant"] if "selected_variant" in selection else selection_path.parent
    histories = [base / f"seed_{s}" / "history.json" for s in selection.get("evaluated_seeds", [])]
    comparison = selection_path.parent / "comparison.json"
    return [h for h in histories if h.exists()], comparison if comparison.exists() else None
