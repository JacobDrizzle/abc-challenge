"""python -m bench <command> ... (see bench/README.md). Console output is ASCII; files are UTF-8."""
import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bench import __version__, contamination, data, efficiency, results, stats
from bench.baselines import BASELINES, make_baseline
from bench.drivers import OnnxModel, Submission
from bench.geometry import geodesic_degrees, random_motion, renormalise
from bench.scoring import BaselinePredictor, score


def _players_file(path):
    return sorted(contamination.parse(str(path))["players"])


def cmd_define(a):
    legacy = {"protocol": data.LEGACY_PROTOCOL} if a.legacy else None
    path = data.define(a.name, a.h5, _players_file(a.players), a.description, legacy=legacy)
    d = json.loads(path.read_text(encoding="utf-8"))
    print(f"Wrote {path}: {len(d['players'])} players, {d['windows']:,} windows, fingerprint {d['cohort_fingerprint'][:16]}")


def cmd_data_extract(a):
    out = data.extract(a.source, _players_file(a.players), a.out)
    print(f"Wrote {out} (sha256 {data.sha256_file(out)})")
    if a.name:
        path = data.define(a.name, out, _players_file(a.players), a.description or "")
        print(f"Wrote {path}")


def cmd_data_verify(a):
    definition = data.load_definition(a.bench)
    sha = data.sha256_file(a.h5)
    print(f"file sha256 {sha} ({'matches' if sha == definition['data']['sha256'] else 'differs from'} the definition's file)")
    with data.Benchmark(definition, a.h5, verify=False) as b:
        ok = b.fingerprint == definition["cohort_fingerprint"]
        print(f"cohort fingerprint {b.fingerprint[:16]}: {'OK' if ok else 'MISMATCH'}; {b.windows:,} windows")
    return 0 if ok else 1


def cmd_register(a):
    registry = results.load_registry()
    name = results.check_name(a.name)
    if name in registry and not a.force:
        raise SystemExit(f"{name} is already registered; use --force to replace it")
    contamination.check([], a.trained_on)  # parses every spec, so typos fail here
    registry[name] = {"onnx_sha256": data.sha256_file(a.onnx), "source": a.source or str(a.onnx), "owner": a.owner,
                      "trained_on": a.trained_on, "declared_trainable_parameters": a.declared_params,
                      "group": a.group, "notes": a.notes}
    results.save_registry(registry)
    print(f"Registered {name} in {results.REGISTRY}")


def _summary_line(name, e):
    m, o = e["macs_per_call"], e["operators"]
    return (f"{name}: {'stateful' if e['stateful'] else 'stateless'}, {e['initializer_elements']:,} initializer "
            f"elements (+{e['constant_node_elements']:,} in Constant nodes), {m['total']:,} MACs per call "
            f"{m['by_operator']}, issues: {', '.join(o['issues']) or 'none'}; off-list ops: "
            f"{', '.join(o['other']) or 'none'}; slow on NPU: {', '.join(o['slow_on_npu']) or 'none'}; "
            f"deterministic: {e['output_checks']['deterministic']}")


def cmd_audit(a):
    e = efficiency.report(a.onnx, a.stateless_reference)
    print(_summary_line(Path(a.onnx).name, e))
    if e["stream_parity"]:
        print(f"stream parity vs {Path(a.stateless_reference).name}: max {e['stream_parity']['max_difference_deg']:.2e} deg")
    if a.json:
        results.write_json(a.json, e)


def cmd_score(a):
    definition = data.load_definition(a.bench)
    machine = results.machine_label(a.machine)
    registry = results.load_registry()
    if a.model.startswith("baseline:"):
        kind, path = "baseline", None
        baseline = a.model.split(":", 1)[1]
        name = results.check_name(a.name or make_baseline(baseline).name)
        entry, trained_on = {}, ["none"]
    else:
        kind, path = "onnx", Path(a.model)
        name = results.check_name(a.name or path.stem.lower())
        entry = registry.get(name, {})
        sha = data.sha256_file(path)
        if entry and entry["onnx_sha256"] != sha:
            raise SystemExit(f"{name} is registered with a different ONNX file ({entry['onnx_sha256'][:12]} != "
                             f"{sha[:12]}); register the new file under a new name")
        trained_on = a.trained_on or entry.get("trained_on") or []
    check = contamination.check(definition["players"], trained_on)
    if check["status"] != "clean":
        detail = (f"{len(check['overlap'])} benchmark player(s) in its training/validation data"
                  if check["overlap"] else check.get("note", ""))
        if not a.allow_contaminated:
            raise SystemExit(f"Refusing to score {name}: {check['status']} ({detail}). "
                             "Pass --allow-contaminated to score it anyway; it will be marked and not ranked.")
        print(f"WARNING: {name} is {check['status']} ({detail}); the result is marked and not ranked.")

    with data.Benchmark(definition, a.h5) as b:
        print(f"Benchmark {b.name}: {len(b.players)} players, {b.windows:,} windows, fingerprint {b.fingerprint[:16]}")
        if kind == "baseline":
            predictor, eff = BaselinePredictor(make_baseline(baseline, b.horizon, b.rate_hz)), None
            predictor.name = name
        else:
            print("Efficiency checks (generated motion, no benchmark data)...")
            eff = efficiency.report(path, a.stateless_reference)
            print("  " + _summary_line(name, eff))
            predictor = OnnxModel(path, name, threads=a.threads)
        protocols = (["from-start", "cold-start"] if a.protocol == "both" else [a.protocol]) if predictor.stateful \
            else ["stateless"]
        progress = print if a.verbose else None
        scored = {}
        for protocol in protocols:
            print(f"Scoring {name} [{protocol}]...")
            accuracy, endpoint, segment = score(predictor, b, protocol, progress=progress)
            scored[protocol] = accuracy
            run = results.RUNS / b.name / f"{name}@{machine}"
            file = results.write_npy(run / f"endpoint_{protocol}.npy", endpoint.astype(np.float32))
            results.write_npy(run / "segment_id.npy", segment)
            accuracy["per_window_file"] = {"path": str(file.relative_to(results.REPO)).replace("\\", "/"),
                                           "sha256": data.sha256_file(file), "dtype": "float32"}
            print(f"  mean error at step 20: {accuracy['mean_deg_step20']:.6f} deg (step 10: "
                  f"{accuracy['mean_deg_step10']:.6f}); median {accuracy['median_deg_step20']:.4f}, p95 "
                  f"{accuracy['p95_deg_step20']:.4f}")
        bench_info = {"name": b.name, "definition_sha256": definition["definition_sha256"],
                      "cohort_fingerprint": b.fingerprint, "legacy_cohort_fingerprint": b.legacy_fingerprint,
                      "windows": b.windows, "players": b.players}
        inputs = {"data_file": b.h5_path.name, "data_sha256": b.data_sha256}

    from bench.latency import machine_info
    record = {
        "schema": "bench-result-v1", "bench_package_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(), "bench": bench_info, "inputs": inputs,
        "model": {"name": name, "kind": kind, "onnx": str(path) if path else None,
                  "onnx_sha256": data.sha256_file(path) if path else None,
                  "baseline": baseline if kind == "baseline" else None,
                  "stateful": bool(predictor.stateful), "owner": entry.get("owner"),
                  "group": a.group or entry.get("group"),
                  "declared_trainable_parameters": a.declared_params or entry.get("declared_trainable_parameters")
                  or (0 if kind == "baseline" else None),
                  "notes": entry.get("notes"), "threads": a.threads},
        "contamination": check, "git": results.git_info(), "machine": {"label": machine, **machine_info()},
        "efficiency": eff, "accuracy": scored[protocols[0]],
        "accuracy_cold_start": scored.get("cold-start") if len(protocols) > 1 else None}
    out = results.write_json(results.result_path(b.name, name, machine), record)
    print(f"Wrote {out}")


def cmd_score_submission(a):
    """Interface check of a submission folder: predict() vs the bare ONNX, and the command-line form."""
    folder = Path(a.folder).resolve()
    onnx_path = Path(a.onnx) if a.onnx else folder / "model_fp32.onnx"
    q = random_motion(400 + a.windows + 19, seed=3)
    bare = renormalise(OnnxModel(onnx_path).stream(q))
    wrapped = renormalise(np.concatenate(list(Submission(folder).predict_segment(q, 400, 20))))
    difference = geodesic_degrees(bare, wrapped)
    print(f"predict() vs {onnx_path.name} on {len(bare)} consecutive windows: max {difference.max():.2e} deg, "
          f"mean {difference.mean():.2e} deg")
    with tempfile.TemporaryDirectory() as tmp:
        sample, output = Path(tmp) / "sample.npy", Path(tmp) / "pred.npy"
        np.save(sample, q[:400])
        run = subprocess.run([sys.executable, "run_inference.py", "--model_dir", ".", "--input", str(sample),
                              "--output", str(output)], cwd=folder, capture_output=True, text=True)
        cli = np.load(output) if run.returncode == 0 and output.exists() else None
    if cli is None:
        print(f"Command-line form FAILED (exit {run.returncode}): {run.stderr.strip()[-500:]}")
        return 1
    cli_ok = cli.shape in ((20, 4), (1, 20, 4)) and np.isfinite(cli).all()
    cli_difference = geodesic_degrees(renormalise(cli.reshape(20, 4)), bare[0]).max() if cli_ok else np.inf
    print(f"Command-line form: shape {cli.shape}, dtype {cli.dtype}, max {cli_difference:.2e} deg vs bare ONNX")
    ok = difference.max() <= a.tolerance and cli_difference <= a.tolerance
    print("OK" if ok else f"MISMATCH above {a.tolerance} deg")
    return 0 if ok else 1


def cmd_compare(a):
    bench = data.load_definition(a.bench)["name"]
    ra, rb = results.find_result(bench, a.a, a.machine), results.find_result(bench, a.b, a.machine)
    if ra["bench"]["cohort_fingerprint"] != rb["bench"]["cohort_fingerprint"]:
        raise SystemExit("Different cohort fingerprints: these results were not scored on the same windows")
    key = "accuracy_cold_start" if a.cold_start else "accuracy"
    xa, xb = ra[key] or ra["accuracy"], rb[key] or rb["accuracy"]
    paired = stats.has_per_player(xa) and stats.has_per_player(xb)
    p = stats.paired(xa, xb, a.draws, a.seed) if paired else None
    print(f"{a.a}: {xa['mean_deg_step20']:.4f} deg, {a.b}: {xb['mean_deg_step20']:.4f} deg at step 20 ({bench})")
    if p is None:
        print(f"{a.a} - {a.b} = {xa['mean_deg_step20'] - xb['mean_deg_step20']:+.4f} deg. No 95% CI: it needs "
              "per-player sums, which published results leave out. Re-score both models with "
              "`python -m bench score` to get it.")
        return
    print(f"{a.a} - {a.b} = {p['mean_diff_deg']:+.4f} deg, 95% CI [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] "
          f"(paired bootstrap over {p['players']} players, {p['draws']} draws); {a.a} better for "
          f"{p['players_better']}/{p['players']} players; equal-player difference {p['equal_player_mean_diff_deg']:+.4f}")


def cmd_latency(a):
    from bench.latency import session
    models = {}
    for item in a.models:
        name, _, path = item.rpartition("=") if "=" in item else (Path(item).stem.lower(), "", item)
        models[results.check_name(name)] = Path(path)
    label = results.machine_label(a.machine)
    record = session(models, label, a.threads, a.warmup, a.runs, a.rounds)
    record["git"] = results.git_info()
    stamp = record["created_utc"][:19].replace(":", "").replace("-", "")
    out = results.write_json(results.RESULTS / "latency" / f"{label}_{stamp}.json", record)
    for name, entry in record["models"].items():
        print(f"{name}: " + ", ".join(f"{t} thr median {v['median_ms']:.3f} ms (p99 {v['p99_ms']:.3f})"
                                      for t, v in entry["threads"].items()))
    print(f"Wrote {out}")


def cmd_leaderboard(a):
    print(f"Wrote {results.write_leaderboard(a.out)}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m bench", description="Team head-pose benchmark")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("define", help="Freeze a new benchmark definition for a player list on an HDF5")
    s.add_argument("--name", required=True)
    s.add_argument("--h5", required=True, type=Path)
    s.add_argument("--players", required=True, type=Path, help="Text file, one player id per line")
    s.add_argument("--description", default="")
    s.add_argument("--legacy", action="store_true", help="Also record the old evaluator's cohort fingerprint")
    s.set_defaults(func=cmd_define)

    s = sub.add_parser("data-extract", help="Write a benchmark-only HDF5 with the given players (for sharing)")
    s.add_argument("--from", dest="source", required=True, type=Path)
    s.add_argument("--players", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--name", help="Also freeze a definition for the extract under this name")
    s.add_argument("--description")
    s.set_defaults(func=cmd_data_extract)

    s = sub.add_parser("data-verify", help="Check that an HDF5 holds a benchmark's exact windows")
    s.add_argument("--bench", default="group-a-v2-test20")
    s.add_argument("--h5", required=True, type=Path)
    s.set_defaults(func=cmd_data_verify)

    s = sub.add_parser("register", help="Add a model to bench/models.json with its training players")
    s.add_argument("name")
    s.add_argument("onnx", type=Path)
    s.add_argument("--trained-on", action="append", required=True, help="Spec, repeatable (see contamination.py)")
    s.add_argument("--declared-params", type=int)
    s.add_argument("--group", help="Seed group, e.g. gru32_velmag")
    s.add_argument("--owner")
    s.add_argument("--source", help="Where the file lives, for teammates")
    s.add_argument("--notes")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_register)

    s = sub.add_parser("audit", help="Efficiency only: params, MACs, operators, output checks (no data)")
    s.add_argument("onnx", type=Path)
    s.add_argument("--stateless-reference", type=Path, help="Stateless graph of the same model, for stream parity")
    s.add_argument("--json", type=Path)
    s.set_defaults(func=cmd_audit)

    s = sub.add_parser("score", help="Score an ONNX model (or baseline:<name>) on a benchmark")
    s.add_argument("model", help=f"model.onnx, or baseline:{{{','.join(BASELINES)}}}")
    s.add_argument("--h5", required=True, type=Path)
    s.add_argument("--bench", default="group-a-v2-test20")
    s.add_argument("--name")
    s.add_argument("--trained-on", action="append", help="Spec, repeatable; default: from bench/models.json")
    s.add_argument("--declared-params", type=int)
    s.add_argument("--group")
    s.add_argument("--protocol", choices=["both", "from-start", "cold-start"], default="both",
                   help="Stateful graphs only; stateless graphs are always re-run per window")
    s.add_argument("--stateless-reference", type=Path)
    s.add_argument("--allow-contaminated", action="store_true")
    s.add_argument("--machine", help="Machine label (default: $BENCH_MACHINE or os-arch)")
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_score)

    s = sub.add_parser("score-submission", help="Check a submission folder's run_inference.py against its ONNX")
    s.add_argument("folder", type=Path)
    s.add_argument("--onnx", type=Path, help="Default: <folder>/model_fp32.onnx")
    s.add_argument("--windows", type=int, default=2000)
    s.add_argument("--tolerance", type=float, default=1e-4, help="Degrees")
    s.set_defaults(func=cmd_score_submission)

    s = sub.add_parser("compare", help="Paired per-player bootstrap between two scored models")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--bench", default="group-a-v2-test20")
    s.add_argument("--machine")
    s.add_argument("--cold-start", action="store_true", help="Use the cold-start scores of stateful models")
    s.add_argument("--draws", type=int, default=10000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("latency", help="Time models in one alternating session on this machine")
    s.add_argument("models", nargs="+", help="name=path.onnx (or just path.onnx)")
    s.add_argument("--machine")
    s.add_argument("--threads", type=int, nargs="+", default=[1])
    s.add_argument("--warmup", type=int, default=200)
    s.add_argument("--runs", type=int, default=3000)
    s.add_argument("--rounds", type=int, default=3)
    s.set_defaults(func=cmd_latency)

    s = sub.add_parser("leaderboard", help="Regenerate LEADERBOARD.md from bench/results/")
    s.add_argument("--out", type=Path, default=results.LEADERBOARD)
    s.set_defaults(func=cmd_leaderboard)

    args = p.parse_args(argv)
    return args.func(args) or 0
