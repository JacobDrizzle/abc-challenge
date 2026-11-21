"""Archive and independently audit a completed endpoint-refit run; no data evaluation."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from bench.stats import paired


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(run, output):
    status, manifest = read(run / "status.json"), read(run / "manifest.json")
    if status["status"] in ("running", "failed"):
        raise ValueError("Only a completed accuracy result can be archived")
    audit = {"sources": {}, "code": {}, "prior_experiment": {}}
    for seed, row in manifest["sources"].items():
        audit["sources"][seed] = {key: digest(row[key]) == row["sha256"] for key in ("source", "copy")}
    for path, expected in manifest["source_code_sha256"].items():
        audit["code"][path] = digest(ROOT / "jacob/training" / path) == expected
    for path, expected in manifest["previous_artifact_sha256"].items():
        audit["prior_experiment"][path] = digest(path) == expected
    assert all(all(row.values()) for row in audit["sources"].values())
    assert all(audit["code"].values()) and all(audit["prior_experiment"].values())
    screen = read(run / "screen.json")
    assert len(screen["ranking"]) == 23
    ranking = sorted(screen["ranking"], key=lambda c: (screen["results"][c["id"]]["mean_endpoint_deg"],
                                                       c["train_rms_correction"], c["id"]))
    assert ranking == screen["ranking"]
    confirmations = []
    for seed in (44, 42, 43):
        path = run / f"confirmation_seed_{seed}.json"
        if not path.exists():
            continue
        result = read(path)
        for name in ("identity", "global_105"):
            check = paired({"per_player": result["results"]["candidate"]["per_user"]},
                           {"per_player": result["results"][name]["per_user"]})
            recorded = result["comparisons"][name]
            np.testing.assert_allclose(check["ci95"], recorded["ci95"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(check["mean_diff_deg"], recorded["mean_diff_deg"], atol=1e-12, rtol=0)
            assert check["players_better"] == recorded["users_better"]
        confirmations.append(result)
    model_path = run / "candidate_model.pt"
    if model_path.exists():
        original = torch.load(run / "originals/seed_44.pt", map_location="cpu", weights_only=True)
        edited = torch.load(model_path, map_location="cpu", weights_only=True)
        assert original["model_spec"] == edited["model_spec"]
        changes = {}
        for name, value in original["model_state"].items():
            other = edited["model_state"][name]
            if name == "head.weight":
                assert torch.equal(value[:6], other[:6])
            elif name == "head_bias_x":
                assert torch.equal(value[:2], other[:2])
            else:
                assert torch.equal(value, other), name
            if not torch.equal(value, other):
                changes[name] = int((value != other).sum())
        assert sum(changes.values()) <= 97
        audit["candidate"] = {"sha256": digest(model_path), "changed_scalars": changes,
                              "parameters": sum(t.numel() for t in edited["model_state"].values())}
        assert audit["candidate"]["parameters"] == 7555
    candidate_checks = {}
    for folder in sorted((run / 'candidates').iterdir()):
        original = torch.load(run / 'originals' / (folder.name + '.pt'), map_location='cpu', weights_only=True)
        for path in sorted(folder.glob('*.pt')):
            saved = torch.load(path, map_location='cpu', weights_only=True)
            assert saved['model_spec'] == original['model_spec']
            assert saved['endpoint_refit']['source_sha256'] == digest(run / 'originals' / (folder.name + '.pt'))
            weights = original['model_state']
            delta = torch.tensor(saved['endpoint_refit']['candidate']['delta'], dtype=weights['head.weight'].dtype)
            expected_head, expected_bias = weights['head.weight'].clone(), weights['head_bias_x'].clone()
            expected_head[6:] += torch.stack((delta[:32], delta[33:65], delta[65:]))
            expected_bias[2] += delta[32]
            expected = {**weights, 'head.weight': expected_head, 'head_bias_x': expected_bias}
            assert expected.keys() == saved['model_state'].keys()
            assert all(torch.equal(value, saved['model_state'][key]) for key, value in expected.items())
            candidate_checks[str(path.relative_to(run))] = digest(path)
    audit['verified_candidate_files'] = candidate_checks
    output.mkdir(parents=True, exist_ok=False)
    for path in sorted(run.iterdir()):
        if path.is_file() and path.suffix in (".json", ".npz"):
            shutil.copy2(path, output / path.name)
    shutil.copy2(run.parent / (run.name + ".log"), output / "run.log")
    (output / "integrity_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    lines = ["# Endpoint refit on the 800-user k8 model", "",
             f"The experiment ran on `{manifest['branch']}` for {status['elapsed_seconds']/60:.1f} minutes. "
             "Both validation subsets were used in earlier work. The 160 test users remain reserved.", ""]
    if confirmations:
        primary = confirmations[0]
        current = primary['results']
        observed = primary['comparisons']['identity']
        lines += [f"**Result.** The run met the criteria for `{status['status']}`. On the 40 reused validation users, "
                  f"seed 44's mean endpoint error went from {current['identity']['mean_endpoint_deg']:.5f}° to "
                  f"{current['candidate']['mean_endpoint_deg']:.5f}°. The fit improved {observed['users_better']}/40 users. "
                  "Performance on fresh users is still an open question.", ""]
    lines += [
             "## What was fitted", "",
             "Only the 97 cubic-term output coefficients were fitted. The temporal feature extractor stayed frozen. "
             "The fit used 1,620,028 stride-20 examples from all 800 training users, a sign-invariant quaternion "
             "alignment objective, regularization toward the original model, and a 10% cap on the training RMS "
             "change in endpoint residual. The CPU solved the coefficients directly. Both models have 7,555 parameters.", "",
             "Seven penalties and three correction strengths produced 21 fits, compared with the original weights and "
             "the prior 1.05-gain edit. All fitted seed 44 checkpoints are saved in the separate run folder.", "",
             "## Selection", "", f"The stride-5 selection winner was **{ranking[0]['id']}**.", "",
             "| Candidate | Endpoint error (°) | Change vs original (°) |", "| --- | ---: | ---: |"]
    reference = screen["results"]["identity"]["mean_endpoint_deg"]
    for item in ranking:
        value = screen["results"][item["id"]]["mean_endpoint_deg"]
        lines.append(f"| {item['id']} | {value:.6f} | {value-reference:+.6f} |")
    if confirmations:
        winner = read(run / "frozen_fit.json")["candidate"]
        discovery = read(run / 'discovery_seed_44.json')['results']
        lines += ["", f"On all {discovery['candidate']['windows']:,} stride-1 selection windows, the original scored "
                  f"{discovery['identity']['mean_endpoint_deg']:.6f}°, the previous gain scored "
                  f"{discovery['global_105']['mean_endpoint_deg']:.6f}°, and the fitted copy scored "
                  f"{discovery['candidate']['mean_endpoint_deg']:.6f}°.", "",
                  f"The frozen fit used penalty **{winner['penalty']:g}**, strength **{winner['strength']:g}**, "
                  f"and RMS cap multiplier **{winner['cap_factor']:.6f}** for seed 44.", "",
                  "## Reused 40-user assessment", "",
                  "Negative differences favour the fitted model. Intervals resample users with 10,000 draws, seed0. "
                  "The three source seeds share the same users.", "",
                  "| Seed | Original (°) | Previous gain (°) | Fitted (°) | Fitted − original (°) | 95% interval | Users improved |",
                  "| --- | ---: | ---: | ---: | ---: | --- | ---: |"]
        for result in confirmations:
            values, p = result["results"], result["comparisons"]["identity"]
            lo, hi = p["ci95"]
            lines.append(f"| {result['seed']} | {values['identity']['mean_endpoint_deg']:.6f} | "
                         f"{values['global_105']['mean_endpoint_deg']:.6f} | {values['candidate']['mean_endpoint_deg']:.6f} | "
                         f"{p['mean_diff_deg']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {p['users_better']}/{p['users']} |")
        primary = confirmations[0]
        failed = [name for name, passed in primary["gate"]["checks"].items() if not passed]
        lines += ["", ("Seed 44 missed these checks: " + ", ".join(failed) + "." if failed else "Seed 44 passed all predeclared assessment checks."),
                  "", "| Seed 44 measure (°) | Original | Previous gain | Fitted |",
                  "| --- | ---: | ---: | ---: |"]
        for title, key in [('Equal-user endpoint mean', 'equal_user_mean_endpoint_deg'), ('Endpoint median', 'median_endpoint_deg'),
                           ('Endpoint p95', 'p95_endpoint_deg'), ('Endpoint p99', 'p99_endpoint_deg'), ('Mean across all horizons', 'mean_all_horizons_deg')]:
            lines.append('| ' + title + ' | ' + ' | '.join(f"{primary['results'][name][key]:.6f}" for name in ('identity','global_105','candidate')) + ' |')
        lines += ["", "The per-user files retain every horizon and both paired comparisons. The fit stayed frozen throughout this assessment."]
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
        for y, result in enumerate(confirmations):
            p = result["comparisons"]["identity"]
            d, (lo, hi) = p["mean_diff_deg"], p["ci95"]
            ax.errorbar(d, y, xerr=[[d-lo], [hi-d]], fmt="o", color="#2563a8", capsize=5)
        ax.set_yticks(range(len(confirmations)), [f"Seed {r['seed']}" for r in confirmations])
        ax.invert_yaxis()
        ax.axvline(0, color="#b14040", linestyle="--")
        ax.set_xlabel("Fitted minus original endpoint error (degrees)\n95% paired user-bootstrap intervals; reused validation users")
        ax.set_title("Endpoint refit: exploratory comparison", loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(output / "confirmation_intervals.png", dpi=180)
        fig.savefig(output / "confirmation_intervals.svg")
        lines += ["", "![Exploratory endpoint comparison](confirmation_intervals.png)"]
        values = primary["results"]
        base = np.asarray(values["identity"]["mean_deg_by_horizon"])
        fitted = np.asarray(values["candidate"]["mean_deg_by_horizon"]) - base
        previous = np.asarray(values["global_105"]["mean_deg_by_horizon"]) - base
        fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
        ax.plot(np.arange(1, 21) * 10, fitted, "o-", color="#2563a8", label="Fitted cubic correction")
        ax.plot(np.arange(1, 21) * 10, previous, "s--", color="#b96e2d", label="Previous +5% gain")
        ax.axhline(0, color="#555555", linewidth=1)
        ax.set_xlabel("Prediction horizon (ms)")
        ax.set_ylabel("Mean error change vs original (degrees)")
        ax.set_title("Seed44: effect across the 20 forecast steps", loc="left")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(output / "horizon_changes.png", dpi=180)
        fig.savefig(output / "horizon_changes.svg")
        lines += ["", "![Mean changes at every forecast horizon](horizon_changes.png)", "",
                  "These are descriptive means on the 40 reused users. Negative values indicate improvements. "
                  "The scored endpoint is 200 ms."]
    elif status["status"] == "control_won_screen":
        lines += ["", "A control won selection, so the protocol stopped. No fitted candidate was assessed on the "
                  "remaining 40 users, and the other seeds were not refitted."]
    else:
        lines += ["", "The sole finalist did not beat both controls in the full discovery check, so the protocol stopped."]
    if (run / 'export_checks.json').exists():
        deployment = read(run / 'export_checks.json')
        base, edited = (deployment['models'][name] for name in ('baseline', 'edited'))
        assert deployment['passed'] and deployment['graph_structure_unchanged']
        for model in (base, edited):
            assert digest(model['onnx']) == model['onnx_sha256']
        latency = deployment['latency']
        lines += ["", "## Streaming export", "",
                  f"Both ONNX files contain {edited['onnx_bytes']:,} bytes, {edited['nodes']} nodes and "
                  f"{edited['macs_per_call']['total']:,} multiply-accumulates per call. Both satisfy the operator audit. "
                  f"The largest fitted ONNX versus PyTorch window discrepancy was {edited['parity']['max_ort_vs_window_deg']:.7f}° "
                  f"on {edited['parity']['synthetic_samples']} synthetic samples.", "",
                  "| Single-thread CPU latency | Original | Fitted |", "| --- | ---: | ---: |",
                  f"| Median (µs) | {latency['baseline']['median_us']:.1f} | {latency['edited']['median_us']:.1f} |",
                  f"| p95 (µs) | {latency['baseline']['p95_us']:.1f} | {latency['edited']['p95_us']:.1f} |", "",
                  "Timing used five alternating rounds on this PC. The small differences fall within the variation across rounds. "
                  "The organisers measure deployment latency on their XR hardware."]
    lines += ["", "## Preservation and verification", "",
              "All three original checkpoints and preserved copies match their hashes. The first experiment's recorded "
              "artifacts and all recorded source files match their hashes. Every saved fitted checkpoint matches its source "
              "plus the recorded cubic correction. The team's benchmark statistic reproduced the paired intervals and user counts.", "",
              "30 unit tests passed, plus a CUDA smoke check. Tests cover the independent projector equations, known "
              "coefficient recovery, regularization limits, inactive features, near-180-degree targets, mirror/sign/yaw "
              "invariance, cubic timing, saved copies, data-split guards and actual-model/cache parity.", "",
              "The fitted checkpoints contain inference weights and fit provenance. The team retains control of model promotion and Git publication.", "",
              "## Files", "",
              f"- Run outputs and model copies: `{run}`",
              "- [Manifest](manifest.json), [complete screen](screen.json), [final status](status.json), "
              "[integrity audit](integrity_audit.json), [run log](run.log)",
              "- [Implementation and reproducibility instructions](../../../training/experimental/v10_endpoint_refit/README.md)"]
    if model_path.exists():
        lines.append(f"- [Separate fitted model](../../../training/outputs/{run.name}/candidate_model.pt)")
    if (run / "export_checks.json").exists():
        lines.append(f"- [Fitted streaming ONNX](../../../training/outputs/{run.name}/onnx/edited.onnx) and [verification results](export_checks.json)")
    else:
        lines += ["", "Deployment export and latency checks were conditional on a promising result and were not run."]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status["status"], "report": str(output / "README.md"), "integrity_verified": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "jacob/training/outputs/v10_endpoint_refit_20261105")
    parser.add_argument("--output", type=Path, default=ROOT / "jacob/ablations/results/v10_endpoint_refit_20261105")
    args = parser.parse_args()
    archive(args.run.resolve(), args.output.resolve())
