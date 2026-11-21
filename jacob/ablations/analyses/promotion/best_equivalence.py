"""One-off check (30 Oct 2026), run before experimental/v9_lean was archived: best/ behaves exactly as v9 did.

1. Every config in the v6, v7 and v9 configs/ folders, and every variant and candidate of their suites and plans on
   every base config, normalises identically (or fails with the same error) under best and v9.
2. The k8_combined, k5_combined and sep+angular checkpoints load to identical weights and give identical outputs
   (max |delta| = 0) on 1,000 validation windows, as trained and as deployed.
3. k8_combined seed 42 exported by best and by v9 (lean, padded, padded + A4) gives byte-identical ONNX files, with
   the SHA-256 of the graphs in jacob/models/.
4. A tiny synthetic training run of best/configs/base.json is bit-identical through best.hooks and v9_lean.hooks.
5. Full validation of k8_combined's selected checkpoint through best.hooks gives selection.json's 4.247104990427709.

python <this file>   (from jacob/training, the abc venv; check 5 uses CUDA)
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(r"<repo>\jacob\training")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch

from baseline_pipeline import DEFAULT_H5, EvaluationConfig, evaluate, manifest_for
from smoke_tcnn import make_fixture
import run_desktop
from tcnn.runtime import configure_runtime

from best import configuration as best_cfg, export as best_export, hooks as best_hooks, model as best_model
from best.quantize import split_segments
from experimental.v9_lean import configuration as v9_cfg, export as v9_export, hooks as v9_hooks, model as v9_model

results = {}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def outcome(normalize, config):
    try:
        return "ok", normalize(copy.deepcopy(config))
    except Exception as error:  # noqa: BLE001 - the error itself is what is compared
        return type(error).__name__, str(error)


# 1. Configs
files = sorted(ROOT.glob("experimental/v*/configs/*.json"))
bases, overrides = [], []
for path in files:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "variants" in data:
        overrides += [(f"{path.name}:{name}", o) for name, o in data["variants"].items()]
    elif "candidates" in data:
        overrides += [(f"{path.name}:{name}", c["overrides"]) for name, c in data["candidates"].items()]
    else:
        bases.append((path.name, data))
bases.append(("best/base.json", json.loads((ROOT/"best/configs/base.json").read_text())))
checked, mismatches, valid = 0, [], 0
for base_name, base in bases:
    for name, extra in [("(base)", {}), *overrides]:
        config = {**base, **extra}
        ours, theirs = outcome(best_cfg.normalize_config, config), outcome(v9_cfg.normalize_config, config)
        checked += 1
        valid += ours[0] == "ok"
        if ours != theirs:
            mismatches.append(f"{base_name} + {name}")
results["1_configs"] = {"files": [p.relative_to(ROOT).as_posix() for p in files], "combinations": checked,
                        "valid_under_both": valid, "mismatches": mismatches, "pass": not mismatches}
print("1 configs:", results["1_configs"]["combinations"], "combinations,", valid, "valid,",
      len(mismatches), "mismatches", flush=True)

# 2. Checkpoints
segments = split_segments(DEFAULT_H5, manifest_for(DEFAULT_H5), "validation")
rng = np.random.default_rng(0)
lengths = np.array([len(q) - 399 for q in segments], dtype=float)
picks = rng.choice(len(segments), 1000, p=lengths / lengths.sum())
windows = np.stack([segments[i][(e := rng.integers(399, len(segments[i]))) - 399:e + 1] for i in picks])
history = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32))
results["2_checkpoints"] = {}
for folder in ("v9_confirm/k8_combined", "v9_confirm/k5_combined", "v8_recipe_sep_base3_addons/angular_loss"):
    path = ROOT/"outputs"/folder/"selected_model.pt"
    ours, theirs = best_model.load_checkpoint(path)[0], v9_model.load_checkpoint(path)[0]
    same_weights = (list(ours.state_dict()) == list(theirs.state_dict())
                    and all(torch.equal(v, theirs.state_dict()[k]) for k, v in ours.state_dict().items()))
    row = {"same_weights": same_weights}
    for label, a, b in (("trained", ours, theirs),
                        ("deployed", best_model.deployed(ours), v9_model.deployed(theirs))):
        with torch.inference_mode():
            delta = (a.eval()(history) - b.eval()(history)).abs().max().item()
        row[f"max_abs_delta_{label}"] = delta
    row["pass"] = same_weights and row["max_abs_delta_trained"] == 0 and row["max_abs_delta_deployed"] == 0
    results["2_checkpoints"][folder] = row
    print("2", folder, row, flush=True)

# 3. ONNX exports of k8_combined seed 42
results["3_exports"] = {}
k8 = ROOT/"outputs/v9_confirm/k8_combined/selected_model.pt"
with tempfile.TemporaryDirectory() as folder:
    for suffix, options in (("", {}), ("_padded", dict(padded_rows=True)),
                            ("_padded_a4", dict(padded_rows=True, fused_head=True))):
        committed = ROOT.parent/"models"/f"tcnn_v9_k8_combined{suffix}_ga2_stream.onnx"
        ours = best_export.export_streaming_onnx(best_model.deployed(best_model.load_checkpoint(k8)[0]),
                                                 Path(folder)/f"best{suffix}.onnx", **options)[0]
        theirs = v9_export.export_streaming_onnx(v9_model.deployed(v9_model.load_checkpoint(k8)[0]),
                                                 Path(folder)/f"v9{suffix}.onnx", **options)[0]
        row = {"best_sha256": sha256(ours), "v9_sha256": sha256(theirs), "committed_sha256": sha256(committed)}
        row["pass"] = row["best_sha256"] == row["v9_sha256"] == row["committed_sha256"]
        results["3_exports"][committed.name] = row
        print("3", committed.name, row["pass"], flush=True)

# 4. Tiny training run of base.json through both hook sets
base = json.loads((ROOT/"best/configs/base.json").read_text())
tiny = {**base, "batch_size": 4, "steps_per_epoch": 3, "epochs": 2, "validation_stride": 20}
with tempfile.TemporaryDirectory() as folder:
    folder = Path(folder)
    data, split = make_fixture(folder/"fixture", samples=900)
    for name, hooks in (("best", best_hooks), ("v9", v9_hooks)):
        with hooks.installed():
            run_desktop.train(data, split, folder/name, tiny, 42, "cpu", synthetic=True)
    states = {name: torch.load(folder/name/"last.pt", weights_only=True) for name in ("best", "v9")}
    differing = []
    for key in sorted(set(states["best"]) | set(states["v9"])):
        a, b = states["best"].get(key), states["v9"].get(key)
        if isinstance(a, dict) and isinstance(b, dict) and all(torch.is_tensor(v) for v in a.values()):
            same = list(a) == list(b) and all(torch.equal(v, b[k]) for k, v in a.items())
        else:
            same = repr(a) == repr(b)
        if not same:
            differing.append(key)
    clocks = ("seconds", "elapsed", "per_second", "windows_per_s", "time")

    def untimed(value):
        if isinstance(value, dict):
            return {k: untimed(v) for k, v in value.items() if not any(c in k for c in clocks)}
        if isinstance(value, list):
            return [untimed(v) for v in value]
        return value

    histories = {name: untimed(json.loads((folder/name/"history.json").read_text())) for name in ("best", "v9")}
    tensors = [k for k in states["best"] if isinstance(states["best"][k], dict)
               and all(torch.is_tensor(v) for v in states["best"][k].values())]
    results["4_training"] = {"checkpoint_keys": sorted(states["best"]), "tensor_dicts_compared": tensors,
                             "differing_keys": differing, "history_equal": histories["best"] == histories["v9"],
                             "epochs": len(histories["best"])}
    results["4_training"]["pass"] = (results["4_training"]["history_equal"]
                                     and not [k for k in differing if k in tensors])
    print("4 training:", results["4_training"], flush=True)

# 5. Full validation of k8_combined through best
configure_runtime()
expected = json.loads((ROOT/"outputs/v9_confirm/k8_combined/selection.json").read_text())
expected = next(c for c in expected["validation_candidates"] if c["seed"] == expected["selected_seed"])
with tempfile.TemporaryDirectory() as folder, best_hooks.installed():
    os.environ["ABC_TCNN_CHECKPOINT"] = str(k8)
    os.environ["ABC_TCNN_DEVICE"] = "cuda"
    report = evaluate(DEFAULT_H5, manifest_for(DEFAULT_H5), Path(folder)/"full_validation",
                      config=EvaluationConfig(batch_size=base["batch_size"]),
                      model_specs=["tcnn_model_adapter:build_model"], latency_steps=200)
got = report["models"]["tiny_tcnn"]["mean_endpoint_deg"]
results["5_full_validation"] = {"got": got, "expected": expected["validation_endpoint_deg"],
                                "cohort_fingerprint_equal": report["cohort_fingerprint"] == expected["cohort_fingerprint"]}
results["5_full_validation"]["pass"] = (got == expected["validation_endpoint_deg"]
                                        and results["5_full_validation"]["cohort_fingerprint_equal"])
print("5 full validation:", results["5_full_validation"], flush=True)

passed = {"1": results["1_configs"]["pass"], "2": all(r["pass"] for r in results["2_checkpoints"].values()),
          "3": all(r["pass"] for r in results["3_exports"].values()), "4": results["4_training"]["pass"],
          "5": results["5_full_validation"]["pass"]}
results["passed"] = passed
results["torch"] = torch.__version__
out = Path(__file__).with_suffix(".json")
out.write_text(json.dumps(results, indent=2), encoding="utf-8")
print("PASSED" if all(passed.values()) else f"FAILED {passed}", "->", out, flush=True)
