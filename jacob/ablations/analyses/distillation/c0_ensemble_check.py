"""C0: how much does an ensemble of the existing checkpoints beat a single model? (validation, stride 1)

Read-only on the repo; writes c0_ensemble_check.json next to this script. Each model's error uses the trainer's
geodesic_degrees_torch, so single-model numbers reproduce full_validation. An ensemble prediction is the sign-aligned
mean of the members' unit quaternions, renormalised (the chordal mean; members agree to a few degrees).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

TRAINING = Path(r"<repo>\jacob\training")
sys.path.insert(0, str(TRAINING))

from baseline_pipeline import DEFAULT_H5, manifest_for  # noqa: E402
from tcnn.data import DeviceWindows, check_corpus  # noqa: E402
from train_tcnn import geodesic_degrees_torch  # noqa: E402
from experimental.v7_compact.model import load_checkpoint  # noqa: E402

OUT = Path(__file__).resolve().parent / "c0_ensemble_check.json"
SEEDS = (42, 43, 44)
FOLDERS = {"sep": "outputs/v8_recipe_sep_base3_addons/angular_loss", "b35": "outputs/v8_recipe_base3_5_addons/angular_loss",
           "v6": "outputs/v6_group_a_v2/control"}
MODELS = {f"{k}_s{s}": TRAINING / v / f"seed_{s}" / "best.pt" for k, v in FOLDERS.items() for s in SEEDS}
ENSEMBLES = {"sep3": [f"sep_s{s}" for s in SEEDS], "b35_3": [f"b35_s{s}" for s in SEEDS],
             "v6_3": [f"v6_s{s}" for s in SEEDS], "angular6": [f"{k}_s{s}" for k in ("sep", "b35") for s in SEEDS],
             "all9": list(MODELS)}


def ensemble(predictions):
    """Sign-aligned chordal mean of unit quaternions [B, H, 4]."""
    reference = predictions[0]
    total = torch.zeros_like(reference)
    for p in predictions:
        total += p * torch.where((p * reference).sum(-1, keepdim=True) < 0, -1., 1.)
    return torch.nn.functional.normalize(total, dim=-1)


def main():
    began = time.time()
    data_path = DEFAULT_H5
    audit = check_corpus(data_path, manifest_for(data_path))
    rows = [r for r in audit["segments"] if r["split"] == "validation"]
    windows = DeviceWindows(data_path, rows, "cuda", 400, 20)
    users = sorted({r["user"] for r in rows})
    owner = np.concatenate([np.full(length - 420 + 1, users.index(r["user"]))
                            for r, (_, length) in zip(rows, windows.bounds)])
    owner = torch.from_numpy(owner).cuda()
    models = {name: load_checkpoint(path)[0].cuda().eval() for name, path in MODELS.items()}
    names = [*MODELS, *ENSEMBLES]
    by_step = {n: torch.zeros(20, dtype=torch.float64, device="cuda") for n in names}
    by_user = {n: torch.zeros(len(users), dtype=torch.float64, device="cuda") for n in names}
    count_user = torch.zeros(len(users), dtype=torch.float64, device="cuda")
    seen = 0
    with torch.inference_mode():
        for history, target in windows.windows(1, 4096):
            who = owner[seen:seen + len(history)]
            seen += len(history)
            count_user += torch.bincount(who, minlength=len(users)).double()
            predictions = {name: torch.nn.functional.normalize(model(history), dim=-1) for name, model in models.items()}
            predictions |= {name: ensemble([predictions[m] for m in members]) for name, members in ENSEMBLES.items()}
            for name, prediction in predictions.items():
                error = geodesic_degrees_torch(prediction, target)  # [B, 20]
                by_step[name] += error.sum(0)
                by_user[name] += torch.zeros(len(users), dtype=torch.float64, device="cuda").index_add_(0, who, error[:, -1])
    if seen != len(owner):
        raise ValueError(f"window count mismatch: {seen} vs {len(owner)}")
    counts = count_user.cpu().numpy()
    result = {"windows": seen, "users": users, "user_windows": counts.tolist(), "models": {}}
    for name in names:
        steps = (by_step[name] / seen).cpu().numpy()
        result["models"][name] = {"step20": float(steps[-1]), "step10": float(steps[9]),
                                  "by_step": steps.tolist(), "user_sum20": by_user[name].cpu().numpy().tolist()}
    singles = {n: result["models"][n]["step20"] for n in MODELS}
    best = min(singles, key=singles.get)
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(users), size=(10000, len(users)))
    summary = {}
    for name in ENSEMBLES:
        members = ENSEMBLES[name]
        best_member = min(members, key=lambda m: singles[m])
        for label, other in (("vs_best_single_overall", best), ("vs_best_member", best_member)):
            diff = np.array(result["models"][name]["user_sum20"]) - np.array(result["models"][other]["user_sum20"])
            boot = diff[draws].sum(1) / counts[draws].sum(1)
            summary.setdefault(name, {})[label] = {
                "other": other, "delta": float(diff.sum() / counts.sum()),
                "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                "users_better": int((diff / counts < 0).sum())}
        summary[name]["vs_mean_member"] = float(result["models"][name]["step20"] - np.mean([singles[m] for m in members]))
    result["best_single"] = best
    result["summary"] = summary
    result["seconds"] = time.time() - began
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{seen:,} windows, {len(users)} users, {result['seconds']:.0f} s")
    for n in MODELS:
        print(f"  {n:10s} step20 {singles[n]:.4f}  step10 {result['models'][n]['step10']:.4f}")
    for n in ENSEMBLES:
        s = summary[n]
        print(f"  {n:10s} step20 {result['models'][n]['step20']:.4f}  step10 {result['models'][n]['step10']:.4f}  "
              f"vs mean member {s['vs_mean_member']:+.4f}  vs best member {s['vs_best_member']['delta']:+.4f} "
              f"{s['vs_best_member']['ci95']} ({s['vs_best_member']['users_better']}/{len(users)})  "
              f"vs best single ({best}) {s['vs_best_single_overall']['delta']:+.4f} {s['vs_best_single_overall']['ci95']}")


if __name__ == "__main__":
    main()
