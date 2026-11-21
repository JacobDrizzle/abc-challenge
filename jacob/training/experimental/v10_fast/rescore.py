"""Score one checkpoint on its validation windows with LeanTCNN.forward and with the fast forward, side by side.

python -m experimental.v10_fast.rescore --checkpoint outputs/<folder>/selected_model.pt [--data <h5>]
    [--stride 1] [--batch 2048] [--device cuda]      (from jacob/training)

The evidence that the two forwards are one function on real data: both step-20 means over every validation window
(stride 1 is the trainer's full validation, so the LeanTCNN mean should be the number in the folder's selection.json),
the largest difference between the two forecasts over all windows, and the time each forward took. Reads only the
validation users of --data and writes nothing.
"""
import argparse
import json
from pathlib import Path
import time

import h5py
import torch

from baseline_pipeline import DEFAULT_H5, EvaluationConfig, inventory, load_split, manifest_for
from tcnn.data import DeviceWindows
from tcnn.runtime import configure_runtime
from train_tcnn import geodesic_degrees_torch

from best.model import load_checkpoint

from experimental.v10_fast.model import FastLeanTCNN


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=DEFAULT_H5, help="ray/ export HDF5; its validation users are scored")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    configure_runtime()  # the trainer's numerical policy
    device = torch.device(args.device)
    _, users = load_split(manifest_for(args.data), "validation")
    with h5py.File(args.data, "r") as h5:
        rows, _ = inventory(h5, users, EvaluationConfig(), False)
    data = DeviceWindows(args.data, rows, device)
    lean, state = load_checkpoint(args.checkpoint)
    fast = FastLeanTCNN(**state["model_spec"]).eval()
    fast.load_state_dict(state["model_state"])
    models = {"LeanTCNN": lean.to(device), "fast": fast.to(device)}
    if fast.positions is None:
        p.error("This model has no short forward (FastLeanTCNN would run LeanTCNN.forward)")

    sums = {name: torch.zeros((), dtype=torch.float64, device=device) for name in models}
    seconds, windows, largest = dict.fromkeys(models, 0.), 0, 0.
    with torch.inference_mode():
        for history, target in data.windows(args.stride, args.batch):
            forecasts = {}
            for name, model in models.items():
                began = time.perf_counter()
                forecasts[name] = model(history)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                seconds[name] += time.perf_counter() - began
                sums[name] += geodesic_degrees_torch(forecasts[name][:, -1], target[:, -1]).sum()
            largest = max(largest, (forecasts["fast"] - forecasts["LeanTCNN"]).abs().max().item())
            windows += len(history)

    means = {name: (value / windows).item() for name, value in sums.items()}
    print(f"{args.checkpoint}: {len(users)} validation users, {windows:,} windows at stride {args.stride}")
    for name in models:
        print(f"  {name:8s} step-20 mean {means[name]:.12f} deg   forward time {seconds[name]:.1f} s")
    print(f"  fast - LeanTCNN: {means['fast'] - means['LeanTCNN']:+.2e} deg in the mean; "
          f"largest difference in any forecast component over all windows {largest:.2e}")
    selection = args.checkpoint.parent / "selection.json"
    if selection.exists() and args.stride == 1:
        locked = json.loads(selection.read_text(encoding="utf-8"))
        recorded = {c["seed"]: c["validation_endpoint_deg"] for c in locked["validation_candidates"]}
        print(f"  recorded in selection.json for seed {locked['selected_seed']}: "
              f"{recorded[locked['selected_seed']]:.12f} deg")


if __name__ == "__main__":
    main()
