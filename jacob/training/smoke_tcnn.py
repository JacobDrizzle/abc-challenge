"""Exercise training, checkpoint loading and scoring on clearly labelled synthetic data."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from baseline_pipeline import SPLITS, EvaluationConfig, evaluate, sha256_file
from jsonio import save_json
from train_tcnn import train

ROOT = Path(__file__).resolve().parent


def make_fixture(directory, samples=500):
    """A synthetic ray/ (group-a-v2) export: one user under each split group, plus its manifest_<run_id>.json."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    run_id = "synthetic"
    data, split = directory / f"qualified_head_100hz_{run_id}.h5", directory / f"manifest_{run_id}.json"
    splits = {name: [f"synthetic_{name}"] for name in SPLITS}
    with h5py.File(data, "w") as h5:
        h5.attrs.update(quaternion_order="wxyz", rate_hz=100, synthetic=True, run_id=run_id)
        h5.attrs.create("splits", list(SPLITS), dtype=h5py.string_dtype())
        for i, name in enumerate(SPLITS):
            t = np.arange(samples)/100
            q = Rotation.from_euler("z", (25*np.sin(1.5*t+i))[:,None], degrees=True).as_quat()[:, [3,0,1,2]]
            group = h5.create_group(f"{name}/{splits[name][0]}/recording/segment_000")
            group.create_dataset("time_s", data=t)
            group.create_dataset("quaternion_wxyz", data=q.astype(np.float32))
            group.attrs["challenge_filter_pass"] = True
    save_json(split, {"synthetic": True, "run_id": run_id, "file": data.name,
                      "data_sha256": sha256_file(data), "splits": splits})
    return data, split


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    output = args.output or ROOT / "outputs" / f"synthetic_smoke_{stamp}"
    if output.exists():
        p.error("Choose a new smoke output directory")
    data, split = make_fixture(output / "data")
    config = json.loads((ROOT / "configs/tcnn_3080.json").read_text())
    config.update(batch_size=4, steps_per_epoch=2, epochs=2, workers=2,
                  cache_mb_per_worker=1, validation_stride=20)
    checkpoint = train(data, split, output / "run", config, 42, args.device, synthetic=True)
    os.environ["ABC_TCNN_CHECKPOINT"] = str(checkpoint)
    os.environ["ABC_TCNN_DEVICE"] = args.device
    report = evaluate(data, split, output / "validation", config=EvaluationConfig(batch_size=8),
                      model_specs=["tcnn_model_adapter:build_model"], latency_steps=5)
    score = report["models"]["tiny_tcnn"]["mean_endpoint_deg"]
    if not np.isfinite(score):
        raise ValueError("Smoke evaluation returned a non-finite score")
    save_json(output / "SMOKE_PASSED.json", {"passed": True, "synthetic": True,
              "device": args.device, "torch": str(torch.__version__), "score_is_not_a_research_result": True})
    print(f"Synthetic smoke check passed: {output}")


if __name__ == "__main__":
    main()
