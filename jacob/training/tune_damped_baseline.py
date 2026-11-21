"""Tune only on frozen validation users, then evaluate the selected baseline."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import h5py
import numpy as np
from baseline_pipeline import (DEFAULT_H5, EvaluationConfig, evaluate, geodesic_degrees, inventory, load_split,
                               manifest_for, sha256_file, window_batches)
from jsonio import save_json
from tcnn.damped import DampedVelocity

ROOT = Path(__file__).resolve().parent


def tune(data, split, output, lags=(1, 3, 5, 10), taus=(25, 50, 100, 200, 500, None), batch_size=1024):
    data, split, output = Path(data), Path(split), Path(output)
    config = EvaluationConfig(batch_size=batch_size)
    candidates = [{"lag": lag, "tau_ms": tau} for lag in lags for tau in taus]
    if not candidates:
        raise ValueError("Empty baseline grid")
    models = [DampedVelocity(config, **candidate) for candidate in candidates]
    protocol = {"data_sha256": sha256_file(data), "split_sha256": sha256_file(split),
                "split": "validation", "config": asdict(config), "candidates": candidates,
                "code_sha256": {p: sha256_file(ROOT/p) for p in
                                ("tune_damped_baseline.py", "tcnn/damped.py", "baseline_pipeline.py")}}
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "selection.json"
    if lock.exists():
        saved = json.loads(lock.read_text(encoding="utf-8"))
        if any(saved[k] != v for k, v in protocol.items()):
            raise ValueError("Existing damped selection uses different inputs or code")
    else:
        _, users = load_split(split, "validation")
        sums, count = np.zeros(len(models)), 0
        with h5py.File(data, "r") as h5:
            segments, skipped = inventory(h5, users, config, False)
            if skipped or set(users) != {s["user"] for s in segments}:
                raise ValueError("Validation cohort is incomplete or contains skipped segments")
            for segment in segments:
                for _, history, target in window_batches(h5[segment["path"]], config):
                    for i, model in enumerate(models):
                        prediction = model.predict_steps(history, [config.horizon])
                        sums[i] += geodesic_degrees(prediction, target[:, -1:]).sum()
                    count += len(history)
                print(f"Damped baseline search: {count:,} validation windows", flush=True)
        if not count:
            raise ValueError("No validation windows")
        winner = int(np.argmin(sums))
        saved = {**protocol, "windows": count, "selected": candidates[winner],
                 "selection_rule": "Lowest full stride-1 validation endpoint mean; grid order breaks ties",
                 "scores": [{**c, "endpoint_deg": float(s/count)} for c, s in zip(candidates, sums)]}
        save_json(lock, saved)
    report_path = output / "full_validation/results.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report["input"]["sha256"] != protocol["data_sha256"]
                or report["input"]["split_sha256"] != protocol["split_sha256"]
                or report["split"] != "validation" or report["config"] != asdict(config)
                or report["environment"]["pipeline_sha256"] != protocol["code_sha256"]["baseline_pipeline.py"]
                or report["models"]["damped_velocity"]["metadata"]["selection_sha256"] != sha256_file(lock)):
            raise ValueError("Damped baseline report does not match selection")
    else:
        os.environ["ABC_DAMPED_SELECTION"] = str(lock.resolve())
        evaluate(data, split, report_path.parent, config=config, split_name="validation",
                 model_specs=["tcnn.damped:build_model"], latency_steps=200)
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_H5)
    parser.add_argument("--split-file", type=Path, help="Default: the manifest_<run_id>.json next to --data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(tune(args.data, args.split_file or manifest_for(args.data), args.output))


if __name__ == "__main__":
    main()
