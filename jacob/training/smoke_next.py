"""Exercise every next-run variant on synthetic data, including a full-size batch."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from jsonio import save_json
from run_next import experiment_configs, run_suite
from smoke_tcnn import make_fixture
from tcnn.model import TinyTCNN, rotation_loss, rotvec_to_quaternion
from tcnn.runtime import configure_runtime

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path)
    p.add_argument("--suite", type=Path, default=ROOT/"configs/v5_run.json")
    p.add_argument("--base-config", type=Path, default=ROOT/"configs/tcnn_v5.json")
    args = p.parse_args()
    output = args.output or ROOT/"outputs"/datetime.now(timezone.utc).strftime("next_smoke_%Y%m%dT%H%M%S%f")
    if output.exists():
        p.error("Choose a new smoke output directory")
    runtime = configure_runtime()
    torch.manual_seed(42)
    suite = json.loads(args.suite.read_text())
    base = json.loads(args.base_config.read_text())
    checks = []
    for name, config in experiment_configs(suite, base).items():
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        model = TinyTCNN(context=400, horizon=20, channels=config["channels"], dropout=config["dropout"],
                         effective_context=config["effective_context"], velocity_features=config["velocity_features"],
                         damped_prior_tau_ms=config["damped_prior_tau_ms"], input_features=config["input_features"],
                         num_blocks=config["num_blocks"]).to(args.device)
        batch = config["batch_size"]
        history = rotvec_to_quaternion(torch.randn(batch, 400, 3, device=args.device)*.1)
        target = rotvec_to_quaternion(torch.randn(batch, 20, 3, device=args.device)*.1)
        loss = rotation_loss(model(history), target, config["loss_kind"], config["endpoint_weight"])
        loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError(f"Non-finite training step: {name}")
        checks.append({"variant": name, "batch_size": batch, "loss": loss.item(),
                       "peak_allocated_mb": torch.cuda.max_memory_allocated()/1e6 if args.device.startswith("cuda") else None})
        print(f"Batch-{batch} training check passed: {name}", flush=True)
        del model, history, target, loss
    data, _ = make_fixture(output/"data")
    base.update(batch_size=4, steps_per_epoch=2, epochs=2, workers=0 if base.get("data_on_device") else 2,
                cache_mb_per_worker=1, validation_stride=20)
    selected = run_suite(data, output/"suite", suite, base, device=args.device, synthetic=True)
    resumed = run_suite(data, output/"suite", suite, base, device=args.device, synthetic=True)
    if selected != resumed:
        raise ValueError("Resume changed the frozen selection")
    save_json(output/"SMOKE_PASSED.json", {"passed": True, "synthetic": True, "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "runtime": runtime, "training_batch_checks": checks, "suite_resume_passed": True,
        "score_is_not_a_research_result": True, "test_evaluated": False})
    print(f"All next-run smoke checks passed: {output}", flush=True)


if __name__ == "__main__":
    main()
