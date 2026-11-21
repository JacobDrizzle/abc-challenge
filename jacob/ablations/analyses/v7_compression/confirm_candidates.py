"""Params, MACs/step, state floats and node counts of the confirm-stage candidates (random weights, no timing)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"<repo>\jacob\training")
sys.path.insert(0, str(ROOT))
import run_next  # noqa: E402
from experimental.v7_compact import hooks  # noqa: E402
from experimental.v7_compact.export import export_streaming_onnx  # noqa: E402
from experimental.v7_compact.model import CompactTCNN  # noqa: E402
from tcnn.onnx_export import audit_operators, count_macs  # noqa: E402

OUT = Path(__file__).parent / "confirm_candidates"
SUITE = ROOT / "experimental/v7_compact/configs"
KEYS = ("context", "horizon", "channels", "dropout", "effective_context", "damped_prior_tau_ms",
        "input_features", "num_blocks", *hooks.MODEL_KEYS)
ADDONS = {"speed_feature": "l1", "head_basis": 3}
CANDIDATES = {"control": {}, "base3_5_addons": {"dilation_base": 3, "num_blocks": 5, **ADDONS},
              "separable_24_addons": {"separable": True, **ADDONS},
              "narrow_full_addons": {"channels": 32, **ADDONS},
              "separable_base3_addons": {"separable": True, "dilation_base": 3, "num_blocks": 5, **ADDONS}}
run = json.loads((SUITE / "v7_run.json").read_text())
run["variants"] = CANDIDATES
with hooks.installed():
    configs = run_next.experiment_configs(run, json.loads((SUITE / "tcnn_v7.json").read_text()))
OUT.mkdir(exist_ok=True)
for name, config in configs.items():
    model = CompactTCNN(**{k: config[k] for k in KEYS}).eval()
    path, shapes = export_streaming_onnx(model, OUT / f"{name}.onnx")
    audit = audit_operators(path)
    print(f"| {name} | {sum(p.numel() for p in model.parameters()):,} | {count_macs(path)['total']:,} | "
          f"{int(sum(np.prod(s) for s in shapes)):,} | {sum(n for op, n in audit['counts'].items() if op != 'Constant')} | "
          f"{', '.join(audit['other']) or 'none'} |")
