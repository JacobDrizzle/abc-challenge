"""Check v9_jitter_combined.json: control == the config k8_combined was trained with; variants differ only in jitter."""
import json
from pathlib import Path
import sys

TRAINING = Path(r"<repo>\jacob\training")
sys.path.insert(0, str(TRAINING))

import run_next
from experimental.v9_lean.hooks import installed

HERE = TRAINING / "experimental/v9_lean/configs"
with installed():
    configs = run_next.experiment_configs(json.loads((HERE / "v9_jitter_combined.json").read_text()),
                                          json.loads((HERE / "tcnn_v9.json").read_text()))
trained = json.loads((TRAINING / "outputs/v9_confirm/k8_combined/experiment.json").read_text())["config"]
print("control == trained k8_combined config:", configs["control"] == trained)
if configs["control"] != trained:
    for key in sorted(set(configs["control"]) | set(trained)):
        if configs["control"].get(key) != trained.get(key):
            print("  differs:", key, configs["control"].get(key), trained.get(key))
for name in ("combined_jitter_1_5ms", "combined_jitter_3ms"):
    diff = {k: (configs["control"].get(k), v) for k, v in configs[name].items() if configs["control"].get(k) != v}
    print(name, diff)
