"""Rescore the Table 3 rows of the paper on the PC (benchmark group-a-v2-test20).

Runs `python -m bench score` for each model and baseline, then copies the result JSONs to outputs/
so prun brings them back. Data is read from /data (the PC's /srv/data, read-only).
"""
import shutil
import subprocess
import sys
from pathlib import Path

H5 = "/data/qualified_head_100hz_20261026_131953.h5"
RUNS = [  # (model argument, registry name or None for baselines)
    ("models/tcnn_v5_ga2_stream.onnx", "v5_tcnn_ga2"),
    ("models/tcnn_v6_ga2_stream.onnx", "v6_tcnn_ga2"),
    ("models/tcnn_v7_sep_base3_addons_angular_ga2_stream.onnx", "v7_sep_base3_addons_angular_ga2"),
    ("models/tcnn_v9_k8_combined_ga2_stream.onnx", "v9_k8_combined_ga2"),
    ("models/final_k8_combined_v10_s42.onnx", "final_k8_combined_v10_s42"),
    ("baseline:damped_velocity", None),
    ("baseline:hold_last", None),
    ("baseline:constant_velocity", None),
]

failed = []
for model, name in RUNS:
    cmd = [sys.executable, "-m", "bench", "score", model, "--h5", H5, "--machine", "nic-pc", "--protocol", "from-start"]
    if name:
        cmd += ["--name", name]
    print("\n>>>", " ".join(cmd), flush=True)
    if subprocess.run(cmd).returncode != 0:
        failed.append(model)

out = Path("outputs/group-a-v2-test20")
out.mkdir(parents=True, exist_ok=True)
for f in Path("bench/results/group-a-v2-test20").glob("*@nic-pc.json"):
    shutil.copy(f, out / f.name)
print("\nfailed:", failed or "none")
sys.exit(1 if failed else 0)
