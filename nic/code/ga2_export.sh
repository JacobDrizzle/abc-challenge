#!/bin/sh
# Run ON THE MAC from the repo root, after ga2.sh finished on the PC:  sh nic_scratch/pc_train/ga2_export.sh
# 1. copy the ga2_round runs back  2. export each seed's two main models (chip-safe, export_main.py)
# 3. register them on the team bench as trained on the group-a-v2 train/val users  4. score them on group-a-v2-test20
set -e
RUNS=nic_scratch/pc_train/outputs/ga2_round
H5=data/qualified_head_100hz_20261026_131953.h5
TRAINED=bench/trained_on/group_a_v2_20261026_131953_trainval.txt
export BENCH_MACHINE=nic-mac
mkdir -p $RUNS
rsync -ah --include='*/' --include='model.pt' --include='metrics.json' --include='log.txt' --include='grid_runs.json' --exclude='*' pc:/srv/abc/runs/ga2_round/ $RUNS/
params() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['summary']['parameters'])" "$1"; }
for s in 42 43 44 45 46; do
  OUT=nic/models/group_a_v2/s$s
  (cd nic_scratch/pc_train && ../abc_onnx/bin/python export_main.py --gads ../../$RUNS/gads_vel_s$s/model.pt \
      --stateful ../../$RUNS/gruvp30_vel+noq_s$s/model.pt --out ../../$OUT)
  .venv-bench/bin/python -m bench register nic_ga2_gru_s$s $OUT/gru_stateful_window.onnx --trained-on $TRAINED --force \
      --declared-params $(params $RUNS/gruvp30_vel+noq_s$s/metrics.json) --group nic_ga2_gru --owner nic \
      --notes "Stateful GRU, velocity-only input, 32 units, patience 30, seed $s; trained only on group-a-v2 /train, stopped on /validation"
  .venv-bench/bin/python -m bench register nic_ga2_gadst_s$s $OUT/gadst.onnx --trained-on $TRAINED --force \
      --declared-params $(params $RUNS/gads_vel_s$s/metrics.json) --group nic_ga2_gadst --owner nic \
      --notes "GADS-T, rotation + velocity input, seed $s; trained only on group-a-v2 /train, stopped on /validation"
done
for s in 42 43 44 45 46; do
  for m in gru:gru_stateful_window gadst:gadst; do
    .venv-bench/bin/python -m bench score nic/models/group_a_v2/s$s/${m#*:}.onnx --name nic_ga2_${m%%:*}_s$s --h5 $H5 --bench group-a-v2-test20
  done
done
.venv-bench/bin/python -m bench leaderboard
