#!/bin/sh
# Refresh this folder from nic_scratch (the workspace). Copies a FIXED list of curated files only, so nothing large or private slips in.
#   nic/publish.sh        then review `git diff`, run betterleaks, and commit yourself.
# To publish something new, add it to the list below (and say why in the commit). archived/ is never touched.
set -e
cd "$(dirname "$0")"
S=../nic_scratch
# findings log (paths inside it point at nic_scratch, which is not in git)
{ echo "> Copy of nic_scratch/FINDINGS.md, refreshed by publish.sh. Paths such as pc_train/outputs/... refer to Nic's local scratch space (not in git)."; echo; cat $S/FINDINGS.md; } > FINDINGS.md
# plain-language results and the comparison table
for f in friday_grid_results saturday_grid_results gads_vs_gru_results where_error_comes_from; do cp $S/docs/results/$f.pdf results/; done
cp $S/pc_train/outputs/eval_table/table.md results/eval_table.md
# the two main ONNX models (made by pc_train/export_main.py)
cp $S/pc_train/outputs/onnx_main/gadst.onnx $S/pc_train/outputs/onnx_main/gru_stateful_step.onnx $S/pc_train/outputs/onnx_main/gru_stateful_window.onnx models/
# training and export code behind models/group_a_v2/ (code/README.md and code/requirements.txt are written by hand, not copied)
mkdir -p code
for f in train.py windows.py grid.py stateful.py onnx_check.py export_main.py h5_to_npz.py ga2.sh ga2_export.sh; do cp $S/pc_train/$f code/; done
# the GRU's warm-up measurement (how many calls after zero state until it scores like the warm stream) and its result
cp $S/warmup/gru_warmup.py code/
cp $S/warmup/gru_s45.json results/gru_s45_warmup.json
# splits: regenerated from the snapshot player lists
(cd $S/pc_train && ../abc_torch/bin/python make_splits.py)
echo "published. files:"; find . -type f ! -name .DS_Store | sort
