#!/usr/bin/env bash
# One bench score from a job line "name|onnx|declared_params|group"; log to bench_logs/<name>.log.
IFS='|' read -r name onnx params group <<< "$1"
SP=<scratch>
cd <repo> || exit 1
extra=()
[ -n "$group" ] && extra=(--group "$group")
start=$(date +%s)
<local_test>/.venv/Scripts/python.exe -m bench score "$onnx" --name "$name" \
  --h5 data/h5/qualified_head_100hz_20261026_131953.h5 \
  --trained-on bench/trained_on/group_a_v2_20261026_131953_trainval.txt --declared-params "$params" "${extra[@]}" \
  > "$SP/bench_logs/$name.log" 2>&1
code=$?
echo "$name exit $code $(( $(date +%s) - start ))s"
