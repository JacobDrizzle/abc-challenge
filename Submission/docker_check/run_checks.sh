#!/usr/bin/env bash
# Build the clean-room image on several Linux bases and run check_submission.py in each, network off.
#
#   bash Submission/docker_check/run_checks.sh                         # the folder next to this one
#   bash Submission/docker_check/run_checks.sh path/to/TEAM_headpose.zip   # the archive you will upload
#   TARGETS="cuda py3.12" bash Submission/docker_check/run_checks.sh   # only some targets
#
# Works in Git Bash (Windows), macOS and Linux. Logs go to Submission/docker_check/logs/.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMISSION="${1:-$HERE/../Motion_Pending_headpose}"
LOGS="${LOGS:-$HERE/logs}"
mkdir -p "$LOGS"

# name   base image                              kind      GPU
ALL=(
  "cuda    nvidia/cuda:12.6.3-base-ubuntu24.04     required  gpu"
  "py3.11  python:3.11-slim                        required  -"
  "py3.12  python:3.12-slim                        required  -"
  "py3.13  python:3.13-slim                        required  -"
  "py3.10  python:3.10-slim                        probe     -"
  "py3.14  python:3.14-slim                        probe     -"
)
TARGETS="${TARGETS:-cuda py3.11 py3.12 py3.13 py3.10 py3.14}"

# Docker Desktop on Windows wants C:/... paths; cygpath exists only in Git Bash / MSYS.
winpath() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }

# A .zip is unpacked first, so the check sees exactly what the organisers will unpack.
if [[ "$SUBMISSION" == *.zip ]]; then
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  if command -v unzip >/dev/null 2>&1; then
    unzip -q "$SUBMISSION" -d "$TMP" || { echo "could not unzip $SUBMISSION"; exit 2; }
  else
    python3 -m zipfile -e "$SUBMISSION" "$TMP" || { echo "could not unzip $SUBMISSION"; exit 2; }
  fi
  TOP=("$TMP"/*)
  if [[ ${#TOP[@]} -ne 1 || ! -d "${TOP[0]}" ]]; then
    echo "FAIL: the zip must hold exactly one top-level folder (TEAMNAME_headpose/), found: ${TOP[*]##*/}"
    exit 1
  fi
  CONTEXT="${TOP[0]}"
  [[ "$(basename "$SUBMISSION" .zip)" == "$(basename "$CONTEXT")" ]] ||
    echo "WARN: zip name $(basename "$SUBMISSION") does not match its folder $(basename "$CONTEXT")/"
else
  CONTEXT="$(cd "$SUBMISSION" && pwd)" || exit 2
fi
NAME="$(basename "$CONTEXT")"
HAVE_GPU=no
docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia && HAVE_GPU=yes
echo "Submission: $CONTEXT"
echo "GPU runtime in Docker: $HAVE_GPU"

SUMMARY=()
REQUIRED_FAILED=0
for row in "${ALL[@]}"; do
  read -r tag image kind gpu <<<"$row"
  [[ " $TARGETS " == *" $tag "* ]] || continue
  echo
  echo "################ $tag ($image, $kind)"

  if ! docker build --progress=plain \
        --build-arg BASE="$image" --build-arg SUBMISSION_NAME="$NAME" \
        --build-context checks="$(winpath "$HERE")" \
        -f "$(winpath "$HERE/Dockerfile")" -t "abc-check:$tag" "$(winpath "$CONTEXT")" \
        >"$LOGS/$tag.build.log" 2>&1; then
    echo "Install FAILED. Last lines of $LOGS/$tag.build.log:"
    grep -v '^#[0-9]* \(sha256\|extracting\|resolve\|DONE\|CACHED\)' "$LOGS/$tag.build.log" | tail -n 15
    SUMMARY+=("$(printf '%-7s %-38s %-9s %s' "$tag" "$image" "$kind" "pip install FAILED")")
    [[ $kind == required ]] && REQUIRED_FAILED=1
    continue
  fi

  flags=(--rm --network none)
  note=""
  if [[ $gpu == gpu ]]; then
    if [[ $HAVE_GPU == yes ]]; then flags+=(--gpus all); else note=" (no GPU runtime, ran without GPU)"; fi
  fi
  docker run "${flags[@]}" "abc-check:$tag" 2>&1 | tee "$LOGS/$tag.log"
  status=${PIPESTATUS[0]}
  result="$(grep '^RESULT:' "$LOGS/$tag.log" | tail -n 1)"
  SUMMARY+=("$(printf '%-7s %-38s %-9s %s' "$tag" "$image" "$kind" "${result:-crashed (exit $status)}$note")")
  [[ $status -ne 0 && $kind == required ]] && REQUIRED_FAILED=1
done

echo
echo "################ Summary (logs in $LOGS)"
printf '%-7s %-38s %-9s %s\n' target image kind result
printf '%s\n' "${SUMMARY[@]}"
exit $REQUIRED_FAILED
