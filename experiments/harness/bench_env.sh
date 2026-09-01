#!/usr/bin/env bash
# ONE definition of where benchmark artefacts live, sourced by every script in this path.
#
# It exists because the roots disagreed. run_h200_math.sh resolved a root and checked the
# benchmark data under it, but math_bench.py defaulted to $HOME and run_math_sweep.sh wrote its
# log to $HOME, so a collaborator saw "benchmark data present (math500 has 500 rows)" and then a
# FileNotFoundError for the same benchmark one second later. A guard that validates a different
# path from the one the program reads is worse than no guard: it manufactures confidence.
#
# Everything below is EXPORTED, so a child process inherits the same answer instead of
# recomputing its own.
#
# Usage:  BENCH_REPO=<repo root> . experiments/harness/bench_env.sh

# Free space in GiB for a path that may not exist yet. df prints nothing for a missing path,
# which rendered a chosen root as "auto: G free" -- a blank where a number belongs. Walk up to
# the nearest existing ancestor, which is the filesystem it will be created on anyway.
bench_free_gb() {
  local d="$1"
  while [ -n "$d" ] && [ "$d" != "/" ] && [ ! -d "$d" ]; do d="$(dirname "$d")"; done
  local v; v="$(df -BG --output=avail "${d:-/}" 2>/dev/null | tail -1 | tr -dc '0-9')"
  echo "${v:-0}"
}

_bench_pick_root() {
  # Candidates: $HOME, and a directory BESIDE the checkout -- beside, never inside, since a venv
  # plus tens of GB of weights inside the working tree would pollute `git status` and a
  # `git clean` would delete it. Take the sibling only when it is meaningfully bigger, so a box
  # where both live on one filesystem keeps its familiar layout.
  local beside home_free beside_free
  beside="$(cd "${BENCH_REPO:-.}/.." 2>/dev/null && pwd)/areal-bench"
  home_free="$(bench_free_gb "$HOME")"
  beside_free="$(bench_free_gb "$(dirname "$beside")")"
  if [ "$beside_free" -gt "$(( home_free * 2 ))" ] && [ "$beside_free" -gt 50 ]; then
    echo "$beside"
  else
    echo "$HOME"
  fi
}

if [ -z "${BENCH_ROOT:-}" ]; then
  BENCH_ROOT="$(_bench_pick_root)"
  BENCH_ROOT_WHY="auto: $(bench_free_gb "$BENCH_ROOT")G free (set BENCH_ROOT=<dir> to override)"
else
  BENCH_ROOT_WHY="set explicitly"
fi
export BENCH_ROOT BENCH_ROOT_WHY

# Every derived path is exported so children agree with the parent by construction.
export BENCH_VENV="${BENCH_VENV:-$BENCH_ROOT/bench-env}"
export OUTROOT="${OUTROOT:-$BENCH_ROOT/runs/math}"
export HF_HOME="${HF_HOME:-$BENCH_ROOT/hf_cache}"
export MATH_EVAL_DATA="${MATH_EVAL_DATA:-$BENCH_ROOT/baselines/Absolute-Zero-Reasoner/evaluation/math_eval/eval/data}"

bench_env_report() {
  echo "  BENCH_ROOT=$BENCH_ROOT  [$BENCH_ROOT_WHY]"
  echo "    venv=$BENCH_VENV"
  echo "    out=$OUTROOT"
  echo "    hf_cache=$HF_HOME"
  echo "    data=$MATH_EVAL_DATA"
}
