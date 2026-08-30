#!/usr/bin/env bash
# Score a checkpoint series on a held-out math benchmark, one job per GPU.
#
# The question this answers: a training run's own task reward cannot say whether the run is
# helping. For step0d it was actively misleading -- it oscillated 0.209-0.721 with no trend
# while held-out MATH-500 fell 0.528 -> 0.33-0.36, and it ranked the checkpoints in the
# wrong order. So every checkpoint series gets scored on a held-out benchmark that can
# resolve the effect: MATH-500's 500 problems, where AIME's 30 cannot.
#
# The base model is always included as the step-0 anchor. Without it a flat series is
# unreadable, because "no decline" and "already declined before the first checkpoint" look
# identical.
#
# Usage:
#   sweep_entropy.sh [LIMIT]            # default series (step0d), LIMIT=0 scores all 500
#   CKPT_ROOT=... sweep_entropy.sh      # any other run's checkpoint directory
#   LIST=1 sweep_entropy.sh             # print the job list and exit, launching nothing
#
# Watchdog: every job runs under a hard `timeout`. For a bounded eval sweep that is the
# right shape -- harness/watchdog.sh watches AReaL step counters, which an eval never
# emits, so it would read every healthy job as stalled. The timeout guarantees the GPU is
# released whether the job finishes, hangs in sglang startup, or wedges mid-generation.
set -u
LIMIT="${1:-0}"
CKPT_ROOT="${CKPT_ROOT:-/home/ubuntu/areal-runs/checkpoints/ubuntu/step0d/t1/default}"
BASE="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
RUN=/home/ubuntu/areal-selfevo/experiments/bench/run_math.sh
STAMP=$(date +%m%d_%H%M)
SUITE="/home/ubuntu/runs/math/sweep_$STAMP"

# Discover checkpoints and order them by global step NUMERICALLY. A lexical sort puts
# globalstep115 before globalstep28, which would silently plot the series out of order.
mapfile -t CKPTS < <(
  find "$CKPT_ROOT" -maxdepth 1 -type d -name "*globalstep*" 2>/dev/null \
  | while read -r d; do echo "$(basename "$d" | sed 's/.*globalstep//') $d"; done \
  | sort -n | cut -d" " -f2-
)
if [ "${#CKPTS[@]}" -eq 0 ]; then
  echo "NO CHECKPOINTS under $CKPT_ROOT -- refusing to run a base-only sweep"; exit 1
fi

# CKPT_STEPS restricts the series to given global steps, comma-separated. Needed for a
# PAIRED comparison against another run: only steps present in BOTH runs can be compared,
# and scoring the union would silently mix paired and unpaired points in one table.
WANT="${CKPT_STEPS:-}"
JOBS=("base:$BASE")
for d in "${CKPTS[@]}"; do
  n=$(basename "$d" | sed 's/.*globalstep//')
  if [ -n "$WANT" ] && ! echo ",$WANT," | grep -q ",$n,"; then continue; fi
  JOBS+=("gs$(printf '%03d' "$n"):$d")
done
if [ -n "$WANT" ] && [ "${#JOBS[@]}" -eq 1 ]; then
  echo "CKPT_STEPS=$WANT matched no checkpoint under $CKPT_ROOT"; exit 1
fi

if [ -n "${LIST:-}" ]; then
  echo "CKPT_ROOT=$CKPT_ROOT"
  printf '%s\n' "${JOBS[@]}"
  echo "(${#JOBS[@]} jobs; nothing launched)"; exit 0
fi

# One GPU per job. More jobs than GPUs would silently share a device and distort timing.
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "${#JOBS[@]}" -gt "$NGPU" ]; then
  echo "REFUSING: ${#JOBS[@]} jobs but only $NGPU GPUs. Split the series."; exit 1
fi

mkdir -p "$SUITE"
echo "sweep -> $SUITE  (limit=$LIMIT, split=${SPLIT:-all}, root=$CKPT_ROOT)"
gpu=0
for j in "${JOBS[@]}"; do
  tag="${j%%:*}"; model="${j#*:}"
  # A missing checkpoint must fail loudly here, not silently score zero later.
  if [ "$model" != "$BASE" ] && [ ! -f "$model/model.safetensors" ]; then
    echo "MISSING: $tag -> $model" | tee -a "$SUITE/errors.txt"; gpu=$((gpu+1)); continue
  fi
  port=$((8410 + gpu))
  ( BENCHES=math500 MAXTOK="${MAXTOK:-8192}" CONC=48 LIMIT="$LIMIT" MEMFRAC=0.82 \
    SPLIT="${SPLIT:-all}" \
    timeout "${JOB_TIMEOUT:-5400}" bash "$RUN" "$model" "sweep_$STAMP/$tag" "$gpu" "$port" \
      > "$SUITE/$tag.log" 2>&1
    echo "$? $tag" >> "$SUITE/exit_codes.txt" ) &
  echo "  launched $tag on GPU $gpu port $port (timeout ${JOB_TIMEOUT:-5400}s)"
  gpu=$((gpu+1))
done
wait
echo "=== sweep complete ==="
# 124 is timeout's signal that the watchdog fired; it must not be mistaken for a score.
sort "$SUITE/exit_codes.txt" 2>/dev/null
grep -q "^124 " "$SUITE/exit_codes.txt" 2>/dev/null && echo "WARNING: a job was killed by the watchdog (exit 124)"
echo "$SUITE" > /home/ubuntu/runs/math/LAST_SWEEP
echo "analyse with: python3 experiments/bench/regrade.py $SUITE"
