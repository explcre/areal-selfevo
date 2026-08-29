#!/usr/bin/env bash
# Score the step0d checkpoint series on a held-out math benchmark.
#
# The question this answers: step0d's policy entropy collapsed 14x (0.253 -> 0.018)
# while its train reward showed no trend at all (0.55, 0.21, 0.72, 0.49, 0.55, 0.48).
# Train reward therefore cannot say whether the entropy collapse cost real capability.
# Only a held-out benchmark can, so every checkpoint is scored on MATH-500 -- 500
# problems, enough to resolve a ~4-point difference, where AIME's 30 cannot.
#
# The base model is included as the step-0 anchor: without it a flat series is
# unreadable, because "no decline" and "already declined before the first checkpoint"
# look identical.
#
# Each job owns one GPU and one port. Usage: sweep_entropy.sh [LIMIT]
#
# Watchdog: every job runs under a hard `timeout`. For a bounded eval sweep that is the
# right shape -- harness/watchdog.sh watches AReaL step counters, which an eval never
# emits, so it would read every healthy job as stalled. The timeout guarantees the GPU is
# released whether the job finishes, hangs in sglang startup, or wedges mid-generation.
set -u
LIMIT="${1:-0}"
CKPT=/home/ubuntu/areal-runs/checkpoints/ubuntu/step0d/t1/default
BASE=Qwen/Qwen2.5-1.5B-Instruct
RUN=/home/ubuntu/areal-selfevo/experiments/bench/run_math.sh
STAMP=$(date +%m%d_%H%M)
SUITE=/home/ubuntu/runs/math/sweep_$STAMP
mkdir -p "$SUITE"

# tag:model pairs, in training order. base first so it takes GPU 0.
JOBS=(
  "base:$BASE"
  "gs028:$CKPT/epoch0epochstep28globalstep28"
  "gs057:$CKPT/epoch1epochstep28globalstep57"
  "gs086:$CKPT/epoch2epochstep28globalstep86"
  "gs115:$CKPT/epoch3epochstep28globalstep115"
  "gs144:$CKPT/epoch4epochstep28globalstep144"
  "gs173:$CKPT/epoch5epochstep28globalstep173"
)

echo "sweep -> $SUITE  (limit=$LIMIT)"
gpu=0
for j in "${JOBS[@]}"; do
  tag="${j%%:*}"; model="${j#*:}"
  # A missing checkpoint must fail loudly here, not silently score zero later.
  if [ "$model" != "$BASE" ] && [ ! -f "$model/model.safetensors" ]; then
    echo "MISSING: $tag -> $model" | tee -a "$SUITE/errors.txt"; gpu=$((gpu+1)); continue
  fi
  port=$((8410 + gpu))
  ( BENCHES=math500 MAXTOK="${MAXTOK:-8192}" CONC=48 LIMIT="$LIMIT" MEMFRAC=0.82 \
    timeout "${JOB_TIMEOUT:-5400}" bash "$RUN" "$model" "sweep_$STAMP/$tag" "$gpu" "$port" \
      > "$SUITE/$tag.log" 2>&1
    echo "$? $tag" >> "$SUITE/exit_codes.txt" ) &
  echo "  launched $tag on GPU $gpu port $port (timeout ${JOB_TIMEOUT:-5400}s)"
  gpu=$((gpu+1))
done
wait
echo "=== sweep complete ==="
# 124 is timeout's signal that the watchdog fired; it must not be mistaken for a score.
cat "$SUITE/exit_codes.txt" 2>/dev/null
grep -q "^124 " "$SUITE/exit_codes.txt" 2>/dev/null && echo "WARNING: a job was killed by the watchdog (exit 124)"
echo "$SUITE" > /home/ubuntu/runs/math/LAST_SWEEP
