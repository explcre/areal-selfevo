#!/usr/bin/env bash
# Supervise an AReaL run: restart it on exit, and kill it when the log stops growing.
#
# GPU utilization is NOT a liveness signal for this stack. When a rollout server drops
# during a weight push the surviving training ranks spin in a dead NCCL collective and
# nvidia-smi keeps reporting 100%. Log growth is the signal that actually tracks progress.
#
# Usage: supervise.sh <launch-script> <run-dir> [max_restarts] [stall_seconds]
set -u -o pipefail
LAUNCH="${1:?launch script}"; RUN="${2:?run dir}"
MAX_RESTARTS="${3:-6}"; STALL_S="${4:-1200}"
LOG="$RUN/train.log"; SUP="$RUN/supervisor.log"
TAG=$(basename "$LAUNCH" .sh)
mkdir -p "$RUN"

say() { echo "[$(date -Is)] $*" >> "$SUP"; }
say "supervisor start: launch=$LAUNCH run=$RUN max_restarts=$MAX_RESTARTS stall=${STALL_S}s"

for attempt in $(seq 0 "$MAX_RESTARTS"); do
  say "attempt $attempt: launching"
  # First attempt keeps whatever the caller set; later attempts must resume, not restart.
  if [ "$attempt" -gt 0 ]; then export EXTRA_ARGS="${EXTRA_ARGS:-} recover.mode=auto"; fi
  bash "$LAUNCH" >> "$SUP" 2>&1 &
  RUN_PID=$!

  # Stall watchdog for this attempt.
  ( while kill -0 "$RUN_PID" 2>/dev/null; do
      sleep 120
      mt=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
      age=$(( $(date +%s) - mt ))
      if [ "$mt" -gt 0 ] && [ "$age" -gt "$STALL_S" ]; then
        say "WATCHDOG: $LOG stalled ${age}s (> ${STALL_S}s); dumping stacks then killing"
        for pid in $(pgrep -u "$USER" -f "rpc_server.*--role actor" | head -4); do
          say "--- py-spy $pid ---"
          (py-spy dump --pid "$pid" 2>&1 | head -30) >> "$SUP" 2>&1 || say "py-spy unavailable"
        done
        pkill -u "$USER" -f "experiment-name ${TAG}" 2>/dev/null
        sleep 15
        pkill -9 -u "$USER" -f "experiment-name ${TAG}" 2>/dev/null
        break
      fi
    done ) &
  WD_PID=$!

  wait "$RUN_PID"; rc=$?
  kill "$WD_PID" 2>/dev/null; wait "$WD_PID" 2>/dev/null
  say "attempt $attempt exited rc=$rc"

  if [ "$rc" -eq 0 ]; then say "run completed cleanly"; exit 0; fi
  if [ "$rc" -eq 3 ]; then say "another run holds the lock; not restarting"; exit 3; fi
  # Let GPU memory drain before the next attempt.
  sleep 45
done
say "exhausted $MAX_RESTARTS restarts"
exit 1
