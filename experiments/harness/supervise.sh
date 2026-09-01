#!/usr/bin/env bash
# Supervise an AReaL run: restart it on exit, and kill it when the log stops growing.
#
# GPU utilization is NOT a liveness signal for this stack. When a rollout server drops
# during a weight push the surviving training ranks spin in a dead NCCL collective and
# nvidia-smi keeps reporting 100%. Log growth is the signal that actually tracks progress.
#
# Usage: supervise.sh <launch-script> <run-dir> [max_restarts] [stall_seconds] [startup_seconds]
#
# Two fuses, because the two failures look different. A steady-state stall shows a log that
# stops growing. A startup failure shows a log that grows steadily -- server boot chatter --
# while no training step ever appears, so the stall check never fires and the run waits
# forever. One rollout server failing to bind is enough to produce it.
set -u -o pipefail
LAUNCH="${1:?launch script}"; RUN="${2:?run dir}"
MAX_RESTARTS="${3:-6}"; STALL_S="${4:-1200}"; STARTUP_S="${5:-900}"
# Poll period. Configurable ONLY so the stall guard can be tested in seconds instead
# of minutes -- a watchdog that cannot be exercised is a watchdog nobody has checked,
# and this one silently failed to fire on two real stalls.
POLL_S="${POLL_S:-120}"
LOG="$RUN/train.log"; SUP="$RUN/supervisor.log"
TAG=$(basename "$LAUNCH" .sh)
mkdir -p "$RUN"

say() { echo "[$(date -Is)] $*" >> "$SUP"; }
say "supervisor start: launch=$LAUNCH run=$RUN max_restarts=$MAX_RESTARTS stall=${STALL_S}s"

for attempt in $(seq 0 "$MAX_RESTARTS"); do
  say "attempt $attempt: launching"
  # Recovery is enabled on EVERY attempt, including the first. Enabling it only on retries
  # was self-defeating: with mode disabled the trainer writes no recover checkpoint, so a
  # retry asking to resume had nothing to resume FROM and restarted at step zero regardless.
  # Measured on the live 30B run -- seventeen hours in, recover.freq_secs 3600, and zero
  # recover checkpoints on disk. The retry path existed and could not have worked.
  #
  # Safe on a fresh run: the documented meaning of 'auto' is to recover only if recover info
  # and checkpoints are available, so with none present it starts normally. What it changes is
  # that the checkpoints now get WRITTEN, which is the thing a later attempt needs.
  case " ${EXTRA_ARGS:-} " in
    *" recover.mode="*) : ;;                       # an explicit caller setting wins
    *) export EXTRA_ARGS="${EXTRA_ARGS:-} recover.mode=auto" ;;
  esac
  bash "$LAUNCH" >> "$SUP" 2>&1 &
  RUN_PID=$!

  # Stall watchdog for this attempt.
  ( started=$(date +%s)
    while kill -0 "$RUN_PID" 2>/dev/null; do
      sleep "$POLL_S"
      # Startup fuse: the log may be growing with server boot chatter while no training step
      # has ever appeared, in which case the stall check below never fires.
      if ! grep -qE "step [0-9]+/" "$LOG" 2>/dev/null; then
        boot=$(( $(date +%s) - started ))
        if [ "$boot" -gt "$STARTUP_S" ]; then
          say "WATCHDOG: no training step after ${boot}s (> ${STARTUP_S}s startup budget)"
          nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$SUP" 2>&1
          pkill -u "$USER" -f "experiment_name=${TAG}" 2>/dev/null
          pkill -9 -u "$USER" -f "inference_service.sglang.launch_server" 2>/dev/null
          sleep 10
          pkill -9 -u "$USER" -f "experiment_name=${TAG}" 2>/dev/null
          break
        fi
        continue
      fi
      # PROGRESS, not log growth. Measured twice on 2026-08-31: `ctx` died at step 162 and
      # `ctxpc` at step 152, and in BOTH cases this watchdog never fired, because the
      # rollout proxy kept writing `ProxyRolloutServer INFO: Cleaned up N stale sessions`
      # into the same log every few seconds for the entire stall. File mtime therefore
      # stayed fresh while training was dead -- 16 minutes undetected in one case, 40 in
      # the other. A log that is growing is not a run that is progressing, and the only
      # thing that distinguishes them is whether the step counter moves.
      #
      # mtime is KEPT as a second signal: a log that stops entirely is also dead, and that
      # is the case this check caught correctly before.
      cur_step=$(grep -aoE "step [0-9]+/" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
      now=$(date +%s)
      if [ -n "$cur_step" ] && [ "$cur_step" != "${last_step:-}" ]; then
        last_step="$cur_step"; last_step_at="$now"
      fi
      : "${last_step_at:=$now}"
      step_age=$(( now - last_step_at ))
      mt=$(stat -c %Y "$LOG" 2>/dev/null || echo 0)
      age=$(( now - mt ))
      if [ "$step_age" -gt "$STALL_S" ]; then
        say "WATCHDOG: no step past ${last_step:-?} for ${step_age}s (> ${STALL_S}s); log mtime age is only ${age}s, so the log was still being written while training was dead; dumping stacks then killing"
      elif [ "$mt" -gt 0 ] && [ "$age" -gt "$STALL_S" ]; then
        say "WATCHDOG: $LOG stalled ${age}s (> ${STALL_S}s); dumping stacks then killing"
        for pid in $(pgrep -u "$USER" -f "rpc_server.*--role actor" | head -4); do
          say "--- stacks for $pid ---"
          # py-spy needs ptrace permission and is silently useless without it, so record the
          # kernel wait-channel too: it is always readable and distinguishes a process
          # blocked in a collective from one blocked on I/O.
          (py-spy dump --pid "$pid" 2>&1 | head -30) >> "$SUP" 2>&1 || say "py-spy failed"
          say "wchan=$(cat /proc/$pid/wchan 2>/dev/null || echo unreadable) state=$(awk '"'"'/^State:/{print $2}'"'"' /proc/$pid/status 2>/dev/null || echo ?)"
        done
        # The trainer is launched with `experiment_name=<tag>` -- an UNDERSCORE and an EQUALS.
        # This previously matched "experiment-name ${TAG}", which appears nowhere on any
        # command line, so the watchdog detected the stall, logged it, and killed nothing:
        # the dead run kept its GPUs and the next attempt collided with it in name_resolve.
        pkill -u "$USER" -f "experiment_name=${TAG}" 2>/dev/null
        pkill -u "$USER" -f "${TAG}\.sh" 2>/dev/null
        # The trainer carries `experiment_name=<tag>` (a Hydra override); the rpc_server
        # workers carry `--experiment-name <tag>` (a CLI flag). ONE run carries both, and
        # matching only the first left 96 workers alive holding 24-70 GB each.
        pkill -u "$USER" -f -- "-experiment-name ${TAG}" 2>/dev/null
        sleep 15
        pkill -9 -u "$USER" -f "experiment_name=${TAG}" 2>/dev/null
        pkill -9 -u "$USER" -f -- "-experiment-name ${TAG}" 2>/dev/null
        # sglang servers do not carry the experiment name on their command line, so they
        # survive the pattern above and hold their GPU memory into the next attempt.
        pkill -9 -u "$USER" -f "inference_service.sglang.launch_server" 2>/dev/null
        sleep 5
        left=$(pgrep -u "$USER" -f "experiment.name.${TAG}" | wc -l)
        say "after kill: $left processes still match experiment_name=${TAG}"
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
