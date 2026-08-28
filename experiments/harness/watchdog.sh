#!/usr/bin/env bash
# Stall watchdog for a Step 0 run.
#   $1 = log file to watch   $2 = file holding the trainer PGID   $3 = stall timeout (s)
# Kills the run only if the "step N/233" counter is UNCHANGED across two samples
# separated by $3 seconds. Sampling a monotonic counter once cannot tell progress
# from a stall -- the previous run was reported as healthy on a single sample.
# Kills by recorded PGID, never by pgrep pattern: a pattern kill previously matched
# the watcher's own command line and killed the controlling SSH session.
LOG="$1"; PGIDF="$2"; STALL="${3:-1800}"
cur() { grep -oE 'step [0-9]+/[0-9]+' "$LOG" 2>/dev/null | grep -oE '[0-9]+' | head -1 | tail -1; }
last=""; strikes=0
while :; do
  sleep "$STALL"
  pgid=$(cat "$PGIDF" 2>/dev/null)
  [ -n "$pgid" ] && kill -0 "-$pgid" 2>/dev/null || { echo "[watchdog] run gone; exiting"; exit 0; }
  now=$(grep -oE step [0-9]+/[0-9]+ "$LOG" 2>/dev/null | tail -1)
  echo "[watchdog] $(date -Is) progress=${now:-none} prev=${last:-none}"
  if [ -n "$last" ] && [ "$now" = "$last" ]; then
    strikes=$((strikes+1))
    echo "[watchdog] NO PROGRESS for $((strikes*STALL))s (strike $strikes/2)"
    if [ "$strikes" -ge 2 ]; then
      echo "[watchdog] STALLED at ${now:-unknown}; killing PGID $pgid"
      kill -TERM "-$pgid" 2>/dev/null; sleep 30; kill -KILL "-$pgid" 2>/dev/null
      exit 1
    fi
  else
    strikes=0
  fi
  last="$now"
done
