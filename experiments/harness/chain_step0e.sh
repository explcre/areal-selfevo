#!/usr/bin/env bash
# Wait for step0c to release the GPUs, then launch step0c with the fixed guards.
set -u
H="$HOME/areal-selfevo/experiments/harness"

# Wait for the step0c process group to exit (bounded).
pg=$(cat "$HOME/runs/step0e/pgid" 2>/dev/null || echo "")
for _ in $(seq 1 120); do
    [ -n "$pg" ] && kill -0 "-$pg" 2>/dev/null || break
    sleep 5
done

# Wait for GPU memory to actually drain; a dead PGID does not mean freed VRAM.
for _ in $(seq 1 120); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "${used:-99999}" -lt 2000 ] && break
    sleep 5
done
echo "GPUs drained (max used=${used:-?} MiB); launching step0c at $(date -Is)"

mkdir -p "$HOME/runs/step0e"
setsid bash "$H/step0e.sh" > "$HOME/runs/step0e/outer.log" 2>&1 < /dev/null &
sleep 3
pgid=$(ps -o pgid= -p $! 2>/dev/null | tr -d ' ')
echo "$pgid" > "$HOME/runs/step0e/pgid"
echo "step0c trainer PGID=$pgid"

# Watch BOTH the filtered launcher log and AReaL's own unfiltered worker log, so the
# liveness signal never depends on a single writer.
setsid bash "$H/watchdog.sh" "$HOME/runs/step0e/pgid" 1800 \
    "$HOME/runs/step0e/train.log" \
    "$HOME/areal-runs/logs/ubuntu/step0c/t1/main.log" \
    > "$HOME/runs/step0e/watchdog.log" 2>&1 < /dev/null &
echo "watchdog pid=$! (kills only after 3 samples / 90 min of a frozen counter)"
