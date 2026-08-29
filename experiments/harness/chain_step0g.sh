#!/usr/bin/env bash
# Launch step0g detached, with a watchdog that reads step0g's own logs.
#
# Two bugs in chain_step0f.sh are fixed here, both from copy-paste of chain_step0c.sh:
#
#   1. It waited on "$HOME/runs/step0f/pgid" -- its OWN pgid file -- while the comment
#      claimed it was waiting for the previous run to exit. It therefore waited on nothing.
#   2. It pointed the watchdog at .../logs/ubuntu/step0c/t1/main.log, a finished run's log.
#      watchdog.sh takes the HIGHEST step counter across all logs it is given, so a stale
#      log pins that maximum at its final value forever. A healthy step0g would have read
#      as frozen and been killed after 90 minutes.
#
# step0e and step0f were both killed abruptly (train.log truncated mid-line, no traceback,
# no STEP0*_EXIT marker, empty watchdog.log) after 7 and 12 minutes. The cause is not in
# the logs. This launcher cannot prevent that, but it records enough to tell next time:
# the pgid, the start time, and a heartbeat file the watchdog updates.
set -u
H="$HOME/areal-selfevo/experiments/harness"
RUN="$HOME/runs/step0g"
mkdir -p "$RUN"

# Preflight: refuse to start on busy GPUs rather than OOM ten minutes in.
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
if [ "${used:-99999}" -ge 2000 ]; then
    echo "REFUSING TO START: a GPU already holds ${used} MiB. Free it first."; exit 1
fi
echo "preflight ok (max GPU used=${used} MiB) at $(date -Is)"

setsid bash "$H/step0g.sh" > "$RUN/outer.log" 2>&1 < /dev/null &
sleep 3
pgid=$(ps -o pgid= -p $! 2>/dev/null | tr -d ' ')
if [ -z "$pgid" ]; then
    echo "TRAINER DIED WITHIN 3s -- see $RUN/outer.log"; tail -20 "$RUN/outer.log"; exit 2
fi
echo "$pgid" > "$RUN/pgid"
echo "step0g trainer PGID=$pgid started $(date -Is)"

# Both logs belong to step0g. Liveness must not depend on one writer, and must never
# depend on a different experiment's log.
setsid bash "$H/watchdog.sh" "$RUN/pgid" 1800 \
    "$RUN/train.log" \
    "$HOME/areal-runs/logs/ubuntu/step0g/t1/main.log" \
    > "$RUN/watchdog.log" 2>&1 < /dev/null &
echo "watchdog pid=$! (earliest kill is 3 samples x 1800s = 90 min of a frozen counter)"
