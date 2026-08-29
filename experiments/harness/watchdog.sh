#!/usr/bin/env bash
# Stall watchdog for an AReaL training run.
#
#   watchdog.sh <pgid-file> <stall-seconds> <log> [log ...]
#
# Kills the run only when the highest "step N/M" counter across ALL given logs is
# unchanged for STRIKES consecutive samples taken STALL seconds apart. Sampling a
# monotonic counter once cannot distinguish progress from a stall -- a previous run was
# reported as healthy on a single sample.
#
# Kill is by recorded PGID, never by pgrep pattern: a pattern kill once matched the
# watcher's own command line and killed the controlling SSH session.
#
# Timing, stated exactly (audit D12): the first sample only records a baseline. A kill
# needs STRIKES further samples that all match it, so the earliest kill is at
# (STRIKES + 1) * STALL seconds. With STALL=1800 and STRIKES=2 that is 90 minutes.
#
# Pass several logs so liveness does not depend on one writer: the filtered launcher log
# AND AReaL's own unfiltered worker log. A counter missing from every log reads as NOSTEP,
# which is a real state that accumulates strikes -- that is how a startup hang is caught.
set -u

PGIDF="${1:?usage: watchdog.sh <pgid-file> <stall-seconds> <log> [log ...]}"
STALL="${2:?}"
shift 2
LOGS=("$@")
STRIKES="${STRIKES:-2}"

# Highest step counter across all logs. Quoting the pattern matters: unquoted, grep takes
# the bracket expression as a second FILE operand, prefixes output with the filename and
# returns a constant forever (audit D1). Only the tail of each log is scanned (D15).
cur() {
    local f n best=""
    for f in "${LOGS[@]}"; do
        [ -f "$f" ] || continue
        n=$(tail -c 2000000 "$f" 2>/dev/null \
            | grep -oE '[Ss]tep [0-9]+/[0-9]+' | tail -1 | grep -oE '^[Ss]tep [0-9]+' \
            | grep -oE '[0-9]+')
        [ -n "$n" ] || continue
        if [ -z "$best" ] || [ "$n" -gt "$best" ] 2>/dev/null; then best="$n"; fi
    done
    printf '%s' "$best"
}

last=""; seen_one=0; strikes=0
while :; do
    sleep "$STALL"

    # A pgid we cannot read is NOT evidence the run ended. Treating it as such silently
    # retires the guard and leaves the run unprotected (audit D4). Reject empty,
    # non-numeric and 0 -- pgid 0 means "my own process group", so killing it would kill
    # the watchdog instead of the trainer (audit D10).
    pgid=$(tr -d '[:space:]' < "$PGIDF" 2>/dev/null || true)
    case "$pgid" in
        ''|*[!0-9]*|0) echo "[watchdog] pgid file unreadable/invalid ('$pgid'); retrying"; continue ;;
    esac
    if ! kill -0 "-$pgid" 2>/dev/null; then
        echo "[watchdog] run (PGID $pgid) has exited; watchdog done"; exit 0
    fi

    now=$(cur); now="${now:-NOSTEP}"
    echo "[watchdog] $(date -Is) progress=${now} prev=${last:-none} strikes=${strikes}"

    if [ "$seen_one" = 1 ] && [ "$now" = "$last" ]; then
        strikes=$((strikes + 1))
        echo "[watchdog] NO PROGRESS across $((strikes + 1)) samples ($(((strikes + 1) * STALL))s)"
        if [ "$strikes" -ge "$STRIKES" ]; then
            echo "[watchdog] STALLED at ${now}; killing PGID $pgid"
            kill -TERM "-$pgid" 2>/dev/null
            sleep 30
            kill -KILL "-$pgid" 2>/dev/null
            exit 1
        fi
    else
        strikes=0
    fi
    last="$now"; seen_one=1
done
