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

# A kill without a cause invites an identical relaunch. step0g deadlocked in the evaluator
# and the watchdog recorded only "no progress" -- the stack that named _evaluate had to be
# taken by hand before the kill landed. Dump once, on the first strike, while the process
# is still hung.
DUMP_DIR="${DUMP_DIR:-$(dirname "$PGIDF")}"
_dumped=0
dump_stacks() {
    [ "$_dumped" -eq 0 ] || return 0
    command -v py-spy >/dev/null 2>&1 || return 0
    local pg out
    pg=$(cat "$PGIDF" 2>/dev/null) || return 0
    out="$DUMP_DIR/stall_stacks_$(date +%s).txt"
    # The trainer is the python process in the group with the largest RSS.
    for p in $(pgrep -g "$pg" -f python 2>/dev/null | head -4); do
        # py-spy needs ptrace rights this host does not grant unprivileged; without the
        # sudo attempt the dump writes "Permission Denied" instead of a stack, which is a
        # guard that looks armed and is not. Fall back to the plain call where sudo is
        # available passwordless-less or unnecessary.
        { echo "===== pid $p ====="
          ( sudo -n env "PATH=$PATH" py-spy dump --pid "$p" 2>/dev/null \
            || timeout 60 py-spy dump --pid "$p" 2>&1 ) | head -60
        } >> "$out"
    done
    [ -s "$out" ] && echo "[watchdog] stall stacks written to $out"
    _dumped=1
}

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
        dump_stacks; echo "[watchdog] NO PROGRESS across $((strikes + 1)) samples ($(((strikes + 1) * STALL))s)"
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
