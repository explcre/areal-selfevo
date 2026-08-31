#!/usr/bin/env bash
# End-to-end test of the stall watchdog against the REAL failure signature.
#
# Both A100 stalls on 2026-08-31 shared the property that defeated the old check: the train
# log kept GROWING (the rollout proxy wrote "Cleaned up N stale sessions" every few seconds)
# while the step counter stood still. A watchdog keyed on file mtime cannot fire on that.
#
# NOTE on the harness contract: supervise.sh runs `bash "$LAUNCH"` with NO arguments, so a
# fake launcher must know its own log path. An earlier version of this test passed the run
# dir as $1, the launcher wrote to /train.log, every write failed with Permission denied, and
# the resulting "watchdog fired" looked like a real false positive. The test was wrong.
set -u
REPO=/home/ubuntu/areal-selfevo
BASE=$(mktemp -d /home/ubuntu/wdtest.XXXXXX)
trap 'rm -rf "$BASE"' EXIT
PASS=0; FAIL=0
ck(){ if [ "$2" = "$3" ]; then echo "  ok   $1"; PASS=$((PASS+1)); else echo "  FAIL $1: got [$2] want [$3]"; FAIL=$((FAIL+1)); fi; }

mk(){ # $1=case dir, $2=launcher body
  mkdir -p "$1"
  { echo '#!/usr/bin/env bash'; echo "LOG=$1/train.log"; echo "$2"; } > "$1/launch.sh"
  chmod +x "$1/launch.sh"
}

# --- case 1: log grows, step frozen -> MUST fire (the bug that cost two runs) ---
C1="$BASE/c1"
mk "$C1" 'echo "Epoch 1/10 Step 5/29 Train step 5/290" >> "$LOG"
while true; do echo "ProxyRolloutServer INFO: Cleaned up 12 stale sessions" >> "$LOG"; sleep 1; done'
POLL_S=2 timeout 60 bash "$REPO/experiments/harness/supervise.sh" "$C1/launch.sh" "$C1" 0 8 40 >/dev/null 2>&1
grep -q "no step past 5" "$C1/supervisor.log" 2>/dev/null && r=fired || r=missed
ck "log growing + step frozen -> fires" "$r" "fired"
n=$(grep -c "Cleaned up" "$C1/train.log" 2>/dev/null || echo 0)
ck "the log really kept growing (>3 chatter lines)" "$([ "${n:-0}" -gt 3 ] && echo yes || echo no)" "yes"
a=$(( $(date +%s) - $(stat -c %Y "$C1/train.log" 2>/dev/null || echo 0) ))
ck "mtime was fresh, so the OLD mtime check could NOT have fired" "$([ "$a" -lt 8 ] && echo fresh || echo stale)" "fresh"

# --- case 2: steps advancing -> MUST NOT fire ---
C2="$BASE/c2"
mk "$C2" 'i=1; while [ $i -le 40 ]; do echo "Epoch 1/10 Step $i/29 Train step $i/290" >> "$LOG"; sleep 1; i=$((i+1)); done'
POLL_S=2 timeout 45 bash "$REPO/experiments/harness/supervise.sh" "$C2/launch.sh" "$C2" 0 8 40 >/dev/null 2>&1
grep -q "WATCHDOG" "$C2/supervisor.log" 2>/dev/null && r=fired || r=quiet
ck "steps advancing -> stays quiet (no false positive)" "$r" "quiet"
s=$(grep -aoE "step [0-9]+/" "$C2/train.log" 2>/dev/null|tail -1|grep -oE "[0-9]+")
ck "case 2 really did advance past step 5" "$([ "${s:-0}" -gt 5 ] && echo yes || echo no)" "yes"

# --- case 3: log stops entirely -> MUST still fire (old signal preserved) ---
C3="$BASE/c3"
mk "$C3" 'echo "Epoch 1/10 Step 3/29 Train step 3/290" >> "$LOG"; sleep 600'
POLL_S=2 timeout 60 bash "$REPO/experiments/harness/supervise.sh" "$C3/launch.sh" "$C3" 0 8 40 >/dev/null 2>&1
grep -q "WATCHDOG" "$C3/supervisor.log" 2>/dev/null && r=fired || r=missed
ck "log stops entirely -> still fires" "$r" "fired"

echo; echo "passed=$PASS failed=$FAIL"
[ "$FAIL" -eq 0 ]
