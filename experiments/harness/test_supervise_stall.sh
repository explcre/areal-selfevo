#!/usr/bin/env bash
# End-to-end test of the stall watchdog against the REAL failure signature.
#
# Both stalls on 2026-08-31 shared one property that defeated the old check: the train log
# kept GROWING (the rollout proxy wrote "Cleaned up N stale sessions" every few seconds)
# while the step counter stood still. A watchdog keyed on file mtime can never fire on that.
# This reproduces exactly that: a fake run that emits one step, then writes cleanup chatter
# forever without ever advancing.
set -u
REPO=/home/ubuntu/areal-selfevo
TMP=$(mktemp -d /home/ubuntu/wdtest.XXXXXX)
trap 'pkill -f "$TMP" 2>/dev/null; rm -rf "$TMP"' EXIT
PASS=0; FAIL=0
ck(){ if [ "$2" = "$3" ]; then echo "  ok   $1"; PASS=$((PASS+1)); else echo "  FAIL $1: got [$2] want [$3]"; FAIL=$((FAIL+1)); fi; }

# --- case 1: log grows, step frozen -> MUST fire (this is the bug that cost two runs) ---
mkdir -p "$TMP/c1"
cat > "$TMP/c1/launch.sh" <<'INNER'
#!/usr/bin/env bash
LOG="$1/train.log"
echo "Epoch 1/10 Step 5/29 Train step 5/290" >> "$LOG"
while true; do
  echo "ProxyRolloutServer INFO: Cleaned up 12 stale sessions" >> "$LOG"
  sleep 1
done
INNER
chmod +x "$TMP/c1/launch.sh"
POLL_S=2 timeout 90 bash "$REPO/experiments/harness/supervise.sh" \
  "$TMP/c1/launch.sh" "$TMP/c1" 0 8 6 >/dev/null 2>&1
grep -q "no step past 5" "$TMP/c1/supervisor.log" 2>/dev/null && r=fired || r=missed
ck "log growing + step frozen -> watchdog fires" "$r" "fired"
# the log must genuinely have kept growing, or the test proved nothing
n=$(grep -c "Cleaned up" "$TMP/c1/train.log" 2>/dev/null || echo 0)
ck "the log really was still growing (>3 chatter lines)" "$([ "${n:-0}" -gt 3 ] && echo yes || echo no)" "yes"
mt_age=$(( $(date +%s) - $(stat -c %Y "$TMP/c1/train.log") ))
ck "mtime was fresh, so the OLD check could not have fired" "$([ "$mt_age" -lt 8 ] && echo fresh || echo stale)" "fresh"

# --- case 2: steps advancing -> MUST NOT fire (no false positives) ---
mkdir -p "$TMP/c2"
cat > "$TMP/c2/launch.sh" <<'INNER'
#!/usr/bin/env bash
LOG="$1/train.log"
i=1
while [ $i -le 40 ]; do
  echo "Epoch 1/10 Step $i/29 Train step $i/290" >> "$LOG"
  sleep 1; i=$((i+1))
done
INNER
chmod +x "$TMP/c2/launch.sh"
POLL_S=2 timeout 45 bash "$REPO/experiments/harness/supervise.sh" \
  "$TMP/c2/launch.sh" "$TMP/c2" 0 8 6 >/dev/null 2>&1
grep -q "WATCHDOG" "$TMP/c2/supervisor.log" 2>/dev/null && r=fired || r=quiet
ck "steps advancing -> watchdog stays quiet" "$r" "quiet"

# --- case 3: log stops entirely -> MUST still fire (the old signal must survive) ---
mkdir -p "$TMP/c3"
cat > "$TMP/c3/launch.sh" <<'INNER'
#!/usr/bin/env bash
LOG="$1/train.log"
echo "Epoch 1/10 Step 3/29 Train step 3/290" >> "$LOG"
sleep 600
INNER
chmod +x "$TMP/c3/launch.sh"
POLL_S=2 timeout 90 bash "$REPO/experiments/harness/supervise.sh" \
  "$TMP/c3/launch.sh" "$TMP/c3" 0 8 6 >/dev/null 2>&1
grep -q "WATCHDOG" "$TMP/c3/supervisor.log" 2>/dev/null && r=fired || r=missed
ck "log stops entirely -> watchdog still fires" "$r" "fired"

echo
echo "passed=$PASS failed=$FAIL"
[ "$FAIL" -eq 0 ]
