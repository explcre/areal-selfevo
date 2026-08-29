#!/usr/bin/env bash
# Behavioural tests for the audit fixes. Run on the A100 host; touches only ~/scratch.
set -u
H="$HOME/harness_new"
T="$HOME/scratch/harness_test"; rm -rf "$T"; mkdir -p "$T"
pass=0; fail=0
ok()   { echo "  PASS  $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL  $1"; fail=$((fail+1)); }

echo "== D2: progress lines must NEVER be suppressed =="
: > "$T/in.txt"
for i in $(seq 1 233); do
  echo "(AReaL) 20260829-00:00:00.000 StatsLogger INFO: Epoch 1/1 Step $i/233 Train step $i/233 done." >> "$T/in.txt"
done
python3 "$H/logfilter.py" < "$T/in.txt" > "$T/out.txt"
got=$(grep -oE 'Step [0-9]+/233' "$T/out.txt" | tail -1)
n=$(grep -c 'Train step' "$T/out.txt")
[ "$got" = "Step 233/233" ] && ok "last step survives ($got, $n lines)" || bad "last step is '$got' (expected Step 233/233), $n lines"

echo "== still caps a genuine storm =="
: > "$T/storm.txt"
for i in $(seq 1 1000); do echo "ERROR: All output_tokens are EOS or PAD tokens; cannot strip" >> "$T/storm.txt"; done
python3 "$H/logfilter.py" < "$T/storm.txt" > "$T/storm_out.txt"
sn=$(wc -l < "$T/storm_out.txt")
[ "$sn" -lt 300 ] && ok "1000 identical lines -> $sn lines out" || bad "storm not capped: $sn lines"

echo "== D13: ordinary words are not collapsed =="
printf 'the facade decade beefed\n' > "$T/w.txt"
python3 "$H/logfilter.py" < "$T/w.txt" > "$T/w_out.txt"
grep -q 'facade decade beefed' "$T/w_out.txt" && ok "words preserved" || bad "words mangled: $(cat "$T/w_out.txt")"

echo "== D1/D6: cur() returns the LAST step number, not a constant =="
cat > "$T/fake.log" <<'LOGEOF'
(AReaL) StatsLogger INFO: Epoch 1/1 Step 41/233 Train step 41/233 done.
(AReaL) StatsLogger INFO: Epoch 1/1 Step 42/233 Train step 42/233 done.
(AReaL) StatsLogger INFO: Epoch 1/1 Step 137/233 Train step 137/233 done.
LOGEOF
# shellcheck disable=SC1090
LOGS=("$T/fake.log")
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
c=$(cur)
[ "$c" = "137" ] && ok "cur() = 137" || bad "cur() = '$c' (expected 137)"

echo "== D4/D10: invalid pgid must NOT retire the watchdog =="
for bad_pgid in "" "abc" "0"; do
  echo "$bad_pgid" > "$T/pgid"
  out=$(STRIKES=1 timeout 8 bash "$H/watchdog.sh" "$T/pgid" 2 "$T/fake.log" 2>&1)
  rc=$?
  if [ $rc -eq 124 ] && echo "$out" | grep -q "unreadable/invalid"; then
    ok "pgid '$bad_pgid' -> keeps retrying (did not exit 0)"
  else
    bad "pgid '$bad_pgid' -> rc=$rc out=$(echo "$out" | head -1)"
  fi
done

echo "== D1 end-to-end: an ADVANCING counter must NOT be killed =="
setsid bash -c 'while :; do sleep 1; done' & victim=$!
sleep 1
vpg=$(ps -o pgid= -p $victim | tr -d ' ')
echo "$vpg" > "$T/pgid2"
: > "$T/live.log"
( for i in 1 2 3 4 5 6 7 8 9 10; do
    echo "(AReaL) StatsLogger INFO: Epoch 1/1 Step $i/233 Train step $i/233 done." >> "$T/live.log"
    sleep 1
  done ) &
STRIKES=1 timeout 9 bash "$H/watchdog.sh" "$T/pgid2" 2 "$T/live.log" > "$T/wd.log" 2>&1
if kill -0 "-$vpg" 2>/dev/null; then ok "advancing counter survived"; else bad "healthy run was KILLED"; fi
kill -TERM "-$vpg" 2>/dev/null
grep -oE 'progress=[^ ]+' "$T/wd.log" | head -3 | sed 's/^/        /'

echo "== a genuinely FROZEN counter must be killed =="
setsid bash -c 'while :; do sleep 1; done' & v2=$!
sleep 1
vpg2=$(ps -o pgid= -p $v2 | tr -d ' ')
echo "$vpg2" > "$T/pgid3"
echo "(AReaL) StatsLogger INFO: Epoch 1/1 Step 77/233 Train step 77/233 done." > "$T/frozen.log"
STRIKES=1 timeout 12 bash "$H/watchdog.sh" "$T/pgid3" 2 "$T/frozen.log" > "$T/wd2.log" 2>&1
sleep 1
if kill -0 "-$vpg2" 2>/dev/null; then bad "frozen run was NOT killed"; kill -TERM "-$vpg2" 2>/dev/null; else ok "frozen run killed"; fi

echo
echo "RESULT: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
