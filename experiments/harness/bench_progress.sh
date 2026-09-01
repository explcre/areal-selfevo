#!/usr/bin/env bash
# Progress for a scoring run that is ALREADY RUNNING, without touching it.
#
# The scorer's own bar exists only in recent revisions, and restarting a run to get it would
# discard hours of finished work. This reads the sglang access log, which counts completed chat
# requests, so it works against any running scorer at any revision.
#
#   bash bench_progress.sh <out-dir>          # one reading
#   bash bench_progress.sh <out-dir> watch    # refresh every 30s
#
# <out-dir> is the directory holding server.log, e.g. $OUTROOT/<tag>_olymp
set -u
D="${1:?usage: bench_progress.sh <out-dir> [watch]}"
MODE="${2:-once}"
S="$D/server.log"
[ -f "$S" ] || { echo "no server.log under $D"; exit 2; }

total_for() {
  case "$1" in
    olympiadbench) echo 675 ;; math500) echo 500 ;; amc23) echo 40 ;;
    aime24) echo 30 ;; aime25) echo 30 ;; *) echo 0 ;;
  esac
}

# Total = sum of the benchmarks this shard actually names in its own output. Guessing from the
# tag was wrong: a "core" shard is not always the same four benchmarks, and a hardcoded total
# silently reports the wrong percentage rather than admitting it does not know.
guess_total() {
  local out="${D}.out" meta="$D/run_meta.json" t=0 b n blist=""
  # Prefer what the run DECLARED at startup; fall back to what it has emitted so far.
  if [ -f "$meta" ]; then
    blist="$(grep -oE '"benches":"[^"]*"' "$meta" 2>/dev/null | sed 's/.*:"//;s/"//')"
  fi
  if [ -n "$blist" ]; then
    for b in ${blist//,/ }; do n="$(total_for "$b")"; t=$(( t + n )); done
    echo "$t"; return
  fi
  [ -f "$out" ] || { echo 0; return; }
  for b in olympiadbench math500 amc23 aime24 aime25; do
    if grep -qaE "^${b} +acc=|^NOTE ${b}:|^CAP-LIMITED ${b}:|RUN_META.*${b}" "$out" 2>/dev/null; then
      n="$(total_for "$b")"; t=$(( t + n ))
    fi
  done
  echo "$t"
}

# Elapsed from the FIRST timestamp sglang wrote, not the file mtime. mtime advances with every
# line, so using it reported ~0 seconds elapsed and a rate of 9 requests/second on a run that
# had been going for minutes -- numbers that look precise and are pure artefact.
started_at() {
  local ts
  ts="$(grep -aoE '^\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' "$S" 2>/dev/null | head -1 | tr -d '[')"
  if [ -n "$ts" ]; then date -d "$ts" +%s 2>/dev/null || stat -c %Y "$S"; else stat -c %Y "$S"; fi
}

done_now() { grep -ac "POST /v1/chat/completions" "$S" 2>/dev/null || echo 0; }

report() {
  local d t elapsed rate eta fill w=28 bar
  d="$(done_now)"; t="$(guess_total)"
  elapsed=$(( $(date +%s) - $(started_at) )); [ "$elapsed" -lt 1 ] && elapsed=1
  rate="$(awk -v d="$d" -v e="$elapsed" 'BEGIN{printf "%.3f", d/e}')"
  if [ "$t" -gt 0 ] && [ "$d" -le "$t" ]; then
    fill=$(awk -v d="$d" -v t="$t" -v w="$w" 'BEGIN{printf "%d", (d/t)*w + 0.5}')
    [ "$fill" -gt "$w" ] && fill="$w"; [ "$fill" -lt 0 ] && fill=0
    bar=""; for ((i=0;i<fill;i++)); do bar="$bar#"; done
    for ((i=fill;i<w;i++)); do bar="$bar-"; done
    awk -v b="$bar" -v d="$d" -v t="$t" -v r="$rate" -v e="$elapsed" \
      'BEGIN{eta=(r>0)?(t-d)/r/60:-1; printf "  [%s] %5.1f%%  %d/%d  %.2f/s  elapsed %.0fm  eta %s\n", b, 100*d/t, d, t, r, e/60, (eta>=0)?sprintf("%.0fm",eta):"?"}'
  else
    # No percentage rather than a wrong one. A shard that has not yet named its benchmarks, or
    # has completed more requests than the benchmarks it named (n>1 sampling), cannot be scaled.
    awk -v d="$d" -v r="$rate" -v e="$elapsed" \
      'BEGIN{printf "  %d requests done  %.2f/s  elapsed %.0fm  (total unknown; no percentage)\n", d, r, e/60}'
  fi
}

if [ "$MODE" = "watch" ]; then while true; do report; sleep 30; done; else report; fi
