#!/usr/bin/env bash
# Progress for scoring runs that are ALREADY RUNNING, without touching them.
#
# The scorer's own bar exists only in recent revisions, and restarting a run to get it would
# discard hours of finished work. This reads the inference server's access log, which counts
# completed requests, so it reports on any running scorer at any revision.
#
#   bash bench_progress.sh                 # every ACTIVE run, found automatically
#   bash bench_progress.sh watch           # the same, refreshing every 30s
#   bash bench_progress.sh <out-dir>       # one specific run
#   bash bench_progress.sh <out-dir> watch
set -u

MODE=once
DIRS=()
for a in "$@"; do
  case "$a" in
    watch) MODE=watch ;;
    *)     DIRS+=("$a") ;;
  esac
done

# Sourcing the shared env keeps this agreeing with the scripts that WRITE the results. Deriving
# a root independently is how a guard and a scorer ended up checking different paths.
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$_HERE/bench_env.sh" ]; then
  BENCH_REPO="$(cd "$_HERE/../.." 2>/dev/null && pwd)" . "$_HERE/bench_env.sh" >/dev/null 2>&1 || true
fi
OUTROOT="${OUTROOT:-$HOME/runs/math}"

# A run counts as ACTIVE when its server log moved recently. Recency, not name matching: a name
# cannot distinguish a live run from its own corpse, and log movement is the signal that can.
ACTIVE_S="${ACTIVE_S:-600}"

total_for() {
  case "$1" in
    olympiadbench) echo 675 ;; math500) echo 500 ;; amc23) echo 40 ;;
    aime24) echo 30 ;; aime25) echo 30 ;; *) echo 0 ;;
  esac
}

# Prefer what the run DECLARED at startup over what it has finished. Inferring the benchmark set
# from completed benchmarks means the total is unknown exactly while it is needed.
guess_total() {
  local d="$1" meta="$1/run_meta.json" out="${1}.out" t=0 b blist=""
  [ -f "$meta" ] && blist="$(sed -n 's/.*"benches":"\([^"]*\)".*/\1/p' "$meta" 2>/dev/null)"
  if [ -n "$blist" ]; then
    for b in ${blist//,/ }; do t=$(( t + $(total_for "$b") )); done
    echo "$t"; return
  fi
  [ -f "$out" ] || { echo 0; return; }
  for b in olympiadbench math500 amc23 aime24 aime25; do
    grep -qaE "^${b} +acc=|^NOTE ${b}:|^CAP-LIMITED ${b}:|RUN_META.*${b}" "$out" 2>/dev/null \
      && t=$(( t + $(total_for "$b") ))
  done
  echo "$t"
}

# Elapsed from the FIRST timestamp the server wrote. The file's mtime advances with every line,
# so using it reported zero elapsed and an impossible rate on a run going for minutes.
started_at() {
  local s="$1/server.log" ts
  ts="$(grep -aoE '^\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' "$s" 2>/dev/null | head -1 | tr -d '[')"
  if [ -n "$ts" ]; then date -d "$ts" +%s 2>/dev/null || stat -c %Y "$s"; else stat -c %Y "$s"; fi
}

done_now() { grep -ac "POST /v1/chat/completions" "$1/server.log" 2>/dev/null || echo 0; }

discover() {
  local now f
  now="$(date +%s)"
  for f in "$OUTROOT"/*/server.log; do
    [ -f "$f" ] || continue
    [ $(( now - $(stat -c %Y "$f") )) -le "$ACTIVE_S" ] && echo "${f%/server.log}"
  done
}

report() {
  local d="$1" name done_ total elapsed rate
  name="$(basename "$d")"
  [ -f "$d/server.log" ] || { printf '  %-26s no server.log\n' "$name"; return; }
  done_="$(done_now "$d")"; total="$(guess_total "$d")"
  elapsed=$(( $(date +%s) - $(started_at "$d") )); [ "$elapsed" -lt 1 ] && elapsed=1
  rate="$(awk -v a="$done_" -v e="$elapsed" 'BEGIN{printf "%.3f", a/e}')"
  if [ "$total" -gt 0 ] && [ "$done_" -le "$total" ]; then
    awk -v n="$name" -v a="$done_" -v t="$total" -v r="$rate" -v e="$elapsed" 'BEGIN{
      w=24; f=int((a/t)*w+0.5); if(f>w)f=w; if(f<0)f=0; b="";
      for(i=0;i<f;i++)b=b"#"; for(i=f;i<w;i++)b=b"-";
      eta=(r>0)?(t-a)/r/60:-1;
      printf "  %-26s [%s] %5.1f%%  %d/%d  %.2f/s  elapsed %.0fm  eta %s\n",
        n, b, 100*a/t, a, t, r, e/60, (eta>=0)?sprintf("%.0fm",eta):"?" }'
  else
    # No percentage rather than a wrong one.
    awk -v n="$name" -v a="$done_" -v r="$rate" -v e="$elapsed" 'BEGIN{
      printf "  %-26s %d requests  %.2f/s  elapsed %.0fm  (total unknown)\n", n, a, r, e/60 }'
  fi
}

if [ "${#DIRS[@]}" -eq 0 ]; then
  mapfile -t DIRS < <(discover)
  if [ "${#DIRS[@]}" -eq 0 ]; then
    echo "no active scoring run under $OUTROOT"
    echo "  a run is active if its server.log moved in the last ${ACTIVE_S}s;"
    echo "  raise ACTIVE_S=<seconds>, or pass a directory explicitly."
    exit 1
  fi
fi

run_all() { local x; for x in "${DIRS[@]}"; do report "$x"; done; }
if [ "$MODE" = "watch" ]; then
  while true; do echo "-- $(date +%H:%M:%S)"; run_all; sleep 30; done
else
  run_all
fi
