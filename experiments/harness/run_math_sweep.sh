#!/usr/bin/env bash
# Score a LIST of models on the frozen math suite, unattended, resuming across restarts.
#
# Built for a dedicated box that can run for days. Every model is fetched, scored on both
# shards, and recorded; a model whose results already exist is skipped, so killing this at any
# point and re-running it loses at most the model in flight.
#
#   bash run_math_sweep.sh --plan     # show what WOULD run, download nothing
#   bash run_math_sweep.sh --run      # smoke-test the path, then do it, with a watchdog
#
# --run ALWAYS smokes first, on a tiny model, and refuses to start the real sweep if that
# fails. SKIP_SMOKE=1 bypasses it; SMOKE_MODEL=<hf id> changes which model is used.
#
# MODELS is a comma-separated list of HF ids. The default is the set we need re-scored at a
# cap large enough for reasoning models: every number we hold for these was measured at a cap
# that truncated a third to three quarters of generations, which makes them lower bounds on a
# token budget rather than measurements of the model.
set -u -o pipefail

MODE="${1:---plan}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTROOT="${OUTROOT:-$HOME/runs/math}"
LOG="$OUTROOT/sweep.log"
MAXTOK="${MAXTOK:-32768}"
# The sweep smoke-tests itself before committing to days of work. A TINY model is used on
# purpose: the point is to prove the path -- venv, sglang, serving, generation, grading -- not
# to measure anything, and a 0.5B downloads in seconds where the real models take many minutes.
# A broken path discovered after a 60GB download costs an evening; discovered here it costs
# five minutes.
SMOKE_MODEL="${SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
MODELS="${MODELS:-\
deepseek-ai/DeepSeek-R1-Distill-Qwen-32B,\
Qwen/Qwen2.5-Math-72B-Instruct,\
Qwen/Qwen2.5-32B-Instruct,\
Qwen/Qwen2.5-Math-7B-Instruct}"

mkdir -p "$OUTROOT"
say(){ echo "[$(date +%m-%d_%H:%M:%S)] $*" | tee -a "$LOG"; }

model_tag(){ echo "$1" | tr '/' '_'; }

model_done(){
  # A model counts as done only when BOTH shards have a parseable results.json. A partially
  # scored model is re-run, because a half-filled row is worse than an empty one.
  local tag; tag="$(model_tag "$1")"
  # run_h200_math.sh writes two shards on a box with enough GPUs and a single _all shard
  # otherwise. Checking only the two-shard layout would re-score a finished model forever on
  # a small box, which is the sort of loop that quietly burns a weekend.
  if [ -s "$OUTROOT/${tag}_all/results.json" ]; then return 0; fi
  [ -s "$OUTROOT/${tag}_olymp/results.json" ] && [ -s "$OUTROOT/${tag}_core/results.json" ]
}

if [ "$MODE" = "--plan" ]; then
  echo "sweep plan (max_tokens=$MAXTOK, out=$OUTROOT)"
  IFS=',' read -ra MS <<< "$MODELS"
  for m in "${MS[@]}"; do
    [ -z "$m" ] && continue
    if model_done "$m"; then echo "  SKIP  $m  (already scored)"; else echo "  RUN   $m"; fi
  done
  echo
  echo "to start:  bash $0 --run"
  exit 0
fi

[ "$MODE" = "--run" ] || { echo "Usage: $0 --plan | --run"; exit 2; }

# Watchdog: the sweep is healthy only while the log keeps growing. It reports, and never kills
# anything -- a scoring run that looks stalled may be mid-generation on a long problem, and
# killing it would discard hours of completed work.
(
  prev=""; stall=0
  while true; do
    sleep 900
    kill -0 "$$" 2>/dev/null || exit 0
    now="$(stat -c %s "$LOG" 2>/dev/null || echo 0)"
    if [ "$now" = "$prev" ]; then stall=$((stall+1)); else stall=0; fi
    [ "$stall" -ge 4 ] && echo "[watchdog] sweep log has not grown in $((stall*15)) min" >> "$LOG"
    prev="$now"
  done
) >/dev/null 2>&1 &
WD=$!
trap 'kill "$WD" 2>/dev/null' EXIT

if [ "${SKIP_SMOKE:-0}" = "1" ]; then
  say "smoke SKIPPED by SKIP_SMOKE=1"
else
  say "smoke: $SMOKE_MODEL (proves the path before any long download)"
  if ! MODEL="$SMOKE_MODEL" timeout 1800 bash "$HERE/run_h200_math.sh" --fetch >> "$LOG" 2>&1; then
    say "SMOKE FAILED: could not fetch $SMOKE_MODEL. Nothing else will work either; stopping."
    say "  see $LOG. Run 'bash $HERE/run_h200_math.sh --install' first if the venv is missing."
    exit 1
  fi
  if ! MODEL="$SMOKE_MODEL" timeout 3600 bash "$HERE/run_h200_math.sh" --smoke >> "$LOG" 2>&1; then
    say "SMOKE FAILED: $SMOKE_MODEL did not score end to end. Stopping before the real sweep."
    say "  last 30 lines of $LOG:"
    tail -30 "$LOG" | sed 's/^/    /'
    exit 1
  fi
  say "smoke PASSED; the serve/generate/grade path works on this box"
fi

say "sweep starting: max_tokens=$MAXTOK"
IFS=',' read -ra MS <<< "$MODELS"
for m in "${MS[@]}"; do
  [ -z "$m" ] && continue
  if model_done "$m"; then say "SKIP $m (already scored)"; continue; fi
  say "=== $m ==="
  if ! MODEL="$m" timeout 21600 bash "$HERE/run_h200_math.sh" --fetch >> "$LOG" 2>&1; then
    say "FETCH FAILED for $m; moving on (nothing else depends on it)"
    continue
  fi
  say "fetched $m; scoring"
  MODEL="$m" TAG="$(model_tag "$m")" MAXTOK="$MAXTOK" \
    timeout 172800 bash "$HERE/run_h200_math.sh" --run >> "$LOG" 2>&1 \
    && say "DONE $m" || say "SCORING FAILED for $m (exit $?); moving on"
done

say "sweep complete"
echo
echo "==== results ===="
grep -hE "^(math500|amc23|aime24|aime25|olympiadbench)" "$OUTROOT"/*.out 2>/dev/null | sort -u
echo
echo "==== cap-limited warnings (these scores are LOWER BOUNDS) ===="
grep -hE "CAP-LIMITED" "$OUTROOT"/*.out 2>/dev/null | sort -u
