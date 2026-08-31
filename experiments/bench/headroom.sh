#!/usr/bin/env bash
# Measure how much room a candidate base model actually has on the target benchmarks.
#
# This gates the decision to scale. A routing method can only demonstrate an effect where
# the base model has room to improve AND the benchmark can resolve the difference. Both
# halves have already bitten this project once: an earlier Qwen3.8-27B run reported 73.3%
# on AIME24/25, but with so many generations hitting the token cap that the true value lay
# anywhere in [0.733, 1.000]. That number measured the budget, not the model.
#
# So every arm here runs at a deliberately large token budget and the truncation rate is
# reported beside the score. A score whose truncation rate is not near zero is not a
# measurement of the model.
#
# One model per GPU group, all in parallel, each under a hard timeout.
set -u
STAMP=$(date +%m%d_%H%M)
SUITE="$HOME/runs/math/headroom_$STAMP"
mkdir -p "$SUITE"
RUN=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_math.sh
BENCHES="${BENCHES:-math500,aime24,aime25,amc23}"

# tag : model : gpus : concurrency
# Concurrency falls with model size: a 32k budget at high concurrency exhausts the KV cache
# and surfaces as timeouts that would be misread as model failures.
# tag : model : gpus : concurrency : max_new_tokens
#
# The budget is PER MODEL because it must leave room for the prompt inside the context.
# Setting it equal to context_len made every request error instantly: the 1.5B and 7B both
# have context_len=32768, so a 32768-token request left zero room and 1100 problems failed
# in under two minutes. The 27B has context_len=262144 and was unaffected, which is exactly
# why the bug first looked like a small-model problem rather than a configuration one.
JOBS=(
  "qwen2.5-1.5b:Qwen/Qwen2.5-1.5B-Instruct:0:32:24576"
  "qwen2.5-7b:Qwen/Qwen2.5-7B-Instruct:1:24:24576"
  "qwen3.8-27b:Qwen/Qwen3.8-27B:2,3:12:32768"
)

echo "headroom -> $SUITE"
echo "benchmarks: $BENCHES   budget: per-model, overridable with MAXTOK"
for j in "${JOBS[@]}"; do
  IFS=: read -r tag model gpus conc mtok <<< "$j"
  port=$((8500 + ${gpus%%,*}))
  ( BENCHES="$BENCHES" MAXTOK="${MAXTOK:-$mtok}" CONC="$conc" MEMFRAC=0.88 \
    TIMEOUT=1800 SPLIT=all \
    timeout "${JOB_TIMEOUT:-21600}" bash "$RUN" "$model" "headroom_$STAMP/$tag" "$gpus" "$port" \
      > "$SUITE/$tag.log" 2>&1
    echo "$? $tag" >> "$SUITE/exit_codes.txt" ) &
  echo "  launched $tag on GPU $gpus port $port conc=$conc maxtok=${MAXTOK:-$mtok}"
done
wait
echo "=== headroom complete ==="
sort "$SUITE/exit_codes.txt" 2>/dev/null
grep -q "^124 " "$SUITE/exit_codes.txt" 2>/dev/null && echo "WARNING: a job hit the watchdog timeout"
echo "$SUITE" > $HOME/runs/math/LAST_HEADROOM
