#!/usr/bin/env bash
# Serve a model and score it on the frozen math suite. Kills the server on every exit path.
set -u
export PATH="$HOME/.local/bin:$PATH"
# Which interpreter serves and scores. Hardcoding one box's venv made this script unusable
# on any other machine, which is exactly what a collaborator hit.
BENCH_VENV="${BENCH_VENV:-$HOME/venv312b}"
if [ -f "$BENCH_VENV/bin/activate" ]; then
  source "$BENCH_VENV/bin/activate"
else
  echo "no venv at $BENCH_VENV (set BENCH_VENV=<path>); falling back to $(command -v python3)" >&2
fi
ulimit -n 131072 || true
MODEL="${1:?model path or hf id}"; TAG="${2:?tag}"; GPUS="${3:-0,1}"; PORT="${4:-8404}"
# The id the server REGISTERS and the id the scorer ASKS FOR, from one variable. They
# must match: an unregistered id is answered HTTP 200 by the BASE model, so two
# literals that drift apart would silently score the wrong weights. math_bench.py now
# refuses an id that is not in /v1/models, which turns that drift into a stop.
SERVED_NAME="${SERVED_NAME:-evalmodel}"
# Follows OUTROOT/BENCH_ROOT so every caller writes to the same volume; $HOME is only the
# fallback, and on a container it is often the wrong (small) one.
OUT="${OUTROOT:-${BENCH_ROOT:-$HOME}/runs/math}/$TAG"; mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES="$GPUS" python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 --port "$PORT" --tp $(echo "$GPUS" | tr ',' '\n' | wc -l) \
  --mem-fraction-static "${MEMFRAC:-0.85}" ${SGL_EXTRA:-} > "$OUT/server.log" 2>&1 &
SRV=$!
cleanup(){ [ -n "${SRV:-}" ] && kill -TERM "$SRV" 2>/dev/null; }
trap cleanup EXIT INT TERM
for _ in $(seq 1 180); do
  kill -0 "$SRV" 2>/dev/null || { echo "SERVER DIED"; tail -25 "$OUT/server.log"; exit 4; }
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || { echo "NOT READY"; tail -25 "$OUT/server.log"; exit 5; }
# The client timeout must scale with the generation budget AND with how many generations share
# the GPU. A fixed 600s was fine at 3072 tokens and 16-way concurrency; at 65536 tokens and 40-way
# it killed 662 of 675 requests, which the scorer then reported as an accuracy over the 13
# survivors. Raising concurrency raises per-request latency, so changing one knob without the
# other converts a speed-up into a mass timeout that looks like a low score.
#   budget/20 tokens-per-second is deliberately pessimistic: measured decode is ~90 tok/s per
#   sequence, so this leaves better than 4x headroom before a healthy request is cut off.
_MT="${MAXTOK:-3072}"; _CC="${CONC:-64}"
DEFAULT_TIMEOUT=$(( _MT / 20 + _CC * 5 + 300 ))
[ "$DEFAULT_TIMEOUT" -lt 600 ] && DEFAULT_TIMEOUT=600
echo "TIMEOUT=${TIMEOUT:-$DEFAULT_TIMEOUT}s (max_tokens=$_MT conc=$_CC)"

echo "endpoint up; scoring $TAG"
# Declare the run's shape BEFORE any of it finishes. Without this a progress reader can only
# learn which benchmarks a shard covers by watching them complete, which is precisely when the
# information stops being useful -- a half-done shard reported "total unknown" while sitting at
# 59%. Written to the output dir as well as stdout so a monitor need not parse the log.
printf '{"tag":"%s","benches":"%s","max_tokens":%s,"concurrency":%s,"started":%s}\n' \
  "$TAG" "${BENCHES:-aime24,aime25,amc23,math500}" "${MAXTOK:-3072}" "${CONC:-64}" "$(date +%s)" \
  > "$OUT/run_meta.json"
echo "RUN_META benches=${BENCHES:-aime24,aime25,amc23,math500} max_tokens=${MAXTOK:-3072} conc=${CONC:-64}"
# Repo copy, never a stale $HOME copy: numbers must come from audited code.
python3 "$(dirname "$0")/math_bench.py" --base-url "http://127.0.0.1:$PORT/v1" \
  --model "$SERVED_NAME" \
  --benchmarks "${BENCHES:-aime24,aime25,amc23,math500}" \
  --max-tokens "${MAXTOK:-3072}" --concurrency "${CONC:-64}" --limit "${LIMIT:-0}" \
  --split "${SPLIT:-all}" \
  --timeout "${TIMEOUT:-$DEFAULT_TIMEOUT}" \
  --out "$OUT/results.json" --gen-out "$OUT/generations.jsonl" 2>&1 | tee "$OUT/math.log"

# Exit with the BENCHMARK's status, not tee's. Without this the script always exits 0, so a
# fatal error inside math_bench.py -- a missing dataset, an unreachable endpoint, a bad cap --
# is reported as a successful scoring run. Measured 2026-09-01: an H200 sweep marked four
# models DONE in under two minutes each while every shard died on FileNotFoundError, and the
# only reason it was caught is that the accuracies were impossible. A harness that cannot fail
# cannot measure anything.
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  echo "BENCH FAILED (exit $rc); see $OUT/math.log" >&2
fi
exit "$rc"
