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
# Follows OUTROOT/BENCH_ROOT so every caller writes to the same volume; $HOME is only the
# fallback, and on a container it is often the wrong (small) one.
OUT="${OUTROOT:-${BENCH_ROOT:-$HOME}/runs/math}/$TAG"; mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES="$GPUS" python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name evalmodel \
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
echo "endpoint up; scoring $TAG"
# Repo copy, never a stale $HOME copy: numbers must come from audited code.
python3 "$(dirname "$0")/math_bench.py" --base-url "http://127.0.0.1:$PORT/v1" \
  --benchmarks "${BENCHES:-aime24,aime25,amc23,math500}" \
  --max-tokens "${MAXTOK:-3072}" --concurrency "${CONC:-64}" --limit "${LIMIT:-0}" \
  --split "${SPLIT:-all}" \
  --timeout "${TIMEOUT:-600}" \
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
