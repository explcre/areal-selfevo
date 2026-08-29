#!/usr/bin/env bash
# Serve a model locally and score it on the two science benchmarks.
#
#   run_science_eval.sh <model-path-or-hf-id> <tag> [gpu-id] [port]
#
# Everything is written under ~/runs/eval/<tag>/. The eval kits in ~/evalkits/ are read
# but never written to, and the original material on the lab machine is untouched.
#
# GeneBench-Pro grades deterministically (`gbp/grading.py`), so it runs end to end here.
# BioMysteryBench's judge shells out to the Codex CLI (`bmb/judge.py:233`), which is not
# installed on this host, so only the SOLVE phase runs here; transcripts are then judged
# where Codex is available with `scripts/rejudge.py`, which replays saved transcripts
# without re-solving.
set -u -o pipefail

MODEL="${1:?usage: run_science_eval.sh <model-path> <tag> [gpu] [port]}"
TAG="${2:?}"
GPU="${3:-0}"
PORT="${4:-8404}"
SERVED_NAME="evalmodel"

export PATH="$HOME/.local/bin:$PATH"
source "$HOME/venv312b/bin/activate"
ulimit -n 131072 || true

OUT="$HOME/runs/eval/$TAG"; mkdir -p "$OUT"
exec 9>"$OUT/.lock"; flock -n 9 || { echo "eval '$TAG' already running"; exit 3; }

echo "[eval] serving $MODEL on GPU $GPU port $PORT"
CUDA_VISIBLE_DEVICES="$GPU" python3 -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$PORT" --mem-fraction-static 0.85 \
    > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
# Kill the server on any exit path, including failure, so a crashed eval never strands a
# GPU. Guarded so we never signal an empty or already-dead pid.
cleanup() { [ -n "${SERVER_PID:-}" ] && kill -TERM "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT INT TERM

echo "[eval] waiting for the endpoint (up to 10 min)"
ready=0
for _ in $(seq 1 120); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[eval] SERVER DIED during startup; last lines:"; tail -20 "$OUT/server.log"; exit 4
    fi
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ready=1; break; fi
    sleep 5
done
[ "$ready" = 1 ] || { echo "[eval] endpoint never became ready"; tail -20 "$OUT/server.log"; exit 5; }
echo "[eval] endpoint up"

BASE="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY="local-no-auth"

echo "[eval] === GeneBench-Pro (10 public problems, deterministic grading) ==="
( cd "$HOME/evalkits/genebench-pro" && python3 -m gbp.runner \
    --backend openrouter --base-url "$BASE" --model "$SERVED_NAME" \
    --attempts 1 --effort high \
    --out "$OUT/gbp.jsonl" \
    --workspace-root "$HOME/scratch/gbp_ws_$TAG" \
    --transcripts "$OUT/gbp_transcripts" ) 2>&1 | tail -25 | tee "$OUT/gbp.log"

echo "[eval] === BioMysteryBench (solve phase only; judge later where Codex lives) ==="
( cd "$HOME/evalkits/bio-mystery" && python3 -m bmb.runner \
    --backend served --base-url "$BASE" --model "$SERVED_NAME" \
    --subset all --attempts 1 --max-steps 25 --workers 4 \
    --out "$OUT/bmb.jsonl" \
    --workspace-root "$HOME/scratch/bmb_ws_$TAG" \
    --transcripts "$OUT/bmb_transcripts" ) 2>&1 | tail -25 | tee "$OUT/bmb.log"

echo "[eval] done. Results under $OUT"
echo "[eval] REMINDER: any BioMysteryBench score must name its judge model and effort,"
echo "[eval] and the 'excluded' counts must be read -- a judge outage silently shrinks n."
