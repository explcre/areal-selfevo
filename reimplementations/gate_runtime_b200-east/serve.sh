#!/usr/bin/env bash
# Launch one sglang server. Detached with setsid: under plain `nohup ... &` from an ssh
# one-liner the server takes a SIGTERM at session close and shuts down cleanly a minute after
# going healthy, which reads as a crash.
set -u
ROOT=/mnt/localssd/gate
source "$ROOT/venv/bin/activate"
export HF_HOME=/mnt/localssd/hf
export TMPDIR=/mnt/localssd/tmp
NAME="${NAME:?}"; GPUS="${GPUS:?}"; PORT="${PORT:?}"
DP="${DP:-1}"; TP="${TP:-1}"
CTX="${CTX:-20480}"
MEMFRAC="${MEMFRAC:-0.85}"
LORA_ARGS=()
if [ -n "${LORA_PATHS:-}" ]; then
  # shellcheck disable=SC2206
  LORA_ARGS=(--enable-lora --max-lora-rank "${MAX_LORA_RANK:-32}" --max-loras-per-batch "${MAX_LORAS:-2}" --lora-paths $LORA_PATHS)
fi
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/server_${NAME}.log"
: > "$LOG"
CUDA_VISIBLE_DEVICES="$GPUS" setsid nohup python -m sglang.launch_server \
  --model-path "$ROOT/models/Qwen3.8-27B" \
  --served-model-name qwen38-27b \
  --host 127.0.0.1 --port "$PORT" \
  --tp "$TP" --dp "$DP" \
  --context-length "$CTX" \
  --mem-fraction-static "$MEMFRAC" \
  --max-running-requests "${MAXREQ:-256}" \
  --allow-auto-truncate \
  "${LORA_ARGS[@]}" \
  >> "$LOG" 2>&1 < /dev/null &
echo "SERVER_${NAME}_PID=$!"
echo "$!" > "$ROOT/logs/server_${NAME}.pid"
