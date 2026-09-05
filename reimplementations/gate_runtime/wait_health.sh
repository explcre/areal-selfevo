#!/usr/bin/env bash
# Block until an sglang server answers /health_generate, or report why it did not.
PORT="${1:?}"; LOG="${2:?}"; N="${3:-90}"
for i in $(seq 1 "$N"); do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "http://127.0.0.1:$PORT/health_generate" 2>/dev/null)
  if [ "$code" = "200" ]; then echo "HEALTHY port=$PORT after ~$((i*10))s"; exit 0; fi
  if ! grep -q . "$LOG" 2>/dev/null; then :; fi
  if grep -qE "Scheduler hit an exception|kill_process_tree called" "$LOG" 2>/dev/null; then
    echo "SERVER DIED port=$PORT"; grep -nE "Error|error|Exception" "$LOG" | tail -8; exit 1
  fi
  sleep 10
done
echo "TIMEOUT port=$PORT"; tail -12 "$LOG"; exit 2
