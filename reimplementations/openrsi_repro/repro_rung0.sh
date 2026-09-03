#!/usr/bin/env bash
# Rung 0 as run on 2026-09-03. Serves Frontis-MA1-30B on ONE GPU and runs the
# OpenRSI NatureBench local quickstart with --smoke (one task, one candidate).
# Currently BLOCKED at the reward stage; see RUNG0.md.
set -euo pipefail
GPU="${GPU:-3}"
PORT="${PORT:-30010}"
SNAP=$(ls -d ~/hf_cache/hub/models--FrontisAI--Frontis-MA1-30B/snapshots/*/)

export HF_HOME=~/hf_cache PATH=~/venv38/bin:$PATH
CUDA_VISIBLE_DEVICES="$GPU" python -m sglang.launch_server \
  --model-path "$SNAP" --served-model-name frontis-ma1-30b \
  --host 127.0.0.1 --port "$PORT" --tp-size 1 --context-length 131072 \
  > ~/sglang_ma1_30b.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

until curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do sleep 5; done
curl -fsS "http://127.0.0.1:$PORT/v1/models"   # record which model actually answers

cd ~/OpenRSI/OpenMLE-Evo
PRIMARY_KEY=EMPTY ~/py312/bin/python scripts/run_naturebench_local.py \
  --naturebench-repo ~/NatureBench \
  --local-python ~/nbenv/bin/python \
  --model-base-url "http://127.0.0.1:$PORT/v1" \
  --model-id frontis-ma1-30b \
  --output-dir ~/rung0_out --experiment-name rung0_smoke --smoke
