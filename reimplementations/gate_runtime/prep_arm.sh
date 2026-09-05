#!/usr/bin/env bash
# Create an arm's run directory with a NULL adapter, then bring up its rollout server.
# The server needs an adapter at boot so `--enable-lora` can size its buffers; the trainer
# overwrites that directory and re-pushes it every step.
set -eu
ROOT=/mnt/localssd/gate
ARM="${ARM:?}"; PORT="${PORT:?}"; SGPU="${SGPU:?}"
RUN="$ROOT/runs/$ARM"
mkdir -p "$RUN"
source "$ROOT/venv/bin/activate"
python "$ROOT/code/mk_probe_adapter.py" --out "$RUN/adapter" --scale 0 \
  --pattern "model\.language_model\.layers\.[0-9]+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.out_proj|mlp\.(gate_proj|up_proj|down_proj))" \
  --target-list "q_proj,k_proj,v_proj,o_proj,out_proj,gate_proj,up_proj,down_proj" | tail -2
NAME="$ARM" GPUS="$SGPU" PORT="$PORT" DP=1 TP=1 CTX="${CTX:-24576}" MEMFRAC="${MEMFRAC:-0.82}" \
  MAXREQ="${MAXREQ:-128}" LORA_PATHS="$ARM=$RUN/adapter" MAX_LORAS=2 \
  bash "$ROOT/code/serve.sh"
