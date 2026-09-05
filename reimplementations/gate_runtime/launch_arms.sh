#!/usr/bin/env bash
# Bring up all four arms: one rollout server and one trainer per arm, one GPU each.
#
# 183 GB per device is what makes this layout possible: a 27B policy in bf16 is 52 GB, so the
# trainer needs no sharding at all and the rollout server still has ~100 GB of KV cache. On
# the 80 GB boxes this programme has used, the same experiment needs FSDP across devices and
# the four arms cannot run concurrently, which is why it was never run.
set -u
ROOT=/mnt/localssd/gate
POOL="${POOL:-$ROOT/out/pool.json}"
CAP="${CAP:-16384}"
GROUP="${GROUP:-8}"
BATCH="${BATCH:-8}"
LR="${LR:-1e-5}"
TOKEN_BUDGET="${TOKEN_BUDGET:-0}"
MAX_STEPS="${MAX_STEPS:-100000}"
MAX_HOURS="${MAX_HOURS:-9}"
SMOKE="${SMOKE:-0}"
ARMS="${ARMS:-T C1 C2 C3}"
GCS="${GCS:-gs://selfevo/runs/b200x8/gate/runs}"

declare -A SGPU=( [T]=0 [C1]=2 [C2]=4 [C3]=6 )
declare -A TGPU=( [T]=1 [C1]=3 [C2]=5 [C3]=7 )
declare -A PORT=( [T]=30031 [C1]=30032 [C2]=30033 [C3]=30034 )

if [ "${STAGE:-all}" = "servers" ] || [ "${STAGE:-all}" = "all" ]; then
  for arm in $ARMS; do
    echo "=== prep $arm (server gpu ${SGPU[$arm]}, port ${PORT[$arm]}) ==="
    ARM="$arm" PORT="${PORT[$arm]}" SGPU="${SGPU[$arm]}" CTX=$((CAP+2048)) \
      MEMFRAC="${MEMFRAC:-0.80}" MAXREQ=$((GROUP*BATCH)) bash "$ROOT/code/prep_arm.sh"
  done
  for arm in $ARMS; do
    bash "$ROOT/code/wait_health.sh" "${PORT[$arm]}" "$ROOT/logs/server_${arm}.log" 60 || exit 1
  done
fi

if [ "${STAGE:-all}" = "trainers" ] || [ "${STAGE:-all}" = "all" ]; then
  source "$ROOT/venv/bin/activate"
  for arm in $ARMS; do
    RUN="$ROOT/runs/$arm"
    echo "=== train $arm (trainer gpu ${TGPU[$arm]}) ==="
    CUDA_VISIBLE_DEVICES="${TGPU[$arm]}" PYTORCH_ALLOC_CONF=expandable_segments:True \
    setsid nohup python "$ROOT/code/train_arm.py" \
      --arm "$arm" --pool "$POOL" --url "http://127.0.0.1:${PORT[$arm]}" \
      --run-dir "$RUN" --gcs "$GCS" \
      --group-size "$GROUP" --prompts-per-step "$BATCH" --cap "$CAP" --lr "$LR" \
      --token-budget "$TOKEN_BUDGET" --max-steps "$MAX_STEPS" --max-hours "$MAX_HOURS" \
      --smoke "$SMOKE" --concurrency $((GROUP*BATCH)) \
      --grad-ckpt "${GRAD_CKPT:-1}" --mb-tokens "${MB_TOKENS:-16384}" \
      --max-len "${MAX_LEN:-$CAP}" --truncated "${TRUNCATED:-wrong}" \
      --ckpt-every "${CKPT_EVERY:-10}" \
      > "$ROOT/logs/train_${arm}.log" 2>&1 < /dev/null &
    echo "TRAIN_${arm}_PID=$!"
    echo "$!" > "$ROOT/logs/train_${arm}.pid"
  done
fi
