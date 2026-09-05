#!/usr/bin/env bash
# One sample end to end before the scaled run, with every stage required to FIRE.
#
# Stages proven here, each on observable state rather than on a call that returned:
#   1. the pool file loads and a selector draws from it
#   2. the serving backend applies all three module families (preflight_families.py)
#   3. the adapter reaches 64/64 decoder layers (assert_coverage, which is shown to fire on
#      the inherited config in the same run)
#   4. right padding does not change a token's logprob (assert_padding_is_a_noop)
#   5. rollouts come back with exact token ids and are graded
#   6. at least one group carries advantage and a gradient step runs with grad_norm > 0
#   7. the adapter's LoRA-B is no longer zero, and the RELOADED adapter changes server output
set -u
ROOT=/mnt/localssd/gate
POOL="${POOL:-$ROOT/out/pool.json}"
CAP="${CAP:-16384}"
GROUP="${GROUP:-8}"
BATCH="${BATCH:-2}"
STEPS="${STEPS:-2}"
RUN="$ROOT/runs/smoke"
rm -rf "$RUN"

echo "=== 1. server for arm T ==="
ARM=T PORT=30031 SGPU=0 CTX=$((CAP+2048)) MEMFRAC=0.86 MAXREQ=$((GROUP*BATCH)) \
  bash "$ROOT/code/prep_arm.sh"
bash "$ROOT/code/wait_health.sh" 30031 "$ROOT/logs/server_T.log" 60 || exit 1

source "$ROOT/venv/bin/activate"
echo "=== 2. does the SERVER apply every module family? ==="
python "$ROOT/code/preflight_families.py" --url http://127.0.0.1:30031 \
  --out "$ROOT/out/preflight_families.json" || exit 2

echo "=== 3-7. one arm, $STEPS steps, every stage asserted ==="
CUDA_VISIBLE_DEVICES=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
python "$ROOT/code/train_arm.py" --arm T --pool "$POOL" --url http://127.0.0.1:30031 \
  --run-dir "$RUN" --group-size "$GROUP" --prompts-per-step "$BATCH" --cap "$CAP" \
  --concurrency $((GROUP*BATCH)) --smoke "$STEPS" --ckpt-every 0 2>&1 | tail -40
rc=${PIPESTATUS[0]}
echo "SMOKE_RC=$rc"
exit $rc
