#!/usr/bin/env bash
# Score the base model and every arm's final adapter on the HELD-OUT report half.
#
# One server holds all the adapters at once, so every arm is scored by the same process on the
# same problems in the same session -- the paired design needs that, and it also removes
# "which server was it on" as an explanation for a difference.
set -u
ROOT=/mnt/localssd/gate
PORT="${PORT:-30040}"
K="${K:-8}"   # avg@8: the paired-difference SE scales with per-item sampling noise, and k=4 widens the interval from ~2.6 to ~3.8 points
CAP="${CAP:?}"
EFFORT="${EFFORT:-low}"
ARMS="${ARMS:-T C1 C2 C3}"
CKPT="${CKPT:-adapter}"          # which snapshot under runs/<arm>/ to score
TAG="${TAG:-final}"

LP=""
for arm in $ARMS; do LP="$LP ${arm}=$ROOT/runs/$arm/$CKPT"; done
NAME=eval GPUS=0,1,2,3,4,5,6,7 PORT="$PORT" DP=8 TP=1 CTX=$((CAP+2048)) MEMFRAC=0.85 \
  MAXREQ=64 LORA_PATHS="$LP" MAX_LORAS=2 bash "$ROOT/code/serve.sh"
bash "$ROOT/code/wait_health.sh" "$PORT" "$ROOT/logs/server_eval.log" 60 || exit 1

source "$ROOT/venv/bin/activate"
cd "$ROOT/code"
for arm in base $ARMS; do
  L=""; [ "$arm" != "base" ] && L="--lora $arm"
  OUT="$ROOT/out/eval_${TAG}_${arm}.jsonl"
  echo "=== eval $arm -> $OUT ==="
  python gen_pool.py --url "http://127.0.0.1:$PORT" --split report --k "$K" --cap "$CAP" \
      --effort "$EFFORT" --concurrency 320 $L --out "$OUT" 2>&1 | tail -3
done

ARGS=""
for arm in base $ARMS; do ARGS="$ARGS --arm ${arm}=$ROOT/out/eval_${TAG}_${arm}.jsonl.graded.jsonl"; done
python analyze_eval.py $ARGS --against base --out "$ROOT/out/eval_${TAG}_summary.json"
python analyze_eval.py $ARGS --against C1   --out "$ROOT/out/eval_${TAG}_vsC1.json"
