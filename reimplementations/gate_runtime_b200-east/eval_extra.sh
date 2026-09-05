#!/usr/bin/env bash
# Two more checkpoints, to pin down whether the held-out curves really cross.
#
# T scores HIGHER at step 30 (0.7051) than at its step-61 endpoint (0.6450) while C2 scores
# LOWER at step 40 (0.6332) than at its step-60 endpoint (0.6981). If both readings are real,
# the T-vs-C2 ordering depends on where the arms are stopped, and a single endpoint comparison
# would report whichever answer the wall clock happened to produce. Two intermediate points
# turn that from an inference off two readings into a curve.
set -u
ROOT=/mnt/localssd/gate
PORT=30046
CAP=8192
K="${K:-8}"
source "$ROOT/venv/bin/activate"
LP="Tmid=$ROOT/runs/T/ckpt/step00050 C2early=$ROOT/runs/C2/ckpt/step00020"
NAME=extra GPUS=0,1,2,3,4,5,6,7 PORT=$PORT DP=8 TP=1 CTX=$((CAP+2048)) MEMFRAC=0.85 \
  MAXREQ=64 LORA_PATHS="$LP" MAX_LORAS=2 bash "$ROOT/code/serve.sh"
bash "$ROOT/code/wait_health.sh" $PORT "$ROOT/logs/server_extra.log" 60 || exit 1
cd "$ROOT/code"
for arm in Tmid C2early; do
  echo "=== extra eval $arm ==="
  python gen_pool.py --url "http://127.0.0.1:$PORT" --split report --k "$K" --cap "$CAP" \
      --effort low --concurrency 320 --lora "$arm" \
      --out "$ROOT/out/eval_extra_${arm}.jsonl" 2>&1 | tail -2
done
python analyze_eval.py --against base \
  --arm base="$ROOT/out/eval_final_base.jsonl.graded.jsonl" \
  --arm T_s30="$ROOT/out/eval_matched_T.jsonl.graded.jsonl" \
  --arm T_s50="$ROOT/out/eval_extra_Tmid.jsonl.graded.jsonl" \
  --arm T_s61="$ROOT/out/eval_final_T.jsonl.graded.jsonl" \
  --arm C2_s20="$ROOT/out/eval_extra_C2early.jsonl.graded.jsonl" \
  --arm C2_s40="$ROOT/out/eval_matched_C2.jsonl.graded.jsonl" \
  --arm C2_s60="$ROOT/out/eval_final_C2.jsonl.graded.jsonl" \
  --out "$ROOT/out/eval_trajectory.json"
gsutil -mq rsync -r "$ROOT/out" gs://selfevo/runs/b200x8/gate/out
echo "=== EXTRA EVAL DONE $(date -Is) ==="
