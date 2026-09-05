#!/usr/bin/env bash
# Score each arm at a COMMON generated-token budget, which is pre-registration rule 3.
#
# The arms cannot be matched on steps: the gate selects harder problems whose rollouts run
# longer, so equal step counts hand the cheap-task arms more rollouts and the expensive-task
# arms more tokens. They also stopped at different points -- C1 was halted by the length guard
# at 6.95M generated tokens, which is therefore the budget every arm is cut back to.
#
# C1's own final adapter IS its state at that budget, so only the other three need a
# checkpoint; C1's existing `final` evaluation is reused unchanged.
set -u
ROOT=/mnt/localssd/gate
PORT=30045
CAP=8192
K="${K:-8}"
source "$ROOT/venv/bin/activate"

LP="T=$ROOT/runs/T/ckpt/step00030 C2=$ROOT/runs/C2/ckpt/step00040 C3=$ROOT/runs/C3/ckpt/step00020"
NAME=matched GPUS=0,1,2,3,4,5,6,7 PORT=$PORT DP=8 TP=1 CTX=$((CAP+2048)) MEMFRAC=0.85 \
  MAXREQ=64 LORA_PATHS="$LP" MAX_LORAS=2 bash "$ROOT/code/serve.sh"
bash "$ROOT/code/wait_health.sh" $PORT "$ROOT/logs/server_matched.log" 60 || exit 1

cd "$ROOT/code"
for arm in T C2 C3; do
  OUT="$ROOT/out/eval_matched_${arm}.jsonl"
  echo "=== matched-budget eval $arm -> $OUT ==="
  python gen_pool.py --url "http://127.0.0.1:$PORT" --split report --k "$K" --cap "$CAP" \
      --effort low --concurrency 320 --lora "$arm" --out "$OUT" 2>&1 | tail -2
done
cp "$ROOT/out/eval_final_C1.jsonl.graded.jsonl" "$ROOT/out/eval_matched_C1.jsonl.graded.jsonl"

python analyze_eval.py --against base \
  --arm base="$ROOT/out/eval_final_base.jsonl.graded.jsonl" \
  --arm T="$ROOT/out/eval_matched_T.jsonl.graded.jsonl" \
  --arm C1="$ROOT/out/eval_matched_C1.jsonl.graded.jsonl" \
  --arm C2="$ROOT/out/eval_matched_C2.jsonl.graded.jsonl" \
  --arm C3="$ROOT/out/eval_matched_C3.jsonl.graded.jsonl" \
  --out "$ROOT/out/eval_matched_vs_base.json"
python analyze_eval.py --against C1 \
  --arm T="$ROOT/out/eval_matched_T.jsonl.graded.jsonl" \
  --arm C1="$ROOT/out/eval_matched_C1.jsonl.graded.jsonl" \
  --arm C2="$ROOT/out/eval_matched_C2.jsonl.graded.jsonl" \
  --arm C3="$ROOT/out/eval_matched_C3.jsonl.graded.jsonl" \
  --out "$ROOT/out/eval_matched_vs_C1.json"
gsutil -mq rsync -r "$ROOT/out" gs://selfevo/runs/b200x8/gate/out
echo "=== MATCHED EVAL DONE $(date -Is) ==="
