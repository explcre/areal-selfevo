#!/usr/bin/env bash
# Restart the four trainers against the SAME servers, at a tighter rollout cap.
#
# Why: at cap 12288 a step cost 430 s, of which 299 s was generation, and the generation is
# straggler-bound -- the group is not complete until its longest rollout ends, and by then the
# batch has drained to a handful of sequences decoding almost alone. Halving the cap bounds the
# straggler and roughly doubles the number of gradient steps the wall clock allows, which is
# the quantity this experiment is short of. The pool is rebuilt at the same cap from the SAME
# discovery samples (a generation that ran past the cap would have been cut at it), so p-hat
# and the rollouts still agree.
#
# The servers are NOT restarted: their context length (14336) already covers the smaller cap.
set -u
ROOT=/mnt/localssd/gate
POOL="$ROOT/out/pool_cap8192.json"
declare -A TGPU=( [T]=1 [C1]=3 [C2]=5 [C3]=7 )
declare -A PORT=( [T]=30031 [C1]=30032 [C2]=30033 [C3]=30034 )
source "$ROOT/venv/bin/activate"
for arm in T C1 C2 C3; do
  if [ -f "$ROOT/logs/train_$arm.pid" ]; then
    p=$(cat "$ROOT/logs/train_$arm.pid"); kill -9 "$p" 2>/dev/null && echo "killed $arm ($p)"
  fi
done
sleep 20
for arm in T C1 C2 C3; do
  curl -s -X POST "http://127.0.0.1:${PORT[$arm]}/unload_lora_adapter" \
    -H "Content-Type: application/json" -d "{\"lora_name\":\"$arm\"}" --max-time 120 >/dev/null
  mv "$ROOT/runs/$arm" "$ROOT/runs/${arm}_cap12288_aborted" 2>/dev/null
done
sleep 5
for arm in T C1 C2 C3; do
  CUDA_VISIBLE_DEVICES="${TGPU[$arm]}" PYTORCH_ALLOC_CONF=expandable_segments:True \
  setsid nohup python "$ROOT/code/train_arm.py" \
    --arm "$arm" --pool "$POOL" --url "http://127.0.0.1:${PORT[$arm]}" \
    --run-dir "$ROOT/runs/$arm" --gcs gs://selfevo/runs/b200x8/gate/runs \
    --group-size 8 --prompts-per-step 6 --cap 8192 --max-len 8192 --lr 2e-4 \
    --token-budget 0 --max-steps 400 --max-hours "${MAX_HOURS:-5.5}" \
    --concurrency 48 --grad-ckpt 1 --mb-tokens 16384 --truncated wrong --ckpt-every 10 \
    > "$ROOT/logs/train_$arm.log" 2>&1 < /dev/null &
  echo "$!" > "$ROOT/logs/train_$arm.pid"
  echo "TRAIN_${arm}_PID=$(cat "$ROOT/logs/train_$arm.pid")"
done
