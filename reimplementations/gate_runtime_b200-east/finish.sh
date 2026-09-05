#!/usr/bin/env bash
# Everything after the last gradient step: verify the served adapters, score them on the
# held-out half, and analyse. Written as one script so it runs unattended on a spot box.
set -u
ROOT=/mnt/localssd/gate
CAP="${CAP:-8192}"
K="${K:-8}"
POOL="${POOL:-$ROOT/out/pool_cap8192.json}"
declare -A PORT=( [T]=30031 [C1]=30032 [C2]=30033 [C3]=30034 )
source "$ROOT/venv/bin/activate"

echo "=== 0. wait for every trainer to exit ==="
for a in T C1 C2 C3; do
  p=$(cat "$ROOT/logs/train_$a.pid" 2>/dev/null || echo 1)
  while ps -p "$p" > /dev/null 2>&1; do sleep 60; done
  echo "  $a exited: $(tail -1 "$ROOT/logs/train_$a.log" | cut -c1-120)"
done

echo "=== 1. calibrated route verification on the FINAL adapters ==="
# Now, not during training: loading the calibration adapter races the trainer's own
# unload/load otherwise, and an unload that collides is exactly what wedges this server.
for a in T C1 C2 C3; do
  python "$ROOT/code/verify_route.py" --url "http://127.0.0.1:${PORT[$a]}" --name "$a" \
    --out "$ROOT/out/route_final_$a.json" 2>&1 | tail -3
done

echo "=== 2. stop the per-arm servers ==="
for a in T C1 C2 C3; do
  p=$(cat "$ROOT/logs/server_$a.pid" 2>/dev/null || echo 1); kill "$p" 2>/dev/null
done
sleep 30
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo

echo "=== 3. score base and every arm on the held-out report half ==="
CAP="$CAP" K="$K" TAG=final bash "$ROOT/code/run_evals.sh"

echo "=== 4. the training curve, token-matched across arms ==="
python "$ROOT/code/analyze_curve.py" --pool "$POOL" --window 10 --token-match \
  --arm T="$ROOT/runs/T/steps.jsonl" --arm C1="$ROOT/runs/C1/steps.jsonl" \
  --arm C2="$ROOT/runs/C2/steps.jsonl" --arm C3="$ROOT/runs/C3/steps.jsonl" \
  --out "$ROOT/out/curve_final.json" > "$ROOT/out/curve_final.txt"
tail -5 "$ROOT/out/curve_final.txt"

echo "=== 5. sync everything ==="
gsutil -mq rsync -r "$ROOT/out" gs://selfevo/runs/b200x8/gate/out
gsutil -mq rsync -r "$ROOT/logs" gs://selfevo/runs/b200x8/gate/logs
for a in T C1 C2 C3; do
  gsutil -mq rsync -r "$ROOT/runs/$a" "gs://selfevo/runs/b200x8/gate/runs/$a"
done
echo "=== FINISH DONE $(date -Is) ==="
