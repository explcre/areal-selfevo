#!/usr/bin/env bash
# One-screen state of the four arms: steps done, generated-token budget consumed, the reward
# curve's last points, and whether any server is retracting requests (a KV pool that is full
# shows up as low throughput, not as an error).
ROOT=/mnt/localssd/gate
echo "=== $(date -Is) ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' '|'; echo
for arm in T C1 C2 C3; do
  f="$ROOT/runs/$arm/steps.jsonl"
  [ -s "$f" ] || { echo "$arm: no steps yet"; continue; }
  python3 - "$f" "$arm" <<'PY'
import json, sys, statistics
f, arm = sys.argv[1], sys.argv[2]
rows=[json.loads(l) for l in open(f)]
acc=[r["rollout_accuracy"] for r in rows if r.get("rollout_accuracy") is not None]
def w(xs,n): 
    xs=xs[-n:]
    return round(statistics.fmean(xs),4) if xs else None
print("%-3s steps=%-4d tok=%-10d info=%.2f acc_last20=%s acc_first20=%s absB=%.3f gen=%.0fs upd=%.0fs" % (
    arm, len(rows), rows[-1]["gen_tokens_cum"],
    statistics.fmean([r["informative_group_fraction"] for r in rows[-20:]]),
    w(acc,20), w(acc[:20],20), rows[-1].get("adapter_absB") or 0,
    statistics.fmean([r["t_gen_s"] for r in rows[-5:]]),
    statistics.fmean([r["t_update_s"] for r in rows[-5:]])))
PY
  n=$(grep -c "KV cache pool is full" "$ROOT/logs/server_${arm}.log" 2>/dev/null || echo 0)
  [ "$n" != "0" ] && echo "    WARNING $arm server retracted requests $n times (KV pool full)"
done
