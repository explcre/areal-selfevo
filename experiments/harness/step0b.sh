#!/usr/bin/env bash
# Step 0: reproduce AReaL published GSM8K GRPO run on 8x A100.
# Deviations from the reference config, and why:
#   attn_impl=sdpa -- the prebuilt flash-attn wheel is ABI-incompatible with torch 2.9.1
#                     (undefined symbol c10_cuda_check_implementation). sdpa is
#                     numerically equivalent, only slower.
#   batch_size=32  -- fits one node; the reference assumes a larger cluster.
# gconfig.max_new_tokens is NOT overridden this time; the reference value (1024) stands.
# The previous run overrode it to 512, which silently truncated chain-of-thought.
set -o pipefail
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/venv312b/bin/activate"
cd "$HOME/areal-selfevo" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Reuse the admin key already present on this host; never re-embed the secret here.
KEY=$(grep -oE "admin_api_key=[A-Za-z0-9_-]+" "$HOME/step0.sh" | head -1 | cut -d= -f2)
[ -n "$KEY" ] || { echo "no admin key found"; exit 2; }

RUN="$HOME/runs/step0b"; mkdir -p "$RUN"
LOG="$RUN/train.log"; : > "$LOG"

python3 examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=local \
  cluster.fileroot="$HOME/areal-runs" \
  train_dataset.batch_size=32 \
  valid_dataset.batch_size=32 \
  total_train_epochs=1 \
  +actor.attn_impl=sdpa +ref.attn_impl=sdpa \
  +rollout.agent.admin_api_key="$KEY" \
  experiment_name=step0b trial_name=t1 2>&1 \
  | stdbuf -oL python3 "$HOME/logfilter.py" >> "$LOG"
echo "STEP0B_EXIT=${PIPESTATUS[0]}" >> "$LOG"
