#!/usr/bin/env bash
# Arm A0 of the M25 plan: vanilla GRPO + ONE shared LoRA. The baseline every other arm is
# measured against, so nothing here may be clever.
#
# What the plan (experiments/m25/PLAN.md) fixes and this script obeys:
#   * no partition, no clustering, no harness ladder -- group_routing is NOT set at all, so
#     RoutingContext.can_evolve_harness stays False and _refuse_dropped_harness stays absolute.
#   * FIXED generation cap. gconfig.max_new_tokens is a constant; no selector, no variants.
#   * adv_norm=null and kl_ctl=0.0. kl_ctl>0 builds a ref model COLOCATED on the actor and
#     OOMs at 32B; mean_level=group is forbidden by the plan (its token-weighted mean destroys
#     sum(a_i)=0 unless all rows match in length).
#   * batch and max_tokens_per_mb are NOT raised: the measured actor headroom is 3.3 GB.
#   * checkpoint every 25 steps, so a reclaimed box costs under five minutes of training.
#
# MATH, not GSM8K: GSM8K is near-saturated at 32B. Same source as the probe batch.
set -u -o pipefail

source "$HOME/venv312b/bin/activate"
cd "$HOME/areal-selfevo" || exit 1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"; mkdir -p "$TMPDIR"
ulimit -n 131072 || echo "WARNING: could not raise the file-descriptor limit"

DRY_RUN="${DRY_RUN:-0}"
EXP="${EXP_NAME:-a0_math}"
TRIAL="${TRIAL_NAME:-t1}"
N_GPUS=4
GPU_GIB=79.19
MODEL_PATH="$HOME/hf_cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
TOTAL_STEPS="${TOTAL_STEPS:-2000}"
SAVE_FREQ="${SAVE_FREQ:-25}"
BATCH_SIZE="${BATCH_SIZE:-8}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MAX_TOK_PER_MB="${MAX_TOK_PER_MB:-2048}"
# What a rollout that hit MAX_NEW_TOKENS contributes to the advantage: keep (A0, the
# historical behaviour -- it grades 0 and carries the same negative advantage as a
# confident wrong answer), zero (no advantage of its own, group baseline untouched) or
# exclude (also dropped from the group mean and std). Defaults to the baseline, so this
# file still launches A0 unchanged.
TRUNC_ADV="${TRUNC_ADV:-keep}"
WANDB_GROUP="${WANDB_GROUP:-A0}"
export WANDB_MODE="${WANDB_MODE:-online}"

RUN="$HOME/runs/${EXP}"; mkdir -p "$RUN"
LOG="$RUN/train.log"
DUMP="$RUN/resolved_config.yaml"

KEY=$(cat "$HOME/.areal_admin_key" 2>/dev/null)
[ -n "${KEY:-}" ] || { echo "no admin key at ~/.areal_admin_key"; exit 5; }

OVERRIDES=(
  scheduler.type=local
  cluster.fileroot="$HOME/areal-runs"
  cluster.n_gpus_per_node=$N_GPUS

  actor.path="$MODEL_PATH"
  ref.path="$MODEL_PATH"
  actor.backend=fsdp:d2p1t1
  rollout.backend=sglang:d1p1t2
  +actor.scheduling_strategy.type=separation
  +rollout.scheduling_strategy.type=separation
  +actor.fsdp.memory_efficient_load=true

  rollout.use_lora=true
  actor.use_lora=true
  actor.lora_rank=32
  actor.lora_alpha=32
  actor.target_modules="[q_proj,k_proj,v_proj,o_proj]"
  gconfig.lora_name="$EXP"
  sglang.enable_lora=true
  sglang.max_lora_rank=32
  sglang.mem_fraction_static=0.80
  sglang.context_length=4096
  ++sglang.disable_cuda_graph=true

  actor.kl_ctl=0.0
  # "+": the field is on PPOActorConfig but not in the YAML this run composes, so a
  # bare override is refused by hydra struct mode -- same as +actor.attn_impl below.
  +actor.truncated_advantage="$TRUNC_ADV"
  actor.adv_norm=null
  actor.optimizer.lr=1.0e-4
  actor.mb_spec.max_tokens_per_mb="$MAX_TOK_PER_MB"

  gconfig.n_samples="$N_SAMPLES"
  gconfig.max_new_tokens="$MAX_NEW_TOKENS"
  gconfig.max_tokens="$MAX_TOKENS"

  train_dataset.path=/home/ubuntu/data/deepmath_decontam
  train_dataset.batch_size="$BATCH_SIZE"
  train_dataset.max_length=1024
  valid_dataset.path=DigitalLearningGmbH/MATH-lighteval
  valid_dataset.batch_size="$BATCH_SIZE"

  rollout.max_concurrent_rollouts=32
  +total_train_steps="$TOTAL_STEPS"
  saver.freq_steps="$SAVE_FREQ"
  saver.freq_epochs=null
  recover.mode="${RECOVER_MODE:-auto}"
  recover.freq_steps="$SAVE_FREQ"
  recover.freq_secs=null
  recover.freq_epochs=null
  evaluator.freq_epochs=null
  evaluator.freq_secs=null
  evaluator.freq_steps=null
  +actor.attn_impl=sdpa
  +ref.attn_impl=sdpa
  +rollout.agent.admin_api_key="$KEY"
  stats_logger.wandb.mode="$WANDB_MODE"
  +stats_logger.wandb.project=selfevo-m25
  +stats_logger.wandb.group="$WANDB_GROUP"
  experiment_name="$EXP"
  trial_name="$TRIAL"
)

HYDRA_ARGS=(--config examples/math/gsm8k_grpo_lora.yaml "${OVERRIDES[@]}")

# Only on a real launch: a DRY_RUN must not touch a running job's artefacts.
[ "$DRY_RUN" = "1" ] || { [ -s "$LOG" ] && mv "$LOG" "$LOG.$(date +%s)"; }
echo "=== ${EXP} (A0): preflight ($(date -Is)) ===" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES="" \
SMOKE_DUMP="$DUMP" SMOKE_NGPUS="$N_GPUS" SMOKE_GPU_GIB="$GPU_GIB" \
SMOKE_MIN_KV_GIB=4 SMOKE_ACTOR_SHARD_FRAC=0.80 SMOKE_TRUNC_ADV="$TRUNC_ADV" \
python "$HOME/harness4/preflight_a0.py" "${HYDRA_ARGS[@]}" 2>&1 | tee -a "$LOG"
pf=${PIPESTATUS[0]}
if [ "$pf" -ne 0 ]; then
  echo "PREFLIGHT REFUSED THE LAUNCH (rc=$pf); nothing was started." | tee -a "$LOG"
  exit 6
fi
if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN=1: preflight passed, resolved config at $DUMP, no GPU touched." | tee -a "$LOG"
  exit 0
fi

busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
       | awk -F", " '$2 > 2048 {printf "%s(%sMiB) ", $1, $2}')
if [ -n "$busy" ]; then
  echo "REFUSING TO LAUNCH: GPUs already hold memory: $busy" | tee -a "$LOG"
  exit 4
fi

echo "=== ${EXP} (A0): launching ($(date -Is)) ===" | tee -a "$LOG"
export ENTRY_DUMP="$RUN/process_env.json"
export ENTRY_TARGET="$HOME/areal-selfevo/examples/math/gsm8k_rl.py"
python "$HOME/harness4/trainer_entry.py" "${HYDRA_ARGS[@]}" >> "$LOG" 2>&1
rc=${PIPESTATUS[0]}
echo "A0_EXIT=$rc" | tee -a "$LOG"
echo "=== ${EXP}: finished ($(date -Is)) rc=$rc ===" | tee -a "$LOG"
exit "$rc"
