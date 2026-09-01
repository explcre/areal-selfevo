#!/usr/bin/env bash
# lora30b -- LoRA RL on Frontis-MA1-30B (Qwen3-MoE, 30.5B), 8x A100 80GB, separation.
#
# This is the CORRECTED successor to lora27b, which never reached step 1: all four
# supervised attempts (runs/lora27b/train.log*) died in `Engine method 'initialize'`
# with a CUDA OOM. The post-mortem below is what the evidence actually says, which is
# NOT what the first diagnosis said, so both are recorded.
#
# WHAT ACTUALLY KILLED lora27b -- read the logs before trusting any of this:
#
#   (a) THE CAPACITY BUG, and it alone is sufficient. areal/engine/fsdp_engine.py:1036
#       `_create_device_model` loads the model in `optimizer_dtype` (float32, not
#       actor.dtype), and with `fsdp.memory_efficient_load: false` -- the DEFAULT, and
#       what lora27b ran -- `loading_device` is the CUDA device, so EVERY rank
#       materialises the WHOLE model in fp32 on its OWN card before FSDP2 shards it.
#       For a 30.5B model that is 113.7 GiB against 79.25 GiB of usable A100. It cannot
#       fit, on any number of GPUs, with any rollout configuration, because sharding
#       happens after the load. The traceback confirms the site:
#         transformers/core_model_loading.py:789 _materialize_copy -> tensor.to(device)
#         torch.OutOfMemoryError ... this process has 70.77 GiB memory in use
#       70.77 GiB is one actor worker, still climbing toward 113.7, on one card.
#       The fix is `actor.fsdp.memory_efficient_load=true`: rank 0 builds from_config on
#       CPU, loads the pretrained state dict there, everyone else builds on `meta`, and
#       fsdp2_load_full_state_dict broadcasts AFTER sharding (fsdp_engine.py:411-465).
#       Peak host RAM on rank 0 is fp32 model 113.7 GiB + bf16 state dict 56.9 GiB =
#       170.6 GiB, against 1693 GiB free on this box. Peak GPU is the shard.
#
#   (b) LoRA WAS HALF-ENABLED, exactly as diagnosed. The lora27b resolved config
#       (areal-runs/logs/ubuntu/lora27b/t1/config.yaml) has actor.use_lora: true at
#       line 487 but rollout.use_lora: false at 320 and sglang.enable_lora: null at 194.
#       That is not a cosmetic mismatch. `get_py_cmd` (cli_args.py:2286-2300) DROPS any
#       flag whose value is None/False/""/[], so `enable_lora: null` means the server is
#       launched with no --enable-lora at all: sglang serves the BASE model, the trainer
#       updates an adapter, and the ratio in the PPO loss compares two different policies.
#       cli_args.py:1374 says LoRA "should be enabled together with vLLM/SGLang" and
#       nothing enforces it, so the run is silently wrong rather than loud.
#
#   (c) A THIRD DEFECT nobody had named: lora27b was built on examples/math/gsm8k_grpo.yaml,
#       whose actor.weight_update_mode is `xccl`. The repo's own LoRA recipe,
#       examples/math/gsm8k_grpo_lora.yaml, carries `weight_update_mode: disk  # must be
#       disk`, and rl_trainer.py:378-381 states why: "LoRA must go through disk (P2P
#       transports cannot carry PEFT-wrapped tensors)". This script therefore starts from
#       gsm8k_grpo_lora.yaml -- the upstream LoRA config -- instead of re-deriving one.
#       That single change also brings rollout.use_lora, sglang.enable_lora and
#       sglang.max_lora_rank in as INTERPOLATIONS of actor's values, so the two sides
#       cannot drift by construction. We still assert it; see the preflight.
#
#   (d) COLOCATION WAS NOT THE CAUSE, contrary to the first diagnosis. In the lora27b
#       resolved config actor.scheduling_strategy.type is `separation` (line 519) and
#       rollout.scheduling_strategy.type is `separation` (line 317). The ONLY colocation
#       is ref.scheduling_strategy.type (line 707), and the ref model was never built:
#       rl_trainer.py:206 builds it only `if config.actor.kl_ctl > 0 and config.ref is not
#       None`, and kl_ctl was 0.0. Nor was the 57.93 GiB process on the card lora27b's own
#       sglang server -- the run died during ACTOR init and the `rollout` role was never
#       created in any of the four attempts (grep "workers for role 'rollout'": no hits).
#       It was a FOREIGN co-tenant from another job. Forcing separation is still right, and
#       this script does it explicitly rather than inheriting a default, but it would not
#       have saved lora27b on its own.
#
#   (e) NOT the cause, confirmed: `target_modules: []` is harmless --
#       fsdp_engine.py:1117-1121 maps an empty list to "all-linear". It is nonetheless
#       WRONG FOR THIS MODEL and is overridden below; see TARGET MODULES.
#
# MODEL. FrontisAI/Frontis-MA1-30B, cached at ~/hf_cache (NOT ~/.cache/huggingface, which
# is where the datasets live -- so HF_HOME is deliberately left alone and the checkpoint is
# passed as an absolute snapshot path). Verified complete: config.json declares
# Qwen3MoeForCausalLM, 48 layers, hidden 2048, 128 experts, top-8, bf16; the index declares
# total_size 61,064,245,248 B and all 12 shards are present, summing to 61,066,578,112 B on
# disk (the 2.3 MB excess is safetensors headers). 61,064,245,248 / 2 = 30.53e9 parameters.
# The Qwen/Qwen3.8-27B fallback under ~/.cache/huggingface is also complete (18 shards,
# 55,562,855,904 B declared) and is selected automatically if the 30B snapshot fails its
# check -- MODEL_PATH= overrides both.
#
# TARGET MODULES. `all-linear` is the wrong default here and the preflight will not stop
# you, so it is stated: this is a 128-expert MoE with decoder_sparse_step 1, so every one of
# the 48 layers has 384 expert linears. PEFT's "all-linear" would attach an adapter to all
# 6144 of them -- about 1.7B trainable parameters, more than a rank-32 attention adapter by
# a factor of 60, and it would need sglang's fused-MoE LoRA path (gate_up_proj_moe in
# sglang/srt/lora/utils.py) rather than the plain one. The default here is the attention
# projections only, q/k/v/o, which sglang normalises to qkv_proj + o_proj and which are both
# in its _KNOWN_LORA_TARGET_MODULES. 26.7M trainable parameters. LORA_TARGETS overrides.
#
# THE MEMORY ARITHMETIC, stated so it can be checked rather than trusted. A100 80GB reports
# 79.25 GiB usable. Base model: 30.53e9 params = 56.87 GiB bf16 = 113.74 GiB fp32.
#
#   ACTOR, fsdp:d4p1t1 -> 4 GPUs, FSDP2 full-shard over dp=4, optimizer_dtype float32:
#     sharded base params            113.74 / 4          = 28.44 GiB/GPU
#     LoRA r=32 on q,k,v,o, 48 layers: 26.74M params
#       (q 2048*32+32*4096=196608, k and v 2048*32+32*512=81920 each,
#        o 4096*32+32*2048=196608; 557056/layer x 48)
#       fp32 param + grad + Adam m,v = 16 B/param = 0.40 GiB, sharded  = 0.10 GiB/GPU
#     transient bf16 all-gather, largest wrapped unit is one MoE layer
#       128 experts x 3 x 2048x768 ~ 604M params x 2 B, two in flight  = 2.3  GiB/GPU
#     activations, gradient_checkpointing on, max_tokens_per_mb 10240  ~ 4-6  GiB/GPU
#     ------------------------------------------------------------------------------
#     ~35 GiB of 79.25 GiB per actor GPU, ~44 GiB spare.
#     LOAD-TIME peak is the number that matters and it is 113.74 GiB/GPU unless
#     memory_efficient_load is on. That is defect (a). The preflight refuses without it.
#
#   ROLLOUT, sglang:d2p1t2 -> 2 servers x TP2 = 4 GPUs:
#     bf16 weights                    56.87 / 2          = 28.44 GiB/GPU
#     adapter buffers, max_loaded_loras 8 x 26.74M x 2 B  = 0.21 GiB/GPU
#     sglang static pool at mem_fraction_static 0.80      = 63.40 GiB/GPU
#     ------------------------------------------------------------------------------
#     KV cache gets 63.40 - 28.44 - 0.21                  = 34.75 GiB/GPU
#     KV/token/GPU = 48 layers x 2 x (4 KV heads / TP2) x 128 head_dim x 2 B = 48 KiB,
#     so ~759k tokens ~ 370 concurrent 2048-token sequences against
#     max_concurrent_rollouts 256. Comfortable.
#     15.85 GiB/GPU stays OUTSIDE the pool for CUDA graphs, the sampler and NCCL.
#     Note what TP buys: at the upstream `sglang:d4` (one server per card) the weights are
#     56.87 GiB on ONE card and the same 0.80 leaves 6.3 GiB of KV. The preflight computes
#     this from whatever backend you set and refuses below MIN_KV_GIB.
#
#   4 actor GPUs + 4 rollout GPUs = 8 = N_GPUS, and they must be DISJOINT.
#   LocalScheduler._allocate_gpus (areal/infra/scheduler/local.py:203-219) is a bare
#   round-robin counter modulo the device list with NO occupancy tracking, so the moment
#   actor.world_size + rollout.world_size exceeds n_gpus the assignment silently WRAPS and
#   two roles land on the same card. The preflight refuses on that sum.
#
# INHERITED FROM step0l, and still true here:
#   attn_impl=sdpa   -- the prebuilt flash-attn wheel is ABI-incompatible with torch 2.9.1
#                       (undefined symbol c10_cuda_check_implementation). sdpa is
#                       numerically equivalent, only slower, and does not touch generation.
#   evaluator.freq_epochs=null, freq_secs=null
#                    -- the in-training evaluator DEADLOCKS this stack in
#                       _evaluate -> rollout_controller.wait (step0g at step 59, step0h at
#                       87/116/145). gsm8k_grpo_lora.yaml ships freq_epochs: 1, so the epoch
#                       trigger is live unless we kill it here. Validation is done by scoring
#                       saved checkpoints offline instead, which is a better signal anyway.
#   saver.freq_steps -- the reference saves on epochs only; without step saves there is no
#                       validation curve for a run this long.
#   ulimit -n        -- 1024 descriptors cannot serve batch x n_samples concurrent rollouts;
#                       exhaustion surfaces as Errno 24 inside the workflow and AReaL scores
#                       a failed workflow as reward 0.0, laundering the failure into the
#                       training signal.
#   PYTHONUNBUFFERED -- stdbuf cannot unbuffer CPython, and a block-buffered trainer stdout
#                       delays the supervisor's stall detection.
#   rc=${PIPESTATUS[0]} -- exit with the TRAINER's status. An earlier harness always exited
#                       0 and reported a failed run as a success.
#
# kl_ctl STAYS 0.0. Not a preference: at kl_ctl > 0 rl_trainer.py:206 builds a REF model,
# and ref.scheduling_strategy in this yaml is colocation-on-actor, which would put a second
# 30B fp32 model on the actor's four cards. If you want a KL term you must first re-budget
# the memory above. The preflight refuses if kl_ctl != 0 while ref is configured.
#
# USAGE. This script REFUSES to run unattended. Launch it through the supervisor:
#     nohup bash experiments/harness/supervise.sh experiments/harness/lora30b.sh \
#           ~/runs/lora30b 3 1800 5400 >> ~/runs/chain_lora30b.log 2>&1 &
# supervise.sh derives its kill patterns from basename(launch .sh), so the experiment name
# must stay `lora30b` for the watchdog to be able to reap this run; the preflight enforces
# that too. For a config-only check that touches no GPU:
#     DRY_RUN=1 bash experiments/harness/lora30b.sh
set -u -o pipefail

export PATH="$HOME/.local/bin:$PATH"
# The two boxes have different venvs; pick whichever exists rather than hardcoding one.
for V in "$HOME/venv312b/bin/activate" "/venv/main/bin/activate"; do
  [ -f "$V" ] && { source "$V"; break; }
done
cd "$HOME/areal-selfevo" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 131072 || echo "WARNING: could not raise the file-descriptor limit"
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------------------
# Knobs. Every value the preflight asserts is a variable, because an assertion only earns
# its keep if the thing it checks is settable wrongly.
# ---------------------------------------------------------------------------------------
DRY_RUN="${DRY_RUN:-0}"                       # 1 = preflight + config dump, no launch
EXP="${EXP_NAME:-lora30b}"                    # must match supervise.sh's basename tag
TRIAL="${TRIAL_NAME:-t1}"
N_GPUS="${N_GPUS:-8}"                         # 8x A100 80GB
GPU_GIB="${GPU_GIB:-79.25}"                   # what torch reports usable on an 80GB A100
ACTOR_BACKEND="${ACTOR_BACKEND:-fsdp:d4p1t1}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-sglang:d2p1t2}"
SCHED_STRATEGY="${SCHED_STRATEGY:-separation}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"
MIN_KV_GIB="${MIN_KV_GIB:-8}"                 # refuse a server with less KV than this
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
MAX_LORA_RANK="${MAX_LORA_RANK:-$LORA_RANK}"  # sglang side; MUST equal LORA_RANK
# `-` and not `:-`: an explicitly EMPTY LORA_NAME must reach the config and be refused
# there, not be silently replaced by the default. The sglang bridge raises on an empty
# adapter name, and a knob whose wrong value cannot reach the check is an unchecked knob.
LORA_NAME="${LORA_NAME-lora30b}"
LORA_TARGETS="${LORA_TARGETS:-[q_proj,k_proj,v_proj,o_proj]}"
ACTOR_USE_LORA="${ACTOR_USE_LORA:-true}"
ROLLOUT_USE_LORA="${ROLLOUT_USE_LORA:-true}"
MEMORY_EFFICIENT_LOAD="${MEMORY_EFFICIENT_LOAD:-true}"
KL_CTL="${KL_CTL:-0.0}"
LR="${LR:-1.0e-5}"                            # LoRA lr; the upstream LoRA recipe uses 1.7e-4
N_SAMPLES="${N_SAMPLES:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SAVE_FREQ_STEPS="${SAVE_FREQ_STEPS:-25}"
GPU_BUSY_MIB="${GPU_BUSY_MIB:-2048}"
export WANDB_MODE="${WANDB_MODE:-online}"
export AREAL_WEIGHT_SYNC_RETRIES="${AREAL_WEIGHT_SYNC_RETRIES:-5}"

# ---------------------------------------------------------------------------------------
# Checkpoint resolution. Prefer the local 30B snapshot; fall back to the 27B only if the
# 30B snapshot is INCOMPLETE, and say which one was picked. Resolving to an absolute
# snapshot directory rather than a repo id keeps HF_HOME out of it -- the model lives in
# ~/hf_cache and the datasets live in ~/.cache/huggingface, and exporting HF_HOME for one
# would hide the other.
# ---------------------------------------------------------------------------------------
resolve_snapshot() {
  # Echo the newest snapshot dir of a cached HF repo, or nothing. $1=cache root, $2=repo id
  local root="$1" repo="$2" d
  d="$root/hub/models--${repo//\//--}"
  [ -d "$d/snapshots" ] || return 1
  ls -1dt "$d"/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::'
}
snapshot_is_complete() {
  # True when every shard named by the safetensors index exists and the bytes add up.
  local s="$1"
  [ -f "$s/config.json" ] || return 1
  python3 - "$s" <<'CKPT'
import json, os, sys
s = sys.argv[1]
idx = os.path.join(s, "model.safetensors.index.json")
if not os.path.exists(idx):
    sys.exit(0 if os.path.exists(os.path.join(s, "model.safetensors")) else 1)
j = json.load(open(idx))
files = sorted(set(j["weight_map"].values()))
missing = [f for f in files if not os.path.exists(os.path.join(s, f))]
if missing:
    sys.exit(1)
have = sum(os.path.getsize(os.path.join(s, f)) for f in files)
sys.exit(0 if have >= float(j["metadata"]["total_size"]) else 1)
CKPT
}

if [ -z "${MODEL_PATH:-}" ]; then
  PRIMARY=$(resolve_snapshot "$HOME/hf_cache" "FrontisAI/Frontis-MA1-30B" || true)
  FALLBACK=$(resolve_snapshot "$HOME/.cache/huggingface" "Qwen/Qwen3.8-27B" || true)
  if [ -n "$PRIMARY" ] && snapshot_is_complete "$PRIMARY"; then
    MODEL_PATH="$PRIMARY"; MODEL_WHICH="FrontisAI/Frontis-MA1-30B (preferred, local)"
  elif [ -n "$FALLBACK" ] && snapshot_is_complete "$FALLBACK"; then
    MODEL_PATH="$FALLBACK"; MODEL_WHICH="Qwen/Qwen3.8-27B (FALLBACK: 30B snapshot incomplete)"
  else
    echo "no complete snapshot for either FrontisAI/Frontis-MA1-30B or Qwen/Qwen3.8-27B"
    exit 2
  fi
else
  MODEL_WHICH="MODEL_PATH override"
fi

RUN="$HOME/runs/${EXP}"; mkdir -p "$RUN"
LOG="$RUN/train.log"
DUMP="$RUN/resolved_config.yaml"
FILTER="$HOME/areal-selfevo/experiments/harness/logfilter.py"   # audit D14: run the repo copy

# Reuse the admin key already on this host; never re-embed the secret in this file.
KEY=$(grep -oE "admin_api_key=[A-Za-z0-9_-]+" "$HOME/step0.sh" 2>/dev/null | head -1 | cut -d= -f2)
[ -n "$KEY" ] || KEY=$(cat "$HOME/.areal_admin_key" 2>/dev/null)
[ -n "$KEY" ] || { echo "no admin key found in ~/step0.sh or ~/.areal_admin_key"; exit 2; }

# ---------------------------------------------------------------------------------------
# The override list. ONE array, used by both the preflight and the trainer, so the config
# that is checked is by construction the config that runs. `+` where the key is absent from
# gsm8k_grpo_lora.yaml and hydra's struct mode would otherwise reject the override.
# ---------------------------------------------------------------------------------------
OVERRIDES=(
  scheduler.type=local
  cluster.fileroot="$HOME/areal-runs"
  cluster.n_gpus_per_node="$N_GPUS"

  actor.path="$MODEL_PATH"
  ref.path="$MODEL_PATH"
  actor.backend="$ACTOR_BACKEND"
  rollout.backend="$ROLLOUT_BACKEND"

  # (d): forced, not inherited. Both engines get their own cards.
  +actor.scheduling_strategy.type="$SCHED_STRATEGY"
  +rollout.scheduling_strategy.type="$SCHED_STRATEGY"

  # (a): the load-time fix. Without this every rank puts the whole fp32 model on one card.
  +actor.fsdp.memory_efficient_load="$MEMORY_EFFICIENT_LOAD"

  # (b): LoRA on BOTH sides, with the sglang counterparts written out rather than left to
  # the yaml's interpolation, so that the preflight is checking values and not a template.
  rollout.use_lora="$ROLLOUT_USE_LORA"
  actor.use_lora="$ACTOR_USE_LORA"
  actor.lora_rank="$LORA_RANK"
  actor.lora_alpha="$LORA_ALPHA"
  actor.target_modules="$LORA_TARGETS"
  gconfig.lora_name="$LORA_NAME"
  sglang.enable_lora="$ROLLOUT_USE_LORA"
  sglang.max_lora_rank="$MAX_LORA_RANK"
  ++sglang.disable_cuda_graph=${DISABLE_CUDA_GRAPH:-true}
  sglang.mem_fraction_static="$MEM_FRACTION_STATIC"

  actor.kl_ctl="$KL_CTL"
  actor.optimizer.lr="$LR"
  gconfig.n_samples="$N_SAMPLES"
  train_dataset.batch_size="$BATCH_SIZE"
  saver.freq_steps="$SAVE_FREQ_STEPS"
  evaluator.freq_epochs=null
  evaluator.freq_secs=null
  evaluator.freq_steps=null
  +actor.attn_impl=sdpa
  +ref.attn_impl=sdpa
  +rollout.agent.admin_api_key="$KEY"
  stats_logger.wandb.mode="$WANDB_MODE"
  +stats_logger.wandb.project="${WANDB_PROJECT:-selfevo-lora30b}"
  +stats_logger.wandb.name="$EXP"
  experiment_name="$EXP"
  trial_name="$TRIAL"
)

# EXTRA_ARGS is word-split on purpose (supervise.sh appends `recover.mode=auto` on every
# restart, and operators use it for one-off overrides). It is spliced in HERE, before the
# preflight, so the preflight validates the config that will actually run -- an override the
# preflight never sees is an override nobody checked.
# ---- routing: OFF by default, so this script's existing behaviour is unchanged ----
# The run this launcher has produced so far has group_routing null, which makes it plain policy
# optimisation with an adapter: the matched CONTROL, not the method. That was mis-described for
# several hours because the run's NAME says lora30b and nothing in the launcher said routing was
# absent. ROUTE=1 turns on the treatment arm.
#
# The two stabilisers default on for the treatment because the unstabilised constant is already
# measured to be harmful at 1.5B: it breaks the zero-mean advantage property and every arm that
# carried it crossed into ~99% truncation between steps 174 and 199. Running the treatment
# without them would reproduce a known failure at twenty times the cost.
ROUTE_ARGS=()
if [ "${ROUTE:-0}" = "1" ]; then
  ROUTE_ARGS=(
    "+actor.group_routing.enabled=true"
    "+actor.group_routing.solved_advantage=${SOLVED_ADV:-0.5}"
    "+actor.group_routing.unsolved_advantage=0.0"
    "+actor.group_routing.router=${ROUTER:-null}"
    "+actor.group_routing.zero_mean=${ZERO_MEAN:-true}"
    "+actor.group_routing.exclude_truncated_from_sft=${EXCLUDE_TRUNC:-true}"
  )
fi

read -r -a EXTRA_ARR <<< "${EXTRA_ARGS:-}"
HYDRA_ARGS=(
  --config examples/math/gsm8k_grpo_lora.yaml
  "${OVERRIDES[@]}"
  ${ROUTE_ARGS[@]+"${ROUTE_ARGS[@]}"} ${EXTRA_ARR[@]+"${EXTRA_ARR[@]}"}
)

# ---------------------------------------------------------------------------------------
# WATCHDOG. supervise.sh is the only thing that reaps this stack when it wedges -- sglang
# servers carry no experiment name on their command line and survive an experiment-scoped
# pkill, holding tens of GB into the next attempt. A 30B run left unattended after a stall
# costs a full box. Refuse rather than start blind.
# ---------------------------------------------------------------------------------------
PARENT_CMD=$(ps -o args= -p "$PPID" 2>/dev/null || true)
case "$PARENT_CMD" in
  *supervise.sh*) SUPERVISED=1 ;;
  *)              SUPERVISED=0 ;;
esac
if [ "$DRY_RUN" != "1" ] && [ "$SUPERVISED" != "1" ] && [ "${ALLOW_UNSUPERVISED:-0}" != "1" ]; then
  echo "REFUSING TO LAUNCH: no supervisor. Parent is: ${PARENT_CMD:-<unknown>}"
  echo "  bash experiments/harness/supervise.sh experiments/harness/lora30b.sh ~/runs/$EXP 3 1800 5400"
  echo "Set ALLOW_UNSUPERVISED=1 only if you are babysitting this by hand."
  exit 5
fi
# supervise.sh kills by basename(launch .sh); a different experiment_name is unreapable.
if [ "$SUPERVISED" = "1" ] && [ "${EXP#lora30b}" = "$EXP" ]; then
  echo "REFUSING TO LAUNCH: EXP_NAME='$EXP' does not start with 'lora30b', so"
  echo "supervise.sh's pkill -f experiment_name=lora30b would not match this run."
  exit 5
fi

# One run at a time. Without this, a second launch truncates the log the live run is
# appending to and the watchdog is reading (audit D7).
exec 9>"$RUN/.lock"
flock -n 9 || { echo "a ${EXP} run is already active in $RUN"; exit 3; }

# Rotate rather than truncate, so a previous run's evidence survives (audit D7).
[ -s "$LOG" ] && mv "$LOG" "$LOG.$(date +%s)"

# ---------------------------------------------------------------------------------------
# PREFLIGHT. Runs on the CPU with the GPUs masked off, resolves the SAME overrides through
# AReaL's own config machinery into the real dataclasses (so __post_init__ validation runs),
# writes the fully-resolved config to $DUMP, and refuses on anything that made lora27b fail
# or would make this run silently wrong. Everything it checks, it prints.
# ---------------------------------------------------------------------------------------
echo "=== ${EXP}: preflight ($(date -Is)) ===" | tee -a "$LOG"
echo "model: $MODEL_WHICH" | tee -a "$LOG"
echo "       $MODEL_PATH" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="" \
LORA30B_DUMP="$DUMP" \
LORA30B_NGPUS="$N_GPUS" \
LORA30B_GPU_GIB="$GPU_GIB" \
LORA30B_MIN_KV_GIB="$MIN_KV_GIB" \
LORA30B_MODEL="$MODEL_PATH" \
python3 - "${HYDRA_ARGS[@]}" <<'PREFLIGHT' 2>&1 | tee -a "$LOG"
"""Refuse a 30B LoRA RL launch that would OOM, sample the wrong policy, or share cards.

Resolves the launcher's overrides through AReaL's own Hydra + dataclass path so that what
is checked is what the trainer will build, dumps the result, and asserts the specific
conditions that broke lora27b. Exits non-zero on the first failure set; prints every check
either way, because a preflight whose output you cannot read is a preflight nobody audits.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys

import yaml
from omegaconf import OmegaConf

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import GRPOConfig, parse_cli_args, to_structured_cfg

GIB = 1024.0 ** 3
N_GPUS = int(os.environ["LORA30B_NGPUS"])
GPU_GIB = float(os.environ["LORA30B_GPU_GIB"])
MIN_KV_GIB = float(os.environ["LORA30B_MIN_KV_GIB"])
MODEL = os.environ["LORA30B_MODEL"]
DUMP = os.environ["LORA30B_DUMP"]

_rows: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str) -> bool:
    """Record one preflight verdict and return it, so callers can short-circuit."""
    _rows.append((bool(ok), label, detail))
    return bool(ok)


def note(label: str, detail: str) -> None:
    """Record an informational line that can never fail the preflight."""
    _rows.append((None, label, detail))


def model_bytes(path: str) -> tuple[float, str]:
    """Return (parameter count, note) read from the checkpoint's safetensors index."""
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        j = json.load(open(idx))
        total = float(j["metadata"]["total_size"])
        files = sorted(set(j["weight_map"].values()))
        on_disk = sum(
            os.path.getsize(os.path.join(path, f))
            for f in files
            if os.path.exists(os.path.join(path, f))
        )
        cfg = json.load(open(os.path.join(path, "config.json")))
        width = 2 if str(cfg.get("torch_dtype", "bfloat16")).endswith("16") else 4
        return total / width, (
            f"{len(files)} shards, index total_size={total:.0f} B, "
            f"on disk {on_disk:.0f} B"
        )
    single = os.path.join(path, "model.safetensors")
    if os.path.exists(single):
        return os.path.getsize(single) / 2.0, "single-file safetensors"
    raise FileNotFoundError(f"no safetensors index or model.safetensors under {path}")


# -- 1. resolve, exactly as the trainer will ------------------------------------------
cfg, _ = parse_cli_args(sys.argv[1:])
obj = OmegaConf.to_object(to_structured_cfg(cfg, config_cls=GRPOConfig))
with open(DUMP, "w") as fh:
    yaml.dump(dataclasses.asdict(obj), fh, default_flow_style=False, sort_keys=False)
note("resolved config", f"written to {DUMP}")
check(
    type(obj.actor).__name__ == "PPOActorConfig",
    "config is the real dataclass",
    f"actor is {type(obj.actor).__name__}; a DictConfig here means __post_init__ was skipped",
)

a, r, sg, g = obj.actor, obj.rollout, obj.sglang, obj.gconfig

# -- 2. the checkpoint exists and is whole --------------------------------------------
ckpt_ok = check(
    os.path.isdir(a.path),
    "checkpoint path exists",
    a.path if os.path.isdir(a.path) else f"{a.path} IS NOT A DIRECTORY ON THIS HOST",
)
n_params = 0.0
if ckpt_ok:
    try:
        n_params, detail = model_bytes(a.path)
        note("checkpoint", f"{n_params / 1e9:.2f}B params; {detail}")
    except Exception as exc:  # noqa: BLE001 - report, do not crash the preflight
        check(False, "checkpoint is readable", f"{type(exc).__name__}: {exc}")

# -- 3. LoRA is enabled on BOTH sides, consistently ------------------------------------
check(
    bool(a.use_lora) and bool(r.use_lora),
    "LoRA consistent across actor and rollout",
    f"actor.use_lora={a.use_lora} rollout.use_lora={r.use_lora} "
    "(training an adapter while the server holds the base model samples one policy and "
    "updates another)",
)
check(
    bool(sg.enable_lora) is bool(a.use_lora),
    "sglang.enable_lora tracks actor.use_lora",
    f"sglang.enable_lora={sg.enable_lora} actor.use_lora={a.use_lora} "
    "(null/false drops --enable-lora from the server command line entirely)",
)
check(
    sg.max_lora_rank == a.lora_rank,
    "sglang.max_lora_rank == actor.lora_rank",
    f"max_lora_rank={sg.max_lora_rank} lora_rank={a.lora_rank}",
)
check(
    bool(g.lora_name),
    "gconfig.lora_name is set",
    f"lora_name={g.lora_name!r} (the sglang bridge raises when it is empty)",
)
check(
    a.weight_update_mode == "disk",
    "actor.weight_update_mode == disk",
    f"weight_update_mode={a.weight_update_mode!r} "
    "(P2P transports cannot carry PEFT-wrapped tensors; rl_trainer.py:378-381)",
)

# -- 4. scheduling: nothing that needs a GPU may be colocated --------------------------
for role, engine in (("actor", a), ("rollout", r)):
    st = getattr(engine.scheduling_strategy, "type", None)
    st = getattr(st, "value", st)
    check(
        st != "colocation",
        f"{role} scheduling strategy is not colocation",
        f"{role}.scheduling_strategy.type={st!r}",
    )

# -- 5. no ref model, so its colocation-on-actor never fires ---------------------------
check(
    float(a.kl_ctl) == 0.0,
    "actor.kl_ctl == 0.0, so no ref model is built",
    f"kl_ctl={a.kl_ctl} (rl_trainer.py:206 builds ref only when kl_ctl > 0; ref is "
    "configured colocation-on-actor and would put a second 30B on the actor's cards)",
)

# -- 6. the load path ------------------------------------------------------------------
check(
    bool(a.fsdp.memory_efficient_load),
    "actor.fsdp.memory_efficient_load is on",
    f"memory_efficient_load={a.fsdp.memory_efficient_load} (off means every rank "
    f"materialises the whole {a.optimizer_dtype} model on its own card before sharding)",
)

# -- 7. the GPUs actually add up -------------------------------------------------------
actor_alloc = ModelAllocation.from_str(a.backend, name="actor")
rollout_alloc = ModelAllocation.from_str(r.backend, name="rollout")
aw = actor_alloc.parallel.world_size
rw = rollout_alloc.parallel.world_size
check(
    aw + rw <= N_GPUS,
    "actor + rollout GPUs fit the node",
    f"{a.backend} needs {aw} + {r.backend} needs {rw} = {aw + rw} of {N_GPUS} "
    "(LocalScheduler._allocate_gpus round-robins with no occupancy tracking, so an "
    "over-subscription silently WRAPS onto cards another role already holds)",
)

# -- 8. the memory arithmetic, computed from the resolved config -----------------------
if n_params:
    store = 4.0 if str(a.optimizer_dtype).endswith("32") else 2.0
    train_total = n_params * store / GIB
    per_actor = train_total / max(actor_alloc.parallel.data_parallel_size, 1)
    note(
        "actor per-GPU base params",
        f"{n_params / 1e9:.2f}B x {store:.0f} B ({a.optimizer_dtype}) = "
        f"{train_total:.1f} GiB / dp{actor_alloc.parallel.data_parallel_size} = "
        f"{per_actor:.1f} GiB of {GPU_GIB:.2f} GiB",
    )
    check(
        per_actor < GPU_GIB * 0.6,
        "actor shard leaves room for activations",
        f"{per_actor:.1f} GiB of {GPU_GIB:.2f} GiB per actor GPU",
    )
    note(
        "actor load-time peak",
        f"rank 0 builds on CPU: {train_total:.1f} GiB fp32 model + "
        f"{n_params * 2 / GIB:.1f} GiB bf16 state dict = "
        f"{train_total + n_params * 2 / GIB:.1f} GiB host RAM"
        if a.fsdp.memory_efficient_load
        else f"{train_total:.1f} GiB PER GPU -- this will OOM",
    )

    tp = rollout_alloc.parallel.tensor_parallel_size
    infer_per_gpu = (n_params * 2 / GIB) / max(tp, 1)
    pool = float(sg.mem_fraction_static) * GPU_GIB
    kv = pool - infer_per_gpu
    note(
        "sglang per-GPU",
        f"weights {n_params * 2 / GIB:.1f} GiB / tp{tp} = {infer_per_gpu:.1f} GiB; "
        f"static pool {sg.mem_fraction_static} x {GPU_GIB:.2f} = {pool:.1f} GiB; "
        f"KV = {kv:.1f} GiB; outside the pool {GPU_GIB - pool:.1f} GiB",
    )
    check(
        kv >= MIN_KV_GIB,
        "sglang KV cache above the floor",
        f"{kv:.1f} GiB >= {MIN_KV_GIB:.1f} GiB (raise tp, or mem_fraction_static, or "
        "lower MIN_KV_GIB deliberately)",
    )

# -- report ----------------------------------------------------------------------------
width = max(len(label) for _, label, _ in _rows)
failed = 0
for ok, label, detail in _rows:
    if ok is None:
        tag = "note"
    elif ok:
        tag = "ok  "
    else:
        tag = "FAIL"
        failed += 1
    print(f"[{tag}] {label.ljust(width)}  {detail}")
print(f"\nPREFLIGHT {'FAIL' if failed else 'PASS'} ({failed} failed)")
sys.exit(1 if failed else 0)
PREFLIGHT
pf=${PIPESTATUS[0]}
if [ "$pf" -ne 0 ]; then
  echo "PREFLIGHT REFUSED THE LAUNCH (rc=$pf); nothing was started." | tee -a "$LOG"
  exit 6
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN=1: preflight passed, resolved config at $DUMP, no GPU touched." | tee -a "$LOG"
  exit 0
fi

# ---------------------------------------------------------------------------------------
# Refuse to start on GPUs that already hold memory. A previous run's sglang servers carry no
# experiment name, so an experiment-scoped kill leaves them alive holding tens of GB; the new
# run's engines then stack on top and die with a CUDA OOM tens of minutes later. lora27b's
# card was in exactly this state -- a 57.93 GiB foreign process it never accounted for.
# ---------------------------------------------------------------------------------------
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
       | awk -F", " -v t="$GPU_BUSY_MIB" '$2 > t {printf "%s(%sMiB) ", $1, $2}')
if [ -n "$busy" ]; then
  echo "REFUSING TO LAUNCH: GPUs already hold memory: $busy" | tee -a "$LOG"
  echo "Free them first (sglang servers survive an experiment-scoped pkill), or raise" | tee -a "$LOG"
  echo "GPU_BUSY_MIB if this is deliberate co-tenancy." | tee -a "$LOG"
  exit 4
fi

echo "=== ${EXP}: launching ($(date -Is)) ===" >> "$LOG"
python3 examples/math/gsm8k_rl.py \
  "${HYDRA_ARGS[@]}" 2>&1 \
  | python3 "$FILTER" >> "$LOG" 2>>"$LOG"

# Exit with the TRAINER's status, not the echo's. Previously the harness always exited 0,
# so a failed run reported success -- the same mis-report the harness exists to prevent
# (audit D5).
rc=${PIPESTATUS[0]}
echo "LORA30B_EXIT=$rc" >> "$LOG"
exit "$rc"
