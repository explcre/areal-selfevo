#!/usr/bin/env bash
# step0m -- the solved-branch A/B, the experiment the silence measurement asked for.
#
# 31.4% of all groups are RL-silent BECAUSE THEY ARE SOLVED (1152 group observations on the
# control arm). Those groups already contain a correct sample, so a supervised target exists
# inside the rollout at zero teacher cost. This run asks the only question that matters
# about them: does using that target help, or does it just spend entropy?
#
# The two arms differ in ONE flag. Everything else -- model, data, seed, optimiser,
# normalisation, token budget -- is identical, and the off arm is bit-identical to vanilla
# GRPO because group_routing defaults to None.
#
#   ARM=off    group_routing absent           vanilla GRPO: the silent groups are computed
#                                             and contribute exactly zero
#   ARM=on     solved_advantage=$SOLVED_ADV   ours: reuse them, SFT on the group's own
#                                             correct samples, no extra generation
#   ARM=dapo   dapo_dynamic_sampling          DAPO (2503.14476): DISCARD them and oversample
#                                             until the batch refills
#   ARM=router router=$ROUTER                  a Router decides the mode for EVERY group from
#                                             observability features, instead of the fixed
#                                             silent-and-solved rule the `on` arm applies
#
# The router arm is NOT a fourth point on the same axis. `on` and `dapo` differ in what they
# DO with a fixed, hand-chosen partition of the batch; `router` differs in WHO CHOOSES the
# partition. Its comparison is against `on` at matched generation budget: same set, same
# modes available, the only difference being that a learned policy picks per group. If it
# does not beat `on`, the honest reading is that the fixed rule was already the right rule
# -- which is a publishable null, not a failed run.
#
# ROUTER names a key in selfevo.compose.ROUTERS. `contextual` is the LinUCB bandit over the 7
# observability features; `random` is its matched control -- same mode proportions, shuffled
# across units -- and any claim about the learned router has to beat THAT, not just `off`.
#
# The three arms act on the SAME set -- a group with zero reward variance -- which is what
# makes DAPO the baseline rather than related work. Comparison axis is matched GENERATION
# BUDGET, not matched steps: DAPO's kept batch is denser per step, so equal-step comparison
# would flatter it. Read rollout/accepted__count and rollout/rejected__count for the real
# multiplier (those counts only reach the trainer after the export_stats fix; before it the
# trainer saw a constant 1.0 and the cost was unmeasurable).
#
# dynamic_bs stays FALSE for the DAPO arm. It reads as the oversampling switch and is the
# opposite: dynamic_bs=true stops after batch_size ATTEMPTS and returns a shrunken batch,
# while false keeps generating until batch_size are ACCEPTED, which is DAPO's oversampling.
#
# SOLVED_ADV defaults to 0.5. With reward_norm std_level=group the informative advantages
# are standardised to |A| ~ 1, so 0.5 is half a typical advantage -- a deliberate first
# value, not a tuned one, and it should be reported as such.
#
# Runs the arms SEQUENTIALLY on all 8 GPUs rather than side by side on 4 each: two AReaL
# jobs on one node share port allocation and temp dirs, and a collision costs more hours
# than serialising does.
set -u -o pipefail

ARM="${ARM:?set ARM=off, on, dapo or router}"
SOLVED_ADV="${SOLVED_ADV:-0.5}"
ROUTER="${ROUTER:-contextual}"
case "$ARM" in
  off)  ROUTING_ARGS=() ;;
  on)   ROUTING_ARGS=(
          "+actor.group_routing.enabled=true"
          "+actor.group_routing.solved_advantage=${SOLVED_ADV}"
        ) ;;
  dapo) ROUTING_ARGS=(
          "+dynamic_filter_fn=selfevo.baselines.dapo.dapo_dynamic_sampling"
        ) ;;
  # solved_advantage is still passed: for a routed group it is the SFT magnitude, so the
  # router arm and the `on` arm write the same number when they agree on the mode. Leaving
  # it at its default would silently make the two arms differ by more than the decision.
  router)
        ROUTING_ARGS=(
          "+actor.group_routing.enabled=true"
          "+actor.group_routing.solved_advantage=${SOLVED_ADV}"
          "+actor.group_routing.router=${ROUTER}"
        ) ;;
  *)    echo "ARM must be 'off', 'on', 'dapo' or 'router', got '$ARM'"; exit 2 ;;
esac

export PATH="$HOME/.local/bin:$PATH"
# The two boxes have different venvs; pick whichever exists rather than hardcoding one.
for V in "$HOME/venv312b/bin/activate" "/venv/main/bin/activate"; do
  [ -f "$V" ] && { source "$V"; break; }
done
cd "$HOME/areal-selfevo" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 131072 || echo "WARNING: could not raise the file-descriptor limit"
export PYTHONUNBUFFERED=1

# Fail here rather than 40 minutes in. An unregistered router name reaches _route_groups only
# after the model is loaded and the first batch is generated, and the ValueError it raises
# there costs a full startup to discover. Placed AFTER the venv activation on purpose: the
# import needs the environment the trainer will actually run in, and an earlier version of
# this check referenced a $PY that is not defined until below -- which under `set -u` fails
# every arm, not just the router one.
if [ "$ARM" = "router" ]; then
  python3 -c "
import sys
from selfevo.compose import ROUTERS
name = '${ROUTER}'
if name not in ROUTERS:
    sys.exit('ROUTER=%r is not registered; known: %s' % (name, sorted(ROUTERS)))
" || exit 2
fi
# Online by default: a run whose curve is only on the box is a run nobody can check,
# and these boxes are rented. WANDB_MODE=offline or disabled opts out.
export WANDB_MODE="${WANDB_MODE:-online}"
# One transient rollout disconnect during a weight push otherwise aborts the whole run,
# and the surviving ranks then spin at 100% utilization looking healthy.
export AREAL_WEIGHT_SYNC_RETRIES="${AREAL_WEIGHT_SYNC_RETRIES:-5}"

# EXP_NAME lets two runs of the SAME arm coexist -- e.g. two solved_advantage values --
# and keeps the run directory written here identical to the one the supervisor watches.
# Without it the watchdog polls a path the launcher never creates, sees it stale forever,
# and kills a perfectly healthy run at the first stall check.
EXP="${EXP_NAME:-step0m-${ARM}}"
RUN="$HOME/runs/${EXP}"; mkdir -p "$RUN"
LOG="$RUN/train.log"
FILTER="$HOME/areal-selfevo/experiments/harness/logfilter.py"

exec 9>"$RUN/.lock"
flock -n 9 || { echo "a ${EXP} run is already active in $RUN"; exit 3; }
[ -s "$LOG" ] && mv "$LOG" "$LOG.$(date +%s)"

KEYFILE="$HOME/.areal_admin_key"
[ -f "$KEYFILE" ] || (umask 077; head -c 24 /dev/urandom | base64 | tr -d "/+=" > "$KEYFILE")
KEY=$(cat "$KEYFILE")
[ -n "$KEY" ] || { echo "no admin key found"; exit 2; }

# Refuse to start on GPUs that already hold memory. A previous run's sglang servers carry no
# experiment name, so an experiment-scoped kill leaves them alive holding ~66 GB each; the new
# run's servers then stack on top and the job dies with "Failed to CUDA calloc" tens of
# minutes later, after the weights are already partially updated. Cheaper to refuse here.
GPU_BUSY_MIB="${GPU_BUSY_MIB:-2048}"
busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
       | awk -F", " -v t="$GPU_BUSY_MIB" '$2 > t {printf "%s(%sMiB) ", $1, $2}')
if [ -n "$busy" ]; then
  echo "REFUSING TO LAUNCH: GPUs already hold memory: $busy" | tee -a "$LOG"
  echo "Free them first (sglang servers survive an experiment-scoped pkill), or raise" | tee -a "$LOG"
  echo "GPU_BUSY_MIB if this is deliberate co-tenancy." | tee -a "$LOG"
  exit 4
fi

echo "=== ${EXP}: ARM=${ARM} SOLVED_ADV=${SOLVED_ADV} args=${ROUTING_ARGS[*]:-none} ===" >> "$LOG"

python3 examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=local \
  cluster.fileroot="$HOME/areal-runs" \
  gconfig.n_samples=8 \
  actor.optimizer.lr=1.0e-6 \
  actor.eps_clip=0.2 \
  saver.freq_steps=25 \
  actor.kl_ctl=0.0 \
  ~actor.adv_norm \
  ++actor.mb_spec.granularity=8 \
  actor.path=Qwen/Qwen2.5-1.5B-Instruct \
  "${ROUTING_ARGS[@]}" \
  stats_logger.wandb.mode="${WANDB_MODE:-online}" \
  +stats_logger.wandb.project="${WANDB_PROJECT:-selfevo-routing}" \
  +stats_logger.wandb.name="${EXP}" \
  ${EXTRA_ARGS:-} \
  evaluator.freq_epochs=null \
  evaluator.freq_secs=null \
  +actor.attn_impl=sdpa +ref.attn_impl=sdpa \
  +rollout.agent.admin_api_key="$KEY" \
  experiment_name="${EXP}" trial_name=t1 2>&1 \
  | python3 "$FILTER" >> "$LOG" 2>>"$LOG"

rc=${PIPESTATUS[0]}
echo "STEP0M_${ARM}_EXIT=$rc" >> "$LOG"
exit "$rc"
