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

ARM="${ARM:?set ARM=off or ARM=on}"
SOLVED_ADV="${SOLVED_ADV:-0.5}"
case "$ARM" in
  off)  ROUTING_ARGS=() ;;
  on)   ROUTING_ARGS=(
          "+actor.group_routing.enabled=true"
          "+actor.group_routing.solved_advantage=${SOLVED_ADV}"
        ) ;;
  dapo) ROUTING_ARGS=(
          "+dynamic_filter_fn=selfevo.baselines.dapo.dapo_dynamic_sampling"
        ) ;;
  *)    echo "ARM must be 'off', 'on' or 'dapo', got '$ARM'"; exit 2 ;;
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
