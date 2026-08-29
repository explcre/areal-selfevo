#!/usr/bin/env bash
# Step 0c: AReaL's published GSM8K GRPO config on 8x A100, with the reference batch size.
#
# History of this run, and why each knob is where it is:
#   step0  overrode gconfig.max_new_tokens=512  -> truncated CoT produced all-EOS
#          generations, 3,468,076 sglang 500s, and a stall at 159/233.
#   step0b restored max_new_tokens=1024 (0 all-EOS errors) but kept
#          train_dataset.batch_size=32 against a published 256. With kl_ctl=0.0,
#          eps_clip=0.4 and group size 4, that 8x smaller batch gave high-variance
#          unregularized updates: entropy collapsed 4.13 -> 1.2e-06, sequence length ran
#          to the 1024 cap, and task_reward fell 0.85 -> 0.23.
#   step0f changes NOTHING the reference sets. Both failures were our own deviations.
#          A third was caught before it ran: we had set total_train_epochs to 1 against a
#          published 10, which at batch 256 gives only ~29 optimizer steps.
#
# The only remaining deviation is forced, and is documented rather than silent:
#   attn_impl=sdpa -- the prebuilt flash-attn wheel is ABI-incompatible with torch 2.9.1
#                     (undefined symbol c10_cuda_check_implementation). sdpa is
#                     numerically equivalent, only slower. It does not touch generation.
#
#   (min_new_tokens was tried and REMOVED: it crashes sglang with AReaL, see EXPERIMENTS.md)
#                     policy samples EOS first; the all-EOS completion makes sglang return
#                     500 (both EOS and PAD are stop tokens), and AReaL scores the failed
#                     trajectory as reward 0.0, which drives entropy lower still. Note this
#                     field was DEAD upstream -- declared in cli_args.py and never
#                     forwarded -- so it is only effective together with our
#                     sglang_remote.py change that puts it in sampling_params.
#
# THE EXPERIMENTAL VARIABLES (two, both justified by measurement):
#   gconfig.n_samples=8 (batch stays at the published 256) -- ISOLATES group size.
#                     step0e also halved the batch to hold sequences-per-step at 1024. That
#                     was compute-neutral for GENERATION but not for OPTIMISATION: batch 128
#                     doubles the optimizer steps per epoch (580 vs 290), so it conflated
#                     "more updates on noisier gradients" with the group-size change. It
#                     declined faster than the published config (reward 0.713 -> 0.297 by
#                     step 7). This run pays 2x generation instead, keeping the update
#                     count identical to published so G is the only variable.
#                     ORIGINAL RATIONALE, unchanged:
#                     The published 256x4 and this 128x8 both generate 1024 sequences per
#                     step, but at the measured solve rate p=0.76 a group of 4 is unanimous
#                     0.76^4 + 0.24^4 = 34% of the time, and a unanimous group has every
#                     A_i = r_i - rbar exactly 0, so it contributes NO gradient at all.
#                     G=8 drops that to ~11%. Our own group-size law (EXPERIMENTS.md) wants
#                     G >= 6.9 at eps=0.1, against the published 4.
#
#   actor.kl_ctl=0.01 -- step0c reproduced the published config faithfully and COLLAPSED:
#                     entropy 4.03 -> 0.025 by step 22, then train reward fell off a cliff
#                     0.748 -> 0.039 between steps 62 and 76, held-out 0.687 -> 0.378.
#                     AReaL offers no entropy regulariser at all (no entropy_coef anywhere),
#                     and the reference sets kl_ctl: 0.0, so nothing constrains the update.
#                     reward_scaling is not a lever: adv_norm normalises by batch std, so a
#                     10x reward scale cancels. kl_ctl is the only regulariser available and
#                     it is genuinely wired (actor.py:255).
#
# Hygiene, not a treatment -- prevents a crash and wasted compute, does not change the
# objective:
#                     which sglang rejects with a 500. Verified end to end across all 9
#                     hops by experiments/harness/verify_chain.py.
#
# One addition, which changes measurement and not training:
#   evaluator.freq_steps=20 -- the reference fires the evaluator once per epoch, so a
#                     1-epoch run yields a single eval point and no validation curve.
set -u -o pipefail

export PATH="$HOME/.local/bin:$PATH"
source "$HOME/venv312b/bin/activate"
cd "$HOME/areal-selfevo" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The default soft limit of 1024 file descriptors cannot serve 1024 concurrent rollouts
# (batch 256 x n_samples 4) plus max_concurrent_rollouts=256. Exhaustion surfaces as
# "OSError: [Errno 24] Too many open files" inside the rollout workflow -- and AReaL
# scores a failed workflow as reward 0.0, so the failure is laundered into the training
# signal instead of raised. The hard limit here is 1048576.
ulimit -n 131072 || echo "WARNING: could not raise the file-descriptor limit"
# stdbuf cannot unbuffer CPython (it overrides libc stdio, which CPython bypasses), so the
# trainer's stdout was block-buffered into the pipe and delayed stall detection (audit D8).
export PYTHONUNBUFFERED=1

RUN="$HOME/runs/step0f"; mkdir -p "$RUN"
LOG="$RUN/train.log"
FILTER="$HOME/areal-selfevo/experiments/harness/logfilter.py"   # audit D14: run the repo copy

# One run at a time. Without this, a second launch truncates the log the live run is
# appending to and the watchdog is reading (audit D7).
exec 9>"$RUN/.lock"
flock -n 9 || { echo "a step0f run is already active in $RUN"; exit 3; }

# Rotate rather than truncate, so a previous run's evidence survives (audit D7).
[ -s "$LOG" ] && mv "$LOG" "$LOG.$(date +%s)"

# Reuse the admin key already on this host; never re-embed the secret in this file.
KEY=$(grep -oE "admin_api_key=[A-Za-z0-9_-]+" "$HOME/step0.sh" | head -1 | cut -d= -f2)
[ -n "$KEY" ] || { echo "no admin key found"; exit 2; }

python3 examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  scheduler.type=local \
  cluster.fileroot="$HOME/areal-runs" \
  evaluator.freq_steps=20 \
  actor.kl_ctl=0.01 \
  gconfig.n_samples=8 \
  +actor.attn_impl=sdpa +ref.attn_impl=sdpa \
  +rollout.agent.admin_api_key="$KEY" \
  experiment_name=step0f trial_name=t1 2>&1 \
  | python3 "$FILTER" >> "$LOG" 2>>"$LOG"

# Exit with the TRAINER's status, not the echo's. Previously the script always exited 0,
# so a failed run reported success -- the same mis-report the harness exists to prevent
# (audit D5).
rc=${PIPESTATUS[0]}
echo "STEP0F_EXIT=$rc" >> "$LOG"
exit "$rc"
