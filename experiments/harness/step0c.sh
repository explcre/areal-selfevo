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
#   step0c changes NOTHING the reference sets. Both failures were our own deviations.
#          A third was caught before it ran: we had set total_train_epochs to 1 against a
#          published 10, which at batch 256 gives only ~29 optimizer steps.
#
# The only remaining deviation is forced, and is documented rather than silent:
#   attn_impl=sdpa -- the prebuilt flash-attn wheel is ABI-incompatible with torch 2.9.1
#                     (undefined symbol c10_cuda_check_implementation). sdpa is
#                     numerically equivalent, only slower. It does not touch generation.
#
#   gconfig.min_new_tokens=1 -- forces at least one non-EOS token. As entropy falls the
#                     policy samples EOS first; the all-EOS completion makes sglang return
#                     500 (both EOS and PAD are stop tokens), and AReaL scores the failed
#                     trajectory as reward 0.0, which drives entropy lower still. Note this
#                     field was DEAD upstream -- declared in cli_args.py and never
#                     forwarded -- so it is only effective together with our
#                     sglang_remote.py change that puts it in sampling_params.
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

RUN="$HOME/runs/step0c"; mkdir -p "$RUN"
LOG="$RUN/train.log"
FILTER="$HOME/areal-selfevo/experiments/harness/logfilter.py"   # audit D14: run the repo copy

# One run at a time. Without this, a second launch truncates the log the live run is
# appending to and the watchdog is reading (audit D7).
exec 9>"$RUN/.lock"
flock -n 9 || { echo "a step0c run is already active in $RUN"; exit 3; }

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
  gconfig.min_new_tokens=1 \
  +actor.attn_impl=sdpa +ref.attn_impl=sdpa \
  +rollout.agent.admin_api_key="$KEY" \
  experiment_name=step0c trial_name=t1 2>&1 \
  | python3 "$FILTER" >> "$LOG" 2>>"$LOG"

# Exit with the TRAINER's status, not the echo's. Previously the script always exited 0,
# so a failed run reported success -- the same mis-report the harness exists to prevent
# (audit D5).
rc=${PIPESTATUS[0]}
echo "STEP0C_EXIT=$rc" >> "$LOG"
exit "$rc"
