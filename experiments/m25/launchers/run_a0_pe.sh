#!/usr/bin/env bash
# A0 with recovery OFF and periodic eval ON.
#
# recover.mode=auto wedged the run: its first recover checkpoint, at step 24, deadlocked both
# actor ranks inside a distributed gather under torch.distributed.checkpoint, with /proc/<pid>/io
# byte counters unchanged and nothing timing out for 36 minutes. It had never fired before today,
# so nothing is lost by turning it off -- it has never once produced a usable resume -- and
# leaving it on wedges the run at step 24 every time.
#
# Every variable is EXPORTED on its own line, never as a prefix assignment on a continued
# command, because a comment spliced into such a chain has previously turned an exported variable
# into an unexported one and silently dropped the setting.
set -u

export RECOVER_MODE=disabled

export SELFEVO_PERIODIC_EVAL=1
export SELFEVO_PERIODIC_EVAL_FREQ_STEPS=50
export SELFEVO_PERIODIC_EVAL_LIMIT=64
export SELFEVO_PERIODIC_EVAL_MODEL=a0_math
export SELFEVO_PERIODIC_EVAL_BASE_MODEL=/home/ubuntu/hf_cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd
# SELFEVO_PERIODIC_EVAL_BASE_URL is deliberately NOT set, and it is no longer needed.
#
# Two corrections to what this comment used to say, both of them recorded because the wrong
# version of each cost a run.
#
#   1. The pin was never WRONG. The reaped run's own log shows its trainer generating against
#      :32735 more than a hundred times, and the current run landed on :32735 again -- read
#      off the serving process' command line and confirmed by listing the models it serves.
#      The ports once quoted as evidence against it (16866, 17672, 17909) came out of
#      /tmp/areal/name_resolve/<user>/<exp>/<trial>/workers/, which holds the RPC ports of
#      the worker PROCESSES. Those are not the sglang HTTP port and never were.
#   2. The pin is still not the right mechanism, for a different reason: the port is
#      allocated afresh on every launch (17727, 21698, 17762, 18284 and 32735 have all been
#      it in one day), so a constant written before the run is right only by luck.
#
# What replaced it: selfevo.periodic_eval.resolve_endpoint walks four sources in order --
# this variable, the engine's `addresses`, the rollout CONTROLLER's `server_infos` (the one
# that answers on this stack; the controller has no `addresses`, which is why the automatic
# path had never once worked), this trial's `gen_servers` record, and finally the serving
# process' own command line fenced by trial ownership. The address it found and the source
# that found it are logged with every point, as periodic_eval/endpoint/{port,source}. If
# nothing answers, the point refuses with status 10 (endpoint_undiscovered) rather than
# guessing at localhost. Setting this variable still wins outright, so the pin remains
# available as an override -- it is simply not required.
export SELFEVO_PERIODIC_EVAL_STATE=/home/ubuntu/areal-runs/a0_math_best_val.json
export SELFEVO_PERIODIC_EVAL_OUT_DIR=/home/ubuntu/runs/a0_periodic
export MATH_EVAL_DATA=/home/ubuntu/evaldata

# The token budget, set EXPLICITLY rather than left on the module default of 16384.
#
# The rollout server is launched with --context-length 4096, so it REFUSES any request whose
# prompt plus max_tokens exceeds that: measured 2026-09-02, max_tokens=16384 comes back HTTP
# 400 "Requested token count exceeds the model's maximum context length of 4096 tokens". The
# model's own config.json says 32768, so the harness clamp that reads config.json passes and
# every request then fails -- which is why this is set here as well as guarded in code.
#
# 2048 is chosen against the split rather than picked: the longest prompt in olympiadbench's
# search half is 1304 tokens, so 1304 + 2048 = 3352 fits inside 4096 for EVERY problem in it.
#
# This is NOT the headline budget. OlympiadBench's headline number was measured at 16384 and a
# score at 2048 is not comparable with it. Every logged point carries max_tokens,
# headline_max_tokens and budget_matches_headline=0 saying so, and the score series is called
# trend_score rather than accuracy for the same reason.
export SELFEVO_PERIODIC_EVAL_MAX_TOKENS=2048

# The tokenizer the evaluation tokenises with. The rollout server runs --skip-tokenizer-init,
# so it holds no tokenizer and every TEXT request to it fails; the evaluation speaks token ids
# and needs the snapshot on disk. It would be resolved from BASE_MODEL above (which IS this
# directory) if left unset; it is written out so the run records what it used.
export SELFEVO_PERIODIC_EVAL_MODEL_PATH=/home/ubuntu/hf_cache/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd

exec bash /home/ubuntu/harness4/run_a0.sh
