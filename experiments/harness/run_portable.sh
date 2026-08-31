#!/usr/bin/env bash
# Unattended training or evaluation on a SHARED GPU box.
#
# Written for a machine we do not control: someone else's jobs may already be running, and
# the set of free GPUs is whatever is left. Nothing here assumes all 8 are ours, nothing
# requires a human at the keyboard, and every failure mode we have actually hit on our own
# boxes has a guard.
#
#   bash run_portable.sh                       # train, default arm, whatever GPUs are free
#   MODE=eval CKPT=/path/to/ckpt bash run_portable.sh
#   ARM=dapo MIN_GPUS=4 bash run_portable.sh
#
# Everything is an environment variable; see CONFIG below. Exit codes are meaningful:
#   0 success   3 another instance holds the lock   4 not enough free GPUs
#   5 setup failed   6 run failed after all retries
set -u -o pipefail

# ----------------------------------------------------------------- CONFIG ----
# cron and bare systemd units start with neither of these set, and `set -u` would abort on
# the first reference -- before anything has been logged.
USER="${USER:-$(id -un)}"
HOME="${HOME:-$(getent passwd "$(id -u)" 2>/dev/null | cut -d: -f6)}"
HOME="${HOME:-/tmp}"
HARNESS_START="$(date +%s)"

REPO_URL="${REPO_URL:-https://github.com/explcre/areal-selfevo.git}"
REPO_BRANCH="${REPO_BRANCH:-selfevo/a100}"
WORKDIR="${WORKDIR:-$HOME/selfevo-portable}"
OUTDIR="${OUTDIR:-$WORKDIR/out}"
MODE="${MODE:-train}"                 # train | eval
ARM="${ARM:-on}"                      # off | on | dapo   (train only)
SOLVED_ADV="${SOLVED_ADV:-0.5}"
EPOCHS="${EPOCHS:-5}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
CKPT="${CKPT:-}"                      # eval only; defaults to the newest we trained
BENCHES="${BENCHES:-math500}"
# Group size. The silent channel is predicted to be prompt HETEROGENEITY rather than
# binomial tail mass, which implies raising this will NOT shrink it. That prediction is
# the cheapest decisive experiment left, so the knob is exposed.
N_SAMPLES="${N_SAMPLES:-8}"
DATASET="${DATASET:-openai/gsm8k}"
# Anything else to hand the trainer verbatim, e.g. a different mb capacity for long data.
PORTABLE_EXTRA="${PORTABLE_EXTRA:-}"

# Shared-box etiquette.
# Explicit pin, for when a collaborator has been told WHICH boards are theirs. Bypasses
# auto-detection entirely: e.g. GPUS=4,5,6,7. Still checked for being free unless
# GPUS_FORCE=1, because being told a GPU is yours and it actually being idle are
# different claims, and starting anyway is how two jobs end up sharing a board.
GPUS="${GPUS:-}"
GPUS_FORCE="${GPUS_FORCE:-0}"
MIN_GPUS="${MIN_GPUS:-4}"             # refuse rather than thrash below this
MAX_GPUS="${MAX_GPUS:-8}"             # never claim more than this even if free
GPU_FREE_MIB="${GPU_FREE_MIB:-4096}"  # a GPU holding more than this belongs to someone else
WAIT_FOR_GPUS_S="${WAIT_FOR_GPUS_S:-0}"   # >0 waits for a neighbour's job to finish

# Resilience.
MAX_ATTEMPTS="${MAX_ATTEMPTS:-4}"
GPU_BUSY_ON_EXIT_MIB="${GPU_BUSY_ON_EXIT_MIB:-2048}"
# Static KV-cache fraction for each rollout server. Passed EXPLICITLY rather than
# inherited from the yaml, because a partial-box claim splits train and rollout
# differently and a measured 4-GPU run on an H200 put rollout servers at ~140 GB of
# 141 GB, leaving no headroom for the weight-sync broadcast -- the run then died in
# dist.broadcast tens of minutes in. Lower this if a partial claim OOMs.
MEM_FRACTION="${MEM_FRACTION:-0.8}"
STALL_S="${STALL_S:-1800}"
STARTUP_S="${STARTUP_S:-1200}"

# Reporting.
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-selfevo-routing}"
# Seconds, not just minutes: two copies started inside the same minute would otherwise
# share a RUN_NAME, and the loser's manifest would clobber the winner's.
RUN_NAME="${RUN_NAME:-${MODE}-${ARM}-$(hostname -s)-$(date +%m%d_%H%M%S)}"

# The manifest is written by a python heredoc that reads os.environ, so every value it
# reports has to be EXPORTED. A plain shell variable arrives there as an empty string, and
# a manifest full of empty strings is the silent failure that reads as a success.
export MODE ARM SOLVED_ADV MODEL BENCHES RUN_NAME WANDB_MODE WANDB_PROJECT

mkdir -p "$WORKDIR" "$OUTDIR"
LOG="$OUTDIR/${RUN_NAME}.log"
MANIFEST="$OUTDIR/${RUN_NAME}.manifest.json"
export PORTABLE_LOG="$LOG"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

write_manifest() {
  # A single structured file is what gets handed back, so it is written on EVERY exit path,
  # including failure -- a manifest that only exists on success cannot report a failure.
  local status="$1" note="${2:-}"
  if ! command -v python3 >/dev/null 2>&1; then
    # A failure that happens before setup finds an interpreter still has to leave a
    # manifest behind. Write the little we know by hand rather than nothing at all.
    printf '{"run_name": "%s", "status": "%s", "note": "%s", "written_at": "%s",\n "host": "%s", "mode": "%s", "arm": "%s", "gpus_claimed": "%s", "log": "%s",\n "checkpoints": [], "evaluations": [], "degraded": "python3 was not on PATH"}\n' \
      "$RUN_NAME" "$status" "${note//[\"\\]/}" "$(date -Is 2>/dev/null)" \
      "${HOSTNAME:-}" "$MODE" "$ARM" "${CUDA_VISIBLE_DEVICES:-}" "$LOG" >"$MANIFEST"
    return 0
  fi
  python3 - "$MANIFEST" "$status" "$note" <<'PY' || log "manifest: python3 failed on $MANIFEST"
import json, os, pathlib, sys, subprocess, datetime
manifest, status, note = sys.argv[1], sys.argv[2], sys.argv[3]
out = pathlib.Path(manifest).parent
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception: return ""
# OUTDIR, not its parent: the parent is WORKDIR, and walking that means walking the venv
# and the HF cache -- minutes of stat() calls inside a signal handler.
ckpts = sorted(str(p) for p in out.rglob("*globalstep*") if p.is_dir())
evals = []
for f in out.rglob("results.json"):
    try: evals.append({"file": str(f), "data": json.loads(f.read_text())})
    except Exception: evals.append({"file": str(f), "data": None})
json.dump({
    "run_name": os.environ.get("RUN_NAME", ""),
    "status": status,
    "note": note,
    "written_at": datetime.datetime.now().astimezone().isoformat(),
    "host": sh("hostname"),
    "mode": os.environ.get("MODE", ""),
    "arm": os.environ.get("ARM", ""),
    "solved_advantage": os.environ.get("SOLVED_ADV", ""),
    "model": os.environ.get("MODEL", ""),
    "gpus_claimed": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "commit": sh("git -C '%s' rev-parse HEAD" % os.environ.get("REPO_DIR", "")),
    "wandb": {"mode": os.environ.get("WANDB_MODE", ""),
              "degraded_from_online": os.environ.get("WANDB_DEGRADED", "0") == "1",
              "project": os.environ.get("WANDB_PROJECT", ""),
              "run": os.environ.get("RUN_NAME", "")},
    "checkpoints": ckpts,
    "evaluations": evals,
    "log": os.environ.get("PORTABLE_LOG", ""),
}, open(manifest, "w"), indent=2)
PY
}

STATUS_WRITTEN=0
MAIN_PID=$$

on_signal() {
  # A signal has to STOP us. The old trap wrote a manifest and returned, so `wait` came
  # back non-zero and the retry loop relaunched the training the operator had just asked
  # to stop -- on a box we do not own, that is the wrong way round.
  log "received a signal; stopping"
  declare -F cleanup_ours >/dev/null 2>&1 && cleanup_ours
  write_manifest "interrupted" "received a signal"
  STATUS_WRITTEN=1
  exit 130
}

on_exit() {
  # Every other way out -- an unset variable, SIGHUP when the ssh session drops, a return
  # we did not anticipate -- still leaves a manifest behind.
  local rc=$?
  [ "${BASHPID:-$$}" = "$MAIN_PID" ] || return 0
  # ALWAYS release the GPUs we claimed. Measured: without this, a run that exhausted its
  # retries left 113 processes and 4 GPUs holding ~131 GB each, indefinitely. On a shared box
  # that is worse than the failure itself -- the collaborator loses half their machine to a
  # job that already gave up, and nothing in the log says why the memory is gone.
  # Guarded on having claimed anything, so the lock-busy path (which owns nothing) is a no-op.
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    cleanup_ours 2>/dev/null
    local held
    held=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
           | awk -F", " -v t="$GPU_BUSY_ON_EXIT_MIB" '"'"'$2 > t {printf "%s ", $1}'"'"')
    [ -n "$held" ] && log "WARNING: GPUs still holding memory after cleanup: $held"
  fi
  [ "$STATUS_WRITTEN" -eq 1 ] || write_manifest "failed" "exited rc=$rc with no status"
}

trap on_signal INT TERM HUP
trap on_exit EXIT

# One instance per WORKDIR. Two copies racing on the same checkout is the fastest way to
# produce a corrupt run that looks like a code bug.
exec 9>"$WORKDIR/.lock"
if ! flock -n 9; then
  echo "another run_portable.sh holds $WORKDIR/.lock"
  # Report even this: no manifest at all is indistinguishable from a crash. Under a name
  # of its own, so the instance that does hold the lock never has its manifest clobbered.
  # shellcheck disable=SC2030,SC2031  # the subshell scoping is the point: $MANIFEST must
  # not be rewritten for anyone else.
  ( MANIFEST="${MANIFEST%.json}.lockbusy.json"
    write_manifest "failed" "another instance holds $WORKDIR/.lock" )
  STATUS_WRITTEN=1
  exit 3
fi

die() { log "FATAL: $1"; write_manifest "failed" "$1"; STATUS_WRITTEN=1; exit "${2:-1}"; }

# A configuration that can never be satisfied should be refused, not looped on.
[ "$MAX_GPUS" -ge "$MIN_GPUS" ] || die "MAX_GPUS=$MAX_GPUS is below MIN_GPUS=$MIN_GPUS" 4


wandb_usable() {
  # Online W&B without a key does not warn -- it raises inside PPOTrainer construction and
  # the whole run dies before step 1. A collaborator will not have our key, so an unattended
  # run must not depend on one.
  [ -n "${WANDB_API_KEY:-}" ] && return 0
  grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null && return 0
  grep -qE "^api_key" "${HOME}/.config/wandb/settings" 2>/dev/null && return 0
  return 1
}

# ------------------------------------------------------------ FREE GPUS ----
free_gpus() {
  # Only GPUs holding less than GPU_FREE_MIB. Someone else's 40 GB job makes its GPU theirs.
  # The split tolerates both "0, 20214" and "0,20214" (drivers differ on the space; with a
  # bare -F", " the no-space form left $2 empty, which read as free and printed the WHOLE
  # line as an index). Both fields must be integers, so a "[N/A]" row is skipped rather
  # than guessed at -- an unparseable row means we do not know, and we do not claim.
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F'[[:space:]]*,[[:space:]]*' -v t="$GPU_FREE_MIB" \
        '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $2+0 < t+0 {print $1}' \
    | head -n "$MAX_GPUS" | paste -sd,
}

claim_gpus() {
  local waited=0 got
  if [ -n "$GPUS" ]; then
    # Validate the pin rather than trusting it.
    local n_pin busy
    n_pin=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
    # Deliberately NOT awk, and not a $(...) one-liner. The previous version carried a
    # baked-in '"'"'-style escape that bash parsed as `syntax error near unexpected token (`
    # every time this ran. It failed SILENTLY on our own boxes -- the error went to stderr,
    # `busy` came back empty, `[ -n "$busy" ]` was false, and the script proceeded -- so the
    # GPU-pin safety check never actually validated anything. A collaborator's 8xH200 box
    # surfaced it as "run_portable.sh: line 226". Measured 2026-08-31.
    busy=""
    while IFS=, read -r _idx _used; do
      _idx=${_idx// /}; _used=${_used// /}
      case "$_idx" in ''|*[!0-9]*) continue ;; esac
      case "$_used" in ''|*[!0-9]*) continue ;; esac
      for _pin in ${GPUS//,/ }; do
        if [ "$_pin" = "$_idx" ] && [ "$_used" -ge "$GPU_FREE_MIB" ]; then
          busy="$busy$_idx(${_used}MiB) "
        fi
      done
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)
    if [ -n "$busy" ] && [ "$GPUS_FORCE" != "1" ]; then
      log "GPUS=$GPUS was pinned but these already hold memory: $busy"
      log "Someone else may be using them. Set GPUS_FORCE=1 to proceed anyway."
      return 1
    fi
    [ $((n_pin % 2)) -eq 1 ] && log "WARNING: GPUS=$GPUS is an odd count; AReaL splits train/rollout"
    echo "$GPUS"; return 0
  fi
  while :; do
    got="$(free_gpus)"
    local n=0
    [ -n "$got" ] && n=$(echo "$got" | tr ',' '\n' | grep -c .)
    # AReaL splits GPUs between training and rollout, so an odd count wastes one. Round
    # BEFORE the MIN_GPUS test: rounding after it returns MIN_GPUS-1 GPUs whenever
    # MIN_GPUS is odd, and at n=1 `cut -f1-0` fails and returns an EMPTY list as a
    # success -- which then reads as "claim nothing, and match every process on the box".
    if [ $((n % 2)) -eq 1 ]; then
      n=$((n - 1))
      if [ "$n" -gt 0 ]; then got="$(echo "$got" | cut -d, -f"1-$n")"; else got=""; fi
    fi
    if [ "$n" -gt 0 ] && [ "$n" -ge "$MIN_GPUS" ]; then
      echo "$got"; return 0
    fi
    [ "$waited" -ge "$WAIT_FOR_GPUS_S" ] && return 1
    log "only $n free GPU(s), need $MIN_GPUS; waiting (${waited}/${WAIT_FOR_GPUS_S}s)"
    sleep 60; waited=$((waited + 60))
  done
}

# --------------------------------------------------------------- SETUP ----
setup() {
  export REPO_DIR="$WORKDIR/areal-selfevo"
  if [ ! -d "$REPO_DIR/.git" ]; then
    log "cloning $REPO_URL ($REPO_BRANCH)"
    git clone --depth 50 --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR" >>"$LOG" 2>&1 \
      || return 1
  else
    log "updating existing checkout"
    git -C "$REPO_DIR" fetch --depth 50 origin "$REPO_BRANCH" >>"$LOG" 2>&1 \
      && git -C "$REPO_DIR" reset --hard "origin/$REPO_BRANCH" >>"$LOG" 2>&1 || return 1
  fi

  # Reuse a usable interpreter rather than building one: on a shared box the environment is
  # often already correct, and a fresh venv is minutes of wheel downloads.
  for cand in "$WORKDIR/venv/bin/activate" "$HOME/venv312b/bin/activate" "/venv/main/bin/activate"; do
    if [ -f "$cand" ]; then
      # shellcheck disable=SC1090
      source "$cand"
      python3 -c "import torch, aiohttp" 2>/dev/null && { log "using env $cand"; break; }
    fi
  done
  if ! python3 -c "import torch, aiohttp" 2>/dev/null; then
    log "no usable environment found; building one at $WORKDIR/venv"
    python3 -m venv "$WORKDIR/venv" >>"$LOG" 2>&1 || return 1
    # shellcheck disable=SC1091
    source "$WORKDIR/venv/bin/activate"
    pip install -q -U pip >>"$LOG" 2>&1
    pip install -q -e "$REPO_DIR" >>"$LOG" 2>&1 || return 1
  fi
  # Never pip-install into an interpreter we did not create: the environment we just
  # reused may be the one a neighbour's job is importing from at this moment, and a
  # resolver that upgrades a shared dependency under a live run is a way to break it.
  if python3 -c "import wandb" 2>/dev/null; then :
  elif [ "${VIRTUAL_ENV:-}" = "$WORKDIR/venv" ]; then
    pip install -q wandb >>"$LOG" 2>&1 || true
  else
    log "wandb missing from the reused environment; not installing into someone else's env"
  fi

  # Data and weights, fetched once and cached. HF_HOME is set explicitly because a shared
  # box often has a stale one pointing at a filesystem this user cannot write.
  export HF_HOME="${HF_HOME_OVERRIDE:-$WORKDIR/hf}"
  mkdir -p "$HF_HOME"
  log "prefetching model and data into $HF_HOME"
  python3 - <<PY >>"$LOG" 2>&1 || return 1
import os
from huggingface_hub import snapshot_download
from datasets import load_dataset
snapshot_download("${MODEL}", allow_patterns=["*.json","*.safetensors","*.txt","*.model"])
load_dataset("openai/gsm8k", "main")
print("prefetch ok")
PY
  return 0
}

# ----------------------------------------------------------------- RUN ----
newest_checkpoint() {
  # AReaL nests these as checkpoints/<user>/<experiment>/<trial>/default/<step>, so the
  # one-level glob this used to be matched nothing at all and MODE=eval could never default
  # to the checkpoint we had just trained -- it failed, and then retried the failure.
  find "$OUTDIR/checkpoints" -maxdepth 6 -type d -name '*globalstep*' \
    -printf '%T@\t%p\n' 2>/dev/null | sort -rn | head -1 | cut -f2-
}

run_once() {
  local gpus="$1" n="$2"
  cd "$REPO_DIR" || return 1
  export WANDB_MODE WANDB_PROJECT
  export CUDA_VISIBLE_DEVICES="$gpus"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
  ulimit -n 131072 2>/dev/null || true

  if [ "$MODE" = "eval" ]; then
    local target="${CKPT:-$(newest_checkpoint)}"
    [ -n "$target" ] || { log "MODE=eval needs CKPT= or a checkpoint under $OUTDIR"; return 2; }
    log "evaluating $target on $BENCHES using GPUs $gpus"
    CKPT_ROOT="$(dirname "$target")" BENCHES="$BENCHES" \
      bash experiments/bench/sweep_entropy.sh >>"$LOG" 2>&1
    return $?
  fi

  local routing=()
  case "$ARM" in
    off)  routing=() ;;
    on)   routing=("+actor.group_routing.enabled=true"
                   "+actor.group_routing.solved_advantage=${SOLVED_ADV}") ;;
    dapo) routing=("+dynamic_filter_fn=selfevo.baselines.dapo.dapo_dynamic_sampling") ;;
    *)    log "ARM must be off|on|dapo, got '$ARM'"; return 2 ;;
  esac

  local key_file="$WORKDIR/.areal_admin_key"
  [ -f "$key_file" ] || (umask 077; head -c 24 /dev/urandom | base64 | tr -d "/+=" > "$key_file")

  log "training arm=$ARM on $n GPU(s) [$gpus], epochs=$EPOCHS, wandb=$WANDB_MODE"
  python3 examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local \
    cluster.fileroot="$OUTDIR" \
    cluster.n_gpus_per_node="$n" \
    sglang.mem_fraction_static="$MEM_FRACTION" \
    actor.path="$MODEL" \
    gconfig.n_samples="$N_SAMPLES" \
    train_dataset.path="$DATASET" valid_dataset.path="$DATASET" \
    actor.optimizer.lr=1.0e-6 actor.eps_clip=0.2 actor.kl_ctl=0.0 \
    ~actor.adv_norm ++actor.mb_spec.granularity=8 \
    saver.freq_steps=25 total_train_epochs="$EPOCHS" \
    evaluator.freq_epochs=null evaluator.freq_secs=null \
    +actor.attn_impl=sdpa +ref.attn_impl=sdpa \
    stats_logger.wandb.mode="$WANDB_MODE" \
    +stats_logger.wandb.project="$WANDB_PROJECT" \
    +stats_logger.wandb.name="$RUN_NAME" \
    ${routing[@]+"${routing[@]}"} \
    +rollout.agent.admin_api_key="$(cat "$key_file")" \
    ${PORTABLE_EXTRA} \
    experiment_name="$RUN_NAME" trial_name=t1 >>"$LOG" 2>&1
}

# Kill only what we started. Patterns cover both spellings AReaL uses and the sglang servers,
# which carry no experiment name and otherwise survive to hold their GPUs into the next try.
ours_gpu() {
  # True when /proc/$1 carries a NON-EMPTY CUDA_VISIBLE_DEVICES whose every device is one
  # we claimed. Substring matching was wrong in both directions: our "0,1" matched a
  # neighbour's "0,1,2,3" and killed them, an empty claim matched EVERY server on the box,
  # our "3,2,1,0" matched nothing, and our full list never matched our own workers --
  # AReaL hands each sglang server a single physical id, so the full list never appears.
  local pid="$1" theirs d
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || return 1
  theirs="$(tr '\0' '\n' <"/proc/$pid/environ" 2>/dev/null \
            | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1)"
  [ -n "$theirs" ] || return 1
  for d in ${theirs//,/ }; do
    case ",${CUDA_VISIBLE_DEVICES}," in *",$d,"*) ;; *) return 1 ;; esac
  done
  return 0
}

cleanup_ours() {
  # With nothing claimed there is nothing of ours to kill, and every test below would
  # match a neighbour instead. Refusing is the only safe reading of an empty claim.
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { log "cleanup skipped: no GPUs claimed"; return 0; }
  pkill -u "$USER" -f "experiment_name=${RUN_NAME}" 2>/dev/null
  pkill -u "$USER" -f -- "-experiment-name ${RUN_NAME}" 2>/dev/null
  sleep 8
  pkill -9 -u "$USER" -f "experiment_name=${RUN_NAME}" 2>/dev/null
  pkill -9 -u "$USER" -f -- "-experiment-name ${RUN_NAME}" 2>/dev/null
  # Only sglang servers whose GPUs are in our claimed set; a neighbour's must survive.
  # A second test, on age: on a shared box the neighbour often runs as the SAME user, so
  # -u $USER protects nobody. A server older than this harness cannot be one of ours.
  local pid age ours_age
  ours_age=$(( $(date +%s) - HARNESS_START ))
  for pid in $(pgrep -u "$USER" -f "inference_service.sglang.launch_server" 2>/dev/null); do
    ours_gpu "$pid" || continue
    age="$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')"
    [ -n "$age" ] && [ "$age" -gt "$ours_age" ] && continue
    kill -9 "$pid" 2>/dev/null
  done
  sleep 5
}

main() {
  log "run_portable: mode=$MODE arm=$ARM name=$RUN_NAME workdir=$WORKDIR"
  if [ "$WANDB_MODE" = "online" ] && ! wandb_usable; then
    # Degrade rather than die, and say so: an unattended run that fails because
    # LOGGING is unconfigured has wasted the machine for no scientific reason. The
    # offline run can be synced later with `wandb sync`, and the manifest records it.
    log "WARNING: WANDB_MODE=online but no API key found (WANDB_API_KEY, ~/.netrc,"
    log "         or ~/.config/wandb/settings). Falling back to offline; sync later"
    log "         with: wandb sync ${OUTDIR}/wandb/offline-*"
    WANDB_MODE=offline
    WANDB_DEGRADED=1
  fi
  export WANDB_MODE WANDB_DEGRADED="${WANDB_DEGRADED:-0}"
  setup || die "setup failed" 5

  local gpus n
  gpus="$(claim_gpus)" || die "fewer than MIN_GPUS=$MIN_GPUS free (raise WAIT_FOR_GPUS_S to wait)" 4
  n=$(echo "$gpus" | tr ',' '\n' | grep -c .)
  export CUDA_VISIBLE_DEVICES="$gpus"
  log "claimed GPUs [$gpus] ($n)"
  write_manifest "running" ""

  local attempt rc
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    log "attempt $attempt/$MAX_ATTEMPTS"
    ( run_once "$gpus" "$n" ) &
    local pid=$!

    # Two fuses: no first step within STARTUP_S, or the log stops growing for STALL_S.
    ( local started; started=$(date +%s)
      while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        local now age mt; now=$(date +%s)
        mt=$(stat -c %Y "$LOG" 2>/dev/null || echo "$now"); age=$((now - mt))
        if ! grep -qE "step [0-9]+/" "$LOG" 2>/dev/null; then
          [ $((now - started)) -gt "$STARTUP_S" ] && { log "no step within ${STARTUP_S}s; killing"; cleanup_ours; break; }
        elif [ "$age" -gt "$STALL_S" ]; then
          log "log stale ${age}s; killing"; cleanup_ours; break
        fi
      done ) &
    local wd=$!
    wait "$pid"; rc=$?
    kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null

    if [ "$rc" -eq 0 ]; then
      log "attempt $attempt succeeded"
      write_manifest "succeeded" ""
      STATUS_WRITTEN=1
      # shellcheck disable=SC2031  # the lock-busy rename above was scoped to its subshell
      log "manifest: $MANIFEST"
      return 0
    fi
    log "attempt $attempt failed rc=$rc"
    cleanup_ours
    # A neighbour may have taken GPUs while we were down; re-claim rather than assume.
    gpus="$(claim_gpus)" || die "GPUs no longer available after a failure" 4
    n=$(echo "$gpus" | tr ',' '\n' | grep -c .)
    export CUDA_VISIBLE_DEVICES="$gpus"
    sleep $((attempt * 30))
  done
  die "all $MAX_ATTEMPTS attempts failed" 6
}

# Sourcing exposes the functions without running anything, which is how
# selfevo/tests/test_run_portable.py exercises them against a stubbed nvidia-smi.
[ "${RUN_PORTABLE_SOURCE_ONLY:-0}" = "1" ] || main "$@"
