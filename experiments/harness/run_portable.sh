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

# Shared-box etiquette.
MIN_GPUS="${MIN_GPUS:-4}"             # refuse rather than thrash below this
MAX_GPUS="${MAX_GPUS:-8}"             # never claim more than this even if free
GPU_FREE_MIB="${GPU_FREE_MIB:-4096}"  # a GPU holding more than this belongs to someone else
WAIT_FOR_GPUS_S="${WAIT_FOR_GPUS_S:-0}"   # >0 waits for a neighbour's job to finish

# Resilience.
MAX_ATTEMPTS="${MAX_ATTEMPTS:-4}"
STALL_S="${STALL_S:-1800}"
STARTUP_S="${STARTUP_S:-1200}"

# Reporting.
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-selfevo-routing}"
RUN_NAME="${RUN_NAME:-${MODE}-${ARM}-$(hostname -s)-$(date +%m%d_%H%M)}"

mkdir -p "$WORKDIR" "$OUTDIR"
LOG="$OUTDIR/${RUN_NAME}.log"
MANIFEST="$OUTDIR/${RUN_NAME}.manifest.json"

log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
die() { log "FATAL: $*"; write_manifest "failed" "$*"; exit "${2:-1}"; }

# One instance per WORKDIR. Two copies racing on the same checkout is the fastest way to
# produce a corrupt run that looks like a code bug.
exec 9>"$WORKDIR/.lock"
flock -n 9 || { echo "another run_portable.sh holds $WORKDIR/.lock"; exit 3; }

write_manifest() {
  # A single structured file is what gets handed back, so it is written on EVERY exit path,
  # including failure -- a manifest that only exists on success cannot report a failure.
  local status="$1" note="${2:-}"
  python3 - "$MANIFEST" "$status" "$note" <<'PY' 2>/dev/null || true
import json, os, pathlib, sys, subprocess, datetime
manifest, status, note = sys.argv[1], sys.argv[2], sys.argv[3]
out = pathlib.Path(manifest).parent
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception: return ""
ckpts = sorted(str(p) for p in out.parent.rglob("*globalstep*") if p.is_dir())
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
              "project": os.environ.get("WANDB_PROJECT", ""),
              "run": os.environ.get("RUN_NAME", "")},
    "checkpoints": ckpts,
    "evaluations": evals,
    "log": os.environ.get("PORTABLE_LOG", ""),
}, open(manifest, "w"), indent=2)
PY
}
export PORTABLE_LOG="$LOG"
trap 'write_manifest "interrupted" "received a signal"' INT TERM

# ------------------------------------------------------------ FREE GPUS ----
free_gpus() {
  # Only GPUs holding less than GPU_FREE_MIB. Someone else's 40 GB job makes its GPU theirs.
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F", " -v t="$GPU_FREE_MIB" '$2 < t {print $1}' | head -n "$MAX_GPUS" | paste -sd,
}

claim_gpus() {
  local waited=0 got
  while :; do
    got="$(free_gpus)"
    local n=0
    [ -n "$got" ] && n=$(echo "$got" | tr ',' '\n' | grep -c .)
    # AReaL splits GPUs between training and rollout, so an odd count wastes one.
    if [ "$n" -ge "$MIN_GPUS" ]; then
      [ $((n % 2)) -eq 1 ] && got="$(echo "$got" | cut -d, -f1-$((n - 1)))" && n=$((n - 1))
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
  python3 -c "import wandb" 2>/dev/null || pip install -q wandb >>"$LOG" 2>&1 || true

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
run_once() {
  local gpus="$1" n="$2"
  cd "$REPO_DIR" || return 1
  export WANDB_MODE WANDB_PROJECT
  export CUDA_VISIBLE_DEVICES="$gpus"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
  ulimit -n 131072 2>/dev/null || true

  if [ "$MODE" = "eval" ]; then
    local target="${CKPT:-$(ls -dt "$OUTDIR"/checkpoints/*globalstep* 2>/dev/null | head -1)}"
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
    actor.path="$MODEL" \
    gconfig.n_samples=8 \
    actor.optimizer.lr=1.0e-6 actor.eps_clip=0.2 actor.kl_ctl=0.0 \
    ~actor.adv_norm ++actor.mb_spec.granularity=8 \
    saver.freq_steps=25 total_train_epochs="$EPOCHS" \
    evaluator.freq_epochs=null evaluator.freq_secs=null \
    +actor.attn_impl=sdpa +ref.attn_impl=sdpa \
    stats_logger.wandb.mode="$WANDB_MODE" \
    +stats_logger.wandb.project="$WANDB_PROJECT" \
    +stats_logger.wandb.name="$RUN_NAME" \
    "${routing[@]}" \
    +rollout.agent.admin_api_key="$(cat "$key_file")" \
    experiment_name="$RUN_NAME" trial_name=t1 >>"$LOG" 2>&1
}

# Kill only what we started. Patterns cover both spellings AReaL uses and the sglang servers,
# which carry no experiment name and otherwise survive to hold their GPUs into the next try.
cleanup_ours() {
  pkill -u "$USER" -f "experiment_name=${RUN_NAME}" 2>/dev/null
  pkill -u "$USER" -f -- "-experiment-name ${RUN_NAME}" 2>/dev/null
  sleep 8
  pkill -9 -u "$USER" -f "experiment_name=${RUN_NAME}" 2>/dev/null
  pkill -9 -u "$USER" -f -- "-experiment-name ${RUN_NAME}" 2>/dev/null
  # Only sglang servers whose GPUs are in our claimed set; a neighbour's must survive.
  for pid in $(pgrep -u "$USER" -f "inference_service.sglang.launch_server" 2>/dev/null); do
    grep -q "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" "/proc/$pid/environ" 2>/dev/null \
      && kill -9 "$pid" 2>/dev/null
  done
  sleep 5
}

main() {
  log "run_portable: mode=$MODE arm=$ARM name=$RUN_NAME workdir=$WORKDIR"
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

main "$@"
