#!/usr/bin/env bash
# Terminal-Bench 2.1 harness swap at FIXED model.
#
#   bash run_tb_swap.sh --smoke     # verify every prerequisite, change nothing, ~2 min
#   bash run_tb_swap.sh --fetch     # download the Terminal-Bench 2.1 task set, resumable
#   bash run_tb_swap.sh --run       # run one arm (ARM=A|B)
#
# GPUs: set GPUS=0,1,2,3. NOTE what these are for. Terminal-Bench talks to a REMOTE model over
# ANTHROPIC_BASE_URL, so the harness itself uses no GPU. GPUS only matters if this script also
# serves your model (SERVE=1), in which case it pins sglang to those devices.
#
# W&B: harbor does NOT support Weights & Biases -- verified, `wandb` is not a harbor dependency
# and appears nowhere in the TB harness. So this script logs the FINAL RESULT itself when
# WANDB_API_KEY is set. Do not expect per-task streaming; there is no hook for it.
#
# Anthropic-compat caveat, stated because it will bite: the shipped config drives Claude Code
# against an ANTHROPIC-compatible endpoint. sglang serves an OpenAI-compatible API. Serving
# your own model therefore needs a translating proxy (e.g. LiteLLM in Anthropic mode) between
# them. SERVE=1 starts sglang and tells you this; it does not magically make it Anthropic-shaped.
#
# The smoke path is the point: it checks each prerequisite INDEPENDENTLY and reports all
# failures at once, rather than dying at the first one and hiding the rest. Every check here
# corresponds to something that actually bit us -- see the messages.
set -u -o pipefail

MODE="${1:-}"
LHH_DIR="${LHH_DIR:-$HOME/LongHorizon-Harness}"
TB_DIR="$LHH_DIR/eval/TB-harness"
VENV="${TB_VENV:-$HOME/tb-env}"
PY="${TB_PYTHON:-python3.12}"
ARM="${ARM:-}"
FAILED=0
ok(){   printf '  \033[32mok\033[0m    %s\n' "$1"; }
bad(){  printf '  \033[31mFAIL\033[0m  %s\n' "$1"; printf '        -> %s\n' "$2"; FAILED=$((FAILED+1)); }
warn(){ printf '  \033[33mwarn\033[0m  %s\n' "$1"; }

echo "== Terminal-Bench 2.1 harness-swap preflight =="

# 1. Docker. Terminal-Bench executes every task in a container; without a usable daemon
#    nothing runs. Our A100 fails here with a docker.sock permission error, which is why
#    this script exists rather than a README instruction.
if ! command -v docker >/dev/null 2>&1; then
  bad "docker not installed" "Terminal-Bench runs tasks in containers. Install Docker."
elif ! docker info >/dev/null 2>&1; then
  bad "docker present but daemon unusable" \
      "Typically the user is not in the docker group: sudo usermod -aG docker \$USER, then re-login. Verify with: docker info"
else
  ok "docker daemon usable"
fi

# 2. Python 3.12 + harbor. conda is NOT required despite the upstream README; harbor 0.18.0
#    installs cleanly into a plain venv (verified 2026-08-31).
# python3.12 is needed only to CREATE the venv. An existing working harbor makes it
# irrelevant -- an earlier version of this check failed the whole preflight on a box where
# harbor was already installed and fine, which is the kind of false blocker that makes people
# ignore preflights.
if [ -x "$VENV/bin/harbor" ] && "$VENV/bin/harbor" --help >/dev/null 2>&1; then
  ok "harbor already installed and runnable ($("$VENV/bin/python" -m pip show harbor 2>/dev/null | awk '/^Version/{print $2}')) at $VENV"
elif ! command -v "$PY" >/dev/null 2>&1; then
  bad "no working harbor, and $PY not found to build one" \
      "Either point TB_VENV at a venv that has harbor, or set TB_PYTHON to a python3.12 binary."
else
  ok "$PY present"
  echo "        creating venv at $VENV and installing harbor==0.18.0 ..."
  "$PY" -m venv "$VENV" >/dev/null 2>&1
  "$VENV/bin/python" -m pip install -q --no-cache-dir harbor==0.18.0 >/dev/null 2>&1
  if "$VENV/bin/harbor" --help >/dev/null 2>&1; then
    ok "harbor installed and runnable ($("$VENV/bin/python" -m pip show harbor 2>/dev/null | awk '/^Version/{print $2}'))"
  else
    bad "harbor not runnable" "pip install harbor==0.18.0 into $VENV failed; run it manually to see the error."
  fi
fi

# 3. The harness checkout.
if [ -d "$TB_DIR" ]; then
  ok "TB-harness found at $TB_DIR"
  [ -d "$TB_DIR/Harness/src" ] && ok "Harness/src present (PYTHONPATH target)" \
    || bad "Harness/src missing" "Checkout looks incomplete; re-clone AMAP-ML/LongHorizon-Harness."
else
  bad "TB-harness not found at $TB_DIR" \
      "git clone https://github.com/AMAP-ML/LongHorizon-Harness.git \"$LHH_DIR\"  (or set LHH_DIR)"
fi

# 4. Credentials. The shipped script exits 2 without BOTH of these: it drives Claude Code as
#    the agent harness against an Anthropic-compatible endpoint.
[ -n "${ANTHROPIC_API_KEY:-}" ] && ok "ANTHROPIC_API_KEY set" \
  || bad "ANTHROPIC_API_KEY unset" "The shipped LHH config drives Claude Code; it exits 2 without this."
[ -n "${ANTHROPIC_BASE_URL:-}" ] && ok "ANTHROPIC_BASE_URL set ($ANTHROPIC_BASE_URL)" \
  || bad "ANTHROPIC_BASE_URL unset" "Point this at the endpoint serving YOUR model, so the swap holds the model fixed."

# 5. Tasks. Not shipped with the repo.
TASKS="$TB_DIR/datasets/terminal-bench-2-1/tasks"
if [ -d "$TASKS" ] || [ -L "$TASKS" ]; then
  n=$(find -L "$TASKS" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && ok "task set present ($n tasks)" \
    || bad "task dir exists but is empty" "Populate $TASKS (a symlink is fine)."
else
  bad "task set missing" "Put Terminal-Bench 2.1 tasks at $TASKS. Try: $VENV/bin/harbor download --help"
fi

# 6. Both arms must be defined. Upstream ships ONLY the LHH arm; the baseline is the work.
LHH_CFG="$TB_DIR/Scripts/tbench21_full_cua_harness_claudecode_qwen37_enable_thinking.yaml"
[ -f "$LHH_CFG" ] && ok "arm B (LHH MEA) config shipped" || bad "arm B config missing" "Expected $LHH_CFG"
if [ -n "${BASELINE_CFG:-}" ] && [ -f "$BASELINE_CFG" ]; then
  ok "arm A (baseline) config: $BASELINE_CFG"
else
  warn "arm A (baseline) NOT defined. Upstream ships no baseline; set BASELINE_CFG=<yaml>."
  warn "A swap without its baseline measures nothing - this is the real work of the task."
fi

echo
if [ "$FAILED" -gt 0 ]; then
  echo "PREFLIGHT: $FAILED blocking problem(s). Tell us rather than working around them -"
  echo "the workaround is usually more interesting to us than the result."
  [ "$MODE" = "--smoke" ] && exit 1
  exit 1
fi
echo "PREFLIGHT: all checks passed."
[ "$MODE" = "--smoke" ] && { echo "(smoke mode: nothing was run)"; exit 0; }

# ---- fetch mode: download the task set, resumably ----
if [ "$MODE" = "--fetch" ]; then
  DEST="${TB_TASKS:-$HOME/tb_tasks}"
  LINK="$TB_DIR/datasets/terminal-bench-2-1/tasks"
  mkdir -p "$DEST" "$TB_DIR/datasets/terminal-bench-2-1"
  # Retry rather than fail: this is a large multi-task download over a public registry and a
  # single transient error should not cost the whole fetch. harbor skips what it already has,
  # so retrying is cheap and resumable.
  for attempt in 1 2 3 4 5; do
    echo "fetch attempt $attempt ..."
    if "$VENV/bin/harbor" download "terminal-bench@2.1" -o "$DEST"; then
      break
    fi
    echo "  attempt $attempt failed; retrying in $((attempt*20))s"
    sleep $((attempt*20))
  done
  n=$(find -L "$DEST" -maxdepth 2 -mindepth 1 -type d 2>/dev/null | wc -l)
  if [ "$n" -eq 0 ]; then
    echo "FETCH FAILED: no tasks under $DEST after 5 attempts."
    echo "  The registry name may have changed. Try: $VENV/bin/harbor download --help"
    echo "  and list what is available, then set TB_TASKS to a manually obtained task tree."
    exit 4
  fi
  [ -e "$LINK" ] || ln -s "$DEST" "$LINK"
  echo "fetched $n task dirs -> $LINK"
  exit 0
fi

# ---- optional: serve your own model on the pinned GPUs ----
if [ "${SERVE:-0}" = "1" ]; then
  : "${MODEL_PATH:?set MODEL_PATH to serve a model}"
  : "${GPUS:?set GPUS=0,1,2,3 to pin the server}"
  NTP=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
  echo "serving $MODEL_PATH on GPUs $GPUS (tp=$NTP), port ${SERVE_PORT:-8700}"
  CUDA_VISIBLE_DEVICES="$GPUS" "$VENV/bin/python" -m sglang.launch_server \
    --model-path "$MODEL_PATH" --served-model-name tbmodel \
    --host 127.0.0.1 --port "${SERVE_PORT:-8700}" --tp "$NTP" \
    --mem-fraction-static "${MEMFRAC:-0.8}" > "$TB_DIR/serve.log" 2>&1 &
  SRV=$!
  trap '[ -n "${SRV:-}" ] && kill -TERM "$SRV" 2>/dev/null' EXIT INT TERM
  for i in $(seq 1 180); do
    kill -0 "$SRV" 2>/dev/null || { echo "SERVER DIED - see $TB_DIR/serve.log"; tail -20 "$TB_DIR/serve.log"; exit 5; }
    curl -sf "http://127.0.0.1:${SERVE_PORT:-8700}/v1/models" >/dev/null 2>&1 && break
    sleep 5
  done
  echo "server up. NOTE: this is an OPENAI-compatible endpoint. The TB config wants an"
  echo "ANTHROPIC-compatible one, so point ANTHROPIC_BASE_URL at a translating proxy, not"
  echo "directly at this port, or the run will fail inside the container with auth/schema errors."
fi

# ---- actual run ----
[ "$MODE" = "--run" ] || { echo "Usage: $0 --smoke | --fetch | --run   (ARM=A|B, BASELINE_CFG=<yaml>)"; exit 2; }
case "$ARM" in
  B) CFG="$LHH_CFG" ;;
  A) CFG="${BASELINE_CFG:?set BASELINE_CFG for arm A}" ;;
  *) echo "set ARM=A or ARM=B"; exit 2 ;;
esac
STAMP=$(date +%Y%m%d-%H%M%S)
OUT="$TB_DIR/logs/arm${ARM}_$STAMP"
mkdir -p "$OUT"
echo "arm=$ARM config=$CFG out=$OUT"
export PYTHONPATH="$TB_DIR/Harness/src${PYTHONPATH:+:$PYTHONPATH}"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
cd "$TB_DIR" || exit 1
# n_attempts 3: the published Terminal-Bench 2.1 protocol is pass@1 averaged over 3 repeats,
# and our own measurements show a single score carries ~1 point of jitter. One repeat cannot
# resolve a harness delta.
# Retry a whole-run failure ONCE. Deliberately once, not many: a harness run is hours long,
# and silently retrying a systematically broken config burns a night. A single retry covers a
# transient container/registry hiccup; a second identical failure is information, not noise.
RC=1
for attempt in 1 2; do
  "$VENV/bin/harbor" run -c "$CFG" 2>&1 | tee -a "$OUT/run.log"
  RC=${PIPESTATUS[0]}
  [ "$RC" -eq 0 ] && break
  echo "attempt $attempt exited $RC" | tee -a "$OUT/run.log"
  [ "$attempt" -eq 1 ] && { echo "retrying once in 60s"; sleep 60; }
done
echo "EXIT=$RC" | tee -a "$OUT/run.log"

# W&B: harbor has no W&B integration (verified), so log the outcome ourselves. Optional and
# non-fatal -- a logging failure must never fail an expensive completed run.
if [ -n "${WANDB_API_KEY:-}" ]; then
  "$VENV/bin/python" -m pip install -q wandb >/dev/null 2>&1 || true
  WANDB_ARM="$ARM" WANDB_RC="$RC" WANDB_OUT="$OUT" WANDB_CFG="$CFG" \
  "$VENV/bin/python" - <<'PY' 2>&1 | tail -3 || echo "W&B logging failed (run itself unaffected)"
import os, glob, json
import wandb
run = wandb.init(project=os.environ.get("WANDB_PROJECT", "tb21-harness-swap"),
                 name=f"arm{os.environ['WANDB_ARM']}-{os.path.basename(os.environ['WANDB_OUT'])}",
                 config={"arm": os.environ["WANDB_ARM"], "config_path": os.environ["WANDB_CFG"]})
run.summary["exit_code"] = int(os.environ["WANDB_RC"])
# Attach whatever harbor wrote, without assuming a schema we have not verified.
for f in glob.glob(os.path.join(os.environ["WANDB_OUT"], "**", "*.json"), recursive=True)[:20]:
    try:
        run.save(f)
    except Exception:
        pass
run.finish()
print("W&B logged")
PY
else
  echo "WANDB_API_KEY unset - skipping W&B (optional)."
fi

echo "Send back: $OUT/run.log AND the per-task results (not just the aggregate)."
