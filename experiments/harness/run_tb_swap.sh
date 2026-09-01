#!/usr/bin/env bash
# Terminal-Bench 2.1 harness swap at FIXED model.
#
#   bash run_tb_swap.sh --install   # set up a BARE box: python3.12, venv, harbor, clone LHH
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
HARBOR_VER="${HARBOR_VER:-0.18.0}"
# The command that runs harbor on THIS box. Set by resolve_harbor/install_harbor, because a
# venv at $VENV is only one of the routes that can supply it.
HARBOR=""
ARM="${ARM:-}"
# Default to EVERY visible GPU. Only used under SERVE=1 (the harness itself talks to a remote
# endpoint and uses no GPU), so defaulting wide is safe and matches an 8-GPU box out of the box.
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)}"
GPUS="${GPUS:-0}"
FAILED=0
ok(){   printf '  \033[32mok\033[0m    %s\n' "$1"; }
bad(){  printf '  \033[31mFAIL\033[0m  %s\n' "$1"; printf '        -> %s\n' "$2"; FAILED=$((FAILED+1)); }
warn(){ printf '  \033[33mwarn\033[0m  %s\n' "$1"; }


# Install harbor into $VENV and, on failure, SHOW WHY. An earlier version piped this to
# /dev/null and told the user to "run it manually to see the error" -- which is the script
# withholding the one thing that would let them fix it. A collaborator hit exactly that.
resolve_harbor() {
  # Locate an already-working harbor without installing anything. Sets HARBOR and returns 0,
  # or returns 1 if none is runnable. Checked in the same order install_harbor creates them.
  HARBOR=""
  if [ -x "$VENV/bin/harbor" ] && "$VENV/bin/harbor" --help >/dev/null 2>&1; then
    HARBOR="$VENV/bin/harbor"; return 0
  fi
  if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import harbor" >/dev/null 2>&1; then
    HARBOR="$VENV/bin/python -m harbor"; return 0
  fi
  if command -v harbor >/dev/null 2>&1 && harbor --help >/dev/null 2>&1; then
    HARBOR="$(command -v harbor)"; return 0
  fi
  local cur; cur="$(command -v python3 || command -v python)"
  if [ -n "$cur" ] && "$cur" -c "import harbor" >/dev/null 2>&1; then
    HARBOR="$cur -m harbor"; return 0
  fi
  return 1
}

install_harbor() {
  # Get a runnable `harbor` by whatever route this box actually supports, in order of
  # cleanliness. A stock python3-venv is preferred, but plenty of real boxes lack it:
  # Debian/Ubuntu split ensurepip into a separate python3.N-venv package, and conda
  # pythons routinely build venvs with no pip. Falling back is the difference between a
  # collaborator running the experiment and a collaborator filing a bug.
  local log="${TMPDIR:-/tmp}/harbor_install.$$.log"
  : > "$log"
  HARBOR=""

  _try_pip_into() {   # $1 = python to install with, $2 = label
    echo "  trying: $2" | tee -a "$log"
    if "$1" -m pip install --disable-pip-version-check -q "harbor==${HARBOR_VER}" >> "$log" 2>&1; then
      local hb; hb="$(dirname "$1")/harbor"
      if [ -x "$hb" ] && "$hb" --help >/dev/null 2>&1; then HARBOR="$hb"; return 0; fi
      if "$1" -c "import harbor" >/dev/null 2>&1; then HARBOR="$1 -m harbor"; return 0; fi
    fi
    return 1
  }

  # 1. stock venv, with ensurepip in case the venv came up without pip
  if "$PY" -m venv "$VENV" >> "$log" 2>&1; then
    "$VENV/bin/python" -m ensurepip --upgrade >> "$log" 2>&1 || true
    _try_pip_into "$VENV/bin/python" "venv at $VENV" && { echo "  harbor: $HARBOR"; return 0; }
  else
    echo "  note: '$PY -m venv' failed (this box likely lacks python3-venv/ensurepip)" | tee -a "$log"
  fi

  # 2. virtualenv, which vendors its own pip and needs no ensurepip
  if command -v virtualenv >/dev/null 2>&1; then
    rm -rf "$VENV"
    if virtualenv -q -p "$PY" "$VENV" >> "$log" 2>&1; then
      _try_pip_into "$VENV/bin/python" "virtualenv at $VENV" && { echo "  harbor: $HARBOR"; return 0; }
    fi
  fi

  # 3. uv, a single static binary that brings its own python
  if command -v uv >/dev/null 2>&1; then
    rm -rf "$VENV"
    if uv venv "$VENV" >> "$log" 2>&1; then
      _try_pip_into "$VENV/bin/python" "uv venv at $VENV" && { echo "  harbor: $HARBOR"; return 0; }
    fi
  fi

  # 4. conda, which can materialise a 3.12 without apt or root. On a box that is already
  #    sitting in a conda base -- common for shared notebook images -- this is usually the
  #    only route that works, because harbor requires Python >=3.12 and the base env is older.
  if command -v conda >/dev/null 2>&1; then
    rm -rf "$VENV"
    echo "  trying: conda create -p $VENV python=3.12" | tee -a "$log"
    if conda create -y -q -p "$VENV" python=3.12 >> "$log" 2>&1; then
      _try_pip_into "$VENV/bin/python" "conda env at $VENV" && { echo "  harbor: $HARBOR"; return 0; }
    fi
  fi

  # 4. the interpreter we are already standing in. On a conda base this is usually the one
  #    that works, and an isolated venv was only ever a nicety.
  local cur; cur="$(command -v python3 || command -v python)"
  if [ -n "$cur" ] && ! "$cur" -c 'import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)'; then
    echo "  skipping direct install: $cur is $("$cur" -V 2>&1), and harbor requires >=3.12" | tee -a "$log"
    cur=""
  fi
  if [ -n "$cur" ]; then
    _try_pip_into "$cur" "direct install into $cur (no venv)" && {
      echo "  harbor: $HARBOR"
      echo "  note: installed into the CURRENT interpreter because no venv route worked."
      return 0
    }
  fi

  echo "  every install route failed. Last 25 lines of $log:"
  sed 's/^/      /' "$log" | tail -25
  echo "  full log: $log"
  echo "  likely causes, in order:"
  echo "    - python too old:    harbor requires >=3.12. With conda already on the box:"
  echo "                           conda create -y -p $VENV python=3.12"
  echo "                           $VENV/bin/pip install harbor==${HARBOR_VER}"
  echo "    - no ensurepip:      apt-get install -y python3.12-venv   (needs root+apt)"
  echo "    - blocked PyPI:      pip config set global.index-url <your internal mirror>"
  echo "    - TLS interception:  pip install --cert /path/to/corp-ca.pem, or"
  echo "                         pip config set global.cert /path/to/corp-ca.pem"
  return 1
}

# ---- install mode: bring a BARE box to the point where --smoke can pass ----
# Written because the collaborator's H200 is a different machine from ours and will have none
# of this. Everything here is idempotent: re-running it after a partial failure is safe.
if [ "$MODE" = "--install" ]; then
  echo "== installing prerequisites =="
  SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"

  # 1. python3.12. Needed ONLY to create the venv; if a working harbor venv already exists
  #    this is skipped entirely.
  if resolve_harbor; then
    echo "  harbor venv already present at $VENV - skipping python install"
  elif command -v "$PY" >/dev/null 2>&1; then
    echo "  $PY already present"
  else
    echo "  installing python3.12 ..."
    if command -v apt-get >/dev/null 2>&1; then
      $SUDO apt-get update -qq && $SUDO apt-get install -y -qq python3.12 python3.12-venv \
        || echo "  apt install failed - try deadsnakes PPA, or set TB_PYTHON to any python3.12"
    else
      echo "  no apt-get. Install python3.12 yourself, then set TB_PYTHON to it."
    fi
  fi

  # 2. venv + harbor.
  if ! resolve_harbor; then
    if command -v "$PY" >/dev/null 2>&1; then
      echo "  creating $VENV and installing harbor==${HARBOR_VER} ..."
      install_harbor || echo "  (see the error above - this is the blocker, not a symptom)"
    else
      echo "  SKIP: no python3.12 available to build the venv"
    fi
  fi
  if resolve_harbor; then echo "  harbor OK ($HARBOR)"; else echo "  harbor NOT installed"; fi

  # 3. the harness checkout.
  if [ -d "$TB_DIR" ]; then
    echo "  LongHorizon-Harness already at $LHH_DIR"
  else
    echo "  cloning LongHorizon-Harness -> $LHH_DIR ..."
    git clone --depth 1 https://github.com/AMAP-ML/LongHorizon-Harness.git "$LHH_DIR" \
      || echo "  clone failed"
  fi

  # 4. docker. Deliberately NOT auto-installed: it needs root, changes a shared machine, and
  #    the usual failure is group membership rather than absence. Diagnose and instruct.
  if ! command -v docker >/dev/null 2>&1; then
    echo "  docker MISSING. Install it (Terminal-Bench runs every task in a container):"
    echo "    curl -fsSL https://get.docker.com | $SUDO sh"
  elif docker info >/dev/null 2>&1; then
    echo "  docker OK"
  else
    echo "  docker present but daemon unusable. Almost always group membership:"
    echo "    $SUDO usermod -aG docker \$USER   # then log out and back in"
    echo "    (verify with: docker info)"
  fi

  echo
  echo "install pass done - now run: bash $0 --smoke"
  exit 0
fi

echo "== Terminal-Bench 2.1 harness-swap preflight =="

# 1. Docker. Terminal-Bench executes every task in a container; without a usable daemon
#    nothing runs. Our A100 fails here with a docker.sock permission error, which is why
#    this script exists rather than a README instruction.
if ! command -v docker >/dev/null 2>&1; then
  bad "docker not installed" "Terminal-Bench runs EVERY task in a container. Install it, e.g.:
             curl -fsSL https://get.docker.com | sh   (as root, or with sudo)
           then start it:  dockerd >/tmp/dockerd.log 2>&1 &   (or: systemctl start docker)
           A container host often needs --privileged or a mounted /var/run/docker.sock;
           if you cannot get a daemon, tell us - that decides whether this task is runnable there."
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
if resolve_harbor; then
  ok "harbor already installed and runnable: $HARBOR"
elif ! command -v "$PY" >/dev/null 2>&1; then
  bad "no working harbor, and $PY not found to build one" \
      "Either point TB_VENV at a venv that has harbor, or set TB_PYTHON to a python3.12 binary."
else
  ok "$PY present"
  echo "        creating venv at $VENV and installing harbor==${HARBOR_VER} ..."
  install_harbor || true
  if resolve_harbor; then
    ok "harbor installed and runnable: $HARBOR"
  else
    bad "harbor not runnable - the pip error is printed above" \
        "Fix that error first. Everything downstream (agent imports, task fetch, both arms) depends on harbor."
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
# The key must be NON-EMPTY because the shipped script exits 2 on an empty one, but its VALUE
# is only checked by whatever serves ANTHROPIC_BASE_URL. Serving your own model behind a local
# proxy usually means any placeholder works, so default one in rather than block the run.
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  export ANTHROPIC_API_KEY="placeholder-local-endpoint"
  warn "ANTHROPIC_API_KEY was empty; using a placeholder. Fine for a self-hosted endpoint that"
  warn "does not check it. If your endpoint DOES authenticate, set the real key or every task fails."
else
  ok "ANTHROPIC_API_KEY set"
fi
# This one cannot be defaulted: it decides WHICH MODEL both arms talk to, and the entire
# experiment is that both arms share one policy. A wrong or absent value silently changes the
# thing being held fixed.
[ -n "${ANTHROPIC_BASE_URL:-}" ] && ok "ANTHROPIC_BASE_URL set ($ANTHROPIC_BASE_URL)" \
  || bad "ANTHROPIC_BASE_URL unset" "REQUIRED. Point it at the endpoint serving YOUR model; both arms must use the same one or the swap does not hold the model fixed."

# 5. Tasks. Not shipped with the repo.
TASKS="$TB_DIR/datasets/terminal-bench-2-1/tasks"
if [ -d "$TASKS" ] || [ -L "$TASKS" ]; then
  n=$(find -L "$TASKS" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && ok "task set present ($n tasks)" \
    || bad "task dir exists but is empty" "Populate $TASKS (a symlink is fine)."
else
  bad "task set missing" "Put Terminal-Bench 2.1 tasks at $TASKS. Try: $HARBOR download --help"
fi

# 6. Both arms must be defined. Upstream ships ONLY the LHH arm; the baseline is the work.
# Both arms' AGENT CLASSES must import, not merely their config files exist. A config naming
# a class that cannot be constructed fails hours into a run, inside a container, after the
# task set has been staged. Verified 2026-08-31: with PYTHONPATH set to Harness/src, the
# harbor venv imports BOTH harbor_agent:CuaHarnessClaudeCodeAgent (arm B) and
# harbor.agents.terminus_2.terminus_2:Terminus2 (arm A) with no extra installs beyond
# harbor itself -- so no separate environment build is needed for the TB path.
if [ -d "$TB_DIR/Harness/src" ] && [ -x "$VENV/bin/python" ]; then
  agent_check=$(cd "$TB_DIR" && PYTHONPATH="$TB_DIR/Harness/src" "$VENV/bin/python" - <<'PYCHK' 2>&1
try:
    import harbor_agent
    getattr(harbor_agent, "CuaHarnessClaudeCodeAgent")
    from harbor.agents.terminus_2.terminus_2 import Terminus2  # noqa: F401
    print("BOTH_OK")
except Exception as e:
    print(f"{type(e).__name__}: {e}")
PYCHK
)
  case "$agent_check" in
    *BOTH_OK*) ok "both arms' agent classes import (no extra env build needed)" ;;
    *"No module named 'harbor'"*)
      # Do not report this as a separate problem: it is the harbor failure above, restated.
      warn "agent import skipped - harbor is not installed yet (see the harbor failure above)" ;;
    *) bad "an agent class does not import: $agent_check" \
           "This fails hours into a run, inside a container. Fix before launching." ;;
  esac
fi

LHH_CFG="$TB_DIR/Scripts/tbench21_full_cua_harness_claudecode_qwen37_enable_thinking.yaml"
[ -f "$LHH_CFG" ] && ok "arm B (LHH MEA) config shipped" || bad "arm B config missing" "Expected $LHH_CFG"
if [ -n "${BASELINE_CFG:-}" ] && [ -f "$BASELINE_CFG" ]; then
  ok "arm A (baseline) config: $BASELINE_CFG"
else
  SHIPPED_BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tb21_baseline_terminus2.yaml"
  if [ -f "$SHIPPED_BASE" ]; then
    BASELINE_CFG="$SHIPPED_BASE"
    ok "arm A (baseline) defaulting to shipped Terminus 2 config: $BASELINE_CFG"
    warn "Terminus 2 is the OFFICIAL leaderboard harness, so this baseline is the comparison"
    warn "the leaderboard implies. Verify model_name matches arm B before trusting the delta."
  else
    warn "arm A (baseline) NOT defined and no shipped config found; set BASELINE_CFG=<yaml>."
    warn "A swap without its baseline measures nothing."
  fi
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
    if $HARBOR download "terminal-bench@2.1" -o "$DEST"; then
      break
    fi
    echo "  attempt $attempt failed; retrying in $((attempt*20))s"
    sleep $((attempt*20))
  done
  n=$(find -L "$DEST" -maxdepth 2 -mindepth 1 -type d 2>/dev/null | wc -l)
  if [ "$n" -eq 0 ]; then
    echo "FETCH FAILED: no tasks under $DEST after 5 attempts."
    echo "  The registry name may have changed. Try: $HARBOR download --help"
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
  $HARBOR run -c "$CFG" 2>&1 | tee -a "$OUT/run.log"
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
