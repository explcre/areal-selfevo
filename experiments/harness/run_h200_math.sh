#!/usr/bin/env bash
# Score models on the frozen math suite on a box with GPUs but NO DOCKER.
#
# Written for a collaborator whose container cannot run Terminal-Bench: harbor puts every
# task in a container, so that benchmark is simply unavailable there. Nothing here needs a
# container, a daemon, or root -- only Python, pip and the GPUs.
#
#   bash run_h200_math.sh --install   # venv + sglang + bench deps, tries several routes
#   bash run_h200_math.sh --fetch     # download the model weights, resumable
#   bash run_h200_math.sh --smoke     # 5 problems end to end on ONE gpu, ~5 min
#   bash run_h200_math.sh --run       # the full suite, sharded across all GPUs
#
# Every step is re-runnable: --install skips what already imports, --fetch resumes, and
# --run skips a benchmark whose results.json already exists unless FORCE=1.
set -u -o pipefail

MODE="${1:-}"
VENV="${BENCH_VENV:-$HOME/bench-env}"
PY="${BENCH_PYTHON:-python3}"
MODEL="${MODEL:-Qwen/Qwen2.5-Math-7B-Instruct}"
TAG="${TAG:-$(echo "$MODEL" | tr '/' '_')}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTROOT="${OUTROOT:-$HOME/runs/math}"
# Reasoning models need room; a truncated generation is graded WRONG, so a cap that is too
# small silently reports a token budget rather than a capability. See BENCH_OVERRIDES.
MAXTOK="${MAXTOK:-32768}"
GPUS_ALL="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)}"
GPUS_ALL="${GPUS_ALL:-0}"
NGPU="$(echo "$GPUS_ALL" | tr ',' '\n' | grep -c .)"
# TP size per server. The suite splits into two groups (OlympiadBench, and the four short
# benchmarks), so the default is chosen to make those two servers consume EVERY GPU: idle
# GPUs are the main way a scoring run wastes a box. An explicit TP is respected, with a
# warning when it would strand cards.
if [ -n "${TP:-}" ]; then
  if [ "$((TP * 2))" -lt "$NGPU" ] && [ "$NGPU" -gt 1 ]; then
    echo "  warn: TP=$TP on $NGPU GPUs leaves $((NGPU - TP * 2)) idle; TP=$((NGPU / 2)) would use all" >&2
  fi
else
  TP=$(( NGPU >= 2 ? NGPU / 2 : 1 ))
fi

ok(){   printf '  \033[32mok\033[0m    %s\n' "$1"; }
bad(){  printf '  \033[31mFAIL\033[0m  %s\n' "$1"; printf '        -> %s\n' "$2"; FAILED=$((FAILED+1)); }
warn(){ printf '  \033[33mwarn\033[0m  %s\n' "$1"; }
FAILED=0

py_of_venv(){ echo "$VENV/bin/python"; }

make_venv() {
  # Same four-route fallback as the harness installer: a stock python3-venv is preferred but
  # is genuinely absent on many images, and a collaborator should not be blocked by that.
  local log="$OUTROOT/venv_setup.log"; mkdir -p "$OUTROOT"; : > "$log"
  [ -x "$VENV/bin/python" ] && { ok "reusing $VENV"; return 0; }
  if "$PY" -m venv "$VENV" >> "$log" 2>&1; then
    "$VENV/bin/python" -m ensurepip --upgrade >> "$log" 2>&1 || true
    [ -x "$VENV/bin/pip" ] || "$VENV/bin/python" -m ensurepip >> "$log" 2>&1 || true
    [ -x "$VENV/bin/python" ] && { ok "venv at $VENV"; return 0; }
  fi
  echo "  note: '$PY -m venv' did not yield a usable venv (missing python3-venv/ensurepip?)"
  if command -v virtualenv >/dev/null 2>&1 && virtualenv -q -p "$PY" "$VENV" >> "$log" 2>&1; then
    ok "virtualenv at $VENV"; return 0
  fi
  if command -v uv >/dev/null 2>&1 && uv venv "$VENV" >> "$log" 2>&1; then
    ok "uv venv at $VENV"; return 0
  fi
  if command -v conda >/dev/null 2>&1 && conda create -y -q -p "$VENV" python=3.12 >> "$log" 2>&1; then
    ok "conda env at $VENV"; return 0
  fi
  echo "  every venv route failed; last 20 lines of $log:"; sed 's/^/      /' "$log" | tail -20
  return 1
}

if [ "$MODE" = "--install" ]; then
  echo "== install: venv + sglang + bench deps (no docker, no root) =="
  make_venv || exit 1
  P="$(py_of_venv)"
  # sglang pulls a matching torch. Pinning the pair is what keeps a fresh box reproducible;
  # letting pip resolve freely is how a box ends up with a torch its driver cannot use.
  if "$P" -c "import sglang" >/dev/null 2>&1; then
    ok "sglang already importable ($("$P" -c 'import sglang;print(sglang.__version__)' 2>/dev/null))"
  else
    echo "  installing sglang==${SGL_VER:-0.5.10.post1} (this pulls torch; several minutes)"
    if ! "$P" -m pip install -q "sglang[all]==${SGL_VER:-0.5.10.post1}" 2>&1 | tail -25; then
      bad "sglang install failed" "Scroll up for the pip error. If it is a TLS failure, set: $P -m pip config set global.cert /path/to/corp-ca.pem"
    fi
  fi
  for pkg in datasets huggingface_hub math_verify; do
    "$P" -c "import ${pkg}" >/dev/null 2>&1 || "$P" -m pip install -q "$pkg" 2>&1 | tail -5
  done
  "$P" -c "import torch; print('  torch', torch.__version__, 'cuda', torch.version.cuda, 'devices', torch.cuda.device_count())" 2>&1 | tail -2
  echo "install pass done - now run: bash $0 --fetch"
  exit $((FAILED > 0))
fi


case "$MODE" in
  --fetch|--smoke|--run) ;;
  *) echo "Usage: $0 --install | --fetch | --smoke | --run"
     echo "  MODEL=<hf id>   default $MODEL"
     echo "  MAXTOK=<n>      default $MAXTOK   TP=<n> default $TP"
     exit 2 ;;
esac
[ -x "$VENV/bin/python" ] || { echo "no venv at $VENV; run: bash $0 --install"; exit 2; }
P="$(py_of_venv)"

if [ "$MODE" = "--fetch" ]; then
  echo "== fetch: $MODEL =="
  export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
  # snapshot_download resumes, so a dropped connection costs only the current shard.
  "$P" - "$MODEL" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(sys.argv[1], max_workers=8)
print("MODEL_PATH", path)
PYEOF
  exit $?
fi

resolve_model_path() {
  # Prefer a local snapshot so a scoring run never depends on the network mid-flight.
  export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
  local p
  p="$("$P" - "$MODEL" <<'PYEOF' 2>/dev/null
import sys
from huggingface_hub import snapshot_download
try:
    print(snapshot_download(sys.argv[1], local_files_only=True))
except Exception:
    print("")
PYEOF
)"
  [ -n "$p" ] && echo "$p" || echo "$MODEL"
}

if [ "$MODE" = "--smoke" ]; then
  echo "== smoke: 5 problems, 1 gpu, end to end =="
  MP="$(resolve_model_path)"; echo "  model: $MP"
  G="$(echo "$GPUS_ALL" | cut -d, -f1)"
  BENCH_VENV="$VENV" BENCHES="math500" MAXTOK=2048 LIMIT=5 CONC=4 \
    timeout 3600 bash "$REPO/experiments/bench/run_math.sh" "$MP" "smoke_$TAG" "$G" 8701
  rc=$?
  echo "smoke exit=$rc"
  [ $rc -eq 0 ] && echo "now run: bash $0 --run"
  exit $rc
fi

shard_done() {
  # True when this shard already produced a parseable results.json. Re-running a finished
  # shard costs GPU-hours and, worse, can overwrite a good result with a worse one from a
  # partial run. FORCE=1 overrides. The header promised this; it now exists.
  local tag="$1" f="$OUTROOT/$1/results.json"
  [ "${FORCE:-0}" = "1" ] && return 1
  [ -s "$f" ] || return 1
  "$P" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d else 1)" "$f" 2>/dev/null
}

if [ "$MODE" = "--run" ]; then
  echo "== run: full suite, $NGPU gpu(s), tp=$TP, max_tokens=$MAXTOK =="
  MP="$(resolve_model_path)"; echo "  model: $MP"
  mkdir -p "$OUTROOT"
  # Two shards: OlympiadBench is by far the longest, so it gets its own server and the four
  # short benchmarks share the other. With fewer than 2*TP GPUs everything runs on one.
  if [ "$NGPU" -ge $((TP * 2)) ]; then
    A="$(echo "$GPUS_ALL" | cut -d, -f1-$TP)"
    B="$(echo "$GPUS_ALL" | cut -d, -f$((TP + 1))-$((TP * 2)))"
    if shard_done "${TAG}_olymp"; then
      echo "  olympiadbench -> SKIP (results.json exists; FORCE=1 to redo)"; P1=""
    else
      BENCH_VENV="$VENV" BENCHES="olympiadbench" MAXTOK="$MAXTOK" CONC="${CONC:-24}" \
        timeout 43200 bash "$REPO/experiments/bench/run_math.sh" "$MP" "${TAG}_olymp" "$A" 8711 \
        > "$OUTROOT/${TAG}_olymp.out" 2>&1 &
      P1=$!
    fi
    if shard_done "${TAG}_core"; then
      echo "  core suite    -> SKIP (results.json exists; FORCE=1 to redo)"; P2=""
    else
      BENCH_VENV="$VENV" BENCHES="math500,amc23,aime24,aime25" MAXTOK="$MAXTOK" CONC="${CONC:-24}" \
        timeout 43200 bash "$REPO/experiments/bench/run_math.sh" "$MP" "${TAG}_core" "$B" 8721 \
        > "$OUTROOT/${TAG}_core.out" 2>&1 &
      P2=$!
    fi
    [ -n "$P1" ] && echo "  olympiadbench -> gpu $A (log $OUTROOT/${TAG}_olymp.out)"
    [ -n "$P2" ] && echo "  core suite    -> gpu $B (log $OUTROOT/${TAG}_core.out)"
    [ -n "$P1" ] && { wait $P1; echo "EXIT_OLYMP=$?" >> "$OUTROOT/${TAG}_olymp.out"; }
    [ -n "$P2" ] && { wait $P2; echo "EXIT_CORE=$?"  >> "$OUTROOT/${TAG}_core.out"; }
  else
    if shard_done "${TAG}_all"; then
      echo "  all -> SKIP (results.json exists; FORCE=1 to redo)"
    else
    BENCH_VENV="$VENV" BENCHES="math500,amc23,aime24,aime25,olympiadbench" \
      MAXTOK="$MAXTOK" CONC="${CONC:-24}" \
      timeout 86400 bash "$REPO/experiments/bench/run_math.sh" "$MP" "${TAG}_all" "$GPUS_ALL" 8711 \
      > "$OUTROOT/${TAG}_all.out" 2>&1
    echo "EXIT_ALL=$?" >> "$OUTROOT/${TAG}_all.out"
    fi
  fi
  echo "== results =="
  grep -hE "^(math500|amc23|aime24|aime25|olympiadbench)" "$OUTROOT/${TAG}"*.out 2>/dev/null
  grep -hE "CAP-LIMITED" "$OUTROOT/${TAG}"*.out 2>/dev/null
  exit 0
fi

echo "Usage: $0 --install | --fetch | --smoke | --run"
echo "  MODEL=<hf id>   default $MODEL"
echo "  MAXTOK=<n>      default $MAXTOK   TP=<n> default $TP"
exit 2
