#!/usr/bin/env bash
# Resolve the deps AReaL's RUNTIME WORKERS need, not the ones its tests need.
#
# An earlier resolver walked the test import chain and declared the box ready. The workers
# import a different set, so two runs died on missing modules discovered one at a time from
# GPU logs: litellm (proxy-rollout) and agents (math_agent workflow). This imports the
# actual worker entrypoints and installs whatever they ask for, until they all import.
set -u
source /venv/main/bin/activate
export UV_HTTP_TIMEOUT=600
cd /root/areal-selfevo
MODULES="areal.workflow.openai.math_agent areal.engine.sglang_remote areal.infra.rpc.guard areal.infra.data_service.worker areal.infra.data_service.router areal.infra.data_service.gateway areal.trainer.ppo.actor areal.trainer.rl_trainer"
PKG_FOR_agents=openai-agents
for i in $(seq 1 20); do
  miss=$(python - <<PY 2>/dev/null
import importlib
for m in "$MODULES".split():
    try:
        importlib.import_module(m)
    except ModuleNotFoundError as e:
        print(e.name); break
    except Exception:
        pass
PY
)
  if [ -z "$miss" ]; then
    echo "=== all worker modules import (after $((i-1)) install(s)) ==="
    python - <<PY
import importlib
ok = 0
for m in "$MODULES".split():
    try:
        importlib.import_module(m); ok += 1
    except ModuleNotFoundError as e:
        print(f"  STILL MISSING for {m}: {e.name}")
    except Exception as e:
        print(f"  {m}: imports but raised {type(e).__name__} (not a dependency problem)")
        ok += 1
print(f"  {ok} worker modules importable")
PY
    exit 0
  fi
  # A few import names differ from their distribution names.
  var="PKG_FOR_${miss}"
  pkg="${!var:-$miss}"
  echo "missing '$miss' -> installing '$pkg'"
  uv pip install "$pkg" >/dev/null 2>&1 || { echo "  FAILED to install $pkg"; exit 1; }
done
echo "gave up after 20 rounds"
