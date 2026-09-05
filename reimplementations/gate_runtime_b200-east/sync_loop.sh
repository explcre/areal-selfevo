#!/usr/bin/env bash
# Incremental sync to the bucket. This box is a SPOT instance: anything not in the bucket is
# assumed lost. Adapters are large, so `runs/*/ckpt` is synced too but only every Nth pass.
set -u
ROOT=/mnt/localssd/gate
DST="${DST:-gs://selfevo/runs/b200x8/gate}"
EVERY="${EVERY:-300}"
HEAVY_EVERY="${HEAVY_EVERY:-4}"
i=0
while true; do
  i=$((i+1))
  gsutil -mq rsync -r -x '.*\.tmp/.*' "$ROOT/code" "$DST/code" 2>/dev/null
  gsutil -mq rsync -r -x '.*\.gz$' "$ROOT/out"  "$DST/out"  2>/dev/null
  gsutil -mq rsync -r "$ROOT/logs" "$DST/logs" 2>/dev/null
  for d in "$ROOT"/runs/*/; do
    [ -d "$d" ] || continue
    arm=$(basename "$d")
    # One file at a time, and errors are NOT discarded. A single `gsutil cp a b c` aborts the
    # whole command when any one path is missing -- route_probe.json only exists after the
    # first gradient step -- and with stderr thrown away that read as a working sync for the
    # whole first hour of the run.
    for f in steps.jsonl meta.json route_probe.json; do
      [ -f "$d/$f" ] && gsutil -q cp "$d/$f" "$DST/runs/$arm/$f" \
        || { [ -f "$d/$f" ] && echo "[sync] FAILED $arm/$f"; }
    done
    if [ $((i % HEAVY_EVERY)) -eq 0 ] && [ -d "$d/ckpt" ]; then
      gsutil -mq rsync -r "$d/ckpt" "$DST/runs/$arm/ckpt" || echo "[sync] FAILED $arm/ckpt"
    fi
  done
  echo "[sync] pass $i $(date -Is)"
  sleep "$EVERY"
done
