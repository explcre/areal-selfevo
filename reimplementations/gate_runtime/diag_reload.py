#!/usr/bin/env python3
"""Find a way to replace a served adapter every step that does not deadlock.

Observed: ``/unload_lora_adapter`` logged "Start unload Lora adapter" and never returned, with
the scheduler idle at 0% GPU. Since the arms replace their adapter after every gradient step,
a reload path that can hang is fatal to the whole experiment, so three candidates are timed
here and the first that survives repeated use is the one the trainer uses.

  1. load(name, new_path) with no unload -- does the server replace in place?
  2. unload(name) then load(name, new_path) -- the path that hung
  3. load under a FRESH name each time, unloading the previous one afterwards
"""
from __future__ import annotations

import json
import shutil
import sys
import time

import requests

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30031"
SRC = "/mnt/localssd/gate/runs/T/adapter"
TMO = 120


def post(path, body):
    """POST with a hard timeout, returning (status, seconds) or ('TIMEOUT', seconds)."""
    t = time.time()
    try:
        r = requests.post(URL + path, json=body, timeout=TMO)
        return r.status_code, round(time.time() - t, 1), r.text[:120]
    except requests.exceptions.Timeout:
        return "TIMEOUT", round(time.time() - t, 1), ""


def copy(i):
    """A distinct copy of the adapter directory, so each load sees a new path."""
    dst = "/mnt/localssd/gate/adapters/reload_%d" % i
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(SRC, dst)
    return dst


out = {}
print("--- candidate 1: load with the SAME name, no unload ---", flush=True)
rows = []
for i in range(3):
    rows.append(post("/load_lora_adapter", {"lora_name": "T", "lora_path": copy(i)}))
    print("  ", rows[-1], flush=True)
    if rows[-1][0] == "TIMEOUT":
        break
out["same_name_no_unload"] = rows

print("--- candidate 2: unload then load ---", flush=True)
rows = []
for i in range(3, 5):
    rows.append(("unload",) + post("/unload_lora_adapter", {"lora_name": "T"}))
    print("  ", rows[-1], flush=True)
    if rows[-1][1] == "TIMEOUT":
        break
    rows.append(("load",) + post("/load_lora_adapter", {"lora_name": "T", "lora_path": copy(i)}))
    print("  ", rows[-1], flush=True)
    if rows[-1][1] == "TIMEOUT":
        break
out["unload_then_load"] = rows

print(json.dumps(out, indent=1))
json.dump(out, open("/mnt/localssd/gate/out/diag_reload.json", "w"), indent=1)
