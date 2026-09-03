"""Run the AReaL trainer, first recording the environment the PROCESS actually received.

A bash comment inside a backslash-continued command splices the continuation and silently
drops env-prefix assignments, so a launcher can run with a DIFFERENT model than its own text
says. Reading the .sh is therefore not evidence. This entrypoint writes os.environ, sys.argv
and the cwd of the REAL trainer process to a file before handing control to the trainer, so
the config the job actually got can be read back from an artifact afterwards.

Set ENTRY_DUMP to the output path and ENTRY_TARGET to the trainer script to run.
"""

import json
import os
import runpy
import socket
import sys

DUMP = os.environ["ENTRY_DUMP"]
TARGET = os.environ["ENTRY_TARGET"]

record = {
    "host": socket.gethostname(),
    "pid": os.getpid(),
    "cwd": os.getcwd(),
    "executable": sys.executable,
    "python": sys.version,
    "target": TARGET,
    "argv_passed_to_trainer": sys.argv[1:],
    "env": dict(os.environ),
}
os.makedirs(os.path.dirname(DUMP), exist_ok=True)
with open(DUMP, "w") as fh:
    json.dump(record, fh, indent=1, sort_keys=True)

print(f"ENTRY: wrote process env/argv to {DUMP}", flush=True)
for k in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "CUDA_VISIBLE_DEVICES", "TMPDIR"):
    print(f"ENTRY: {k} = {os.environ.get(k)!r}", flush=True)
# The model path is the thing most likely to be silently wrong; print it from argv.
for a in sys.argv[1:]:
    if a.startswith("actor.path=") or a.startswith("experiment_name=") or a.startswith(
        "actor.backend="
    ) or a.startswith("rollout.backend="):
        print(f"ENTRY: {a}", flush=True)

sys.argv = [TARGET] + sys.argv[1:]
runpy.run_path(TARGET, run_name="__main__")
