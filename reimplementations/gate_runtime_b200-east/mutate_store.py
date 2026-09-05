#!/usr/bin/env python3
"""Break each guarantee of the store on purpose; the suite must notice every one."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

SRC = "/mnt/localssd/gate/code"
MUTANTS = [
    ("a tie is presented as a ranking with no marker",
     "tasksource/routing.py",
     "    tied = len(set(vals)) == 1 and len(vals) > 1",
     "    tied = False",
     "test_16"),
    ("cheap_tokens ranks on production+verification added together",
     "tasksource/routing.py",
     '    "cheap_tokens": ("tokens_per_accepted", "lower"),',
     '    "cheap_tokens": ("verify_tokens_per_accepted", "lower"),',
     "test_15"),
    ("above_floor reverts to the threshold test it was misnamed for (the v1 defect)",
     "tasksource/store.py",
     '            "above_floor": None if floor is None else bool(score >= floor)}',
     '            "above_floor": None if floor is None else bool(score >= threshold)}',
     "test_14"),
    ("schema-version check removed (older rows silently reinterpreted)",
     "tasksource/store.py",
     "            if v != SCHEMA_VERSION:",
     "            if False:",
     "test_10"),
    ("missing-version check removed (a row with no version is guessed at)",
     "tasksource/store.py",
     "            if v is None:",
     "            if False:",
     "test_10"),
    ("a refutation no longer needs its witness",
     "tasksource/store.py",
     '        if self.verification["verdict"] == "refuted" and not self.verification.get("witness"):',
     "        if False:",
     "test_05"),
    ("a success rate no longer needs the prompt it was measured under",
     "tasksource/store.py",
     '            if self.difficulty.get("success_rate") is not None and not self.difficulty.get(',
     "            if False and self.difficulty.get(",
     "test_06"),
    ("a dedup score no longer needs its false-positive floor",
     "tasksource/store.py",
     '            if "false_positive_floor" not in d:',
     "            if False:",
     "test_07"),
    ("distilled tasks no longer need to name their teacher",
     "tasksource/store.py",
     "        if missing:",
     "        if False:",
     "test_01"),
    ("the store rewrites instead of appending",
     "tasksource/store.py",
     '        with open(self.path, "a") as fh:\n            fh.write(json.dumps(rec.to_json(), sort_keys=True) + "\\n")\n        return rec.task_id',
     '        with open(self.path, "w") as fh:\n            fh.write(json.dumps(rec.to_json(), sort_keys=True) + "\\n")\n        return rec.task_id',
     "test_11"),
    ("routing ranks an unmeasured field as zero",
     "tasksource/routing.py",
     "    measured = [p for p in cand if getattr(p, field_name) is not None]\n    unmeasured = [p for p in cand if getattr(p, field_name) is None]",
     "    measured = list(cand)\n    unmeasured = []",
     "test_13"),
]


def run_one(name, relpath, old, new, expect):
    with tempfile.TemporaryDirectory() as d:
        shutil.copytree(os.path.join(SRC, "tasksource"), os.path.join(d, "tasksource"))
        shutil.copy(os.path.join(SRC, "test_store.py"), d)
        path = os.path.join(d, relpath)
        src = open(path).read()
        if old not in src:
            return name, "COULD NOT APPLY", relpath
        open(path, "w").write(src.replace(old, new, 1))
        r = subprocess.run([sys.executable, "test_store.py"], cwd=d, capture_output=True,
                           text=True)
        caught = any(l.startswith(("FAIL " + expect, "ERROR " + expect))
                     for l in r.stdout.splitlines())
        return name, ("CAUGHT" if caught else "SURVIVED"), ""


if __name__ == "__main__":
    print("%-62s %s" % ("mutation", "verdict"))
    survived = 0
    for m in MUTANTS:
        n, v, note = run_one(*m)
        if v != "CAUGHT":
            survived += 1
        print("%-62s %s %s" % (n[:62], v, note))
    print()
    print(("%d mutant(s) SURVIVED" % survived) if survived
          else "all %d mutants caught: every guarantee is constrained by a test" % len(MUTANTS))
    raise SystemExit(1 if survived else 0)
