#!/usr/bin/env python3
"""Test the run-level smoke check against a real run and against the two defects it must catch.

A check that has never been shown to fail is not evidence. These cases are built by mutating a
REAL run's records, so the pass case and the fail cases differ only in the thing under test.
"""
from __future__ import annotations

import copy
import json
import sys

sys.path.insert(0, "/mnt/localssd/gate/code")
sys.path.insert(0, "/mnt/localssd/gate/ornith")

from postscaffold_train import smoke_verdict  # noqa: E402

REAL = "/mnt/localssd/gate/runs/PRE_fixed/iters.jsonl"


def main() -> int:
    """Run the four cases and report; non-zero if any behaves wrongly."""
    recs = [json.loads(l) for l in open(REAL) if l.strip()]
    fails = []

    v = smoke_verdict(recs, "pre")
    print("real repaired run                       -> %s" % (v or "PASS"))
    if v:
        fails.append("the real repaired run should pass")

    # Defect 1, the original: the harness stage is handed an empty list every iteration.
    m = copy.deepcopy(recs)
    for r in m:
        if r.get("updates"):
            r["updates"]["harness"] = {"rows": 0, "grad_norm": 0.0, "fp_delta": 0.0,
                                       "loss": None, "tokens": 0}
    v = smoke_verdict(m, "pre")
    print("harness starved every iteration         -> %s" % (v or "PASS"))
    if not any("harness" in x for x in v):
        fails.append("a starved harness stage was not caught")

    # Defect 2, the one the last check had: the FINAL iteration formed no group, so a
    # last-record-only check tested nothing. Here the run is otherwise sound and must pass.
    m2 = copy.deepcopy(recs)
    m2[-1]["formed_task_group"] = False
    m2[-1]["updates"] = None
    v = smoke_verdict(m2, "pre")
    print("sound run whose last iteration is empty -> %s" % (v or "PASS"))
    if v:
        fails.append("a sound run was failed because its last iteration formed no group")

    # And a run where NO iteration formed a group must fail rather than pass vacuously.
    m3 = copy.deepcopy(recs)
    for r in m3:
        r["formed_task_group"] = False
        r["updates"] = None
    v = smoke_verdict(m3, "pre")
    print("no iteration formed a group             -> %s" % (v or "PASS"))
    if not v:
        fails.append("a run in which nothing trained passed vacuously")

    # A stage that stepped but moved no parameter must not count as trained.
    m4 = copy.deepcopy(recs)
    for r in m4:
        if r.get("updates"):
            r["updates"]["harness"]["fp_delta"] = 0.0
    v = smoke_verdict(m4, "pre")
    print("harness stepped but moved no parameter  -> %s" % (v or "PASS"))
    if not any("harness" in x for x in v):
        fails.append("a harness step that moved no parameter was accepted")

    if fails:
        print("\nFAILED: %s" % fails)
        return 1
    print("\nAll five cases behaved correctly: the check passes the repaired run and catches "
          "both defects it exists to catch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
