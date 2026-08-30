#!/usr/bin/env python3
"""Confirm the packed tests catch the F10 failure they were written for."""
import subprocess, sys, pathlib
SRC = pathlib.Path.home() / "areal-selfevo" / "selfevo" / "integration" / "token_routing.py"
ROOT = pathlib.Path.home() / "areal-selfevo"

MUTANTS = [
    ("P1", "1-D treated as one sequence (the exact F10 bug)",
     "    packed = is_packed(loss_mask, cu_seqlens)",
     "    packed = False\n    loss_mask = loss_mask.view(1, -1) if loss_mask.ndim == 1 else loss_mask"),
    ("P2", "refusal on missing cu_seqlens removed",
     "    if loss_mask.ndim == 1 and not packed:", "    if False:"),
    ("P3", "output not repacked",
     "    if original_shape_1d:", "    if False:"),
]

def run():
    r = subprocess.run([sys.executable, "-m", "pytest", "selfevo/tests/test_packed_routing.py",
                        "-q", "--no-header"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    return r.returncode == 0

orig = SRC.read_text()
if not run():
    sys.exit("BASELINE FAILS")
print("baseline: PASS")
surv = []
try:
    for mid, desc, old, new in MUTANTS:
        if old not in orig:
            print(f"  SKIPPED  {mid}  {desc}"); surv.append(mid); continue
        SRC.write_text(orig.replace(old, new, 1))
        ok = run()
        if ok: surv.append(mid)
        print(f"  {'SURVIVED' if ok else 'killed  '}  {mid}  {desc}")
finally:
    SRC.write_text(orig)
    assert SRC.read_text() == orig, "RESTORE FAILED"
    print("restored:", "PASS" if run() else "FAIL")
sys.exit(1 if surv else 0)
