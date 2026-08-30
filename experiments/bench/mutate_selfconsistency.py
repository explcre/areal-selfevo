#!/usr/bin/env python3
"""Mutation test for the self-consistency baseline.

This baseline decides whether our method's gains are real, so a bug that flatters it is a
bug that makes our method look worse, and one that cripples it makes our method look better.
Both directions matter.
"""
import subprocess, sys, shutil, pathlib
ROOT = pathlib.Path.home() / "areal-selfevo" / "experiments" / "bench"
SRC = ROOT / "selfconsistency.py"
MUTANTS = [
    ("V01", "least common answer wins", "collections.Counter(vs).most_common(1)[0][0]", "collections.Counter(vs).most_common()[-1][0]"),
    # V02 is NEAR-EQUIVALENT and left in deliberately. Correctness is DERIVED from the
    # boxed answer, so every sample inside one normalised vote-group grades the same way and
    # preferring a "correct" member changes nothing. It differs only where normalisation is
    # coarser than grading -- merging 0.5 with 1/2, say -- which is why norm() is kept
    # deliberately conservative. Recorded rather than silently dropped.
    ("V02", "ties broken by correctness (near-equivalent, see note)", "winner = next(i for i in sub if votes[i] == top)", "winner = next((i for i in sub if votes[i] == top and correct[i]), next(i for i in sub if votes[i] == top))"),
    ("V03", "unparseable samples vote as a group", "vs = [votes[i] for i in sub if votes[i] is not None]", "vs = [votes[i] for i in sub]"),
    ("V04", "maj@k silently reports pass@k", "m += 1.0 if correct[winner] else 0.0", "m += 1.0 if any(correct[i] for i in sub) else 0.0"),
    ("V05", "pass@k requires ALL correct", "p += 1.0 if any(correct[i] for i in sub) else 0.0", "p += 1.0 if all(correct[i] for i in sub) else 0.0"),
    ("V06", "single-sample refusal removed", "    if K < 2:", "    if False:"),
    ("V07", "normalisation merges everything", "    return s or None", "    return \"X\""),
    ("V08", "frac/dfrac no longer merged", 's.replace("\\\\dfrac", "\\\\frac")', "s"),
    ("V09", "empty-file guard removed", "    if not rows:", "    if False:"),
]
def run():
    r = subprocess.run([sys.executable, "-m", "pytest", "test_selfconsistency.py", "-q", "-x", "--no-header"],
                       cwd=ROOT, capture_output=True, text=True, timeout=400)
    return r.returncode == 0
orig = SRC.read_text(); shutil.copy2(SRC, SRC.with_suffix(".py.vbak"))
if not run(): SRC.write_text(orig); sys.exit("BASELINE FAILS")
surv, killed, skip = [], [], []
try:
    for mid, desc, old, new in MUTANTS:
        if old not in orig: skip.append(mid); print(f"  SKIPPED   {mid}  {desc}"); continue
        if orig.count(old) > 1: skip.append(mid); print(f"  SKIPPED   {mid}  {desc} [>1]"); continue
        SRC.write_text(orig.replace(old, new, 1))
        try: ok = run()
        except subprocess.TimeoutExpired: ok = False
        (surv if ok else killed).append(mid)
        print(f"  {'SURVIVED' if ok else 'killed  '}  {mid}  {desc}")
finally:
    SRC.write_text(orig); assert SRC.read_text() == orig, "RESTORE FAILED"
    SRC.with_suffix(".py.vbak").unlink()
print(f"\nkilled {len(killed)}/{len(killed)+len(surv)}  skipped {len(skip)}")
sys.exit(1 if surv or skip else 0)
