#!/usr/bin/env python3
"""Mutation test for extract_boxed's truncated-box fallback.

The fallback changes reported accuracy on real data, so the tests around it must actually
constrain it. Each mutant is a plausible way to get the fallback subtly wrong.
"""
import subprocess, sys, shutil, pathlib

ROOT = pathlib.Path.home() / "areal-selfevo" / "experiments" / "bench"
SRC = ROOT / "math_bench.py"
MUTANTS = [
    ("B01", "first balanced box wins instead of last", "for start in reversed(starts):", "for start in starts:"),
    ("B02", "no fallback (the original defect)", "for start in reversed(starts):", "for start in [starts[-1]]:"),
    ("B03", "strip() dropped", 'return "".join(buf).strip()', 'return "".join(buf)'),
    ("B04", "depth never increments on {", "            depth += 1", "            depth += 0"),
    ("B05", "depth never decrements on }", "            depth -= 1", "            depth -= 0"),
    ("B06", "returns at depth 1 not 0", "                if depth == 0:", "                if depth == 1:"),
    ("B07", "empty starts not guarded", "    if not starts:", "    if False:"),
    ("B08", "closing brace appended to output", '            if depth == 0:\n                return "".join(buf).strip()', '            if depth == 0:\n                buf.append(c)\n                return "".join(buf).strip()'),
]

def run() -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", "test_math_bench.py", "-q", "-x", "--no-header"],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    return r.returncode == 0

orig = SRC.read_text()
shutil.copy2(SRC, SRC.with_suffix(".py.mb"))
if not run():
    SRC.write_text(orig); sys.exit("BASELINE FAILS")
surv, killed, skip = [], [], []
try:
    for mid, desc, old, new in MUTANTS:
        if old not in orig:
            skip.append((mid, desc)); print(f"  SKIPPED  {mid}  {desc}"); continue
        if orig.count(old) > 1:
            skip.append((mid, desc + " [AMBIGUOUS]")); print(f"  SKIPPED  {mid}  {desc} [>1 match]"); continue
        SRC.write_text(orig.replace(old, new, 1))
        ok = run()
        (surv if ok else killed).append((mid, desc))
        print(f"  {'SURVIVED' if ok else 'killed  '}  {mid}  {desc}")
finally:
    SRC.write_text(orig)
    assert SRC.read_text() == orig, "RESTORE FAILED"
    SRC.with_suffix(".py.mb").unlink()
print(f"\nkilled {len(killed)}/{len(killed)+len(surv)}  skipped {len(skip)}")
sys.exit(1 if surv or skip else 0)
