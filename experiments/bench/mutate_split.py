#!/usr/bin/env python3
"""Mutation test for the committed-split guards in math_bench.py.

These guards exist so a number cannot be produced against the wrong 250 problems. A guard
that looks armed and is not would be worse than none, so each is checked by breaking it.
"""
import subprocess, sys, shutil, pathlib

ROOT = pathlib.Path.home() / "areal-selfevo" / "experiments" / "bench"
SRC = ROOT / "math_bench.py"
MUTANTS = [
    ("S01", "checksum guard removed", '    if actual != d["dataset_md5"]:', "    if False:"),
    ("S02", "checksum compared to itself", '    if actual != d["dataset_md5"]:', "    if actual != actual:"),
    ("S03", "'all' returns a half instead of everything", '    if split == "all":\n        return None', '    if split == "all":\n        return set(range(250))'),
    ("S04", "search and report swapped", "    return set(d[split])", '    return set(d["report" if split == "search" else "search"])'),
    ("S05", "unknown split silently allowed", "    if split not in d:", "    if False:"),
    ("S06", "missing split file ignored", "    if not sf.exists():", "    if False:"),
    ("S07", "idx not recorded on rows", '"answer": str(r["answer"]), "idx": i}', '"answer": str(r["answer"]), "idx": 0}'),
    ("S08", "filter inverted (keeps the other half)", "        if keep is not None and i not in keep:", "        if keep is not None and i in keep:"),
    ("S09", "filter disabled entirely", "        if keep is not None and i not in keep:", "        if False:"),
]

def run() -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", "test_math_bench.py", "-q", "-x", "--no-header"],
                       cwd=ROOT, capture_output=True, text=True, timeout=400)
    return r.returncode == 0

orig = SRC.read_text()
shutil.copy2(SRC, SRC.with_suffix(".py.sbak"))
if not run():
    SRC.write_text(orig); sys.exit("BASELINE FAILS")
surv, killed, skip = [], [], []
try:
    for mid, desc, old, new in MUTANTS:
        if old not in orig:
            skip.append(mid); print(f"  SKIPPED   {mid}  {desc}"); continue
        if orig.count(old) > 1:
            skip.append(mid); print(f"  SKIPPED   {mid}  {desc} [>1 match]"); continue
        SRC.write_text(orig.replace(old, new, 1))
        try: ok = run()
        except subprocess.TimeoutExpired: ok = False
        (surv if ok else killed).append(mid)
        print(f"  {'SURVIVED' if ok else 'killed  '}  {mid}  {desc}")
finally:
    SRC.write_text(orig)
    assert SRC.read_text() == orig, "RESTORE FAILED"
    SRC.with_suffix(".py.sbak").unlink()
print(f"\nkilled {len(killed)}/{len(killed)+len(surv)}  skipped {len(skip)}")
sys.exit(1 if surv or skip else 0)
