#!/usr/bin/env python3
"""Mutation test for selfevo/meta_critics.py.

The class exists to tell an inverted critic from a silent one, so the mutants concentrate
on the tie rule, the verdict boundaries, and every path that is supposed to REFUSE.
"""
import subprocess, sys, shutil, pathlib

ROOT = pathlib.Path.home() / "areal-selfevo"
SRC = ROOT / "selfevo" / "meta_critics.py"
TESTS = "selfevo/tests/test_meta_critics.py"

MUTANTS = [
    ("N01", "ties get first-rank not mid-rank", "        mid = (i + j) / 2.0 + 1.0", "        mid = i + 1.0"),
    ("N02", "AUC direction flipped", "    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)",
                                     "    return 1.0 - (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)"),
    ("N03", "mis-ordering collapsed into uninformative", "        elif auc <= 0.5 - self.margin:", "        elif False:"),
    ("N04", "informative boundary >= becomes >", "        if auc >= 0.5 + self.margin:", "        if auc > 0.5 + self.margin:"),
    ("N05", "margin ignored on the informative side", "        if auc >= 0.5 + self.margin:", "        if auc >= 0.5:"),
    ("N06", "min_paired check dropped", "        if len(usable) < self.min_paired:", "        if False:"),
    ("N07", "uniform-outcome refusal dropped", "        if all(outs) or not any(outs):", "        if False:"),
    ("N08", "coarse scores no longer dropped", "            if not self.use_coarse and s.coarse:", "            if False:"),
    ("N09", "coarse always dropped even when asked", "            if not self.use_coarse and s.coarse:", "            if s.coarse:"),
    ("N10", "unpaired scores silently used", "            if s.unit_id is None or s.unit_id not in outcomes:", "            if False:"),
    ("N11", "None unit_id treated as pairable", "            if s.unit_id is None or s.unit_id not in outcomes:", "            if s.unit_id not in outcomes:"),
    ("N12", "constant-scorer branch removed", "        if len(set(vals)) == 1:", "        if False:"),
    ("N13", "min_paired validation dropped", "        if self.min_paired < 2:", "        if False:"),
    ("N14", "margin validation dropped", "        if not 0.0 <= self.margin < 0.5:", "        if False:"),
    ("N15", "auc range check dropped", "        if self.auc is not None and not 0.0 <= self.auc <= 1.0:", "        if False:"),
    ("N16", "INSUFFICIENT may carry an AUC", "        if self.verdict is CalibrationVerdict.INSUFFICIENT and self.auc is not None:", "        if False:"),
    ("N17", "verdict may lack an AUC", "        if self.verdict is not CalibrationVerdict.INSUFFICIENT and self.auc is None:", "        if False:"),
    ("N18", "empty basis allowed", "        if not self.basis:", "        if False:"),
    ("N19", "single-class AUC allowed", "    if not pos or not neg:", "    if False:"),
    ("N20", "n_paired misreported as n_scored", "auc=auc, verdict=verdict, n_scored=n_scored, n_paired=len(usable),", "auc=auc, verdict=verdict, n_scored=n_scored, n_paired=n_scored,"),
]

def run() -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", TESTS, "-q", "-x", "--no-header"],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    return r.returncode == 0

orig = SRC.read_text()
shutil.copy2(SRC, SRC.with_suffix(".py.mbak"))
if not run():
    SRC.write_text(orig); sys.exit("BASELINE FAILS")
surv, killed, skip = [], [], []
try:
    for mid, desc, old, new in MUTANTS:
        if old not in orig:
            skip.append((mid, desc)); print(f"  SKIPPED   {mid}  {desc}"); continue
        if orig.count(old) > 1:
            skip.append((mid, desc)); print(f"  SKIPPED   {mid}  {desc} [>1 match]"); continue
        SRC.write_text(orig.replace(old, new, 1))
        try:
            ok = run()
        except subprocess.TimeoutExpired:
            ok = False
        (surv if ok else killed).append((mid, desc))
        print(f"  {'SURVIVED' if ok else 'killed  '}  {mid}  {desc}")
finally:
    SRC.write_text(orig)
    assert SRC.read_text() == orig, "RESTORE FAILED"
    SRC.with_suffix(".py.mbak").unlink()
print(f"\nkilled {len(killed)}/{len(killed)+len(surv)}  skipped {len(skip)}")
sys.exit(1 if surv or skip else 0)
