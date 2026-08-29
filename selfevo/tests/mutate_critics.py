#!/usr/bin/env python3
"""Mutation test for selfevo/critics.py.

The first version of this module passed 123 tests while 17 mutations survived, so a green
suite is not evidence. Each mutation below is a defect a reader could plausibly introduce;
a mutation that survives marks a line the tests do not actually constrain.

Every mutation targets something the audit found unconstrained, plus the invariants the
rewrite is supposed to establish (normalisation, the p_hat split, the carried unit_id).
"""
from __future__ import annotations
import subprocess, sys, shutil, pathlib

ROOT = pathlib.Path.home() / "areal-selfevo"
SRC = ROOT / "selfevo" / "critics.py"
TESTS = "selfevo/tests/test_critics.py"

# (id, description, old, new)
MUTANTS = [
    ("M01", "group_size hardcoded to 8",            "g = ctx.group_size", "g = 8"),
    ("M02", "no normalisation (raw I_RL)",          "value = info / ceiling", "value = info"),
    ("M03", "ceiling fixed at 1.0",                 "ceiling = rl_informativeness(0.5, g)", "ceiling = 1.0"),
    ("M04", "ceiling uses p not 0.5",               "ceiling = rl_informativeness(0.5, g)", "ceiling = rl_informativeness(p, g) or 1.0"),
    ("M05", "solved_at default 1.0 -> 0.9",         "solved_at: float = 1.0", "solved_at: float = 0.9"),
    ("M06", "unsolved_at default 0.0 -> 0.1",       "unsolved_at: float = 0.0", "unsolved_at: float = 0.1"),
    ("M07", "solved_value default 0.0 -> 0.5",      "solved_value: float = 0.0", "solved_value: float = 0.5"),
    ("M08", "unsolved_floor default 0.5 -> 0.9",    "unsolved_floor: float = 0.5", "unsolved_floor: float = 0.9"),
    ("M09", "coarse_below default 8 -> 2",          "coarse_below: int = 8", "coarse_below: int = 2"),
    ("M10", "unit_id dropped (the live bug)",       "unit_id=unit_id if unit_id is not None else ctx.unit_id", "unit_id=unit_id"),
    ("M11", "context unit_id wins over explicit",   "unit_id=unit_id if unit_id is not None else ctx.unit_id", "unit_id=ctx.unit_id"),
    ("M12", "singleton group accepted",             "if g < 2:", "if g < 0:"),
    ("M13", "solved test >= becomes >",             "if p >= self.solved_at:", "if p > self.solved_at:"),
    ("M14", "unsolved test <= becomes <",           "elif p <= self.unsolved_at:", "elif p < self.unsolved_at:"),
    ("M15", "teacher branch ignored",               "if ctx.has_teacher:", "if True:"),
    ("M16", "teacher branch inverted",              "if ctx.has_teacher:", "if not ctx.has_teacher:"),
    ("M17", "coarse never flagged",                 "coarse = g <= self.coarse_below", "coarse = False"),
    ("M18", "coarse always flagged",                "coarse = g <= self.coarse_below", "coarse = True"),
    ("M19", "solved/unsolved basis swapped",        'basis = f"solved (p_hat={p:.3f} >= {self.solved_at}): no gradient, nothing to learn"',
                                                    'basis = f"unsolved (p_hat={p:.3f}) and no teacher: unlearnable now"'),
    ("M20", "informative basis reports wrong value", 'f"I_RL={info:.4f}/{ceiling:.4f}={value:.4f} at G={g}; predicts a non-zero "',
                                                     'f"I_RL={info:.4f}/{ceiling:.4f}={info:.4f} at G={g}; predicts a non-zero "'),
    ("M21", "side always INFORMATIVE",              "side = SilenceSide.SOLVED", "side = SilenceSide.INFORMATIVE"),
    ("M22", "history never appended",               "self._history.append(s)", "pass"),
    ("M23", "history returns the live list",        "return list(self._history)", "return self._history"),
    ("M24", "param range check dropped",            "if not 0.0 <= v <= 1.0:", "if False:"),
    ("M25", "ordering check dropped",               "if self.unsolved_at >= self.solved_at:", "if False:"),
    ("M26", "value range check dropped",            "if not 0.0 <= self.value <= 1.0:", "if False:"),
    ("M27", "empty basis allowed",                  "if not self.basis:", "if False:"),
    ("M28", "group_size not recorded on score",     "group_size=g,", "group_size=2,"),
    ("M29", "sample-granularity note dropped",      "if ctx.granularity is Granularity.SAMPLE:", "if False:"),
    ("M30", "unsolved floor ignored",               "value = self.unsolved_floor", "value = 0.0"),
]

def run_tests() -> bool:
    """True if the suite passes."""
    r = subprocess.run([sys.executable, "-m", "pytest", TESTS, "-q", "-x", "--no-header"],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    return r.returncode == 0

def main() -> int:
    original = SRC.read_text()
    backup = SRC.with_suffix(".py.mutbak")
    shutil.copy2(SRC, backup)
    # A dirty tree makes every later result meaningless -- a prior finding.
    if not run_tests():
        SRC.write_text(original); backup.unlink()
        print("BASELINE FAILS -- fix the suite before mutating"); return 2
    survived, killed, skipped = [], [], []
    try:
        for mid, desc, old, new in MUTANTS:
            if old not in original:
                skipped.append((mid, desc)); continue
            if original.count(old) > 1:
                skipped.append((mid, desc + " [AMBIGUOUS: pattern appears >1x]")); continue
            SRC.write_text(original.replace(old, new, 1))
            try:
                ok = run_tests()
            except subprocess.TimeoutExpired:
                ok = False
            (survived if ok else killed).append((mid, desc))
            print(f"  {'SURVIVED' if ok else 'killed  '}  {mid}  {desc}")
    finally:
        SRC.write_text(original)
        assert SRC.read_text() == original, "RESTORE FAILED"
        backup.unlink()
    print(f"\nkilled {len(killed)}/{len(killed)+len(survived)}   skipped {len(skipped)}")
    for mid, desc in skipped: print(f"  SKIPPED {mid} {desc}")
    if survived:
        print("\nSURVIVORS (unconstrained lines):")
        for mid, desc in survived: print(f"  {mid}  {desc}")
    return 1 if survived or skipped else 0

if __name__ == "__main__":
    sys.exit(main())
