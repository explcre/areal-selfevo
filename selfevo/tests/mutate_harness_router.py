#!/usr/bin/env python3
"""Mutation-test selfevo/tests/test_harness_router.py against a COPY of the repo.

A copy, not the live checkout: the training supervisor relaunches on process exit, so a
mutated harness.py sitting on disk for even a few seconds could be imported by a real run.
Every mutation is a single-line defect a careless edit to ``propose_threshold`` could
produce; the file is restored and its sha256 re-checked after each one.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(sys.argv[1]).resolve()
TARGET = REPO / "selfevo/routing/harness.py"
TESTS = "selfevo/tests/test_harness_router.py"

# (label, find, replace) -- each is a single-line defect a careless edit could produce.
MUTATIONS = [
    ("boundary: solve_rate == propose_threshold no longer proposes",
     "if not self.partition and ctx.solve_rate <= self.effective_propose_threshold:",
     "if not self.partition and ctx.solve_rate < self.effective_propose_threshold:"),
    ("partition conjunct dropped: Co-Harness both trains and evolves",
     "if not self.partition and ctx.solve_rate <= self.effective_propose_threshold:",
     "if ctx.solve_rate <= self.effective_propose_threshold:"),
    ("decision reads failed_threshold, so the setting is silently ignored",
     "if not self.partition and ctx.solve_rate <= self.effective_propose_threshold:",
     "if not self.partition and ctx.solve_rate <= self.failed_threshold:"),
    ("property default inverted to solved_threshold: the extension is on by default",
     "return self.failed_threshold if self.propose_threshold is None else self.propose_threshold",
     "return self.solved_threshold if self.propose_threshold is None else self.propose_threshold"),
    ("property test inverted: is None -> is not None",
     "return self.failed_threshold if self.propose_threshold is None else self.propose_threshold",
     "return self.failed_threshold if self.propose_threshold is not None else self.propose_threshold"),
    ("field default inverted to 1.0",
     "    propose_threshold: float | None = None",
     "    propose_threshold: float | None = 1.0"),
    ("lower ordering guard weakened to a range check, so it never fires",
     "            if self.propose_threshold < self.failed_threshold:",
     "            if self.propose_threshold < 0.0:"),
    ("upper ordering guard weakened, so an unreachable band is accepted",
     "            if self.propose_threshold > self.solved_threshold:",
     "            if self.propose_threshold > 1.0:"),
    ("range check turns exclusive, rejecting the widest legal relaxation",
     "            if not 0.0 <= self.propose_threshold <= 1.0:",
     "            if not 0.0 <= self.propose_threshold < 1.0:"),
    ("mixed branch emits VALIDATE instead of PROPOSE",
     "                harness = HarnessAction.PROPOSE",
     "                harness = HarnessAction.VALIDATE"),
    ("mixed branch also skips the gradient, collapsing the two axes into one",
     "                harness = HarnessAction.PROPOSE",
     "                harness, mode = HarnessAction.PROPOSE, TrainingMode.SKIP"),
    ("new path's reason annotation dropped, so the audit record omits it",
     '                why = f"{why}; failed samples also proposed to the harness"',
     "                why = why"),
    ("harness-arm guard narrowed to VALIDATE, leaking the new PROPOSE path",
     "        if not ctx.can_evolve_harness and harness is not HarnessAction.NONE:",
     "        if not ctx.can_evolve_harness and harness is HarnessAction.VALIDATE:"),
    # Round 2: off-by-one on the new guards, and the reason annotations, which are the
    # audit record and have been wrong here before.
    ("upper ordering guard turns exclusive, rejecting the widest legal relaxation",
     "            if self.propose_threshold > self.solved_threshold:",
     "            if self.propose_threshold >= self.solved_threshold:"),
    ("lower ordering guard turns exclusive, rejecting the explicit default",
     "            if self.propose_threshold < self.failed_threshold:",
     "            if self.propose_threshold <= self.failed_threshold:"),
    ("effective threshold ignores the field entirely",
     "return self.failed_threshold if self.propose_threshold is None else self.propose_threshold",
     "return self.failed_threshold if self.propose_threshold is None else self.failed_threshold"),
    ("solved branch proposes instead of validating",
     "                HarnessAction.VALIDATE",
     "                HarnessAction.PROPOSE"),
    ("'harness only' note appended whenever an action exists, training or not",
     "        if harness is not HarnessAction.NONE and mode == TrainingMode.SKIP:",
     "        if harness is not HarnessAction.NONE or mode == TrainingMode.SKIP:"),
    ("'harness only' note appended to units no consumer takes at all",
     "        if harness is not HarnessAction.NONE and mode == TrainingMode.SKIP:",
     "        if mode == TrainingMode.SKIP:"),
    # Round 3: conjuncts a careless edit could bolt onto the new condition, and a
    # comparison written against the wrong left-hand side.
    ("new path silently gated on an external teacher",
     "if not self.partition and ctx.solve_rate <= self.effective_propose_threshold:",
     "if not self.partition and ctx.has_teacher and ctx.solve_rate <= self.effective_propose_threshold:"),
    ("new path silently gated on the ABSENCE of a teacher",
     "if not self.partition and ctx.solve_rate <= self.effective_propose_threshold:",
     "if not self.partition and not ctx.has_teacher and ctx.solve_rate <= self.effective_propose_threshold:"),
    ("new path silently excludes TOKEN granularity",
     "if not self.partition and ctx.solve_rate <= self.effective_propose_threshold:",
     "if not self.partition and ctx.granularity is not Granularity.TOKEN and ctx.solve_rate <= self.effective_propose_threshold:"),
    ("comparison written against the threshold, not the unit's solve rate",
     "if not self.partition and ctx.solve_rate <= self.effective_propose_threshold:",
     "if not self.partition and self.failed_threshold <= self.effective_propose_threshold:"),
    ("failed branch stops proposing",
     "            mode, why = self._failed_mode(ctx)\n            harness = HarnessAction.PROPOSE",
     "            mode, why = self._failed_mode(ctx)\n            harness = HarnessAction.NONE"),
    ("solved-side comparison turns exclusive",
     "        solved = ctx.solve_rate >= self.solved_threshold",
     "        solved = ctx.solve_rate > self.solved_threshold"),
]


def run_tests() -> bool:
    """True if the suite passes."""
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x", "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True, timeout=900, env=env,
    )
    return r.returncode == 0


def _assert_isolated() -> None:
    """Refuse to run unless the copy, not the live checkout, is what pytest imports."""
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run(
        [sys.executable, "-c",
         "import selfevo.routing.harness as h; print(h.__file__)"],
        cwd=REPO, capture_output=True, text=True, env=env, timeout=300,
    )
    got = pathlib.Path(r.stdout.strip()).resolve()
    if got != TARGET:
        raise SystemExit(f"ISOLATION FAILED: pytest would import {got}, not {TARGET}")
    print(f"isolated: imports resolve to {got}")


def main() -> int:
    _assert_isolated()
    original = TARGET.read_text()
    digest = hashlib.sha256(original.encode()).hexdigest()

    if not run_tests():
        print("BASELINE IS RED -- mutation results would be meaningless")
        return 2
    print(f"baseline green; {len(MUTATIONS)} mutations\n")

    survivors = []
    for label, find, repl in MUTATIONS:
        if original.count(find) != 1:
            print(f"SKIP  {label}: anchor appears {original.count(find)}x")
            survivors.append((label, "anchor not unique"))
            continue
        TARGET.write_text(original.replace(find, repl, 1))
        passed = run_tests()
        TARGET.write_text(original)
        assert hashlib.sha256(TARGET.read_text().encode()).hexdigest() == digest, "restore failed"
        if passed:
            print(f"SURVIVED  {label}")
            survivors.append((label, "tests still passed"))
        else:
            print(f"killed    {label}")

    print(f"\n{len(MUTATIONS) - len(survivors)}/{len(MUTATIONS)} killed")
    if survivors:
        print("\nSURVIVORS (the tests do not constrain these):")
        for label, why in survivors:
            print(f"  - {label}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
