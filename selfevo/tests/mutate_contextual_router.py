#!/usr/bin/env python3
"""Mutation-test selfevo/tests/test_contextual_router.py against a COPY of the repo.

A copy, not the live checkout: the training supervisor relaunches on process exit, so a
mutated router sitting on disk for even a few seconds could be imported by a real run.

The mutations are chosen around the ways this router could stop being contextual without
saying so: a zero substituted for a missing feature, a dropped intercept, a confidence term
computed without the arm's own data, an outcome credited to whichever arm is convenient.
Each is a single-line defect a careless edit could produce; the file is restored and its
sha256 re-checked after each one.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(sys.argv[1]).resolve()
TARGET = REPO / "selfevo/routing/contextual.py"
TESTS = "selfevo/tests/test_contextual_router.py"

MUTATIONS = [
    # ---------------------------------------------------------------- the argmax ----
    ("the UCB argmax keeps the LAST tie instead of the first, inverting the tie-break",
     "            if score > best_score:",
     "            if score >= best_score:"),
    ("modes are visited in the caller's order, so the tie-break depends on the config",
     "        for m in sorted(usable):",
     "        for m in usable:"),
    ("the confidence term is dropped, so the router never explores",
     "            score = float(theta @ x + self.alpha * np.sqrt(max(x @ A_inv @ x, 0.0)))",
     "            score = float(theta @ x)"),
    ("the confidence term ignores the arm's own data, so it never anneals",
     "            score = float(theta @ x + self.alpha * np.sqrt(max(x @ A_inv @ x, 0.0)))",
     "            score = float(theta @ x + self.alpha * np.sqrt(max(x @ x, 0.0)))"),
    ("theta is solved with A instead of its inverse",
     "            theta = A_inv @ self._b[m]",
     "            theta = self._A[m] @ self._b[m]"),
    ("the score ignores the context entirely: the arm's mean, not its prediction",
     "            score = float(theta @ x + self.alpha * np.sqrt(max(x @ A_inv @ x, 0.0)))",
     "            score = float(theta.sum() + self.alpha * np.sqrt(max(x @ A_inv @ x, 0.0)))"),
    # -------------------------------------------------------------- the features ----
    ("a missing feature becomes a zero instead of raising",
     "                if self.require_features:",
     "                if False:"),
    ("require_features is inverted",
     "                if self.require_features:",
     "                if not self.require_features:"),
    ("the non-finite feature guard is dropped",
     "                vals.append(v if np.isfinite(v) else 0.0)",
     "                vals.append(v)"),
    ("the intercept is dropped, forcing every arm's value through the origin",
     "        vals.append(1.0)",
     "        vals.append(0.0)"),
    ("the feature check runs only after the no-usable-mode short-circuit",
     "        x = self._vector(ctx)\n        usable = self._usable(ctx)",
     "        usable = self._usable(ctx)\n        x = self._vector(ctx) if usable else None"),
    ("features are read in sorted order rather than the configured order",
     "        for name in self.feature_names:",
     "        for name in sorted(self.feature_names):"),
    # ------------------------------------------------------------------- learning ----
    ("the mode-match check is skipped, so an outcome credits whichever arm it names",
     "            if out.mode != mode or mode not in self._A:",
     "            if mode not in self._A:"),
    ("the update is applied to the mode the OUTCOME names, not the one that was routed",
     "            mode, x = remembered",
     "            mode, x = out.mode, remembered[1]"),
    ("the remembered decision is not consumed, so one outcome can be credited twice",
     "            remembered = self._pending.pop(unit_id, None)",
     "            remembered = self._pending.get(unit_id, None)"),
    ("the finiteness guard on the update is dropped",
     "            if not (np.isfinite(delta_A).all() and np.isfinite(delta_b).all()):",
     "            if False:"),
    ("a rejected update is counted as applied",
     "                self.rejected += 1",
     "                self.updates += 1"),
    ("cost is ignored, so a cheap mode cannot compete on value per unit cost",
     "                delta_b = (out.value / out.cost) * x",
     "                delta_b = out.value * x"),
    ("A is updated with the identity instead of the outer product, losing the context",
     "                delta_A = np.outer(x, x)",
     "                delta_A = np.eye(len(x))"),
    ("the ridge prior is dropped from A, so the first observation is infinitely confident",
     "            self._A[m] = np.eye(d) * self.ridge",
     "            self._A[m] = np.eye(d)"),
    # -------------------------------------------------------------- pending cache ----
    ("the pending cache is unbounded",
     "            if len(self._pending) >= self.pending_cap:",
     "            if False:"),
    ("the cap is off by one, so the cache holds one more than configured",
     "            if len(self._pending) >= self.pending_cap:",
     "            if len(self._pending) > self.pending_cap:"),
    ("eviction drops the NEWEST decision instead of the oldest",
     "                self._pending.popitem(last=False)",
     "                self._pending.popitem(last=True)"),
    ("evictions stop being counted, so a lost decision looks like an unlucky arm",
     "                self.evicted += 1",
     "                self.evicted += 0"),
    ("re-routing a unit evicts a bystander instead of refreshing the entry",
     "            self._pending.pop(ctx.unit_id, None)\n            if len(self._pending) >= self.pending_cap:",
     "            if len(self._pending) >= self.pending_cap:"),
    ("a decision with no unit id is remembered under the key None",
     "        if ctx.unit_id is not None:",
     "        if True:"),
    # --------------------------------------------------------------- teacher guard ----
    ("the teacher guard reads has_teacher, so a free self-target is thrown away",
     "        return [m for m in self.modes if ctx.has_target or not known_modes()[m]]",
     "        return [m for m in self.modes if ctx.has_teacher or not known_modes()[m]]"),
    ("the teacher guard is removed",
     "        return [m for m in self.modes if ctx.has_target or not known_modes()[m]]",
     "        return list(self.modes)"),
    ("the teacher guard is inverted",
     "        return [m for m in self.modes if ctx.has_target or not known_modes()[m]]",
     "        return [m for m in self.modes if ctx.has_target or known_modes()[m]]"),
]

EXPECTED_SURVIVORS: dict[str, str] = {}


def run_tests() -> bool:
    """True if the suite passes."""
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x", "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True, timeout=1800, env=env,
    )
    return r.returncode == 0


def _assert_isolated() -> None:
    """Refuse to run unless the copy, not the live checkout, is what pytest imports."""
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run(
        [sys.executable, "-c", "import selfevo.routing.contextual as m; print(m.__file__)"],
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

    survivors, expected = [], []
    for label, find, repl in MUTATIONS:
        if original.count(find) != 1:
            print(f"SKIP  {label}: anchor appears {original.count(find)}x")
            survivors.append((label, "anchor not unique"))
            continue
        TARGET.write_text(original.replace(find, repl, 1))
        passed = run_tests()
        TARGET.write_text(original)
        assert hashlib.sha256(TARGET.read_text().encode()).hexdigest() == digest, "restore failed"
        if passed and label in EXPECTED_SURVIVORS:
            print(f"EQUIVALENT {label}")
            expected.append(label)
        elif passed:
            print(f"SURVIVED  {label}")
            survivors.append((label, "tests still passed"))
        else:
            print(f"killed    {label}")

    killed = len(MUTATIONS) - len(survivors) - len(expected)
    print(f"\n{killed}/{len(MUTATIONS)} killed, {len(expected)} equivalent by construction")
    for label in expected:
        print(f"  = {label}: {EXPECTED_SURVIVORS[label]}")
    if survivors:
        print("\nSURVIVORS (the tests do not constrain these):")
        for label, why in survivors:
            print(f"  - {label}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
