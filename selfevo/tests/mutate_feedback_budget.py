"""Mutation test for the feedback-budget counter, including its vendor-tree call site.

Run against a COPY of the tree, never the live checkout:

    rsync -a --exclude .git ~/areal-selfevo/ /tmp/mut_fb/
    python3 selfevo/tests/mutate_feedback_budget.py /tmp/mut_fb

Half these mutations target ``areal/api/reward_api.py`` rather than ``selfevo/``, and that is
the point. A counter proved only against its own module cannot tell an instrumented call site
from an uninstrumented one, and the number that closes ``GOAL.md``'s matched-feedback-budget
row is the one produced at the real dispatch site.

Three columns, not two: SKIP is neither a kill nor a survival, and scoring it as either
reports a number that is not the truth.
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys
from dataclasses import dataclass

TESTS = ["selfevo/tests/test_feedback_budget.py"]

LIVE = pathlib.Path("/home/ubuntu/areal-selfevo").resolve()


@dataclass(frozen=True)
class Mutation:
    """One deliberate defect, keyword-only so no field can silently rebind.

    Attributes:
        label: What the defect is.
        rel: Source file, relative to the repo root.
        find: Exact text to replace; must appear exactly once.
        repl: Its replacement.
    """

    label: str
    rel: str
    find: str
    repl: str


MUTATIONS = [
    Mutation(
        label="THE COORDINATOR'S MUTATION: the counter never increments at the real call site",
        rel="areal/api/reward_api.py",
        find="                _selfevo_budget.record_call(attempt)",
        repl="                pass",
    ),
    Mutation(
        label="the instrumentation is removed from the dispatch loop entirely",
        rel="areal/api/reward_api.py",
        find="                from selfevo import feedback_budget as _selfevo_budget\n\n                _selfevo_budget.record_call(attempt)",
        repl="                _selfevo_unused = 0",
    ),
    Mutation(
        label="counting moves OUTSIDE the retry loop, so a retry costs nothing",
        rel="areal/api/reward_api.py",
        find="                _selfevo_budget.record_call(attempt)",
        repl="                _selfevo_budget.record_call(0) if attempt == 0 else None",
    ),
    Mutation(
        label="the final-timeout refusal is not counted, so a non-verdict looks like a verdict",
        rel="areal/api/reward_api.py",
        find="                        _selfevo_budget.record_refusal()",
        repl="                        pass",
    ),
    Mutation(
        label="a retry is counted as a call but not as a retry",
        rel="selfevo/feedback_budget.py",
        find="            if attempt > 0:\n                self._retries += 1",
        repl="            if False:\n                self._retries += 1",
    ),
    Mutation(
        label="every attempt is counted as a retry, making the series a constant",
        rel="selfevo/feedback_budget.py",
        find="            if attempt > 0:\n                self._retries += 1",
        repl="            if True:\n                self._retries += 1",
    ),
    Mutation(
        label="a cache hit is folded into verifier calls, so work avoided reads as work done",
        rel="selfevo/feedback_budget.py",
        find="            self._cache_hits += 1\n            self._touched = True",
        repl="            self._cache_hits += 1\n            self._calls += 1\n            self._touched = True",
    ),
    Mutation(
        label="the visibility flag is always on, so an unreadable counter reports a confident zero",
        rel="selfevo/feedback_budget.py",
        find="                visible=self._touched,",
        repl="                visible=True,",
    ),
    Mutation(
        label="the visibility flag never turns on, so a real count is discarded",
        rel="selfevo/feedback_budget.py",
        find="                visible=self._touched,",
        repl="                visible=False,",
    ),
    Mutation(
        label="a negative token count is subtracted from the budget instead of ignored",
        rel="selfevo/feedback_budget.py",
        find="        if n <= 0:\n            return",
        repl="        if False:\n            return",
    ),
    Mutation(
        label="the counter drops increments under concurrency (the lock is removed)",
        rel="selfevo/feedback_budget.py",
        find="        with self._lock:\n            self._calls += 1",
        repl="        if True:\n            self._calls += 1",
    ),
    Mutation(
        label="an invisible counter reports zero rather than NaN plus a flag",
        rel="selfevo/periodic_eval.py",
        find='                "budget/verifier_calls_total": nan,\n                "budget/verifier_calls_step": nan,',
        repl='                "budget/verifier_calls_total": 0.0,\n                "budget/verifier_calls_step": 0.0,',
    ),
    Mutation(
        label="the per-step series reports the cumulative total instead of the delta",
        rel="selfevo/periodic_eval.py",
        find="        delta = counts - prev if prev is not None else counts",
        repl="        delta = counts",
    ),
    Mutation(
        label="budget counters are emitted only on evaluation steps, not on every step",
        rel="selfevo/periodic_eval.py",
        find="        metrics = self.budget_metrics()\n        if not self.config.should_run(global_step):\n            return metrics",
        repl="        metrics = {}\n        if not self.config.should_run(global_step):\n            return metrics",
    ),
    Mutation(
        label="the counter is allowed to raise into the reward path",
        rel="selfevo/feedback_budget.py",
        find="    try:\n        fn(*args)\n    except Exception:  # pragma: no cover - defensive; asserted by test_feedback_budget\n        pass",
        repl="    fn(*args)",
    ),
]


#: Mutations that survive for a MEASURED reason rather than because a test is missing.
#: A survivor is a claim that the tests do not constrain the code, and making that claim when
#: the mutation has no observable effect overstates the gap exactly as scoring a SKIP as a
#: kill understates it. Each entry carries the measurement that justifies it; nothing goes in
#: here because it was inconvenient.
EXPECTED_SURVIVORS = {
    "the final-timeout refusal is not counted, so a non-verdict looks like a verdict": (
        "UNREACHABLE CODE on python<3.11. asyncio.TimeoutError is neither the builtin "
        "TimeoutError nor a subclass of it before 3.11, and AsyncRewardWrapper.__call__ "
        "catches the builtin, so its timeout branch -- and the record_refusal() inside it -- "
        "never executes on this box (python 3.10.12, measured). No test can kill a mutation "
        "in code that cannot run. Expected to be KILLED on python>=3.11 by "
        "test_a_reward_the_verifier_never_returned_is_counted_as_a_refusal, which skips here."
    ),
    "the counter drops increments under concurrency (the lock is removed)": (
        "BEHAVIOURALLY INERT on CPython 3.10. Measured directly: 64 threads x 50,000 "
        "unlocked increments at a 1e-9 switch interval lost 0 increments across 3 trials, as "
        "did 32x20,000 and 16x5,000. The GIL does not preempt this attribute increment, so "
        "the defect the mutation names cannot be observed by any test on this interpreter. "
        "The lock is kept because it is correct under free-threaded builds, not because a "
        "test proves it here -- and saying so is the point of this entry."
    ),
}


def run_tests(repo: pathlib.Path) -> bool:
    """Run the pinned tests inside one tree.

    Args:
        repo: The tree to run in.

    Returns:
        True when every test passed.
    """
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header", "-x", "-p", "no:cacheprovider"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=2400,
    )
    return r.returncode == 0


def main() -> int:
    """Apply every mutation in turn and report killed / survived / skipped.

    Returns:
        0 when every mutation was killed, 1 otherwise.
    """
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    repo = pathlib.Path(sys.argv[1]).resolve()
    if repo == LIVE:
        print(f"REFUSING: {repo} is the live checkout. Copy the tree and point this at the copy.")
        return 2

    originals, digests = {}, {}
    for m in MUTATIONS:
        if m.rel not in originals:
            originals[m.rel] = (repo / m.rel).read_text()
            digests[m.rel] = hashlib.sha256(originals[m.rel].encode()).hexdigest()

    if not run_tests(repo):
        print("BASELINE IS RED -- fix the tree before reading any mutation result")
        return 2
    print(f"baseline green; {len(MUTATIONS)} mutations\n")

    killed, survived, skipped, expected = [], [], [], []
    for m in MUTATIONS:
        p = repo / m.rel
        original = originals[m.rel]
        n = original.count(m.find)
        if n != 1:
            print(f"SKIP      {m.label}  (anchor appears {n}x)")
            skipped.append(m.label)
            continue
        mutated = original.replace(m.find, m.repl, 1)
        if mutated == original:
            print(f"SKIP      {m.label}  (replacement changed no bytes)")
            skipped.append(m.label)
            continue
        try:
            compile(mutated, str(p), "exec")
        except SyntaxError as exc:
            print(f"SKIP      {m.label}  (mutant does not compile: {exc})")
            skipped.append(m.label)
            continue
        p.write_text(mutated)
        try:
            still_green = run_tests(repo)
        finally:
            p.write_text(original)
            assert hashlib.sha256(p.read_text().encode()).hexdigest() == digests[m.rel]
        if still_green and m.label in EXPECTED_SURVIVORS:
            print(f"expected  {m.label}")
            expected.append(m.label)
        elif still_green:
            print(f"SURVIVED  {m.label}")
            survived.append(m.label)
        else:
            print(f"killed    {m.label}")
            killed.append(m.label)

    print(
        f"\nkilled {len(killed)}  survived {len(survived)}  skipped {len(skipped)}"
        f"  expected-survivor {len(expected)}  of {len(MUTATIONS)}"
    )
    for x in survived:
        print(f"  SURVIVOR: {x}")
    for x in skipped:
        print(f"  SKIPPED:  {x}")
    for x in expected:
        print(f"  EXPECTED SURVIVOR: {x}\n      because {EXPECTED_SURVIVORS[x]}")
    return 1 if (survived or skipped) else 0


if __name__ == "__main__":
    raise SystemExit(main())
