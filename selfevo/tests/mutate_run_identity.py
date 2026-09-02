"""Mutation test for the tracker-identity rules: break them on purpose, one way at a time.

Run against a COPY of the tree, never the live checkout:

    rsync -a --exclude .git ~/areal-selfevo/ ~/mutwork/ri/
    python3 selfevo/tests/mutate_run_identity.py ~/mutwork/ri

Same refusal and the same three-column reporting as ``mutate_periodic_eval.py``: a mutation
whose anchor was not unique, whose replacement changed no bytes, or whose result does not
compile has not been TESTED, and scoring it as a kill reports a number higher than the truth.

WHAT THE MUTATIONS ARE AIMED AT. The defect being fixed was silent by construction -- the
tracker warned once per commit and nothing failed -- so the tests that replace that silence
have to be shown to fail when the silence comes back. The two most important rows below are
therefore "the collision check passes when the step is in the past" and "the identifier is
the vendor constant again": each restores exactly the state A0 ran in for four hours.
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys
from dataclasses import dataclass

TESTS = ["selfevo/tests/test_run_identity.py"]

LIVE = pathlib.Path("/home/ubuntu/areal-selfevo").resolve()


@dataclass(frozen=True)
class Mutation:
    """One deliberate defect.

    Attributes:
        label: What the defect is, in the words a reader needs.
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
        label="THE DEFECT ITSELF: the identifier is the vendor constant again",
        rel="selfevo/run_identity.py",
        find='    return f"{stem}_{launch_token() if token is None else token}", False',
        repl="    return stem, False",
    ),
    Mutation(
        label="THE BRIEF'S MUTATION: the collision check passes when the step is in the past",
        rel="selfevo/run_identity.py",
        find="    if intended_resume or int(resumed_step) <= 0:\n        return",
        repl="    if True:\n        return",
    ),
    Mutation(
        label="the collision check treats a resumed step as fresh (off by one on zero)",
        rel="selfevo/run_identity.py",
        find="    if intended_resume or int(resumed_step) <= 0:",
        repl="    if intended_resume or int(resumed_step) >= 0:",
    ),
    Mutation(
        label="every launch is treated as an intended resume, so nothing is ever refused",
        rel="selfevo/run_identity.py",
        find='    return f"{stem}_{launch_token() if token is None else token}", False',
        repl='    return f"{stem}_{launch_token() if token is None else token}", True',
    ),
    Mutation(
        label="an explicitly named identifier is ignored, so a real resume forks the curve",
        rel="selfevo/run_identity.py",
        find="    if pinned:\n        return pinned, True",
        repl="    if False:\n        return pinned, True",
    ),
    Mutation(
        label="the launch token is time only, so a crash-relaunch inside one second collides",
        rel="selfevo/run_identity.py",
        find='    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now)) + f"_{int(entropy) % 100000:05d}"',
        repl='    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))',
    ),
    Mutation(
        label="THE BRIEF'S MUTATION: the first-write check passes when the step is behind",
        rel="selfevo/run_identity.py",
        find="    if int(resumed_step) <= 0 or int(log_step) >= int(resumed_step):\n        return",
        repl="    if True:\n        return",
    ),
    Mutation(
        label="the first-write check refuses a resume that picks up exactly where it left off",
        rel="selfevo/run_identity.py",
        find="    if int(resumed_step) <= 0 or int(log_step) >= int(resumed_step):",
        repl="    if int(resumed_step) <= 0 or int(log_step) > int(resumed_step):",
    ),
    Mutation(
        label="the vendor id is built from the constant suffix again (the wiring is reverted)",
        rel="areal/utils/stats_logger.py",
        find="            id=run_id,",
        repl='            id=f"{self.config.experiment_name}_{self.config.trial_name}_{suffix}",',
    ),
    Mutation(
        label="the startup freshness check is never called",
        rel="areal/utils/stats_logger.py",
        find="        assert_id_is_fresh(run_id, self._resumed_step, self._intended_resume)",
        repl="        pass",
    ),
    Mutation(
        label="the first-commit check is never called",
        rel="areal/utils/stats_logger.py",
        find="            assert_step_advances(\n",
        repl="            _unused = (\n",
    ),
]


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
        timeout=1200,
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
        print(
            f"REFUSING: {repo} is the live checkout. A training run imports this tree across "
            f"worker processes that relaunch, so a mutated file on disk can be imported by "
            f"it. Copy the tree first and point this at the copy."
        )
        return 2

    originals, digests = {}, {}
    for m in MUTATIONS:
        p = repo / m.rel
        if m.rel not in originals:
            originals[m.rel] = p.read_text()
            digests[m.rel] = hashlib.sha256(originals[m.rel].encode()).hexdigest()

    if not run_tests(repo):
        print("BASELINE IS RED -- fix the tree before reading any mutation result")
        return 2
    print(f"baseline green; {len(MUTATIONS)} mutations\n")

    killed, survived, skipped = [], [], []
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
            assert hashlib.sha256(p.read_text().encode()).hexdigest() == digests[m.rel], (
                f"failed to restore {m.rel}"
            )
        if still_green:
            print(f"SURVIVED  {m.label}")
            survived.append(m.label)
        else:
            print(f"killed    {m.label}")
            killed.append(m.label)

    print(
        f"\nkilled {len(killed)}  survived {len(survived)}  skipped {len(skipped)}"
        f"  of {len(MUTATIONS)}"
    )
    for x in survived:
        print(f"  SURVIVOR: {x}")
    for x in skipped:
        print(f"  SKIPPED:  {x}")
    return 1 if (survived or skipped) else 0


if __name__ == "__main__":
    raise SystemExit(main())
