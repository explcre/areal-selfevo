#!/usr/bin/env python3
"""Mutation-test selfevo/tests/test_gold_target_reachability.py against a COPY of the repo.

A copy, not the live checkout, for the reason every harness here says: a GPU job imports this
tree via PYTHONPATH, so a mutated production file sitting on disk for even a few seconds
could be read by a real run. This one mutates FOUR files rather than one -- the actor, the
rollout workflow, the MATH adapter and the apply seam -- because the claim under test spans
them: that a gold target is absent at every point between the dataset and the advantage
tensor, and that unsolved groups therefore reach no update.

Every target is sha256-compared against the LIVE checkout before the first mutation and
again after the last, so a harness that failed to restore something cannot be mistaken for a
clean run.

Usage: mutate_gold_target_reachability.py <path-to-copy-of-repo>
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

COPY = pathlib.Path(sys.argv[1]).resolve()
LIVE = pathlib.Path(__file__).resolve().parents[2]
TESTS = "selfevo/tests/test_gold_target_reachability.py"

ACTOR = "areal/trainer/ppo/actor.py"
WORKFLOW = "areal/workflow/rlvr.py"
ADAPTER = "areal/dataset/competition_math.py"
ROUTERS = "selfevo/routing/routers.py"
SEAM = "selfevo/integration/group_apply.py"
TARGETS = [ACTOR, WORKFLOW, ADAPTER, ROUTERS, SEAM]

_UNSOLVED_WRITE = """                    if gr.unsolved_advantage != 0.0:
                        row_adv = row_adv + torch.repeat_interleave(
                            silent * unsolved, sizes_t
                        ).to(row_adv.dtype) * gr.unsolved_advantage"""

_SOLVED_ROWS = """                        solved_rows = torch.repeat_interleave(
                            silent * solved, sizes_t
                        ).to(row_adv.dtype) * gr.solved_advantage"""

_APPLY_SIG = """    sft_weight: float,
    sft_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ApplyStats]:
    \"\"\"Apply one mode per group to the advantage tensor."""

# (label, file, find, replace) -- each a single-edit defect a careless change could produce.
MUTATIONS = [
    # -- the unsolved branch actually reaching the update ---------------------------------
    ("unsolved branch writes the SOLVED constant, so wrong rollouts are reinforced",
     ACTOR, _UNSOLVED_WRITE,
     _UNSOLVED_WRITE.replace("gr.unsolved_advantage", "gr.solved_advantage")),
    ("solved/unsolved classification swapped, so the constant lands on the wrong side",
     ACTOR,
     "                    solved = torch.tensor(\n"
     "                        [float(g.min() > 0.5) for g in per_group_r], device=advantages.device\n"
     "                    )",
     "                    solved = torch.tensor(\n"
     "                        [float(g.max() <= 0.5) for g in per_group_r], device=advantages.device\n"
     "                    )"),
    ("solved mask negated, so the SFT constant lands on the unsolved groups",
     ACTOR, _SOLVED_ROWS, _SOLVED_ROWS.replace("silent * solved", "silent * (1.0 - solved)")),
    # Kept although this file does NOT kill it; recorded in EXPECTED_SURVIVORS below. It is
    # a property of the SOLVED branch, which is not what this file claims, and the repo's
    # own suite already constrains it.
    ("silence gate dropped, so every group takes the constant whether silent or not",
     ACTOR, _SOLVED_ROWS, _SOLVED_ROWS.replace("silent * solved", "solved")),
    ("the write is never applied, so no group reaches the update at all",
     ACTOR,
     "                    if bool((row_adv != 0).any()):",
     "                    if False:"),
    # -- the has_target convention that closes the branch at every router -----------------
    ("RandomRouter's target guard removed, so an SFT draw fires with no target",
     ROUTERS,
     "        if known_modes()[chosen] and not ctx.has_target:",
     "        if False:"),
    ("RandomRouter's guard reads has_teacher, the defect measured on 2026-08-31",
     ROUTERS,
     "        if known_modes()[chosen] and not ctx.has_target:",
     "        if known_modes()[chosen] and not ctx.has_teacher:"),
    # -- the gold arriving, or a known field leaving, at the rollout schema ---------------
    ("a gold field is added to the trajectory while the seam still cannot spend it",
     WORKFLOW,
     '            "rewards": torch.tensor(reward, dtype=torch.float32),',
     '            "rewards": torch.tensor(reward, dtype=torch.float32),\n'
     '            "gold_ids": torch.tensor([0], dtype=torch.int32),'),
    ("a known trajectory key is dropped, so the schema read is checking nothing",
     WORKFLOW,
     '            "turn_ids": torch.tensor(turn_ids, dtype=torch.int32),\n',
     ""),
    # -- the adapter keeping or losing a column ------------------------------------------
    ("the adapter keeps the gold solution, which is the premise this file denies",
     ADAPTER,
     '    keep = {"messages", "answer"}',
     '    keep = {"messages", "answer", "solution"}'),
    ("the adapter drops the gold answer as well, leaving nothing to grade against",
     ADAPTER,
     '    keep = {"messages", "answer"}',
     '    keep = {"messages"}'),
    # -- the seam advertising a target it can never be given -----------------------------
    ("the apply seam grows a target argument with nothing to supply it",
     SEAM, _APPLY_SIG,
     _APPLY_SIG.replace(
         "    sft_rows: torch.Tensor | None = None,\n)",
         "    sft_rows: torch.Tensor | None = None,\n"
         "    sft_targets: torch.Tensor | None = None,\n)")),
]


# Mutants this file is not the right place to kill, each named with the test that DOES kill
# it. Recorded rather than deleted: a mutation that survives for a reason is evidence about
# this file's scope, and dropping it would leave the same defect uncovered on the day the
# other test moves. Verified 2026-09-01 by applying the mutation to the copy and running the
# whole selfevo suite, which failed exactly the test named here and nothing else.
EXPECTED_SURVIVORS = {
    "silence gate dropped, so every group takes the constant whether silent or not":
        "selfevo/tests/test_group_routing.py::test_routing_keys_on_silence_not_on_the_outcome"
        " (a solved-branch property, out of scope for this file)",
}


def run_tests() -> bool:
    """True if the suite passes against the copy."""
    env = dict(os.environ, PYTHONPATH=str(COPY))
    r = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x",
         "-p", "no:cacheprovider"],
        cwd=COPY, capture_output=True, text=True, timeout=1800, env=env,
    )
    return r.returncode == 0


def _sha(path: pathlib.Path) -> str:
    """sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_identical_to_live(when: str) -> None:
    """Refuse to proceed unless every target in the copy matches the live checkout."""
    for rel in TARGETS + [TESTS]:
        a, b = _sha(COPY / rel), _sha(LIVE / rel)
        if a != b:
            raise SystemExit(f"COPY DIVERGED {when}: {rel} ({a[:12]} != {b[:12]})")
    print(f"copy is sha256-identical to the live checkout {when}")


def _assert_isolated() -> None:
    """Refuse to run unless pytest would import the COPY, not the live checkout."""
    env = dict(os.environ, PYTHONPATH=str(COPY))
    r = subprocess.run(
        [sys.executable, "-c",
         "import selfevo.integration.group_apply as g, areal.trainer.ppo.actor as a;"
         " print(g.__file__); print(a.__file__)"],
        cwd=COPY, capture_output=True, text=True, env=env, timeout=600,
    )
    got = [pathlib.Path(p).resolve() for p in r.stdout.split()]
    want = [COPY / SEAM, COPY / ACTOR]
    if got != want:
        raise SystemExit(f"ISOLATION FAILED: pytest would import {got}, not {want}")
    print(f"isolated: imports resolve under {COPY}")


def main() -> int:
    """Apply each mutation to the copy, run the tests, restore, and report."""
    _assert_isolated()
    _assert_identical_to_live("at start")

    originals = {rel: (COPY / rel).read_text() for rel in TARGETS}
    digests = {rel: _sha(COPY / rel) for rel in TARGETS}

    if not run_tests():
        print("BASELINE IS RED -- mutation results would be meaningless")
        return 2
    print(f"baseline green; {len(MUTATIONS)} mutations\n")

    survivors = []
    skipped = []
    expected = []
    for label, rel, find, repl in MUTATIONS:
        original = originals[rel]
        n = original.count(find)
        if n != 1:
            print(f"SKIP      {label}: anchor appears {n}x in {rel}")
            skipped.append((label, f"anchor appears {n}x in {rel}"))
            continue
        mutated = original.replace(find, repl, 1)
        if mutated == original:
            print(f"SKIP      {label}: replacement leaves {rel} byte-identical")
            skipped.append((label, "equivalent mutant: file unchanged"))
            continue
        target = COPY / rel
        target.write_text(mutated)
        try:
            compile(mutated, str(target), "exec")
        except SyntaxError as exc:
            target.write_text(original)
            print(f"SKIP      {label}: mutant does not compile ({exc.msg})")
            skipped.append((label, f"mutant does not compile: {exc.msg}"))
            continue
        passed = run_tests()
        target.write_text(original)
        assert _sha(target) == digests[rel], f"restore failed for {rel}"
        if passed and label in EXPECTED_SURVIVORS:
            print(f"expected  {label}: killed by {EXPECTED_SURVIVORS[label]}")
            expected.append(label)
        elif passed:
            print(f"SURVIVED  {label}")
            survivors.append((label, "tests still passed"))
        else:
            print(f"killed    {label}")

    _assert_identical_to_live("at finish")
    killed = len(MUTATIONS) - len(survivors) - len(skipped) - len(expected)
    print(f"\n{killed}/{len(MUTATIONS)} killed, {len(survivors)} survived, "
          f"{len(expected)} survived as expected, {len(skipped)} skipped")
    if skipped:
        print("\nSKIPPED (not evidence either way):")
        for label, why in skipped:
            print(f"  - {label}: {why}")
    if survivors:
        print("\nSURVIVORS (the tests do not constrain these):")
        for label, why in survivors:
            print(f"  - {label}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
