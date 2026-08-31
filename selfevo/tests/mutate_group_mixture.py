#!/usr/bin/env python3
"""Mutation-test selfevo/tests/test_group_mixture.py against a COPY of the repo.

A copy, not the live checkout: ``group_apply.py`` and ``actor.py`` are imported by real
training processes and the supervisor relaunches on process exit, so a mutated file sitting
on disk for even a few seconds could be picked up by a run.

Usage: ``python3 selfevo/tests/mutate_group_mixture.py /path/to/repo/copy``

Three files are mutated rather than one, because the feature is three edits that can each be
individually wrong and each look fine on its own:

    selfevo/integration/group_apply.py   the blend itself
    areal/trainer/ppo/actor.py           the gate that decides whether the blend runs
    areal/api/cli_args.py                the validation that stops a mixture arm being
                                         configured into something that is not one

RESTORATION IS THE POINT OF THE STRUCTURE HERE. A mutation harness that is interrupted
leaves its target MUTATED on disk, and the next run then treats the mutation as its
baseline -- the defect is now "original" and every mutant built on top of it looks killed.
That has happened in this repo. So: every target is restored in a ``finally``, SIGINT and
SIGTERM are trapped and restore before exiting, and the last thing the run prints is a
digest check of every target against the bytes read at startup. If that check does not say
INTACT, the results are void and the checkout needs ``git checkout``.
"""
from __future__ import annotations

import hashlib
import pathlib
import signal
import subprocess
import sys

REPO = pathlib.Path(sys.argv[1])
TESTS = "selfevo/tests/test_group_mixture.py"

GROUP_APPLY = "selfevo/integration/group_apply.py"
ACTOR = "areal/trainer/ppo/actor.py"
CLI_ARGS = "areal/api/cli_args.py"

# (file, label, find, replace) -- each is a single defect a careless edit could produce.
MUTATIONS = [
    # ---- the blend itself -----------------------------------------------------------
    (GROUP_APPLY, "the RL component is dropped from the blend",
     "            terms.append(a * block)",
     "            terms.append(0.0 * block)"),
    (GROUP_APPLY, "the RL and SFT coefficients are swapped",
     "        a = w.get(TrainingMode.RL, 0.0)\n        b = w.get(TrainingMode.SFT, 0.0)",
     "        a = w.get(TrainingMode.SFT, 0.0)\n        b = w.get(TrainingMode.RL, 0.0)"),
    (GROUP_APPLY, "the SFT component blends the ORIGINAL advantages, not the SFT write",
     "            terms.append(sft_only[rows] if b == 1.0 else b * sft_only[rows])",
     "            terms.append(block if b == 1.0 else b * block)"),
    (GROUP_APPLY, "the second component is dropped, so a mixture is only its first term",
     "            blended = terms[0] + terms[1]",
     "            blended = terms[0]"),
    (GROUP_APPLY, "pure SKIP leaves the advantages in place instead of zeroing them",
     "            blended = torch.zeros_like(block)          # pure SKIP: c * 0",
     "            blended = block          # pure SKIP: c * 0"),
    (GROUP_APPLY, "the mask no longer bounds the write, only its non-zero part",
     "        out[rows] = torch.where(m != 0, blended, block)",
     "        out[rows] = blended"),
    (GROUP_APPLY, "weights are used unnormalised",
     "    return {m: float(w) / total for m, w in mixture.items()}",
     "    return {m: float(w) for m, w in mixture.items()}"),
    (GROUP_APPLY, "the pure-RL short circuit fires on pure SFT instead",
     "        if a == 1.0:",
     "        if b == 1.0:"),
    (GROUP_APPLY, "the RL extreme is computed as SKIP",
     "        [TrainingMode.RL] * n_decisions, sft_weight=sft_weight,",
     "        [TrainingMode.SKIP] * n_decisions, sft_weight=sft_weight,"),
    (GROUP_APPLY, "the SFT extreme is computed as SKIP",
     "        [TrainingMode.SFT] * n_decisions, sft_weight=sft_weight,",
     "        [TrainingMode.SKIP] * n_decisions, sft_weight=sft_weight,"),
    (GROUP_APPLY, "the blend is written into the SFT extreme rather than the RL one",
     "    out = base\n",
     "    out = sft_only\n"),
    # ---- group slicing ---------------------------------------------------------------
    (GROUP_APPLY, "off-by-one: the last row of every group is left out",
     "        rows = slice(start, start + g)",
     "        rows = slice(start, start + g - 1)"),
    (GROUP_APPLY, "group cursor advances one row too far",
     "        start = start + g\n",
     "        start = start + g + 1\n"),
    # ---- counting --------------------------------------------------------------------
    (GROUP_APPLY, "mode mass counted as whole groups",
     "            mass[mode] = mass[mode] + weight",
     "            mass[mode] = mass[mode] + 1.0"),
    (GROUP_APPLY, "mixed_groups counts every group, one-hot included",
     "        if max(w.values()) < 1.0:",
     "        if max(w.values()) <= 1.0:"),
    (GROUP_APPLY, "mixed_groups never counts anything",
     "        if max(w.values()) < 1.0:",
     "        if max(w.values()) < 0.0:"),
    (GROUP_APPLY, "changed_rows always zero",
     "        changed_rows=int((out != advantages).any(dim=-1).sum()),",
     "        changed_rows=0,"),
    (GROUP_APPLY, "changed_rows counts every row of the batch",
     "        changed_rows=int((out != advantages).any(dim=-1).sum()),",
     "        changed_rows=int(advantages.shape[0]),"),
    (GROUP_APPLY, "n_rows reports the group count",
     "        n_rows=int(advantages.shape[0]),",
     "        n_rows=len(group_sizes),"),
    (GROUP_APPLY, "n_groups reports the row count",
     "\n        n_groups=len(group_sizes),\n",
     "\n        n_groups=int(advantages.shape[0]),\n"),
    # ---- validation guards -----------------------------------------------------------
    (GROUP_APPLY, "empty-mixture guard dropped",
     "    if not mixture:",
     "    if False:"),
    (GROUP_APPLY, "unknown-mode guard dropped (a teacher mode becomes a silent SKIP)",
     "    if unknown_modes:",
     "    if False:"),
    (GROUP_APPLY, "negative-weight guard dropped",
     "        if value < 0:",
     "        if value < -1e9:"),
    (GROUP_APPLY, "finite-weight guard dropped",
     "        if not math.isfinite(value):",
     "        if False:"),
    (GROUP_APPLY, "zero-sum guard dropped",
     "    if total <= 0:",
     "    if False:"),
    (GROUP_APPLY, "one-mixture-per-group guard dropped",
     "    if len(mixtures) != len(group_sizes):",
     "    if False:"),
    # ---- the gate in the actor -------------------------------------------------------
    (ACTOR, "the mixture gate never fires",
     '        use_mixture = _decision == "mixture"',
     "        use_mixture = False"),
    (ACTOR, "the mixture gate always fires, whatever the config says",
     '        use_mixture = _decision == "mixture"',
     "        use_mixture = True"),
    (ACTOR, "the decision is collapsed to one-hot before it reaches the seam",
     "            mixtures.append(decision.normalised())",
     "            mixtures.append({decision.argmax(): 1.0})"),
    (ACTOR, "the router is asked twice per unit",
     "            mixtures.append(decision.normalised())",
     "            mixtures.append(router.route(ctx).normalised())"),
    (ACTOR, "the mixture path is fed the argmax modes instead of the weights",
     "                list(sizes),\n                mixtures,",
     "                list(sizes),\n                [{m: 1.0} for m in modes],"),
    (ACTOR, "the mixture path hardcodes the SFT magnitude",
     "                mixtures,\n                sft_weight=float(gr.solved_advantage),",
     "                mixtures,\n                sft_weight=1.0,"),
    # ---- config validation -----------------------------------------------------------
    (CLI_ARGS, "an unknown decision value is silently accepted",
     '        if self.decision not in ("argmax", "mixture"):',
     "        if False:"),
    (CLI_ARGS, "a mixture configured without a router is silently accepted",
     '        if self.decision == "mixture" and not self.router:',
     "        if False:"),
    (CLI_ARGS, "the shipped default is mixture rather than argmax",
     '    decision: str = "argmax"',
     '    decision: str = "mixture"'),
]


def run_tests() -> bool:
    """True if the suite passes.

    Returns:
        Whether pytest exited 0. ``-x`` because a mutant only has to break one assertion to
        count as killed, and the remaining tests cost seconds each.
    """
    r = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x"],
        cwd=REPO, capture_output=True, text=True, timeout=1800,
    )
    return r.returncode == 0


def main() -> int:
    """Run every mutation against a restored baseline and report the score.

    Returns:
        0 if every mutant was killed and every target verified intact, 1 if a mutant
        survived or an anchor was not unique, 2 if the baseline was already red, 3 if a
        target could NOT be restored -- which is the case that must never be silent.
    """
    targets = sorted({rel for rel, *_ in MUTATIONS})
    originals = {rel: (REPO / rel).read_text() for rel in targets}
    digests = {
        rel: hashlib.sha256(text.encode()).hexdigest() for rel, text in originals.items()
    }

    def restore() -> None:
        """Put every target back exactly as it was read at startup."""
        for rel, text in originals.items():
            (REPO / rel).write_text(text)

    def on_signal(signum, _frame):
        """Restore before dying, so an interrupted run cannot poison the next baseline."""
        restore()
        print(f"\ninterrupted by signal {signum}; targets restored")
        raise SystemExit(130)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    survivors: list[tuple[str, str]] = []
    try:
        if not run_tests():
            print("BASELINE IS RED -- mutation results would be meaningless")
            return 2
        print(f"baseline green; {len(MUTATIONS)} mutations over {len(targets)} files\n")

        for rel, label, find, repl in MUTATIONS:
            original = originals[rel]
            n = original.count(find)
            if n != 1:
                print(f"SKIP      [{rel}] {label}: anchor appears {n}x")
                survivors.append((label, f"anchor appears {n}x in {rel}"))
                continue
            (REPO / rel).write_text(original.replace(find, repl, 1))
            try:
                passed = run_tests()
            finally:
                (REPO / rel).write_text(original)
            if passed:
                print(f"SURVIVED  [{rel}] {label}")
                survivors.append((label, "tests still passed"))
            else:
                print(f"killed    [{rel}] {label}")
    finally:
        restore()

    # The verification the module docstring promises. Printed last and unconditionally,
    # because a harness that silently leaves a mutation behind is worse than no harness.
    dirty = [
        rel for rel in targets
        if hashlib.sha256((REPO / rel).read_text().encode()).hexdigest() != digests[rel]
    ]
    print()
    for rel in targets:
        state = "MUTATED" if rel in dirty else "INTACT"
        print(f"{state:8s} {rel}")
    if dirty:
        print("\nRESTORE FAILED -- results are void; run `git checkout` on the files above")
        return 3

    print(f"\n{len(MUTATIONS) - len(survivors)}/{len(MUTATIONS)} killed")
    if survivors:
        print("\nSURVIVORS (the tests do not constrain these):")
        for label, why in survivors:
            print(f"  - {label}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
