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
That has happened in this repo. So: every target is restored in a ``finally``; SIGINT and
SIGTERM are trapped, and the handler kills the pytest child (it runs in its own session, so
a signal sent to this process does not reach it), restores, PRINTS THE DIGEST CHECK, and
only then exits. An earlier version raised ``SystemExit`` from the handler, which sailed
straight past the digest block -- the restore happened but the run could not prove it, and
the orphaned pytest child kept running.
"""
from __future__ import annotations

import hashlib
import os
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
     "            terms.append(b * sft_only[rows])",
     "            terms.append(b * block)"),
    (GROUP_APPLY, "the second component is dropped, so a mixture is only its first term",
     "            blended = terms[0] + terms[1]",
     "            blended = terms[0]"),
    (GROUP_APPLY, "pure SKIP leaves the advantages in place instead of zeroing them",
     "            blended = torch.zeros_like(block)          # pure SKIP: c * 0",
     "            blended = block          # pure SKIP: c * 0"),
    (GROUP_APPLY, "the mask no longer bounds the write, only its non-zero part",
     "        new_block = torch.where(m != 0, blended, block)",
     "        new_block = blended"),
    (GROUP_APPLY, "weights are used unnormalised",
     "    return {m: float(w) / total for m, w in mixture.items()}",
     "    return {m: float(w) for m, w in mixture.items()}"),
    (GROUP_APPLY, "the RL extreme is computed as SKIP",
     "        [TrainingMode.RL] * n_decisions, sft_weight=sft_weight,",
     "        [TrainingMode.SKIP] * n_decisions, sft_weight=sft_weight,"),
    (GROUP_APPLY, "the SFT extreme is computed as SKIP",
     "                    [TrainingMode.SFT] * n_decisions, sft_weight=sft_weight,",
     "                    [TrainingMode.SKIP] * n_decisions, sft_weight=sft_weight,"),
    (GROUP_APPLY, "the blend is written into the SFT extreme rather than the RL one",
     "    out = base\n",
     "    out = sft_only\n"),
    # ---- the two zero-weight guards, which are load-bearing for DIFFERENT reasons ----
    (GROUP_APPLY, "a zero-weighted RL term is included as `0.0 * block` (NaN poisoning)",
     "        if a != 0.0:\n            terms.append(a * block)",
     "        if True:\n            terms.append(a * block)"),
    (GROUP_APPLY, "a zero-weighted SFT term is included (flips -0.0 to +0.0)",
     "        if b != 0.0:\n            if sft_only is None:",
     "        if True:\n            if sft_only is None:"),
    (GROUP_APPLY, "the pure-RL short circuit is removed (NaN rows self-report as changed)",
     "        if a == 1.0:\n            # Pure RL",
     "        if False:\n            # Pure RL"),
    (GROUP_APPLY, "the pure-RL short circuit fires on pure SFT instead",
     "        if a == 1.0:\n            # Pure RL",
     "        if b == 1.0:\n            # Pure RL"),
    (GROUP_APPLY, "the SFT extreme is built eagerly (raises where apply_decisions succeeds)",
     "    sft_only = None\n",
     "    sft_only, _ = apply_decisions(advantages, loss_mask, group_sizes,\n"
     "                                  [TrainingMode.SFT] * n_decisions,\n"
     "                                  sft_weight=sft_weight)[0], None\n"),
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
    (GROUP_APPLY, "changed count inverted",
     "        changed = changed + int((block != new_block).any(dim=-1).sum())",
     "        changed = changed + int((block == new_block).any(dim=-1).sum())"),
    (GROUP_APPLY, "every row of a written group counted, changed or not",
     "        changed = changed + int((block != new_block).any(dim=-1).sum())",
     "        changed = changed + int(block.shape[0])"),
    (GROUP_APPLY, "changed counted AFTER the write, when `block` is a view of the result",
     "        changed = changed + int((block != new_block).any(dim=-1).sum())\n"
     "        out[rows] = new_block",
     "        out[rows] = new_block\n"
     "        changed = changed + int((block != new_block).any(dim=-1).sum())"),
    (GROUP_APPLY, "changed_rows diffed once at the end (diverges on NaN)",
     "        changed_rows=changed,",
     "        changed_rows=int((out != advantages).any(dim=-1).sum()),"),
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
     "    if total <= 0 or not math.isfinite(total):",
     "    if False:"),
    (GROUP_APPLY, "overflowing-sum guard dropped (an inf sum becomes a silent SKIP)",
     "    if total <= 0 or not math.isfinite(total):",
     "    if total <= 0:"),
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
     "            mixtures.append(dict(decision.weights))",
     "            mixtures.append({decision.argmax(): 1.0})"),
    (ACTOR, "the actor normalises too, so a refusal names weights nobody emitted",
     "            mixtures.append(dict(decision.weights))",
     "            mixtures.append(decision.normalised())"),
    (ACTOR, "the router is asked twice per unit",
     "            mixtures.append(dict(decision.weights))",
     "            mixtures.append(dict(router.route(ctx).weights))"),
    (ACTOR, "the mixture path is fed the argmax modes instead of the weights",
     "                list(sizes),\n                mixtures,",
     "                list(sizes),\n                [{m: 1.0} for m in modes],"),
    (ACTOR, "the mixture path hardcodes the SFT magnitude",
     "                mixtures,\n                sft_weight=float(gr.solved_advantage),",
     "                mixtures,\n                sft_weight=1.0,"),
    (ACTOR, "mixed_groups logged on the mixture branch only, so the key sets differ",
     '        stats_tracker.scalar(**{"route/mixed_groups": float(stats.mixed_groups)})',
     '        if use_mixture:\n'
     '            stats_tracker.scalar('
     '**{"route/mixed_groups": float(stats.mixed_groups)})'),
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

_CHILD: subprocess.Popen | None = None


def _kill_child() -> None:
    """Kill the pytest child and its whole session, if one is running.

    It is launched with ``start_new_session=True`` so that a Ctrl-C aimed at this harness
    does not race it into the same terminal; the price is that the child does NOT receive
    signals sent here, so it has to be killed explicitly or it keeps running after the
    harness is gone, holding a checkout that is about to be restored under it.
    """
    child = _CHILD
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(child.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        child.kill()
    try:
        child.wait(timeout=60)
    except subprocess.TimeoutExpired:
        pass


def run_tests() -> bool:
    """True if the suite passes.

    Returns:
        Whether pytest exited 0. ``-x`` because a mutant only has to break one assertion to
        count as killed, and the remaining tests cost seconds each.
    """
    global _CHILD
    _CHILD = subprocess.Popen(
        [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x"],
        cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        return _CHILD.wait(timeout=1800) == 0
    except subprocess.TimeoutExpired:
        _kill_child()
        raise
    finally:
        _CHILD = None


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

    def verify() -> list[str]:
        """Print an INTACT/MUTATED line per target and return the ones that differ."""
        dirty = [
            rel for rel in targets
            if hashlib.sha256((REPO / rel).read_text().encode()).hexdigest() != digests[rel]
        ]
        print()
        for rel in targets:
            print(f"{'MUTATED' if rel in dirty else 'INTACT':8s} {rel}")
        if dirty:
            print("\nRESTORE FAILED -- results are void; run `git checkout` on the above")
        return dirty

    def on_signal(signum, _frame):
        """Reap the child, restore, PROVE the restore, then exit.

        Ordering matters and is the whole point: the proof has to be printed by the same
        code path that does the restoring, or an interrupted run leaves a claim nobody
        checked. ``os._exit`` rather than ``SystemExit`` because SystemExit raised from a
        handler unwinds past this function and past the verification below it.
        """
        _kill_child()
        restore()
        print(f"\ninterrupted by signal {signum}; targets restored")
        verify()
        sys.stdout.flush()
        os._exit(130)

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

    if verify():
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
