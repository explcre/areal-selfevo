#!/usr/bin/env python3
"""Mutation-test selfevo/tests/test_routing_stabilisers.py against a COPY of the repo.

A copy, not the live checkout: ``actor.py`` and ``group_apply.py`` are imported by real
training processes and the supervisor relaunches on process exit, so a mutated file sitting
on disk for even a few seconds could be picked up by a run.

Usage: ``python3 selfevo/tests/mutate_routing_stabilisers.py /path/to/repo/copy``

Three files are mutated rather than one, because each stabiliser is several edits that can
be individually wrong and each look fine on its own:

    areal/trainer/ppo/actor.py           the truncation signal, the re-centring, the gates
    selfevo/integration/group_apply.py   the per-row veto on the SFT constant
    areal/api/cli_args.py                the defaults, which decide what an unedited config
                                         is now running

RESTORATION IS THE POINT OF THE STRUCTURE HERE, and it is copied deliberately from
mutate_group_mixture.py rather than reinvented. A mutation harness that is interrupted
leaves its target MUTATED on disk, and the next run then treats the mutation as its
baseline -- the defect is now "original" and every mutant built on top of it looks killed.
That has happened in this repo. So: every target is restored in a ``finally``; SIGINT and
SIGTERM are trapped, and the handler kills the pytest child (it runs in its own session, so
a signal sent to this process does not reach it), restores, PRINTS THE DIGEST CHECK, and
only then exits.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import signal
import subprocess
import sys

REPO = pathlib.Path(sys.argv[1])
TESTS = "selfevo/tests/test_routing_stabilisers.py"

GROUP_APPLY = "selfevo/integration/group_apply.py"
ACTOR = "areal/trainer/ppo/actor.py"
CLI_ARGS = "areal/api/cli_args.py"

# (file, label, find, replace) -- each is a single defect a careless edit could produce.
MUTATIONS = [
    # ---- zero_mean: what is averaged --------------------------------------------------
    (ACTOR, "zero_mean subtracts the UNMASKED mean over the whole tensor",
     "    before = float(advantages[keep].to(torch.float64).mean())",
     "    before = float(advantages.to(torch.float64).mean())"),
    (ACTOR, "zero_mean re-centres against the TOKEN mask instead of the loss's own mask",
     "                    recentred, adv_mean_before, adv_mean_after = _recentre_advantages(\n"
     "                        advantages, loss_mask\n"
     "                    )",
     "                    recentred, adv_mean_before, adv_mean_after = _recentre_advantages(\n"
     "                        advantages, data[\"loss_mask\"]\n"
     "                    )"),
    (ACTOR, "zero_mean centres per GROUP instead of per batch",
     "                    recentred, adv_mean_before, adv_mean_after = _recentre_advantages(\n"
     "                        advantages, loss_mask\n"
     "                    )",
     "                    _g0 = _recentre_advantages(advantages[:sizes[0]], loss_mask[:sizes[0]])[0]\n"
     "                    _g1 = _recentre_advantages(advantages[sizes[0]:], loss_mask[sizes[0]:])[0]\n"
     "                    recentred = torch.cat([_g0, _g1], dim=0)\n"
     "                    adv_mean_before = adv_mean_after = 0.0"),
    # ---- zero_mean: the arithmetic ----------------------------------------------------
    (ACTOR, "the offset is ADDED instead of subtracted, doubling it",
     "    out = torch.where(keep, advantages - shift, advantages)",
     "    out = torch.where(keep, advantages + shift, advantages)"),
    (ACTOR, "the shift is applied everywhere, including positions the loss never reads",
     "    out = torch.where(keep, advantages - shift, advantages)",
     "    out = advantages - shift"),
    (ACTOR, "the shift is scaled per position, so relative differences do not survive",
     "    out = torch.where(keep, advantages - shift, advantages)",
     "    out = torch.where(keep, advantages * (1.0 - shift), advantages)"),
    (ACTOR, "an empty mask writes NaN across the batch instead of being a no-op",
     "    if not bool(keep.any()):",
     "    if False:"),
    # ---- zero_mean: the gate ----------------------------------------------------------
    (ACTOR, "zero_mean never fires, whatever the config says",
     '                    if getattr(gr, "zero_mean", False):',
     "                    if False:"),
    (ACTOR, "zero_mean always fires, whatever the config says",
     '                    if getattr(gr, "zero_mean", False):',
     "                    if True:"),
    (ACTOR, "the re-centred tensor is computed and then discarded",
     '                        advantages = data["advantages"] = recentred',
     '                        data["advantages"] = advantages'),
    (ACTOR, "the corrected mean is reported as 0.0 rather than measured",
     "    after = float(out[keep].to(torch.float64).mean())",
     "    after = 0.0"),
    (ACTOR, "the uncorrected arm reports 0.0 instead of the offset it is still carrying",
     "                        adv_mean_after = adv_mean_before",
     "                        adv_mean_after = 0.0"),
    (ACTOR, "the offset is only logged when the correction ran, so the arms differ",
     "                    stats_tracker.scalar(\n"
     '                        **{\n'
     '                            "route/adv_mean_before": adv_mean_before,\n'
     '                            "route/adv_mean_after": adv_mean_after,\n'
     "                        }\n"
     "                    )",
     '                    if getattr(gr, "zero_mean", False):\n'
     "                        stats_tracker.scalar(\n"
     '                            **{\n'
     '                                "route/adv_mean_before": adv_mean_before,\n'
     '                                "route/adv_mean_after": adv_mean_after,\n'
     "                            }\n"
     "                        )"),
    # ---- the truncation signal --------------------------------------------------------
    (ACTOR, "truncation measured against the padded width, i.e. the defect being replaced",
     "    return loss_mask.sum(dim=-1) >= int(max_new_tokens)",
     "    return loss_mask.sum(dim=-1) >= loss_mask.shape[-1]"),
    (ACTOR, "a response of exactly the cap is not counted as truncated",
     "    return loss_mask.sum(dim=-1) >= int(max_new_tokens)",
     "    return loss_mask.sum(dim=-1) > int(max_new_tokens)"),
    (ACTOR, "truncation keyed on the wrong side of the comparison",
     "    return loss_mask.sum(dim=-1) >= int(max_new_tokens)",
     "    return loss_mask.sum(dim=-1) <= int(max_new_tokens)"),
    (ACTOR, "a missing cap silently reports that nothing is truncated",
     "    if max_new_tokens is None or int(max_new_tokens) < 1:",
     "    if False:"),
    # ---- the veto: the gate in the actor ----------------------------------------------
    (ACTOR, "exclude_truncated excludes NOTHING, whatever the config says",
     '                    if getattr(gr, "exclude_truncated_from_sft", False):\n'
     "                        sft_rows = ~truncated",
     '                    if getattr(gr, "exclude_truncated_from_sft", False):\n'
     "                        sft_rows = torch.ones_like(truncated)"),
    (ACTOR, "exclude_truncated excludes EVERYTHING",
     '                    if getattr(gr, "exclude_truncated_from_sft", False):\n'
     "                        sft_rows = ~truncated",
     '                    if getattr(gr, "exclude_truncated_from_sft", False):\n'
     "                        sft_rows = torch.zeros_like(truncated)"),
    (ACTOR, "the veto is inverted: terminated rows are excluded and truncated ones kept",
     "                        sft_rows = ~truncated",
     "                        sft_rows = truncated"),
    (ACTOR, "the flag is ignored, so every routed run excludes truncated rows",
     '                    if getattr(gr, "exclude_truncated_from_sft", False):',
     "                    if True:"),
    (ACTOR, "the veto never reaches the router branch",
     "                        gr, data, raw_reward, advantages, sizes, sft_rows=sft_rows",
     "                        gr, data, raw_reward, advantages, sizes"),
    (ACTOR, "the veto never reaches the fixed-rule branch",
     "                            solved_rows = solved_rows * sft_rows.to(row_adv.dtype)",
     "                            solved_rows = solved_rows"),
    (ACTOR, "the fixed rule counts every solved row as excluded, not just the truncated ones",
     "                            sft_excluded = int(\n"
     "                                ((solved_rows != 0) & (~sft_rows)).sum()\n"
     "                            )",
     "                            sft_excluded = int((solved_rows != 0).sum())"),
    (ACTOR, "the exclusion count is not logged on the fixed-rule branch",
     "                    stats_tracker.scalar(\n"
     '                        **{"route/sft_excluded_rows": float(sft_excluded)}\n'
     "                    )",
     "                    pass"),
    (ACTOR, "the exclusion count is not logged on the router branch",
     "        stats_tracker.scalar(\n"
     '            **{"route/sft_excluded_rows": float(stats.sft_excluded_rows)}\n'
     "        )",
     "        pass"),
    (ACTOR, "the truncation rate is not logged",
     "                    stats_tracker.scalar(\n"
     '                        **{\n'
     '                            "route/truncated_row_fraction": float(\n'
     "                                truncated.float().mean()\n"
     "                            )\n"
     "                        }\n"
     "                    )",
     "                    pass"),
    # ---- the veto: the seam -----------------------------------------------------------
    (GROUP_APPLY, "the veto is not applied at all",
     "        if allow is not None and mode == TrainingMode.SFT:",
     "        if False:"),
    (GROUP_APPLY, "the veto is inverted, withholding from exactly the wrong rows",
     "            new = torch.where(row_ok, new, before)",
     "            new = torch.where(row_ok, before, new)"),
    (GROUP_APPLY, "the veto also zeroes SKIP groups, which is a second intervention",
     "        if allow is not None and mode == TrainingMode.SFT:",
     "        if allow is not None:"),
    (GROUP_APPLY, "an excluded row is SKIPPED rather than left alone",
     "            new = torch.where(row_ok, new, before)",
     "            new = torch.where(row_ok, new, torch.zeros_like(before))"),
    (GROUP_APPLY, "excluded rows are never counted",
     "            excluded += int((~row_ok).sum())",
     "            excluded += 0"),
    (GROUP_APPLY, "the excluded count is inverted, reporting the rows that WERE written",
     "            excluded += int((~row_ok).sum())",
     "            excluded += int(row_ok.sum())"),
    (GROUP_APPLY, "the shape guard on sft_rows is dropped",
     "        if sft_rows.dim() != 1 or int(sft_rows.shape[0]) != b:",
     "        if False:"),
    (GROUP_APPLY, "the veto is reshaped along the wrong axis, so it vetoes by COLUMN",
     "        allow = sft_rows.to(torch.bool).reshape(-1, 1)",
     "        allow = sft_rows.to(torch.bool).reshape(1, -1)"),
    (GROUP_APPLY, "apply_mixtures does not forward the veto to the SFT extreme",
     "                    [TrainingMode.SFT] * n_decisions, sft_weight=sft_weight,\n"
     "                    sft_rows=sft_rows,",
     "                    [TrainingMode.SFT] * n_decisions, sft_weight=sft_weight,"),
    (GROUP_APPLY, "apply_mixtures does not count the rows it withheld from",
     "                excluded = excluded + int((~allow[rows]).sum())",
     "                excluded = excluded + 0"),
    # ---- the defaults, which decide what an unedited config now runs -------------------
    (CLI_ARGS, "zero_mean defaults to True",
     "    zero_mean: bool = False",
     "    zero_mean: bool = True"),
    (CLI_ARGS, "exclude_truncated_from_sft defaults to True",
     "    exclude_truncated_from_sft: bool = False",
     "    exclude_truncated_from_sft: bool = True"),
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
        [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x",
         "-p", "no:cacheprovider"],
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
