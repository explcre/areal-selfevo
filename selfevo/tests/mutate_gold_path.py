#!/usr/bin/env python3
"""Mutation-test selfevo/tests/test_gold_batch_path.py against a COPY of the repo.

A copy and never the live checkout, for the reason every harness here gives: an 8xA100 job
imports this tree through PYTHONPATH, so a mutated production file sitting on disk for even a
few seconds could be read by a real run. Four files are mutated, because the claim under test
spans them -- the dataset adapter that keeps the gold, the workflow that carries it, the
executor seam that serves the proxy path, and the substitution that spends it.

Every target is sha256-compared against the LIVE checkout before the first mutation and again
after the last, so a harness that failed to restore something cannot be mistaken for a clean
run. A mutant that leaves the file byte-identical, or that does not compile, is reported as
SKIPPED and never as evidence either way -- an earlier harness in this repo learned that the
hard way, twice, by counting inert mutants as survivors.

Usage: mutate_gold_path.py <path-to-copy-of-repo>
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

COPY = pathlib.Path(sys.argv[1]).resolve()
LIVE = pathlib.Path(__file__).resolve().parents[2]
TESTS = "selfevo/tests/test_gold_batch_path.py"

ADAPTER = "areal/dataset/competition_math.py"
WORKFLOW = "areal/workflow/rlvr.py"
EXECUTOR = "areal/infra/workflow_executor.py"
ATTACH = "selfevo/gold/attach.py"
SUBST = "selfevo/gold/substitute.py"
TARGETS = [ADAPTER, WORKFLOW, EXECUTOR, ATTACH, SUBST]

# (label, file, find, replace) -- each a single-edit defect a careless change could produce.
MUTATIONS = [
    # -- the dataset flag -----------------------------------------------------------------
    ("the gold column is kept unconditionally, so the default run changes",
     ADAPTER, "        if keep_solution:\n", "        if True:\n"),
    ("the tokenizer guard is dropped, so a gold arm silently gets empty golds",
     ADAPTER, "    if keep_solution and tokenizer is None:", "    if False:"),
    ("the EOS is never appended, so the gold row is the only one that never terminates",
     ADAPTER,
     "            if ids and append_eos and tokenizer.eos_token_id is not None:",
     "            if False:"),
    ("the gold template is ignored, so the seam that adapts to a chat template is dead",
     ADAPTER,
     "            text = gold_template.format(solution=solution) if solution else \"\"",
     "            text = solution"),
    # -- carrying it to the batch ----------------------------------------------------------
    ("the workflow stops attaching, so the gold never reaches a trajectory",
     WORKFLOW, "        if \"gold_ids\" in data:", "        if False:"),
    ("the executor stops attaching, so the proxy path the live runs use gets no gold",
     EXECUTOR,
     "                if traj is not None and \"gold_ids\" in pending_task.data:",
     "                if False:"),
    ("the executor attaches unguarded, so every workflow pays for a key it lacks",
     EXECUTOR,
     "                if traj is not None and \"gold_ids\" in pending_task.data:",
     "                if traj is not None:"),
    # -- the padding that makes collation safe ---------------------------------------------
    ("the gold keeps its natural length, which survives collation and breaks at packing",
     ATTACH,
     "    row_ids = torch.zeros(width, dtype=ids.dtype)\n"
     "    row_mask = torch.zeros(width, dtype=torch.int32)",
     "    width = max(len(flat), 1)\n"
     "    row_ids = torch.zeros(width, dtype=ids.dtype)\n"
     "    row_mask = torch.zeros(width, dtype=torch.int32)"),
    ("the mask counts padding as gold, so every gold reads as full width",
     ATTACH, "        row_mask[: len(flat)] = 1", "        row_mask[:] = 1"),
    ("an over-long gold is truncated instead of refused",
     ATTACH,
     "    if len(flat) > width:\n        raise GoldAttachError(",
     "    if len(flat) > width:\n        flat = flat[:width]\n    if False:\n        raise GoldAttachError("),
    ("the prompt boundary ignores rows with no response",
     ATTACH,
     "    return torch.where(has_response, first, fallback)",
     "    return first"),
    # -- the rules -------------------------------------------------------------------------
    ("DyME's predicate is inverted, so only groups WITH a correct sample get gold",
     SUBST,
     "        return bool((rewards > _CORRECT).sum() == 0)",
     "        return bool((rewards > _CORRECT).sum() != 0)"),
    ("LSPO's cliff is aliased to DyME's predicate, so the two stop being distinct arms",
     SUBST,
     "        return bool(torch.isclose(rewards.sum(), torch.zeros((), dtype=rewards.dtype)))",
     "        return bool((rewards > _CORRECT).sum() == 0)"),
    ("the last row of a group is sacrificed instead of DyME's first",
     SUBST, "        victim = rows.start", "        victim = rows.stop - 1"),
    # -- what the gold row is --------------------------------------------------------------
    ("the old response mask survives, so the row trains on gold AND leftover rollout",
     SUBST, "    lm[row, :] = 0\n    lm[row, prompt_len:end] = 1",
     "    lm[row, prompt_len:end] = 1"),
    ("the attention mask keeps the old tail, so packing reads padding as tokens",
     SUBST,
     "    if end < width:\n        am[row, end:] = torch.zeros((), dtype=am.dtype)",
     "    if False:\n        am[row, end:] = torch.zeros((), dtype=am.dtype)"),
    ("the gold row keeps the wrong rollout's reward, so it gets a negative advantage",
     SUBST, "    out[\"rewards\"][row] = gold_reward", "    pass"),
    ("the prompt is overwritten too, so the target is detached from its question",
     SUBST, "    ids[row, prompt_len:end] = gold.to(ids.dtype)",
     "    ids[row, :n_gold] = gold.to(ids.dtype)"),
    ("is_gold becomes per-row, the shape that does not survive packing",
     SUBST, "    is_gold = torch.zeros((n_rows, width), dtype=torch.int32)",
     "    is_gold = torch.zeros((n_rows, 1), dtype=torch.int32)"),
    # -- the off-policy value --------------------------------------------------------------
    ("the sentinel becomes NaN, the value the loss audit measured destroying a batch",
     SUBST, "GOLD_LOGP_SENTINEL = 1.0", "GOLD_LOGP_SENTINEL = float(\"nan\")"),
    ("the sentinel becomes 0.0, a legal log-probability that cannot be detected",
     SUBST, "GOLD_LOGP_SENTINEL = 1.0", "GOLD_LOGP_SENTINEL = 0.0"),
    ("the unfilled-sentinel guard never fires",
     SUBST, "    bad = (~torch.isfinite(vals)) | (vals > 0)",
     "    bad = (~torch.isfinite(vals)) | (vals > 1)"),
    ("reconcile writes prox_logp unshifted, so every gold ratio is off by one position",
     SUBST, "    shifted = torch.roll(prox, shifts=1, dims=-1).to(data[\"logprobs\"].dtype)",
     "    shifted = prox.to(data[\"logprobs\"].dtype)"),
    ("reconcile shifts the wrong way",
     SUBST, "    shifted = torch.roll(prox, shifts=1, dims=-1).to(data[\"logprobs\"].dtype)",
     "    shifted = torch.roll(prox, shifts=-1, dims=-1).to(data[\"logprobs\"].dtype)"),
    ("reconcile overwrites every row, not only the gold ones",
     SUBST, "    out[\"logprobs\"] = torch.where(gold_tok, shifted, out[\"logprobs\"])",
     "    out[\"logprobs\"] = shifted"),
    ("ratio_one silently reports success instead of refusing",
     SUBST,
     "    if policy is GoldLogprobPolicy.RATIO_ONE:\n        raise GoldPolicyError(",
     "    if False:\n        raise GoldPolicyError("),
    # -- the guards ------------------------------------------------------------------------
    ("the reach guard never fires, which is the silent no-op this path exists to prevent",
     SUBST, "    if qualifying and not substituted:", "    if False:"),
    ("a batch whose golds are all empty is accepted",
     SUBST, "    if int(gold_mask.sum()) == 0:", "    if False:"),
    ("substituting after compute_logp is allowed, corrupting prox_logp silently",
     SUBST, "    if stale:", "    if False:"),
    ("a grouping that does not partition the batch is accepted",
     SUBST,
     "    if not sizes or any(s <= 0 for s in sizes) or sum(sizes) != n_rows:",
     "    if False:"),
    ("the off arm copies the batch instead of returning it",
     SUBST,
     "        return dict(batch), GoldStats(",
     "        return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}, GoldStats("),
    ("the token-mass denominator is measured before substitution",
     SUBST, "        loss_tokens=_loss_tokens(out),", "        loss_tokens=_loss_tokens(batch),"),
    ("a refusal drops its counts, so the list form understates the lost reach",
     SUBST, "            st = exc.stats or GoldStats(", "            st = GoldStats("),
]


def run_tests() -> bool:
    """True if the gold path suite passes against the copy."""
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
         "import selfevo.gold.substitute as g, areal.dataset.competition_math as d;"
         " print(g.__file__); print(d.__file__)"],
        cwd=COPY, capture_output=True, text=True, env=env, timeout=600,
    )
    got = [pathlib.Path(p).resolve() for p in r.stdout.split()]
    want = [COPY / SUBST, COPY / ADAPTER]
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
        if passed:
            print(f"SURVIVED  {label}")
            survivors.append((label, "tests still passed"))
        else:
            print(f"killed    {label}")

    _assert_identical_to_live("at finish")
    killed = len(MUTATIONS) - len(survivors) - len(skipped)
    print(f"\n{killed}/{len(MUTATIONS)} killed, {len(survivors)} survived, "
          f"{len(skipped)} skipped")
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
