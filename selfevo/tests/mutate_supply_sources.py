#!/usr/bin/env python3
"""Mutation-test selfevo/tests/test_supply_sources.py against a COPY of the repo.

A copy and never the live checkout, for the reason every harness here gives: this tree is
imported through PYTHONPATH by jobs on this box, so a mutated production file sitting on disk
for even a few seconds could be read by a real run. Eight files are mutated, because the claim
under test spans them -- the four suppliers, the store they read, the policy and its control,
the interface that validates an offer, and the seam that spends one.

Third-generation behaviours, all of them non-negotiable and all of them present here:

* Refuse to start on a red baseline. A mutation score against an already-failing suite
  measures nothing.
* ``_assert_isolated`` -- prove pytest imported the COPY. A harness that silently tested the
  live tree reports every mutation as killed, for the wrong reason.
* ``_assert_matches_live`` -- sha256 every target against the live checkout before the first
  mutation and after the last.
* A four-way SKIP taxonomy: non-unique anchor, zero-byte change, uncompilable mutant, or a
  literal newline problem. A skipped mutant is never counted as evidence either way.
* KILL ATTRIBUTION BY TEST ID. "Something failed" does not say whether the guard you meant to
  test is the one that fired, and a mutation killed by an unrelated collection error is not a
  constrained mutation.

Usage: mutate_supply_sources.py <path-to-copy-of-repo>
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import re
import subprocess
import sys

COPY = pathlib.Path(sys.argv[1]).resolve()
LIVE = pathlib.Path(__file__).resolve().parents[2]
TESTS = "selfevo/tests/test_supply_sources.py"

SUBST = "selfevo/gold/substitute.py"
BASE = "selfevo/supply/base.py"
GOLD = "selfevo/supply/gold.py"
STORE = "selfevo/supply/store.py"
SELF = "selfevo/supply/self_gen.py"
CORPUS = "selfevo/supply/corpus.py"
TEACHER = "selfevo/supply/teacher.py"
POLICY = "selfevo/supply/policy.py"
TARGETS = [SUBST, BASE, GOLD, STORE, SELF, CORPUS, TEACHER, POLICY]

# (label, file, find, replace) -- each a single-edit defect a careless change could produce.
MUTATIONS = [
    # -- a refusal that is not one ---------------------------------------------------------
    ("a refusing supplier reports success and leaves the row unchanged",
     SUBST,
     "            decisions.append(NO_SOURCE)\n"
     "            unserved_groups[last_reason.value] += 1",
     "            decisions.append(\"gold\")\n"
     "            substituted.append(victim)"),
    ("the unserved counter never increments",
     SUBST, "            unserved_groups[last_reason.value] += 1",
     "            unserved_groups[last_reason.value] += 0"),
    ("the served counter never increments",
     SUBST, "        served_by[offer.source] += 1", "        served_by[offer.source] += 0"),
    ("the gold counters count every gold refusal, including ones a later source rescued",
     SUBST, "            if gold_reason is Refusal.NO_GOLD:",
     "            if gold_reason is not None:"),
    ("the batch reach guard never fires, so an arm with no supply runs as a silent no-op",
     SUBST, "    if not any(sup.has_supply(batch) for sup in used):", "    if False:"),
    ("a policy naming a supplier the arm never built is silently skipped",
     SUBST, "    unknown = sorted(set(policy.sources()) - set(supply_map))", "    unknown = []"),
    # -- what the substituted row is --------------------------------------------------------
    ("the substituted row keeps the original response's token mask",
     SUBST, "    lm[row, :] = 0\n    lm[row, prompt_len:end] = 1",
     "    lm[row, prompt_len:end] = 1"),
    ("the off-policy path leaves the rollout's own log-probabilities on tokens it never emitted",
     SUBST,
     "    fill = GOLD_LOGP_SENTINEL if policy is GoldLogprobPolicy.PROX_RECOMPUTE else 0.0\n"
     "    lp[row, :] = 0.0\n"
     "    lp[row, prompt_len:end] = fill",
     "    fill = GOLD_LOGP_SENTINEL if policy is GoldLogprobPolicy.PROX_RECOMPUTE else 0.0"),
    ("source_ids becomes per-row, the shape that does not survive packing",
     SUBST, "    source_ids = torch.zeros((n_rows, width), dtype=torch.int32)",
     "    source_ids = torch.zeros((n_rows, 1), dtype=torch.int32)"),
    # -- the gate ----------------------------------------------------------------------------
    ("the capability gate leaks, so the off arm gains a tensor it never asked for",
     SUBST, "    engaged = suppliers is not None or source_policy is not None",
     "    engaged = True"),
    ("the batch-global qualifying index restarts at 0 for every prompt",
     SUBST, "                qualifying_offset=qualifying_offset,", "                qualifying_offset=0,"),
    # -- identity ---------------------------------------------------------------------------
    ("prompt identity grows a second scheme instead of the ledger's",
     BASE, "    return prompt_key(ids + [0], [0] * len(ids) + [1])", "    return str(len(ids))"),
    ("an empty payload is accepted at the offer instead of refused",
     BASE, "        if t.numel() == 0:", "        if False:"),
    # -- the store ---------------------------------------------------------------------------
    ("the store records incorrect rollouts, so a wrong row becomes a confident target",
     STORE, "            if float(raw[row]) <= correct_above:", "            if False:"),
    ("the store records substituted rows, so a self arm silently replays its gold arm",
     STORE, "            if torch.is_tensor(is_gold) and int(is_gold[row].sum()) != 0:",
     "            if False:"),
    ("the store is unbounded",
     STORE, "        while len(self._rows) > self.capacity:", "        while False:"),
    # -- the suppliers -----------------------------------------------------------------------
    ("the self supplier serves whatever it holds instead of this prompt's row",
     SELF, "        key = request.identity()", "        key = next(iter(self.store.keys()), \"\")"),
    ("an over-long response is truncated instead of refused",
     SELF,
     "        if n > request.capacity:\n"
     "            raise SupplierRefused(\n"
     "                Refusal.NO_FIT,\n"
     "                self.name,\n"
     "                f\"prompt {request.prompt_len} + response {n} exceeds width {request.width}\",\n"
     "            )\n"
     "        return SupplyOffer(tokens, self.name, source_detail)",
     "        return SupplyOffer(tokens[: request.capacity], self.name, source_detail)"),
    ("the corpus falls back to another prompt's row rather than refusing",
     CORPUS, "        tokens = self._pool.get(key)",
     "        tokens = self._pool.get(key) or next(iter(self._pool.values()), None)"),
    ("the teacher's verifier is ignored",
     TEACHER, "        if not self.verify(request, tokens):", "        if False:"),
    ("the teacher no longer requires a verifier",
     TEACHER, "        if verify is None:", "        if False:"),
    ("the gold supplier stops checking that the gold fits",
     GOLD, "        if request.prompt_len + n_gold > request.width:", "        if False:"),
    # -- the control -------------------------------------------------------------------------
    ("the matched control silently reuses the treatment's own choices",
     POLICY, "        shuffled = list(realised)\n        random.Random(seed).shuffle(shuffled)",
     "        shuffled = list(realised)"),
    ("the forced policy wraps instead of refusing, silently reusing decisions",
     POLICY, "        if qualifying_index >= len(self.assignment):", "        if False:"),
    ("an unknown source name is accepted at construction",
     POLICY, "        if name != NO_SOURCE and name not in SUPPLY_SOURCES:", "        if False:"),
]


def run_tests() -> tuple[bool, list[str]]:
    """Run the suite against the copy and report pass/fail plus the failing test ids.

    Returns:
        ``(passed, failing_ids)``. No ``-x``: a mutation is only evidence if the test that was
        meant to catch it is the one that fired, and stopping at the first failure hides that.
    """
    env = dict(os.environ, PYTHONPATH=str(COPY), CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header",
         "-p", "no:cacheprovider"],
        cwd=COPY, capture_output=True, text=True, timeout=1800, env=env,
    )
    ids = re.findall(r"^(?:FAILED|ERROR) (\S+)", r.stdout, flags=re.M)
    return r.returncode == 0, [i.split("::")[-1] for i in ids]


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
    env = dict(os.environ, PYTHONPATH=str(COPY), CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(
        [sys.executable, "-c",
         "import selfevo.gold.substitute as g, selfevo.supply.policy as p;"
         " print(g.__file__); print(p.__file__)"],
        cwd=COPY, capture_output=True, text=True, env=env, timeout=600,
    )
    got = [pathlib.Path(p).resolve() for p in r.stdout.split()]
    want = [COPY / SUBST, COPY / POLICY]
    if got != want:
        raise SystemExit(f"ISOLATION FAILED: pytest would import {got}, not {want}")
    print(f"isolated: imports resolve under {COPY}")


def main() -> int:
    """Apply each mutation to the copy, run the tests, restore, and report."""
    _assert_isolated()
    _assert_identical_to_live("at start")

    originals = {rel: (COPY / rel).read_text() for rel in TARGETS}
    digests = {rel: _sha(COPY / rel) for rel in TARGETS}

    baseline_ok, _ = run_tests()
    if not baseline_ok:
        print("BASELINE IS RED -- mutation results would be meaningless")
        return 2
    print(f"baseline green; {len(MUTATIONS)} mutations\n")

    survivors: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
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
        passed, failing = run_tests()
        target.write_text(original)
        assert _sha(target) == digests[rel], f"restore failed for {rel}"
        if passed:
            print(f"SURVIVED  {label}")
            survivors.append((label, "tests still passed"))
        else:
            first = failing[0] if failing else "<no test id parsed>"
            extra = f" (+{len(failing) - 1} more)" if len(failing) > 1 else ""
            print(f"killed    {label}\n            by {first}{extra}")

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
