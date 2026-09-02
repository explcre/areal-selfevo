#!/usr/bin/env python3
"""Mutation-test selfevo/tests/test_deepmath_dataset.py against a COPY of the repo.

A copy and never the live checkout, for the reason every harness here gives: an 8xA100 job
imports this tree through PYTHONPATH, so a mutated production file sitting on disk for even a
few seconds could be read by a real run. Two files are mutated -- the adapter, and the dataset
registry whose BRANCH ORDER is itself part of the claim, since ``zwhe99/DeepMath-103K`` satisfies
the MATH branch's own ``"math" in path.lower()`` predicate and would be captured by it.

Every target is sha256-compared against the LIVE checkout before the first mutation and again
after the last, so a harness that failed to restore something cannot be mistaken for a clean
run. A mutant that leaves the file byte-identical, or that does not compile, is reported as
SKIPPED and never as evidence either way.

Usage: mutate_deepmath_dataset.py <path-to-copy-of-repo>
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

COPY = pathlib.Path(sys.argv[1]).resolve()
LIVE = pathlib.Path(__file__).resolve().parents[2]
TESTS = "selfevo/tests/test_deepmath_dataset.py"

ADAPTER = "areal/dataset/deepmath.py"
INIT = "areal/dataset/__init__.py"
TARGETS = [ADAPTER, INIT]

# (label, file, find, replace) -- each a single-edit defect a careless change could produce.
MUTATIONS = [
    # -- the close-tag guard, which is the one defect unique to THIS corpus ----------------
    ("the close-tag guard never fires, so MATH's template trains a doubled </think>",
     ADAPTER, "    if dup > len(probe_texts) // 2:", "    if False:"),
    ("the guard becomes an any-row test, so the ONE genuinely malformed row vetoes the "
     "correct default template",
     ADAPTER, "    if dup > len(probe_texts) // 2:", "    if dup > 0:"),
    ("the guard becomes an all-rows test, so a majority-bad template passes",
     ADAPTER, "    if dup > len(probe_texts) // 2:", "    if dup >= len(probe_texts):"),
    ("the guard counts the OPENING tag, which this corpus strips, so it never fires",
     ADAPTER, '_THINK_CLOSE = "</think>"', '_THINK_CLOSE = "<think>"'),
    ("the guard is computed and then not called",
     ADAPTER, "        _assert_one_think_close(probe, gold_template)", "        pass"),
    ("the probe reads no rows, so the guard is handed an empty sample and abstains",
     ADAPTER, "_PROBE_ROWS = 200", "_PROBE_ROWS = 0"),
    # -- the gold column -------------------------------------------------------------------
    ("the gold column is kept unconditionally, so a default run gains a column",
     ADAPTER, "        if keep_solution:\n", "        if True:\n"),
    ("the tokenizer guard is dropped, so a gold arm silently gets empty golds",
     ADAPTER, "    if keep_solution and tokenizer is None:", "    if False:"),
    ("an unknown solution_field is accepted, yielding an empty gold for every row",
     ADAPTER, "    if solution_field not in SOLUTION_FIELDS:", "    if False:"),
    ("solution_field is ignored, so the three derivations stop being distinct arms",
     ADAPTER,
     '            solution = sample.get(solution_field) or ""',
     '            solution = sample.get("r1_solution_1") or ""'),
    ("the EOS is never appended, so the gold row is the only one that never terminates",
     ADAPTER,
     "            if ids and append_eos and tokenizer.eos_token_id is not None:",
     "            if False:"),
    ("append_eos is ignored, so the seam cannot be turned off",
     ADAPTER,
     "            if ids and append_eos and tokenizer.eos_token_id is not None:",
     "            if ids and tokenizer.eos_token_id is not None:"),
    # -- the answer ------------------------------------------------------------------------
    ("the answer is handed over bare, which self-verifies on only 83.8% of structured golds",
     ADAPTER, '            "answer": _boxed_gold(answer),', '            "answer": answer,'),
    ("the answer is extracted from the trace instead of the corpus's curated field",
     ADAPTER,
     '        answer = (sample.get("final_answer") or "").strip()',
     '        answer = ""'),
    # -- the unanswerable rows -------------------------------------------------------------
    ("unanswerable rows are kept, becoming fake permanent members of the UNSOLVED branch",
     ADAPTER, "    if drop_unanswerable:", "    if False:"),
    ("the unanswerable filter is inverted, keeping only the rows with no answer",
     ADAPTER,
     '        dataset = dataset.filter(lambda s: bool((s["final_answer"] or "").strip()))',
     '        dataset = dataset.filter(lambda s: not bool((s["final_answer"] or "").strip()))'),
    ("an all-empty corpus is accepted rather than refused",
     ADAPTER, "    n_probe = min(_PROBE_ROWS, len(dataset))", "    n_probe = 0"),
    ("the empty-answer schema check never fires",
     ADAPTER, "    if n_gold == 0:", "    if False:"),
    # -- the difficulty filter, which is the whole reason for the corpus switch -------------
    ("the difficulty filter is never applied, so the harder operating point is not selected",
     ADAPTER,
     "    if min_difficulty is not None or max_difficulty is not None:",
     "    if False:"),
    ("the difficulty bound becomes strict, silently dropping every row exactly at the bound",
     ADAPTER,
     '        dataset = dataset.filter(lambda s: lo <= float(s["difficulty"]) <= hi)',
     '        dataset = dataset.filter(lambda s: lo < float(s["difficulty"]) <= hi)'),
    ("a filter that keeps nothing is accepted, so an empty training set reaches the loader",
     ADAPTER,
     '        if len(dataset) == 0:\n            raise ValueError(\n                f"difficulty filter',
     '        if False:\n            raise ValueError(\n                f"difficulty filter'),
    # -- registration, and the branch ORDER --------------------------------------------------
    ("the DeepMath branch is removed, so the corpus falls through to the MATH adapter",
     INIT, '    elif "deepmath" in path.lower() and type == "rl":', "    elif False:"),
    ("the branch predicate becomes case-sensitive, so the real id 'DeepMath' misses it and "
     "the MATH branch below captures the corpus -- the exact ordering hazard",
     INIT, '    elif "deepmath" in path.lower() and type == "rl":',
     '    elif "deepmath" in path and type == "rl":'),
    ("the branch stops gating on type, so an sft request silently receives RL rows",
     INIT, '    elif "deepmath" in path.lower() and type == "rl":',
     '    elif "deepmath" in path.lower():'),
    ("deepmath is dropped from VALID_DATASETS, so the error message hides the adapter",
     INIT, '    "gsm8k",\n    "deepmath",\n', '    "gsm8k",\n'),
]


def run_tests() -> bool:
    """True if the DeepMath suite passes against the copy."""
    env = dict(os.environ, PYTHONPATH=str(COPY))
    r = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x",
         "-p", "no:cacheprovider"],
        cwd=COPY, capture_output=True, text=True, timeout=3600, env=env,
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
         "import areal.dataset as i, areal.dataset.deepmath as d;"
         " print(i.__file__); print(d.__file__)"],
        cwd=COPY, capture_output=True, text=True, env=env, timeout=600,
    )
    got = [pathlib.Path(p).resolve() for p in r.stdout.split()]
    want = [COPY / INIT, COPY / ADAPTER]
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
