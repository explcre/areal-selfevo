#!/usr/bin/env python3
"""Mutation test for the BIRD execution-accuracy grader.

Usage::

    python3 mutate_sql_bench.py                 # copies experiments/bench/ itself
    python3 mutate_sql_bench.py <bench-copy> [<live-bench-dir>]

Green tests are not evidence that a guard is constrained; this project has shipped 350
passing tests that let five real defects through. So each guard that decides a BIRD score is
broken on purpose, one at a time, and the suite must go red. The mutants are the plausible
ways this grader could silently lie: the multiset rule collapsing to a set, the ordering rule
disappearing, a comparison that stops looking at row values, an execution error or a timeout
scoring as a correct answer, a failure quietly leaving the denominator, the read-only URI
losing its ``mode=ro``, the ATTACH authorizer not being installed, and an undecidable
comparison being decided anyway.

**A copy, not the live checkout, and the reason is on the box.** Six agents write to
``~/areal-selfevo`` at once and a training run imports it, so a mutated ``sql_bench.py``
sitting on disk for even a few seconds is a hazard. This harness goes one step further than
``mutate_harness_selectors.py``: given no argument it MAKES the copy itself in a temporary
directory, and given one it REFUSES to run if the path resolves inside the live checkout.
The first-generation harnesses in this tree set ``ROOT = Path.home()/"areal-selfevo"`` and
mutated it directly; that is the hazard being designed out, and passing the live path by hand
is the one remaining way to reintroduce it.

**What counts as evidence, and what does not.** A mutation whose anchor was not unique, whose
replacement left the bytes unchanged, whose text does not ``compile()``, or whose replacement
contains a literal backslash-n, has not been TESTED. All four are reported in their own
column. A SKIP is never a SURVIVED and never a KILLED: a harness that scores them as kills
reports a number higher than the truth, which is worse than reporting nothing. The
uncompilable case matters most here, because a ``SyntaxError`` fails every test in the suite
and would otherwise be recorded as the strongest kill in the table while proving nothing.

**Deliberately absent, because they are EQUIVALENT mutants.**

* Removing the pruning in ``_column_candidates`` (returning every column as a candidate for
  every column) changes only how many permutations are walked, never which comparisons
  succeed: the true permutation is in both candidate sets. It would alter bytes, compile, run
  and survive, and scoring it a survivor would report a no-op as a gap in the tests.
* Deleting the ``if not gold_rows and not pred_rows`` early return is nearly equivalent -- the
  code below reaches the same verdict for two empty results -- and where it is NOT equivalent
  it raises ``IndexError`` rather than mis-grading. The reachable defect in the same place,
  dropping the column-count guard that protects the same indexing, is in the table and is
  killed.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile

LIVE_DEFAULT = pathlib.Path(__file__).resolve().parent
TARGET = "sql_bench.py"
TESTS = ["test_sql_bench.py"]
BACKSLASH_N = chr(92) + "n"

# (id, description, find, replace). Every anchor was confirmed to occur exactly once.
MUTATIONS = [
    # ------------------------------------------------------------------- the comparator
    ("C01", "rows compare as a SET, so duplicate rows stop mattering (BIRD's own rule)",
     '    return collections.Counter(gold_rows) == collections.Counter(permuted)',
     '    return set(gold_rows) == set(permuted)'),
    ("C02", "the unordered comparison looks only at row COUNT, not at row values",
     '    return collections.Counter(gold_rows) == collections.Counter(permuted)',
     '    return len(gold_rows) == len(permuted)'),
    ("C03", "the ordered comparison looks only at row COUNT, not at row values",
     '        return list(gold_rows) == permuted',
     '        return len(gold_rows) == len(permuted)'),
    ("C04", "cells are sorted WITHIN each row, so any per-row anagram is accepted",
     '    permuted = [tuple(r[j] for j in perm) for r in pred_rows]',
     '    permuted = [tuple(sorted(r, key=repr)) for r in pred_rows]\n'
     '    gold_rows = [tuple(sorted(r, key=repr)) for r in gold_rows]'),
    ("C05", "the column count is not compared, so a wider or narrower answer is considered",
     '    if len(pred_rows[0]) != ncols:',
     '    if False:'),
    ("C06", "the row count is not compared, which also removes the guard on the indexing below",
     '    if len(gold_rows) != len(pred_rows):',
     '    if False:'),
    ("C07", "exhausting the permutation budget is reported as a decided verdict",
     '            return False, True',
     '            return False, False'),
    ("C08", "official_equal stops being BIRD's function, so accuracy_official is not comparable",
     '    return set(gold_rows) == set(pred_rows)',
     '    return results_equal(gold_rows, pred_rows, False)[0]'),

    # ---------------------------------------------------------------------- the ordering
    ("O01", "row order never matters, so a top-N answer ranked backwards passes",
     '    return _ORDER_BY.search(_top_level_only(_blank_noise(gold_sql))) is not None',
     '    return False'),
    ("O02", "row order always matters, so a correct unordered answer is graded wrong",
     '    return _ORDER_BY.search(_top_level_only(_blank_noise(gold_sql))) is not None',
     '    return True'),
    ("O03", "nesting is ignored, so a subquery's ORDER BY imposes an order on the outer rows",
     '            out.append(ch if depth == 0 else " ")',
     '            out.append(ch)'),

    # ----------------------------------------------------------------------- the verdicts
    ("V01", "an execution error is scored as a CORRECT answer",
     '        rec["status"] = ST_EXEC_ERROR',
     '        rec["status"] = ST_PASS\n        rec["passed"] = True'),
    ("V02", "an execution error is not recognised and degrades into a wrong answer",
     '    if pred.status == "error":',
     '    if False and pred.status == "error":'),
    ("V03", "a timeout is scored as a CORRECT answer",
     '        rec["status"] = ST_TIMEOUT',
     '        rec["status"] = ST_PASS\n        rec["passed"] = True'),
    ("V04", "a timeout is not recognised and degrades into a wrong answer",
     '    if pred.status == "timeout":',
     '    if False and pred.status == "timeout":'),
    ("V05", "a completion with no SQL in it is scored as a CORRECT answer",
     '        rec["status"] = ST_NO_SQL',
     '        rec["status"] = ST_PASS\n        rec["passed"] = True'),
    ("V06", "a broken dataset row is scored as a CORRECT answer",
     '        rec["status"] = ST_GOLD_BROKEN',
     '        rec["status"] = ST_PASS\n        rec["passed"] = True'),
    ("V07", "a gold over the row cap is compared anyway, deciding an undecidable comparison",
     '    if gold.truncated:',
     '    if False:'),

    # --------------------------------------------------------------------- the accounting
    ("A01", "unparseable completions silently leave the denominator",
     '    n_graded = n_problems - counts[ST_GEN_FAILED]',
     '    n_graded = n_problems - counts[ST_GEN_FAILED] - counts[ST_NO_SQL]'),
    ("A02", "accuracy_all repeats accuracy, so exclusions stop being visible",
     '        "accuracy_all": (n_pass / n_problems) if n_problems else float("nan"),',
     '        "accuracy_all": acc,'),
    ("A03", "a misaligned prediction list grades every question against the wrong answer",
     '    if len(problems) != len(sqls):',
     '    if False:'),
    ("A04", "non-finite floats are scrubbed only at the top level of the results row",
     '        return {k: json_safe(v) for k, v in obj.items()}',
     '        return dict(obj)'),

    # ------------------------------------------------------------------------ the fixture
    ("F01", "the database is opened READ-WRITE, so a generated query can modify the fixture",
     '    return "file:" + urllib.request.pathname2url(str(path)) + "?mode=ro"',
     '    return "file:" + urllib.request.pathname2url(str(path))'),
    ("F02", "the authorizer is not installed, so ATTACH can reach a second file",
     '    conn.set_authorizer(auth)',
     '    pass'),

    # --------------------------------------------------------------- extraction and load
    ("E01", "any completion is submitted as SQL, so prose reaches the database",
     '    if _BARE_START.match(text):',
     '    if True:'),
    ("L01", "a blank gold SQL loads instead of being fatal",
     '        if not str(r["SQL"]).strip():',
     '        if False:'),
    ("L02", "an unknown difficulty label loads, and drops out of the per-difficulty table",
     '        if diff and diff not in DIFFICULTIES:',
     '        if False:'),
]


# The guard each mutant is AIMED at. A mutant killed only by some other test is still a
# kill, but it is not evidence that the assertion written for that defect constrains it --
# and this table exists because the first run of this harness credited the set-vs-multiset
# mutant to the permutation-budget test and both ordering mutants to the fixture PREMISE
# test, purely because `-x` stopped at whichever test ran first. Every entry below is
# checked, and a mutant whose intended guard stays green is reported as UNAIMED even though
# the suite went red.
EXPECTED_GUARD = {
    "C01": "test_duplicate_rows_count_which_is_where_this_differs_from_bird",
    "C02": "test_row_values_are_compared_not_just_shapes",
    "C03": "test_row_order_is_significant_when_the_gold_orders",
    "C04": "test_a_column_permutation_must_permute_every_row_the_same_way",
    "C05": "test_a_different_number_of_rows_or_columns_is_wrong",
    "C06": "test_two_empty_results_are_equal_and_that_is_bird_s_rule",
    "C07": "test_the_permutation_budget_reports_exhaustion_instead_of_grading_wrong",
    "C08": "test_official_equal_is_bird_s_function_and_nothing_else",
    "O01": "test_a_reordered_answer_fails_when_the_gold_ordered_it",
    "O02": "test_an_equivalent_rewrite_passes",
    "O03": "test_an_order_by_inside_a_subquery_does_not_order_the_result",
    "V01": "test_a_query_that_errors_is_wrong_not_an_exclusion",
    "V02": "test_a_query_that_errors_is_wrong_not_an_exclusion",
    "V03": "test_a_query_that_hangs_is_wrong_and_bucketed_as_a_timeout",
    "V04": "test_a_query_that_hangs_is_wrong_and_bucketed_as_a_timeout",
    "V05": "test_a_missing_query_is_a_fail_not_an_exclusion",
    "V06": "test_a_broken_gold_is_named_as_a_broken_row_not_charged_to_the_model",
    "V07": "test_a_gold_over_the_row_cap_is_undecidable_not_wrong",
    "A01": "test_failures_stay_in_the_denominator",
    "A02": "test_only_generation_failures_leave_the_denominator_and_both_numbers_"
           "are_reported",
    "A03": "test_grade_many_rejects_mismatched_lengths",
    "A04": "test_json_safe_scrubs_at_every_depth",
    "F01": "test_the_database_is_opened_read_only",
    "F02": "test_attach_is_denied_so_a_query_cannot_reach_a_second_file",
    "E01": "test_prose_is_not_a_query",
    "L01": "test_an_empty_gold_sql_is_fatal_at_load",
    "L02": "test_an_unknown_difficulty_label_is_fatal",
}


def _sha(p: pathlib.Path) -> str:
    """sha256 of a file's bytes."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_tests(repo: pathlib.Path) -> tuple:
    """Run the BIRD suite against ``repo`` and collect EVERY failing test id.

    Deliberately not ``-x``. Stopping at the first failure attributes a kill to whichever
    test happens to run earliest, which is a fact about file order rather than about the
    guard: the first version of this harness credited the set-vs-multiset mutant to the
    permutation-budget test and the ordering mutants to the fixture PREMISE test. Both were
    real failures and neither was the assertion written to catch that defect. Collecting the
    whole set lets each mutant be checked against the guard it was aimed at.

    Args:
        repo: The copied bench directory under test.

    Returns:
        ``(passed, {failing test ids})``.
    """
    env = dict(os.environ, PYTHONPATH=str(repo))
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header",
         "-p", "no:cacheprovider"],
        cwd=repo, capture_output=True, text=True, timeout=3600, env=env,
    )
    who = {line.split(" ")[1].split("::")[-1]
           for line in r.stdout.splitlines()
           if line.startswith("FAILED ") or line.startswith("ERROR ")}
    return r.returncode == 0, who


def _assert_isolated(repo: pathlib.Path) -> None:
    """Refuse to run unless pytest imports the COPY, not any other checkout.

    A harness that silently tested the live tree would report every mutation as killed, for
    the wrong reason, and would report it while the live tree was mutated.

    Args:
        repo: The copied bench directory.

    Raises:
        SystemExit: If the import does not resolve inside ``repo``.
    """
    env = dict(os.environ, PYTHONPATH=str(repo))
    r = subprocess.run(
        [sys.executable, "-c", "import sql_bench; print(sql_bench.__file__)"],
        cwd=repo, capture_output=True, text=True, env=env, timeout=600,
    )
    lines = [pathlib.Path(x.strip()).resolve()
             for x in r.stdout.splitlines() if x.strip()]
    want = [(repo / TARGET).resolve()]
    if lines != want:
        raise SystemExit(
            f"ISOLATION FAILED: import resolves to {lines}, not {want}\n{r.stderr}")
    print(f"isolated: sql_bench resolves inside {repo}")


def _assert_matches_live(repo: pathlib.Path, live) -> None:
    """Assert the copy is byte-identical to the live checkout, so the anchors are real.

    Without this the harness would be mutating a re-derivation, and a kill would say nothing
    about the file a scoring run actually imports.

    Args:
        repo: The copied bench directory.
        live: The live bench directory, or ``None``.

    Raises:
        SystemExit: On any divergence.
    """
    if live is None:
        print("no live directory given; skipping byte-identity check")
        return
    a, b = _sha(repo / TARGET), _sha(live / TARGET)
    if a != b:
        raise SystemExit(f"COPY DIVERGED from live at {TARGET}: {a} != {b}")
    print(f"copy is byte-identical to {live / TARGET}")


def main() -> int:
    """Apply every mutation to a copy and report kills, survivors and skips separately."""
    live = LIVE_DEFAULT
    tmp = None
    if len(sys.argv) > 1:
        repo = pathlib.Path(sys.argv[1]).resolve()
        live = pathlib.Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else LIVE_DEFAULT
        if repo == live or live in repo.parents:
            raise SystemExit(
                f"REFUSED: {repo} is inside the live checkout {live}. This harness writes "
                f"mutated source to disk, and a training run imports that tree. Pass a copy, "
                f"or pass no argument and one will be made."
            )
    else:
        tmp = tempfile.mkdtemp(prefix="mutate_sql_bench_")
        repo = pathlib.Path(tmp) / "bench"
        shutil.copytree(live, repo,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        print(f"copied {live} -> {repo}")

    src_path = repo / TARGET
    original = src_path.read_text()
    digest = _sha(src_path)

    def _restore(*_):
        """Put the source back and leave, however this process is asked to stop."""
        src_path.write_text(original)
        print(f"\ninterrupted; restored {TARGET}: sha256 matches = "
              f"{_sha(src_path) == digest}", flush=True)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        os._exit(130)

    signal.signal(signal.SIGINT, _restore)
    signal.signal(signal.SIGTERM, _restore)

    try:
        _assert_isolated(repo)
        _assert_matches_live(repo, live)

        ok, _ = run_tests(repo)
        if not ok:
            print("BASELINE IS RED -- mutation results would be meaningless")
            return 2
        print(f"baseline green; {len(MUTATIONS)} mutations\n")

        killed, survivors, skips, unaimed = [], [], [], []
        for label, why, find, repl in MUTATIONS:
            if BACKSLASH_N in find or BACKSLASH_N in repl:
                skips.append((label, why, "literal backslash-n in the mutation text"))
                print(f"SKIP      {label}: literal backslash-n")
                continue
            n = original.count(find)
            if n != 1:
                skips.append((label, why, f"anchor appears {n}x"))
                print(f"SKIP      {label}: anchor appears {n}x")
                continue
            mutated = original.replace(find, repl, 1)
            if mutated == original:
                skips.append((label, why, "replacement left the file byte-identical"))
                print(f"SKIP      {label}: file byte-identical")
                continue
            try:
                compile(mutated, str(src_path), "exec")
            except SyntaxError as exc:
                skips.append((label, why, f"mutant does not compile ({exc.msg})"))
                print(f"SKIP      {label}: mutant does not compile")
                continue

            src_path.write_text(mutated)
            try:
                passed, who = run_tests(repo)
            finally:
                src_path.write_text(original)
                assert _sha(src_path) == digest, f"restore failed for {TARGET}"

            if passed:
                survivors.append((label, why))
                print(f"SURVIVED  {label}: {why}")
            else:
                aimed = EXPECTED_GUARD.get(label)
                hit = aimed in who
                killed.append((label, why, aimed, hit, sorted(who)))
                if hit:
                    print(f"killed    {label}: {why}  <- {aimed}")
                else:
                    unaimed.append((label, why, aimed, sorted(who)))
                    print(f"UNAIMED   {label}: {why}  <- red, but {aimed} stayed GREEN; "
                          f"failures were {sorted(who)[:4]}")

        assert _sha(src_path) == digest, "final restore check failed"
        _assert_matches_live(repo, live)

        print(f"\n{len(killed)} killed ({len(killed) - len(unaimed)} by the guard aimed "
              f"at them), {len(survivors)} survived, {len(skips)} skipped "
              f"({len(MUTATIONS)} total)")
        if unaimed:
            print("\nUNAIMED (the suite went red, but not at the assertion written for "
                  "this defect -- add one that fires here):")
            for label, why, aimed, got in unaimed:
                print(f"  - {label} {why}: expected {aimed}, got {got[:4]}")
        if skips:
            print("\nSKIPPED (not applied, so not evidence either way):")
            for label, why, reason in skips:
                print(f"  - {label} {why}: {reason}")
        if survivors:
            print("\nSURVIVORS (the tests do not constrain these):")
            for label, why in survivors:
                print(f"  - {label}: {why}")
        missing = sorted(set(m[0] for m in MUTATIONS) - set(EXPECTED_GUARD))
        if missing:
            print(f"\nNO EXPECTED GUARD DECLARED for {missing}")
        return 1 if (survivors or unaimed or missing) else 0
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
