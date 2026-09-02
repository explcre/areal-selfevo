#!/usr/bin/env python3
"""Verify the BIRD GRADER, not the model: gold must pass, wrong must fail.

Companion to ``lcb_selfcheck.py`` and the same idea: a grader is only worth its output if
BOTH directions have been measured. This script runs five checks and exits non-zero if any
deviates, so it can be run as a gate rather than read as a report. It touches no GPU and no
endpoint.

1. GOLD SELF-CHECK  -- every gold SQL, run through :func:`sql_bench.grade_prediction`
   against its own database. It must pass. BIRD ships the reference answer, so unlike
   LiveCodeBench this is available directly and there is no substitute to argue about.
   Any gold that does not self-verify is a BROKEN ROW and is listed, not dropped: published
   work documents pervasive annotation errors in BIRD, so the count is measured here rather
   than assumed.

   It also counts EMPTY golds, which self-verify but are ungradeable in practice: against a
   gold returning no rows, every prediction that returns nothing -- including ``WHERE 1=0``
   and a query that selects the wrong column entirely -- scores correct. A benchmark can be
   silently unable to distinguish a right answer from a wrong one, and this is what that
   looks like.

2. EQUIVALENCE  -- gold self-checking against gold is nearly vacuous on its own: the same
   bytes appear on both sides and it passes under any REFLEXIVE comparator, including one
   that ignores row values entirely. So each order-insensitive gold is also graded against a
   REWRITTEN query returning the same rows in a different order, and those must pass too.
   The check reports how many of the rewrites actually changed the row sequence, because a
   rewrite that returned the same order would have proved nothing.

3. KNOWN-WRONG BATTERY  -- deliberately wrong predictions, each of which must FAIL. A
   grader that passes everything is worse than no grader. Two families are chosen because
   they separate this grader from BIRD's own: ``duplicated_rows`` (gold UNION ALL gold) is
   wrong here and CORRECT under ``calculate_ex``'s set comparison, and ``reversed_order``
   is wrong here on an ORDER BY gold and correct under a grader that ignores order. If
   either of those stops failing, the multiset and ordering rules have stopped existing.

4. READ-ONLY  -- a write, a DDL and an ATTACH are each rejected, and the smallest database
   is sha256'd before and after the whole battery to show the 1.4 GiB fixture was not
   modified by anything that ran.

5. TIMEOUT  -- an unbounded recursive query is graded as ``timeout``, in bounded wall-clock
   time, and does not hang or abort the run.

Usage::

    python3 sql_selfcheck.py                      # bird_mini_dev
    python3 sql_selfcheck.py --benchmarks bird_dev,bird_mini_dev
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sql_bench as sb  # noqa: E402


def _grade_all(problems, sqls, workers, **kw):
    """Grade many predictions, preserving order.

    Args:
        problems: Questions from :func:`sql_bench.load`.
        sqls: Predictions aligned with ``problems``.
        workers: Concurrent gradings.
        **kw: Passed to :func:`sql_bench.grade_prediction`.

    Returns:
        Records in problem order.
    """
    return sb.grade_many(problems, sqls, workers=workers, **kw)


def check_gold(problems, workers, timeout, max_rows) -> tuple:
    """Every gold, graded against its own database.

    Args:
        problems: Questions from :func:`sql_bench.load`.
        workers: Concurrent gradings.
        timeout: Wall-clock seconds per execution.
        max_rows: Row cap per execution.

    Returns:
        ``(n_pass, n_total, broken, empty, records)`` where ``broken`` lists
        ``(question_id, status, detail)`` for every gold that did not self-verify and
        ``empty`` lists the question ids whose gold returns no rows.
    """
    recs = _grade_all(problems, [p["gold_sql"] for p in problems], workers,
                      timeout=timeout, max_rows=max_rows)
    broken = [(r["question_id"], r["status"], r["detail"][:200])
              for r in recs if not r["passed"]]
    empty = [r["question_id"] for r in recs if r["gold_rows"] == 0]
    return sum(1 for r in recs if r["passed"]), len(recs), broken, empty, recs


def reorder_rewrite(gold_sql: str) -> str:
    """A query returning the gold's rows in a deliberately different order.

    Args:
        gold_sql: The reference query.

    Returns:
        The gold wrapped in a subquery ordered by its first column descending. Valid for any
        SELECT, and only used where the gold does NOT impose an order of its own.
    """
    return f"SELECT * FROM ( {gold_sql} ) AS __selfcheck ORDER BY 1 DESC"


def check_equivalence(problems, gold_recs, workers, timeout, max_rows) -> tuple:
    """Rewritten golds returning the same rows in another order must still pass.

    This is what makes check 1 non-vacuous: a comparator that ignored row VALUES entirely
    would pass gold-against-itself and fail here.

    Args:
        problems: Questions from :func:`sql_bench.load`.
        gold_recs: Records from :func:`check_gold`, for the gold row counts.
        workers: Concurrent gradings.
        timeout: Wall-clock seconds per execution.
        max_rows: Row cap per execution.

    Returns:
        ``(n_pass, n_total, n_actually_reordered, failures, plan_dependent)``.
        ``n_actually_reordered`` counts the rewrites whose row SEQUENCE really differed from
        gold's, since a rewrite that returned the same order would not have exercised
        anything. ``plan_dependent`` lists the golds excluded because the rewrite returned a
        different SET of rows -- see the note in the body.
    """
    cand = []
    for p, g in zip(problems, gold_recs):
        if sb.order_matters(p["gold_sql"]):
            continue
        if (g["gold_rows"] or 0) < 2:
            continue
        cand.append((p, reorder_rewrite(p["gold_sql"])))

    def _shape(pair):
        """Gold rows and rewrite rows, so plan-dependence can be told from a grader bug."""
        p, q = pair
        a = sb.execute_sql(p["db_path"], p["gold_sql"], timeout, max_rows)
        b = sb.execute_sql(p["db_path"], q, timeout, max_rows)
        return a, b

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        shapes = list(ex.map(_shape, cand))

    sel, sqls, plan_dependent, reordered = [], [], [], 0
    for (p, q), (a, b) in zip(cand, shapes):
        if a.status != "ok" or b.status != "ok":
            continue
        if collections.Counter(a.rows) != collections.Counter(b.rows):
            # Not a grader fault. The gold carries a LIMIT with no total order, so WHICH
            # rows it returns is the query planner's arbitrary choice; wrapping it changes
            # the plan and a different set of rows comes back. Excluded from the
            # requirement and reported, because a semantically equivalent prediction is
            # graded wrong on these rows through no fault of its own.
            plan_dependent.append(p["question_id"])
            continue
        if a.rows != b.rows:
            reordered += 1
        sel.append(p)
        sqls.append(q)
    if not sel:
        return 0, 0, 0, [], plan_dependent
    recs = _grade_all(sel, sqls, workers, timeout=timeout, max_rows=max_rows)
    failures = [(r["question_id"], r["status"], r["detail"][:200])
                for r in recs if not r["passed"]]
    return (sum(1 for r in recs if r["passed"]), len(recs), reordered, failures,
            plan_dependent)


# Each entry is (family, builder, whether the family needs an ORDER BY gold). The builder
# receives the problem and the gold record and returns a prediction that MUST be graded
# wrong, or None when this question cannot support that family.
def _w_constant(p, g):
    """A one-row, one-column constant no gold can return.

    A numeric constant was tried first and had to be skipped whenever the gold returned a
    single row, because a count could coincide with it. A string literal that does not occur
    anywhere in the data cannot coincide, so this family runs on every question instead of a
    minority of them, and a pass here is a real escape rather than an ambiguity.
    """
    return "SELECT '__selfcheck_wrong_answer__'"


def _w_different_column(p, g):
    """The right rows with the wrong number of columns -- each column duplicated."""
    return f"SELECT *, * FROM ( {p['gold_sql']} ) AS __w"


def _w_inverted_where(p, g):
    """The gold with its result set inverted away to nothing."""
    return f"SELECT * FROM ( {p['gold_sql']} ) AS __w WHERE NOT (1 = 1)"


def _w_duplicated_rows(p, g):
    """Every row twice. WRONG under the multiset rule, CORRECT under BIRD's set comparison."""
    return (f"SELECT * FROM ( {p['gold_sql']} ) AS __a "
            f"UNION ALL SELECT * FROM ( {p['gold_sql']} ) AS __b")


def _w_dropped_row(p, g):
    """One row short."""
    if (g["gold_rows"] or 0) < 2:
        return None
    return f"SELECT * FROM ( {p['gold_sql']} ) AS __w LIMIT {g['gold_rows'] - 1}"


def _w_reversed_order(p, g):
    """The right rows in the wrong order, for a gold that orders its own output."""
    if not sb.order_matters(p["gold_sql"]):
        return None
    if (g["gold_rows"] or 0) < 2:
        return None
    return f"SELECT * FROM ( {p['gold_sql']} ) AS __w ORDER BY 1 DESC"


def _w_syntax_error(p, g):
    """Not SQL at all."""
    return "SELEKT * FRM nowhere WHEER"


def _w_empty_string(p, g):
    """The empty prediction."""
    return ""


def _w_prose(p, g):
    """A completion that answers in words and never writes a query."""
    return "I would need to look at the schema more carefully to answer this."


def _eligible_reversed_order(p, sql, timeout, max_rows) -> bool:
    """Whether ``sql`` really is the gold's rows in a DIFFERENT order.

    ``ORDER BY 1 DESC`` over the gold sometimes reproduces the gold's own sequence -- a gold
    ordered by a computed metric can already be descending in its output column. When it
    does, the "wrong" prediction is not wrong, and grading it a pass is correct behaviour.
    A first version omitted this check and reported five grader ESCAPES on bird_dev that
    were all this: the probe had reversed nothing. A known-wrong case that is not wrong
    measures the probe, not the grader.

    Args:
        p: The question.
        sql: The candidate wrong prediction.
        timeout: Wall-clock seconds per execution.
        max_rows: Row cap per execution.

    Returns:
        Whether the rewrite holds the same rows in a genuinely different sequence.
    """
    a = sb.execute_sql(p["db_path"], p["gold_sql"], timeout, max_rows)
    b = sb.execute_sql(p["db_path"], sql, timeout, max_rows)
    if a.status != "ok" or b.status != "ok":
        return False
    # Same multiset (so ORDER is the only difference) and a different sequence (so there IS
    # a difference). Both halves are needed: without the first the case would fail because
    # the rows differ, which tests nothing about ordering.
    return (collections.Counter(a.rows) == collections.Counter(b.rows)
            and a.rows != b.rows)


# (family, builder, eligibility). The builder returns a prediction that MUST be graded
# wrong, or None when this question cannot support the family. The eligibility hook, when
# present, EXECUTES to confirm the case really is wrong before it is required to fail.
WRONG_FAMILIES = [
    ("wrong_constant", _w_constant, None),
    ("different_columns", _w_different_column, None),
    ("inverted_where", _w_inverted_where, None),
    ("duplicated_rows", _w_duplicated_rows, None),
    ("dropped_row", _w_dropped_row, None),
    ("reversed_order", _w_reversed_order, _eligible_reversed_order),
    ("syntax_error", _w_syntax_error, None),
    ("empty_string", _w_empty_string, None),
    ("prose", _w_prose, None),
]

# Families whose whole point is that BIRD's own set-based comparison would call them
# CORRECT. If one of these ever stops failing, the rule it tests has stopped existing.
DISCRIMINATING = {"duplicated_rows", "reversed_order"}


def check_wrong(problems, gold_recs, workers, timeout, max_rows, cap: int = 250) -> tuple:
    """Deliberately wrong predictions, every one of which must be graded wrong.

    Eligibility is decided PER FAMILY over the whole benchmark and only then sampled, rather
    than sampling questions first and asking which families they support. A first version did
    it the other way round and gave ``reversed_order`` -- one of the two families that
    separate this grader from BIRD's own -- zero cases, so the run reported nine green
    families while the ordering rule was never exercised at all.

    Args:
        problems: Every question of the benchmark.
        gold_recs: Their gold records, for row counts.
        workers: Concurrent gradings.
        timeout: Wall-clock seconds per execution.
        max_rows: Row cap per execution.
        cap: Most questions per family, taken by an even stride over the eligible ones so
            the sample is deterministic and spans the file.

    Returns:
        ``(n_failed_as_required, n_total, per_family, escapes)`` where ``per_family`` maps a
        family to ``(n_required_wrong, n_graded_wrong, n_official_would_pass)`` and
        ``escapes`` lists ``(family, question_id, status)`` for anything that scored correct.
    """
    per_family, escapes, total, ok = {}, [], 0, 0
    for name, builder, eligibility in WRONG_FAMILIES:
        eligible = []
        for p, g in zip(problems, gold_recs):
            if not g["passed"]:
                continue  # a broken gold cannot support a known-wrong claim
            q = builder(p, g)
            if q is None:
                continue
            eligible.append((p, q))
        stride = max(1, len(eligible) // cap) if cap else 1
        chosen = eligible[::stride][:cap] if cap else eligible
        if eligibility is not None:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                keep = list(ex.map(lambda pq: eligibility(pq[0], pq[1], timeout, max_rows),
                                   chosen))
            chosen = [pq for pq, k in zip(chosen, keep) if k]
        sel = [p for p, _ in chosen]
        sqls = [q for _, q in chosen]
        if not sel:
            per_family[name] = (0, 0, 0)
            continue
        recs = _grade_all(sel, sqls, workers, timeout=timeout, max_rows=max_rows)
        wrong = sum(1 for r in recs if not r["passed"])
        official_pass = sum(1 for r in recs if r.get("passed_official"))
        per_family[name] = (len(recs), wrong, official_pass)
        total += len(recs)
        ok += wrong
        escapes.extend((name, r["question_id"], r["status"])
                       for r in recs if r["passed"])
    return ok, total, per_family, escapes


def check_readonly(problem, timeout, max_rows) -> tuple:
    """A write, a DDL, an ATTACH and a chained statement must all be refused.

    Every probe names a table that really EXISTS in this database, looked up from
    ``sqlite_master`` rather than hard-coded. A first version wrote
    ``DROP TABLE IF EXISTS superhero`` against whichever database question 0 used; the table
    was not there, ``IF EXISTS`` made the statement a legal no-op, SQLite returned success
    without attempting a write, and the probe recorded a read-only BREACH that had not
    happened. A guard that fires on the wrong thing is not a guard.

    The last probe is the reason ``PRAGMA writable_schema`` is not tested on its own: the
    pragma is harmless unless a write follows it on the SAME connection, and this grader
    executes one statement per fresh connection. What must therefore hold is that a chained
    payload cannot run at all, which is what is asserted.

    Args:
        problem: Any question, used for its database.
        timeout: Wall-clock seconds per execution.
        max_rows: Row cap per execution.

    Returns:
        ``(ok, details)`` where ``details`` maps a probe name to the status and error.
    """
    got = sb.execute_sql(problem["db_path"],
                         "SELECT name FROM sqlite_master WHERE type='table' "
                         "AND name NOT LIKE 'sqlite_%' ORDER BY name LIMIT 1",
                         timeout, max_rows)
    if got.status != "ok" or not got.rows:
        return False, {"lookup": (got.status, got.error[:160])}
    table = got.rows[0][0]
    probes = {
        "insert": f'INSERT INTO "{table}" DEFAULT VALUES',
        "delete": f'DELETE FROM "{table}"',
        "create_table": "CREATE TABLE __selfcheck_should_not_exist (a INT)",
        "drop_existing_table": f'DROP TABLE "{table}"',
        "attach": "ATTACH DATABASE '/tmp/__selfcheck_attach.db' AS evil",
        "chained_write": ("PRAGMA writable_schema = ON; "
                          f'DELETE FROM "{table}"'),
    }
    details, ok = {}, True
    for name, sql in probes.items():
        r = sb.execute_sql(problem["db_path"], sql, timeout, max_rows)
        details[name] = (r.status, r.error[:160])
        # "ok" here would mean the statement was ACCEPTED. Every one of these must be
        # refused by the read-only URI, by the authorizer, or by sqlite3's one-statement rule.
        if r.status == "ok":
            ok = False
    return ok, details


def check_timeout(problem, limit: float = 3.0) -> tuple:
    """An unbounded query must be graded ``timeout`` in bounded wall-clock time.

    Args:
        problem: Any question, used for its database.
        limit: The wall-clock limit to enforce for this probe.

    Returns:
        ``(ok, status, seconds)``.
    """
    runaway = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
               "SELECT count(*) FROM c")
    t0 = time.time()
    rec = sb.grade_prediction(problem, runaway, timeout=limit)
    el = time.time() - t0
    # Bounded: the gold runs first, so allow the gold's time plus the limit plus slack.
    return (rec["status"] == sb.ST_TIMEOUT and not rec["passed"] and el < limit + 30.0,
            rec["status"], round(el, 2))


def build_parser() -> argparse.ArgumentParser:
    """The command-line parser this script actually runs with."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmarks", default=",".join(sb.SUITE))
    ap.add_argument("--limit", type=int, default=0, help="questions per benchmark, 0 = all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--exec-timeout", type=float, default=sb.DEFAULT_EXEC_TIMEOUT)
    ap.add_argument("--max-rows", type=int, default=sb.DEFAULT_MAX_ROWS)
    ap.add_argument("--wrong-sample", type=int, default=250,
                    help="most questions PER FAMILY the known-wrong battery runs on, taken "
                         "by an even stride over the questions that family applies to, so "
                         "the sample is deterministic and spans the file; 0 = all of them")
    ap.add_argument("--out", default="", help="JSON summary")
    return ap


def main() -> int:
    """Run every check and return a process exit status."""
    args = build_parser().parse_args()
    report, failed = {}, False
    for bench in [b.strip() for b in args.benchmarks.split(",") if b.strip()]:
        t0 = time.time()
        problems = sb.load(bench, limit=args.limit)
        print(f"\n=== {bench}: {len(problems)} questions, "
              f"{len({p['db_id'] for p in problems})} databases", flush=True)

        small = min(problems, key=lambda p: Path(p["db_path"]).stat().st_size)
        before = hashlib.sha256(Path(small["db_path"]).read_bytes()).hexdigest()

        n_ok, n, broken, empty, gold_recs = check_gold(
            problems, args.workers, args.exec_timeout, args.max_rows)
        print(f"1. GOLD SELF-CHECK      {n_ok}/{n}"
              f"   broken={len(broken)}  empty_gold={len(empty)}", flush=True)
        for qid, st, det in broken[:20]:
            print(f"     BROKEN qid={qid} {st}: {det}", flush=True)
        if empty[:20]:
            print(f"     EMPTY GOLD qids: {empty[:20]}", flush=True)
        if n_ok != n:
            failed = True

        eq_ok, eq_n, eq_reordered, eq_fail, eq_plandep = check_equivalence(
            problems, gold_recs, args.workers, args.exec_timeout, args.max_rows)
        print(f"2. EQUIVALENCE          {eq_ok}/{eq_n}"
              f"   (row sequence really changed on {eq_reordered};"
              f" {len(eq_plandep)} plan-dependent gold(s) excluded)", flush=True)
        if eq_plandep:
            print(f"     PLAN-DEPENDENT GOLD qids {sorted(eq_plandep)}: LIMIT with no total "
                  f"order, so WHICH rows the gold returns is the planner's choice. A "
                  f"semantically equivalent prediction is graded wrong on these.",
                  flush=True)
        for qid, st, det in eq_fail[:20]:
            print(f"     FAILED qid={qid} {st}: {det}", flush=True)
        if eq_ok != eq_n or eq_reordered == 0:
            failed = True

        w_ok, w_n, per_family, escapes = check_wrong(
            problems, gold_recs, args.workers, args.exec_timeout, args.max_rows,
            cap=args.wrong_sample)
        print(f"3. KNOWN-WRONG          {w_ok}/{w_n}   "
              f"(<= {args.wrong_sample or 'all'} questions per family)", flush=True)
        for name, (req, got, off) in per_family.items():
            mark = "OK " if req == got else "ESCAPED"
            extra = f"   (BIRD's set comparison would pass {off})" if off else ""
            print(f"     {mark} {name:<20} {got}/{req}{extra}", flush=True)
        for fam, qid, st in escapes[:20]:
            print(f"     ESCAPE family={fam} qid={qid} status={st}", flush=True)
        if w_ok != w_n:
            failed = True
        for fam in DISCRIMINATING:
            req, got, off = per_family.get(fam, (0, 0, 0))
            if req == 0 or got != req or off == 0:
                # off == 0 would mean the family is not actually discriminating on this
                # data, so it is no longer evidence that our rule differs from BIRD's.
                print(f"     WEAK {fam}: required={req} wrong={got} official_pass={off}",
                      flush=True)
                failed = True

        ro_ok, ro_details = check_readonly(problems[0], args.exec_timeout, args.max_rows)
        print(f"4. READ-ONLY            {'OK' if ro_ok else 'BREACHED'}", flush=True)
        for name, (st, err) in ro_details.items():
            print(f"     {name:<24} {st}: {err}", flush=True)
        after = hashlib.sha256(Path(small["db_path"]).read_bytes()).hexdigest()
        same = before == after
        print(f"     fixture sha256 unchanged ({Path(small['db_path']).name}): {same}",
              flush=True)
        if not ro_ok or not same:
            failed = True

        to_ok, to_status, to_secs = check_timeout(problems[0])
        print(f"5. TIMEOUT              {'OK' if to_ok else 'FAILED'} "
              f"status={to_status} {to_secs}s", flush=True)
        if not to_ok:
            failed = True

        report[bench] = {
            "n_questions": n,
            "n_databases": len({p["db_id"] for p in problems}),
            "gold_pass": n_ok,
            "gold_broken": broken,
            "gold_empty": empty,
            "equivalence_pass": eq_ok,
            "equivalence_n": eq_n,
            "equivalence_reordered": eq_reordered,
            "equivalence_plan_dependent_golds": sorted(eq_plandep),
            "wrong_failed_as_required": w_ok,
            "wrong_n": w_n,
            "wrong_per_family_cap": args.wrong_sample,
            "wrong_per_family": {k: {"n": v[0], "graded_wrong": v[1],
                                     "official_would_pass": v[2]}
                                 for k, v in per_family.items()},
            "wrong_escapes": escapes,
            "readonly_ok": ro_ok,
            "readonly_probes": {k: v[0] for k, v in ro_details.items()},
            "fixture_unchanged": same,
            "timeout_ok": to_ok,
            "timeout_status": to_status,
            "exec_timeout": args.exec_timeout,
            "max_rows": args.max_rows,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"    ({report[bench]['seconds']}s)", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    print("\nSELFCHECK " + ("FAILED" if failed else "PASSED"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
