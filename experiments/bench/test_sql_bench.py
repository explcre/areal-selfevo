#!/usr/bin/env python3
"""Tests for the BIRD text-to-SQL harness.

Three things are being constrained here, in descending order of how badly they would lie.

The first is the COMPARATOR. Execution accuracy is decided entirely by "are these two
tables the same table", and every way of getting that wrong is a way of reporting a score
nobody can use: a set comparison silently stops counting duplicate rows, an order-blind
comparison passes a top-N answer that ranked the wrong way round, and a comparison that
looks only at shapes passes any query returning the right number of rows. Each of those is
asserted directly, in both directions -- the equivalent rearrangement must PASS and the
wrong answer must FAIL -- because a comparator tested only on things it should accept is
indistinguishable from ``return True``.

The second is the ACCOUNTING: a query that errors, hangs or was never written must be a FAIL
that stays in the denominator, and the counts must add up to the number of questions no
matter which way each one failed. This project has repeatedly reported inflated scores
because excluded samples quietly left the denominator.

The third is the FIXTURE. The databases are a 1.4 GiB download that nothing here can
regenerate, and the queries being run are written by an unaligned model, so "read-only" is
asserted rather than assumed.

The end of the file drives the real CLI over a synthetic database and recorded completions,
because a harness that is only ever exercised through its parts keeps working after the loop
that joins them has broken.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sql_bench as sb  # noqa: E402


def has_dataset():
    """Whether the real BIRD archive is unpacked on this machine."""
    try:
        sb.require_dataset("bird_mini_dev")
        return True
    except Exception:
        return False


needs_dataset = pytest.mark.skipif(not has_dataset(),
                                   reason="BIRD dev archive not unpacked on this box")


# ------------------------------------------------------------------ synthetic fixture ----


def _make_db(path: Path) -> None:
    """Build a small database with the properties the comparator tests need.

    ``t`` carries a DUPLICATE row and a NULL so that multiset and null handling are
    exercisable; ``u`` carries two columns so column permutation is exercisable.

    Args:
        path: Destination ``.sqlite`` file.
    """
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE t (a INTEGER, b TEXT);
        INSERT INTO t VALUES (1, 'x'), (2, 'y'), (2, 'y'), (3, NULL);
        CREATE TABLE u (p INTEGER, q INTEGER);
        INSERT INTO u VALUES (1, 10), (2, 20), (3, 30);
        """
    )
    conn.commit()
    conn.close()


@pytest.fixture()
def bird(tmp_path, monkeypatch):
    """A complete synthetic $BIRD_DATA tree, laid out exactly as the real archive unpacks.

    Returns:
        The data root, with ``dev_20240627/dev.json`` holding four questions against one
        database.
    """
    root = tmp_path / "bird"
    dbdir = root / sb.DB_SUBDIR / "toy"
    dbdir.mkdir(parents=True)
    _make_db(dbdir / "toy.sqlite")
    rows = [
        {"question_id": 0, "db_id": "toy", "question": "all of t",
         "evidence": "", "SQL": "SELECT a, b FROM t", "difficulty": "simple"},
        {"question_id": 1, "db_id": "toy", "question": "u ordered",
         "evidence": "hint", "SQL": "SELECT p FROM u ORDER BY p DESC",
         "difficulty": "moderate"},
        {"question_id": 2, "db_id": "toy", "question": "two columns",
         "evidence": "", "SQL": "SELECT p, q FROM u", "difficulty": "challenging"},
        {"question_id": 3, "db_id": "toy", "question": "count",
         "evidence": "", "SQL": "SELECT count(*) FROM t", "difficulty": "simple"},
    ]
    f = root / sb.RELEASES["bird_dev"]
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rows))
    monkeypatch.setenv(sb.DATA_ENV, str(root))
    sb.schema_prompt.cache_clear()
    return root


@pytest.fixture()
def toy(bird):
    """The four synthetic questions, loaded through the real loader."""
    return sb.load("bird_dev")


# ----------------------------------------------------------------------------- premise ----


def test_premise_the_fixture_really_contains_a_duplicate_row_and_an_order(toy):
    """If this fails, every multiset and ordering test below is testing nothing.

    A duplicate-blind comparator can only be caught by data that HAS a duplicate row, and an
    order-blind one only by a gold that really orders more than one row.
    """
    r = sb.execute_sql(toy[0]["db_path"], toy[0]["gold_sql"])
    assert r.status == "ok"
    assert len(r.rows) == 4 and len(set(r.rows)) == 3, "no duplicate row in the fixture"
    o = sb.execute_sql(toy[1]["db_path"], toy[1]["gold_sql"])
    assert o.rows == [(3,), (2,), (1,)], "the ordered gold does not order anything"
    assert sb.order_matters(toy[1]["gold_sql"])
    assert not sb.order_matters(toy[0]["gold_sql"])


# -------------------------------------------------------------------------- extract_sql ----


def test_extracts_a_fenced_sql_block():
    assert sb.extract_sql("blah\n```sql\nSELECT 1\n```\n") == "SELECT 1"


def test_last_sql_block_wins():
    """A model that explores before committing is graded on what it committed."""
    text = "```sql\nSELECT 1\n```\nno wait\n```sql\nSELECT 2\n```"
    assert sb.extract_sql(text) == "SELECT 2"


def test_an_untagged_block_is_accepted_when_nothing_is_tagged():
    assert sb.extract_sql("```\nSELECT 3\n```") == "SELECT 3"


def test_a_tagged_block_outranks_a_later_untagged_one():
    """Sample output in a bare fence must not displace the submitted query."""
    text = "```sql\nSELECT 1\n```\nresult:\n```\n1\n```"
    assert sb.extract_sql(text) == "SELECT 1"


def test_an_empty_trailing_fence_does_not_discard_the_submission():
    """The exact fault a live 32B run hit on the code benchmark, asserted here before it can."""
    assert sb.extract_sql("```sql\nSELECT 1\n```\n```sql\n```") == "SELECT 1"


def test_a_bare_query_is_accepted_because_the_prompt_asks_for_one():
    """BIRD's own instruction block says not to fence and to start from SELECT."""
    assert sb.extract_sql("SELECT a FROM t") == "SELECT a FROM t"
    assert sb.extract_sql("  WITH c AS (SELECT 1) SELECT * FROM c").startswith("WITH")


def test_prose_is_not_a_query():
    """Prose must be refused, not handed to the database so its syntax error looks like SQL."""
    assert sb.extract_sql("I would need the schema to answer that.") is None
    assert sb.extract_sql("") is None
    assert sb.extract_sql("```sql\n\n```") is None


def test_an_explanation_before_a_bare_query_is_refused_not_salvaged():
    """Only a completion that BEGINS with a statement keyword is taken unfenced.

    Salvaging by searching for the first SELECT anywhere would submit the model's
    description of a query as the query.
    """
    assert sb.extract_sql("Here is my answer: SELECT a FROM t") is None


# ------------------------------------------------------------------------ order_matters ----


def test_a_top_level_order_by_makes_order_significant():
    assert sb.order_matters("SELECT a FROM t ORDER BY a")
    assert sb.order_matters("SELECT a FROM t UNION SELECT b FROM u ORDER BY 1")


def test_an_order_by_inside_a_subquery_does_not_order_the_result():
    """A subquery's ORDER BY does not order the outer rows.

    Demanding one would mark a correct prediction wrong for an ordering the gold never
    produced. Spider's exec_eval, which substring-matches the raw text, gets this wrong.
    """
    assert not sb.order_matters(
        "SELECT a FROM t WHERE a IN (SELECT p FROM u ORDER BY p LIMIT 2)")
    assert not sb.order_matters(
        "WITH c AS (SELECT p FROM u ORDER BY p) SELECT * FROM c")


def test_order_by_inside_a_string_literal_or_comment_is_not_a_keyword():
    assert not sb.order_matters("SELECT a FROM t WHERE b = 'order by'")
    assert not sb.order_matters("SELECT a FROM t -- order by a")
    assert not sb.order_matters("SELECT a FROM t /* order by a */")
    assert not sb.order_matters('SELECT "order by" FROM t')
    assert not sb.order_matters("SELECT `order by` FROM t")


def test_an_unterminated_quote_does_not_hang_the_detector():
    """The literal patterns are the linear-time forms; the naive ones backtrack forever.

    A generated query supplies an unterminated quote sooner or later, and a grader that
    stops responding on one is worse than a grader that mis-grades it.
    """
    t0 = time.time()
    sb.order_matters("SELECT a FROM t WHERE b = '" + "x" * 400)
    assert time.time() - t0 < 5.0


# ------------------------------------------------------------------------ results_equal ----

G = [(1, "x"), (2, "y"), (2, "y"), (3, None)]


def test_row_order_is_ignored_when_the_gold_does_not_order():
    assert sb.results_equal(G, list(reversed(G)), ordered=False)[0]


def test_row_order_is_significant_when_the_gold_orders():
    """An order-blind comparison passes a top-N answer that ranked the wrong way round."""
    assert not sb.results_equal([(3,), (2,), (1,)], [(1,), (2,), (3,)], ordered=True)[0]
    assert sb.results_equal([(3,), (2,), (1,)], [(3,), (2,), (1,)], ordered=True)[0]


def test_duplicate_rows_count_which_is_where_this_differs_from_bird():
    """Rows compare as a MULTISET, so dropping or doubling a repeated row is wrong.

    BIRD's own calculate_ex compares sets and calls all of these correct; that verdict is
    still computed and reported separately, but it is not this grader's verdict.

    The SAME-LENGTH pair is the one that constrains anything, and it was added after the
    mutation harness showed why. With only the dropped and doubled cases this test passed
    against a grader whose multiset comparison had been replaced by a set comparison: those
    two differ from gold in ROW COUNT, so ``results_equal`` rejects them at the length guard
    and never reaches the comparison being tested. A test that is satisfied before the code
    it names runs is not evidence about that code.
    """
    same_length_different_multiset = [(1, "x"), (1, "x"), (2, "y"), (3, None)]
    deduped = [(1, "x"), (2, "y"), (3, None)]
    doubled = G + G
    assert len(same_length_different_multiset) == len(G), "premise: the length guard is passed"
    assert not sb.results_equal(G, same_length_different_multiset, ordered=False)[0]
    assert not sb.results_equal(G, deduped, ordered=False)[0]
    assert not sb.results_equal(G, doubled, ordered=False)[0]
    assert sb.official_equal(G, same_length_different_multiset), \
        "premise: BIRD's rule really does accept these"
    assert sb.official_equal(G, deduped)
    assert sb.official_equal(G, doubled)


def test_column_order_is_ignored():
    gold = [(1, 10), (2, 20), (3, 30)]
    swapped = [(10, 1), (20, 2), (30, 3)]
    assert sb.results_equal(gold, swapped, ordered=False)[0]
    assert sb.results_equal(gold, swapped, ordered=True)[0]
    assert not sb.official_equal(gold, swapped), "premise: BIRD's rule is positional"


def test_a_column_permutation_must_permute_every_row_the_same_way():
    """Reordering the cells of ONE row is not a column permutation and must not pass.

    A comparator that sorted the cells within each row independently would accept this, and
    would then accept any answer whose rows are anagrams of gold's.
    """
    gold = [(1, 10), (2, 20), (3, 30)]
    per_row_shuffle = [(1, 10), (20, 2), (3, 30)]
    assert not sb.results_equal(gold, per_row_shuffle, ordered=False)[0]


def test_row_values_are_compared_not_just_shapes():
    """A comparator that checked only counts would pass this, and would pass anything."""
    assert not sb.results_equal([(1,)], [(2,)], ordered=False)[0]
    assert not sb.results_equal(G, [(9, "z")] * 4, ordered=False)[0]


def test_a_different_number_of_rows_or_columns_is_wrong():
    assert not sb.results_equal(G, G[:2], ordered=False)[0]
    assert not sb.results_equal([(1, 2)], [(1, 2, 3)], ordered=False)[0]


def test_two_empty_results_are_equal_and_that_is_bird_s_rule():
    """Reachable only on a dataset with an empty gold; dev and mini-dev have none (measured)."""
    assert sb.results_equal([], [], ordered=False)[0]
    assert not sb.results_equal([(1,)], [], ordered=False)[0]
    assert not sb.results_equal([], [(1,)], ordered=False)[0]


def test_null_is_compared_as_a_value():
    assert sb.results_equal([(None,)], [(None,)], ordered=False)[0]
    assert not sb.results_equal([(None,)], [(0,)], ordered=False)[0]


def test_an_integer_and_the_same_float_are_the_same_answer():
    """CAST(x AS REAL) must not be graded wrong; BIRD compares raw sqlite tuples and agrees."""
    assert sb.results_equal([(5,)], [(5.0,)], ordered=False)[0]
    assert not sb.results_equal([(5,)], [("5",)], ordered=False)[0]


def test_the_permutation_budget_reports_exhaustion_instead_of_grading_wrong():
    """Hitting the budget is "not shown equal", never a silent wrong answer.

    A budget that quietly returned False would be a lever on the score with no artifact
    behind it.
    """
    gold = [(1, 2, 3, 4), (1, 2, 3, 4), (4, 3, 2, 1)]
    pred = [(4, 3, 2, 1), (4, 3, 2, 1), (1, 2, 3, 4)]
    equal, exhausted = sb.results_equal(gold, pred, ordered=False, max_perms=0)
    assert not equal and exhausted


def test_official_equal_is_bird_s_function_and_nothing_else():
    """accuracy_official must stay comparable to a published BIRD score."""
    assert sb.official_equal([(1,), (1,)], [(1,)])
    assert sb.official_equal([(1,), (2,)], [(2,), (1,)])
    assert not sb.official_equal([(1, 2)], [(2, 1)])


# --------------------------------------------------------------------------- execution ----


def test_a_wrong_query_is_an_outcome_not_an_exception(toy):
    r = sb.execute_sql(toy[0]["db_path"], "SELECT nosuchcolumn FROM t")
    assert r.status == "error" and "nosuchcolumn" in r.error


def test_the_database_is_opened_read_only(toy):
    """The fixture is a download nothing here can regenerate, so writes must be refused."""
    for sql in ("CREATE TABLE zzz (a INT)", "DELETE FROM t", "DROP TABLE t",
                "INSERT INTO t VALUES (9, 'z')", "UPDATE t SET a = 0"):
        r = sb.execute_sql(toy[0]["db_path"], sql)
        assert r.status == "error", f"{sql!r} was ACCEPTED against a read-only database"
    still = sb.execute_sql(toy[0]["db_path"], "SELECT count(*) FROM t")
    assert still.rows == [(4,)], "the fixture was modified"


def test_attach_is_denied_so_a_query_cannot_reach_a_second_file(toy, tmp_path):
    """mode=ro protects the file that was opened; ATTACH would open another one."""
    target = tmp_path / "attached.db"
    r = sb.execute_sql(toy[0]["db_path"], f"ATTACH DATABASE '{target}' AS evil")
    assert r.status == "error" and "authorized" in r.error.lower()
    assert not target.exists()


def test_two_statements_cannot_be_chained(toy):
    """One statement per connection is what makes a stateful pragma inert."""
    r = sb.execute_sql(toy[0]["db_path"], "SELECT 1; DELETE FROM t")
    assert r.status == "error"


def test_a_runaway_query_is_killed_at_the_limit(toy):
    """A hang must be a bounded timeout, not a grader that stops responding.

    sqlite releases the GIL inside execute(), so a limit implemented in the parent's own
    interpreter would bound nothing; this asserts the child is really killed.
    """
    runaway = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
               "SELECT count(*) FROM c")
    t0 = time.time()
    r = sb.execute_sql(toy[0]["db_path"], runaway, timeout=2.0)
    el = time.time() - t0
    assert r.status == "timeout"
    assert el < 30.0, f"the limit did not bound anything: {el:.1f}s"


def test_the_row_cap_is_reported_rather_than_trimming_the_answer(toy):
    r = sb.execute_sql(toy[0]["db_path"], "SELECT a, b FROM t", max_rows=2)
    assert r.truncated and len(r.rows) == 2


def test_a_path_needing_quoting_still_opens_the_right_file(tmp_path):
    """A database directory with a space or a ? would otherwise start the URI query early."""
    d = tmp_path / "od d?x#y"
    d.mkdir()
    _make_db(d / "toy.sqlite")
    r = sb.execute_sql(d / "toy.sqlite", "SELECT count(*) FROM t")
    assert r.status == "ok" and r.rows == [(4,)]


# ------------------------------------------------------------------------- the verdicts ----


def test_the_gold_passes_against_itself(toy):
    for p in toy:
        rec = sb.grade_prediction(p, p["gold_sql"])
        assert rec["status"] == sb.ST_PASS and rec["passed"], (p["question_id"], rec)


def test_an_equivalent_rewrite_passes(toy):
    """The rewrite returns the same rows in another order; a reflexive-only comparator fails it."""
    rec = sb.grade_prediction(toy[0], "SELECT a, b FROM t ORDER BY a DESC")
    assert rec["passed"]


def test_a_reordered_answer_fails_when_the_gold_ordered_it(toy):
    rec = sb.grade_prediction(toy[1], "SELECT p FROM u ORDER BY p ASC")
    assert rec["status"] == sb.ST_WRONG and not rec["passed"]
    assert rec["passed_official"], "premise: BIRD's own rule would have passed this"


def test_a_doubled_answer_fails_but_bird_would_pass_it(toy):
    rec = sb.grade_prediction(
        toy[0], "SELECT a, b FROM t UNION ALL SELECT a, b FROM t")
    assert not rec["passed"] and rec["passed_official"]


def test_a_query_that_errors_is_wrong_not_an_exclusion(toy):
    rec = sb.grade_prediction(toy[0], "SELECT nosuchcolumn FROM t")
    assert rec["status"] == sb.ST_EXEC_ERROR and not rec["passed"]


def test_a_query_that_hangs_is_wrong_and_bucketed_as_a_timeout(toy):
    runaway = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
               "SELECT count(*) FROM c")
    rec = sb.grade_prediction(toy[0], runaway, timeout=2.0)
    assert rec["status"] == sb.ST_TIMEOUT and not rec["passed"]


def test_a_query_returning_nothing_is_wrong_when_the_gold_returns_rows(toy):
    rec = sb.grade_prediction(toy[0], "SELECT a, b FROM t WHERE 1 = 0")
    assert rec["status"] == sb.ST_WRONG and not rec["passed"]


def test_a_missing_query_is_a_fail_not_an_exclusion(toy):
    for pred in (None, "", "   "):
        rec = sb.grade_prediction(toy[0], pred)
        assert rec["status"] == sb.ST_NO_SQL and not rec["passed"]


def test_a_broken_gold_is_named_as_a_broken_row_not_charged_to_the_model(toy):
    """The dataset's own query failing is a fact about the dataset."""
    p = dict(toy[0], gold_sql="SELECT nosuchcolumn FROM t")
    rec = sb.grade_prediction(p, "SELECT a, b FROM t")
    assert rec["status"] == sb.ST_GOLD_BROKEN and not rec["passed"]


def test_a_gold_over_the_row_cap_is_undecidable_not_wrong(toy):
    """Truncating both sides and comparing them would invent a verdict."""
    rec = sb.grade_prediction(toy[0], "SELECT a, b FROM t", max_rows=2)
    assert rec["status"] == sb.ST_ROW_LIMIT and not rec["passed"]


def test_a_prediction_over_the_row_cap_is_wrong_because_the_gold_is_not(toy):
    rec = sb.grade_prediction(
        dict(toy[0], gold_sql="SELECT a, b FROM t LIMIT 1"),
        "SELECT a, b FROM t, u, u AS u2", max_rows=3)
    assert rec["status"] == sb.ST_WRONG and not rec["passed"]


def test_grade_many_rejects_mismatched_lengths(toy):
    """A misalignment would grade every question against the wrong answer and still score."""
    with pytest.raises(ValueError):
        sb.grade_many(toy, ["SELECT 1"])


# -------------------------------------------------------------------------- accounting ----


def _row(toy, preds, params=None):
    """Grade predictions and summarise, as the CLI does."""
    recs = sb.grade_many(toy, preds, workers=4)
    return recs, sb.summarise("bird_dev", toy, recs, params or {})


def test_every_question_lands_in_exactly_one_bucket(toy):
    recs, row = _row(toy, [p["gold_sql"] for p in toy])
    assert sum(row["counts"].values()) == len(toy)
    assert set(row["counts"]) == set(sb.STATUSES)


def test_failures_stay_in_the_denominator(toy):
    preds = ["SELECT nosuchcol FROM t", None, "SELECT 1", toy[3]["gold_sql"]]
    recs, row = _row(toy, preds)
    assert row["n_problems"] == 4 and row["n_graded"] == 4
    assert row["n_pass"] == 1
    assert row["accuracy"] == pytest.approx(0.25)
    assert row["n_exec_error"] == 1 and row["n_no_sql"] == 1 and row["n_wrong_answer"] == 1


def test_only_generation_failures_leave_the_denominator_and_both_numbers_are_reported(toy):
    recs = sb.grade_many(toy, [p["gold_sql"] for p in toy], workers=4)
    recs[0]["status"] = sb.ST_GEN_FAILED
    recs[0]["passed"] = False
    row = sb.summarise("bird_dev", toy, recs, {})
    assert row["n_graded"] == 3 and row["n_pass"] == 3
    assert row["accuracy"] == pytest.approx(1.0)
    assert row["accuracy_all"] == pytest.approx(0.75)


def test_a_gold_broken_row_is_counted_as_a_fail_and_named_separately(toy):
    broken = [dict(toy[0], gold_sql="SELECT nosuchcol FROM t")] + toy[1:]
    recs = sb.grade_many(broken, [p["gold_sql"] for p in broken], workers=4)
    row = sb.summarise("bird_dev", broken, recs, {})
    assert row["n_gold_broken"] == 1
    assert row["n_pass"] == 3 and row["n_graded"] == 4


def test_both_verdicts_are_reported_and_their_disagreement_is_counted(toy):
    """The number comparable to a published BIRD score must be in the row, not inferred."""
    preds = [
        "SELECT a, b FROM t UNION ALL SELECT a, b FROM t",   # ours wrong, BIRD's right
        "SELECT p FROM u ORDER BY p ASC",                     # ours wrong, BIRD's right
        toy[2]["gold_sql"],
        toy[3]["gold_sql"],
    ]
    recs, row = _row(toy, preds)
    assert row["n_pass"] == 2
    assert row["n_pass_official"] == 4
    assert row["n_verdict_differs"] == 2
    assert row["accuracy"] == pytest.approx(0.5)
    assert row["accuracy_official"] == pytest.approx(1.0)


def test_the_per_difficulty_table_covers_every_question(toy):
    recs, row = _row(toy, [p["gold_sql"] for p in toy])
    assert sum(v["n"] for v in row["by_difficulty"].values()) == len(toy)
    assert row["by_difficulty"]["simple"]["n"] == 2


def test_wilson_is_nan_on_an_empty_denominator(toy):
    recs = sb.grade_many(toy, [p["gold_sql"] for p in toy], workers=4)
    for r in recs:
        r["status"] = sb.ST_GEN_FAILED
    row = sb.summarise("bird_dev", toy, recs, {})
    assert row["n_graded"] == 0
    assert row["accuracy"] != row["accuracy"]  # NaN, not a confident zero


# ------------------------------------------------------------------------------- load ----


def test_load_reads_the_expected_shape(toy):
    assert len(toy) == 4
    assert toy[0]["db_id"] == "toy" and Path(toy[0]["db_path"]).exists()
    assert toy[0]["gold_sql"].startswith("SELECT")


def test_load_filters_are_applied_and_an_empty_selection_is_fatal(bird):
    assert len(sb.load("bird_dev", difficulty="simple")) == 2
    assert len(sb.load("bird_dev", limit=2)) == 2
    assert [p["question_id"] for p in sb.load("bird_dev", question_ids={1, 2})] == [1, 2]
    with pytest.raises(ValueError):
        sb.load("bird_dev", question_ids={999})


def test_a_missing_dataset_is_fatal_not_an_empty_suite(tmp_path, monkeypatch):
    """An empty question list scores zero and looks exactly like a model that answered wrong."""
    monkeypatch.setenv(sb.DATA_ENV, str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError):
        sb.load("bird_dev")


def test_an_unknown_benchmark_name_is_fatal(bird):
    with pytest.raises(ValueError):
        sb.load("bird_enormous")


def test_an_empty_gold_sql_is_fatal_at_load(bird):
    """A blank gold makes a row ungradeable while looking like an ordinary row."""
    f = bird / sb.RELEASES["bird_dev"]
    rows = json.loads(f.read_text())
    rows[0]["SQL"] = "   "
    f.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        sb.load("bird_dev")


def test_an_unknown_difficulty_label_is_fatal(bird):
    """Otherwise the per-difficulty table silently omits those questions."""
    f = bird / sb.RELEASES["bird_dev"]
    rows = json.loads(f.read_text())
    rows[0]["difficulty"] = "trivial"
    f.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        sb.load("bird_dev")


def test_a_missing_database_is_fatal(bird):
    f = bird / sb.RELEASES["bird_dev"]
    rows = json.loads(f.read_text())
    rows[0]["db_id"] = "absent"
    f.write_text(json.dumps(rows))
    with pytest.raises(FileNotFoundError):
        sb.load("bird_dev")


def test_the_prompt_carries_the_schema_the_question_and_the_evidence(toy):
    p = sb.build_prompt(toy[1])
    assert "CREATE TABLE u" in p and "CREATE TABLE t" in p
    assert toy[1]["question"] in p
    assert "External Knowledge: hint" in p
    assert "SQLite" in p


def test_the_prompt_omits_the_knowledge_line_when_there_is_none(toy):
    assert "External Knowledge" not in sb.build_prompt(toy[0])


# -------------------------------------------------------------------------------- CLI ----


def test_main_grades_recorded_completions_and_writes_an_auditable_artifact(
        bird, tmp_path, monkeypatch, capsys):
    """The whole CLI, end to end, with no endpoint and no GPU."""
    gens = tmp_path / "gens.jsonl"
    with open(gens, "w") as fh:
        for qid, text in ((0, "```sql\nSELECT a, b FROM t\n```"),
                          (1, "SELECT p FROM u ORDER BY p ASC"),
                          (2, "SELECT q, p FROM u"),
                          (3, "I cannot answer this.")):
            fh.write(json.dumps({"question_id": qid, "text": text,
                                 "finish_reason": "stop", "status": "ok"}) + "\n")
    out = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", ["sql_bench.py", "--benchmarks", "bird_dev",
                                      "--from-generations", str(gens),
                                      "--out", str(out), "--workers", "2"])
    assert sb.main() == 0
    rows = json.loads(out.read_text())
    assert len(rows) == 1
    row = rows[0]
    assert row["n_problems"] == 4 and row["n_graded"] == 4
    # qid 0 right; qid 1 wrong order; qid 2 columns swapped, which this grader accepts;
    # qid 3 no SQL.
    assert row["n_pass"] == 2 and row["n_no_sql"] == 1 and row["n_wrong_answer"] == 1
    assert row["params"]["dataset_md5"]
    recs = [json.loads(x) for x in
            (tmp_path / "res.records.jsonl").read_text().splitlines()]
    assert len(recs) == 4
    assert {r["question_id"] for r in recs} == {0, 1, 2, 3}
    assert all("status" in r and "passed" in r for r in recs)


def test_main_counts_a_missing_generation_as_a_failure_not_a_shorter_benchmark(
        bird, tmp_path, monkeypatch):
    """Two things at once, because the second is what makes the first safe.

    A question with no recorded completion stays in ``n_problems`` and is counted a failure
    rather than shortening the benchmark; and an outage this large ABORTS instead of
    reporting ``accuracy`` over the one survivor, which here would have been a confident
    1.0000 from a run that answered a quarter of the questions.
    """
    gens = tmp_path / "gens.jsonl"
    gens.write_text(json.dumps({"question_id": 0, "text": "SELECT a, b FROM t",
                                "finish_reason": "stop", "status": "ok"}) + "\n")
    out = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", ["sql_bench.py", "--benchmarks", "bird_dev",
                                      "--from-generations", str(gens),
                                      "--out", str(out), "--workers", "2"])
    with pytest.raises(SystemExit, match="ABORT"):
        sb.main()
    row = json.loads(out.read_text())[0]
    assert row["n_problems"] == 4 and row["n_graded"] == 1 and row["n_failed"] == 3
    assert row["accuracy"] == pytest.approx(1.0)
    assert row["accuracy_all"] == pytest.approx(0.25)


def test_a_results_row_with_an_empty_difficulty_bucket_is_still_writable(
        bird, tmp_path, monkeypatch):
    """NaN must be scrubbed at every depth, not only at the top level of the row.

    The per-difficulty table carries a NaN accuracy for any difficulty with nothing graded,
    two levels down, and `json.dumps(allow_nan=False)` refuses it. The scrub is recursive
    and this is what fails if it stops being.
    """
    gens = tmp_path / "gens.jsonl"
    with open(gens, "w") as fh:
        for qid in (0, 1, 2, 3):
            fh.write(json.dumps({"question_id": qid, "text": "SELECT 1",
                                 "finish_reason": "stop", "status": "ok"}) + "\n")
    out = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", ["sql_bench.py", "--benchmarks", "bird_dev",
                                      "--difficulty", "simple",
                                      "--from-generations", str(gens),
                                      "--out", str(out), "--workers", "2"])
    assert sb.main() == 0
    row = json.loads(out.read_text())[0]
    assert row["by_difficulty"]["moderate"]["n"] == 0
    assert row["by_difficulty"]["moderate"]["accuracy"] is None


def test_json_safe_scrubs_at_every_depth():
    nan = float("nan")
    got = sb.json_safe({"a": nan, "b": {"c": [nan, 1.5]}, "d": float("inf")})
    assert got == {"a": None, "b": {"c": [None, 1.5]}, "d": None}
    json.dumps(got, allow_nan=False)


def test_load_generations_rejects_an_empty_file(toy, tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError):
        sb.load_generations(empty, toy)


def test_generating_without_a_model_refuses_before_touching_the_endpoint(bird, monkeypatch):
    """An unregistered id is answered HTTP 200 by the BASE model, so there is no safe default."""
    monkeypatch.setattr(sys, "argv", ["sql_bench.py", "--benchmarks", "bird_dev"])
    with pytest.raises(SystemExit):
        sb.main()


def test_the_model_flag_has_no_default_that_could_silently_resolve():
    assert sb.build_parser().get_default("model") is None


def test_the_grading_knobs_are_recorded_in_the_results_row(bird):
    """A grading tolerance that is not reported is a silent lever on the score."""
    args = sb.build_parser().parse_args(["--from-generations", "x"])
    params = sb.resolve_params("bird_dev", args)
    for k in ("exec_timeout", "max_rows", "max_col_perms", "comparison", "dataset_md5"):
        assert k in params, k


# --------------------------------------------------------------- the real BIRD archive ----


@needs_dataset
def test_the_real_releases_load_with_the_expected_shape():
    """Counts asserted against what was measured on download, so a re-download is noticed."""
    dev = sb.load("bird_dev")
    mini = sb.load("bird_mini_dev")
    assert len(dev) == 1534
    assert len(mini) == 500
    assert len({p["db_id"] for p in dev}) == 11
    assert len({p["db_id"] for p in mini}) == 11
    assert all(Path(p["db_path"]).exists() for p in dev)


@needs_dataset
def test_mini_dev_is_not_a_filter_of_dev_because_its_golds_differ():
    """Building mini-dev as an id split would score 14 questions against superseded golds."""
    dev = {p["question_id"]: p for p in sb.load("bird_dev")}
    mini = sb.load("bird_mini_dev")
    assert all(p["question_id"] in dev for p in mini)
    differing = [p["question_id"] for p in mini
                 if dev[p["question_id"]]["gold_sql"].strip() != p["gold_sql"].strip()]
    assert len(differing) == 14, differing


@needs_dataset
def test_a_sample_of_real_golds_self_verifies():
    """The full 1534 + 500 gold self-check lives in sql_selfcheck.py; this is the tripwire."""
    problems = sb.load("bird_mini_dev")[::25]
    recs = sb.grade_many(problems, [p["gold_sql"] for p in problems], workers=4)
    bad = [(r["question_id"], r["status"], r["detail"][:120]) for r in recs
           if not r["passed"]]
    assert not bad, bad


@needs_dataset
def test_no_real_gold_returns_an_empty_result():
    """An empty gold is ungradeable: every prediction returning nothing scores correct."""
    problems = sb.load("bird_mini_dev")[::25]
    recs = sb.grade_many(problems, [p["gold_sql"] for p in problems], workers=4)
    assert [r["question_id"] for r in recs if r["gold_rows"] == 0] == []
