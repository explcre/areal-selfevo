#!/usr/bin/env python3
"""Score a served model on BIRD text-to-SQL by EXECUTING the SQL it writes.

Third domain beside :mod:`math_bench` and :mod:`code_bench`, and deliberately the same
shape: a ``SUITE`` of frozen benchmarks, generation through the sglang OpenAI-compatible
endpoint, per-question records written to disk, Wilson intervals, and every excluded or
failed sample counted in the output rather than dropped. What differs is again the grader.
A math answer is a string compared to a string and a code answer is a program that has to
be run; a SQL answer is a query that has to be run AGAINST A DATABASE, and the whole
question is what "the same answer" means once two queries return two tables.

DATASET
-------
BIRD dev, ``dev.zip`` from ``bird-bench.oss-cn-beijing.aliyuncs.com`` (linked from
https://bird-bench.github.io/), unpacked under ``$BIRD_DATA`` (default ``~/evaldata/bird``).
MEASURED on this box 2026-09-02, by reading the files rather than trusting the downloader:

* ``bird_dev``       -- ``dev_20240627/dev.json``, **1534** questions, 11 databases,
  difficulty simple 925 / moderate 464 / challenging 145.
* ``bird_mini_dev``  -- ``minidev_json/mini_dev_sqlite.json``, **500** questions, the same
  11 databases, difficulty simple 148 / moderate 250 / challenging 102.

``dev_databases/`` holds **11** ``.sqlite`` files totalling **1,493,538,934 bytes**
(1.4 GiB), from 232 KiB (superhero) to 570 MiB (european_football_2).

**Mini-dev is NOT a filter of dev.json and must not be built as one.** All 500 of its
``question_id`` values do occur in ``dev.json``, but 4 of its questions and **14 of its gold
SQL strings differ** from the dev.json row with the same id -- mini-dev ships its own
revised golds. So it is loaded from its own file, exactly as a separate release, and the
alternative (a committed id split, the shape ``math500_split.json`` uses) would have scored
14 questions against superseded golds while reporting the mini-dev name.

This matters for one specific comparison. Our anchor for Qwen2.5-32B-Instruct is 43.8% EX
on BIRD **Mini-Dev** (arXiv 2511.04153), with Qwen2.5-Coder-32B at 49.4 and the leaderboard
top near 82. ``bird_mini_dev`` is the benchmark that number is comparable to.

GRADING: EXECUTION ACCURACY
---------------------------
Run the predicted SQL and the gold SQL against the same database and compare the two result
sets. The comparison is the entire benchmark, so it is spelled out:

* **Row order is ignored unless the gold query orders its output.** Two result sets are
  compared as MULTISETS -- duplicate rows count -- except when the gold query carries a
  TOP-LEVEL ``ORDER BY``, in which case the row sequence must match exactly.
  :func:`order_matters` decides this after blanking string literals, quoted identifiers and
  comments, and after blanking everything at parenthesis depth > 0, so an ``ORDER BY``
  inside a subquery or a CTE -- which does not order the OUTER result -- does not impose an
  ordering requirement. MEASURED: **276 of 1534** dev golds and **88 of 500** mini-dev golds
  order their output under this rule. Spider's ``exec_eval`` uses the cruder
  ``"order by" in sql.lower()`` over the raw text, which counts **305** and **106** -- the
  two rules disagree on **29** dev and **18** mini-dev golds, every one of them an ORDER BY
  that does not order the returned rows. The rule also bites less often than either count
  suggests: only **55 of 276** dev and **18 of 88** mini-dev order-sensitive golds return
  more than one row, and a single-row result has only one order.

* **Column order and column naming are ignored.** Column NAMES are never read: a result set
  here is a list of value tuples. Column ORDER is quantified over -- a prediction matches if
  ANY permutation of its columns matches gold. The permutations searched are not all ``n!``
  of them: gold column ``i`` can only correspond to a predicted column whose multiset of
  values is equal to gold column ``i``'s, which is an exact necessary condition (a column
  permutation composed with a row permutation preserves each column's multiset), so the
  candidate sets are computed first and only their product is walked. The identity
  permutation is tried first because it is what almost every correct answer uses.
  MEASURED column counts: 1-6 on both releases (dev: 1255 single-column of 1534), so the
  search is small in practice; ``--max-col-perms`` bounds it anyway and a run that hits the
  bound reports ``n_perm_exhausted`` instead of silently grading the row wrong.

* **A query that errors, times out or returns the wrong rows is WRONG, not an exception.**
  It is bucketed by how it failed, it stays in the denominator, and it never aborts the run.

* **Every execution has a wall-clock timeout** -- ``--exec-timeout``, default **30.0 s**,
  which is the ``meta_time_out`` default of BIRD's own ``evaluation_ex.py``. The slowest
  gold measured here is 3.99 s (dev question 596), so the limit carries 7.5x headroom over
  the slowest query the dataset itself needs. Timeouts are counted in their own bucket.
  The limit is enforced by KILLING a child process, not by a thread that asks politely: a
  runaway ``CROSS JOIN`` does not return to check a flag.

* **Every database is opened READ-ONLY.** ``file:...?mode=ro`` plus an authorizer that
  denies ``ATTACH``/``DETACH``, so a generated query cannot write the fixture and cannot
  reach a second file to write that instead. The fixture is 1.4 GiB of downloaded SQLite
  that nothing in this repo can regenerate.

DIVERGENCE FROM THE OFFICIAL SCRIPT, AND WHY BOTH NUMBERS ARE REPORTED
----------------------------------------------------------------------
BIRD's ``evaluation_ex.py`` grades with exactly this::

    def calculate_ex(predicted_res, ground_truth_res):
        return 1 if set(predicted_res) == set(ground_truth_res) else 0

That is a SET comparison of raw tuples. It differs from the rules above in three ways, and
the differences do not point the same way:

1. it ignores DUPLICATE rows, which is LOOSER than the multiset rule here (MEASURED: 132 of
   1534 dev golds and 33 of 500 mini-dev golds return a result containing duplicate rows, so
   the two rules can disagree on those and only those);
2. it ignores row ORDER even for an ``ORDER BY`` gold, which is LOOSER;
3. it compares tuples positionally, so column order matters, which is STRICTER.

A grader that silently picked either side would produce a number that is not the published
number and does not say so. So both verdicts are computed from the SAME executions and both
are reported: ``accuracy`` under the rules above, and ``accuracy_official`` under
``calculate_ex`` reproduced byte-for-byte in :func:`official_equal`. ``n_verdict_differs``
counts the rows where they disagree. Compare our arms with ``accuracy``; compare against a
published BIRD score with ``accuracy_official``.

FAILURE ACCOUNTING
------------------
Every question lands in exactly one bucket and every bucket is printed:

``pass`` ``wrong_answer`` ``exec_error`` ``timeout`` ``no_sql`` ``row_limit``
``gold_broken`` ``harness_error`` ``gen_failed``

Only ``gen_failed`` (the endpoint returned nothing after retries) leaves the denominator of
``accuracy``, exactly as math_bench and code_bench do, and because excluding it biases the
score UPWARD both numbers are reported: ``accuracy`` over graded questions and
``accuracy_all`` over all questions with generation failures counted wrong.

``gold_broken`` and ``harness_error`` are scored as FAILS so they cannot inflate anything,
and both make the process exit non-zero, because a grader that is quietly malfunctioning is
worse than one that is loudly down. ``gold_broken`` specifically means the DATASET's own
query failed on the dataset's own database; the row is then ungradeable and no verdict about
the model can be read from it.

GOLD: what verifies this harness
--------------------------------
Unlike LiveCodeBench, BIRD ships the reference answer, so the OlympiadBench move is
available directly: run every gold through this grader against its own database and require
100%. ``sql_selfcheck.py`` does that and three more things, and exits non-zero on any
deviation. MEASURED 2026-09-02 (numbers restated in ``EXPERIMENTS.md``):

* **GOLD SELF-CHECK -- 1534/1534 dev, 500/500 mini-dev.** Zero broken rows: zero golds fail
  to execute, zero time out, and -- worth stating because it is the failure that would make
  a benchmark ungradeable without looking broken -- zero golds return an EMPTY result. An
  empty gold is silently ungradeable: any prediction returning nothing, including a
  ``WHERE 1=0``, scores correct against it. There are none here, and that was measured
  rather than assumed.
* **EQUIVALENCE -- gold must still pass when the answer is rearranged.** Self-checking gold
  against itself is nearly vacuous on its own: it is the same bytes on both sides and passes
  under any comparator that is reflexive, including one that ignores row values entirely. So
  the check also grades each gold against a REWRITTEN form of itself that returns the same
  rows in a different order, and requires those to pass too, which no reflexive-only
  comparator does.
* **KNOWN-WRONG BATTERY -- every one must FAIL.** A grader that passes everything is worse
  than no grader.
* **MUTATION -- ``mutate_sql_bench.py``.**

WHAT IS AND IS NOT VALIDATED HERE
----------------------------------
GRADING is verified end to end with no GPU and no endpoint: the golds, the equivalence set,
the known-wrong battery, and ``--from-generations`` driving the real CLI over recorded
completions.

GENERATION has NOT been exercised against a served model. The prompt is BIRD's own
(:func:`build_prompt`, transcribed from ``llm/src/prompt.py`` and ``table_schema.py``), and
the endpoint client, the model-identity check and the retry policy are math_bench's,
already exercised by the other two domains -- but no token has been generated against this
benchmark and no score exists. Do not read one into these files.

LICENCE -- carries into the paper
----------------------------------
The dev archive is a public, ungated download and is used on a run-and-cite basis. The
``bird-bench/mini_dev`` repository, from which the Mini-Dev question file and the reference
evaluation semantics come, has **NO LICENSE FILE** (its root holds only ``.gitignore``,
``README.md``, ``requirements.txt`` and directories; GitHub's licence API answers 404 and
the repository's ``license`` field is null, both checked 2026-09-02). Nothing here may
become a redistributed dependency: fetch it, cite it, do not vendor it. No BIRD data is
committed to this repository -- ``$BIRD_DATA`` is fetched per box.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import functools
import hashlib
import itertools
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reused rather than reimplemented: the endpoint client, the model-identity refusal and the
# interval are benchmark agnostic and are already tested against a misbehaving local server
# in test_math_bench.py. Copying them would have produced a third version to drift.
from math_bench import chat_url, generate, verify_model, wilson  # noqa: E402

# One entry per frozen SQL benchmark, mirroring math_bench.SUITE and code_bench.RELEASES.
# The value is the question file relative to $BIRD_DATA. Mini-dev is a separate FILE and not
# a filter over dev.json, because 14 of its 500 golds differ from the dev.json row carrying
# the same question_id -- see the module docstring.
RELEASES = {
    "bird_dev": "dev_20240627/dev.json",
    "bird_mini_dev": "minidev_json/mini_dev_sqlite.json",
}
SUITE = ["bird_mini_dev"]

# Where the unpacked archive lives. An environment variable rather than a constant, for the
# same reason math_bench uses MATH_EVAL_DATA: this path differs on every box.
DATA_ENV = "BIRD_DATA"
_DEFAULT_DATA = "~/evaldata/bird"
# Both releases score against the databases shipped in dev.zip. minidev.zip contains a
# byte-identical second copy of them and 1.9 GiB of MySQL/PostgreSQL dumps we do not use, so
# only its question file is unpacked.
DB_SUBDIR = "dev_20240627/dev_databases"

# Generation parameters that differ from the CLI defaults, per benchmark. Nothing differs
# yet; the table exists so that when one does it is a line here rather than an edit to a
# default that also moves the other benchmark.
BENCH_OVERRIDES: dict[str, dict] = {}

# Above this share of endpoint failures the run is not a measurement. Same rule and same
# rationale as math_bench.FAILED_RATE_ABORT.
FAILED_RATE_ABORT = 0.10
# Above this share of truncated generations the score measures the token budget.
CAP_LIMITED_RATE = 0.10
PROGRESS_EVERY_S = float(os.environ.get("PROGRESS_EVERY_S", "30"))

# Buckets. Exactly one per question, all of them printed, and every one except PASS and
# GEN_FAILED is a failure that stays in the denominator.
ST_PASS = "pass"
ST_WRONG = "wrong_answer"
ST_EXEC_ERROR = "exec_error"
ST_TIMEOUT = "timeout"
ST_NO_SQL = "no_sql"
ST_ROW_LIMIT = "row_limit"
ST_GOLD_BROKEN = "gold_broken"
ST_HARNESS = "harness_error"
ST_GEN_FAILED = "gen_failed"
STATUSES = (ST_PASS, ST_WRONG, ST_EXEC_ERROR, ST_TIMEOUT, ST_NO_SQL, ST_ROW_LIMIT,
            ST_GOLD_BROKEN, ST_HARNESS, ST_GEN_FAILED)

DIFFICULTIES = ("simple", "moderate", "challenging")

# Defaults set from measurement, not from round numbers -- the rule code_bench arrived at
# after a 1 MiB output cap silently subtracted four problems from its score.
#   exec timeout: BIRD's own evaluation_ex.py default; slowest gold measured here is 3.99 s.
#   max rows: the largest gold result measured here is 278,230 rows (dev question 1015-ish
#   scale); 1,000,000 is 3.6x that, and a result that exceeds it is reported, never trimmed.
DEFAULT_EXEC_TIMEOUT = 30.0
DEFAULT_MAX_ROWS = 1_000_000
DEFAULT_MAX_COL_PERMS = 5000


def _ids(args) -> set:
    """The ``--question-ids`` filter as a set of ints.

    Args:
        args: Parsed CLI namespace.

    Returns:
        The requested ``question_id`` values, empty when the flag was not given.
    """
    raw = getattr(args, "question_ids", "") or ""
    return {int(x) for x in re.split(r"[,\s]+", raw) if x.strip()}


# ------------------------------------------------------------------------ the dataset ----


def data_root() -> Path:
    """Directory holding the unpacked BIRD archive.

    Returns:
        ``$BIRD_DATA`` when set, otherwise ``~/evaldata/bird``.

    Raises:
        FileNotFoundError: When the directory does not exist. A missing dataset must stop
            the run: left unchecked it yields an empty question list, which scores zero and
            is indistinguishable in the output from a model that answered nothing right.
    """
    p = Path(os.path.expanduser(os.environ.get(DATA_ENV) or _DEFAULT_DATA))
    if not p.is_dir():
        raise FileNotFoundError(
            f"BIRD data root {p} is not a directory. Set {DATA_ENV}, or unpack dev.zip "
            f"from https://bird-bench.github.io/ there."
        )
    return p


def require_dataset(bench: str) -> Path:
    """Path to one release's question file, checked to exist.

    Args:
        bench: A key of :data:`RELEASES`.

    Returns:
        The path to the question JSON.

    Raises:
        ValueError: For an unknown benchmark name.
        FileNotFoundError: When the file is absent, naming what IS present so the fix is
            obvious from the message alone.
    """
    if bench not in RELEASES:
        raise ValueError(f"unknown sql benchmark {bench!r}; known: {sorted(RELEASES)}")
    root = data_root()
    f = root / RELEASES[bench]
    if not f.exists():
        present = sorted(str(p.relative_to(root)) for p in root.rglob("*.json")
                         if p.stat().st_size > 1000)[:10]
        raise FileNotFoundError(
            f"{f} not found. {root} holds {present or 'no question files'}; unpack "
            f"dev.zip (and, for mini-dev, mini_dev_sqlite.json out of minidev.zip) under "
            f"{DATA_ENV}."
        )
    return f


def database_path(db_id: str) -> Path:
    """Path to one BIRD database file.

    Args:
        db_id: The ``db_id`` field of a question.

    Returns:
        The ``.sqlite`` file for that database.

    Raises:
        FileNotFoundError: When it is absent. The databases are the benchmark; a question
            whose database is missing cannot be graded and must not be quietly skipped.
    """
    p = data_root() / DB_SUBDIR / db_id / f"{db_id}.sqlite"
    if not p.exists():
        raise FileNotFoundError(
            f"database {p} not found; unpack dev_databases.zip inside dev.zip"
        )
    return p


def load(bench: str, difficulty: str = "", limit: int = 0, question_ids=()) -> list:
    """Load a release's questions, normalised.

    Args:
        bench: A key of :data:`RELEASES`.
        difficulty: Keep only this difficulty (``simple``/``moderate``/``challenging``);
            empty keeps all. Recorded in the results row, since scoring a subset and
            reporting it as the benchmark is how a number stops meaning what it says.
        limit: Keep only the first N after filtering; 0 keeps all.
        question_ids: Keep only these ``question_id`` values; empty keeps all. Also
            recorded, and an id that matches nothing is fatal rather than ignored -- a typo
            would otherwise silently shrink the benchmark.

    Returns:
        A list of dicts with keys ``idx``, ``question_id``, ``db_id``, ``question``,
        ``evidence``, ``gold_sql``, ``difficulty`` and ``db_path``.

    Raises:
        ValueError: If a row lacks the expected schema, if a gold SQL is empty, if a
            difficulty label is not one this benchmark defines, or if the filters select
            nothing.
    """
    f = require_dataset(bench)
    rows = json.loads(f.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{bench}: {f} did not contain a non-empty list of questions")
    out = []
    for i, r in enumerate(rows):
        missing = [k for k in ("question_id", "db_id", "question", "SQL") if k not in r]
        if missing:
            raise ValueError(f"{bench} row {i}: missing {missing}")
        if not str(r["SQL"]).strip():
            # A blank gold makes the row ungradeable while looking like an ordinary row:
            # every prediction would be compared against a query that cannot run.
            raise ValueError(f"{bench} row {i} (qid {r['question_id']}): empty gold SQL")
        diff = r.get("difficulty", "")
        if diff and diff not in DIFFICULTIES:
            raise ValueError(
                f"{bench} row {i} (qid {r['question_id']}): difficulty {diff!r} is not one "
                f"of {DIFFICULTIES}; the per-difficulty table would silently omit it"
            )
        q = {
            "idx": i,
            "question_id": r["question_id"],
            "db_id": r["db_id"],
            "question": r["question"],
            "evidence": r.get("evidence", "") or "",
            "gold_sql": r["SQL"],
            "difficulty": diff,
            "db_path": str(database_path(r["db_id"])),
        }
        if difficulty and q["difficulty"] != difficulty:
            continue
        if question_ids and q["question_id"] not in question_ids:
            continue
        out.append(q)
    if question_ids:
        absent = sorted(set(question_ids) - {q["question_id"] for q in out})
        if absent:
            raise ValueError(f"{bench}: no such question_id: {absent}")
    if limit:
        out = out[:limit]
    if not out:
        raise ValueError(
            f"{bench}: no questions selected (difficulty={difficulty!r} "
            f"question_ids={len(question_ids)} limit={limit})"
        )
    return out


def dataset_md5(bench: str) -> str:
    """md5 of the question file, recorded so two scores can be shown to be of the same data."""
    return hashlib.md5(require_dataset(bench).read_bytes()).hexdigest()


# -------------------------------------------------------------------------- the prompt ----


@functools.lru_cache(maxsize=None)
def schema_prompt(db_path: str) -> str:
    """The ``CREATE TABLE`` statements for one database, as BIRD presents them.

    Transcribed from ``bird-bench/mini_dev``'s ``table_schema.py``
    (``generate_schema_prompt_sqlite`` with ``num_rows=None``): every table's DDL out of
    ``sqlite_master``, joined by blank lines, ``sqlite_sequence`` excluded. Reproduced
    rather than invented, for the reason code_bench states about LiveCodeBench's prompt --
    a different prompt is a different benchmark, and a score that is not comparable to the
    published numbers is much less useful than one that is.

    Args:
        db_path: The ``.sqlite`` file.

    Returns:
        The schema block of the prompt.
    """
    conn = sqlite3.connect(readonly_uri(Path(db_path)), uri=True)
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        ddls = []
        for name in names:
            if name == "sqlite_sequence":
                continue
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (name,)).fetchone()
            if row and row[0]:
                ddls.append(row[0])
    finally:
        conn.close()
    return "\n\n".join(ddls)


def build_prompt(problem: dict) -> str:
    """BIRD's own prompt for one question.

    Transcribed from ``llm/src/prompt.py::generate_combined_prompts_one`` for the SQLite
    dialect: schema, then the comment block (question BEFORE the external knowledge, which
    is that file's ordering and is kept), then the chain-of-thought line, then the
    instruction block, joined by blank lines.

    Args:
        problem: One entry from :func:`load`.

    Returns:
        The user message sent to the endpoint.
    """
    dialect = "SQLite"
    knowledge = problem.get("evidence") or ""
    knowledge_text = " and understanding External Knowledge" if knowledge else ""
    knowledge_prompt = f"-- External Knowledge: {knowledge}" if knowledge else ""
    comment = (
        f"-- Using valid {dialect}{knowledge_text}, answer the following questions for the "
        f"tables provided above.\n"
        f"-- {problem['question']}\n"
        f"{knowledge_prompt}"
    )
    cot = f"\nGenerate the {dialect} for the above question after thinking step by step: "
    instruction = f"""
        \nIn your response, you do not need to mention your intermediate steps.
        Do not include any comments in your response.
        Do not need to start with the symbol ```
        You only need to return the result {dialect} SQL code
        start from SELECT
        """
    return "\n\n".join([schema_prompt(problem["db_path"]), comment, cot, instruction])


# ---------------------------------------------------------------------- SQL extraction ----

_FENCE = re.compile(r"```(?P<lang>[A-Za-z0-9_+-]*)[ \t]*\r?\n(?P<body>.*?)```", re.S)
_SQL_TAGS = ("sql", "sqlite")
# A bare completion is accepted only when it BEGINS with a statement keyword. Anything else
# is refused rather than handed to the database, so prose can never become a query whose
# syntax error is then reported as the model's answer.
_BARE_START = re.compile(r"^\s*(?:WITH|SELECT)\b", re.I)


def extract_sql(text: str):
    """The SQL a completion actually submits, or ``None``.

    Precedence, highest first, mirroring :func:`code_bench.extract_code`:

    1. the LAST non-empty ```` ```sql ```` (or ``sqlite``) fenced block -- a model that
       explores before committing is graded on what it committed;
    2. the LAST non-empty fenced block of any other tag, once no tagged block has content;
    3. the whole completion, but only if it begins with ``SELECT`` or ``WITH``, which is the
       unfenced form BIRD's own instruction block asks for ("Do not need to start with the
       symbol ```", "start from SELECT").

    Anything else -- prose, an empty fence, an explanation with no query -- returns ``None``
    and is scored ``no_sql``. That is a FAIL and stays in the denominator; it is never an
    exclusion.

    Args:
        text: The raw completion.

    Returns:
        The SQL string, stripped, or ``None``.
    """
    if not text:
        return None
    blocks = [(m.group("lang").lower(), m.group("body")) for m in _FENCE.finditer(text)]
    tagged = [b for lang, b in blocks if lang in _SQL_TAGS]
    for level in (tagged, [b for _, b in blocks]):
        for body in reversed(level):
            if body.strip():
                return body.strip()
    if _BARE_START.match(text):
        return text.strip()
    return None


# ------------------------------------------------------------------ execution, read-only ----

# The child that actually touches a database. It is a separate PROCESS, spawned fork+exec
# through subprocess, for two reasons that a thread cannot give:
#   * a wall-clock limit that is enforced by SIGKILL. A runaway CROSS JOIN never returns to
#     check a flag, and sqlite3 releases the GIL inside execute(), so a timeout implemented
#     in the parent's own interpreter can bound nothing.
#   * blast radius. It runs with -I (isolated: no PYTHONPATH, no user site), opens the file
#     read-only, and denies ATTACH, so a generated query cannot write the 1.4 GiB fixture
#     and cannot reach a second file to write instead.
# It speaks JSON over stdin/stdout. Cells are passed through as JSON scalars so that Python's
# own numeric equality survives the round trip -- 5 and 5.0 must still compare equal, because
# BIRD's own grader compares raw sqlite3 tuples and would call CAST(x AS REAL) correct. The
# two values JSON cannot carry are tagged as two-element lists, which sqlite3 can never
# itself return: ["b", hex] for a BLOB and ["nf", name] for a non-finite float.
_CHILD_PROGRAM = r'''
import json, math, sqlite3, sys

def cell(v):
    if isinstance(v, bytes):
        return ["b", v.hex()]
    if isinstance(v, float) and not math.isfinite(v):
        return ["nf", "nan" if v != v else ("inf" if v > 0 else "-inf")]
    return v

req = json.loads(sys.stdin.read())
out = {"status": "ok", "rows": [], "ncols": 0, "truncated": False, "error": ""}
conn = None
try:
    conn = sqlite3.connect(req["uri"], uri=True, isolation_level=None)
    # SQLite's own numeric action codes are the fallbacks, not None. The module-level
    # constants only appeared in Python 3.11, and a getattr default of None here meant the
    # authorizer was silently NOT INSTALLED on an older interpreter -- a guard that
    # disappears on the boxes least likely to have anything else protecting them.
    deny = getattr(sqlite3, "SQLITE_DENY", 1)
    ok = getattr(sqlite3, "SQLITE_OK", 0)
    attach = getattr(sqlite3, "SQLITE_ATTACH", 24)
    detach = getattr(sqlite3, "SQLITE_DETACH", 25)

    def auth(action, a1, a2, dbname, source):
        if action in (attach, detach):
            return deny
        return ok
    conn.set_authorizer(auth)
    cur = conn.execute(req["sql"])
    out["ncols"] = len(cur.description) if cur.description else 0
    cap = req["max_rows"]
    rows = []
    while True:
        batch = cur.fetchmany(2000)
        if not batch:
            break
        rows.extend(batch)
        if len(rows) > cap:
            out["truncated"] = True
            rows = rows[:cap]
            break
    out["rows"] = [[cell(v) for v in r] for r in rows]
except BaseException as e:
    out = {"status": "error", "rows": [], "ncols": 0, "truncated": False,
           "error": type(e).__name__ + ": " + str(e)[:400]}
finally:
    if conn is not None:
        try:
            conn.close()
        except BaseException:
            pass
sys.stdout.write(json.dumps(out))
sys.stdout.flush()
'''


class _NonFinite:
    """A non-finite float that compares equal to another of the same name.

    ``float("nan") != float("nan")``, so a result set containing NaN would fail to match
    ITSELF and a correct prediction would be graded wrong for a reason that has nothing to
    do with SQL. Result cells are hashed and compared here (multisets, column multisets), so
    the sentinel defines both ``__eq__`` and ``__hash__``.

    Args:
        name: ``nan``, ``inf`` or ``-inf``.
    """

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, _NonFinite) and other.name == self.name

    def __hash__(self):
        return hash(("_NonFinite", self.name))

    def __repr__(self):
        return f"<{self.name}>"


def _decode_cell(v):
    """Turn one JSON cell from the child back into a comparable Python value.

    Args:
        v: A JSON scalar, or a ``["b", hex]`` / ``["nf", name]`` tag.

    Returns:
        ``bytes``, a :class:`_NonFinite`, or ``v`` unchanged.
    """
    if isinstance(v, list) and len(v) == 2:
        if v[0] == "b":
            return bytes.fromhex(v[1])
        if v[0] == "nf":
            return _NonFinite(v[1])
    return v


def readonly_uri(path: Path) -> str:
    """A ``file:`` URI that opens ``path`` read-only.

    Built with :func:`urllib.request.pathname2url` rather than by string concatenation:
    a database directory containing a space, a ``?`` or a ``#`` would otherwise produce a
    URI whose query string starts in the middle of the path, and sqlite would open (or fail
    to open) a different file than the one named.

    Args:
        path: The ``.sqlite`` file.

    Returns:
        ``file:<quoted path>?mode=ro``.
    """
    return "file:" + urllib.request.pathname2url(str(path)) + "?mode=ro"


class ExecResult:
    """One SQL execution against one database.

    Attributes:
        status: ``ok``, ``error``, ``timeout`` or ``harness``.
        rows: Result rows as tuples of decoded cells; empty unless ``status == "ok"``.
        ncols: Column count reported by the cursor.
        truncated: Whether the row cap was hit, which makes any comparison undecidable.
        error: The exception text, for ``error`` and ``harness``.
        seconds: Wall-clock time.
    """

    __slots__ = ("status", "rows", "ncols", "truncated", "error", "seconds")

    def __init__(self, status, rows=(), ncols=0, truncated=False, error="", seconds=0.0):
        self.status = status
        self.rows = list(rows)
        self.ncols = ncols
        self.truncated = truncated
        self.error = error
        self.seconds = seconds


def execute_sql(db_path, sql: str, timeout: float = DEFAULT_EXEC_TIMEOUT,
                max_rows: int = DEFAULT_MAX_ROWS) -> ExecResult:
    """Run one query read-only, under a wall-clock limit enforced by killing it.

    Args:
        db_path: The ``.sqlite`` file.
        sql: The query.
        timeout: Seconds before the child is killed.
        max_rows: Most rows read back. Exceeding it does not truncate the ANSWER, it marks
            the result undecidable -- see :func:`grade_prediction`.

    Returns:
        An :class:`ExecResult`. A query that raises is ``error``, not an exception here: a
        wrong query is an ordinary outcome of this benchmark and must never abort the run.
    """
    req = json.dumps({"uri": readonly_uri(Path(db_path)), "sql": sql,
                      "max_rows": int(max_rows)})
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "-I", "-c", _CHILD_PROGRAM],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    try:
        out, err = proc.communicate(req.encode(), timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return ExecResult("timeout", seconds=time.time() - t0)
    el = time.time() - t0
    if proc.returncode != 0 or not out:
        # The child died without answering: killed by the OOM killer, or a bug here. It is
        # NOT evidence about the query, so it is reported as a harness fault, which is loud
        # and makes the run exit non-zero.
        return ExecResult("harness", error=(err.decode(errors="replace")[:400]
                                            or f"child exited {proc.returncode}"),
                          seconds=el)
    try:
        d = json.loads(out.decode())
    except Exception as e:
        return ExecResult("harness", error=f"unparseable child output: {e}", seconds=el)
    if d["status"] != "ok":
        return ExecResult("error", error=d["error"], seconds=el)
    rows = [tuple(_decode_cell(v) for v in r) for r in d["rows"]]
    return ExecResult("ok", rows=rows, ncols=d["ncols"], truncated=d["truncated"],
                      seconds=el)


# ------------------------------------------------------------------------ the comparison ----

# Blanked before any keyword is looked for: string literals, all three identifier quotings,
# and both comment forms. The literal patterns are the linear-time forms -- `'[^']*(?:''[^']*)*'`
# rather than `'(?:[^']|'')*'` -- because the naive alternation backtracks exponentially on
# an unterminated quote, which a generated query supplies sooner or later.
_SQL_NOISE = re.compile(
    r"""'[^']*(?:''[^']*)*'      # single-quoted string
      | "[^"]*(?:""[^"]*)*"      # double-quoted identifier
      | `[^`]*`                  # backtick identifier
      | \[[^\]]*\]               # bracket identifier
      | --[^\n]*                 # line comment
      | /\*.*?\*/                # block comment
    """,
    re.S | re.X,
)
_ORDER_BY = re.compile(r"\border\s+by\b", re.I)


def _blank_noise(sql: str) -> str:
    """Replace literals, quoted identifiers and comments with spaces of the same length.

    Same length so that positions are preserved and the parenthesis scan that follows stays
    aligned with the original text.

    Args:
        sql: A query.

    Returns:
        The query with every non-code region blanked.
    """
    return _SQL_NOISE.sub(lambda m: " " * len(m.group(0)), sql)


def _top_level_only(sql: str) -> str:
    """Blank everything nested inside parentheses, keeping depth-0 text.

    Args:
        sql: A query whose literals have already been blanked.

    Returns:
        The query with every parenthesised region replaced by spaces.
    """
    out = []
    depth = 0
    for ch in sql:
        if ch == "(":
            depth += 1
            out.append(" ")
        elif ch == ")":
            depth = max(0, depth - 1)
            out.append(" ")
        else:
            out.append(ch if depth == 0 else " ")
    return "".join(out)


def order_matters(gold_sql: str) -> bool:
    """Whether the gold query's own row ORDER is part of the answer.

    True exactly when the gold carries a TOP-LEVEL ``ORDER BY`` -- one that orders the rows
    the query returns. An ``ORDER BY`` inside a subquery, a CTE or a scalar expression does
    not order the outer result, and demanding an order the gold does not actually produce
    would mark correct predictions wrong. String literals, quoted identifiers and comments
    are blanked first, so a value like ``'order by'`` in a WHERE clause is not a keyword.

    This is stricter than BIRD's own grader, which ignores order entirely, and more precise
    than Spider's ``exec_eval``, which tests ``"order by" in sql.lower()`` over the raw text.

    Args:
        gold_sql: The dataset's reference query.

    Returns:
        Whether row order must match.
    """
    return _ORDER_BY.search(_top_level_only(_blank_noise(gold_sql))) is not None


def _column_candidates(gold_rows, pred_rows, ncols):
    """For each gold column, the predicted columns it could correspond to.

    A column permutation composed with any row permutation leaves each column's MULTISET of
    values unchanged, so gold column ``i`` can only map to a predicted column with an equal
    value multiset. That is an exact necessary condition, not a heuristic: the true
    permutation always survives it, so pruning with it can never turn a correct answer into
    a wrong one. It replaces the randomised sampling Spider's ``exec_eval`` uses, whose
    output depends on ``random.choice`` and is therefore not reproducible.

    Args:
        gold_rows: Gold result rows.
        pred_rows: Predicted result rows.
        ncols: Column count, equal on both sides.

    Returns:
        A list of ``ncols`` tuples of candidate predicted-column indices.
    """
    gold_ms = [collections.Counter(r[i] for r in gold_rows) for i in range(ncols)]
    pred_ms = [collections.Counter(r[j] for r in pred_rows) for j in range(ncols)]
    return [tuple(j for j in range(ncols) if pred_ms[j] == gold_ms[i])
            for i in range(ncols)]


def _matches(gold_rows, pred_rows, perm, ordered: bool) -> bool:
    """Compare two result sets under one column permutation.

    Args:
        gold_rows: Gold rows.
        pred_rows: Predicted rows.
        perm: ``perm[i]`` is the predicted column matched to gold column ``i``.
        ordered: Whether row sequence must match; otherwise rows compare as a MULTISET, so
            duplicate rows count.

    Returns:
        Whether they match.
    """
    permuted = [tuple(r[j] for j in perm) for r in pred_rows]
    if ordered:
        return list(gold_rows) == permuted
    return collections.Counter(gold_rows) == collections.Counter(permuted)


def results_equal(gold_rows, pred_rows, ordered: bool,
                  max_perms: int = DEFAULT_MAX_COL_PERMS) -> tuple:
    """Execution-accuracy comparison of two result sets.

    Row order is ignored unless ``ordered``; rows otherwise compare as multisets, so a
    prediction that duplicates or drops a repeated row is wrong. Column order is quantified
    over. Column NAMES are never consulted.

    Args:
        gold_rows: Gold result rows.
        pred_rows: Predicted result rows.
        ordered: From :func:`order_matters` on the GOLD query.
        max_perms: Most column permutations to try before giving up.

    Returns:
        ``(equal, budget_exhausted)``. ``budget_exhausted`` is True only when the search
        stopped early, in which case the verdict is "not shown equal" rather than "shown
        unequal" and the run reports the count separately instead of quietly scoring it
        wrong.
    """
    if not gold_rows and not pred_rows:
        # Both empty. BIRD's own grader calls this equal without looking at column counts,
        # and so does this one. It is unreachable on dev and mini-dev, where every gold
        # returns at least one row (measured), but it is the shape that would make a
        # benchmark ungradeable and it is asserted rather than left to chance.
        return True, False
    if len(gold_rows) != len(pred_rows):
        return False, False
    ncols = len(gold_rows[0])
    if len(pred_rows[0]) != ncols:
        return False, False
    identity = tuple(range(ncols))
    if _matches(gold_rows, pred_rows, identity, ordered):
        return True, False
    tried = 0
    for perm in itertools.product(*_column_candidates(gold_rows, pred_rows, ncols)):
        if perm == identity or len(set(perm)) != ncols:
            continue
        tried += 1
        if tried > max_perms:
            return False, True
        if _matches(gold_rows, pred_rows, perm, ordered):
            return True, False
    return False, False


def official_equal(gold_rows, pred_rows) -> bool:
    """BIRD's own verdict, reproduced so both numbers can be reported from one run.

    ``evaluation_ex.py::calculate_ex`` is exactly ``set(pred) == set(gold)``: it ignores
    duplicate rows and row order, and compares tuples positionally so column order matters.
    Reproduced rather than approximated -- ``accuracy_official`` is what a published BIRD
    score is comparable to, and ``accuracy`` is what our arms should be compared to.

    Args:
        gold_rows: Gold result rows.
        pred_rows: Predicted result rows.

    Returns:
        The official verdict.
    """
    return set(gold_rows) == set(pred_rows)


# ------------------------------------------------------------------------- the grader ----


def grade_prediction(problem: dict, sql, timeout: float = DEFAULT_EXEC_TIMEOUT,
                     max_rows: int = DEFAULT_MAX_ROWS,
                     max_perms: int = DEFAULT_MAX_COL_PERMS) -> dict:
    """Grade one predicted query by executing it beside its gold.

    The gold runs first, so a dataset row whose own query is broken is reported as a broken
    ROW rather than charged to the model.

    Args:
        problem: One entry from :func:`load`.
        sql: The extracted prediction, or ``None``/empty when nothing was extractable.
        timeout: Wall-clock seconds per execution.
        max_rows: Row cap per execution.
        max_perms: Column-permutation budget.

    Returns:
        A record with ``status`` (one of :data:`STATUSES`), ``passed``, ``passed_official``,
        the two row counts, both execution times and any error text.
    """
    rec = {
        "idx": problem["idx"],
        "question_id": problem["question_id"],
        "db_id": problem["db_id"],
        "difficulty": problem.get("difficulty", ""),
        "order_matters": order_matters(problem["gold_sql"]),
        "status": ST_WRONG,
        "passed": False,
        "passed_official": False,
        "perm_exhausted": False,
        "gold_rows": None,
        "pred_rows": None,
        "gold_seconds": None,
        "pred_seconds": None,
        "detail": "",
    }
    gold = execute_sql(problem["db_path"], problem["gold_sql"], timeout, max_rows)
    rec["gold_seconds"] = round(gold.seconds, 3)
    if gold.status != "ok":
        # The dataset's own query failed on the dataset's own database. Nothing about the
        # model can be read from this row.
        rec["status"] = ST_GOLD_BROKEN
        rec["detail"] = f"gold {gold.status}: {gold.error}"[:400]
        return rec
    rec["gold_rows"] = len(gold.rows)
    if gold.truncated:
        rec["status"] = ST_ROW_LIMIT
        rec["detail"] = (f"gold returned more than max_rows={max_rows}; the comparison is "
                         f"not decidable")
        return rec
    if not sql or not str(sql).strip():
        rec["status"] = ST_NO_SQL
        rec["detail"] = "no SQL could be extracted from the completion"
        return rec

    pred = execute_sql(problem["db_path"], sql, timeout, max_rows)
    rec["pred_seconds"] = round(pred.seconds, 3)
    if pred.status == "timeout":
        rec["status"] = ST_TIMEOUT
        rec["detail"] = f"predicted query exceeded {timeout}s"
        return rec
    if pred.status == "error":
        rec["status"] = ST_EXEC_ERROR
        rec["detail"] = pred.error[:400]
        return rec
    if pred.status == "harness":
        rec["status"] = ST_HARNESS
        rec["detail"] = pred.error[:400]
        return rec
    rec["pred_rows"] = len(pred.rows)
    if pred.truncated:
        # Gold is not truncated (checked above), so the prediction returned strictly more
        # rows than gold has and is wrong on row count alone. No cap-dependent guessing.
        rec["status"] = ST_WRONG
        rec["detail"] = f"predicted query returned more than max_rows={max_rows} rows"
        return rec

    equal, exhausted = results_equal(gold.rows, pred.rows, rec["order_matters"], max_perms)
    rec["perm_exhausted"] = exhausted
    rec["passed"] = equal
    rec["passed_official"] = official_equal(gold.rows, pred.rows)
    rec["status"] = ST_PASS if equal else ST_WRONG
    if exhausted:
        rec["detail"] = (f"column-permutation budget {max_perms} exhausted; not shown "
                         f"equal rather than shown unequal")
    return rec


def grade_many(problems: list, sqls: list, workers: int = 8, **kw) -> list:
    """Grade a whole benchmark, one thread per concurrent pair of executions.

    Args:
        problems: Questions from :func:`load`.
        sqls: Extracted predictions, aligned with ``problems``.
        workers: Concurrent gradings. Each holds a full result set in memory, so this is
            deliberately lower than code_bench's sandbox count.
        **kw: Passed to :func:`grade_prediction`.

    Returns:
        Records in problem order.

    Raises:
        ValueError: If the two lists differ in length -- a misalignment would grade every
            question against the wrong answer and still produce a plausible score.
    """
    if len(problems) != len(sqls):
        raise ValueError(f"{len(problems)} problems but {len(sqls)} predictions")
    out = [None] * len(problems)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(grade_prediction, p, s, **kw): i
                for i, (p, s) in enumerate(zip(problems, sqls))}
        for fut, i in futs.items():
            out[i] = fut.result()
    return out


def summarise(bench: str, problems: list, records: list, params: dict) -> dict:
    """Aggregate per-question verdicts into a results row.

    Every bucket is counted and reported. ``accuracy`` excludes questions whose GENERATION
    failed, matching math_bench and code_bench, and because that exclusion biases the score
    upward ``accuracy_all`` reports the same passes over every question. ``accuracy_official``
    is the same executions scored by BIRD's own ``calculate_ex``.

    Args:
        bench: Benchmark name.
        problems: The questions scored.
        records: Verdicts from :func:`grade_prediction`, aligned with ``problems``.
        params: Generation and grading parameters to record for provenance.

    Returns:
        The results row.
    """
    counts = {s: 0 for s in STATUSES}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    n_problems = len(problems)
    n_graded = n_problems - counts[ST_GEN_FAILED]
    n_pass = counts[ST_PASS]
    n_official = sum(1 for r in records if r.get("passed_official"))
    acc = (n_pass / n_graded) if n_graded else float("nan")
    lo, hi = wilson(n_pass, n_graded) if n_graded else (float("nan"), float("nan"))
    n_trunc = sum(1 for r in records if r.get("finish_reason") == "length")
    by_diff = {}
    for d in DIFFICULTIES:
        sel = [r for r in records if r.get("difficulty") == d]
        gr = [r for r in sel if r["status"] != ST_GEN_FAILED]
        by_diff[d] = {
            "n": len(sel),
            "n_graded": len(gr),
            "n_pass": sum(1 for r in gr if r["passed"]),
            "accuracy": (sum(1 for r in gr if r["passed"]) / len(gr)) if gr
                        else float("nan"),
        }
    return {
        "benchmark": bench,
        "n_problems": n_problems,
        "n_graded": n_graded,
        "n_pass": n_pass,
        "n_failed": counts[ST_GEN_FAILED],
        "n_truncated": n_trunc,
        "n_no_sql": counts[ST_NO_SQL],
        "n_wrong_answer": counts[ST_WRONG],
        "n_exec_error": counts[ST_EXEC_ERROR],
        "n_timeout": counts[ST_TIMEOUT],
        "n_row_limit": counts[ST_ROW_LIMIT],
        "n_gold_broken": counts[ST_GOLD_BROKEN],
        "n_harness_error": counts[ST_HARNESS],
        "n_perm_exhausted": sum(1 for r in records if r.get("perm_exhausted")),
        "accuracy": acc,
        "accuracy_all": (n_pass / n_problems) if n_problems else float("nan"),
        "wilson_lo": lo,
        "wilson_hi": hi,
        # BIRD's own semantics over the same executions, so this row can be compared with a
        # published BIRD number without re-running anything.
        "n_pass_official": n_official,
        "accuracy_official": (n_official / n_graded) if n_graded else float("nan"),
        "n_verdict_differs": sum(1 for r in records
                                 if r["passed"] != r.get("passed_official")),
        "by_difficulty": by_diff,
        "truncation_rate": (n_trunc / n_problems) if n_problems else float("nan"),
        "cap_limited": bool(n_problems) and (n_trunc / n_problems) > CAP_LIMITED_RATE,
        "params": dict(params),
        "counts": counts,
    }


def resolve_params(bench: str, args, explicit=frozenset()) -> dict:
    """Generation and grading parameters this benchmark actually runs with.

    :data:`BENCH_OVERRIDES` states only what differs from the CLI defaults, and an explicitly
    named CLI flag outranks the table -- without that distinction a deliberate ``--max-tokens``
    is discarded while the run still reports success.

    Args:
        bench: Benchmark name.
        args: Parsed CLI namespace.
        explicit: Flag names the caller named on the command line.

    Returns:
        The resolved parameter dict, recorded in the results row.
    """
    out = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "n": 1,
        "concurrency": args.concurrency,
        "timeout": args.timeout,
        "seed": args.seed,
        "exec_timeout": args.exec_timeout,
        "max_rows": args.max_rows,
        "max_col_perms": args.max_col_perms,
        "difficulty": args.difficulty,
        "question_ids": sorted(_ids(args)) or "all",
        "dataset_md5": dataset_md5(bench),
        "comparison": "multiset rows, order-sensitive iff gold has a top-level ORDER BY, "
                      "column order quantified over, column names ignored",
    }
    for k, v in BENCH_OVERRIDES.get(bench, {}).items():
        if k not in explicit:
            out[k] = v
    return out


async def _generate_all(bench, problems, args, params) -> list:
    """Fetch one completion per question from the endpoint, in question order.

    Args:
        bench: Benchmark name, for the progress line.
        problems: Questions from :func:`load`.
        args: Parsed CLI namespace.
        params: Resolved generation parameters.

    Returns:
        A list of ``{"text", "finish_reason", "status"}`` aligned with ``problems``.
    """
    import aiohttp

    url = chat_url(args.base_url)
    sem = asyncio.Semaphore(params["concurrency"])
    conn = aiohttp.TCPConnector(limit=params["concurrency"])
    results = [None] * len(problems)

    async with aiohttp.ClientSession(connector=conn) as session:
        # BEFORE any generation: refuse a model id this endpoint does not serve, and fold
        # the resolved id, the URL and the served list into the parameters this row reports.
        # Without the first half the run scores whatever weights happen to be loaded;
        # without the second half nobody can tell afterwards which ones those were.
        params.update(await verify_model(session, args.base_url,
                                         getattr(args, "model", None)))

        async def one(i, p):
            async with sem:
                results[i] = await generate(session, url, args.model, build_prompt(p),
                                            params)

        tasks = [one(i, p) for i, p in enumerate(problems)]
        done, t0, nxt = 0, time.time(), 0.0
        for fut in asyncio.as_completed(tasks):
            await fut
            done += 1
            el = time.time() - t0
            if el >= nxt or done == len(tasks):
                nxt = el + PROGRESS_EVERY_S
                print(f"  {bench} generate {done}/{len(tasks)}  {el:.0f}s",
                      file=sys.stderr, flush=True)
    return results


def score_completions(bench, problems, completions, args, params, workers=8) -> tuple:
    """Grade a set of completions and return ``(per-question records, results row)``.

    Separated from generation on purpose: it is what lets the whole grading path be exercised
    from recorded completions with no model running, which is the only way this harness could
    be validated on a box whose GPUs must not be touched.

    Args:
        bench: Benchmark name.
        problems: Questions from :func:`load`.
        completions: ``{"text", "finish_reason", "status"}`` aligned with ``problems``.
        args: Parsed CLI namespace.
        params: Resolved parameters.
        workers: Concurrent gradings.

    Returns:
        ``(records, row)``.
    """
    sqls = [extract_sql(c["text"]) for c in completions]
    records = grade_many(problems, sqls, workers=workers, timeout=args.exec_timeout,
                         max_rows=args.max_rows, max_perms=args.max_col_perms)
    for rec, comp, sql in zip(records, completions, sqls):
        rec["finish_reason"] = comp.get("finish_reason")
        rec["gen_status"] = comp.get("status", "ok")
        rec["benchmark"] = bench
        rec["sql_sha256"] = hashlib.sha256((sql or "").encode()).hexdigest()
        rec["sql_chars"] = len(sql or "")
        if rec["gen_status"] == "failed":
            # The endpoint never answered. That is not a statement about the model's SQL, so
            # it is bucketed separately -- and because excluding it biases the score upward,
            # summarise() reports accuracy_all alongside accuracy.
            rec["status"] = ST_GEN_FAILED
            rec["passed"] = False
            rec["passed_official"] = False
            rec["detail"] = "endpoint returned nothing after retries"
    return records, summarise(bench, problems, records, params)


def json_safe(obj):
    """Replace every non-finite float with ``None``, at any depth.

    ``json.dumps(..., allow_nan=False)`` refuses NaN, which is the behaviour we want -- a
    results file containing the token ``NaN`` is not JSON and every downstream reader breaks
    on it. But the scrub has to be RECURSIVE. A first version replaced non-finite floats only
    at the top level of a results row and crashed on the per-difficulty table, where a
    difficulty with nothing graded carries a NaN accuracy two levels down. The crash was the
    good outcome; the bad one was one line away, because a ``default=`` handler would have
    turned it into a silent null.

    Args:
        obj: Any JSON-serialisable structure.

    Returns:
        The same structure with non-finite floats replaced by ``None``.
    """
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def write_records(path, records) -> None:
    """Write per-question verdicts as JSONL.

    An aggregate with no per-question record cannot be audited later, and every number this
    project has had to re-examine was one that had not been persisted.

    Args:
        path: Destination file.
        records: Verdicts.
    """
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def load_generations(path, problems) -> list:
    """Read recorded completions and align them with ``problems``.

    Args:
        path: JSONL written by ``--gen-out``.
        problems: Questions from :func:`load`.

    Returns:
        Completions aligned with ``problems``. A question with no recorded completion is
        filled in as a generation FAILURE, so it stays in ``n_problems`` and is counted --
        a missing generation must not silently shorten the benchmark.

    Raises:
        ValueError: If the file holds no completions at all.
    """
    by_qid = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            by_qid[d.get("question_id")] = d
    if not by_qid:
        raise ValueError(f"{path} contains no completions")
    out = []
    for p in problems:
        d = by_qid.get(p["question_id"])
        if d is None:
            out.append({"text": "", "finish_reason": None, "status": "failed"})
        else:
            out.append({"text": d.get("text", ""),
                        "finish_reason": d.get("finish_reason"),
                        "status": d.get("status", "ok")})
    return out


def build_parser() -> argparse.ArgumentParser:
    """The command-line parser this script actually runs with."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8404/v1")
    ap.add_argument("--model", default=None,
                    help="model id to score, as listed by <base-url>/models. Required "
                         "unless --from-generations, and deliberately without a default: "
                         "an unregistered id is answered HTTP 200 by the BASE model, so a "
                         "default silently scores the wrong weights.")
    ap.add_argument("--benchmarks", default=",".join(SUITE),
                    help=f"comma-separated, from {sorted(RELEASES)}")
    ap.add_argument("--limit", type=int, default=0, help="questions per benchmark, 0 = all")
    ap.add_argument("--difficulty", default="", choices=["", *DIFFICULTIES])
    ap.add_argument("--question-ids", default="",
                    help="comma-separated question_id values to score. Scoring a subset and "
                         "calling it the benchmark is how a number stops meaning what it "
                         "says, so the selection is recorded in the results row.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent gradings; each holds a full result set in memory")
    ap.add_argument("--exec-timeout", type=float, default=DEFAULT_EXEC_TIMEOUT,
                    help="hard wall-clock limit per SQL execution, enforced by killing the "
                         "child. Default matches BIRD's own evaluation_ex.py meta_time_out; "
                         "the slowest gold measured here is 3.99s.")
    ap.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS,
                    help="most rows read back per execution. The largest gold result "
                         "measured is 278,230 rows; a result above the cap is reported as "
                         "undecidable, never trimmed and compared anyway.")
    ap.add_argument("--max-col-perms", type=int, default=DEFAULT_MAX_COL_PERMS,
                    help="column-permutation budget per comparison. Exhausting it is "
                         "counted and reported, not silently scored wrong.")
    ap.add_argument("--from-generations", default="",
                    help="grade a recorded JSONL of completions instead of calling a model; "
                         "needs no GPU and no endpoint")
    ap.add_argument("--out", default="", help="results JSON")
    ap.add_argument("--gen-out", default="", help="JSONL of every completion")
    ap.add_argument("--records-out", default="",
                    help="JSONL of per-question verdicts (defaults beside --out)")
    return ap


def main() -> int:
    """Run the requested SQL benchmarks and return a process exit status."""
    ap = build_parser()
    args = ap.parse_args()
    explicit = {a.lstrip("-").replace("-", "_") for a in sys.argv[1:] if a.startswith("--")}
    # Refused here, before the dataset is loaded and before the endpoint is touched at all:
    # a missing flag should be reported as a missing flag. --from-generations grades recorded
    # text and needs no model, so it is exempt.
    if not args.from_generations and not args.model:
        raise SystemExit(
            "ERROR: --model is required when generating. It names which weights answer: an "
            "id the server does not serve is answered HTTP 200 by the BASE model, so there "
            "is no safe default. Pass an id from <base-url>/models, or use "
            "--from-generations to grade recorded completions."
        )
    rows, exit_code = [], 0
    gen_fh = open(args.gen_out, "w") if args.gen_out else None
    try:
        for bench in [b.strip() for b in args.benchmarks.split(",") if b.strip()]:
            t0 = time.time()
            problems = load(bench, difficulty=args.difficulty, limit=args.limit,
                            question_ids=_ids(args))
            params = resolve_params(bench, args, explicit)
            print(f"{bench}: {len(problems)} questions, "
                  f"{len({p['db_id'] for p in problems})} databases, "
                  f"exec_timeout={params['exec_timeout']}s", file=sys.stderr, flush=True)
            # Every row carries the same attribution schema, so "which model produced this
            # score" is answerable from the artifact alone -- and a regrade says so rather
            # than inheriting a model id it never called.
            params["generations_source"] = args.from_generations or None
            if args.from_generations:
                params.update({"model": None, "endpoint": None, "served_models": None})
                completions = load_generations(args.from_generations, problems)
            else:
                completions = asyncio.run(_generate_all(bench, problems, args, params))
            if gen_fh:
                for p, c in zip(problems, completions):
                    gen_fh.write(json.dumps({"benchmark": bench, "idx": p["idx"],
                                             "question_id": p["question_id"], **c}) + "\n")
                gen_fh.flush()
            records, row = score_completions(bench, problems, completions, args, params,
                                             workers=args.workers)
            row["seconds"] = round(time.time() - t0, 1)
            rows.append(row)
            rec_out = args.records_out or (
                str(Path(args.out).with_suffix(".records.jsonl")) if args.out else "")
            if rec_out:
                write_records(rec_out, records)
                print(f"wrote {rec_out}", file=sys.stderr)
            print(
                f"{row['benchmark']:<16} acc={row['accuracy']:.4f} "
                f"(all={row['accuracy_all']:.4f}, official={row['accuracy_official']:.4f}) "
                f"wilson=[{row['wilson_lo']:.3f},{row['wilson_hi']:.3f}] "
                f"n={row['n_graded']}/{row['n_problems']} pass={row['n_pass']} "
                f"wrong={row['n_wrong_answer']} err={row['n_exec_error']} "
                f"to={row['n_timeout']} nosql={row['n_no_sql']} "
                f"goldbroken={row['n_gold_broken']} rowlimit={row['n_row_limit']} "
                f"harness={row['n_harness_error']} genfail={row['n_failed']} "
                f"({row['seconds']}s)", flush=True)
    finally:
        if gen_fh:
            gen_fh.close()

    if args.out:
        # allow_nan=False is kept deliberately: it turns a NaN the scrub missed into a
        # loud crash rather than a `NaN` token that is not valid JSON.
        Path(args.out).write_text(
            json.dumps(json_safe(rows), indent=2, allow_nan=False))
        print(f"wrote {args.out}")

    for r in rows:
        if r["n_graded"] == 0:
            print(f"WARNING {r['benchmark']}: graded nothing; accuracy is meaningless",
                  file=sys.stderr)
            exit_code = 1
        elif r["n_graded"] < r["n_problems"]:
            print(f"WARNING {r['benchmark']}: only {r['n_graded']}/{r['n_problems']} graded "
                  f"({r['n_failed']} generation failures); `accuracy` is over survivors and "
                  f"is biased upward -- accuracy_all={r['accuracy_all']:.4f} counts them "
                  f"wrong", file=sys.stderr)
        if r["n_gold_broken"]:
            # The DATASET's query failed on the DATASET's database. Those rows are
            # ungradeable and no verdict about the model can be read from them.
            print(f"GOLD-BROKEN {r['benchmark']}: {r['n_gold_broken']} question(s) whose "
                  f"own gold SQL failed to execute. Those rows are ungradeable; read the "
                  f"records file before using this score.", file=sys.stderr)
            exit_code = 1
        if r["n_row_limit"]:
            print(f"ROW-LIMIT {r['benchmark']}: {r['n_row_limit']} gold result(s) exceeded "
                  f"max_rows={r['params']['max_rows']}, so those comparisons are not "
                  f"decidable and are scored as fails. Raise --max-rows.", file=sys.stderr)
            exit_code = 1
        if r["n_perm_exhausted"]:
            print(f"PERM-BUDGET {r['benchmark']}: {r['n_perm_exhausted']} comparison(s) hit "
                  f"--max-col-perms={r['params']['max_col_perms']} and were not shown equal. "
                  f"Those are scored wrong; raise the budget to decide them.",
                  file=sys.stderr)
        if r["n_harness_error"]:
            # The grader itself broke. Scored as a fail so it cannot inflate anything, and
            # loud so it cannot be mistaken for the model writing bad SQL.
            print(f"HARNESS-ERROR {r['benchmark']}: {r['n_harness_error']} execution(s) "
                  f"failed inside the GRADER, not in the query. This is a grader fault; "
                  f"read the records file before using this score.", file=sys.stderr)
            exit_code = 1
        if r["n_failed"] / max(1, r["n_problems"]) > FAILED_RATE_ABORT:
            raise SystemExit(
                f"ABORT {r['benchmark']}: {r['n_failed']}/{r['n_problems']} generation "
                f"requests FAILED. This is not a score. Usual cause is max_tokens="
                f"{r['params']['max_tokens']} exceeding the model's context, which makes the "
                f"server reject every request; also check the endpoint and model name."
            )
        if r["cap_limited"]:
            print(f"CAP-LIMITED {r['benchmark']}: {r['n_truncated']}/{r['n_problems']} "
                  f"({r['truncation_rate']:.1%}) hit max_tokens="
                  f"{r['params']['max_tokens']}. Truncated queries are graded as fails, so "
                  f"this score is partly a property of the token budget; do NOT compare it "
                  f"against a run at a different cap.", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
