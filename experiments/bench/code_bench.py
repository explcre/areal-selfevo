#!/usr/bin/env python3
"""Score a served model on LiveCodeBench by EXECUTING the code it writes.

Companion to :mod:`math_bench`, and deliberately the same shape: a ``SUITE`` of frozen
benchmarks, generation through the sglang OpenAI-compatible endpoint, per-sample records
written to disk, Wilson intervals, and every excluded or failed sample counted in the
output rather than dropped. What differs is the grader. A math answer is a string compared
to a string; a code answer is a program that has to be RUN, and running a program an
unaligned model wrote is a security problem before it is a measurement problem. That half
lives in :mod:`code_sandbox`, whose module docstring states exactly what it does and does
not contain -- read it before trusting a number from here.

DATASET
-------
``livecodebench/code_generation_lite``, release ``v6`` (``test6.jsonl``), already in the HF
cache on this box: 175 problems, contest dates 2025-01-04 to 2025-04-06, difficulty
hard 80 / medium 52 / easy 43, platform atcoder 112 / leetcode 63. All 175 carry private
tests; 7000 test cases in total (463 public, 6537 private). The dataset's own loading
script is incompatible with ``datasets 5.0.1`` ("Dataset scripts are no longer supported"),
so the raw JSONL is read directly and its md5 is recorded in the results row.

The release ships NO reference solutions. That matters for how this harness is verified and
is discussed under GOLD below.

TWO PROBLEM FAMILIES
--------------------
``stdin`` (atcoder, 112 problems)  -- a whole program reading stdin, writing stdout.
``functional`` (leetcode, 63)      -- a ``Solution`` class with a named method; the test
                                      input is one JSON literal per line, one per argument,
                                      and the expected output is a JSON literal.

Both are graded by execution. A submission passes a problem only if it passes EVERY test
of that problem, public and private, which is LiveCodeBench's own criterion.

FAILURE ACCOUNTING -- the thing this project keeps getting bitten by
--------------------------------------------------------------------
Every problem lands in exactly one bucket and every bucket is printed:

``pass`` ``wrong_answer`` ``runtime_error`` ``timeout`` ``output_limit`` ``no_code``
``harness_error`` ``gen_failed``

A crashed, timed-out, or unparseable submission is a FAIL and stays in the denominator --
it is never dropped. Only ``gen_failed`` (the endpoint returned nothing after retries) is
excluded from ``accuracy``, exactly as :mod:`math_bench` does, and because excluding it
biases the score UPWARD both numbers are reported: ``accuracy`` over graded problems and
``accuracy_all`` over all problems with generation failures counted wrong. ``harness_error``
means the SANDBOX broke, not the submission; it is scored as a fail so it cannot inflate
anything, and it also makes the process exit non-zero, because a grader that is quietly
malfunctioning is worse than one that is loudly down.

GOLD: what verifies this harness, and what it does not
------------------------------------------------------
``code_generation_lite`` has no reference solutions, so the OlympiadBench move -- run the
dataset's own gold through the grader and require 100% -- is not directly available. Two
substitutes are used instead, and they establish different things:

1. **Replay oracle** (:func:`replay_oracle`, driven by ``lcb_selfcheck.py``). For each
   problem a submission is synthesised that returns the dataset's own expected output,
   looked up by the SHA-256 of the exact bytes the harness delivered. It is not a solution
   and is never called one: it computes nothing. What it does prove, over all 175 problems
   and all 7000 tests, is that the harness delivers stdin byte-exactly, splats functional
   arguments correctly, dispatches the right method name, serialises the return value, and
   compares outputs without mangling them. A mismatch is a harness bug by construction,
   because the "solution" is the answer key.

   MEASURED 2026-09-02, bwrap tier: **175/175**. The first run of this check scored
   171/175, and the four misses were one bug: the sandbox read back at most 1 MiB of a
   program's output, while 14 test cases across 4 problems (abc388_d, abc396_f, arc190_c,
   3759) expect up to 3.28 MiB. A truncated answer compares unequal, so a
   reference-correct submission was being scored a wrong answer. Two things changed. The
   cap is now 16 MiB, set from the measured maximum with headroom rather than from a round
   number; and truncation is no longer silent -- if the answer was cut off, the comparison
   is not decidable and the problem is reported as a harness fault, which is loud and exits
   non-zero, instead of quietly subtracting from the score.

2. **Hand-written reference solutions** (``lcb_reference_solutions.json``), a small set of
   genuinely computed solutions covering both families. These are what show the grader
   accepts code that solves the problem rather than code that replays the key. They cover a
   handful of problems, not 175, and that limit is stated rather than glossed.

MEASURED 2026-09-02: **10/10** reference solutions pass.

Both are run by ``lcb_selfcheck.py``, together with a battery of known-WRONG submissions
(empty output, infinite loop, crash, syntax error, prose with no code block, fork attempt,
memory hog, network attempt, a write outside the working directory, a wrong method name,
and two oracles deliberately missing an answer) each of which must FAIL. A grader that
passes everything is worse than no grader. MEASURED 2026-09-02: **31/31** behave as
required, across one stdin and one functional problem.

Two findings from that battery are worth carrying here, because both look like grader bugs
and neither is:

* **The dataset repeats test inputs.** 70 of the 175 problems contain at least one
  duplicated input, 372 of the 7000 cases; abc387_b has 43 tests but only 26 distinct
  inputs. So an oracle deprived of "the last test" can still answer it out of an earlier
  identical one. The check now withholds a test whose input appears nowhere else.
* **Stray stdout is fatal for a stdin problem and irrelevant for a functional one**, because
  a functional verdict is the RETURNED value, read from a file, not what the program
  printed. That asymmetry is asserted in both directions rather than assumed.

OUTPUT COMPARISON
-----------------
``stdin`` output is compared after stripping trailing whitespace per line and trailing
blank lines, which is what every judge in this space does. Beyond that, a token differing
only as a FLOAT is accepted within a relative tolerance -- but only when the expected token
is not an integer literal, so ``2025`` never matches ``2024`` while ``1.41421356`` matches
``1.414213562``. The tolerance is a knob (``--float-rel-tol``, default 1e-6) and it is
recorded in the results row, because a grading tolerance that is not reported is a silent
lever on the score.

WHAT IS AND IS NOT VALIDATED HERE
--------------------------------
GRADING is verified: gold 175/175, known-wrong 31/31, and every mutant killed. Grading,
loading, prompting and the artifact writer are exercised end to end from recorded
completions (``--from-generations``), with no GPU and no endpoint.

GENERATION was first exercised against a real served 32B on 35 problems in September 2026,
and that run found two silent scoring faults, both fixed here and both pinned by tests:

* a valid submission followed by an EMPTY trailing ```python``` fence was discarded and
  scored ``no_code`` (see :func:`extract_code`), understating the score by 1/20 on that
  slice;
* the request payload names a MODEL ID and nothing else -- an adapter is reached only
  because sglang registers ``--lora-paths NAME=path`` as a model id -- and an unregistered
  id is answered HTTP 200 by the BASE model. The harness recorded no model id at all, so a
  run left on a default would have scored the base weights with nothing in results.json to
  say so. Every run now verifies the id against ``<base-url>/models`` first and records the
  id, the URL and the served list (see :func:`math_bench.verify_model`).

Still unexercised against real output: the ``[PYTHON]``-tag and unfenced submission forms,
which no observed completion has used and which this harness deliberately refuses.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import pickle
import re
import statistics
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from code_sandbox import (  # noqa: E402
    SandboxLimits,
    describe_tier,
    detect_tier,
    run_python,
)
# Reused rather than reimplemented: the endpoint client and the interval are benchmark
# agnostic and are already tested against a misbehaving local server in test_math_bench.py.
from math_bench import chat_url, generate, verify_model, wilson  # noqa: E402

# One entry per frozen code benchmark, mirroring math_bench.SUITE. The value is the file
# inside the HF snapshot, so adding LiveCodeBench v5 later is one line and no new code.
RELEASES = {
    "livecodebench_v6": "test6.jsonl",
    "livecodebench_v5": "test5.jsonl",
    "livecodebench_v4": "test4.jsonl",
}
SUITE = ["livecodebench_v6"]

# Where the snapshot lives. An environment variable rather than a constant for the same
# reason math_bench uses MATH_EVAL_DATA: this path differs on every box.
DATA_ENV = "LCB_DATA"
# Where the HF hub cache keeps this dataset, relative to the cache root, and the root
# used when HF_HOME says nothing.
_HF_SUBPATH = "hub/datasets--livecodebench--code_generation_lite/snapshots"
_HF_DEFAULT_HOME = "~/.cache/huggingface"

# Generation parameters that differ from the CLI defaults, per benchmark. Code answers are
# long -- a full program plus reasoning -- so the cap is higher than the math default.
BENCH_OVERRIDES = {
    "livecodebench_v6": {"max_tokens": 16384},
}

# Above this share of endpoint failures the run is not a measurement. Same rule and same
# rationale as math_bench.FAILED_RATE_ABORT.
FAILED_RATE_ABORT = 0.10
# Above this share of truncated generations the score measures the token budget.
CAP_LIMITED_RATE = 0.10
PROGRESS_EVERY_S = float(os.environ.get("PROGRESS_EVERY_S", "30"))

# Buckets. Exactly one per problem, all of them printed, and every one except PASS and
# GEN_FAILED is a wrong answer that stays in the denominator.
ST_PASS = "pass"
ST_WRONG = "wrong_answer"
ST_RUNTIME = "runtime_error"
ST_TIMEOUT = "timeout"
ST_OUTPUT = "output_limit"
ST_NO_CODE = "no_code"
ST_HARNESS = "harness_error"
ST_GEN_FAILED = "gen_failed"
STATUSES = (ST_PASS, ST_WRONG, ST_RUNTIME, ST_TIMEOUT, ST_OUTPUT, ST_NO_CODE,
            ST_HARNESS, ST_GEN_FAILED)


def _ids(args):
    """The ``--question-ids`` filter as a set.

    Args:
        args: Parsed CLI namespace.

    Returns:
        A set of ids, empty when the flag was not given.
    """
    raw = getattr(args, "question_ids", "") or ""
    return {x.strip() for x in raw.split(",") if x.strip()}


def hf_snapshot_roots() -> list:
    """Cache directories that may hold the release, in the order they are tried.

    ``$HF_HOME`` first, then the default cache. Hardcoding ``~/.cache/huggingface`` sent
    the eval box -- which sets ``HF_HOME=~/hf_cache`` -- looking in a directory that does
    not exist, so ``LCB_DATA`` had to be set by hand for a run that should have needed no
    arguments. The old path stays as a SECOND candidate rather than a replacement,
    because a box can hold the snapshot under the default cache while ``HF_HOME`` points
    somewhere else entirely.

    Returns:
        The candidate snapshot directories, whether or not they exist, in search order.
    """
    roots = []
    home = os.environ.get("HF_HOME")
    if home:
        roots.append(Path(os.path.expanduser(home)) / _HF_SUBPATH)
    default = Path(os.path.expanduser(_HF_DEFAULT_HOME)) / _HF_SUBPATH
    if default not in roots:
        roots.append(default)
    return roots


def snapshot_dir() -> Path:
    """Directory holding the LiveCodeBench JSONL releases.

    Returns:
        ``$LCB_DATA`` when set, otherwise the newest snapshot under the first cache root
        from :func:`hf_snapshot_roots` that holds one.

    Raises:
        FileNotFoundError: When ``$LCB_DATA`` is not a directory, or when no cache root
            holds a snapshot. A missing dataset must stop the run: left unchecked it
            yields an empty problem list, which scores zero and is indistinguishable in
            the output from a model that failed every problem.
    """
    env = os.environ.get(DATA_ENV)
    if env:
        p = Path(os.path.expanduser(env))
        if not p.is_dir():
            raise FileNotFoundError(f"{DATA_ENV}={env} is not a directory")
        return p
    roots = hf_snapshot_roots()
    for root in roots:
        if not root.is_dir():
            continue
        snaps = sorted(d for d in root.iterdir() if d.is_dir())
        if snaps:
            return snaps[-1]
    raise FileNotFoundError(
        f"no LiveCodeBench snapshot under any of {[str(r) for r in roots]} "
        f"(HF_HOME={os.environ.get('HF_HOME') or 'unset'}). Set {DATA_ENV} to a directory "
        f"holding the release JSONL files ({', '.join(sorted(RELEASES.values()))})."
    )


def require_dataset(bench: str) -> Path:
    """Path to one release's JSONL, checked to exist.

    Args:
        bench: A key of :data:`RELEASES`.

    Returns:
        The path to the release file.

    Raises:
        ValueError: For an unknown benchmark name.
        FileNotFoundError: When the file is absent, naming what IS present so the fix is
            obvious from the message alone.
    """
    if bench not in RELEASES:
        raise ValueError(f"unknown code benchmark {bench!r}; known: {sorted(RELEASES)}")
    d = snapshot_dir()
    f = d / RELEASES[bench]
    if not f.exists():
        present = sorted(x.name for x in d.iterdir() if x.suffix == ".jsonl")
        raise FileNotFoundError(
            f"{f} not found. Snapshot {d} holds {present or 'no jsonl files'}; set "
            f"{DATA_ENV} to a directory containing {RELEASES[bench]}."
        )
    return f


def decode_tests(field: str) -> list:
    """Decode one of the dataset's test-case fields.

    LiveCodeBench stores public tests as plain JSON and private tests as
    base64(zlib(pickle(json-string))) to keep them out of a plain-text search. Both forms
    appear, so both are handled.

    Args:
        field: The raw field value.

    Returns:
        A list of ``{"input", "output", "testtype"}`` dicts.

    Raises:
        ValueError: If neither encoding yields a list. A benchmark that silently loads zero
            test cases would pass every submission, which is the single worst failure this
            grader could have, so it is fatal rather than empty.
    """
    try:
        out = json.loads(field)
    except Exception:
        # The compressed form nests a pickle. It is the dataset's own encoding and the file
        # is a pinned local snapshot whose md5 is recorded in every results row; that is the
        # only reason unpickling it is acceptable. Never point LCB_DATA at a file you did
        # not put there yourself.
        out = json.loads(pickle.loads(zlib.decompress(base64.b64decode(field.encode()))))
    if not isinstance(out, list):
        raise ValueError(f"test-case field decoded to {type(out).__name__}, expected list")
    return out


def load(bench: str, difficulty: str = "", limit: int = 0, question_ids=()) -> list:
    """Load a release's problems, normalised.

    Args:
        bench: A key of :data:`RELEASES`.
        difficulty: Keep only this difficulty (``easy``/``medium``/``hard``); empty keeps
            all. Recorded in the results row, since scoring a subset and reporting it as
            the benchmark is how a number stops meaning what it says.
        question_ids: Keep only these ``question_id`` values; empty keeps all. Also
            recorded, and an id that matches nothing is fatal rather than ignored -- a
            typo would otherwise silently shrink the benchmark.
        limit: Keep only the first N after filtering; 0 keeps all.

    Returns:
        A list of problem dicts with keys ``idx``, ``question_id``, ``platform``,
        ``difficulty``, ``contest_date``, ``title``, ``question``, ``starter_code``,
        ``func_name`` and ``tests``. ``tests`` is public tests followed by private tests,
        each carrying ``visibility``.

    Raises:
        ValueError: If a row lacks the expected schema, if a functional problem has no
            ``func_name`` to call, if a problem carries no test cases at all, or if the
            filters select nothing.
    """
    f = require_dataset(bench)
    raw = f.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    out = []
    for i, r in enumerate(rows):
        missing = [k for k in ("question_content", "question_id", "platform", "difficulty",
                               "public_test_cases", "private_test_cases", "metadata")
                   if k not in r]
        if missing:
            raise ValueError(f"{bench} row {i}: missing {missing}")
        tests = []
        for vis, key in (("public", "public_test_cases"), ("private", "private_test_cases")):
            for t in decode_tests(r[key]):
                t = dict(t)
                t["visibility"] = vis
                tests.append(t)
        if not tests:
            raise ValueError(f"{bench} row {i} ({r['question_id']}): no test cases")
        meta = json.loads(r["metadata"]) if r["metadata"] else {}
        func_name = meta.get("func_name", "")
        if any(t["testtype"] == "functional" for t in tests) and not func_name:
            raise ValueError(
                f"{bench} row {i} ({r['question_id']}): functional tests but metadata has "
                f"no func_name, so there is nothing to call"
            )
        p = {
            "idx": i,
            "question_id": r["question_id"],
            "platform": r["platform"],
            "difficulty": r["difficulty"],
            "contest_date": r.get("contest_date", ""),
            "title": r.get("question_title", ""),
            "question": r["question_content"],
            "starter_code": r.get("starter_code", "") or "",
            "func_name": func_name,
            "tests": tests,
        }
        if difficulty and p["difficulty"] != difficulty:
            continue
        if question_ids and p["question_id"] not in question_ids:
            continue
        out.append(p)
    if question_ids:
        missing = sorted(set(question_ids) - {p["question_id"] for p in out})
        if missing:
            raise ValueError(f"{bench}: no such question_id: {missing}")
    if limit:
        out = out[:limit]
    if not out:
        raise ValueError(
            f"{bench}: no problems selected (difficulty={difficulty!r} "
            f"question_ids={len(question_ids)} limit={limit})"
        )
    return out


def dataset_md5(bench: str) -> str:
    """md5 of the release file, recorded so two scores can be shown to be of the same data."""
    return hashlib.md5(require_dataset(bench).read_bytes()).hexdigest()


# LiveCodeBench's own prompt, reproduced rather than invented: a different prompt is a
# different benchmark, and a score that is not comparable to the published numbers is much
# less useful than one that is.
SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question (problem "
    "specification) and will generate a correct Python program that matches the "
    "specification and passes all tests."
)
FORMAT_WITH_STARTER = (
    "You will use the following starter code to write the solution to the problem and "
    "enclose your code within delimiters."
)
FORMAT_STDIN = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not "
    "directly test on the sample inputs). Enclose your code within delimiters as follows."
)


def build_prompt(problem: dict) -> str:
    """The user message for one problem.

    Args:
        problem: A dict from :func:`load`.

    Returns:
        The prompt string, in LiveCodeBench's format, with the starter code inlined for
        functional problems and an empty ``# YOUR CODE HERE`` stub otherwise.
    """
    if problem["starter_code"].strip():
        fmt, body = FORMAT_WITH_STARTER, problem["starter_code"]
    else:
        fmt, body = FORMAT_STDIN, "# YOUR CODE HERE"
    return (
        f"{SYSTEM_MESSAGE}\n\n"
        f"### Question:\n{problem['question']}\n\n"
        f"### Format: {fmt}\n"
        f"```python\n{body}\n```\n\n"
        f"### Answer: (use the provided format with backticks)\n"
    )


_FENCE = re.compile(r"```(?P<lang>[A-Za-z0-9_+-]*)[ \t]*\r?\n(?P<body>.*?)```", re.S)
# Fence tags that mean "this block is the program".
_PY_TAGS = ("python", "py", "python3")


def extract_code(text: str):
    """Return the submitted program, or ``None`` when the completion contains none.

    PRECEDENCE, highest level first. Within a level the LAST candidate wins; an EMPTY
    candidate is never returned, and never ends the search either:

    1. python-tagged fenced blocks (``python``, ``py``, ``python3``): last non-empty one.
    2. fenced blocks of any other tag, or of none: last non-empty one.
    3. nothing else -- ``None``, graded ``no_code``.

    "Last wins" matches :func:`math_bench.extract_boxed` taking the last box: a model that
    reasons in code before committing must be graded on what it committed to. Python tags
    outrank untagged blocks so that a trailing block of example output or shell commands
    does not displace the answer.

    "NON-EMPTY" is the half a live run found missing. One 32B completion ended with an
    empty ```python``` fence after a 1102-character program that compiled; taking the last
    python block, finding it empty and returning None discarded a valid submission and
    scored it ``no_code`` -- 1 of 20 problems on that slice, understating the score. The
    empty-block guard itself is right and is kept: an empty block is never executed, and a
    completion whose ONLY block is empty is still ``no_code``. What changed is that an
    empty choice falls back to the previous non-empty candidate instead of ending the
    search. The fallback also crosses levels: if every python-tagged block is empty, the
    untagged blocks are considered before giving up.

    LEVEL 3 IS A REFUSAL, NOT AN OMISSION, and it is UNEXERCISED against real output. The
    two other forms this space uses -- a ``[PYTHON]`` ... ``[/PYTHON]`` tagged submission,
    and an unfenced program -- are deliberately NOT accepted, and no completion has ever
    needed them: 0 of 35 completions in the live 32B run used ``[PYTHON]`` tags and 0 were
    unfenced. Accepting them cannot be validated on evidence that does not exist, and it
    can only make MORE text executable: returning the whole completion as "probably code"
    would let a prose answer run, and a prose answer that happens to be a valid Python
    expression would then be graded on whatever it evaluated to. ``None`` is a real
    outcome and is graded ``no_code`` -- a FAIL that stays in the denominator.

    Args:
        text: The raw completion.

    Returns:
        The code, or ``None``.
    """
    if not text:
        return None
    blocks = [(m.group("lang").lower(), m.group("body")) for m in _FENCE.finditer(text)]
    if not blocks:
        return None
    py = [b for lang, b in blocks if lang in _PY_TAGS]
    for level in (py, [b for _, b in blocks]):
        for body in reversed(level):
            if body.strip():
                return body
    return None


# Names leetcode-style submissions assume are already imported. Only explicit names and
# module imports: a star-import of `math` would shadow builtins such as `pow` and `gcd` and
# change the meaning of otherwise-correct submissions.
FUNCTIONAL_PREAMBLE = """import sys, math, json, re, string, random, itertools, functools
import collections, heapq, bisect, operator, array, copy, decimal, fractions
from typing import *
from collections import Counter, defaultdict, deque, OrderedDict
from functools import lru_cache, cache, reduce, cmp_to_key
from itertools import accumulate, combinations, permutations, product, groupby
from heapq import heappush, heappop, heapify, nlargest, nsmallest
from bisect import bisect_left, bisect_right, insort
from math import gcd, lcm, inf, isqrt, comb, perm
"""

# Appended after a functional submission. Deliberately obscure names: a submission is free
# to define anything at module level, and a collision would make the harness call the wrong
# object and report a wrong answer that the model did not give.
_CALL_TEMPLATE = """

# ---- grading harness appended by experiments/bench/code_bench.py ----
import json as _lcb_json
with open("_args.json") as _lcb_f:
    _lcb_args = _lcb_json.load(_lcb_f)
_lcb_out = Solution().{func}(*_lcb_args)
with open("_result.json", "w") as _lcb_f:
    _lcb_json.dump(_lcb_out, _lcb_f)
"""

RESULT_FILE = "_result.json"
ARGS_FILE = "_args.json"


def build_program(problem: dict, code: str, test: dict):
    """Compose the exact program run for one test, plus the files it needs.

    Args:
        problem: A dict from :func:`load`.
        code: The submitted code.
        test: One entry of ``problem["tests"]``.

    Returns:
        ``(source, stdin_bytes, extra_files, read_back)``.

    Raises:
        ValueError: For an unknown ``testtype``, or when a functional test's input does not
            parse as one JSON literal per line. Guessing at either would grade a submission
            against arguments the problem never had.
    """
    if test["testtype"] == "stdin":
        return code, test["input"].encode(), {}, ()
    if test["testtype"] != "functional":
        raise ValueError(f"unknown testtype {test['testtype']!r}")
    args = []
    for lineno, part in enumerate(test["input"].split("\n")):
        if not part.strip():
            continue
        try:
            args.append(json.loads(part))
        except Exception as exc:
            raise ValueError(
                f"{problem['question_id']}: functional argument {lineno} is not JSON "
                f"({part[:80]!r}): {exc}"
            ) from exc
    source = (FUNCTIONAL_PREAMBLE + "\n" + code
              + _CALL_TEMPLATE.format(func=problem["func_name"]))
    return source, b"", {ARGS_FILE: json.dumps(args)}, (RESULT_FILE,)


_INT_RE = re.compile(r"^[+-]?\d+$")


def _tokens_match(got: str, want: str, rel_tol: float) -> bool:
    """Whether two whitespace-separated tokens agree.

    Exact string equality first. A float tolerance is applied ONLY when the expected token
    is not an integer literal, so ``2025`` can never match ``2024`` while ``1.41421356``
    matches ``1.414213562``. Judges in this space accept a relative error on real-valued
    answers, and refusing to would mark correct submissions wrong; accepting it on integers
    would mark wrong submissions correct. The rule is therefore asymmetric on purpose.

    Args:
        got: Token produced.
        want: Token expected.
        rel_tol: Relative tolerance for the float case; 0 disables it entirely.

    Returns:
        True when they agree.
    """
    if got == want:
        return True
    if rel_tol <= 0 or _INT_RE.match(want):
        return False
    try:
        a, b = float(got), float(want)
    except ValueError:
        return False
    if math.isnan(a) or math.isnan(b):
        return False
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=rel_tol)


def compare_stdout(got: str, want: str, rel_tol: float = 1e-6) -> bool:
    """Whether a program's stdout matches the expected output.

    Trailing whitespace on each line and trailing blank lines are ignored, which is what
    every judge for these platforms does and what the dataset's own outputs assume (many
    expected outputs carry a trailing newline that no submission is required to emit).
    Interior blank lines and line ORDER are significant.

    Args:
        got: Captured stdout.
        want: Expected output.
        rel_tol: Passed to :func:`_tokens_match`.

    Returns:
        True when they match.
    """
    def norm(s):
        return [ln.rstrip() for ln in s.replace("\r\n", "\n").rstrip().split("\n")]

    g, w = norm(got), norm(want)
    if g == w:
        return True
    if len(g) != len(w):
        return False
    for gl, wl in zip(g, w):
        gt, wt = gl.split(), wl.split()
        if len(gt) != len(wt) or not all(_tokens_match(a, b, rel_tol)
                                         for a, b in zip(gt, wt)):
            return False
    return True


def compare_value(got, want, rel_tol: float = 1e-6) -> bool:
    """Whether a functional return value matches the expected JSON value.

    Lists and tuples are compared elementwise (a submission returning a tuple where the
    key holds a list is accepted, since JSON has only one sequence type). Floats use the
    tolerance; ``bool`` is compared strictly against ``bool`` so that ``True`` never
    satisfies an expected ``1``, which is a real distinction in these problems.

    Args:
        got: Parsed return value.
        want: Parsed expected value.
        rel_tol: Relative tolerance for floats.

    Returns:
        True when they match.
    """
    if isinstance(want, bool) or isinstance(got, bool):
        return isinstance(got, bool) and isinstance(want, bool) and got == want
    if isinstance(want, (list, tuple)):
        if not isinstance(got, (list, tuple)) or len(got) != len(want):
            return False
        return all(compare_value(a, b, rel_tol) for a, b in zip(got, want))
    if isinstance(want, float) or isinstance(got, float):
        try:
            return math.isclose(float(got), float(want), rel_tol=rel_tol, abs_tol=rel_tol)
        except (TypeError, ValueError):
            return False
    return got == want


def _clip(s, n: int = 400) -> str:
    """Shorten a string for the artifact file, marking that it was shortened."""
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n] + f"... [+{len(s) - n} chars]"


def grade_submission(problem: dict, code, limits=None, tier=None, rel_tol: float = 1e-6,
                     max_tests: int = 0, stop_on_first_failure: bool = True) -> dict:
    """Run one submission against one problem's tests and return the verdict.

    A submission passes only if EVERY selected test passes, which is LiveCodeBench's own
    criterion. Testing stops at the first failure by default, because a submission that
    fails test 3 of 40 cannot pass however the other 37 go, and running them costs 37
    sandboxed processes for no information.

    Args:
        problem: A dict from :func:`load`.
        code: The submitted program, or ``None`` when the completion held no code block.
        limits: :class:`code_sandbox.SandboxLimits`; defaults apply when omitted.
        tier: Isolation tier; defaults to the strongest this machine supports.
        rel_tol: Float tolerance for output comparison.
        max_tests: Test cases to run per problem, 0 for all. A cap makes the score cheaper
            and WEAKER (fewer chances to catch a wrong submission), so it is recorded in
            the results row rather than being an invisible knob.
        stop_on_first_failure: Whether to stop at the first failing test.

    Returns:
        A record with ``status`` (one of :data:`STATUSES`), ``passed``, ``n_tests``,
        ``n_tests_run``, ``n_tests_passed``, ``first_fail`` and ``detail``. Never raises for
        anything the submission does: a crash, a hang and a missing result file are
        outcomes to be scored.
    """
    limits = limits or SandboxLimits()
    tier = tier or detect_tier()
    tests = problem["tests"][:max_tests] if max_tests else problem["tests"]
    base = {
        "question_id": problem["question_id"],
        "idx": problem["idx"],
        "platform": problem["platform"],
        "difficulty": problem["difficulty"],
        "n_tests": len(tests),
        "n_tests_run": 0,
        "n_tests_passed": 0,
        "first_fail": None,
        "sandbox_tier": tier,
        "elapsed_s": 0.0,
        "detail": "",
    }
    if not code or not code.strip():
        # A completion with no code block is a FAIL, not an exclusion. Dropping it would
        # shrink the denominator and inflate the score, which is precisely the silent-zero
        # path this project has been bitten by repeatedly.
        base["status"] = ST_NO_CODE
        base["passed"] = False
        base["detail"] = "completion contained no fenced code block"
        return base

    t0 = time.time()
    for i, test in enumerate(tests):
        try:
            source, stdin_data, extra, read_back = build_program(problem, code, test)
        except ValueError as exc:
            base.update(status=ST_HARNESS, passed=False, first_fail=i,
                        detail=f"could not build the program for test {i}: {exc}",
                        elapsed_s=round(time.time() - t0, 3))
            return base
        r = run_python(source, stdin_data=stdin_data, limits=limits, extra_files=extra,
                       read_back=read_back, tier=tier)
        base["n_tests_run"] = i + 1
        if r.status == "harness_error":
            base.update(status=ST_HARNESS, passed=False, first_fail=i,
                        detail=f"sandbox failure on test {i}: {r.detail} {_clip(r.stderr)}",
                        elapsed_s=round(time.time() - t0, 3))
            return base
        if r.status == "timeout":
            base.update(status=ST_TIMEOUT, passed=False, first_fail=i,
                        detail=f"test {i} ({test['visibility']}) exceeded "
                               f"{limits.wall_seconds}s",
                        elapsed_s=round(time.time() - t0, 3))
            return base
        if r.status == "output_limit":
            base.update(status=ST_OUTPUT, passed=False, first_fail=i,
                        detail=f"test {i} exceeded the {limits.file_bytes}-byte write limit",
                        elapsed_s=round(time.time() - t0, 3))
            return base
        if r.status == "error":
            base.update(status=ST_RUNTIME, passed=False, first_fail=i,
                        detail=f"test {i} exited {r.returncode}: {_clip(r.stderr)}",
                        elapsed_s=round(time.time() - t0, 3))
            return base
        if r.stdout_truncated or any(r.files_truncated.values()):
            # The answer was cut off by the READ-BACK cap, so we do not know whether it
            # matched. Calling that a wrong answer is exactly the silent deflation this
            # harness exists to avoid: it was measured costing 4 of 175 problems at a 1 MiB
            # cap, on submissions that were reference-correct. Reported as a grader fault,
            # which also makes the process exit non-zero.
            base.update(status=ST_HARNESS, passed=False, first_fail=i,
                        detail=f"test {i}: output exceeded the {limits.output_bytes}-byte "
                               f"read-back cap, so the comparison is not decidable; raise "
                               f"--output-mb",
                        elapsed_s=round(time.time() - t0, 3))
            return base

        if test["testtype"] == "stdin":
            ok = compare_stdout(r.stdout, test["output"], rel_tol)
            why = f"expected {_clip(test['output'], 200)!r} got {_clip(r.stdout, 200)!r}"
        else:
            raw = r.files.get(RESULT_FILE)
            if raw is None:
                base.update(status=ST_RUNTIME, passed=False, first_fail=i,
                            detail=f"test {i}: the solution never returned a value "
                                   f"({_clip(r.stderr)})",
                            elapsed_s=round(time.time() - t0, 3))
                return base
            try:
                got = json.loads(raw)
                want = json.loads(test["output"])
            except Exception as exc:
                base.update(status=ST_HARNESS, passed=False, first_fail=i,
                            detail=f"test {i}: could not parse the values ({exc})",
                            elapsed_s=round(time.time() - t0, 3))
                return base
            ok = compare_value(got, want, rel_tol)
            why = f"expected {_clip(test['output'], 200)!r} got {_clip(raw, 200)!r}"

        if ok:
            base["n_tests_passed"] += 1
        else:
            base.update(status=ST_WRONG, passed=False, first_fail=i,
                        detail=f"test {i} ({test['visibility']}) {why}",
                        elapsed_s=round(time.time() - t0, 3))
            if stop_on_first_failure:
                return base
    base["elapsed_s"] = round(time.time() - t0, 3)
    if base["n_tests_passed"] == base["n_tests"]:
        base.update(status=ST_PASS, passed=True)
    else:
        base.setdefault("status", ST_WRONG)
        base["passed"] = False
    return base


def replay_oracle(problem: dict) -> str:
    """A submission that returns the dataset's own expected output for every test.

    NOT a solution, and never reported as one: it computes nothing. It exists to separate
    harness bugs from solution bugs, the way OlympiadBench's gold answers do on the math
    side. The lookup key is the SHA-256 of the EXACT bytes the harness delivers -- stdin for
    a stdin problem, the JSON of the splatted argument list for a functional one -- so an
    input the harness mangles by even one byte produces an unknown key and a loud
    ``KeyError`` rather than a quiet mismatch.

    Args:
        problem: A dict from :func:`load`.

    Returns:
        Python source for the oracle submission.

    Raises:
        ValueError: For a problem whose tests are of an unknown type.
    """
    kinds = {t["testtype"] for t in problem["tests"]}
    if kinds == {"stdin"}:
        table = {hashlib.sha256(t["input"].encode()).hexdigest(): t["output"]
                 for t in problem["tests"]}
        return (
            "import sys, json, hashlib\n"
            f"_T = json.loads({json.dumps(json.dumps(table))})\n"
            "_data = sys.stdin.buffer.read()\n"
            "_k = hashlib.sha256(_data).hexdigest()\n"
            "if _k not in _T:\n"
            "    raise KeyError('replay oracle: stdin not in the answer key '\n"
            "                   '(%d bytes delivered)' % len(_data))\n"
            "sys.stdout.write(_T[_k])\n"
        )
    if kinds == {"functional"}:
        table = {}
        for t in problem["tests"]:
            args = [json.loads(p) for p in t["input"].split("\n") if p.strip()]
            table[hashlib.sha256(json.dumps(args).encode()).hexdigest()] = t["output"]
        return (
            "import json, hashlib\n"
            f"_T = json.loads({json.dumps(json.dumps(table))})\n"
            "class Solution:\n"
            f"    def {problem['func_name']}(self, *a):\n"
            "        _k = hashlib.sha256(json.dumps(list(a)).encode()).hexdigest()\n"
            "        if _k not in _T:\n"
            "            raise KeyError('replay oracle: arguments not in the answer key')\n"
            "        return json.loads(_T[_k])\n"
        )
    raise ValueError(f"{problem['question_id']}: mixed or unknown test types {kinds}")


def grade_many(problems: list, codes: list, workers: int = 16, **kw) -> list:
    """Grade a list of submissions in parallel, preserving order.

    Threads are safe here because every execution is a separate process and the sandbox
    uses no ``preexec_fn`` -- resource limits are applied inside the child instead, which is
    the reason that choice was made.

    Args:
        problems: Problems from :func:`load`.
        codes: Submissions aligned with ``problems``; ``None`` entries grade ``no_code``.
        workers: Concurrent sandboxes.
        **kw: Forwarded to :func:`grade_submission`.

    Returns:
        Verdict records in the order of ``problems``.

    Raises:
        ValueError: If the two lists differ in length, which would silently pair
            submissions with the wrong problems.
    """
    if len(problems) != len(codes):
        raise ValueError(f"{len(problems)} problems but {len(codes)} submissions")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(lambda pc: grade_submission(pc[0], pc[1], **kw),
                             zip(problems, codes)))


def summarise(bench: str, problems: list, records: list, params: dict) -> dict:
    """Aggregate per-problem verdicts into a results row.

    Every bucket is counted and reported. ``accuracy`` excludes problems whose GENERATION
    failed, matching math_bench, and because that exclusion biases the score upward
    ``accuracy_all`` reports the same passes over every problem.

    Args:
        bench: Benchmark name.
        problems: The problems scored.
        records: Verdicts from :func:`grade_submission`, aligned with ``problems``.
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
    acc = (n_pass / n_graded) if n_graded else float("nan")
    lo, hi = wilson(n_pass, n_graded) if n_graded else (float("nan"), float("nan"))
    n_trunc = sum(1 for r in records if r.get("finish_reason") == "length")
    return {
        "benchmark": bench,
        "n_problems": n_problems,
        "n_graded": n_graded,
        "n_pass": n_pass,
        "n_failed": counts[ST_GEN_FAILED],
        "n_truncated": n_trunc,
        "n_no_code": counts[ST_NO_CODE],
        "n_wrong_answer": counts[ST_WRONG],
        "n_runtime_error": counts[ST_RUNTIME],
        "n_timeout": counts[ST_TIMEOUT],
        "n_output_limit": counts[ST_OUTPUT],
        "n_harness_error": counts[ST_HARNESS],
        "accuracy": acc,
        "accuracy_all": (n_pass / n_problems) if n_problems else float("nan"),
        "wilson_lo": lo,
        "wilson_hi": hi,
        "truncation_rate": (n_trunc / n_problems) if n_problems else float("nan"),
        "cap_limited": bool(n_problems) and (n_trunc / n_problems) > CAP_LIMITED_RATE,
        "params": dict(params),
        "counts": counts,
    }


def resolve_params(bench: str, args, explicit=frozenset()) -> dict:
    """Generation and grading parameters this benchmark actually runs with.

    :data:`BENCH_OVERRIDES` states only what differs from the CLI defaults, and an
    explicitly named CLI flag outranks the table -- without that distinction a deliberate
    ``--max-tokens`` is discarded while the run still reports success.

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
        "wall_seconds": args.wall_seconds,
        "memory_mb": args.memory_mb,
        "output_mb": getattr(args, "output_mb", 16),
        "max_tests": args.max_tests,
        "float_rel_tol": args.float_rel_tol,
        "difficulty": args.difficulty,
        "question_ids": sorted(_ids(args)) or "all",
        "sandbox_tier": args.sandbox or detect_tier(),
        "dataset_md5": dataset_md5(bench),
    }
    for k, v in BENCH_OVERRIDES.get(bench, {}).items():
        if k not in explicit:
            out[k] = v
    out["sandbox_describes"] = describe_tier(out["sandbox_tier"])
    return out


async def _generate_all(bench, problems, args, params) -> list:
    """Fetch one completion per problem from the endpoint, in problem order.

    Args:
        bench: Benchmark name, for the progress line.
        problems: Problems from :func:`load`.
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
        # the resolved id, the URL and the served list into the parameters this row
        # reports. Without the first half the run scores whatever weights happen to be
        # loaded; without the second half nobody can tell afterwards which ones those were.
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


def score_completions(bench, problems, completions, args, params, workers=16) -> tuple:
    """Grade a set of completions and return ``(per-problem records, results row)``.

    Separated from generation on purpose: it is what lets the whole grading path be
    exercised from recorded completions with no model running, which is the only way this
    harness could be validated on a box whose GPUs must not be touched.

    Args:
        bench: Benchmark name.
        problems: Problems from :func:`load`.
        completions: ``{"text", "finish_reason", "status"}`` aligned with ``problems``.
        args: Parsed CLI namespace.
        params: Resolved parameters.
        workers: Concurrent sandboxes.

    Returns:
        ``(records, row)``.
    """
    limits = SandboxLimits(wall_seconds=args.wall_seconds,
                           memory_bytes=int(args.memory_mb) * 1024 * 1024,
                           output_bytes=int(args.output_mb) * 1024 * 1024)
    tier = params["sandbox_tier"]
    codes = [extract_code(c["text"]) for c in completions]
    records = grade_many(problems, codes, workers=workers, limits=limits, tier=tier,
                         rel_tol=args.float_rel_tol, max_tests=args.max_tests)
    for rec, comp, code in zip(records, completions, codes):
        rec["finish_reason"] = comp.get("finish_reason")
        rec["gen_status"] = comp.get("status", "ok")
        rec["benchmark"] = bench
        rec["code_sha256"] = hashlib.sha256((code or "").encode()).hexdigest()
        rec["code_chars"] = len(code or "")
        if rec["gen_status"] == "failed":
            # The endpoint never answered. That is not a statement about the model's code,
            # so it is bucketed separately -- and because excluding it biases the score
            # upward, summarise() reports accuracy_all alongside accuracy.
            rec["status"] = ST_GEN_FAILED
            rec["passed"] = False
            rec["detail"] = "endpoint returned nothing after retries"
    return records, summarise(bench, problems, records, params)


def write_records(path, records) -> None:
    """Write per-problem verdicts as JSONL.

    An aggregate with no per-problem record cannot be audited later, and every number this
    project has had to re-examine was one that had not been persisted.

    Args:
        path: Destination file.
        records: Verdicts.
    """
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def build_parser() -> argparse.ArgumentParser:
    """The command-line parser this script actually runs with."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8404/v1")
    ap.add_argument("--model", default=None,
                    help="model id to score, as listed by <base-url>/models. Required "
                         "unless --from-generations, and deliberately without a default: "
                         "an unregistered id is answered HTTP 200 by the BASE model, so a "
                         "default silently scores the wrong weights. A LoRA adapter "
                         "registered with --lora-paths NAME=path is addressed as NAME.")
    ap.add_argument("--benchmarks", default=",".join(SUITE))
    ap.add_argument("--limit", type=int, default=0, help="problems per benchmark, 0 = all")
    ap.add_argument("--difficulty", default="", choices=["", "easy", "medium", "hard"])
    ap.add_argument("--question-ids", default="",
                    help="comma-separated question_id values to score. Scoring a subset "
                         "and calling it the benchmark is how a number stops meaning what "
                         "it says, so the selection is recorded in the results row.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16,
                    help="concurrent sandboxes during grading")
    ap.add_argument("--wall-seconds", type=float, default=12.0,
                    help="hard wall-clock limit per test case")
    ap.add_argument("--memory-mb", type=int, default=2048,
                    help="address-space limit per test case")
    ap.add_argument("--output-mb", type=int, default=16,
                    help="most output read back per test case. The largest expected "
                         "output in v6 is 3.28 MiB; below it, correct answers are "
                         "truncated and become undecidable comparisons.")
    ap.add_argument("--max-tests", type=int, default=0,
                    help="test cases per problem, 0 = all. A cap makes the grade WEAKER "
                         "and is recorded in the results row.")
    ap.add_argument("--float-rel-tol", type=float, default=1e-6,
                    help="relative tolerance for non-integer expected tokens; 0 disables")
    ap.add_argument("--sandbox", default="", choices=["", "bwrap", "netns", "subprocess"],
                    help="force an isolation tier instead of using the strongest available")
    ap.add_argument("--from-generations", default="",
                    help="grade a recorded JSONL of completions instead of calling a "
                         "model; needs no GPU and no endpoint")
    ap.add_argument("--out", default="", help="results JSON")
    ap.add_argument("--gen-out", default="", help="JSONL of every completion")
    ap.add_argument("--records-out", default="",
                    help="JSONL of per-problem verdicts (defaults beside --out)")
    return ap


def load_generations(path, problems) -> list:
    """Read recorded completions and align them with ``problems``.

    Args:
        path: JSONL with one object per completion, carrying ``idx`` (the row's position in
            the SOURCE file) or ``question_id``, and ``text``.
        problems: Problems from :func:`load`.

    Returns:
        Completions aligned with ``problems``; a problem with no record is marked
        ``failed``, never skipped.

    Raises:
        ValueError: If the file yields no usable record, which would otherwise grade every
            problem as a generation failure and print a confident nan.
    """
    by_idx, by_qid = {}, {}
    n = 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        n += 1
        if "idx" in d:
            by_idx[d["idx"]] = d
        if "question_id" in d:
            by_qid[d["question_id"]] = d
    if not n:
        raise ValueError(f"{path} holds no completions")
    out = []
    for p in problems:
        d = by_idx.get(p["idx"]) or by_qid.get(p["question_id"])
        if d is None:
            out.append({"text": "", "finish_reason": None, "status": "failed"})
        else:
            out.append({"text": d.get("text", ""),
                        "finish_reason": d.get("finish_reason"),
                        "status": d.get("status", "ok")})
    return out


def main() -> int:
    """Run the requested code benchmarks and return a process exit status."""
    ap = build_parser()
    args = ap.parse_args()
    explicit = {a.lstrip("-").replace("-", "_") for a in sys.argv[1:] if a.startswith("--")}
    # Refused here, before the dataset is loaded and before the endpoint is touched at all:
    # a missing flag should be reported as a missing flag. --from-generations grades
    # recorded text and needs no model, so it is exempt.
    if not args.from_generations and not args.model:
        raise SystemExit(
            "ERROR: --model is required when generating. It names which weights answer: "
            "an id the server does not serve is answered HTTP 200 by the BASE model, so "
            "there is no safe default. Pass an id from <base-url>/models, or use "
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
            print(f"{bench}: {len(problems)} problems, sandbox="
                  f"{params['sandbox_tier']} ({params['sandbox_describes']})",
                  file=sys.stderr, flush=True)
            # Every row carries the same attribution schema, so "which model produced
            # this score" is answerable from the artifact alone -- and a regrade says so
            # rather than inheriting a model id it never called.
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
                f"{row['benchmark']:<18} acc={row['accuracy']:.4f} "
                f"(all={row['accuracy_all']:.4f}) "
                f"wilson=[{row['wilson_lo']:.3f},{row['wilson_hi']:.3f}] "
                f"n={row['n_graded']}/{row['n_problems']} pass={row['n_pass']} "
                f"wrong={row['n_wrong_answer']} rt={row['n_runtime_error']} "
                f"to={row['n_timeout']} nocode={row['n_no_code']} "
                f"harness={row['n_harness_error']} genfail={row['n_failed']} "
                f"({row['seconds']}s)", flush=True)
    finally:
        if gen_fh:
            gen_fh.close()

    if args.out:
        Path(args.out).write_text(json.dumps(
            [{k: (None if isinstance(v, float) and math.isnan(v) else v)
              for k, v in r.items()} for r in rows], indent=2, allow_nan=False))
        print(f"wrote {args.out}")

    for r in rows:
        if r["n_graded"] == 0:
            print(f"WARNING {r['benchmark']}: graded nothing; accuracy is meaningless",
                  file=sys.stderr)
            exit_code = 1
        elif r["n_graded"] < r["n_problems"]:
            print(f"WARNING {r['benchmark']}: only {r['n_graded']}/{r['n_problems']} "
                  f"graded ({r['n_failed']} generation failures); `accuracy` is over "
                  f"survivors and is biased upward -- accuracy_all="
                  f"{r['accuracy_all']:.4f} counts them wrong", file=sys.stderr)
        if r["n_harness_error"]:
            # The grader itself broke. Scored as a fail so it cannot inflate anything, and
            # loud so it cannot be mistaken for the model writing bad code.
            print(f"HARNESS-ERROR {r['benchmark']}: {r['n_harness_error']} problem(s) "
                  f"failed inside the SANDBOX, not in the submission. This is a grader "
                  f"fault; read the records file before using this score.",
                  file=sys.stderr)
            exit_code = 1
        if r["n_failed"] / max(1, r["n_problems"]) > FAILED_RATE_ABORT:
            raise SystemExit(
                f"ABORT {r['benchmark']}: {r['n_failed']}/{r['n_problems']} generation "
                f"requests FAILED. This is not a score. Usual cause is max_tokens="
                f"{r['params']['max_tokens']} exceeding the model's context, which makes "
                f"the server reject every request; also check the endpoint and model name."
            )
        if r["cap_limited"]:
            print(f"CAP-LIMITED {r['benchmark']}: {r['n_truncated']}/{r['n_problems']} "
                  f"({r['truncation_rate']:.1%}) hit max_tokens="
                  f"{r['params']['max_tokens']}. Truncated programs are graded as fails, "
                  f"so this score is partly a property of the token budget; do NOT compare "
                  f"it against a run at a different cap.", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
