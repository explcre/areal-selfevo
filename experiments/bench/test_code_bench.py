#!/usr/bin/env python3
"""Tests for the LiveCodeBench harness.

Two things are being constrained here. The first is the grading LOGIC -- extraction,
comparison, dispatch -- where the risk is a comparator that is too generous and passes
wrong answers. The second, and the reason most of these exist, is the ACCOUNTING: a
crashed, hung or unparseable submission must be a FAIL that stays in the denominator, and
the counts must add up to the number of problems no matter which way each one failed. This
project has repeatedly reported inflated scores because excluded samples quietly left the
denominator, so those invariants are asserted directly rather than inferred from a score.

The end of the file drives the real CLI over a synthetic dataset and recorded completions,
because a harness that is only ever exercised through its parts keeps working after the
loop that joins them has broken.
"""

from __future__ import annotations

import base64
import json
import pickle
import sys
import zlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import code_bench as cb  # noqa: E402
from code_sandbox import SandboxLimits  # noqa: E402
# The stub endpoint lives beside the shared plumbing it stands in for; a second copy here
# would drift from the one the math suite tests.
from test_math_bench import _StubSession  # noqa: E402

FAST = SandboxLimits(wall_seconds=8.0, memory_bytes=512 * 1024 ** 2)


def has_dataset():
    """Whether the real release is present on this machine."""
    try:
        cb.require_dataset("livecodebench_v6")
        return True
    except Exception:
        return False


needs_dataset = pytest.mark.skipif(not has_dataset(),
                                   reason="livecodebench v6 snapshot not on this box")


# ------------------------------------------------------------------- extract_code


def test_extracts_a_fenced_python_block():
    assert cb.extract_code("blah\n```python\nprint(1)\n```\n") == "print(1)\n"


def test_last_python_block_wins():
    """A model that explores in code before committing is graded on what it committed."""
    text = "```python\nwrong()\n```\nactually\n```python\nright()\n```"
    assert cb.extract_code(text) == "right()\n"


def test_untagged_block_is_accepted_when_nothing_is_tagged():
    assert cb.extract_code("```\nprint(1)\n```") == "print(1)\n"


def test_python_tag_beats_a_later_untagged_block():
    """A trailing block of sample output must not displace the answer."""
    text = "```python\nsolve()\n```\nOutput:\n```\n42\n```"
    assert cb.extract_code(text) == "solve()\n"


def test_prose_without_a_fence_is_not_code():
    """Returning the whole completion would EXECUTE prose. None is graded no_code, a fail
    that stays in the denominator."""
    assert cb.extract_code("I cannot solve this problem.") is None
    assert cb.extract_code("") is None
    assert cb.extract_code("```python\n\n```") is None


# The live 32B run, on abc390_e: a 1102-character program that compiled, followed by an
# EMPTY python fence. Taking the last python block and refusing it for being empty threw
# the submission away and scored it no_code -- 1 of 20 problems on that slice.
_TRAILING_EMPTY_FENCE = (
    "Here is my solution.\n\n"
    "```python\nimport sys\nprint(sum(map(int, sys.stdin.read().split())))\n```\n"
    "```python\n```\n"
)


def test_an_empty_trailing_fence_does_not_discard_the_submission():
    got = cb.extract_code(_TRAILING_EMPTY_FENCE)
    assert got == "import sys\nprint(sum(map(int, sys.stdin.read().split())))\n"
    compile(got, "<submission>", "exec")  # it really was a program


def test_a_completion_whose_only_block_is_empty_is_still_no_code():
    """The empty-block guard is right; only the discard-the-submission half was wrong."""
    assert cb.extract_code("```python\n```") is None
    assert cb.extract_code("```python\n   \n```") is None
    assert cb.extract_code("```\n\n```") is None
    assert cb.extract_code("```python\n\n```\nnothing here\n```python\n\n```") is None


def test_precedence_level_1_is_the_last_non_empty_python_block():
    text = ("```python\nfirst()\n```\n```python\nsecond()\n```\n"
            "```python\n\n```\n```\nuntagged()\n```")
    assert cb.extract_code(text) == "second()\n"


def test_precedence_level_2_is_any_fence_once_no_python_block_has_content():
    """Only reached when level 1 yields nothing: an all-empty python level is not an answer."""
    assert cb.extract_code("```python\n\n```\n```\nuntagged()\n```") == "untagged()\n"
    assert cb.extract_code("```text\nnotes\n```\n```\nlast()\n```") == "last()\n"


def test_precedence_level_3_the_python_tag_form_is_refused_not_extracted():
    """Documented as a refusal and UNEXERCISED: 0/35 live completions used this form."""
    assert cb.extract_code("[PYTHON]\nprint(1)\n[/PYTHON]") is None


def test_precedence_level_4_an_unfenced_program_is_refused_not_extracted():
    """Also a documented refusal: executing unfenced text would execute prose."""
    assert cb.extract_code("import sys\nprint(sum(map(int, sys.stdin.read().split())))") is None


# ------------------------------------------------------------------ compare_stdout


def test_trailing_whitespace_and_newlines_are_ignored():
    assert cb.compare_stdout("42\n", "42")
    assert cb.compare_stdout("42", "42\n")
    assert cb.compare_stdout("1  \n2\n\n", "1\n2")


def test_line_order_and_content_are_significant():
    assert not cb.compare_stdout("1\n2", "2\n1")
    assert not cb.compare_stdout("1\n2", "1\n2\n3")
    assert not cb.compare_stdout("", "42")


def test_integer_answers_are_compared_exactly():
    """The float tolerance must never reach an integer expectation."""
    assert not cb.compare_stdout("2024", "2025")
    assert not cb.compare_stdout("2025.0000001", "2025")
    assert not cb.compare_stdout("0", "1")


def test_float_answers_use_the_tolerance_but_only_within_it():
    assert cb.compare_stdout("1.414213562", "1.4142135623")
    assert not cb.compare_stdout("1.4142", "1.4142135623")
    assert not cb.compare_stdout("1.414213562", "1.4142135623", rel_tol=0)


def test_token_counts_must_match():
    assert not cb.compare_stdout("1 2", "1")
    assert cb.compare_stdout("1 2\n3", "1  2\n3")


# ------------------------------------------------------------------ compare_value


def test_values_compare_by_structure():
    assert cb.compare_value([1, 4], [1, 4])
    assert cb.compare_value((1, 4), [1, 4])
    assert not cb.compare_value([1, 4], [4, 1])
    assert not cb.compare_value([1], [1, 4])
    assert cb.compare_value([[1], [2]], [[1], [2]])


def test_bool_is_not_an_integer():
    """These problems genuinely distinguish true from 1."""
    assert not cb.compare_value(True, 1)
    assert not cb.compare_value(1, True)
    assert cb.compare_value(True, True)
    assert not cb.compare_value(True, False)


def test_float_values_use_the_tolerance():
    assert cb.compare_value(0.1 + 0.2, 0.3)
    assert not cb.compare_value(0.31, 0.3)


# ------------------------------------------------------------------ build_program


def _stdin_problem(**kw):
    p = {"idx": 0, "question_id": "q1", "platform": "atcoder", "difficulty": "easy",
         "contest_date": "", "title": "t", "question": "add them", "starter_code": "",
         "func_name": "",
         "tests": [{"input": "1 2", "output": "3", "testtype": "stdin",
                    "visibility": "public"},
                   {"input": "4 5", "output": "9", "testtype": "stdin",
                    "visibility": "private"}]}
    p.update(kw)
    return p


def _functional_problem(**kw):
    p = {"idx": 1, "question_id": "q2", "platform": "leetcode", "difficulty": "easy",
         "contest_date": "", "title": "t", "question": "add them",
         "starter_code": "class Solution:\n    def add(self, a, b):\n        ",
         "func_name": "add",
         "tests": [{"input": "1\n2", "output": "3", "testtype": "functional",
                    "visibility": "public"},
                   {"input": "10\n20", "output": "30", "testtype": "functional",
                    "visibility": "private"}]}
    p.update(kw)
    return p


ADD_STDIN = "import sys\na, b = map(int, sys.stdin.read().split())\nprint(a + b)\n"
ADD_FUNC = "class Solution:\n    def add(self, a, b):\n        return a + b\n"


def test_stdin_program_is_passed_through_verbatim():
    """Prefixing a whole-program submission would shift every line number in its
    traceback and could break a __future__ import."""
    p = _stdin_problem()
    src, stdin, extra, read_back = cb.build_program(p, ADD_STDIN, p["tests"][0])
    assert src == ADD_STDIN
    assert stdin == b"1 2"
    assert extra == {} and read_back == ()


def test_functional_arguments_are_one_json_literal_per_line():
    p = _functional_problem()
    src, stdin, extra, read_back = cb.build_program(p, ADD_FUNC, p["tests"][0])
    assert json.loads(extra[cb.ARGS_FILE]) == [1, 2]
    assert read_back == (cb.RESULT_FILE,)
    assert "Solution().add(*_lcb_args)" in src
    assert ADD_FUNC in src


def test_unparseable_functional_argument_is_fatal_not_guessed():
    p = _functional_problem()
    p["tests"][0] = dict(p["tests"][0], input="not json")
    with pytest.raises(ValueError):
        cb.build_program(p, ADD_FUNC, p["tests"][0])


def test_unknown_testtype_is_fatal():
    p = _stdin_problem()
    p["tests"][0] = dict(p["tests"][0], testtype="carrier-pigeon")
    with pytest.raises(ValueError):
        cb.build_program(p, ADD_STDIN, p["tests"][0])


# --------------------------------------------------------------- grade_submission


def _grade(problem, code):
    return cb.grade_submission(problem, code, limits=FAST)


def test_a_correct_stdin_solution_passes_every_test():
    r = _grade(_stdin_problem(), ADD_STDIN)
    assert r["status"] == cb.ST_PASS and r["passed"]
    assert r["n_tests_passed"] == r["n_tests"] == 2


def test_a_correct_functional_solution_passes():
    r = _grade(_functional_problem(), ADD_FUNC)
    assert r["status"] == cb.ST_PASS and r["passed"]


def test_a_solution_that_only_passes_the_public_test_fails():
    """The private tests must actually be executed, or an overfitted submission scores
    as correct."""
    code = "import sys\nsys.stdin.read()\nprint(3)\n"
    r = _grade(_stdin_problem(), code)
    assert r["status"] == cb.ST_WRONG and not r["passed"]
    assert r["first_fail"] == 1


def test_missing_code_is_a_fail_not_an_exclusion():
    for code in (None, "", "   "):
        r = _grade(_stdin_problem(), code)
        assert r["status"] == cb.ST_NO_CODE and r["passed"] is False


def test_a_crash_is_a_fail():
    r = _grade(_stdin_problem(), "raise RuntimeError('x')\n")
    assert r["status"] == cb.ST_RUNTIME and not r["passed"]


def test_a_hang_is_a_fail_and_does_not_hang_the_grader():
    r = cb.grade_submission(_stdin_problem(), "while True:\n    pass\n",
                            limits=SandboxLimits(wall_seconds=2.0))
    assert r["status"] == cb.ST_TIMEOUT and not r["passed"]


def test_a_functional_solution_that_never_returns_is_a_fail():
    r = _grade(_functional_problem(), "import sys\nsys.exit(0)\n")
    assert r["status"] == cb.ST_RUNTIME and not r["passed"]


def test_stopping_early_still_reports_how_far_it_got():
    r = _grade(_stdin_problem(), "import sys\nsys.stdin.read()\nprint(3)\n")
    assert r["n_tests_run"] == 2 and r["n_tests_passed"] == 1 and r["n_tests"] == 2


def test_running_every_test_still_requires_every_test_to_pass():
    """Kills the mutant that passes a submission once ANY test passes.

    With the early exit on, a failing submission returns before the final tally is ever
    reached, so the tally itself was unconstrained. Turning the early exit off is what
    exercises it.
    """
    r = cb.grade_submission(_stdin_problem(), "import sys\nsys.stdin.read()\nprint(3)\n",
                            limits=FAST, stop_on_first_failure=False)
    assert r["status"] == cb.ST_WRONG and not r["passed"]
    assert r["n_tests_run"] == 2 and r["n_tests_passed"] == 1


def test_a_sandbox_fault_is_not_blamed_on_the_submission(monkeypatch):
    """A grader that is broken must say so, not report the model as wrong.

    Forced rather than provoked: a genuine sandbox launch failure cannot be produced
    reliably on a healthy box, and leaving the branch untested let a mutant that deletes it
    survive -- the submission then silently graded as a wrong answer instead.
    """
    from code_sandbox import SandboxResult
    monkeypatch.setattr(cb, "run_python", lambda *a, **k: SandboxResult(
        status="harness_error", returncode=None, stdout="", stderr="",
        detail="sandbox could not start"))
    r = _grade(_stdin_problem(), ADD_STDIN)
    assert r["status"] == cb.ST_HARNESS and not r["passed"]
    assert "sandbox" in r["detail"]


def test_max_tests_is_recorded_because_it_weakens_the_grade():
    r = cb.grade_submission(_stdin_problem(), "import sys\nsys.stdin.read()\nprint(3)\n",
                            limits=FAST, max_tests=1)
    assert r["status"] == cb.ST_PASS and r["n_tests"] == 1


def test_output_truncation_is_a_grader_fault_not_a_wrong_answer():
    """MEASURED: at a 1 MiB read-back cap, 4 of the 175 real problems failed the replay
    oracle because their expected output is up to 3.28 MiB. A truncated output compares
    unequal, so a reference-correct answer was being scored wrong. The comparison is not
    decidable when the answer was cut off, so it is a harness fault -- which is loud and
    exits non-zero -- rather than a quiet subtraction from the score."""
    big = "x" * 5000
    problem = _stdin_problem(tests=[{"input": "1", "output": big, "testtype": "stdin",
                                     "visibility": "public"}])
    r = cb.grade_submission(problem, "print('x' * 5000)\n",
                            limits=SandboxLimits(wall_seconds=8.0, output_bytes=1000))
    assert r["status"] == cb.ST_HARNESS
    assert "read-back cap" in r["detail"]


def test_a_big_answer_is_graded_normally_at_the_default_cap():
    """The counterpart: the fix must not be a blanket refusal to grade large outputs."""
    big = "x" * 5000
    problem = _stdin_problem(tests=[{"input": "1", "output": big, "testtype": "stdin",
                                     "visibility": "public"}])
    assert cb.grade_submission(problem, "print('x' * 5000)\n",
                               limits=FAST)["status"] == cb.ST_PASS


def test_grade_many_rejects_mismatched_lengths():
    """Silently zipping would pair submissions with the wrong problems."""
    with pytest.raises(ValueError):
        cb.grade_many([_stdin_problem()], [ADD_STDIN, ADD_STDIN])


# --------------------------------------------------------------- the replay oracle


def test_the_replay_oracle_passes_both_families():
    for problem in (_stdin_problem(), _functional_problem()):
        r = _grade(problem, cb.replay_oracle(problem))
        assert r["status"] == cb.ST_PASS, r["detail"]


def test_the_replay_oracle_fails_when_the_answer_key_is_perturbed():
    """If it passed anyway the oracle would prove nothing about the comparator."""
    problem = _stdin_problem()
    oracle = cb.replay_oracle(problem)
    problem["tests"][1] = dict(problem["tests"][1], output="999")
    assert _grade(problem, oracle)["status"] != cb.ST_PASS


# ------------------------------------------------------------------- the accounting


def _rec(status):
    return {"status": status, "passed": status == cb.ST_PASS, "finish_reason": None}


def test_every_problem_lands_in_exactly_one_bucket():
    records = [_rec(s) for s in (cb.ST_PASS, cb.ST_PASS, cb.ST_WRONG, cb.ST_RUNTIME,
                                 cb.ST_TIMEOUT, cb.ST_NO_CODE, cb.ST_HARNESS,
                                 cb.ST_GEN_FAILED)]
    row = cb.summarise("b", [None] * len(records), records, {})
    assert sum(row["counts"].values()) == row["n_problems"] == 8


def test_failures_stay_in_the_denominator():
    """A crash, a hang and an unparseable answer are wrong answers, not exclusions."""
    records = [_rec(cb.ST_PASS), _rec(cb.ST_RUNTIME), _rec(cb.ST_TIMEOUT),
               _rec(cb.ST_NO_CODE)]
    row = cb.summarise("b", [None] * 4, records, {})
    assert row["n_graded"] == 4
    assert row["accuracy"] == pytest.approx(0.25)
    assert row["accuracy_all"] == pytest.approx(0.25)


def test_only_generation_failures_leave_the_denominator_and_both_numbers_are_reported():
    """Excluding them biases accuracy upward, so accuracy_all counts them wrong."""
    records = [_rec(cb.ST_PASS), _rec(cb.ST_GEN_FAILED)]
    row = cb.summarise("b", [None] * 2, records, {})
    assert row["n_graded"] == 1 and row["n_failed"] == 1
    assert row["accuracy"] == pytest.approx(1.0)
    assert row["accuracy_all"] == pytest.approx(0.5)


def test_a_harness_error_is_counted_as_a_fail_and_named_separately():
    records = [_rec(cb.ST_PASS), _rec(cb.ST_HARNESS)]
    row = cb.summarise("b", [None] * 2, records, {})
    assert row["n_harness_error"] == 1
    assert row["n_graded"] == 2 and row["accuracy"] == pytest.approx(0.5)


def test_every_status_has_a_reported_count():
    """A bucket with no counter would be a silent exclusion waiting to happen."""
    row = cb.summarise("b", [None], [_rec(cb.ST_PASS)], {})
    for status in cb.STATUSES:
        assert status in row["counts"]


def test_truncation_flags_are_counted():
    records = [dict(_rec(cb.ST_PASS), finish_reason="length"), _rec(cb.ST_PASS)]
    row = cb.summarise("b", [None] * 2, records, {})
    assert row["n_truncated"] == 1


# ------------------------------------------------------------- the real CLI, end to end


def _write_dataset(tmp_path):
    """A synthetic release in the dataset's own schema, including its compressed form."""
    priv_stdin = [{"input": "4 5", "output": "9", "testtype": "stdin"}]
    rows = [
        {"question_title": "add", "question_content": "add two numbers",
         "platform": "atcoder", "question_id": "s1", "contest_id": "c",
         "contest_date": "2025-01-01", "starter_code": "", "difficulty": "easy",
         "public_test_cases": json.dumps(
             [{"input": "1 2", "output": "3", "testtype": "stdin"}]),
         # The real dataset stores private tests base64(zlib(pickle(json))); exercising
         # that path here is what keeps decode_tests honest.
         "private_test_cases": base64.b64encode(
             zlib.compress(pickle.dumps(json.dumps(priv_stdin)))).decode(),
         "metadata": "{}"},
        {"question_title": "add2", "question_content": "add two numbers",
         "platform": "leetcode", "question_id": "f1", "contest_id": "c",
         "contest_date": "2025-01-02",
         "starter_code": "class Solution:\n    def add(self, a, b):\n        ",
         "difficulty": "medium",
         "public_test_cases": json.dumps(
             [{"input": "1\n2", "output": "3", "testtype": "functional"}]),
         "private_test_cases": json.dumps(
             [{"input": "10\n20", "output": "30", "testtype": "functional"}]),
         "metadata": json.dumps({"func_name": "add"})},
        {"question_title": "add3", "question_content": "add two numbers",
         "platform": "atcoder", "question_id": "s2", "contest_id": "c",
         "contest_date": "2025-01-03", "starter_code": "", "difficulty": "hard",
         "public_test_cases": json.dumps(
             [{"input": "7 8", "output": "15", "testtype": "stdin"}]),
         "private_test_cases": json.dumps(
             [{"input": "1 1", "output": "2", "testtype": "stdin"}]),
         "metadata": "{}"},
    ]
    f = tmp_path / "test6.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


@pytest.fixture()
def synthetic(tmp_path, monkeypatch):
    monkeypatch.setenv(cb.DATA_ENV, str(_write_dataset(tmp_path)))
    return tmp_path


def test_load_reads_both_encodings_and_both_families(synthetic):
    problems = cb.load("livecodebench_v6")
    assert [p["question_id"] for p in problems] == ["s1", "f1", "s2"]
    assert [len(p["tests"]) for p in problems] == [2, 2, 2]
    assert problems[0]["tests"][1]["visibility"] == "private"
    assert problems[1]["func_name"] == "add"


def test_load_filters_are_applied_and_an_empty_selection_is_fatal(synthetic):
    assert len(cb.load("livecodebench_v6", difficulty="hard")) == 1
    assert len(cb.load("livecodebench_v6", limit=2)) == 2
    # A filter that selects nothing must be fatal: an empty problem list grades to zero
    # and reads in the output exactly like a model that solved nothing.
    with pytest.raises(ValueError):
        cb.load("livecodebench_v6", difficulty="impossible")


def test_a_missing_dataset_is_fatal_not_an_empty_suite(tmp_path, monkeypatch):
    """An empty problem list scores zero and is indistinguishable from a failing model."""
    monkeypatch.setenv(cb.DATA_ENV, str(tmp_path))
    with pytest.raises(FileNotFoundError):
        cb.load("livecodebench_v6")


def test_a_problem_with_functional_tests_but_no_func_name_is_fatal(tmp_path, monkeypatch):
    row = {"question_title": "x", "question_content": "x", "platform": "leetcode",
           "question_id": "bad", "contest_id": "c", "contest_date": "", "starter_code": "",
           "difficulty": "easy",
           "public_test_cases": json.dumps(
               [{"input": "1", "output": "1", "testtype": "functional"}]),
           "private_test_cases": json.dumps([]), "metadata": "{}"}
    (tmp_path / "test6.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setenv(cb.DATA_ENV, str(tmp_path))
    with pytest.raises(ValueError):
        cb.load("livecodebench_v6")


def test_a_problem_with_no_test_cases_is_fatal(tmp_path, monkeypatch):
    """Zero test cases means every submission passes, which is the worst thing a grader
    can do quietly. It must stop the run rather than score a perfect model."""
    row = {"question_title": "x", "question_content": "x", "platform": "atcoder",
           "question_id": "empty", "contest_id": "c", "contest_date": "",
           "starter_code": "", "difficulty": "easy", "public_test_cases": "[]",
           "private_test_cases": "[]", "metadata": "{}"}
    (tmp_path / "test6.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setenv(cb.DATA_ENV, str(tmp_path))
    with pytest.raises(ValueError):
        cb.load("livecodebench_v6")


def test_question_id_filter_selects_and_a_typo_is_fatal(synthetic):
    """A misspelt id must not silently shrink the benchmark."""
    assert [p["question_id"] for p in
            cb.load("livecodebench_v6", question_ids={"s2", "f1"})] == ["f1", "s2"]
    with pytest.raises(ValueError):
        cb.load("livecodebench_v6", question_ids={"s1", "nope"})


def test_build_prompt_states_the_right_format_for_each_family(synthetic):
    problems = cb.load("livecodebench_v6")
    assert cb.FORMAT_STDIN in cb.build_prompt(problems[0])
    assert cb.FORMAT_WITH_STARTER in cb.build_prompt(problems[1])
    assert "def add" in cb.build_prompt(problems[1])


def test_main_grades_recorded_completions_and_writes_an_auditable_artifact(
        synthetic, tmp_path, monkeypatch, capsys):
    """The REAL loop: CLI -> load -> extract -> sandbox -> summarise -> artifact.

    One correct submission, one that fails only the private test, one with no code block.
    Two of the three are failures of different kinds and both must remain in the
    denominator, so the expected accuracy is exactly 1/3.
    """
    gens = tmp_path / "gens.jsonl"
    gens.write_text("\n".join(json.dumps(g) for g in [
        {"question_id": "s1", "idx": 0, "status": "ok", "finish_reason": "stop",
         "text": "```python\nimport sys\na,b=map(int,sys.stdin.read().split())\n"
                 "print(a+b)\n```"},
        {"question_id": "f1", "idx": 1, "status": "ok", "finish_reason": "stop",
         "text": "```python\nclass Solution:\n    def add(self, a, b):\n"
                 "        return 3\n```"},
        {"question_id": "s2", "idx": 2, "status": "ok", "finish_reason": "length",
         "text": "I am not able to solve this."},
    ]) + "\n")
    out = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", [
        "code_bench.py", "--from-generations", str(gens), "--out", str(out),
        "--workers", "3", "--wall-seconds", "8",
    ])
    assert cb.main() == 0

    row = json.loads(out.read_text())[0]
    assert row["n_problems"] == 3 and row["n_graded"] == 3
    assert row["n_pass"] == 1 and row["n_wrong_answer"] == 1 and row["n_no_code"] == 1
    assert row["n_harness_error"] == 0 and row["n_failed"] == 0
    assert row["accuracy"] == pytest.approx(1 / 3)
    assert row["accuracy_all"] == pytest.approx(1 / 3)
    assert row["n_truncated"] == 1
    assert row["params"]["dataset_md5"]
    assert row["params"]["sandbox_tier"] in ("bwrap", "netns", "subprocess")
    # A regrade calls no endpoint, and must say so rather than inherit a model id.
    assert row["params"]["model"] is None
    assert row["params"]["endpoint"] is None and row["params"]["served_models"] is None
    assert row["params"]["generations_source"] == str(gens)

    records = [json.loads(x) for x in
               (tmp_path / "res.records.jsonl").read_text().splitlines()]
    assert [r["question_id"] for r in records] == ["s1", "f1", "s2"]
    assert [r["status"] for r in records] == [cb.ST_PASS, cb.ST_WRONG, cb.ST_NO_CODE]
    # An aggregate with no per-problem record cannot be audited later.
    assert records[1]["first_fail"] == 1
    assert records[1]["detail"]
    assert all(r["code_sha256"] for r in records)


def test_main_counts_a_missing_generation_as_a_failure_not_a_shorter_benchmark(
        synthetic, tmp_path, monkeypatch):
    gens = tmp_path / "gens.jsonl"
    gens.write_text(json.dumps(
        {"question_id": "s1", "idx": 0, "status": "ok",
         "text": "```python\nimport sys\na,b=map(int,sys.stdin.read().split())\n"
                 "print(a+b)\n```"}) + "\n")
    out = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", [
        "code_bench.py", "--from-generations", str(gens), "--out", str(out),
        "--workers", "3", "--wall-seconds", "8",
    ])
    with pytest.raises(SystemExit):
        cb.main()  # 2 of 3 generations missing is 67%, far above FAILED_RATE_ABORT
    row = json.loads(out.read_text())[0]
    assert row["n_problems"] == 3 and row["n_graded"] == 1 and row["n_failed"] == 2
    assert row["accuracy"] == pytest.approx(1.0)
    assert row["accuracy_all"] == pytest.approx(1 / 3)


def test_load_generations_rejects_an_empty_file(synthetic, tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError):
        cb.load_generations(empty, cb.load("livecodebench_v6"))


# --------------------------------------------------------------------- the real data


@needs_dataset
def test_the_real_release_loads_with_the_expected_shape():
    problems = cb.load("livecodebench_v6")
    assert len(problems) == 175
    assert sum(len(p["tests"]) for p in problems) == 7000
    assert sum(1 for p in problems if p["difficulty"] == "hard") == 80
    assert {p["platform"] for p in problems} == {"atcoder", "leetcode"}
    assert all(p["tests"] for p in problems)
    assert all(p["func_name"] for p in problems if p["platform"] == "leetcode")


@needs_dataset
def test_every_problem_has_private_tests():
    """A problem graded on public tests alone would pass an overfitted submission."""
    for p in cb.load("livecodebench_v6"):
        assert any(t["visibility"] == "private" for t in p["tests"]), p["question_id"]


@needs_dataset
def test_the_suite_names_a_release_that_exists():
    for bench in cb.SUITE:
        assert cb.require_dataset(bench).exists()


# ------------------------------------------------------ which model actually answered
#
# Same exposure as the math suite, through the same shared plumbing: the payload names a
# MODEL ID, an unregistered id is answered HTTP 200 by the BASE model, and a results row
# that does not record the id cannot be attributed afterwards. Driven through the REAL
# main() with a stub session, so the wiring is what is under test, not a helper.


@pytest.fixture()
def stub_endpoint(monkeypatch):
    """Swap aiohttp's session for a stub, so main() runs its real loop with no server.

    Args:
        monkeypatch: pytest's patcher.

    Returns:
        A factory ``(models, completion) -> _StubSession`` that installs the stub.
    """
    import aiohttp

    def install(models=("harnessT49",), completion=""):
        stub = _StubSession(models=models, completion=completion)

        def _open(**kw):
            stub.opened += 1
            return stub

        monkeypatch.setattr(aiohttp, "TCPConnector", lambda **kw: None)
        monkeypatch.setattr(aiohttp, "ClientSession", _open)
        return stub

    return install


def _argv(tmp_path, *extra):
    return ["code_bench.py", "--base-url", "http://stub.invalid/v1",
            "--out", str(tmp_path / "res.json"), "--workers", "2",
            "--wall-seconds", "8", *extra]


def _solution(payload):
    """A submission that solves whichever synthetic problem was asked."""
    prompt = payload["messages"][0]["content"]
    if "class Solution" in prompt:
        return "```python\nclass Solution:\n    def add(self, a, b):\n        return a + b\n```"
    return ("```python\nimport sys\n"
            "print(sum(map(int, sys.stdin.read().split())))\n```")


def test_generating_against_an_unserved_model_id_is_refused(
        synthetic, tmp_path, monkeypatch, stub_endpoint):
    """The severe case: sglang would have answered 200 and served the BASE model."""
    stub = stub_endpoint(models=("base-32b", "evalmodel"))
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--model", "harnessT49"))
    with pytest.raises(SystemExit) as e:
        cb.main()
    assert "harnessT49" in str(e.value) and "base-32b" in str(e.value)
    assert stub.post_calls == [], "refused BEFORE generating, not after"
    assert not (tmp_path / "res.json").exists(), "a refused run writes no score"


def test_the_results_row_records_which_model_answered(
        synthetic, tmp_path, monkeypatch, stub_endpoint):
    stub = stub_endpoint(models=("harnessT49", "base-32b"), completion=_solution)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, "--model", "harnessT49"))
    assert cb.main() == 0
    row = json.loads((tmp_path / "res.json").read_text())[0]
    assert row["params"]["model"] == "harnessT49"
    assert row["params"]["endpoint"] == "http://stub.invalid/v1/chat/completions"
    assert row["params"]["served_models"] == ["harnessT49", "base-32b"]
    assert row["params"]["generations_source"] is None
    assert stub.get_calls == ["http://stub.invalid/v1/models"]
    assert row["n_pass"] == 3, "the stub solved all three synthetic problems"


def test_the_model_flag_has_no_default_that_could_silently_resolve():
    assert cb.build_parser().parse_args([]).model is None


def test_generating_without_a_model_refuses_before_touching_the_endpoint(
        synthetic, tmp_path, monkeypatch, stub_endpoint):
    stub = stub_endpoint(models=("evalmodel",))
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))
    with pytest.raises(SystemExit) as e:
        cb.main()
    # The message names the flag AND the offline alternative -- which is what distinguishes
    # this refusal from the endpoint's own "that id is not served".
    assert "--model" in str(e.value) and "--from-generations" in str(e.value)
    assert stub.opened == 0, "refused before a session was even opened"
    assert stub.get_calls == [] and stub.post_calls == []


# ------------------------------------------------------------- where the data lives


def test_the_snapshot_search_follows_hf_home(tmp_path, monkeypatch):
    """HF_HOME=~/hf_cache on the eval box sent this looking in a directory that is not there."""
    monkeypatch.delenv(cb.DATA_ENV, raising=False)
    snap = tmp_path / "hf" / cb._HF_SUBPATH / "abc123"
    snap.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert cb.snapshot_dir() == snap


def test_the_old_default_cache_stays_a_second_candidate(tmp_path, monkeypatch):
    monkeypatch.delenv(cb.DATA_ENV, raising=False)
    default = tmp_path / "dot_cache"
    monkeypatch.setattr(cb, "_HF_DEFAULT_HOME", str(default))
    snap = default / cb._HF_SUBPATH / "zzz"
    snap.mkdir(parents=True)
    # HF_HOME is set and its cache directory EXISTS but holds no snapshot: the old path
    # must still be tried, and the empty root must not be taken as an answer.
    (tmp_path / "empty_hf" / cb._HF_SUBPATH).mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty_hf"))
    assert cb.snapshot_dir() == snap
    # And HF_HOME outranks it when both hold a snapshot.
    theirs = tmp_path / "empty_hf" / cb._HF_SUBPATH / "aaa"
    theirs.mkdir(parents=True)

    assert cb.snapshot_dir() == theirs


def test_hf_home_is_first_and_the_default_is_always_present(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    roots = [str(r) for r in cb.hf_snapshot_roots()]
    assert len(roots) == 2 and roots[0].startswith(str(tmp_path / "hf"))
    assert roots[1].endswith(cb._HF_SUBPATH) and roots[0].endswith(cb._HF_SUBPATH)
    monkeypatch.delenv("HF_HOME", raising=False)
    assert [str(r) for r in cb.hf_snapshot_roots()] == [roots[1]]


def test_a_missing_snapshot_names_every_root_it_tried(tmp_path, monkeypatch):
    """A dataset that cannot be found must say where it looked, or LCB_DATA gets set by hand."""
    monkeypatch.delenv(cb.DATA_ENV, raising=False)
    monkeypatch.setattr(cb, "_HF_DEFAULT_HOME", str(tmp_path / "dot_cache"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    with pytest.raises(FileNotFoundError) as e:
        cb.snapshot_dir()
    assert str(tmp_path / "hf") in str(e.value)
    assert str(tmp_path / "dot_cache") in str(e.value)
