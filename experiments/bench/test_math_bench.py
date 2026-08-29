"""Tests for the math benchmark harness.

The grader tests came first and an audit confirmed they constrain it (all 4 grader mutants
killed). The audit also found that **8 of 12 mutants survived** in `run_bench`, `generate`
and `load`, which had no tests at all -- failed-generation accounting, the interval,
per-problem vs flattened averaging, swapped `verify` args, a `--limit` off-by-one, removed
retries, and empty-string handling. The second half of this file closes that gap by running
`run_bench` against a real local HTTP server that can be told to misbehave.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from math_bench import (  # noqa: E402
    extract_boxed,
    grade,
    load,
    run_bench,
    wilson,
)

# --------------------------------------------------------------------- extract_boxed


def test_simple():
    assert extract_boxed(r"so \boxed{42}") == "42"


def test_nested_braces_not_truncated():
    assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_boxed(r"\boxed{\frac{\sqrt{3}}{2}}") == r"\frac{\sqrt{3}}{2}"


def test_last_boxed_wins():
    assert extract_boxed(r"first \boxed{1} then \boxed{2}") == "2"


def test_missing_and_unbalanced():
    assert extract_boxed("no box here") is None
    assert extract_boxed(r"\boxed{1") is None


def test_truncated_final_box_discards_earlier_box():
    """A completion cut off mid-answer has not answered, even if an earlier box exists."""
    assert extract_boxed(r"\boxed{1} then \boxed{") is None


def test_text_after_final_brace_is_ignored():
    assert extract_boxed(r"\boxed{7}. Therefore done.") == "7"


# ----------------------------------------------------------------------------- grade


def test_grade_exact_and_equivalent():
    assert grade(r"\boxed{42}", "42")
    assert not grade(r"\boxed{41}", "42")
    assert not grade("no answer", "42")


def test_grade_symbolic_equivalence():
    assert grade(r"\boxed{\frac{1}{2}}", r"\frac{1}{2}")
    assert grade(r"\boxed{0.5}", r"\frac{1}{2}")


def test_grade_tolerates_formatting():
    assert grade(r"\boxed{ 42 }", "42")
    assert grade(r"\boxed{1,000}", "1000") or grade(r"\boxed{1000}", "1,000")


def test_percent_answer_against_bare_gold():
    """The audit's single false negative: math_verify parses 10\\% as 0.1."""
    assert grade(r"\boxed{10\%}", "10")
    assert grade(r"\boxed{10}", "10")


def test_percent_stripping_does_not_create_false_positives():
    assert not grade(r"\boxed{10\%}", "42")


def test_verify_argument_order_is_not_symmetric():
    """The audit found a real case where swapping verify(gold,pred) yields a false positive."""
    assert not grade(r"\boxed{0}", "5x-7y+11z+4=0")


# ---------------------------------------------------------------------------- wilson


def test_wilson_is_not_degenerate_at_the_extremes():
    """A binomial SE is exactly 0 at 0/30 and 30/30, asserting certainty. Wilson is not."""
    lo, hi = wilson(0, 30)
    assert lo == 0.0 and hi > 0.05, (lo, hi)
    lo, hi = wilson(30, 30)
    assert hi == 1.0 and lo < 0.95, (lo, hi)


def test_wilson_never_leaves_the_unit_interval():
    for k in (0, 1, 15, 29, 30):
        lo, hi = wilson(k, 30)
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_is_asymmetric_at_low_counts():
    lo, hi = wilson(1, 30)
    p = 1 / 30
    assert (hi - p) > (p - lo)


def test_wilson_empty_is_nan_not_zero():
    lo, hi = wilson(0, 0)
    assert lo != lo and hi != hi  # NaN


# ------------------------------------------------------------------------------ load


def test_load_rejects_missing_benchmark():
    with pytest.raises(FileNotFoundError):
        load("no_such_benchmark_xyz")


def test_load_real_benchmark_schema():
    rows = load("aime24")
    assert len(rows) >= 29
    assert all(set(r) == {"problem", "answer"} for r in rows)
    assert all(isinstance(r["answer"], str) for r in rows)


# -------------------------------------------------------------- run_bench, end to end


class _Args:
    """Minimal args object; mirrors the argparse namespace run_bench reads."""

    def __init__(self, base_url, **kw):
        self.base_url = base_url
        self.model = "test"
        self.limit = 2
        self.n = 1
        self.temperature = 0.0
        self.top_p = 1.0
        self.max_tokens = 64
        self.concurrency = 4
        self.timeout = 5
        self.seed = 0
        for k, v in kw.items():
            setattr(self, k, v)


async def _serve(handler):
    """Start a local aiohttp server on an ephemeral port; yield its /v1 base url."""
    from aiohttp import web

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = list(runner.addresses)[0][1]
    return runner, f"http://127.0.0.1:{port}/v1"


def _run(handler, **kw):
    async def go():
        runner, url = await _serve(handler)
        try:
            return await run_bench("aime24", _Args(url, **kw))
        finally:
            await runner.cleanup()

    return asyncio.run(go())


def _reply(content, finish="stop"):
    from aiohttp import web

    async def h(request):
        return web.json_response(
            {"choices": [{"message": {"content": content}, "finish_reason": finish}]}
        )

    return h


def test_all_correct_scores_one():
    rows = load("aime24")[:2]
    from aiohttp import web

    async def h(request):
        body = await request.json()
        prompt = body["messages"][0]["content"]
        gold = next(r["answer"] for r in rows if r["problem"] in prompt)
        return web.json_response(
            {"choices": [{"message": {"content": f"\\boxed{{{gold}}}"},
                          "finish_reason": "stop"}]}
        )

    r = _run(h)
    assert r["accuracy"] == 1.0
    assert r["n_graded"] == 2 and r["n_failed"] == 0


def test_all_wrong_scores_zero_and_is_graded():
    r = _run(_reply(r"\boxed{-99999}"))
    assert r["accuracy"] == 0.0
    assert r["n_graded"] == 2, "wrong answers must stay in the denominator"


def test_empty_200_response_is_wrong_not_excluded():
    """Excluding empty completions from the denominator INFLATES accuracy."""
    r = _run(_reply(""))
    assert r["n_graded"] == 2, "empty completion is a wrong answer, not a failure"
    assert r["n_failed"] == 0
    assert r["accuracy"] == 0.0


def test_truncation_is_counted_not_just_graded_wrong():
    r = _run(_reply(r"\boxed{1", finish="length"))
    assert r["n_truncated"] == 2, "truncation must be visible, not silently wrong"
    assert r["accuracy"] == 0.0


def test_no_box_is_counted():
    r = _run(_reply("the answer is 42"))
    assert r["n_no_box"] == 2


def test_server_errors_are_failures_and_shrink_the_denominator_visibly():
    from aiohttp import web

    async def h(request):
        return web.Response(status=500)

    r = _run(h)
    assert r["n_failed"] == 2
    assert r["n_graded"] == 0, "nothing gradable"
    assert r["accuracy"] != r["accuracy"], "must be NaN, not 0.0"


def test_retries_happen_before_giving_up():
    from aiohttp import web

    calls = {"n": 0}

    async def h(request):
        calls["n"] += 1
        if calls["n"] <= 2:
            return web.Response(status=503)
        return web.json_response(
            {"choices": [{"message": {"content": r"\boxed{0}"}, "finish_reason": "stop"}]}
        )

    _run(h, limit=1)
    assert calls["n"] >= 3, "must retry, not fail on the first error"


def test_limit_is_exact_not_off_by_one():
    r = _run(_reply(r"\boxed{0}"), limit=3)
    assert r["n_problems"] == 3


def test_per_problem_averaging_not_flattened():
    """avg@n averages within a problem first; flattening would weight problems unequally
    whenever some samples fail."""
    from aiohttp import web

    rows = load("aime24")[:2]
    # Fail PERMANENTLY for one problem. A transient failure is not enough: the 3 retries
    # would succeed and nothing would be recorded -- which is correct behaviour, and was
    # a bug in an earlier version of this test rather than in the harness.
    async def h(request):
        body = await request.json()
        if rows[1]["problem"] in body["messages"][0]["content"]:
            return web.Response(status=500)
        return web.json_response(
            {"choices": [{"message": {"content": r"\boxed{0}"}, "finish_reason": "stop"}]}
        )

    async def go():
        runner, url = await _serve(h)
        try:
            return await run_bench("aime24", _Args(url, limit=2, n=2, temperature=0.7))
        finally:
            await runner.cleanup()

    r = asyncio.run(go())
    assert r["n_failed"] == 2, "both samples of the permanently-failing problem"
    assert r["n_graded"] == 1, "only the problem with surviving samples is graded"
    assert r["n_problems"] == 2, "the denominator shortfall must remain visible"


def test_metadata_is_recorded_for_reproducibility():
    r = _run(_reply(r"\boxed{0}"))
    assert r["seed"] == 0
    assert r["temperature"] == 0.0


def test_generations_are_persisted(tmp_path):
    """Nothing else matters if a number cannot be re-audited without regenerating it."""
    out = tmp_path / "gen.jsonl"

    async def go():
        runner, url = await _serve(_reply(r"\boxed{7}"))
        try:
            with out.open("w") as fh:
                return await run_bench("aime24", _Args(url, limit=2), fh)
        finally:
            await runner.cleanup()

    asyncio.run(go())
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(recs) == 2
    for r in recs:
        assert r["text"] == r"\boxed{7}"
        assert r["boxed"] == "7"
        assert r["gold"] and r["status"] == "ok"
        assert "correct" in r and "finish_reason" in r
