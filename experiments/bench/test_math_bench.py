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


def test_truncated_final_box_falls_back_to_the_earlier_box():
    """Reversed on 2026-08-29 after measuring what the old rule cost.

    This test used to assert None, on the reasoning that a completion cut off mid-answer
    has not answered. An audit measured the rule's effect across seven checkpoints scored
    on MATH-500: it flipped 0 items for the base model and 21 and 23 items for the two most
    degraded checkpoints (4.2% and 4.6%). Every flip was a finish_reason=="length" item,
    and 92-93% of those repeat the SAME boxed value three or more times before the cut.

    So the rule did not measure "did not answer", it measured "rambled" -- and charged it
    only to the checkpoints under comparison. One item was caught scoring correct at a
    2048-token cap and wrong at 8192 on byte-identical prefix text, meaning a LARGER budget
    produced a LOWER score. A grader whose bias tracks the effect being measured cannot be
    used to measure it.
    """
    assert extract_boxed(r"\boxed{1} then \boxed{") == "1"


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
    # `idx` was added so a scored row traces back to its position in the source file,
    # which is what the committed search/report split addresses.
    assert all(set(r) == {"problem", "answer", "idx"} for r in rows)
    assert all(isinstance(r["answer"], str) for r in rows)
    assert [r["idx"] for r in rows] == list(range(len(rows))), "idx must be file order"


def test_split_halves_are_disjoint_and_cover_the_benchmark():
    """A split that overlapped would let a problem be searched on and reported on."""
    allp = load("math500", "all")
    si = {r["idx"] for r in load("math500", "search")}
    ri = {r["idx"] for r in load("math500", "report")}
    assert len(allp) == 500 and len(si) == 250 and len(ri) == 250
    assert not (si & ri), "halves overlap"
    assert si | ri == {r["idx"] for r in allp}, "halves do not cover the benchmark"


def test_split_is_stable_across_calls():
    """Re-rolling the split per run would let a half be chosen to flatter the method."""
    a = [r["idx"] for r in load("math500", "search")]
    b = [r["idx"] for r in load("math500", "search")]
    assert a == b


def test_split_refuses_when_the_dataset_checksum_moves():
    """Indices address rows by position, so a changed file silently rescopes the split."""
    import json as _json
    import math_bench
    sf = Path(math_bench.__file__).resolve().parent / "math500_split.json"
    orig = sf.read_text()
    d = _json.loads(orig)
    d["dataset_md5"] = "0" * 32
    sf.write_text(_json.dumps(d))
    try:
        with pytest.raises(ValueError, match="md5"):
            load("math500", "search")
    finally:
        sf.write_text(orig)
    assert len(load("math500", "search")) == 250


def test_unknown_split_name_is_rejected():
    with pytest.raises(ValueError):
        load("math500", "trainish")


def test_split_identity_is_pinned_not_merely_structural():
    """Size, disjointness and coverage ALL survive exchanging the two halves.

    A mutation test showed exactly that: swapping search and report, and inverting the
    filter, both passed every structural check. Every number would then have been computed
    on the wrong half with nothing to notice. So the halves are pinned by content.
    """
    search = [r["idx"] for r in load("math500", "search")]
    report = [r["idx"] for r in load("math500", "report")]
    assert search[:8] == [3, 4, 5, 7, 8, 10, 13, 14]
    assert report[:8] == [0, 1, 2, 6, 9, 11, 12, 15]
    assert sum(search) == 63286 and sum(report) == 61464
    assert 3 in search and 3 not in report
    assert 0 in report and 0 not in search


def test_missing_split_file_refuses_rather_than_scoring_everything():
    """Absent the file, silently scoring all 500 would report a search-set number as held out."""
    import math_bench
    sf = Path(math_bench.__file__).resolve().parent / "math500_split.json"
    orig = sf.read_text()
    sf.unlink()
    try:
        # Match the MESSAGE, not just the type: without the explicit guard, read_text()
        # raises the same FileNotFoundError, so a type-only assertion passes with the
        # guard deleted and the explanation of why splits are committed silently lost.
        with pytest.raises(FileNotFoundError, match="committed"):
            load("math500", "search")
    finally:
        sf.write_text(orig)
    assert len(load("math500", "search")) == 250


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


def test_extract_boxed_falls_back_past_a_truncated_final_box():
    """The token cap landing mid-box must not discard an earlier complete answer.

    Returning None here charged 4.2-4.6% of items to the two most degraded checkpoints and
    0.0% to the base model, so it biased exactly the comparison it was used for.
    """
    assert extract_boxed(r"\boxed{7} and \boxed{8") == "7"
    # The shape actually observed in gs173: a verbatim loop cut mid-box.
    assert extract_boxed(r"\boxed{\frac{14}{3}} x \boxed{\frac{14") == r"\frac{14}{3}"
    # Many repeats, final one truncated.
    assert extract_boxed(r"\boxed{5} " * 4 + r"\boxed{5") == "5"


def test_extract_boxed_still_prefers_the_last_complete_box():
    """The fallback must not turn into "first box wins" -- later answers supersede."""
    assert extract_boxed(r"\boxed{1} then \boxed{2}") == "2"
    assert extract_boxed(r"\boxed{1} \boxed{2} \boxed{3}") == "3"


def test_extract_boxed_returns_none_when_no_box_is_balanced():
    """A completion with only a truncated box has genuinely not answered."""
    assert extract_boxed(r"\boxed{a") is None
    assert extract_boxed("no box here") is None
    assert extract_boxed(r"\boxed{\frac{1") is None


def test_extract_boxed_keeps_nested_braces():
    """Guards the original bug: a naive regex mangles \\boxed{\\frac{1}{2}}."""
    assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_boxed(r"\boxed{\text{a}b}") == r"\text{a}b"


def test_extract_boxed_strips_surrounding_whitespace():
    """Models emit \\boxed{ 42 }; an unstripped answer fails string comparison."""
    assert extract_boxed(r"\boxed{ 42 }") == "42"
    assert extract_boxed("\\boxed{\n  7\n}") == "7"
