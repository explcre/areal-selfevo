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
import math_bench as mb  # noqa: E402
from math_bench import (  # noqa: E402
    BENCH_OVERRIDES,
    GEN_KEYS,
    build_parser,
    chat_url,
    explicit_gen_keys,
    extract_boxed,
    grade,
    list_served_models,
    load,
    models_url,
    resolve_params,
    run_bench,
    verify_model,
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


async def _serve(handler, models=("test",)):
    """Start a local aiohttp server on an ephemeral port; yield its /v1 base url.

    It answers /v1/models as well as /v1/chat/completions, because run_bench now verifies
    the requested id against that list BEFORE generating -- an endpoint that cannot say
    what it serves is refused, since an unregistered id is answered HTTP 200 by the base
    model. ``models`` defaults to the id _Args asks for, so the check is satisfied; pass a
    list without it to exercise the refusal.

    Args:
        handler: The chat-completions handler.
        models: Ids this endpoint claims to serve.

    Returns:
        ``(runner, base_url)``.
    """
    from aiohttp import web

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)

    async def _models(request):
        return web.json_response({"object": "list",
                                  "data": [{"id": m} for m in models]})

    app.router.add_get("/v1/models", _models)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = list(runner.addresses)[0][1]
    return runner, f"http://127.0.0.1:{port}/v1"


def _run(handler, models=("test",), **kw):
    async def go():
        runner, url = await _serve(handler, models)
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


# ------------------------------------------------- explicit CLI vs BENCH_OVERRIDES
#
# The bug these close: BENCH_OVERRIDES was applied unconditionally, so an explicit
# `--max-tokens 32768` was discarded in silence. A full 30B re-score therefore ran
# olympiadbench at 16384 and aime at 8192 -- the exact caps it was launched to raise --
# and reported success. Nothing in the results row said the cap had been overruled.


def _explicit(argv):
    """Parse `argv` with the REAL parser and return (namespace, explicit key set)."""
    ap = build_parser()
    args = ap.parse_args(argv)
    return args, explicit_gen_keys(ap, argv)


def test_explicit_gen_keys_reports_a_flag_that_was_named():
    _, keys = _explicit(["--max-tokens", "32768"])
    assert keys == {"max_tokens"}


def test_explicit_gen_keys_reports_nothing_when_a_flag_was_defaulted():
    """A default is not a request, and must not outrank the per-benchmark table."""
    _, keys = _explicit([])
    assert keys == set()
    _, keys = _explicit(["--benchmarks", "aime24", "--limit", "3", "--out", "x.json"])
    assert keys == set()


def test_explicit_gen_keys_handles_the_equals_form():
    """`--max-tokens=32768` is how run_math.sh and the shell chains often write it."""
    _, keys = _explicit(["--max-tokens=32768"])
    assert keys == {"max_tokens"}


def test_explicit_gen_keys_handles_an_unambiguous_abbreviation():
    """argparse accepts `--max-tok`, so a scan of raw argv strings would miss it."""
    args, keys = _explicit(["--max-tok", "32768"])
    assert keys == {"max_tokens"} and args.max_tokens == 32768


def test_explicit_gen_keys_covers_every_generation_key():
    """Each key in GEN_KEYS must be detectable on its own, and only itself.

    A key that has no flag, or whose flag spells its dest differently, would be
    silently undetectable and would go on being overridden by the table.
    """
    values = {"max_tokens": "1234", "temperature": "0.7", "top_p": "0.9", "n": "3",
              "concurrency": "8", "timeout": "77", "seed": "5"}
    assert set(values) == set(GEN_KEYS)
    for key, val in values.items():
        _, keys = _explicit([f"--{key.replace('_', '-')}", val])
        assert keys == {key}, key


def test_explicit_gen_keys_reports_several_flags_at_once():
    _, keys = _explicit(["--max-tokens", "32768", "--seed", "7", "--limit", "5"])
    assert keys == {"max_tokens", "seed"}


def test_explicit_gen_keys_does_not_disturb_the_parser():
    """It re-parses argv; a later parse_args must see exactly what it saw before."""
    ap = build_parser()
    before = vars(ap.parse_args(["--max-tokens", "32768"]))
    explicit_gen_keys(ap, ["--max-tokens", "32768"])
    assert vars(ap.parse_args(["--max-tokens", "32768"])) == before


def test_an_explicit_cap_outranks_the_table():
    """The exact bug: `--max-tokens 32768` on a benchmark the table caps at 8192."""
    args, keys = _explicit(["--max-tokens", "32768"])
    assert BENCH_OVERRIDES["aime24"]["max_tokens"] != 32768
    assert resolve_params("aime24", args, keys)["max_tokens"] == 32768
    assert resolve_params("olympiadbench", args, keys)["max_tokens"] == 32768


def test_the_table_outranks_the_cli_default():
    """Without an explicit flag the per-benchmark cap is the point of the table."""
    args, keys = _explicit([])
    assert args.max_tokens == 4096
    assert resolve_params("aime24", args, keys)["max_tokens"] == 8192
    assert resolve_params("olympiadbench", args, keys)["max_tokens"] == 16384


def test_the_cli_default_is_used_where_the_table_is_silent():
    args, keys = _explicit([])
    assert "math500" not in BENCH_OVERRIDES
    assert resolve_params("math500", args, keys)["max_tokens"] == 4096


def test_the_table_never_touches_a_key_it_does_not_name():
    """An override for max_tokens must not disturb temperature, seed or the rest."""
    args, keys = _explicit(["--temperature", "0.7", "--seed", "9"])
    p = resolve_params("aime24", args, keys)
    assert p["temperature"] == 0.7 and p["seed"] == 9 and p["max_tokens"] == 8192


def test_an_unknown_override_key_is_rejected(monkeypatch):
    """A typo in the table would otherwise run at the default while the results row
    claimed the override -- a number that documents a cap it never used."""
    monkeypatch.setitem(BENCH_OVERRIDES, "aime24", {"max_toknes": 8192})
    args, keys = _explicit([])
    with pytest.raises(ValueError, match="max_toknes"):
        resolve_params("aime24", args, keys)


def test_every_shipped_override_names_a_real_generation_key():
    """Guards the table as committed, not just the checking code."""
    args, keys = _explicit([])
    for bench in BENCH_OVERRIDES:
        resolve_params(bench, args, keys)


def test_skipping_the_table_is_announced(capsys):
    """Silence is what made the 30B re-score look fine; the skip has to be visible."""
    args, keys = _explicit(["--max-tokens", "32768"])
    resolve_params("aime24", args, keys)
    out = capsys.readouterr().out
    assert "aime24" in out and "32768" in out and "8192" in out


def test_no_note_when_the_explicit_value_matches_the_table(capsys):
    args, keys = _explicit(["--max-tokens", "8192"])
    assert resolve_params("aime24", args, keys)["max_tokens"] == 8192
    assert capsys.readouterr().out == ""


def test_resolve_params_returns_exactly_the_generation_keys():
    """The results row's provenance block is these keys; a missing one is a lie."""
    args, keys = _explicit([])
    assert set(resolve_params("aime24", args, keys)) == set(GEN_KEYS)


def test_run_bench_records_the_cap_it_actually_used():
    """The table's cap has to reach the results row, not just the local variable."""
    r = _run(_reply(r"\boxed{1}"))
    assert r["params"]["max_tokens"] == BENCH_OVERRIDES["aime24"]["max_tokens"]


def test_run_bench_honours_an_explicit_cap_passed_to_it():
    """`explicit` is threaded through the call, not read off a module global."""
    async def go():
        runner, url = await _serve(_reply(r"\boxed{1}"))
        try:
            return await run_bench("aime24", _Args(url, max_tokens=32768), None,
                                   {"max_tokens"})
        finally:
            await runner.cleanup()

    r = asyncio.run(go())
    assert r["params"]["max_tokens"] == 32768


def test_main_threads_the_explicit_flags_into_every_benchmark(monkeypatch, capsys):
    """The global has to be set BEFORE the first benchmark resolves its parameters.

    Reading the call order is not enough: this drives the real main() and asserts on
    what run_bench was actually handed, so moving the assignment after the loop, or
    dropping the argument at the call site, fails here.
    """
    import math_bench

    seen = []

    async def fake_run_bench(bench, args, gen_fh=None, explicit=None):
        seen.append((bench, explicit))
        return {"benchmark": bench, "n_problems": 1, "n_graded": 1, "n_failed": 0,
                "n_truncated": 0, "n_no_box": 0, "accuracy": 1.0, "wilson_lo": 0.2,
                "wilson_hi": 1.0, "params": {"max_tokens": 32768}, "cap_limited": False,
                "truncation_rate": 0.0, "seed": 0, "temperature": 0.0}

    monkeypatch.setattr(math_bench, "run_bench", fake_run_bench)
    monkeypatch.setattr(math_bench, "_EXPLICIT", set())
    monkeypatch.setattr(sys, "argv",
                        ["math_bench.py", "--model", "test", "--benchmarks",
                         "aime24,aime25", "--max-tokens", "32768"])
    assert math_bench.main() == 0
    capsys.readouterr()
    assert seen == [("aime24", {"max_tokens"}), ("aime25", {"max_tokens"})]
    assert math_bench._EXPLICIT == {"max_tokens"}


# ----------------------------------------------------- which model actually answered
#
# The request payload names a MODEL ID and nothing else. sglang routes a LoRA adapter by
# that id (--lora-paths NAME=path registers NAME), and answers HTTP 200 for an id it has
# never heard of by serving the BASE model. So an unverified id scores the wrong weights
# in silence, and a results row without the id cannot be attributed afterwards. These
# tests pin the refusal and the record. A STUB endpoint, not a server: the interesting
# cases are an endpoint that answers wrongly or not at all.


class _StubResponse:
    """One canned HTTP reply, shaped like aiohttp's response context manager."""

    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _StubSession:
    """Stand-in for ``aiohttp.ClientSession``: canned /models and /chat/completions.

    Args:
        models: Ids the endpoint claims to serve; an Exception to raise instead of
            answering at all; or None to answer 200 with a body that is not a model list.
        completion: The assistant text every chat request returns, or a callable taking
            the request payload and returning it.
        models_status: HTTP status for the model-list reply.
    """

    def __init__(self, models=("test",), completion="", models_status=200):
        self.models = models
        self.completion = completion
        self.models_status = models_status
        self.get_calls = []
        self.post_calls = []
        # Incremented by whoever installs this as aiohttp.ClientSession. A refusal that
        # is supposed to happen before the endpoint is touched must leave this at 0.
        self.opened = 0

    def get(self, url, timeout=None):
        self.get_calls.append(url)
        if isinstance(self.models, Exception):
            raise self.models
        payload = ({"object": "list", "data": [{"id": m} for m in self.models]}
                   if self.models is not None else {"not": "a model list"})
        return _StubResponse(self.models_status, payload)

    def post(self, url, json=None, timeout=None):
        self.post_calls.append((url, json))
        text = self.completion(json) if callable(self.completion) else self.completion
        return _StubResponse(200, {"choices": [{"message": {"content": text},
                                                "finish_reason": "stop"}]})

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _verify(model, **kw):
    stub = _StubSession(**kw)
    return asyncio.run(verify_model(stub, "http://stub.invalid/v1", model)), stub


def test_the_urls_are_built_from_one_base():
    assert chat_url("http://h:8404/v1") == "http://h:8404/v1/chat/completions"
    assert chat_url("http://h:8404/v1/") == "http://h:8404/v1/chat/completions"
    assert models_url("http://h:8404/v1/") == "http://h:8404/v1/models"


def test_the_served_list_is_read_off_the_endpoint():
    stub = _StubSession(models=("a", "b"))
    assert asyncio.run(list_served_models(stub, "http://stub.invalid/v1")) == ["a", "b"]
    assert stub.get_calls == ["http://stub.invalid/v1/models"]


def test_a_non_200_model_list_is_not_a_model_list():
    stub = _StubSession(models=("a",), models_status=404)
    with pytest.raises(RuntimeError):
        asyncio.run(list_served_models(stub, "http://stub.invalid/v1"))


def test_a_body_that_is_not_a_model_list_is_rejected():
    stub = _StubSession(models=None)
    with pytest.raises(RuntimeError):
        asyncio.run(list_served_models(stub, "http://stub.invalid/v1"))


def test_an_empty_model_list_is_rejected_rather_than_returned():
    """"serves nothing" and "could not tell" must not both look like a bad name."""
    stub = _StubSession(models=())
    with pytest.raises(RuntimeError):
        asyncio.run(list_served_models(stub, "http://stub.invalid/v1"))


def test_an_id_the_endpoint_does_not_serve_is_refused_and_both_sides_are_named():
    with pytest.raises(SystemExit) as e:
        _verify("harnessT49", models=("base-32b", "other"))
    msg = str(e.value)
    assert "harnessT49" in msg, "the refusal must name what was asked for"
    assert "base-32b" in msg and "other" in msg, "and what is available"


def test_an_unverifiable_endpoint_is_a_refusal_not_a_run():
    with pytest.raises(SystemExit):
        _verify("harnessT49", models=OSError("connection refused"))
    with pytest.raises(SystemExit):
        _verify("harnessT49", models=("harnessT49",), models_status=503)


def test_no_model_named_is_itself_a_refusal():
    for missing in (None, ""):
        with pytest.raises(SystemExit) as e:
            _verify(missing)
        assert "--model" in str(e.value)


def test_a_served_id_returns_the_attribution_block():
    block, _ = _verify("harnessT49", models=("harnessT49", "base-32b"))
    assert block == {"model": "harnessT49",
                     "endpoint": "http://stub.invalid/v1/chat/completions",
                     "served_models": ["harnessT49", "base-32b"]}


def test_run_bench_refuses_a_model_the_endpoint_does_not_serve():
    """The live exposure, end to end: the default name would have scored the base model."""
    with pytest.raises(SystemExit) as e:
        _run(_reply(r"\boxed{0}"), models=("base-32b",))
    assert "test" in str(e.value) and "base-32b" in str(e.value)


def test_the_row_records_the_model_the_endpoint_and_the_served_list():
    r = _run(_reply(r"\boxed{0}"), models=("test", "base-32b"))
    assert r["params"]["model"] == "test"
    assert r["params"]["endpoint"].endswith("/v1/chat/completions")
    assert r["params"]["served_models"] == ["test", "base-32b"]


def test_the_model_flag_has_no_default_that_could_silently_resolve():
    """A default id is exactly the failure: unregistered names are served by the base."""
    assert build_parser().parse_args([]).model is None
    assert build_parser().parse_args(["--model", "harnessT49"]).model == "harnessT49"


def test_main_refuses_without_a_model(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["math_bench.py"])
    with pytest.raises(SystemExit) as e:
        mb.main()
    assert e.value.code == 2
    assert "--model" in capsys.readouterr().err
