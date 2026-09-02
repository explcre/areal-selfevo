"""The periodic eval must not read `report`, must not score nothing, and must see a dead adapter.

Three failures this file exists to catch, each of which has already happened here or came
within one weak probe of happening.

1. **Reading the reporting half.** Every arm decision quotes `search`; `report` is quoted
   once, in the write-up. A periodic evaluation runs dozens of times per run, so if it can
   reach `report` it spends the reporting half of the paper. The tests below check this
   twice and in two independent ways -- through the real production call, and by proving the
   guard FIRES on a hand-built tainted record set. A guard that has never been observed to
   fail is not evidence.

2. **Scoring nothing and reporting success.** A benchmark that grades zero problems still
   returns a normal-looking results row. Plotted, its accuracy is indistinguishable from a
   model that got everything wrong. `test_a_dead_endpoint_does_not_produce_a_score` drives
   the whole path against a stub that fails every request and asserts the outcome is a
   status code and a NaN, never a zero.

3. **Declaring a live adapter dead, or a dead one live.** At A0's `globalstep149` the manual
   probe compared greedy TEXT on three trivial prompts, found 0 of 3 differed, printed
   `WARNING: adapter indistinguishable from base` -- and was WRONG: comparing logprobs showed
   the adapter differing by up to 0.041. `test_liveness_sees_an_adapter_that_moves_logprobs_
   but_not_the_argmax` is that exact scenario, and it is the single most important test here.

HOUSE STYLE. Every test drives the real production entry point. Generation goes through a
stub OpenAI endpoint served by the standard library on localhost, so `math_bench.run_bench`
runs unmodified -- real aiohttp, real `verify_model`, real grading, real split filtering. No
GPU, no network beyond loopback, no second scorer anywhere in the file.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "experiments" / "bench"
sys.path.insert(0, str(BENCH))

# math_bench resolves its data root at import time, so this must precede the import. The
# default points at an AZR clone that does not exist on every box; ~/evaldata is where the
# OlympiadBench copy the committed split was built against actually lives.
if not os.environ.get("MATH_EVAL_DATA"):
    _guess = Path(os.path.expanduser("~/evaldata"))
    if (_guess / "olympiadbench" / "test.jsonl").exists():
        os.environ["MATH_EVAL_DATA"] = str(_guess)

mb = pytest.importorskip("math_bench")
pe = pytest.importorskip("selfevo.periodic_eval")

ADAPTER = "a0_math"
BASE = "/models/qwen2.5-32b-instruct"

_HAS_DATA = (mb.DATA / "olympiadbench" / "test.jsonl").exists()
needs_data = pytest.mark.skipif(not _HAS_DATA, reason="olympiadbench data not on this box")


# ----------------------------------------------------------------- the stub endpoint ----
#
# A plain class in the file that uses it, per the house style. It exists so the REAL harness
# can be driven end to end on CPU: anything less than a real HTTP endpoint would mean
# re-deriving what run_bench does, and a test that pins a copy of the code cannot notice the
# copy drifting from the original.


class _Policy:
    """What the stub endpoint answers, mutable per test.

    Attributes:
        models: Ids reported by ``/v1/models``.
        on_models: Called with the list that was just SENT and the 1-based number of the
            ``/v1/models`` request, and may replace the list for subsequent requests by
            returning a new one. After, not before: the real server evicts on its own clock
            between two reads, so a hook that rewrote the list being answered would move the
            window one request earlier than the scenario under test.
        n_models: ``/v1/models`` requests served.
        chat_models: The ``model`` field of every chat completion asked for, in order, so a
            test can prove which weights were generated against rather than which were
            resolved.
        status: HTTP status for chat completions.
        text_for: ``(model, prompt) -> str``, the completion.
        logprobs_for: ``(model, prompt) -> list[float] | None``; None omits the logprobs
            block entirely, which is how a server that does not support them behaves.
        finish_reason: Reported finish reason.
        n_chat: Completions served, so a test can prove the endpoint was really exercised.
    """

    def __init__(self):
        """Start from a healthy endpoint serving both ids."""
        self.models = [BASE, ADAPTER]
        self.on_models = None
        self.n_models = 0
        self.chat_models = []
        self.status = 200
        self.text_for = lambda model, prompt: "Reasoning. The answer is \\boxed{42}"
        self.logprobs_for = lambda model, prompt: [-0.10, -0.20, -0.30]
        self.finish_reason = "stop"
        self.n_chat = 0


class _Handler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible endpoint: ``/v1/models`` and ``/v1/chat/completions``."""

    def log_message(self, *a):
        """Silence the default per-request logging, which floods pytest output."""

    def _send(self, code: int, payload: dict) -> None:
        """Write one JSON response.

        Args:
            code: HTTP status.
            payload: Body.
        """
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """Answer the model list the harness verifies against before generating."""
        p = self.server.policy
        if self.path.rstrip("/").endswith("/models"):
            p.n_models += 1
            sent = list(p.models)
            self._send(200, {"object": "list", "data": [{"id": m} for m in sent]})
            if p.on_models is not None:
                replacement = p.on_models(sent, p.n_models)
                if replacement is not None:
                    p.models = replacement
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        """Answer one chat completion according to the policy."""
        p = self.server.policy
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        p.n_chat += 1
        if p.status != 200:
            self._send(p.status, {"error": "stub failure"})
            return
        model = req.get("model", "")
        p.chat_models.append(model)
        prompt = (req.get("messages") or [{}])[0].get("content", "")
        choice = {
            "message": {"content": p.text_for(model, prompt)},
            "finish_reason": p.finish_reason,
        }
        lps = p.logprobs_for(model, prompt)
        if lps is not None:
            choice["logprobs"] = {"content": [{"token": "t", "logprob": x} for x in lps]}
        self._send(200, {"choices": [choice]})


@pytest.fixture
def endpoint():
    """A running stub endpoint.

    Yields:
        ``(base_url, policy)``. The policy is mutated by the test to choose behaviour.
    """
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.policy = _Policy()
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/v1", srv.policy
    finally:
        srv.shutdown()
        srv.server_close()


def _config(base_url: str, **kw) -> pe.PeriodicEvalConfig:
    """A working configuration pointed at the stub, overridable per test.

    Args:
        base_url: The stub's ``/v1`` url.
        **kw: Field overrides.

    Returns:
        The configuration.
    """
    defaults = dict(
        enabled=True,
        freq_steps=5,
        benchmarks=("olympiadbench",),
        limit=4,
        max_tokens=64,
        concurrency=4,
        timeout=30,
        base_url=base_url,
        model=ADAPTER,
        base_model=BASE,
        probe_prompts=("probe one", "probe two"),
        probe_max_tokens=4,
        patience=3,
    )
    defaults.update(kw)
    return pe.PeriodicEvalConfig(**defaults)


# ------------------------------------------------------------------------- premise ----


def test_premise_the_committed_split_has_two_disjoint_nonempty_halves():
    """If this fails, every split test below is testing nothing."""
    search, report = pe.require_committed_split("olympiadbench")
    assert len(search) == 338 and len(report) == 337
    assert not (search & report)


@needs_data
def test_premise_the_stub_really_drives_the_real_harness(endpoint):
    """The fixture must exercise run_bench end to end, or nothing below is an integration test.

    Asserts the endpoint was hit, the harness graded every problem it asked for, and the
    scoring came back through math_bench's own results row rather than anything local.
    """
    url, policy = endpoint
    cfg = _config(url)
    row, records, calls = _run(pe.run_one_benchmark(cfg, "olympiadbench", ADAPTER))
    assert policy.n_chat == 4, "the harness did not reach the endpoint four times"
    assert row["n_problems"] == 4 and row["n_graded"] == 4
    assert len(records) == 4
    assert calls == 4, "the grader counter did not see the real grader calls"


def _run(coro):
    """Drive one coroutine to completion.

    Args:
        coro: The coroutine.

    Returns:
        Its result.
    """
    import asyncio

    return asyncio.run(coro)


# ------------------------------------------------------- the report half is unreachable ----


@needs_data
def test_the_real_path_grades_only_problems_from_the_search_half(endpoint):
    """The whole point. Checked against the committed file, not against what the code chose.

    The expectation is read from `olympiadbench_split.json` on disk; the actual is the `idx`
    of every generation the harness returned. Neither side is derived from the other, so this
    fails if `--split` is ever wired to the wrong half.
    """
    url, _ = endpoint
    _row, records, _ = _run(
        pe.run_one_benchmark(_config(url, limit=12), "olympiadbench", ADAPTER)
    )
    committed = json.loads((BENCH / "olympiadbench_split.json").read_text())
    seen = {r["idx"] for r in records}
    assert seen, "no generations came back, so nothing was checked"
    assert seen <= set(committed["search"])
    assert not (seen & set(committed["report"]))


@pytest.mark.parametrize("freq", [1, 7, 25, 50])
@pytest.mark.parametrize("limit", [1, 4, 64])
def test_the_namespace_always_carries_the_search_split(freq, limit):
    """Over a grid of configurations, `split` is never anything but `search`.

    `bench_namespace` is the only place the harness's split argument is assigned, so pinning
    it here pins it for every caller.
    """
    cfg = _config("http://x/v1", freq_steps=freq, limit=limit)
    ns = pe.bench_namespace(cfg, "olympiadbench", ADAPTER)
    assert ns.split == pe.EVAL_SPLIT == "search"
    assert ns.split != pe.REPORT_SPLIT


def test_the_split_guard_fires_on_a_report_tainted_record_set():
    """The guard must be observed to FAIL, or it is not evidence that it works.

    Builds a record set from real `report` indices and asserts the refusal names the half.
    Without this, a guard that returned unconditionally would pass every other test here.
    """
    _search, report = pe.require_committed_split("olympiadbench")
    tainted = [{"idx": i} for i in sorted(report)[:3]]
    with pytest.raises(pe.ReportSplitTouched) as exc:
        pe.assert_search_only("olympiadbench", tainted)
    assert "report" in str(exc.value)


def test_the_split_guard_also_refuses_an_index_in_neither_half():
    """A split file that no longer describes the data must not pass silently."""
    with pytest.raises(pe.ReportSplitTouched):
        pe.assert_search_only("olympiadbench", [{"idx": 10**9}])


def test_the_split_guard_accepts_a_clean_search_only_set():
    """The guard must not refuse everything; a guard that always fires guards nothing."""
    search, _ = pe.require_committed_split("olympiadbench")
    assert pe.assert_search_only("olympiadbench", [{"idx": i} for i in sorted(search)[:5]]) == 5


def test_a_benchmark_with_no_committed_split_is_refused_not_scored_whole():
    """No fallthrough to `split=all`, which CONTAINS the reporting half.

    This is the shape of the retracted `partition_from_config` bug: an unlisted name fell
    through to the control's mechanism under a new label. Here the fallthrough would silently
    widen the evaluation to include `report`.
    """
    with pytest.raises(FileNotFoundError) as exc:
        pe.require_committed_split("amc23")
    assert "search" in str(exc.value)


def test_config_refuses_an_unsplit_benchmark_before_any_gpu_work():
    """Validated at config parse, like `harness_variants`, not at the first evaluation."""
    env = _env(BENCHMARKS="olympiadbench,amc23")
    with pytest.raises(FileNotFoundError):
        pe.PeriodicEvalConfig.from_env(env)


def _env(**kw) -> dict:
    """A minimally valid enabling environment, overridable.

    Args:
        **kw: Variables without the `SELFEVO_PERIODIC_EVAL_` prefix.

    Returns:
        The environment mapping.
    """
    env = {
        pe.ENV_ENABLE: "1",
        pe.ENV_PREFIX + "MODEL": ADAPTER,
        pe.ENV_PREFIX + "BASE_MODEL": BASE,
        pe.ENV_PREFIX + "BASE_URL": "http://127.0.0.1:1/v1",
    }
    env.update({pe.ENV_PREFIX + k: str(v) for k, v in kw.items()})
    return env


# ----------------------------------------------------------------- no silent zeroes ----


def test_an_evaluation_that_graded_nothing_is_refused():
    """`n_graded == 0` is a broken run, not a score of zero."""
    with pytest.raises(pe.EmptyEvaluation):
        pe.assert_scored_something({"benchmark": "b", "n_graded": 0, "accuracy": 0.0}, 64)


def test_an_evaluation_that_graded_a_minority_is_refused():
    """Accuracy over survivors is biased upward and is not comparable with the rest of the curve."""
    with pytest.raises(pe.EmptyEvaluation):
        pe.assert_scored_something({"benchmark": "b", "n_graded": 10, "accuracy": 0.9}, 64)


def test_a_nan_accuracy_over_graded_problems_is_refused():
    """A row that graded problems but produced no number is a harness failure."""
    with pytest.raises(pe.EmptyEvaluation):
        pe.assert_scored_something(
            {"benchmark": "b", "n_graded": 64, "accuracy": float("nan")}, 64
        )


def test_a_healthy_row_passes_the_empty_guard():
    """The guard must accept a real row, or it would refuse every evaluation."""
    pe.assert_scored_something({"benchmark": "b", "n_graded": 64, "accuracy": 0.47}, 64)


@needs_data
def test_a_dead_endpoint_does_not_produce_a_score(endpoint, tmp_path):
    """An endpoint that fails every request must yield a status code and NaN, never 0.0.

    Drives the whole hook. A zero accuracy on the curve reads as "the model got everything
    wrong", which is the exact misreading this refuses to allow.
    """
    url, policy = endpoint
    policy.status = 500
    cfg = _config(url, state_path=str(tmp_path / "best.json"))
    tracker = pe.BestValTracker(cfg.patience, cfg.state_path)
    m = pe.run_periodic_eval(cfg, 5, tracker)
    assert m["periodic_eval/status_code"] != pe.STATUS["ok"]
    assert math.isnan(m["periodic_eval/olympiadbench/accuracy"])
    assert m["periodic_eval/olympiadbench/accuracy"] != 0.0
    assert tracker.best_step == -1, "a failed evaluation must not become the best checkpoint"


@needs_data
def test_every_declared_key_is_emitted_even_when_the_evaluation_fails(endpoint, tmp_path):
    """A series with invisible gaps cannot be read; every key is emitted every time."""
    url, policy = endpoint
    policy.status = 500
    cfg = _config(url, state_path=str(tmp_path / "best.json"))
    m = pe.run_periodic_eval(cfg, 5, pe.BestValTracker(cfg.patience, cfg.state_path))
    missing = cfg.metric_keys() - set(m)
    assert not missing, f"declared but not emitted: {sorted(missing)}"


@needs_data
def test_no_key_is_emitted_that_was_never_declared(endpoint, tmp_path):
    """The converse. An undeclared key is a key no test is watching.

    Modelled on `test_harness_selectors.py`'s `SELECTOR_METRIC_KEYS` assertion, the one
    namespace in this tree that pins its own key set.
    """
    url, _ = endpoint
    cfg = _config(url, state_path=str(tmp_path / "best.json"))
    m = pe.run_periodic_eval(cfg, 5, pe.BestValTracker(cfg.patience, cfg.state_path))
    undeclared = set(m) - cfg.metric_keys()
    assert not undeclared, f"emitted but not declared: {sorted(undeclared)}"


# ---------------------------------------------------------------- adapter liveness ----


def test_liveness_sees_an_adapter_that_moves_logprobs_but_not_the_argmax(endpoint):
    """THE step-149 regression. Identical greedy text, different distribution: LIVE.

    The manual probe at A0 `globalstep149` compared greedy text on three trivial prompts,
    found 0 of 3 differed and printed `adapter indistinguishable from base`. Comparing
    logprobs showed max |dlogprob| 0.041 -- the adapter was live and the probe was too weak
    to see it. If this test ever fails, the verdict has been moved back onto text.
    """
    url, policy = endpoint
    policy.text_for = lambda model, prompt: "identical text either way"
    policy.logprobs_for = lambda model, prompt: (
        [-0.100, -0.200] if model == ADAPTER else [-0.141, -0.212]
    )
    rep = _liveness(_config(url))
    assert rep.greedy_differ_frac == 0.0, "premise: the texts must be identical"
    assert rep.max_abs_dlogprob == pytest.approx(0.041, abs=1e-9)
    assert rep.is_live == 1, "a live adapter was called inert on identical greedy text"


def test_liveness_calls_a_bit_identical_adapter_inert(endpoint):
    """The other direction: no distributional difference at all is genuinely inert."""
    url, policy = endpoint
    policy.logprobs_for = lambda model, prompt: [-0.1, -0.2, -0.3]
    rep = _liveness(_config(url))
    assert rep.max_abs_dlogprob == 0.0
    assert rep.is_live == 0


def test_greedy_differ_frac_is_reported_but_does_not_decide(endpoint):
    """Both series exist, and only one of them is the verdict.

    Differing text with bit-identical logprobs is physically odd but is exactly the state a
    text-only probe would call "live"; the verdict must still come from the distribution.
    """
    url, policy = endpoint
    policy.text_for = lambda model, prompt: f"answer for {model}"
    policy.logprobs_for = lambda model, prompt: [-0.5, -0.5]
    rep = _liveness(_config(url))
    assert rep.greedy_differ_frac == 1.0
    assert rep.is_live == 0


def test_liveness_refuses_when_the_endpoint_returns_no_logprobs(endpoint):
    """Neither "assume live" nor "assume inert": both are answers nobody measured."""
    url, policy = endpoint
    policy.logprobs_for = lambda model, prompt: None
    with pytest.raises(pe.LivenessUnavailable):
        _liveness(_config(url))


def test_liveness_counts_the_tokens_it_actually_compared(endpoint):
    """A verdict resting on almost no evidence must be visible as such."""
    url, policy = endpoint
    policy.logprobs_for = lambda model, prompt: [-0.1, -0.2, -0.3, -0.4]
    rep = _liveness(_config(url, probe_prompts=("a", "b", "c")))
    assert rep.n_probes == 3
    assert rep.n_tokens_compared == 12


def _liveness(cfg):
    """Run the liveness probe against a stub endpoint.

    Args:
        cfg: The configuration.

    Returns:
        The :class:`LivenessReport`.
    """
    import asyncio

    import aiohttp

    async def go():
        """Open a session and take one liveness measurement.

        Returns:
            The :class:`LivenessReport`.
        """
        async with aiohttp.ClientSession() as s:
            return await pe.measure_liveness(s, cfg, cfg.model)

    return asyncio.run(go())


# ------------------------------------------------------- best-validation selection ----


def test_best_val_keeps_the_peak_not_the_latest():
    """A decreasing curve: the discriminating case.

    "Keep the latest checkpoint" passes on an increasing curve and fails here, which is the
    whole reason the pair below is a pair.
    """
    t = pe.BestValTracker(patience=99)
    for step, score in [(10, 0.50), (20, 0.42), (30, 0.31), (40, 0.20)]:
        t.update(step, score, f"ckpt{step}")
    assert t.best_step == 10 and t.best_score == 0.50
    assert t.best_checkpoint == "ckpt10"


def test_best_val_keeps_the_last_when_the_curve_is_rising():
    """The other half of the pair: rising, so best genuinely IS the latest.

    Alone this test is passed by an implementation that always keeps the latest; together
    with the decreasing case it is not.
    """
    t = pe.BestValTracker(patience=99)
    for step, score in [(10, 0.20), (20, 0.31), (30, 0.42), (40, 0.50)]:
        t.update(step, score, f"ckpt{step}")
    assert t.best_step == 40 and t.best_score == 0.50


def test_best_val_selects_the_peak_of_a_non_monotone_curve():
    """The realistic shape, where neither "first" nor "last" is right."""
    t = pe.BestValTracker(patience=99)
    for step, score in [(10, 0.30), (20, 0.55), (30, 0.41), (40, 0.52), (50, 0.29)]:
        t.update(step, score, f"ckpt{step}")
    assert t.best_step == 20 and t.best_score == 0.55


def test_a_tie_keeps_the_earlier_checkpoint():
    """Strictly greater than, so a plateau does not walk `best` forward to the latest.

    On a flat curve, greater-or-equal makes selection indistinguishable from no selection.
    """
    t = pe.BestValTracker(patience=99)
    t.update(10, 0.47, "ckpt10")
    t.update(20, 0.47, "ckpt20")
    assert t.best_step == 10 and t.best_checkpoint == "ckpt10"


def test_patience_fires_only_after_n_evaluations_without_improvement():
    """Off by one here silently truncates or never stops a multi-day run."""
    t = pe.BestValTracker(patience=3)
    assert not t.update(10, 0.5, "a").should_stop
    assert not t.update(20, 0.4, "b").should_stop
    assert not t.update(30, 0.4, "c").should_stop
    assert t.update(40, 0.4, "d").should_stop


def test_an_improvement_resets_patience():
    """Otherwise a run stops for a drought it has already ended."""
    t = pe.BestValTracker(patience=3)
    t.update(10, 0.5, "a")
    t.update(20, 0.4, "b")
    t.update(30, 0.4, "c")
    d = t.update(40, 0.6, "d")
    assert d.is_best and d.steps_since_best == 0 and not d.should_stop


def test_best_val_state_survives_a_reload(tmp_path):
    """A 12853-step-per-epoch run WILL be resumed; a forgotten best is a lost selection."""
    p = tmp_path / "best.json"
    t = pe.BestValTracker(patience=3, state_path=p)
    t.update(10, 0.5, "ckpt10")
    t.update(20, 0.4, "ckpt20")
    again = pe.BestValTracker(patience=3, state_path=p)
    assert again.best_step == 10 and again.best_checkpoint == "ckpt10"
    assert again.n_since_best == 1


# ------------------------------------------------------------------- the diagnosis ----


def _rep(is_live: int) -> pe.LivenessReport:
    """A liveness report with a chosen verdict.

    Args:
        is_live: The verdict.

    Returns:
        The report.
    """
    return pe.LivenessReport(2, 0.0, 0.04 if is_live else 0.0, 0.01, is_live, 8)


def test_diagnosis_separates_not_learning_from_learning_but_not_helping():
    """The distinction requirement 5 exists for, and it must be visible in ONE series.

    Same accuracy, same baseline interval; only liveness differs, and the code must differ.
    """
    inert = pe.diagnose(_rep(0), 0.47, (0.43, 0.51))
    live_flat = pe.diagnose(_rep(1), 0.47, (0.43, 0.51))
    assert inert == pe.DIAGNOSIS["adapter_inert"]
    assert live_flat == pe.DIAGNOSIS["live_no_gain"]
    assert inert != live_flat


def test_diagnosis_reports_a_gain_only_outside_the_baseline_interval():
    """Inside the interval of the point it is compared with is not a difference."""
    assert pe.diagnose(_rep(1), 0.60, (0.43, 0.51)) == pe.DIAGNOSIS["live_better"]
    assert pe.diagnose(_rep(1), 0.30, (0.43, 0.51)) == pe.DIAGNOSIS["live_worse"]
    assert pe.diagnose(_rep(1), 0.50, (0.43, 0.51)) == pe.DIAGNOSIS["live_no_gain"]


def test_an_unmeasurable_liveness_is_unknown_not_inert():
    """"We could not tell" must not be plotted as "the adapter is dead"."""
    assert pe.diagnose(None, 0.47, (0.43, 0.51)) == pe.DIAGNOSIS["unknown"]
    assert pe.DIAGNOSIS["unknown"] != pe.DIAGNOSIS["adapter_inert"]


def test_liveness_is_checked_before_accuracy():
    """An inert adapter's score is the base model's; no accuracy reading applies to it."""
    assert pe.diagnose(_rep(0), 0.99, (0.43, 0.51)) == pe.DIAGNOSIS["adapter_inert"]


# --------------------------------------------------------------- off is really off ----


@pytest.mark.parametrize(
    "env,step,why",
    [
        ({}, 50, "absent"),
        ({pe.ENV_ENABLE: "0"}, 50, "explicitly disabled"),
        (_env(FREQ_STEPS=50), 49, "enabled but not a multiple of the cadence"),
        (_env(FREQ_STEPS=50), 0, "enabled, a multiple, but step zero"),
    ],
)
def test_four_ways_of_being_off_are_all_off(env, step, why):
    """Four off-configurations, because "absent" and "disabled" are different bugs.

    Copied in shape from `test_group_routing.py::test_rollback_is_bit_identical`, which
    parametrises over four off-configurations for the same reason.
    """
    cfg = pe.PeriodicEvalConfig.from_env(env)
    assert not cfg.should_run(step), why


def test_the_hook_emits_nothing_evaluation_shaped_when_disabled():
    """With the feature off the trainer's stats dict gains no evaluation key at all."""
    hook = pe.PeriodicEvalHook(env={})
    assert not hook.enabled
    m = hook.maybe_run(global_step=50)
    assert not any(k.startswith("periodic_eval/") for k in m)


def test_a_non_primary_rank_evaluates_nothing():
    """Several ranks each running an hour-long evaluation before a shared barrier is a deadlock.

    A0's own preflight records that AReaL's in-training evaluator "deadlocks this stack",
    which is why this hook gates on rank itself rather than reusing that path.
    """
    hook = pe.PeriodicEvalHook(env=_env(FREQ_STEPS=1))
    assert hook.maybe_run(global_step=50, is_primary=False) == {}


def test_a_bad_cadence_is_refused_at_construction_not_at_the_first_evaluation():
    """A typo must not survive config parse and cost an hour of training to discover."""
    with pytest.raises(ValueError):
        pe.PeriodicEvalConfig.from_env(_env(FREQ_STEPS="every-50"))
    with pytest.raises(ValueError):
        pe.PeriodicEvalConfig.from_env(_env(FREQ_STEPS=0))


def test_a_missing_model_id_is_refused():
    """An unregistered id is answered HTTP 200 by the BASE model, so there is no safe default."""
    env = _env()
    del env[pe.ENV_PREFIX + "MODEL"]
    with pytest.raises(ValueError):
        pe.PeriodicEvalConfig.from_env(env)


def test_the_adapter_and_base_ids_may_not_be_the_same():
    """Otherwise the liveness probe compares the adapter with itself and reports 0 forever."""
    with pytest.raises(ValueError):
        pe.PeriodicEvalConfig.from_env(_env(BASE_MODEL=ADAPTER))


def test_limit_zero_is_refused_because_the_harness_reads_it_as_no_limit():
    """`--limit 0` means "all problems" to math_bench: a very different, much longer run."""
    with pytest.raises(ValueError):
        pe.PeriodicEvalConfig.from_env(_env(LIMIT=0))


# ------------------------------------------------------------------ the cost report ----


def test_throughput_fraction_is_nan_when_the_step_time_is_unknown():
    """An unknown cost is not a free one; 0.0 would read as "this is free"."""
    assert math.isnan(pe.throughput_fraction(55.0, 50, 0.0))


def test_throughput_fraction_is_the_measured_ratio():
    """Cost is reported against the run's own measured step time, not a constant."""
    assert pe.throughput_fraction(54.9, 50, 28.0) == pytest.approx(54.9 / 1400.0)


def test_the_hook_reports_no_step_time_until_it_has_seen_enough_steps():
    """One sample is not a median; reporting one would fabricate a cost.

    The zero it reports instead propagates to a NaN throughput fraction, which is the honest
    reading of "we have not measured this yet".
    """
    hook = pe.PeriodicEvalHook(env=_env(FREQ_STEPS=1000))
    hook.maybe_run(global_step=1)
    hook.maybe_run(global_step=2)
    assert hook.step_seconds() == 0.0, "a cost was reported from one interval"
    for step in range(3, 8):
        hook.maybe_run(global_step=step)
    assert hook.step_seconds() > 0.0


# ------------------------------------------------- the adapter id is a moving target ----
#
# A0's rollout server publishes the adapter as `a0_math-vN` and keeps a rolling window of
# about four; the window advances every 10 to 20 seconds as training publishes weights. A
# fixed id written into the environment at launch is evicted long before the first evaluation
# at step 50, and asking for an evicted id returns HTTP 500 -- so before this section existed
# every point on the curve failed. These tests are the resolution, the pin, and the three
# ways it must refuse rather than substitute.


def _window(newest: int, name: str = ADAPTER, size: int = 4) -> list[str]:
    """A served list shaped like the real one: the base snapshot plus a rolling window.

    Args:
        newest: Version number of the newest member.
        name: Adapter family name.
        size: How many versions the window holds.

    Returns:
        The ids `/v1/models` would report, deliberately NOT in version order -- the real
        server reports whatever order it likes and a resolver that relied on position would
        pass a sorted fixture and fail in production.
    """
    versions = list(range(newest - size + 1, newest + 1))
    return [f"{name}-v{n}" for n in reversed(versions)] + [BASE]


class _FakeInner:
    """The engine `RemoteSGLangEngine` composes, which is where AReaL keeps the addresses."""

    def __init__(self, addresses):
        """Hold one address list.

        Args:
            addresses: ``host:port`` strings.
        """
        self.addresses = list(addresses)


class _FakeEngine:
    """A rollout engine shaped like the real one: composition, no re-exported `addresses`.

    `RemoteSGLangEngine.__init__` sets `self._engine = RemoteInfEngine(...)` and passes ~30
    methods through without passing `addresses` through, so the shape this fixture pins is
    the shape `base_url_from_rollout` actually meets.
    """

    def __init__(self, addresses):
        """Wrap an inner engine holding the addresses.

        Args:
            addresses: ``host:port`` strings.
        """
        self._engine = _FakeInner(addresses)


class _FakeEngineDirect:
    """An engine that exposes `addresses` itself, which the public accessor would look like."""

    def __init__(self, addresses):
        """Hold one address list.

        Args:
            addresses: ``host:port`` strings.
        """
        self.addresses = list(addresses)


class _ExplodingEngine:
    """An engine whose every attribute raises, to prove it was not consulted at all."""

    def __getattr__(self, name):
        """Refuse every attribute access.

        Args:
            name: The attribute asked for.

        Raises:
            AssertionError: Always. Reaching this means the engine was read when the
                documented precedence says an explicit base url wins outright.
        """
        raise AssertionError(f"the engine was consulted for {name!r} despite an explicit base url")


# ------------------------------------------------------------- resolving the newest ----


def test_the_newest_version_in_the_window_is_the_one_resolved():
    """The whole defect in one line: the served list moves, so the id must be read from it."""
    assert pe.select_newest_adapter(_window(13), ADAPTER, BASE) == f"{ADAPTER}-v13"


def test_versions_are_compared_as_numbers_not_strings():
    """`-v9` sorts after `-v10` lexicographically, which is the wrong adapter by four steps."""
    served = [BASE, f"{ADAPTER}-v9", f"{ADAPTER}-v10", f"{ADAPTER}-v2"]
    assert pe.select_newest_adapter(served, ADAPTER, BASE) == f"{ADAPTER}-v10"


def test_an_id_from_another_adapter_family_never_matches():
    """`a0_math_code-v99` is not a newer `a0_math`; a prefix match would score another arm."""
    served = [BASE, f"{ADAPTER}_code-v99", f"{ADAPTER}-v3"]
    assert pe.select_newest_adapter(served, ADAPTER, BASE) == f"{ADAPTER}-v3"


def test_a_configured_version_suffix_is_ignored_because_it_is_stale():
    """The environment names the FAMILY. Whatever version it carries is gone by step 50."""
    assert pe.select_newest_adapter(_window(13), f"{ADAPTER}-v3", BASE) == f"{ADAPTER}-v13"


def test_an_unversioned_id_is_used_only_when_the_endpoint_serves_it():
    """The stable-alias case: the configured id verbatim, and only because it is listed.

    Both halves matter. The first is what a hand-launched single-adapter server looks like;
    the second is the difference between "use the configured id" and "use it whether or not
    anyone serves it", which is the unregistered-id trap that serves the BASE model.
    """
    assert pe.select_newest_adapter([BASE, ADAPTER], ADAPTER, BASE) == ADAPTER
    with pytest.raises(pe.AdapterUnresolved):
        pe.select_newest_adapter([BASE], ADAPTER, BASE)


def test_an_endpoint_serving_only_the_base_model_is_a_refusal_not_a_fallback():
    """THE failure this project has already recorded: a base score wearing an arm's name."""
    with pytest.raises(pe.AdapterUnresolved) as exc:
        pe.select_newest_adapter([BASE], ADAPTER, BASE)
    assert "base" in str(exc.value).lower()


def test_resolution_refuses_to_return_the_base_model_id():
    """The explicit base guard must be observed to FIRE, or it is not evidence.

    Reached by asking for the base id itself, which the endpoint really does serve: without
    the guard the `configured in served` branch would hand it straight back.
    """
    with pytest.raises(pe.AdapterUnresolved) as exc:
        pe.select_newest_adapter([BASE], BASE, BASE)
    assert "BASE" in str(exc.value)


def test_the_version_shape_is_the_one_areal_writes():
    """The expectation comes from AReaL's own formatter, not from a copy of the pattern.

    `get_versioned_lora_name` is the single writer of the served id. Deriving the expectation
    from it means a change there fails here instead of silently making the resolver blind.
    """
    io_struct = pytest.importorskip("areal.api.io_struct")
    served_id = io_struct.get_versioned_lora_name(ADAPTER, 7)
    assert pe.split_adapter_version(served_id) == (ADAPTER, 7)


def test_an_id_with_no_version_reports_none_rather_than_a_number():
    """A base snapshot path has no version, and inventing one would mislabel a curve point."""
    assert pe.split_adapter_version(BASE) == (BASE, None)
    assert pe.split_adapter_version(ADAPTER) == (ADAPTER, None)


# ------------------------------------------------ resolution through the real path ----


@needs_data
def test_the_evaluation_resolves_the_newest_version_at_every_point_not_at_launch(endpoint, tmp_path):
    """Two evaluations, a window that advanced between them, two different versions recorded.

    The discriminating case for caching: an implementation that resolves once and reuses the
    id passes every single-evaluation test in this file and produces a curve whose every
    point after the first is generated against an id the server no longer has.
    """
    url, policy = endpoint
    policy.models = _window(13)
    cfg = _config(url, state_path=str(tmp_path / "best.json"))
    tracker = pe.BestValTracker(cfg.patience, cfg.state_path)

    first = pe.run_periodic_eval(cfg, 5, tracker)
    assert first["periodic_eval/status_code"] == pe.STATUS["ok"]
    assert first["periodic_eval/adapter/version"] == 13.0

    policy.models = _window(17)
    second = pe.run_periodic_eval(cfg, 10, tracker)
    assert second["periodic_eval/status_code"] == pe.STATUS["ok"]
    assert second["periodic_eval/adapter/version"] == 17.0
    assert second["periodic_eval/adapter/n_served"] == 5.0


@needs_data
def test_generation_uses_the_resolved_id_and_never_the_configured_one(endpoint, tmp_path):
    """Asserted on what the ENDPOINT was asked for, not on what the resolver returned.

    The two are different claims: a resolver can return the right id and the benchmark still
    generate against `cfg.model`, which is exactly the defect being fixed.
    """
    url, policy = endpoint
    policy.models = _window(13)
    cfg = _config(url, model=f"{ADAPTER}-v3", state_path=str(tmp_path / "best.json"))
    m = pe.run_periodic_eval(cfg, 5, pe.BestValTracker(cfg.patience, cfg.state_path))
    assert m["periodic_eval/status_code"] == pe.STATUS["ok"]
    assert policy.chat_models, "no completion reached the endpoint, so nothing was checked"
    assert set(policy.chat_models) == {f"{ADAPTER}-v13", BASE}
    assert f"{ADAPTER}-v3" not in policy.chat_models


@needs_data
def test_the_recorded_version_is_the_resolved_one_not_the_configured_one(endpoint, tmp_path):
    """A curve point must be traceable to the weights that produced it, in three places.

    The metric series, the persisted artifact and the best-val history all carry the resolved
    id; the artifact additionally carries the configured one, so the gap between what was
    asked for and what answered is readable after the fact rather than inferred.
    """
    url, policy = endpoint
    policy.models = _window(13)
    cfg = _config(
        url,
        model=f"{ADAPTER}-v3",
        out_dir=str(tmp_path / "out"),
        state_path=str(tmp_path / "best.json"),
    )
    tracker = pe.BestValTracker(cfg.patience, cfg.state_path)
    m = pe.run_periodic_eval(cfg, 5, tracker)

    assert m["periodic_eval/adapter/version"] == 13.0
    saved = json.loads((tmp_path / "out" / "step5" / "results.json").read_text())
    assert saved["resolved_adapter"]["model"] == f"{ADAPTER}-v13"
    assert saved["configured_model"] == f"{ADAPTER}-v3"
    assert tracker.history[-1]["model"] == f"{ADAPTER}-v13"


# ------------------------------------------------------- the window moves under us ----


@needs_data
def test_an_eviction_between_resolving_and_generating_is_refused_not_scored(endpoint, tmp_path):
    """The window advances after resolution and before the first token. Refuse, do not adapt.

    Also pins that the refusal cannot kill the trainer: the harness answers an unserved id
    with `SystemExit`, a BaseException that an `except Exception` would let straight through
    a 2000-step run.
    """
    url, policy = endpoint
    policy.models = _window(13)
    policy.on_models = lambda served, n: _window(17) if n == 1 else None
    cfg = _config(url, state_path=str(tmp_path / "best.json"))
    tracker = pe.BestValTracker(cfg.patience, cfg.state_path)

    m = pe.run_periodic_eval(cfg, 5, tracker)
    assert m["periodic_eval/status_code"] == pe.STATUS["adapter_evicted"]
    assert math.isnan(m["periodic_eval/olympiadbench/accuracy"])
    assert tracker.best_step == -1, "an evicted evaluation must not become the best checkpoint"


@needs_data
def test_an_eviction_after_generating_is_refused_by_the_post_check(endpoint, tmp_path):
    """The generations succeeded and the number is still not recorded.

    The window advances only after the SECOND model listing -- after resolution and after the
    harness verified the id -- so every completion was served and a results row exists. It is
    refused anyway: some of those completions may have come from weights this point does not
    name, and a point that cannot be attributed to one adapter version is not a measurement.
    """
    url, policy = endpoint
    policy.models = _window(13)
    policy.on_models = lambda served, n: _window(19) if n == 2 else None
    cfg = _config(url, state_path=str(tmp_path / "best.json"))
    tracker = pe.BestValTracker(cfg.patience, cfg.state_path)

    m = pe.run_periodic_eval(cfg, 5, tracker)
    assert policy.n_chat > 0, "premise: the benchmark must have generated before the eviction"
    assert m["periodic_eval/status_code"] == pe.STATUS["adapter_evicted"]
    assert math.isnan(m["periodic_eval/olympiadbench/accuracy"])
    assert tracker.best_step == -1


@needs_data
def test_a_window_that_holds_still_is_scored_normally(endpoint, tmp_path):
    """The other direction: the post-check must not refuse every evaluation.

    A guard that always fires guards nothing, and this one sits on the only path that
    produces a curve point at all.
    """
    url, policy = endpoint
    policy.models = _window(13)
    cfg = _config(url, state_path=str(tmp_path / "best.json"))
    m = pe.run_periodic_eval(cfg, 5, pe.BestValTracker(cfg.patience, cfg.state_path))
    assert m["periodic_eval/status_code"] == pe.STATUS["ok"]
    assert not math.isnan(m["periodic_eval/olympiadbench/accuracy"])


@needs_data
def test_a_failed_evaluation_still_names_the_weights_it_was_pinned_to(endpoint, tmp_path):
    """A gap in the curve must say which adapter it was probing, not `UNRESOLVED`.

    Observed on A0 at step 50 with the first version of this code: the evaluation resolved
    `a0_math-vN` perfectly well, every generation then failed for an unrelated reason, and the
    log line read `model=UNRESOLVED` — because the pin travelled only in the return value,
    which the failure destroyed. The failure points are precisely the ones whose weights a
    reader needs named, so the pin is now recovered on every path out.

    The discriminating pair is with `test_a_resolution_failure_is_a_status_code_and_a_nan_
    not_a_zero`, where nothing was ever pinned and the version is genuinely NaN.
    """
    url, policy = endpoint
    policy.models = _window(13)
    policy.on_models = lambda served, n: _window(19) if n == 2 else None
    cfg = _config(url, out_dir=str(tmp_path / "out"), state_path=str(tmp_path / "best.json"))
    tracker = pe.BestValTracker(cfg.patience, cfg.state_path)

    m = pe.run_periodic_eval(cfg, 5, tracker)
    assert m["periodic_eval/status_code"] == pe.STATUS["adapter_evicted"]
    assert math.isnan(m["periodic_eval/olympiadbench/accuracy"]), "a failure is not a score"
    assert m["periodic_eval/adapter/version"] == 13.0, "the failed point forgot its weights"
    saved = json.loads((tmp_path / "out" / "step5" / "results.json").read_text())
    assert saved["resolved_adapter"]["model"] == f"{ADAPTER}-v13"


# --------------------------------------------- a refusal is not a score of anything ----


@needs_data
def test_a_resolution_failure_is_a_status_code_and_a_nan_not_a_zero(endpoint, tmp_path):
    """Nothing matching on the endpoint: the point is absent from the curve, with a reason."""
    url, policy = endpoint
    policy.models = [BASE]
    cfg = _config(url, state_path=str(tmp_path / "best.json"))
    tracker = pe.BestValTracker(cfg.patience, cfg.state_path)

    m = pe.run_periodic_eval(cfg, 5, tracker)
    assert m["periodic_eval/status_code"] == pe.STATUS["adapter_unresolved"]
    assert math.isnan(m["periodic_eval/olympiadbench/accuracy"])
    assert math.isnan(m["periodic_eval/adapter/version"])
    assert policy.n_chat == 0, "a token was generated against an unresolved adapter"
    assert tracker.best_step == -1
    assert not (cfg.metric_keys() - set(m)), "a failed point must still fill every series"


@needs_data
def test_a_refusal_and_a_genuine_zero_are_different_points_in_the_series(endpoint, tmp_path):
    """The discriminating pair. Both are "no accuracy to speak of" and they must not read alike.

    A model that gets every problem wrong is a real measurement of 0.0 with status ok; an
    adapter that could not be resolved is NaN with its own status code. Collapsing the second
    onto the first is how a broken evaluation becomes a plotted collapse.
    """
    url, policy = endpoint
    policy.models = _window(13)
    policy.text_for = lambda model, prompt: "Reasoning. The answer is \\boxed{-99999}"
    cfg = _config(url, state_path=str(tmp_path / "zero.json"))
    genuine = pe.run_periodic_eval(cfg, 5, pe.BestValTracker(cfg.patience, cfg.state_path))

    policy.models = [BASE]
    refusal = pe.run_periodic_eval(
        _config(url, state_path=str(tmp_path / "refused.json")),
        5,
        pe.BestValTracker(3, str(tmp_path / "refused.json")),
    )

    assert genuine["periodic_eval/status_code"] == pe.STATUS["ok"]
    assert genuine["periodic_eval/olympiadbench/accuracy"] == 0.0
    assert refusal["periodic_eval/status_code"] == pe.STATUS["adapter_unresolved"]
    assert math.isnan(refusal["periodic_eval/olympiadbench/accuracy"])


# ------------------------------------------------------------ where to evaluate ----


def test_the_endpoint_comes_from_the_engine_the_trainer_generates_against():
    """The port is allocated by the launcher and the host is not localhost, so it is read.

    Both engine shapes: the composed one the real `RemoteSGLangEngine` is, and a direct
    `addresses` attribute, because a future passthrough on the wrapper must not break this.
    """
    assert pe.base_url_from_rollout(_FakeEngine(["172.28.127.18:32735"])) == (
        "http://172.28.127.18:32735/v1"
    )
    assert pe.base_url_from_rollout(_FakeEngineDirect(["10.0.0.2:9000"])) == (
        "http://10.0.0.2:9000/v1"
    )


def test_an_engine_with_no_address_is_a_refusal_not_a_localhost_guess():
    """A0's server binds the host interface: a probe of localhost finds nothing at all."""
    with pytest.raises(RuntimeError) as exc:
        pe.base_url_from_rollout(_FakeEngine([]))
    assert "127.0.0.1" not in str(exc.value)


def test_the_hook_takes_the_address_from_the_engine_when_none_is_configured():
    """With no BASE_URL in the environment the hook still knows where to evaluate."""
    env = _env()
    del env[pe.ENV_PREFIX + "BASE_URL"]
    hook = pe.PeriodicEvalHook(env=env, rollout=_FakeEngine(["172.28.127.18:32735"]))
    assert hook.config.base_url == ""
    assert hook._eval_config().base_url == "http://172.28.127.18:32735/v1"


def test_an_explicit_base_url_wins_and_the_engine_is_not_consulted():
    """The documented precedence, asserted on the engine never being touched.

    Two places to look for an address is how a run scores the wrong server, so the rule is
    one or the other and never a merge.
    """
    hook = pe.PeriodicEvalHook(env=_env(), rollout=_ExplodingEngine())
    assert hook._eval_config().base_url == "http://127.0.0.1:1/v1"


def test_a_missing_endpoint_is_a_status_code_not_a_crash():
    """An engine that cannot say where it serves must not take the training run down."""
    env = _env(FREQ_STEPS=1)
    del env[pe.ENV_PREFIX + "BASE_URL"]
    hook = pe.PeriodicEvalHook(env=env, rollout=_FakeEngine([]))
    m = hook.maybe_run(global_step=1)
    assert m["periodic_eval/status_code"] == pe.STATUS["endpoint_error"]
    assert math.isnan(m["periodic_eval/olympiadbench/accuracy"])


def test_the_base_url_is_still_required_outside_a_trainer():
    """Without an engine there is nothing to read the address off, so it must be configured."""
    env = _env()
    del env[pe.ENV_PREFIX + "BASE_URL"]
    with pytest.raises(ValueError):
        pe.PeriodicEvalConfig.from_env(env)
