"""The feedback counter must advance on real work, and must never break the reward path.

`GOAL.md` section 4 lists "Matched feedback budget -- query counts not logged" as NOT MET.
Closing that row with a counter that silently reads zero would be worse than leaving it open,
because a zero would be quoted as evidence. So the tests here are in two groups.

**The counter must bite.** `test_the_counter_does_not_stay_at_zero_across_real_reward_work`
drives AReaL's own `AsyncRewardWrapper.__call__` -- the real retry loop, the real process
pool -- with a real reward function and fails if the counter has not moved. The retry tests
below exist because `batch_size x group_size` is the arithmetic this counter replaces, and
that arithmetic is wrong exactly where retries and refusals happen.

**The counter must not be able to hurt anything.** It sits inside the reward path of a
multi-day run. `test_counting_does_not_change_the_reward` asserts exact equality with and
without it, and `test_a_counter_that_raises_does_not_break_the_reward` proves the failure is
contained -- no budget number is worth ending a run over.
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

fb = pytest.importorskip("selfevo.feedback_budget")
pe = pytest.importorskip("selfevo.periodic_eval")


@pytest.fixture(autouse=True)
def clean_budget():
    """Zero the process-local counter around every test.

    The counter is module-level by necessity -- the instrumented call site is inside upstream
    code with nowhere to thread an argument through -- so tests must not leak into each other.
    """
    fb.reset()
    yield
    fb.reset()


# ------------------------------------------------------------------------- premise ----


def test_premise_a_fresh_counter_is_invisible_not_zero():
    """If this fails, every "counter did not move" assertion below is meaningless.

    `visible` is what separates "no verifier calls happened" from "this process cannot see
    them". A counter that reported a confident zero for the second case is the defect this
    whole file guards.
    """
    c = fb.snapshot()
    assert c.calls == 0
    assert c.visible is False


def test_recording_makes_the_counter_visible():
    """The flag must be able to turn on, or it is a constant and pins nothing."""
    fb.record_call()
    assert fb.snapshot().visible is True


# --------------------------------------------------------------- counting real work ----


def test_the_counter_does_not_stay_at_zero_across_real_reward_work():
    """The test the counter exists to be held to: real dispatches, real count.

    Drives `AsyncRewardWrapper.__call__` itself rather than a stand-in, so this fails if the
    instrumentation is removed from the real call site, not merely from a copy of it.
    """
    api = pytest.importorskip("areal.api.reward_api")
    before = fb.snapshot().calls
    wrapper = api.AsyncRewardWrapper(_reward_one, max_workers=1)
    rewards = [asyncio.run(wrapper("p", "c", [1], [2])) for _ in range(4)]
    after = fb.snapshot()
    assert rewards == [1.0, 1.0, 1.0, 1.0], "premise: the reward function must have run"
    assert after.calls > before, "four rewards were computed and the counter did not move"
    assert after.calls - before == 4
    assert after.visible is True


def _reward_one(*args, **kwargs) -> float:
    """A reward function that always succeeds.

    Module-level and picklable, because `AsyncRewardWrapper` dispatches through a
    `ProcessPoolExecutor` and a closure would not survive the trip.

    Returns:
        1.0.
    """
    return 1.0


def _reward_slow(*args, **kwargs) -> float:
    """A reward function that always outlives the wrapper's timeout.

    Module-level and picklable, because `AsyncRewardWrapper` dispatches through a
    `ProcessPoolExecutor`.

    Returns:
        1.0, eventually -- but never before the wrapper has given up on it.
    """
    import time

    time.sleep(3.0)
    return 1.0


#: On Python 3.10 `asyncio.TimeoutError` is neither the builtin `TimeoutError` nor a subclass
#: of it (they were unified in 3.11). `AsyncRewardWrapper.__call__` catches the BUILTIN, so on
#: this interpreter its timeout branch -- and the `return 0.0` inside it -- is unreachable: a
#: slow verifier falls through to `except Exception` and RAISES on the last attempt instead of
#: degrading to a zero reward. Measured on this box, python 3.10.12; see EXPERIMENTS.md.
_TIMEOUT_BRANCH_REACHABLE = issubclass(asyncio.TimeoutError, TimeoutError)


def test_every_retry_of_one_reward_is_counted_as_its_own_dispatch():
    """Four dispatches for ONE reward: the number `batch_size x group_size` cannot produce.

    Drives the real retry loop in `AsyncRewardWrapper.__call__` with a verifier that always
    outlives the timeout, so the wrapper exhausts `max_retries`. If counting is ever moved
    outside that loop this reads 1 instead of 4, and the budget silently undercounts exactly
    where the verifier is misbehaving -- which is the regime a matched-budget claim is about.

    Asserted around BOTH endings the wrapper can have, because which one occurs is a property
    of the interpreter (see `_TIMEOUT_BRANCH_REACHABLE`) and the invariant under test -- every
    dispatch is counted -- holds either way. Counting happens before the dispatch, so it does
    not depend on which handler catches the failure.
    """
    api = pytest.importorskip("areal.api.reward_api")
    wrapper = api.AsyncRewardWrapper(
        _reward_slow, timeout_seconds=0.3, max_workers=4, max_retries=3
    )
    before = fb.snapshot()
    gave_up = False
    try:
        gave_up = asyncio.run(wrapper("p", "c", [1], [2])) == 0.0
    except Exception:
        gave_up = True
    after = fb.snapshot()
    assert gave_up, "premise: the wrapper must have given up, not succeeded"
    assert after.calls - before.calls == 4, "retries were not counted as dispatches"
    assert after.retries - before.retries == 3


@pytest.mark.skipif(
    not _TIMEOUT_BRANCH_REACHABLE,
    reason="python<3.11: AsyncRewardWrapper's `except TimeoutError` cannot catch "
    "asyncio.TimeoutError, so the `return 0.0` refusal path is unreachable and a slow "
    "verifier raises instead. Recorded in EXPERIMENTS.md rather than worked around.",
)
def test_a_reward_the_verifier_never_returned_is_counted_as_a_refusal():
    """The final-timeout `return 0.0` is a zero that is NOT the verifier saying "wrong".

    Uncounted, it enters the reward distribution as an ordinary wrong answer, which is the
    silent-zero shape this repo keeps being bitten by.
    """
    api = pytest.importorskip("areal.api.reward_api")
    wrapper = api.AsyncRewardWrapper(
        _reward_slow, timeout_seconds=0.3, max_workers=4, max_retries=1
    )
    before = fb.snapshot()
    assert asyncio.run(wrapper("p", "c", [1], [2])) == 0.0
    after = fb.snapshot()
    assert after.refusals - before.refusals == 1, "a non-verdict was recorded as a verdict"
    assert after.calls - before.calls == 2


def test_a_slow_verifier_does_not_silently_become_a_zero_reward_on_this_interpreter():
    """Pin the interpreter-dependent behaviour so a python upgrade cannot change it silently.

    This is not a test of our counter; it is a test of an assumption our counter's refusal
    series rests on. On python<3.11 a verifier timeout PROPAGATES rather than being absorbed
    as reward 0.0, so `budget/verifier_refusals_total` is structurally zero here -- and a zero
    whose cause is unrecorded is exactly what this repo refuses to ship.
    """
    api = pytest.importorskip("areal.api.reward_api")
    wrapper = api.AsyncRewardWrapper(
        _reward_slow, timeout_seconds=0.3, max_workers=4, max_retries=0
    )
    if _TIMEOUT_BRANCH_REACHABLE:
        assert asyncio.run(wrapper("p", "c", [1], [2])) == 0.0
        assert fb.snapshot().refusals == 1
    else:
        with pytest.raises(Exception):
            asyncio.run(wrapper("p", "c", [1], [2]))
        assert fb.snapshot().refusals == 0, (
            "refusals must read zero here BECAUSE the branch is unreachable, not because "
            "the counter is broken"
        )
        assert fb.snapshot().calls == 1, "the dispatch itself must still have been counted"


def test_counting_does_not_change_the_reward():
    """Instrumentation on the live path must be provably inert on the value it observes."""
    api = pytest.importorskip("areal.api.reward_api")
    wrapper = api.AsyncRewardWrapper(_reward_one, max_workers=1)
    with_counting = asyncio.run(wrapper("p", "c", [1], [2]))
    fb.reset()
    assert with_counting == 1.0
    assert with_counting == _reward_one()


def test_a_counter_that_raises_does_not_break_the_reward(monkeypatch):
    """The failure must be contained. A counter that can raise is a counter that can end a run."""

    def explode(*a, **k):
        """Fail the way a broken counter would.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("counter is broken")

    monkeypatch.setattr(fb.current(), "record_call", explode)
    fb.record_call()  # must not propagate
    api = pytest.importorskip("areal.api.reward_api")
    wrapper = api.AsyncRewardWrapper(_reward_one, max_workers=1)
    assert asyncio.run(wrapper("p", "c", [1], [2])) == 1.0


# --------------------------------------------------- retries, refusals, cache hits ----


def test_a_retry_is_counted_as_another_call_and_also_as_a_retry():
    """Two dispatches for one reward. `batch x group` would report one, which is the point."""
    b = fb.FeedbackBudget()
    b.record_call(attempt=0)
    b.record_call(attempt=1)
    c = b.snapshot()
    assert c.calls == 2 and c.retries == 1


def test_the_first_attempt_is_a_call_but_not_a_retry():
    """Otherwise every reward would report a retry and the series would be a constant."""
    b = fb.FeedbackBudget()
    b.record_call(attempt=0)
    c = b.snapshot()
    assert c.calls == 1 and c.retries == 0


def test_a_refusal_is_counted_separately_from_a_verdict():
    """The final-timeout path returns 0.0 -- a zero reward that is not a judgement."""
    b = fb.FeedbackBudget()
    b.record_call(attempt=0)
    b.record_refusal()
    c = b.snapshot()
    assert c.calls == 1 and c.refusals == 1


def test_a_cache_hit_is_never_folded_into_calls():
    """Work avoided and work done are two numbers; adding them answers neither question."""
    b = fb.FeedbackBudget()
    b.record_cache_hit()
    b.record_cache_hit()
    c = b.snapshot()
    assert c.cache_hits == 2
    assert c.calls == 0, "a cache hit was counted as verifier work"


def test_cache_enabled_distinguishes_no_cache_from_a_cache_that_never_hit():
    """A `cache_hits` of zero is ambiguous without this, and every zero here needs an artifact."""
    assert fb.FeedbackBudget(cache_enabled=False).snapshot().cache_enabled is False
    assert fb.FeedbackBudget(cache_enabled=True).snapshot().cache_enabled is True


def test_generated_tokens_accumulate_and_ignore_nonsense():
    """A negative token count is a bug upstream, not a credit against the budget."""
    b = fb.FeedbackBudget()
    b.record_generated_tokens(120)
    b.record_generated_tokens(80)
    b.record_generated_tokens(-5)
    b.record_generated_tokens(0)
    assert b.snapshot().generated_tokens == 200


def test_the_counter_is_thread_safe():
    """Rollout workflows run concurrently; a lost increment undercounts the budget silently."""
    import threading

    b = fb.FeedbackBudget()

    def hammer():
        """Record five thousand calls."""
        for _ in range(5000):
            b.record_call()

    # A short switch interval maximises preemption inside `self._calls += 1`, which is three
    # bytecodes and not atomic. Without this the race is real but rarely observed, and a
    # mutation that removes the lock would survive -- a test that cannot see the defect it
    # names is the "guard that cannot fail" shape this repo keeps rediscovering.
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=hammer) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old_interval)
    assert b.snapshot().calls == 16 * 5000


# ------------------------------------------------------------ what reaches the log ----


def test_the_hook_reports_nan_and_a_flag_when_the_counter_is_invisible():
    """A silent zero would be read as "this arm used no verifier calls"."""
    hook = pe.PeriodicEvalHook(env={})
    m = hook.budget_metrics()
    assert m["budget/counter_visible"] == 0.0
    assert math.isnan(m["budget/verifier_calls_total"])
    assert m["budget/verifier_calls_total"] != 0.0


def test_the_hook_reports_cumulative_and_per_step_counts():
    """Both are asked for, and the per-step one is the difference between readings."""
    hook = pe.PeriodicEvalHook(env={})
    fb.record_call()
    fb.record_call()
    fb.record_generated_tokens(500)
    first = hook.budget_metrics()
    assert first["budget/counter_visible"] == 1.0
    assert first["budget/verifier_calls_total"] == 2.0
    assert first["budget/verifier_calls_step"] == 2.0

    fb.record_call()
    fb.record_generated_tokens(300)
    second = hook.budget_metrics()
    assert second["budget/verifier_calls_total"] == 3.0
    assert second["budget/verifier_calls_step"] == 1.0, "per-step must be a delta, not the total"
    assert second["budget/generated_tokens_total"] == 800.0
    assert second["budget/generated_tokens_step"] == 300.0


def test_budget_metrics_are_emitted_on_every_step_not_only_evaluation_steps():
    """A matched-budget claim is about the whole run, not the steps an eval landed on."""
    hook = pe.PeriodicEvalHook(env={})
    m = hook.maybe_run(global_step=7)  # not an evaluation step; the feature is off entirely
    assert set(m) == pe.BUDGET_KEYS


def test_every_budget_key_is_declared():
    """An undeclared key is a key no test is watching."""
    hook = pe.PeriodicEvalHook(env={})
    assert set(hook.budget_metrics()) == pe.BUDGET_KEYS


def test_a_delta_never_double_counts_across_readings():
    """Two readings with no work between them must show a per-step zero, not a repeat."""
    hook = pe.PeriodicEvalHook(env={})
    fb.record_call()
    hook.budget_metrics()
    again = hook.budget_metrics()
    assert again["budget/verifier_calls_step"] == 0.0
    assert again["budget/verifier_calls_total"] == 1.0
