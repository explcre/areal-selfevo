"""Count verifier and grader invocations, so a matched-feedback-budget claim is checkable.

WHY THIS EXISTS. ``GOAL.md`` section 4 tracks compliance with the evaluation bar of arXiv
2607.12227, and two rows read NOT MET. One of them, *"Matched feedback budget -- query counts
not logged"*, is the direct attack surface: an arm that simply calls the verifier more times
can score better without being better, and with no counter there is no way to answer the
reviewer except by rerunning everything.

WHY THE COUNT IS NOT ARITHMETIC. The obvious substitute -- ``batch_size x group_size`` -- is
wrong on this stack, and wrong in a direction that flatters us. ``AsyncRewardWrapper.__call__``
(``areal/api/reward_api.py``) dispatches the reward function inside
``for attempt in range(self.max_retries + 1)``, so a timeout costs **two or more** verifier
invocations for one reward, and on the final timeout it returns ``0.0`` -- a reward that was
never actually computed. Counting is therefore done at the dispatch site, one increment per
dispatch, with refusals counted separately from calls that produced a verdict.

WHAT THE SERIES MEAN.

* ``calls`` -- dispatches to the verifier. Retries included, because each one is real work.
* ``retries`` -- of those, the ones that were not the first attempt for their reward.
* ``refusals`` -- rewards returned without a verdict (the final-timeout ``return 0.0`` path).
  A refusal is a zero reward that is not a judgement, and it must be visible as such.
* ``cache_hits`` -- verdicts served without a dispatch. Counted separately, never folded into
  ``calls``, so a reader sees the work done and the work avoided as two numbers.
* ``cache_enabled`` -- whether any cache is installed at all. Without it a ``cache_hits`` of
  zero is ambiguous between "no cache exists on this path" and "a cache that never hit", and
  this repo's standing rule is that every zero has an artifact behind it. **On the current
  reward path no cache exists, so this reads 0 and ``cache_hits`` is structurally zero.**
* ``generated_tokens`` -- completion tokens produced. The cheap half of the other NOT MET row
  (*matched inference budget*), free at the reward call site because the response object is
  already in hand. Prompt tokens and rollout-internal retries are NOT counted here; that
  would mean instrumenting the rollout path, which this module deliberately does not touch.

SAFETY. Counting must never be able to change or break a training run. Every public entry
point swallows its own exceptions (:func:`_safe`), and ``test_feedback_budget.py`` asserts
both that the reward value is unchanged bit-for-bit with counting on and that a counter which
raises on every call still lets the reward through.

VISIBILITY. The counter is process-local. Whether the process that publishes metrics is the
same process that computes rewards is a property of the deployment, not of this module, so
:func:`snapshot` reports ``visible`` -- False when nothing has ever been counted here. A
reader of the W&B curve must be able to tell "no verifier calls happened" from "this process
cannot see the verifier calls", and a counter that silently reports zero is worse than no
counter.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

__all__ = [
    "COUNTER_KEYS",
    "FeedbackBudget",
    "FeedbackCounts",
    "current",
    "record_cache_hit",
    "record_call",
    "record_generated_tokens",
    "record_refusal",
    "reset",
    "snapshot",
]

#: The names this module counts. Declared so the emitting side can be asserted against a
#: registry rather than trusting an f-string, the way ``SELECTOR_METRIC_KEYS`` is.
COUNTER_KEYS = ("calls", "retries", "refusals", "cache_hits", "generated_tokens")


@dataclass(frozen=True)
class FeedbackCounts:
    """One reading of the counters.

    Attributes:
        calls: Verifier dispatches, retries included.
        retries: Dispatches that were not the first attempt for their reward.
        refusals: Rewards returned without a verdict.
        cache_hits: Verdicts served without a dispatch.
        generated_tokens: Completion tokens seen at the reward call site.
        cache_enabled: Whether a cache is installed on the counted path.
        visible: Whether anything has ever been counted in this process. False means the
            reading is uninformative, NOT that the work did not happen.
    """

    calls: int = 0
    retries: int = 0
    refusals: int = 0
    cache_hits: int = 0
    generated_tokens: int = 0
    cache_enabled: bool = False
    visible: bool = False

    def __sub__(self, other: "FeedbackCounts") -> "FeedbackCounts":
        """Difference between two readings, for a per-step delta.

        Args:
            other: The earlier reading.

        Returns:
            A reading holding the differences. ``cache_enabled`` and ``visible`` are taken
            from ``self``, since they describe the path rather than an amount.
        """
        return FeedbackCounts(
            calls=self.calls - other.calls,
            retries=self.retries - other.retries,
            refusals=self.refusals - other.refusals,
            cache_hits=self.cache_hits - other.cache_hits,
            generated_tokens=self.generated_tokens - other.generated_tokens,
            cache_enabled=self.cache_enabled,
            visible=self.visible,
        )


class FeedbackBudget:
    """A thread-safe tally of verifier work, and the work a cache avoided.

    Thread-safe because AReaL runs rollout workflows concurrently: ``WorkflowExecutor``
    drives many ``arun_episode`` coroutines at once, and every one of them reaches the reward
    call site.
    """

    def __init__(self, cache_enabled: bool = False):
        """Create an empty tally.

        Args:
            cache_enabled: Whether a verdict cache exists on the path being counted. Left
                False on the current reward path, where none does.
        """
        self._lock = threading.Lock()
        self._calls = 0
        self._retries = 0
        self._refusals = 0
        self._cache_hits = 0
        self._tokens = 0
        self._touched = False
        self.cache_enabled = bool(cache_enabled)

    def record_call(self, attempt: int = 0) -> None:
        """Count one dispatch to the verifier.

        Called once per loop iteration of ``AsyncRewardWrapper.__call__``, which is what
        makes retries countable instead of invisible.

        Args:
            attempt: Zero-based attempt index. Anything above zero is also a retry.
        """
        with self._lock:
            self._calls += 1
            if attempt > 0:
                self._retries += 1
            self._touched = True

    def record_refusal(self) -> None:
        """Count one reward returned without a verdict.

        The final-timeout path returns ``0.0``. That zero is not a judgement about the
        completion, and folding it into the reward distribution unremarked is the
        silent-zero shape this project keeps being bitten by.
        """
        with self._lock:
            self._refusals += 1
            self._touched = True

    def record_cache_hit(self) -> None:
        """Count one verdict served without dispatching to the verifier.

        Never increments :meth:`record_call`. The whole point of a separate counter is that
        work avoided and work done are different numbers.
        """
        with self._lock:
            self._cache_hits += 1
            self._touched = True

    def record_generated_tokens(self, n: int) -> None:
        """Count completion tokens seen at the reward call site.

        Args:
            n: Tokens in this completion. Non-positive values are ignored rather than
                subtracted, since a negative token count is a bug upstream, not a credit.
        """
        if n <= 0:
            return
        with self._lock:
            self._tokens += int(n)
            self._touched = True

    def snapshot(self) -> FeedbackCounts:
        """Read the counters without resetting them.

        Returns:
            The current cumulative reading.
        """
        with self._lock:
            return FeedbackCounts(
                calls=self._calls,
                retries=self._retries,
                refusals=self._refusals,
                cache_hits=self._cache_hits,
                generated_tokens=self._tokens,
                cache_enabled=self.cache_enabled,
                visible=self._touched,
            )

    def reset(self) -> None:
        """Zero the counters, including the visibility flag.

        Used by tests. A live run never resets: the cumulative series is the point.
        """
        with self._lock:
            self._calls = self._retries = self._refusals = self._cache_hits = 0
            self._tokens = 0
            self._touched = False


#: The process-local tally. Module-level rather than passed around, because the counted call
#: site is inside upstream AReaL code that has nowhere to thread an argument through.
_BUDGET = FeedbackBudget()


def current() -> FeedbackBudget:
    """The process-local budget.

    Returns:
        The single :class:`FeedbackBudget` every recording function writes to.
    """
    return _BUDGET


def _safe(fn, *args) -> None:
    """Run one recording call, swallowing anything it raises.

    Instrumentation sits inside the reward path of a multi-day run. A counter that can raise
    is a counter that can end the run, and no budget number is worth that.

    Args:
        fn: The bound recording method.
        *args: Its arguments.
    """
    try:
        fn(*args)
    except Exception:  # pragma: no cover - defensive; asserted by test_feedback_budget
        pass


def record_call(attempt: int = 0) -> None:
    """Count one verifier dispatch on the process-local budget.

    Args:
        attempt: Zero-based attempt index; above zero also counts a retry.
    """
    _safe(_BUDGET.record_call, attempt)


def record_refusal() -> None:
    """Count one reward returned without a verdict on the process-local budget."""
    _safe(_BUDGET.record_refusal)


def record_cache_hit() -> None:
    """Count one cached verdict on the process-local budget."""
    _safe(_BUDGET.record_cache_hit)


def record_generated_tokens(n: int) -> None:
    """Count completion tokens on the process-local budget.

    Args:
        n: Tokens in this completion.
    """
    _safe(_BUDGET.record_generated_tokens, n)


def snapshot() -> FeedbackCounts:
    """Read the process-local budget.

    Returns:
        The cumulative reading, whose ``visible`` field says whether it means anything.
    """
    try:
        return _BUDGET.snapshot()
    except Exception:  # pragma: no cover - defensive
        return FeedbackCounts()


def reset() -> None:
    """Zero the process-local budget. Tests only."""
    _safe(_BUDGET.reset)
