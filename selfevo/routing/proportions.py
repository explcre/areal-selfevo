"""Measuring realised mode proportions, and the control that matches them exactly.

The design requires a control run "at the proportions the criterion router actually
produced, **measured, not assumed**". Configuring a :class:`RandomRouter` with nominal
probabilities does **not** achieve that, and the failure is data-dependent rather than
cosmetic:

    RandomRouter({rl: 0.5, sft: 0.5}) on teacherless contexts realises {rl: 0.499,
    skip: 0.501} -- half the mass migrates to SKIP.

Because :class:`SolveRateRouter` degrades to SKIP on a *different*, solve-rate-dependent
subset, the two arms are then not proportion-matched even when configured identically. On
a teacher-sparse dataset the control quietly becomes a mostly-SKIP arm, which reads as a
win for the criterion router for reasons that have nothing to do with the criterion --
exactly the confound the control exists to prevent.

:class:`MatchedPermutationControl` sidesteps configuration entirely: it replays the
criterion router's own realised decisions, shuffled across units. Proportions match by
construction, for any dataset, with no probability to mis-specify.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Iterable, Sequence

from .base import Router, RoutingContext, RoutingDecision

__all__ = ["measure_proportions", "MatchedPermutationControl"]


def measure_proportions(
    router: Router, contexts: Iterable[RoutingContext]
) -> dict[str, float]:
    """Realised mode proportions of ``router`` over ``contexts``.

    Uses :meth:`RoutingDecision.argmax`, so a soft decision is attributed to its dominant
    mode. This is what a downstream arm actually experiences.

    Args:
        router: Any object satisfying the :class:`~selfevo.routing.base.Router` Protocol.
        contexts: The contexts to route. Consumed once.

    Returns:
        ``{mode: proportion}`` summing to 1. Empty dict for an empty stream -- callers
        must not treat that as "uniform".
    """
    counts: Counter[str] = Counter()
    for ctx in contexts:
        counts[router.route(ctx).argmax()] += 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {m: c / total for m, c in counts.items()}


class MatchedPermutationControl:
    """Replays a router's realised decisions in shuffled order. The falsification control.

    Proportions are matched **exactly**, not in expectation, because the same multiset of
    decisions is reused -- only the assignment of decision to unit is destroyed. That is
    precisely the thing under test: whether the criterion's *choice of which unit gets
    which mode* carries information, as opposed to the mode mixture alone.

    If a criterion router does not beat this, its gain came from mixing modes rather than
    from choosing between them.

    Args:
        decisions: The criterion router's realised decisions, in any order.
        seed: Shuffle seed. Uses a private ``random.Random`` so the control neither
            perturbs nor is perturbed by sampling elsewhere.

    Raises:
        ValueError: If ``decisions`` is empty -- a control with nothing to replay would
            silently become a no-op arm.
    """

    def __init__(self, decisions: Sequence[RoutingDecision], seed: int = 0) -> None:
        if not decisions:
            raise ValueError("decisions must be non-empty to form a matched control")
        self._pool = list(decisions)
        self._rng = random.Random(seed)
        self._rng.shuffle(self._pool)
        self._i = 0

    @classmethod
    def from_router(
        cls,
        router: Router,
        contexts: Sequence[RoutingContext],
        seed: int = 0,
    ) -> "MatchedPermutationControl":
        """Build a control by measuring ``router`` on ``contexts`` first.

        Args:
            router: The criterion router to match.
            contexts: Contexts to measure over; the control is then valid for a run of the
                same length.
            seed: Shuffle seed.

        Returns:
            A control whose realised proportions equal ``router``'s on these contexts.
        """
        return cls([router.route(c) for c in contexts], seed=seed)

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        """Return the next replayed decision, ignoring ``ctx``.

        Ignoring the context is the point: the decision is drawn from the criterion
        router's own output distribution but is deliberately uncorrelated with the unit.

        A teacher-requiring decision landing on a teacherless unit degrades to SKIP, since
        the signal layer could not honour it. That degradation is reported by
        :meth:`realised_proportions`, so a shift away from the intended match is visible
        rather than silent.

        Raises:
            RuntimeError: If called more times than there were decisions. Wrapping around
                would silently correlate the control with the unit ordering.
        """
        if self._i >= len(self._pool):
            raise RuntimeError(
                f"control exhausted after {len(self._pool)} decisions; build it over the "
                "same number of contexts as the run it controls"
            )
        d = self._pool[self._i]
        self._i += 1
        from .base import known_modes  # local import keeps module import order simple

        mode = d.argmax()
        if known_modes()[mode] and not ctx.has_teacher:
            return RoutingDecision({"skip": 1.0}, reason=f"control {mode}, no teacher")
        return d

    def realised_proportions(self) -> dict[str, float]:
        """Proportions of the decisions actually served so far.

        Compare against :func:`measure_proportions` of the criterion router: any gap is
        SKIP-migration, and it must be reported alongside a routing result rather than
        assumed to be zero.
        """
        counts: Counter[str] = Counter(d.argmax() for d in self._pool[: self._i])
        total = sum(counts.values())
        if total == 0:
            return {}
        return {m: c / total for m, c in counts.items()}
