"""Credit assignment for routing decisions: the channel a learned router needs.

The audit's finding was exact: ``Router`` lets a learned policy *act* but not *learn* --
there is no observe/update path, so "a future learned router implements the same Protocol;
nothing else changes" was only half true. This adds the missing half.

**The confound this API is shaped around.** A routing decision's effect is not directly
observable. What we see after an update is one scalar per *batch*, produced by every
decision in that batch jointly, plus the optimiser, plus the data. Attributing a reward
delta to one unit's mode choice is a credit-assignment problem, not a measurement -- and
getting it wrong produces a learned router that confidently optimises noise.

Two consequences are baked in rather than left to the caller:

* :class:`DecisionOutcome` carries ``batch_id``, and :class:`BanditRouter` refuses to
  update from a batch in which every unit got the same mode. With no within-batch variation
  the mode is perfectly confounded with the batch, so any apparent difference between modes
  is a difference between batches.
* Learning requires exploration. A router that always picks its current argmax observes
  outcomes only for that mode and cannot discover it was wrong. ``explore_prob`` defaults
  to a non-zero value for that reason, and setting it to 0 is rejected.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

from .base import (
    Router,
    RoutingContext,
    RoutingDecision,
    TrainingMode,
    known_modes,
)

__all__ = [
    "DecisionOutcome",
    "LearningRouter",
    "ConfoundedUpdate",
    "BanditRouter",
]


class ConfoundedUpdate(RuntimeError):
    """Raised when an update could not distinguish mode effects from batch effects."""


@dataclass(frozen=True)
class DecisionOutcome:
    """What happened after a routing decision.

    Args:
        mode: The mode that was actually applied (a decision's ``argmax``).
        value: The observed scalar to credit -- a reward delta, an entropy delta, a
            solve-rate change. Sign convention is the caller's, but it must be consistent:
            higher is better.
        batch_id: Identifier of the update this decision took part in. Required, because
            two decisions from different batches are not comparable -- everything else
            about the update differed too.
        unit_id: Optional unit identifier, for tracing.
        cost: Compute actually spent, in whatever unit the caller uses. Lets a router
            optimise value per unit cost rather than value alone, which matters because
            SKIP is nearly free and would otherwise never look attractive.
    """

    mode: str
    value: float
    batch_id: str
    unit_id: str | None = None
    cost: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in known_modes():
            raise ValueError(f"unknown mode {self.mode!r}")
        if not self.batch_id:
            raise ValueError(
                "batch_id is required: outcomes from different batches are not comparable"
            )
        if self.cost <= 0:
            raise ValueError(f"cost must be positive, got {self.cost}")
        if not math.isfinite(self.value):
            raise ValueError(f"value must be finite, got {self.value}")


@runtime_checkable
class LearningRouter(Protocol):
    """A :class:`Router` that can also be updated from observed outcomes."""

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        """Return mode weights for ``ctx``."""
        ...

    def observe(self, outcomes: Mapping[str, DecisionOutcome]) -> None:
        """Update from one batch's outcomes, keyed by unit id."""
        ...


@dataclass
class BanditRouter:
    """Per-mode value estimates with forced exploration. The minimal router that learns.

    Deliberately a *contextual-free* bandit over modes rather than anything richer: with a
    confounded, low-signal reward channel, a model with more capacity would mostly fit
    noise. This is the simplest thing that can be shown to beat a fixed router, which is
    the bar the design sets before a learned router is worth building at all.

    Value is tracked as mean ``value / cost`` per mode, so SKIP -- which is nearly free --
    competes fairly instead of being dominated by any mode that ever does anything.

    Args:
        explore_prob: Probability of ignoring the current estimates and sampling a mode
            uniformly. Must be > 0: without exploration the router only ever observes the
            mode it already prefers, and cannot discover that preference is wrong.
        modes: Modes to choose among. Defaults to RL, SFT and SKIP; DISTILL is excluded by
            default only because its transport is not built.
        seed: RNG seed. Private ``random.Random``, so exploration neither perturbs nor is
            perturbed by sampling elsewhere.
        min_observations: Observations of a mode before its estimate is trusted; below
            this the router keeps exploring that mode.

    Raises:
        ValueError: If ``explore_prob`` is not in (0, 1], if ``modes`` is empty or contains
            an unregistered mode, or if ``min_observations`` < 1.
    """

    explore_prob: float = 0.1
    modes: tuple[str, ...] = (TrainingMode.RL, TrainingMode.SFT, TrainingMode.SKIP)
    seed: int = 0
    min_observations: int = 5
    _sum: dict[str, float] = field(default_factory=lambda: defaultdict(float), init=False)
    _n: dict[str, int] = field(default_factory=lambda: defaultdict(int), init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.explore_prob <= 1.0:
            raise ValueError(
                f"explore_prob must be in (0, 1], got {self.explore_prob}: a router with "
                "no exploration observes only the mode it already prefers"
            )
        if not self.modes:
            raise ValueError("modes must be non-empty")
        for m in self.modes:
            if m not in known_modes():
                raise ValueError(f"unknown mode {m!r}")
        if self.min_observations < 1:
            raise ValueError(f"min_observations must be >= 1, got {self.min_observations}")
        self._rng = random.Random(self.seed)

    def _available(self, ctx: RoutingContext) -> list[str]:
        """Modes usable for this unit, honouring the teacher invariant."""
        return [m for m in self.modes if ctx.has_teacher or not known_modes()[m]]

    def value_estimates(self) -> dict[str, float]:
        """Mean value-per-cost by mode. Modes never observed are absent, not zero."""
        return {m: self._sum[m] / self._n[m] for m in self.modes if self._n[m] > 0}

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        """Pick a mode: explore with probability ``explore_prob``, else exploit."""
        usable = self._available(ctx)
        if not usable:
            return RoutingDecision({TrainingMode.SKIP: 1.0}, reason="bandit: no usable mode")

        under_observed = [m for m in usable if self._n[m] < self.min_observations]
        if under_observed or self._rng.random() < self.explore_prob:
            pool = under_observed or usable
            return RoutingDecision(
                {self._rng.choice(pool): 1.0},
                reason="bandit explore" if not under_observed else "bandit warmup",
            )

        est = {m: self._sum[m] / self._n[m] for m in usable if self._n[m] > 0}
        if not est:
            return RoutingDecision({self._rng.choice(usable): 1.0}, reason="bandit explore")
        # Ties resolve by name so a run is reproducible given the seed.
        best = max(sorted(est), key=lambda m: est[m])
        return RoutingDecision({best: 1.0}, reason=f"bandit exploit v={est[best]:.4f}")

    def observe(self, outcomes: Mapping[str, DecisionOutcome]) -> None:
        """Update estimates from one batch.

        Args:
            outcomes: ``{unit_id: DecisionOutcome}`` for a single update.

        Raises:
            ConfoundedUpdate: If the batch contains fewer than two distinct modes, or spans
                more than one ``batch_id``. In the first case the mode is perfectly
                confounded with the batch; in the second, outcomes that are not comparable
                would be pooled. Both would corrupt the estimates silently, which is worse
                than refusing.
        """
        if not outcomes:
            return
        batch_ids = {o.batch_id for o in outcomes.values()}
        if len(batch_ids) > 1:
            raise ConfoundedUpdate(
                f"outcomes span {len(batch_ids)} batches; call observe() once per batch, "
                "since outcomes from different updates are not comparable"
            )
        modes = {o.mode for o in outcomes.values()}
        if len(modes) < 2:
            raise ConfoundedUpdate(
                f"batch used only mode {modes.pop()!r}; with no within-batch variation the "
                "mode is perfectly confounded with the batch, so any difference between "
                "modes would be a difference between batches"
            )
        for o in outcomes.values():
            self._sum[o.mode] += o.value / o.cost
            self._n[o.mode] += 1


# A BanditRouter is a Router as well; assert it here so a Protocol drift is caught at
# import time rather than by a caller.
_: Router = BanditRouter()
