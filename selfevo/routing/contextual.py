"""A contextual controller: choose the mode from features, not from one scalar.

``BanditRouter`` is deliberately context-free -- per-mode value estimates and nothing else.
That is the right first thing to build, and it is also the reason a learned router has not
yet been worth more than a threshold: a policy that sees only what a hand-written rule sees
cannot beat the best hand-written rule by more than noise.

This is the contextual version. It runs LinUCB over the features in
:mod:`selfevo.observability`, so the decision can depend on *why* a group looks the way it
does -- whether an unsolved group was out of token budget, whether an unanimous group was
also unanimous in its reasoning -- distinctions a solve-rate threshold cannot express.

LinUCB rather than a neural policy on purpose. The reward channel here is confounded and
low-signal (one scalar per batch, produced by every decision in it), and a model with more
capacity would mostly fit that noise. Ridge regression with an explicit confidence term
degrades to the prior when the evidence is thin, which is the behaviour this setting needs.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from selfevo.observability import FEATURE_NAMES
from selfevo.routing.base import (
    RoutingContext,
    RoutingDecision,
    TrainingMode,
    known_modes,
)
from selfevo.routing.feedback import DecisionOutcome

__all__ = ["ContextualBanditRouter", "MissingFeatures"]


class MissingFeatures(KeyError):
    """Raised when a context lacks a feature the policy was configured to use."""


@dataclass
class ContextualBanditRouter:
    """LinUCB over observability features, one ridge model per mode.

    Args:
        modes: Modes to choose between. Teacher-requiring modes are dropped for a context
            with no target, the same guard every other router in this package applies.
        feature_names: Features read from ``ctx.extra``, in this order. The vector is
            appended with a constant 1.0 so each arm has an intercept.
        alpha: Exploration weight on the confidence term. 0.0 is greedy.
        ridge: Ridge parameter; also the prior strength when an arm has no data.
        require_features: If True (default) a context missing any named feature raises. The
            alternative -- substituting zeros -- would silently turn this into the
            context-free bandit while still reporting itself as contextual, which is the
            failure this project keeps finding in other guises.
        pending_cap: How many un-observed decisions to remember. Bounded because a unit that
            is never observed would otherwise leak; evictions are counted, not silent.

    Attributes:
        evicted: Pending decisions dropped because the cache was full.
        updates: Outcomes actually credited to an arm.
        rejected: Outcomes dropped because the update they implied was not finite.

    Note:
        ``alpha = 0`` is greedy, and greedy here does not merely explore less -- it does not
        explore at all. Every arm starts at ``theta = 0``, so with no confidence term all
        arms tie forever on any context the router has not been forced away from, the
        tie-break picks the first mode by name, and that mode is the only one ever observed.
        :class:`~selfevo.routing.feedback.BanditRouter` rejects ``explore_prob = 0`` for
        exactly this reason; ``alpha = 0`` is kept legal only because a no-exploration
        control arm is worth being able to run deliberately.

    Raises:
        ValueError: If ``modes`` is empty, names an unregistered mode, or the numeric
            parameters are out of range.
    """

    modes: tuple[str, ...] = (TrainingMode.RL, TrainingMode.SFT, TrainingMode.SKIP)
    feature_names: tuple[str, ...] = FEATURE_NAMES
    alpha: float = 1.0
    ridge: float = 1.0
    require_features: bool = True
    pending_cap: int = 4096

    _A: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _b: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _pending: OrderedDict = field(default_factory=OrderedDict, init=False, repr=False)
    evicted: int = field(default=0, init=False)
    updates: int = field(default=0, init=False)
    rejected: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.modes:
            raise ValueError("modes must be non-empty")
        for m in self.modes:
            if m not in known_modes():
                raise ValueError(f"unknown mode {m!r}; register it first")
        if not self.feature_names:
            raise ValueError("feature_names must be non-empty")
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        if self.ridge <= 0:
            raise ValueError(f"ridge must be > 0, got {self.ridge}")
        if self.pending_cap < 1:
            raise ValueError(f"pending_cap must be >= 1, got {self.pending_cap}")
        d = len(self.feature_names) + 1
        for m in self.modes:
            self._A[m] = np.eye(d) * self.ridge
            self._b[m] = np.zeros(d)

    # ------------------------------------------------------------------ features ----

    def _vector(self, ctx: RoutingContext) -> np.ndarray:
        """Feature vector for ``ctx``, with an intercept appended."""
        vals = []
        for name in self.feature_names:
            if name not in ctx.extra:
                if self.require_features:
                    raise MissingFeatures(
                        f"context is missing feature {name!r}; present: "
                        f"{sorted(ctx.extra)}. Populate RoutingContext.extra from "
                        f"selfevo.observability.group_features, or construct this router "
                        f"with require_features=False to accept a zero in its place."
                    )
                vals.append(0.0)
            else:
                v = float(ctx.extra[name])
                # A non-finite feature would poison this arm's ridge model permanently.
                vals.append(v if np.isfinite(v) else 0.0)
        vals.append(1.0)
        return np.asarray(vals, dtype=np.float64)

    def _usable(self, ctx: RoutingContext) -> list[str]:
        """Modes selectable for ``ctx``: teacher-requiring ones need a target."""
        return [m for m in self.modes if ctx.has_target or not known_modes()[m]]

    # --------------------------------------------------------------------- acting ----

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        """Pick the mode with the highest upper confidence bound.

        Args:
            ctx: The unit to route. ``ctx.extra`` must carry ``feature_names``.

        Returns:
            A decision naming one mode. Ties break by mode name -- the argmax keeps the
            first strict improvement over modes visited in sorted order, which is the same
            rule :meth:`RoutingDecision.argmax` and ``BanditRouter`` use -- so a run is
            reproducible.

        Raises:
            MissingFeatures: If ``require_features`` and ``ctx.extra`` lacks a named
                feature. Checked before the no-usable-mode short-circuit below, because the
                requirement is a statement about the CONTEXT: a run whose features never
                arrived must not be able to hide behind its target-free units.
        """
        x = self._vector(ctx)
        usable = self._usable(ctx)
        if not usable:
            return RoutingDecision(
                {TrainingMode.SKIP: 1.0}, reason="contextual: no mode has a target"
            )
        best, best_score = None, -np.inf
        for m in sorted(usable):
            A_inv = np.linalg.inv(self._A[m])
            theta = A_inv @ self._b[m]
            score = float(theta @ x + self.alpha * np.sqrt(max(x @ A_inv @ x, 0.0)))
            if score > best_score:
                best, best_score = m, score
        assert best is not None
        if ctx.unit_id is not None:
            # Drop any earlier decision for this unit FIRST: re-routing a unit already held
            # is a refresh, and evicting a bystander to make room for a key the cache
            # already has would lose a live decision and overcount ``evicted`` as well.
            self._pending.pop(ctx.unit_id, None)
            if len(self._pending) >= self.pending_cap:
                self._pending.popitem(last=False)
                self.evicted += 1
            self._pending[ctx.unit_id] = (best, x)
        return RoutingDecision({best: 1.0}, reason=f"contextual: ucb {best_score:.4f}")

    # -------------------------------------------------------------------- learning ----

    def observe(self, outcomes: Mapping[str, DecisionOutcome]) -> None:
        """Update the ridge model of each arm that was actually applied.

        An outcome is only credited when the remembered decision for that unit names the
        same mode. A mismatch means the caller applied something other than what was routed
        -- crediting the wrong arm would corrupt the model in a way no metric would show.

        An update that is not finite is dropped and counted in ``rejected`` rather than
        applied. :class:`DecisionOutcome` already rejects a non-finite ``value``, but
        ``value / cost`` and the outer product can still overflow, and one non-finite entry
        makes this arm's ``theta`` NaN forever -- a NaN never wins an argmax, so the arm
        would become permanently unselectable with nothing in the log to say so.

        Args:
            outcomes: ``{unit_id: DecisionOutcome}`` for one batch.
        """
        for unit_id, out in outcomes.items():
            remembered = self._pending.pop(unit_id, None)
            if remembered is None:
                continue
            mode, x = remembered
            if out.mode != mode or mode not in self._A:
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                delta_A = np.outer(x, x)
                delta_b = (out.value / out.cost) * x
            if not (np.isfinite(delta_A).all() and np.isfinite(delta_b).all()):
                self.rejected += 1
                continue
            self._A[mode] += delta_A
            self._b[mode] += delta_b
            self.updates += 1

    # ----------------------------------------------------------------- inspection ----

    def weights(self, mode: str) -> dict[str, float]:
        """Fitted coefficients for one arm, by feature name plus ``intercept``.

        Exposed so a run can be audited for WHICH feature drove a decision. A controller
        whose learned weights are never looked at cannot be distinguished from a random one.
        """
        if mode not in self._A:
            raise ValueError(f"{mode!r} is not one of this router's modes: {self.modes}")
        theta = np.linalg.inv(self._A[mode]) @ self._b[mode]
        return dict(zip((*self.feature_names, "intercept"), (float(t) for t in theta)))
