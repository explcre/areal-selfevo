"""Routers: fixed, criterion-based, and the controls needed to falsify them.

The controls are not optional extras. A routing gain is only attributable to the criterion
if it beats :class:`RandomRouter` matched to the *same mode proportions* -- otherwise the
gain may come from mixing modes at all, which any random assignment would also deliver.
:class:`InvertedRouter` is the sharper test: if the criterion carries signal, deliberately
routing against it should do worse than random.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Mapping

from .base import (
    RoutingContext,
    RoutingDecision,
    TrainingMode,
    known_modes,
)
from .criteria import SilenceSide, rl_informativeness, silence_side

__all__ = [
    "StaticRouter",
    "SolveRateRouter",
    "RandomRouter",
    "InvertedRouter",
]


@dataclass(frozen=True)
class StaticRouter:
    """Always returns the same mode weights. The fixed-mode baseline.

    Args:
        weights: Mode weights to return for every unit, e.g. ``{TrainingMode.RL: 1.0}``.

    Raises:
        ValueError: If ``weights`` is invalid (checked by :class:`RoutingDecision`).
    """

    weights: Mapping[str, float] = field(
        default_factory=lambda: {TrainingMode.RL: 1.0}
    )

    def __post_init__(self) -> None:
        RoutingDecision(self.weights)  # validate eagerly, not at first route()

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        """Return the fixed weights, minus any mode the context cannot supply.

        Honours the invariant stated in :mod:`selfevo.routing.base`: a router must not
        select a teacher-requiring mode when no teacher is available. A fixed-mode
        baseline is not exempt -- an all-SFT arm run on teacherless data would otherwise
        emit decisions the signal layer cannot fulfil.
        """
        if not ctx.has_teacher:
            usable = {m: w for m, w in self.weights.items() if not known_modes()[m]}
            if not usable:
                return RoutingDecision(
                    {TrainingMode.SKIP: 1.0}, reason="static, no teacher available"
                )
            if len(usable) != len(self.weights):
                return RoutingDecision(usable, reason="static, teacher modes dropped")
        return RoutingDecision(dict(self.weights), reason="static")


@dataclass(frozen=True)
class SolveRateRouter:
    """Route on RL informativeness and which side of silence a unit falls on.

    The rule, and the reasoning for each branch:

    - **informative** (``I_RL >= threshold``): the group disagrees, so the advantage is
      non-zero. Use RL.
    - **unsolved** (``I_RL < threshold`` and ``p < 0.5``): every sample failed, so every
      advantage is zero and RL cannot move. Only an external target helps -- route to the
      teacher mode if one exists, else ``SKIP``. Forcing RL here spends compute for a
      gradient that is identically zero.
    - **solved** (``I_RL < threshold`` and ``p >= 0.5``): every sample succeeded. RL is
      silent because there is nothing left to learn. ``SKIP`` rather than SFT: adding a
      supervised gradient to an already-correct policy only sharpens it, burning entropy
      for no accuracy. This asymmetry is the whole point of splitting the two silent sides.

    Args:
        threshold: Informativeness below which a unit counts as silent, in [0, 1].
        teacher_mode: Mode to use for unsolved units when a teacher is available.
        blend: If True, return a soft mixture of RL and the teacher mode weighted by
            informativeness, instead of a hard choice. Off by default: a hard rule is
            easier to attribute in an ablation, and the soft version should have to earn
            its extra degree of freedom.

    Raises:
        ValueError: If ``threshold`` is outside [0, 1] or ``teacher_mode`` is unregistered
            or does not actually require a teacher.
    """

    threshold: float = 0.1
    teacher_mode: str = TrainingMode.SFT
    blend: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        modes = known_modes()
        if self.teacher_mode not in modes:
            raise ValueError(f"unknown teacher_mode {self.teacher_mode!r}")
        if not modes[self.teacher_mode]:
            raise ValueError(
                f"teacher_mode {self.teacher_mode!r} does not require a teacher; "
                "routing unsolved units to it would not supply the missing target"
            )

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        """Choose weights from the unit's solve rate and teacher availability."""
        info = rl_informativeness(ctx.solve_rate, ctx.group_size)
        side = silence_side(ctx.solve_rate, ctx.group_size, self.threshold)

        if self.blend and ctx.has_teacher:
            # Soft mixture: RL where the group disagrees, teacher where it does not.
            # Only meaningful on the unsolved side -- blending a teacher into an already
            # solved unit is the sharpening failure described in the class docstring.
            if side is not SilenceSide.SOLVED:
                return RoutingDecision(
                    {TrainingMode.RL: info, self.teacher_mode: 1.0 - info},
                    reason=f"blend I_RL={info:.3f}",
                )

        if side is SilenceSide.INFORMATIVE:
            return RoutingDecision({TrainingMode.RL: 1.0}, reason=f"I_RL={info:.3f}")
        if side is SilenceSide.UNSOLVED:
            if ctx.has_teacher:
                return RoutingDecision(
                    {self.teacher_mode: 1.0}, reason=f"unsolved I_RL={info:.3f}"
                )
            return RoutingDecision(
                {TrainingMode.SKIP: 1.0}, reason=f"unsolved, no teacher I_RL={info:.3f}"
            )
        return RoutingDecision({TrainingMode.SKIP: 1.0}, reason=f"solved I_RL={info:.3f}")


@dataclass
class RandomRouter:
    """Assign modes at random from fixed proportions. **The mandatory control.**

    Matches a criterion router's mode *proportions* while discarding which unit gets which
    mode. If a criterion router does not beat this, its gain came from mixing modes, not
    from choosing between them -- so this must be run at the proportions the criterion
    router actually produced, measured, not assumed.

    Args:
        proportions: ``{mode: probability}``; normalised internally.
        seed: Seed for reproducibility. Uses a private ``random.Random`` rather than the
            global RNG so routing cannot perturb, or be perturbed by, sampling elsewhere.

    Raises:
        ValueError: If ``proportions`` is invalid.
    """

    proportions: Mapping[str, float] = field(
        default_factory=lambda: {TrainingMode.RL: 1.0}
    )
    seed: int = 0

    def __post_init__(self) -> None:
        RoutingDecision(self.proportions)  # validate names and weights
        self._rng = random.Random(self.seed)
        total = sum(self.proportions.values())
        self._modes = sorted(self.proportions)  # sorted => deterministic given the seed
        self._cum: list[float] = []
        acc = 0.0
        for m in self._modes:
            acc += self.proportions[m] / total
            self._cum.append(acc)

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        """Sample a mode from the fixed proportions, ignoring ``ctx``."""
        u = self._rng.random()
        for mode, c in zip(self._modes, self._cum):
            if u <= c:
                chosen = mode
                break
        else:  # pragma: no cover - only reachable on float round-off at the top of range
            chosen = self._modes[-1]
        # has_TARGET, not has_teacher. A teacher-requiring mode is honourable whenever ANY
        # target exists, and for a group with solve_rate > 0 the group's own correct sample
        # is one -- that self-target is the method's central claim. Checking has_teacher
        # instead made this control fall back to SKIP for every SFT draw in every run here
        # (no run has an external teacher), so the "matched" control could not emit the mode
        # it was supposed to match, and its mix collapsed to rl/skip. Measured 2026-08-31.
        if known_modes()[chosen] and not ctx.has_target:
            # Never emit a decision the signal layer cannot honour; fall back to SKIP so
            # the control stays comparable rather than silently erroring mid-batch.
            return RoutingDecision(
                {TrainingMode.SKIP: 1.0}, reason=f"random {chosen} but no teacher"
            )
        return RoutingDecision({chosen: 1.0}, reason="random")


@dataclass(frozen=True)
class InvertedRouter:
    """Route against the criterion. The sharper falsification test.

    Sends informative units to the teacher mode and silent ones to RL -- exactly backwards.
    If :class:`SolveRateRouter` carries real signal, this should perform *worse than*
    :class:`RandomRouter`. If instead it matches random, the criterion is not doing work.

    Args:
        threshold: Same informativeness threshold as the router being tested.
        teacher_mode: Mode to misroute informative units into.
    """

    threshold: float = 0.1
    teacher_mode: str = TrainingMode.SFT

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        """Return the opposite of what :class:`SolveRateRouter` would choose."""
        side = silence_side(ctx.solve_rate, ctx.group_size, self.threshold)
        if side is SilenceSide.INFORMATIVE:
            if ctx.has_teacher:
                return RoutingDecision({self.teacher_mode: 1.0}, reason="inverted")
            return RoutingDecision({TrainingMode.SKIP: 1.0}, reason="inverted, no teacher")
        return RoutingDecision({TrainingMode.RL: 1.0}, reason="inverted")
