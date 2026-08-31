"""Routing that treats the harness as a destination for a trajectory, not just the model.

Co-Harness (2607.22688) sends successful trajectories to the model and failed ones to the
harness. That rule is reproduced here exactly (``partition=True``) and relaxed by default,
because the two consumers read different things from a trajectory and a partition discards
one of the readings.

The thresholds are not arbitrary. For a group of ``G`` samples with per-sample success
probability ``p``, the probability that GRPO produces a non-zero advantage anywhere in the
group is ``I_RL(p, G) = 1 - p**G - (1 - p)**G``, which is zero at both ``p = 0`` and
``p = 1``. So the all-solved and all-failed groups are precisely the groups RL cannot learn
from, and they are the groups this router redirects:

    all-solved (p = 1)   RL is dead, but a correct sample exists -> SFT on it, no teacher
                         needed. Optionally also a harness VALIDATE case.
    all-failed (p = 0)   RL is dead and no self-target exists. Without an external teacher
                         the model arm has nothing, so the harness is the only consumer
                         that can use this unit at all -> PROPOSE.
    mixed (0 < p < 1)    RL has signal; leave it alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from selfevo.routing.base import (
    HarnessAction,
    RoutingContext,
    RoutingDecision,
    TrainingMode,
    known_modes,
)

__all__ = ["CoHarnessRouter"]


def _target_source(ctx: RoutingContext) -> str:
    """Name where an SFT target for ``ctx`` would come from, for an honest decision reason.

    Args:
        ctx: The unit being routed.

    Returns:
        A phrase naming the origin of the target. ``has_self_target`` is checked first
        because a group that drew a correct sample needs no teacher even when one exists,
        and a reason that credits the teacher there would misreport the run.
    """
    if ctx.has_self_target:
        return "target from its own correct sample"
    if ctx.has_teacher:
        return "target from an external teacher"
    return "the mode needs no target"


@dataclass(frozen=True)
class CoHarnessRouter:
    """Route a unit to the model, the harness, or both.

    Args:
        solved_threshold: ``solve_rate >= solved_threshold`` counts as solved. Defaults to
            1.0, the only value at which the RL gradient is *provably* zero; lowering it
            trades that guarantee for reach.
        failed_threshold: ``solve_rate <= failed_threshold`` counts as failed. Defaults to
            0.0, for the same reason.
        partition: If True, a unit feeds either the model or the harness but never both,
            reproducing Co-Harness. If False (default), a failed unit can carry a harness
            action *and* a training mode, which is the point of keeping the two axes
            orthogonal.
        validate_on_success: If True, solved units are also offered to the harness as
            regression cases, so a harness edit that breaks something already working is
            detectable. Ignored when ``partition`` is True.
        self_target_mode: Mode used for a solved unit whose target comes from its own
            correct sample. SFT by design -- hard distillation is the same estimator.
        propose_threshold: Units with ``solve_rate <= propose_threshold`` that are not
            solved contribute a PROPOSE action, *whatever mode they were given*. ``None``
            (default) means ``failed_threshold``, so only wholly-failed units propose and
            behaviour is unchanged. Raising it lets a MIXED unit feed the gradient and the
            harness at once: such a unit still contains failed samples, and those failures
            carry harness-relevant information that the RL update does not consume. This
            is the general form of "one trajectory, two consumers"; at the default it is
            switched off, at ``solved_threshold`` every mixed unit proposes, and under
            ``partition`` it is ignored. It never reaches a *solved* unit -- a unit with
            no failures has no failure mode to report -- which is why it is bounded above
            by ``solved_threshold`` rather than by 1.

    Raises:
        ValueError: If the thresholds cross, any lies outside [0, 1], or
            ``self_target_mode`` is not a registered mode. ``propose_threshold`` outside
            ``[failed_threshold, solved_threshold]`` is also rejected: below
            ``failed_threshold`` it would silence the harness on units the rule has
            already called failures, and above ``solved_threshold`` it names solve rates
            the branch never sees, so two configurations that read differently would
            behave identically and no log would tell them apart.
    """

    solved_threshold: float = 1.0
    failed_threshold: float = 0.0
    partition: bool = False
    validate_on_success: bool = True
    self_target_mode: str = TrainingMode.SFT
    propose_threshold: float | None = None

    @property
    def effective_propose_threshold(self) -> float:
        """``propose_threshold``, defaulting to ``failed_threshold``.

        Resolved in one place so the default cannot drift between the validation in
        ``__post_init__`` and the decision in :meth:`route`.
        """
        return self.failed_threshold if self.propose_threshold is None else self.propose_threshold

    def __post_init__(self) -> None:
        for name, v in (
            ("solved_threshold", self.solved_threshold),
            ("failed_threshold", self.failed_threshold),
        ):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if self.failed_threshold >= self.solved_threshold:
            raise ValueError(
                f"failed_threshold ({self.failed_threshold}) must be below "
                f"solved_threshold ({self.solved_threshold}); otherwise a unit would be "
                "both solved and failed and the rule would depend on check order"
            )
        if self.propose_threshold is not None:
            if not 0.0 <= self.propose_threshold <= 1.0:
                raise ValueError(
                    f"propose_threshold must be in [0, 1], got {self.propose_threshold}"
                )
            if self.propose_threshold < self.failed_threshold:
                raise ValueError(
                    f"propose_threshold ({self.propose_threshold}) must be at least "
                    f"failed_threshold ({self.failed_threshold}); a lower value would "
                    "silence the harness on units the rule already calls failures"
                )
            if self.propose_threshold > self.solved_threshold:
                raise ValueError(
                    f"propose_threshold ({self.propose_threshold}) must be at most "
                    f"solved_threshold ({self.solved_threshold}); a solved unit is offered "
                    "to the harness as a VALIDATE case and has no failure to propose, so "
                    "the excess range changes no decision and only misreports the config"
                )
        if self.self_target_mode not in known_modes():
            raise ValueError(f"unknown mode {self.self_target_mode!r}")

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        """Choose a training mode and a harness action for ``ctx``.

        Args:
            ctx: The unit to route.

        Returns:
            A decision whose ``harness`` field is ``NONE`` whenever
            ``ctx.can_evolve_harness`` is False, so a run without a harness arm behaves
            exactly as it would without this router.
        """
        solved = ctx.solve_rate >= self.solved_threshold
        failed = ctx.solve_rate <= self.failed_threshold

        if solved:
            mode, why = self._solved_mode(ctx)
            harness = (
                HarnessAction.VALIDATE
                if self.validate_on_success and not self.partition
                else HarnessAction.NONE
            )
        elif failed:
            mode, why = self._failed_mode(ctx)
            harness = HarnessAction.PROPOSE
            if self.partition:
                # Co-Harness proper: a failed unit belongs to the harness alone.
                mode, why = TrainingMode.SKIP, "failed; partitioned to harness"
        else:
            mode, why = TrainingMode.RL, "mixed group; RL has signal"
            # A mixed unit contains failed samples, and their failure mode is information the
            # RL update does not consume. Off at the default threshold; when enabled this is
            # the general case of one trajectory feeding both consumers.
            if not self.partition and ctx.solve_rate <= self.effective_propose_threshold:
                harness = HarnessAction.PROPOSE
                why = f"{why}; failed samples also proposed to the harness"
            else:
                harness = HarnessAction.NONE

        if not ctx.can_evolve_harness and harness is not HarnessAction.NONE:
            harness = HarnessAction.NONE
            why = f"{why}; harness action dropped (no harness arm)"

        if harness is not HarnessAction.NONE and mode == TrainingMode.SKIP:
            why = f"{why}; unit is consumed by the harness only"

        return RoutingDecision({mode: 1.0}, reason=why, harness=harness)

    def _solved_mode(self, ctx: RoutingContext) -> tuple[str, str]:
        """Mode for a unit RL cannot learn from because every sample was correct."""
        if known_modes()[self.self_target_mode] and not ctx.has_target:
            # has_self_target is False here only at TOKEN granularity, where a group mean
            # does not describe a sibling sample.
            return TrainingMode.SKIP, "solved; RL is dead and no target is available"
        return self.self_target_mode, f"solved; RL is dead, {_target_source(ctx)}"

    def _failed_mode(self, ctx: RoutingContext) -> tuple[str, str]:
        """Mode for a unit RL cannot learn from because every sample was wrong.

        Gated on ``has_target``, not ``has_teacher``: at the default ``failed_threshold``
        the two agree, because ``solve_rate == 0`` admits no self-target. Above it a
        "failed" group can still contain a correct sample, and skipping that group would
        discard a target the rollout already paid for.
        """
        if ctx.has_target:
            return TrainingMode.SFT, f"failed; RL is dead, {_target_source(ctx)}"
        return TrainingMode.SKIP, "failed; RL is dead and no target is available"
