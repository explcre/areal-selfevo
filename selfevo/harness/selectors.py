"""Feature-driven dispatch rules for the harness axis, and the matched control for one.

``HarnessDispatcher`` takes a ``selector`` -- a rule that answers "which variant next?" --
and defaults to :func:`~selfevo.harness.dispatch.round_robin`, which ignores the batch
entirely. Round-robin is the right default (deterministic, always moves, visits every
member) but it is not a controller: it cannot be said to have ROUTED anything, because
nothing about the batch reaches it. This module holds the rules that do read the batch, and
the control that isolates *reading the batch* from *moving at all*.

**The axis these rules move, and why it is not ``step_limit``.**
``HarnessVariant.step_limit`` documents itself as "maximum agent steps". On a single-turn
math RLVR run there are no agent steps: the workflow issues exactly one completion request
per rollout, so a variant set that differed only in ``step_limit`` would be dispatched
between scaffolds that behave identically -- an arm that trains exactly like its own control
while logging switches, which is the failure ``HarnessDispatcher`` was built to refuse and
which it cannot detect here because the field it compares is one nothing reads.

The axis that IS real on this run is the GENERATION BUDGET. Truncation is non-termination:
a response that reaches the cap carries no answer, grades wrong, and this repo has measured
``n_truncated == n_no_box`` everywhere it looked. ``truncated_fraction`` is therefore both a
genuine routing feature and a quantity the budget directly controls, which is what makes a
budget ladder a harness variant set rather than a relabelling. So a variant on this path
carries its budget in ``settings[GENERATION_BUDGET_KEY]`` and :func:`budget_of` is the one
place that mapping is read.

**The one-observation-one-decision contract.** A selector here is consulted through TWO
entry points and they are not interchangeable. :meth:`LadderSelector.observe` is given the
batch and DECIDES; ``__call__`` is what ``HarnessDispatcher.apply`` invokes and only reports
the decision already taken. Splitting them is what makes the call site auditable: an
``observe`` is stamped with the epoch (the training step) it belongs to, a second observation
inside one epoch is REFUSED and counted rather than silently overwriting the first, and a
``__call__`` with no live decision raises instead of inventing one. Without that, a selector
driven per group instead of per batch would decide on one group's features and look
identical in the log.

A refusal is a first-class outcome, not an error: the treatment rule below deliberately
does nothing while truncation sits between its thresholds, and a run whose selector refused
on 90% of steps is a different run from one that was never consulted. Both are counted.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence

from selfevo.harness.base import HarnessVariant

__all__ = [
    "GENERATION_BUDGET_KEY",
    "budget_of",
    "ladder",
    "SelectorDecision",
    "LadderSelector",
    "TruncationStepLimitSelector",
    "RateMatchedControlSelector",
    "SELECTORS",
    "build_selector",
]

#: Key under ``HarnessVariant.settings`` holding the variant's generation budget, in tokens.
#: Named for the field the rollout ultimately sets (``gconfig.max_new_tokens``, which the
#: OpenAI-proxy path carries as ``max_completion_tokens``) so that a reader can follow the
#: value from the variant definition to the request without a translation table.
GENERATION_BUDGET_KEY = "max_new_tokens"


def budget_of(variant: HarnessVariant) -> int:
    """The generation budget a variant configures, in tokens.

    The single place the variant-to-behaviour mapping is read. It refuses rather than
    defaults, because a variant with no budget would dispatch to the SAME generation length
    as every other variant, producing two arms that are byte-identical while the log reports
    switches -- the exact silent no-op this axis exists to make impossible.

    Args:
        variant: The variant to read.

    Returns:
        The budget in tokens.

    Raises:
        ValueError: If the variant carries no budget, or one that is not a positive integer.
            A float is refused too: a budget is a token count, and a silently truncated 255.6
            would make two runs that quote the same config differ.
    """
    if GENERATION_BUDGET_KEY not in variant.settings:
        raise ValueError(
            f"variant {variant.name!r} has no {GENERATION_BUDGET_KEY!r} in its settings "
            f"({sorted(variant.settings)}), so dispatching to it would not change the "
            f"generation length and the arm would train exactly like its own control"
        )
    value = variant.settings[GENERATION_BUDGET_KEY]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"variant {variant.name!r} has {GENERATION_BUDGET_KEY}={value!r} of type "
            f"{type(value).__name__}; a generation budget must be an int number of tokens"
        )
    if value < 1:
        raise ValueError(
            f"variant {variant.name!r} has {GENERATION_BUDGET_KEY}={value}; a budget below "
            f"1 token cannot produce a response"
        )
    return value


def ladder(variants: Sequence[HarnessVariant]) -> tuple[HarnessVariant, ...]:
    """Order a variant set by generation budget, ascending.

    "One rung longer" and "one rung shorter" are only defined against an order, and the
    CONFIGURED order is not it: a config that happened to list ``[long, short, mid]`` would
    otherwise make "one rung longer" mean "shorter", and the arm would be reported as the
    opposite of what it did. Sorting here, once, keeps that off every caller.

    Args:
        variants: The configured set. Every member must carry a budget.

    Returns:
        The same variants ordered by ascending budget.

    Raises:
        ValueError: If fewer than two variants are given -- a ladder with one rung has no
            neighbours and every decision would be refused, which is a control arm and must
            be configured as one rather than arrived at by accident. Also if two variants
            share a budget, since a "move" between them would change nothing.
    """
    if len(variants) < 2:
        raise ValueError(
            f"a budget ladder needs at least 2 rungs, got {len(variants)}; a single-rung "
            f"ladder refuses every decision and is a control arm, not a selector arm"
        )
    budgets = [budget_of(v) for v in variants]
    dupes = sorted({b for b in budgets if budgets.count(b) > 1})
    if dupes:
        raise ValueError(
            f"variants share generation budget(s) {dupes}; a move between two rungs of the "
            f"same length is a switch the log reports and the rollout cannot show"
        )
    return tuple(v for _, v in sorted(zip(budgets, variants), key=lambda p: p[0]))


@dataclass(frozen=True)
class SelectorDecision:
    """What one observation decided, and why.

    Args:
        epoch: The epoch (training step) this decision belongs to. Carried so that a caller
            can assert the decision it is applying is the one it just asked for.
        move: ``+1`` one rung longer, ``-1`` one rung shorter, ``0`` refused.
        reason: Short human-readable justification, in the same spirit as
            ``DispatchRecord.reason``.
        observation: The batch feature the decision was taken on, kept so a step's decision
            can be re-derived from the log alone.
        blocked: True when a non-zero move was reduced to a refusal because the ladder had
            no rung in that direction. Distinct from an ordinary refusal: the rule WANTED to
            move and the configuration would not let it, which is a statement about the
            ladder rather than about the batch.
    """

    epoch: int
    move: int
    reason: str
    observation: float
    blocked: bool = False


class LadderSelector:
    """Base class: a dispatch rule over a budget ladder with an auditable decision epoch.

    Subclasses implement :meth:`_propose`, which sees the batch feature and returns a
    direction. Everything else here is the contract that makes WHERE the rule is called a
    checkable property rather than a comment: epoch stamping, refusal of a second
    observation inside one epoch, clamping at the ends of the ladder, and the counters a
    matched control is configured from.

    Args:
        name: Identifier used in logs and metric keys.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._decision: SelectorDecision | None = None
        self._last_epoch: int | None = None
        # Counters. Every one of these is emitted every step, because the two-pass matched
        # control is configured from the treatment's REALISED rates and a rate reconstructed
        # from a nominal setting is not a match, it is a hope.
        self.decisions = 0
        self.moves = 0
        self.refusals = 0
        self.blocked = 0
        self.up_moves = 0
        self.down_moves = 0
        self.repeat_observations = 0
        self.consumed = 0

    # -- the decision half -------------------------------------------------------------

    def observe(
        self,
        epoch: int,
        truncated_fraction: float,
        *,
        variants: Sequence[HarnessVariant],
        current: HarnessVariant,
    ) -> SelectorDecision:
        """Open an epoch, decide once, and return the decision.

        Args:
            epoch: Monotonically increasing epoch id -- the global training step. One
                decision per epoch is the whole contract.
            truncated_fraction: Batch-mean fraction of rollouts that reached the ACTIVE
                generation budget. Must be in ``[0, 1]``.
            variants: The configured variant set, used to clamp at the ends of the ladder.
            current: The active variant, i.e. where on the ladder the decision starts.

        Returns:
            The decision for this epoch. On a repeated or stale epoch, the decision already
            standing for it, unchanged.

        Raises:
            ValueError: If ``truncated_fraction`` is not a finite number in ``[0, 1]``.
                Accepting a NaN would send it through a comparison that is False against
                every threshold, which reads in the log as a deliberate refusal.
        """
        value = float(truncated_fraction)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"truncated_fraction must be a finite fraction in [0, 1], got "
                f"{truncated_fraction!r}; a NaN compares False against every threshold and "
                f"would be logged as a considered refusal"
            )
        if self._last_epoch is not None and epoch <= self._last_epoch:
            # Refused and counted, not raised: a second observation is a caller bug that
            # should be visible in the metrics of the run it damaged, and raising here would
            # kill a training job over a logging mistake.
            self.repeat_observations += 1
            assert self._decision is not None
            return self._decision

        rungs = ladder(variants)
        index = [v.name for v in rungs].index(current.name)
        move, reason = self._propose(value)
        decision_blocked = False
        if move > 0 and index == len(rungs) - 1:
            move, decision_blocked = 0, True
            reason = f"{reason}; blocked: {current.name!r} is the longest rung"
        elif move < 0 and index == 0:
            move, decision_blocked = 0, True
            reason = f"{reason}; blocked: {current.name!r} is the shortest rung"

        self._last_epoch = epoch
        self._decision = SelectorDecision(epoch, move, reason, value, decision_blocked)
        self.decisions += 1
        if move == 0:
            self.refusals += 1
            if decision_blocked:
                self.blocked += 1
        else:
            self.moves += 1
            if move > 0:
                self.up_moves += 1
            else:
                self.down_moves += 1
        return self._decision

    def _propose(self, truncated_fraction: float) -> tuple[int, str]:
        """Direction this rule wants, before the ladder ends are considered.

        Args:
            truncated_fraction: Batch-mean truncation under the active budget.

        Returns:
            ``(move, reason)`` with ``move`` in ``{-1, 0, +1}``.
        """
        raise NotImplementedError

    # -- the reporting half ------------------------------------------------------------

    def __call__(
        self, variants: Sequence[HarnessVariant], current: HarnessVariant
    ) -> HarnessVariant:
        """Report the standing decision as a variant. The ``HarnessDispatcher`` seam.

        Deliberately does NOT decide. ``HarnessDispatcher.apply`` calls this once per
        PROPOSE it acts on, and a rule that decided here would decide on whatever the
        dispatcher happened to ask for rather than on the batch, with no way to tell the two
        apart afterwards.

        Args:
            variants: The configured set.
            current: The active variant.

        Returns:
            The neighbouring variant the standing decision selected.

        Raises:
            RuntimeError: If no decision is standing, or the standing one refused. Either
                means the caller issued a PROPOSE the selector never asked for, and the
                dispatcher would otherwise be handed an arbitrary variant.
        """
        if self._decision is None:
            raise RuntimeError(
                f"selector {self.name!r} was asked to choose before any observation; "
                f"call observe() with the batch first, or the choice is not a routing "
                f"decision"
            )
        if self._decision.move == 0:
            raise RuntimeError(
                f"selector {self.name!r} refused to move at epoch {self._decision.epoch} "
                f"({self._decision.reason}), but a PROPOSE reached it; a caller must not "
                f"propose on a refusal"
            )
        rungs = ladder(variants)
        index = [v.name for v in rungs].index(current.name)
        self.consumed += 1
        return rungs[index + self._decision.move]

    @property
    def decision(self) -> SelectorDecision | None:
        """The decision standing for the most recent epoch, or ``None`` before the first."""
        return self._decision

    def as_metrics(self) -> dict[str, float]:
        """Cumulative selector counters, under the actor's ``route/`` namespace.

        The full key set every step, so two arms sharing a panel emit the same keys, and so
        that the realised move rate a matched control is configured from can be read off the
        run rather than assumed from its config.
        """
        return {
            "route/harness_decisions": float(self.decisions),
            "route/harness_moves": float(self.moves),
            "route/harness_refusals": float(self.refusals),
            "route/harness_blocked": float(self.blocked),
            "route/harness_up_moves": float(self.up_moves),
            "route/harness_down_moves": float(self.down_moves),
            "route/harness_repeat_observations": float(self.repeat_observations),
            "route/harness_move_rate": float(self.moves) / float(max(self.decisions, 1)),
        }


class TruncationStepLimitSelector(LadderSelector):
    """Lengthen the budget when the batch is mostly truncated, shorten it when it is not.

    The treatment rule. It is a bang-bang controller on ``truncated_fraction`` with a dead
    band between the thresholds, and the dead band is the point: a rule that moved on every
    batch would be indistinguishable from one that moved on a coin flip, because with a
    two-or-three-rung ladder "always move" is a deterministic cycle. Refusing in the band is
    what makes the moves informative about the batch.

    The rule is expected to be ABSORBING: it drives the budget towards the rung whose
    truncation lies inside the band and then refuses, so a long run makes few moves. That is
    correct behaviour for a controller, and it is the reason the matched control is matched
    on realised moves per decision rather than on a nominal rate -- a control that moved on
    a schedule would move many times more often than the treatment and the comparison would
    be between activity levels rather than between placements.

    Args:
        up_threshold: Batch-mean truncated fraction at or above which the budget moves one
            rung LONGER. The default 0.5 means "most of this batch never terminated".
        down_threshold: Batch-mean truncated fraction at or below which the budget moves one
            rung SHORTER, recovering rollout time the batch is not using.

    Raises:
        ValueError: If the thresholds are not ordered and inside ``[0, 1]``. Crossed
            thresholds would make the two branches overlap, and which one fired would depend
            on the order they are tested in rather than on the batch.
    """

    def __init__(self, up_threshold: float = 0.5, down_threshold: float = 0.05) -> None:
        super().__init__("truncation_step_limit")
        if not 0.0 <= down_threshold < up_threshold <= 1.0:
            raise ValueError(
                f"need 0 <= down_threshold < up_threshold <= 1, got "
                f"down={down_threshold} up={up_threshold}; crossed or equal thresholds make "
                f"the branch that fires depend on test order rather than on the batch"
            )
        self.up_threshold = float(up_threshold)
        self.down_threshold = float(down_threshold)

    def _propose(self, truncated_fraction: float) -> tuple[int, str]:
        """Longer above ``up_threshold``, shorter at or below ``down_threshold``, else refuse."""
        if truncated_fraction >= self.up_threshold:
            return 1, (
                f"truncated_fraction {truncated_fraction:.4f} >= {self.up_threshold}: "
                f"most of the batch never terminated"
            )
        if truncated_fraction <= self.down_threshold:
            return -1, (
                f"truncated_fraction {truncated_fraction:.4f} <= {self.down_threshold}: "
                f"the batch is not using its budget"
            )
        return 0, (
            f"truncated_fraction {truncated_fraction:.4f} is inside the dead band "
            f"({self.down_threshold}, {self.up_threshold})"
        )


class RateMatchedControlSelector(LadderSelector):
    """Move as OFTEN as the treatment did, but never because of the batch.

    The control that isolates the claim. A harness-routing arm asserts that moving the
    budget *in response to truncation* helps; an unmatched control -- round-robin, or a fixed
    budget -- confounds that with moving at all, and with how often. This rule takes the
    treatment's REALISED moves-per-decision and up-share, measured from the treatment's own
    counters after it ran, and reproduces them from a seeded RNG that never sees the batch.

    Configuring it from a NOMINAL rate would not be a control. The treatment's rate is an
    outcome of the data: its thresholds, the ladder, and where truncation happened to sit
    determine how often it moved, and this project has already recorded a routing result
    where the arm and its control differed in activity rather than in policy.

    The residual mismatch is expected to be non-zero and must be REPORTED, not tuned away.
    Two mechanisms cause it. The draw is Bernoulli, so a finite run realises a rate near but
    not equal to ``move_rate``. And a drawn direction that would leave the ladder is flipped
    inward rather than refused, which preserves the move COUNT -- the quantity being matched
    -- at the cost of the direction mix; every flip is counted in :attr:`flips`.

    Args:
        move_rate: Probability that a decision moves. Set from the treatment's
            ``moves / decisions``.
        up_share: Probability that a move is one rung longer. Set from the treatment's
            ``up_moves / moves``. Ignored when ``moves`` was zero, in which case any value
            is unused because no move is ever drawn.
        seed: RNG seed, so the control is reproducible and two control runs differing only
            in seed are a variance estimate rather than two anecdotes.

    Raises:
        ValueError: If either probability is outside ``[0, 1]``.
    """

    def __init__(self, move_rate: float, up_share: float = 0.5, seed: int = 0) -> None:
        super().__init__("rate_matched_control")
        for label, p in (("move_rate", move_rate), ("up_share", up_share)):
            if not math.isfinite(p) or not 0.0 <= p <= 1.0:
                raise ValueError(f"{label} must be a probability in [0, 1], got {p!r}")
        self.move_rate = float(move_rate)
        self.up_share = float(up_share)
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self.flips = 0

    def _propose(self, truncated_fraction: float) -> tuple[int, str]:
        """Draw a move, ignoring the batch. The argument is accepted and deliberately unused."""
        if self._rng.random() >= self.move_rate:
            return 0, f"rate-matched control: no move drawn at rate {self.move_rate:.4f}"
        up = self._rng.random() < self.up_share
        return (1 if up else -1), (
            f"rate-matched control: move drawn at rate {self.move_rate:.4f}, "
            f"direction {'longer' if up else 'shorter'} at up_share {self.up_share:.4f}"
        )

    def observe(self, epoch, truncated_fraction, *, variants, current):
        """Draw a move, then flip a blocked direction inward rather than losing the move.

        Overridden because the base class turns a blocked move into a refusal, and for the
        control that would silently lower the very rate it exists to match: the treatment is
        absorbing, so a control sharing its ladder spends much of the run at an end.
        """
        decision = super().observe(
            epoch, truncated_fraction, variants=variants, current=current
        )
        if not decision.blocked:
            return decision
        rungs = ladder(variants)
        index = [v.name for v in rungs].index(current.name)
        move = 1 if index == 0 else -1
        self.flips += 1
        # Undo the refusal the base class recorded and book the flipped move instead, so the
        # counters keep describing what the arm DID.
        self.refusals -= 1
        self.blocked -= 1
        self.moves += 1
        if move > 0:
            self.up_moves += 1
        else:
            self.down_moves += 1
        self._decision = SelectorDecision(
            decision.epoch,
            move,
            f"{decision.reason}; flipped inward to keep the move rate matched",
            decision.observation,
            blocked=False,
        )
        return self._decision

    def as_metrics(self) -> dict[str, float]:
        """Base counters plus the flip count, which is half the residual mismatch."""
        out = super().as_metrics()
        out["route/harness_control_flips"] = float(self.flips)
        out["route/harness_control_target_rate"] = float(self.move_rate)
        out["route/harness_control_target_up_share"] = float(self.up_share)
        return out


#: Selectors an arm may name in ``group_routing.harness_selector``. Values are factories
#: taking the keyword arguments in ``group_routing.harness_selector_args``, so a control can
#: be configured from a measured rate without a second config schema.
SELECTORS: dict[str, Callable[..., LadderSelector]] = {
    "truncation_step_limit": TruncationStepLimitSelector,
    "rate_matched_control": RateMatchedControlSelector,
}


def build_selector(name: str, args: dict | None = None) -> LadderSelector:
    """Resolve a configured selector name and its arguments into a selector.

    The single production entry point, so "which selectors exist?" has one answer and the
    trainer does not reimplement the registry lookup.

    Args:
        name: A key of :data:`SELECTORS`.
        args: Keyword arguments for the factory. ``seed`` and integer-valued floats are
            coerced to ``int`` so that a config that arrives through OmegaConf as ``7.0``
            builds the same selector as one that arrives as ``7``.

    Returns:
        The constructed selector.

    Raises:
        ValueError: If the name is unregistered, or the arguments do not fit the factory.
            Falling back to a default selector would report an arm that never ran.
    """
    factory = SELECTORS.get(name)
    if factory is None:
        raise ValueError(
            f"unknown harness selector {name!r}; registered: {sorted(SELECTORS)}. "
            f"Falling back to the default rule would report a selector arm that never ran"
        )
    kwargs = dict(args or {})
    if "seed" in kwargs:
        kwargs["seed"] = int(kwargs["seed"])
    try:
        return factory(**kwargs)
    except TypeError as exc:
        raise ValueError(
            f"harness_selector_args {sorted(kwargs)} do not fit selector {name!r}: {exc}"
        ) from exc
