"""Harness selection that reads a FEATURE, and the rate-matched control that makes it evidence.

:class:`~selfevo.harness.dispatch.HarnessDispatcher` takes a ``selector`` -- a rule answering
"given the configured variant set and the active member, which member should a PROPOSE move
to?" -- and ships :func:`~selfevo.harness.dispatch.round_robin` in that slot. Round-robin is
a placeholder and its own docstring says so: it is deterministic, it always moves, and it
visits every member, which is what a *measurement* arm needs and is all it is for. What it
cannot do is test the prediction the paper actually makes, which is not "the harness changes"
but

    the harness should follow a property of the trajectories -- when the agent keeps running
    out of steps, the STEP BUDGET is the thing that should move.

Round-robin proposes blindly, so an arm running it answers a different question. This module
implements the rule that answers the intended one and, of equal importance, the control
without which the answer cannot be read.

**Why the control is half of this module.** This project has now reached the same finding
three times from three directions: a "smart" rule that beats a baseline turns out to be
indistinguishable from a random rule applied at the same RATE.
:mod:`selfevo.routing.proportions` exists for exactly that reason and says so in its first
paragraph; :mod:`selfevo.routing.rule_policy` opens with a retraction of a seven-feature rule
that collapsed onto one predicate; and the routing-targeting audit found the targeted and
random arms tying once their intervention rates were matched. A feature-driven harness arm
inherits the hazard in full, and in a sharper form than a router does: changing the step
limit changes the rollout distribution whatever the reason for the change, so "switches
sometimes" is a treatment all by itself. The comparison this module is built to support is
therefore

    treatment  :class:`TruncationStepLimitSelector`   switches at rate p, chosen BY the feature
    control    :class:`RateMatchedControlSelector`    switches at rate p, chosen INDEPENDENTLY

and the difference between those two arms is the only quantity attributable to TARGETING. An
arm reported against a no-harness control measures p. It does not measure targeting and must
not be written up as if it did.

**One decision per observation, and why the denominator has to be nailed down first.** A
switch RATE is a fraction, and a fraction is worthless until its denominator is fixed. The
denominator here is not obvious, because ``HarnessDispatcher.consume`` calls a selector a
data-dependent number of times per batch: it stops at the first proposal that MOVES, so a
batch whose selector moves calls it once, while a batch whose selector declines calls it once
per proposing group. Counting raw calls would make the denominator a function of the outcome
-- refusing batches would contribute more denominator than moving ones -- and the treatment
and control rates would not be comparable even when both behaved identically.

So the unit of decision here is the OBSERVATION: :meth:`observe` marks a new batch, the first
call after it takes the decision, and every further call in the same batch is refused and
counted separately as a repeat. Both selectors inherit that machinery from one place, so the
two arms cannot drift apart on the definition of the very quantity being matched.

**Determinism.** The treatment holds no RNG at all: its decision is a pure function of the
observed features, the configured set and the active member, with positional tie-breaks. The
control holds a private :class:`random.Random` seeded at construction, so its schedule is
fixed by ``seed`` alone and is reproducible without reference to anything the run does. An
unreproducible arm is not evidence, and a control whose schedule depended on the features
would not be a control.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from selfevo.harness.base import HarnessVariant
from selfevo.harness.dispatch import HarnessSelectionRefused
from selfevo.routing.contextual import MissingFeatures

__all__ = [
    "TRUNCATION_FEATURE",
    "SELECTOR_METRIC_KEYS",
    "SelectionRecord",
    "TruncationStepLimitSelector",
    "RateMatchedControlSelector",
]

# The feature the treatment rule reads, by the name `selfevo.observability.GroupFeatures`
# gives it. Named once so a test can assert the rule reads a feature production actually
# produces, rather than a string that only ever appears in the test that made it up.
TRUNCATION_FEATURE = "truncated_fraction"

# The metric key set both selectors emit, IN FULL, on every call to `as_metrics`.
#
# Shared between treatment and control on purpose, and asserted equal by test. Two arms that
# emit different keys cannot be put on one panel, and this axis exists to be read as a
# difference between two arms; a control whose refusals were counted under a key the
# treatment never emits would have to be compared by eye against a different chart.
SELECTOR_METRIC_KEYS: tuple[str, ...] = (
    "route/harness_sel_decisions",
    "route/harness_sel_moves",
    "route/harness_sel_rate",
    "route/harness_sel_refused_no_move_wanted",
    "route/harness_sel_refused_no_variant",
    "route/harness_sel_repeat_calls",
)

# The two reasons a decision can decline, kept as constants because they are also metric key
# suffixes and a typo in one of the two places would produce a counter nobody increments.
#
#   NO_MOVE_WANTED  the rule looked and did not want to move. For the treatment that is the
#                   dead band; for the control it is a STAY drawn from the schedule. This is
#                   the ordinary case for both arms and is NOT a failure.
#   NO_VARIANT      the rule wanted to move and the CONFIGURED SET could not oblige -- the
#                   active variant is already the longest (or shortest) member. This one is a
#                   statement about the configuration, and an arm whose refusals are mostly
#                   of this kind is an arm whose variant set is too small to express the rule.
_NO_MOVE_WANTED = "no_move_wanted"
_NO_VARIANT = "no_variant"
_CATEGORIES = (_NO_MOVE_WANTED, _NO_VARIANT)


@dataclass(frozen=True)
class SelectionRecord:
    """What one selector decision looked at, and what it decided.

    The audit record for the selection half of a dispatch, one level below
    :class:`~selfevo.harness.dispatch.DispatchRecord`: that one records what the DISPATCHER
    did with an action, this one records why the RULE proposed what it proposed. Both are
    needed, because "the harness did not move" has several causes and an arm whose refusals
    are all dead-band is a different run from one whose refusals are all ceiling.

    Args:
        index: Position of this decision in the selector's own history, counting only real
            decisions -- so it is the index into the sequence whose length is the rate's
            denominator.
        statistic: The scalar the rule decided on. ``nan`` for a selector that reads no
            feature, which is what the control is; recorded rather than omitted so that "this
            arm read nothing" is a value in the record instead of an absence to be inferred.
        before: Name of the variant active when the decision was taken.
        after: Name of the variant proposed, or ``None`` for a refusal.
        moved: Whether a variant was proposed. Never inferred from ``after``, and never from
            the fact that a decision was taken at all.
        reason: The rule's own explanation, carried into
            :attr:`~selfevo.harness.dispatch.DispatchRecord.reason` when the dispatcher turns
            a refusal into a record, so a run's logs quote the rule rather than paraphrasing
            it.
    """

    index: int
    statistic: float
    before: str
    after: str | None
    moved: bool
    reason: str


def _check_set(
    variants: Sequence[HarnessVariant], current: HarnessVariant | None, who: str
) -> None:
    """Refuse a call whose variant set and active member cannot both be true.

    Args:
        variants: The configured set, as the dispatcher holds it.
        current: The active member.
        who: Name of the calling rule, for the message.

    Raises:
        ValueError: If the set is empty, or ``current`` is not one of its members. These are
            PROGRAMMER conditions, not data conditions, so they raise rather than becoming a
            recorded refusal: the caller's idea of what exists and the dispatcher's have
            diverged, and choosing either one would dispatch under a configuration that no
            arm declared. Same rule, and same reasoning, as
            :func:`~selfevo.harness.dispatch.round_robin`.
    """
    if not variants:
        raise ValueError(f"{who} needs a non-empty variant set")
    names = [v.name for v in variants]
    if current is None or current.name not in names:
        got = None if current is None else current.name
        raise ValueError(
            f"{who}: current variant {got!r} is not in the configured set {names}; "
            "the caller and the dispatcher disagree about what exists"
        )


def _neighbour(
    variants: Sequence[HarnessVariant], current: HarnessVariant, *, longer: bool
) -> HarnessVariant | None:
    """The NEAREST configured variant with a strictly larger (or smaller) ``step_limit``.

    Nearest rather than extreme, which is the single most consequential choice in this
    module's rule. Jumping straight to the largest budget on the first truncated batch would
    make the arm's trajectory a function of the SET'S DIAMETER rather than of the feature: the
    same run over ``[plain, long]`` and over ``[plain, long, enormous]`` would land in
    different places from identical evidence, and the second arm's result could not be
    attributed to the rule. A one-step move keeps exactly one variant boundary crossed per
    decision, so a run's harness history reads as a walk whose every step has a reason
    attached, and it lets the budget settle wherever truncation stops rather than at whichever
    end of the set was configured.

    A variant whose ``step_limit`` EQUALS the current one is neither longer nor shorter and is
    never returned. The set may legitimately contain such a pair -- variants differing only in
    ``settings`` are explicitly allowed by the dispatcher's construction guard -- and moving
    between them would be a change the feature did not ask for, dressed up as one it did.

    Args:
        variants: The configured set, in configured order.
        current: The active member.
        longer: True to look upward in ``step_limit``, False to look downward.

    Returns:
        The nearest member in the requested direction, or ``None`` when the active variant is
        already at that end of the set. ``None`` is a legitimate answer and the caller turns
        it into an explicit refusal; it is never a reason to return ``current``.
    """
    best_key: tuple[int, int] | None = None
    best: HarnessVariant | None = None
    for i, v in enumerate(variants):
        delta = v.step_limit - current.step_limit
        if longer and delta <= 0:
            continue
        if not longer and delta >= 0:
            continue
        # Ties in distance break by CONFIGURED POSITION, so the rule is reproducible over a
        # set that carries two variants at the same step budget.
        key = (abs(delta), i)
        if best_key is None or key < best_key:
            best_key, best = key, v
    return best


class _ObservationSelector:
    """Machinery shared by the treatment and its control: one decision per observed batch.

    Not an abstraction for its own sake. The two selectors have to agree on what a DECISION
    is, because the whole design rests on matching a rate and a rate is only as meaningful as
    its denominator; if each class counted decisions its own way, the two arms could be
    reported as rate-matched while counting different things. So the counting, the
    once-per-observation rule, the audit records and the emitted key set all live here, and a
    subclass supplies only :meth:`_decide`.

    The lifecycle is:

    * :meth:`observe` -- called once per batch, before the batch's actions are consumed. It
      opens a new decision epoch and hands the subclass whatever the batch carries.
    * ``__call__`` -- the dispatcher's ``selector`` seam. The FIRST call in an epoch takes the
      decision and records it. Every later call in the same epoch is refused and counted as a
      repeat, because ``consume`` calls the selector again for each proposing group after a
      refusal, and letting those become decisions would inflate the denominator on exactly
      the batches where nothing happened.

    A missing :meth:`observe` between two batches is therefore not silent either: the second
    batch's calls land in the first batch's epoch, are refused as repeats, and
    ``route/harness_sel_repeat_calls`` climbs while ``route/harness_sel_decisions`` does not.
    """

    def __init__(self) -> None:
        self._epoch = 0
        self._decided_epoch = 0
        self._records: list[SelectionRecord] = []
        self._repeat_calls = 0
        self._refusals = {c: 0 for c in _CATEGORIES}

    # ------------------------------------------------------------------ observing ----

    def observe(self, rows: Iterable[Mapping[str, float]] | None = None) -> None:
        """Open a new decision epoch, carrying this batch's per-group features.

        Args:
            rows: One mapping per routed group, as
                :meth:`selfevo.observability.GroupFeatures.as_extra` produces and as
                ``RoutingContext.extra`` carries. A selector that reads no feature ignores
                this, and the parameter is present on both arms anyway so that production
                wires the treatment and the control through an IDENTICAL call site -- two
                arms that differ in how they are called differ in more than the thing under
                test.

        Raises:
            Whatever the subclass's :meth:`_on_observe` raises for evidence it cannot use.
        """
        self._epoch += 1
        self._on_observe(rows)

    def _on_observe(self, rows: Iterable[Mapping[str, float]] | None) -> None:
        """Take whatever this rule needs from the batch. Subclass hook."""
        raise NotImplementedError

    # ------------------------------------------------------------------- deciding ----

    def __call__(
        self, variants: Sequence[HarnessVariant], current: HarnessVariant
    ) -> HarnessVariant:
        """The ``selector`` seam: propose a variant, or refuse and say why.

        Args:
            variants: The configured set, in configured order.
            current: The active member.

        Returns:
            A member of ``variants`` other than ``current``.

        Raises:
            ValueError: If the set is empty or ``current`` is not a member (see
                :func:`_check_set`), or if no observation has been made -- deciding on
                evidence that never arrived is the silent substitution this whole module
                refuses.
            HarnessSelectionRefused: When the rule declines. The dispatcher turns this into a
                :class:`~selfevo.harness.dispatch.DispatchRecord` with ``changed=False`` and
                ``refused=True``, so a decline is visible in the metrics and is never counted
                as a switch.
        """
        _check_set(variants, current, type(self).__name__)
        if self._epoch == 0:
            raise ValueError(
                f"{type(self).__name__} was called before observe(); a selection taken "
                "before any batch was observed would be a decision on evidence that never "
                "arrived, and a rule that quietly substituted a default there would report "
                "an arm that never ran"
            )
        if self._decided_epoch == self._epoch:
            self._repeat_calls += 1
            raise HarnessSelectionRefused(
                "the decision for this observation has already been taken; a harness is a "
                "shared artefact and one batch is one decision, so the later proposals in "
                "this batch are evidence for a decision already made"
            )
        self._decided_epoch = self._epoch

        target, statistic, reason, category = self._decide(variants, current)
        index = len(self._records)
        if target is None:
            if category not in self._refusals:
                raise ValueError(
                    f"{type(self).__name__} refused with unknown category {category!r}; "
                    f"known: {list(_CATEGORIES)}. An unrecognised category would be counted "
                    "nowhere and the refusal would vanish from the metrics"
                )
            self._refusals[category] += 1
            self._records.append(
                SelectionRecord(index, statistic, current.name, None, False, reason)
            )
            raise HarnessSelectionRefused(reason)
        self._records.append(
            SelectionRecord(index, statistic, current.name, target.name, True, reason)
        )
        return target

    def _decide(
        self, variants: Sequence[HarnessVariant], current: HarnessVariant
    ) -> tuple[HarnessVariant | None, float, str, str]:
        """Decide for one observation. Subclass hook.

        Returns:
            ``(target, statistic, reason, category)``. ``target`` is ``None`` for a refusal,
            in which case ``category`` names which of :data:`_CATEGORIES` it was; on a move
            the category is ignored.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ reporting ----

    @property
    def records(self) -> tuple[SelectionRecord, ...]:
        """One record per DECISION, in order. Repeat calls are not decisions and are absent."""
        return tuple(self._records)

    @property
    def decisions(self) -> int:
        """The rate's denominator: observations on which this rule was actually asked."""
        return len(self._records)

    @property
    def moves(self) -> int:
        """The rate's numerator: decisions on which this rule proposed a different variant."""
        return sum(1 for r in self._records if r.moved)

    @property
    def repeat_calls(self) -> int:
        """Calls that arrived after this epoch's decision was already taken."""
        return self._repeat_calls

    @property
    def switch_rate(self) -> float:
        """Moves per decision, or ``nan`` before any decision has been taken.

        ``nan`` rather than 0.0, deliberately. An arm that was never asked and an arm that was
        asked and always declined are different runs with different diagnoses, and a 0.0 in
        this slot merges them into one number that reads as "the rule does nothing". The
        denominator is emitted beside it so the distinction survives into the logs.
        """
        if not self._records:
            return math.nan
        return self.moves / len(self._records)

    def outcomes(self) -> tuple[bool, ...]:
        """The realised move/stay sequence, which is what a matched control replays."""
        return tuple(r.moved for r in self._records)

    def as_metrics(self) -> dict[str, float]:
        """Metrics for ``stats_tracker.scalar``, under the actor's ``route/`` namespace.

        Emits :data:`SELECTOR_METRIC_KEYS` in full every time, so the treatment and the
        control produce the same key set on every step and can share a panel.

        CUMULATIVE, unlike :meth:`selfevo.harness.dispatch.DispatchBatch.as_metrics`, which is
        per batch. That is not an inconsistency: a rate over a single step is 0 or 1 and
        carries no information, while the quantity this module exists to match is a run-level
        fraction. The per-step view of the same axis is already emitted by the dispatcher.

        Returns:
            ``{metric name: float}``, every value a float -- ``nan`` included, for a rate that
            is not yet defined.
        """
        return {
            "route/harness_sel_decisions": float(self.decisions),
            "route/harness_sel_moves": float(self.moves),
            "route/harness_sel_rate": float(self.switch_rate),
            "route/harness_sel_refused_no_move_wanted": float(
                self._refusals[_NO_MOVE_WANTED]
            ),
            "route/harness_sel_refused_no_variant": float(self._refusals[_NO_VARIANT]),
            "route/harness_sel_repeat_calls": float(self._repeat_calls),
        }


class TruncationStepLimitSelector(_ObservationSelector):
    """Move the step budget in the direction the truncation rate points, one variant at a time.

    **The rule.** Let ``t`` be the mean of ``truncated_fraction`` over the batch's groups --
    the fraction of this batch's rollouts that hit the step budget rather than terminating.
    Then

        ``t >= raise_above``   propose the NEAREST variant with a LARGER ``step_limit``
        ``t <= lower_below``   propose the NEAREST variant with a SMALLER ``step_limit``
        otherwise              refuse, recorded as ``no_move_wanted``

    and in either of the first two cases, if the set has no member in that direction, refuse,
    recorded as ``no_variant``, naming the ceiling or floor it hit.

    **Why this shape, clause by clause.**

    *Direction, not destination.* The feature says the budget is binding, or that it is not.
    It does not say by how much: nothing in ``truncated_fraction`` measures how many more
    steps a truncated rollout needed. A rule that jumped to the largest budget would be
    claiming a magnitude the evidence does not carry, and would make the arm's behaviour a
    function of the configured set's diameter. See :func:`_neighbour`.

    *Symmetric, and this is the clause that makes the axis about ADAPTATION.* Without the
    downward move, "the harness follows the feature" is confounded with "the harness gets more
    compute": an arm that only ever grows its step limit is also an arm that spends more, and
    a gain over a fixed-budget control could be bought entirely with the extra steps. The
    downward branch is what makes the treatment a rule about MATCHING the budget to the
    workload rather than about raising it, and it is why a run's mean step budget is a number
    worth logging beside its reward.

    *The two thresholds are not mirror images, on purpose.* Getting the upward move wrong
    costs compute; getting the downward move wrong destroys solves that were within reach.
    The defaults price that asymmetry in: ``raise_above=0.5`` says raise the budget once it
    binds for the MEDIAN rollout rather than for a tail, since the extra steps are paid for by
    every rollout in the batch while only the truncated ones can benefit; ``lower_below=0.05``
    says cut it only when it bound for at most about one rollout in twenty.

    **What is measured and what is not.** The branch's PREMISE is measured and is recorded in
    :mod:`selfevo.routing.rule_policy`: on OlympiadBench, doubling the token cap moved
    truncation 79 -> 78, and ``n_truncated == n_no_box`` in every MATH/AMC/AIME row, so a
    truncated sample is one that never terminated usefully rather than one that was a few
    tokens short. **The two threshold VALUES are not pinned by any measurement, and this
    docstring says so rather than inventing one**, in the same terms
    ``RulePolicyRouter.truncated_threshold`` uses for the same reason. They are defensible
    prices for an asymmetric error, not estimates. The experiment that would ground them is a
    sweep of ``raise_above`` at a fixed ``lower_below``, and until it is run the honest
    statement is that this rule's SHAPE is the claim and its thresholds are hyperparameters.

    **The statistic is a mean over groups.** With the equal group sizes GRPO uses, the mean of
    the per-group truncated fractions is exactly the fraction of the batch's SAMPLES that were
    truncated, which is the quantity the step budget acts on; with unequal groups it is the
    unweighted mean of per-group rates, and this docstring is the only place that distinction
    is recorded. A max would let one pathological group move a shared artefact; a min would
    require unanimity from a batch that is deliberately heterogeneous.

    **Determinism.** No RNG anywhere in this class. Two instances with the same configuration,
    shown the same observations, produce identical histories -- there is no seed to get wrong,
    which is a stronger guarantee than reproducibility under one. Ties in
    :func:`_neighbour` break by configured position for the same reason.

    Args:
        raise_above: ``t`` at or above this proposes a longer budget. In [0, 1].
        lower_below: ``t`` at or below this proposes a shorter budget. In [0, 1], and
            strictly below ``raise_above``.
        feature: Name of the feature read from each row, defaulting to
            :data:`TRUNCATION_FEATURE`. A parameter only so the rule can be pointed at a
            differently-named producer; it is not a knob for changing what the rule means.

    Raises:
        ValueError: If either threshold is outside [0, 1]; if ``lower_below >= raise_above``,
            which would either erase the dead band or let one value of ``t`` satisfy both
            branches so that the rule's answer depended on the order the branches are written
            in; or if ``feature`` is empty.
    """

    def __init__(
        self,
        *,
        raise_above: float = 0.5,
        lower_below: float = 0.05,
        feature: str = TRUNCATION_FEATURE,
    ) -> None:
        for name, value in (("raise_above", raise_above), ("lower_below", lower_below)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if lower_below >= raise_above:
            raise ValueError(
                f"lower_below ({lower_below}) must be strictly below raise_above "
                f"({raise_above}); at or above it a single truncated_fraction would satisfy "
                "both branches and the rule's answer would depend on which branch is written "
                "first, which is a coin flip wearing a threshold's clothes"
            )
        if not feature:
            raise ValueError("feature name must be non-empty")
        super().__init__()
        self.raise_above = raise_above
        self.lower_below = lower_below
        self.feature = feature
        self._statistic = math.nan

    def _on_observe(self, rows: Iterable[Mapping[str, float]] | None) -> None:
        """Reduce this batch's per-group features to the one scalar the rule decides on.

        Args:
            rows: One feature mapping per routed group.

        Raises:
            ValueError: If ``rows`` is empty or ``None``. A batch with no groups carries no
                evidence, and a rule that decided anyway -- on a stale statistic, or on a
                default -- would be reporting a decision it did not make. Also raised if a
                value is non-finite or outside [0, 1]: ``group_features`` guarantees both, so
                a violation means these rows did not come from it and the rule is being fed
                something else.
            MissingFeatures: If a row lacks the feature. The same exception type
                :class:`selfevo.routing.contextual.ContextualBanditRouter` and
                :class:`selfevo.routing.rule_policy.RulePolicyRouter` raise, so a caller
                handling one handles all three; and raised rather than defaulted because a
                substituted 0.0 here reads as "nothing truncated" and would quietly propose
                a SHORTER budget on a batch whose features never arrived.
        """
        rows = [] if rows is None else list(rows)
        if not rows:
            raise ValueError(
                f"{type(self).__name__}.observe() was given no group features; a rule that "
                "decided from an empty batch would be deciding from no evidence at all"
            )
        total = 0.0
        for i, row in enumerate(rows):
            if self.feature not in row:
                raise MissingFeatures(
                    f"group {i} is missing feature {self.feature!r}; present: "
                    f"{sorted(row)}. Populate these rows from "
                    "selfevo.observability.group_features -- a substituted default here "
                    "would propose a budget change on evidence that never arrived"
                )
            value = float(row[self.feature])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"group {i} has {self.feature}={value}; group_features guarantees a "
                    "finite fraction in [0, 1], so this row came from somewhere else"
                )
            total += value
        self._statistic = total / len(rows)

    def _decide(
        self, variants: Sequence[HarnessVariant], current: HarnessVariant
    ) -> tuple[HarnessVariant | None, float, str, str]:
        """Apply the rule to the observed statistic. See the class docstring for the rule.

        Raises:
            ValueError: If every configured variant carries the same ``step_limit``. That set
                is legal for the dispatcher -- variants differing only in ``settings`` are an
                axis an adapter really varies, and its construction guard accepts them -- but
                this rule moves along the step budget and cannot move over it at all. An arm
                configured that way would refuse every proposal while reporting a
                feature-driven harness, and would train exactly like the control it is meant
                to be compared against. Raised rather than recorded as a refusal, because it
                is a statement about the configuration and not about the batch, and because
                the alternative is a run whose most common log line is a refusal nobody reads.
        """
        t = self._statistic
        names = [v.name for v in variants]
        limits = {v.step_limit for v in variants}
        if len(variants) > 1 and len(limits) == 1:
            raise ValueError(
                f"variants {names} all carry step_limit {next(iter(limits))}; this rule "
                "moves along the step budget and can never move over such a set, so the arm "
                "would refuse every proposal and train exactly like its own control while "
                "reporting a feature-driven harness"
            )

        if t >= self.raise_above:
            target = _neighbour(variants, current, longer=True)
            if target is None:
                return (
                    None,
                    t,
                    f"{self.feature}={t:.3f} >= {self.raise_above} calls for a longer budget, "
                    f"but {current.name!r} already has the largest step_limit "
                    f"({current.step_limit}) in {names}; the configured harness has nothing "
                    "longer to offer",
                    _NO_VARIANT,
                )
            return (
                target,
                t,
                f"{self.feature}={t:.3f} >= {self.raise_above}: step_limit "
                f"{current.step_limit} -> {target.step_limit}",
                "",
            )

        if t <= self.lower_below:
            target = _neighbour(variants, current, longer=False)
            if target is None:
                return (
                    None,
                    t,
                    f"{self.feature}={t:.3f} <= {self.lower_below} calls for a shorter "
                    f"budget, but {current.name!r} already has the smallest step_limit "
                    f"({current.step_limit}) in {names}; the configured harness has nothing "
                    "shorter to offer",
                    _NO_VARIANT,
                )
            return (
                target,
                t,
                f"{self.feature}={t:.3f} <= {self.lower_below}: step_limit "
                f"{current.step_limit} -> {target.step_limit}",
                "",
            )

        return (
            None,
            t,
            f"{self.feature}={t:.3f} is inside the dead band "
            f"({self.lower_below}, {self.raise_above}): the budget binds for neither most "
            "rollouts nor almost none of them, so the feature names no direction",
            _NO_MOVE_WANTED,
        )


class RateMatchedControlSelector(_ObservationSelector):
    """Switch at the treatment's MEASURED rate, choosing when and where without the feature.

    The arm that makes a feature-driven result mean something. Its only job is to hold the
    intervention rate fixed while destroying the link between the feature and the
    intervention, so that the treatment-minus-control difference is attributable to TARGETING
    and not to the fact that switching the step limit at all perturbs the run. It is the same
    device as :class:`selfevo.routing.proportions.MatchedPermutationControl`, one level up:
    that one replays a router's realised decision multiset shuffled across units, this one
    replays a selector's realised move/stay multiset shuffled across decision points.

    **How the rate is matched.** From a completed treatment run, take ``moves`` and
    ``decisions`` -- realised, measured, never nominal, for the reason
    :mod:`selfevo.routing.proportions` gives at length: a control configured with the
    probability the treatment was SUPPOSED to realise is not matched to the one it did.
    :meth:`from_treatment` reads both off the treatment selector. The control then builds a
    deck of ``moves`` MOVE tokens and ``decisions - moves`` STAY tokens, shuffles it with a
    private :class:`random.Random`, and serves one token per decision, reshuffling a fresh
    deck each time it is exhausted.

    **Per-step or over the run? Over the run, and the choice is the whole design.** Matching
    per step -- moving exactly on the steps the treatment moved -- would match the rate
    perfectly and would be the WRONG control: the times of intervention would then be a
    deterministic function of the feature, so the control would inherit the treatment's
    targeting in the time dimension and could only test the choice of destination. The
    hypothesis under test is that the harness should follow the feature, and a control that
    follows the feature's timing has conceded the hypothesis before the run starts. Matching
    over the run keeps the marginal rate identical and makes the intervention times
    independent of the feature, which is exactly the null this arm has to represent.

    **Residual mismatch, stated exactly.** After ``n = q * decisions + r`` calls the control
    has made ``q * moves + s`` moves, where ``s`` is the number of MOVE tokens in the first
    ``r`` cards of a shuffled deck, so ``0 <= s <= min(r, moves)``:

    * at every multiple of ``decisions`` the realised rate is EXACTLY the treatment's -- not
      matched in expectation, matched by construction, which a Bernoulli draw at probability
      ``p`` could not offer (its count has standard deviation ``sqrt(n p (1-p))``);
    * in between, the realised rate differs by at most ``r / n`` in absolute value, worst case
      ``(decisions - 1) / n``, and its expectation is the treatment's rate exactly;
    * the two arms are separate RUNS, so the control's own number of proposing batches need
      not equal the treatment's. That is why the deck recycles rather than being consumed
      once: an in-order replay would only match if the two runs happened to take the same
      number of decisions, and :class:`MatchedPermutationControl` records measuring an 8.5%
      realised rate against a 32% target when it made that assumption.

    **What is NOT matched, said out loud.** The DESTINATION mixture. When the control moves it
    draws uniformly among the members that are not active, while the treatment's destinations
    are whatever its rule chose -- typically skewed toward longer budgets. So this control
    isolates "switching at rate p" from "switching where the feature says", and it does not
    separate "the feature says where" from "longer budgets are simply better". If the
    treatment beats this control, the follow-up that separates those two is a second control
    replaying the treatment's realised DESTINATION multiset on this same feature-independent
    schedule. Naming that here rather than discovering it in review is the point; a run
    reported against this control alone must carry the caveat.

    **Determinism.** The schedule is a function of ``seed`` and nothing else -- not of the
    features, not of the batch contents, not of global RNG state, since the generator is
    private in the same way and for the same reason as ``RandomRouter``'s. Two runs at the
    same seed have identical schedules; two at different seeds do not.

    Args:
        moves: MOVE tokens per deck -- the treatment's realised switch count.
        decisions: Deck size -- the treatment's realised decision count, i.e. the denominator
            of its switch rate.
        seed: Schedule seed. Vary it across replicate control runs; hold it fixed to reproduce
            one.

    Raises:
        ValueError: If ``decisions`` is not positive, or ``moves`` is negative or exceeds
            ``decisions``. A control with an empty deck would be a silent no-op arm -- the
            same guard, for the same reason, as
            :class:`~selfevo.routing.proportions.MatchedPermutationControl` -- and a rate
            above 1 is not a rate.
    """

    def __init__(self, *, moves: int, decisions: int, seed: int = 0) -> None:
        if decisions <= 0:
            raise ValueError(
                f"decisions must be positive, got {decisions}; a control with an empty deck "
                "would never move and would be a no-op arm wearing a control's name"
            )
        if not 0 <= moves <= decisions:
            raise ValueError(
                f"moves must be in [0, {decisions}], got {moves}; a switch count above the "
                "decision count is not a rate this control could realise"
            )
        super().__init__()
        self.target_moves = moves
        self.block = decisions
        self.seed = seed
        self._rng = random.Random(seed)
        self._deck: list[bool] = []

    @classmethod
    def from_treatment(
        cls, treatment: _ObservationSelector, *, seed: int = 0
    ) -> "RateMatchedControlSelector":
        """Build the control that matches a treatment selector's REALISED rate.

        The intended construction path, and the only one that cannot mis-specify the rate: it
        reads the numerator and the denominator off the arm being matched rather than
        accepting a probability someone believed the arm would realise.

        Args:
            treatment: A selector that has already been run. Its
                :attr:`~_ObservationSelector.moves` and
                :attr:`~_ObservationSelector.decisions` become this control's deck.
            seed: Schedule seed.

        Returns:
            A control whose realised switch rate equals ``treatment``'s exactly at every
            multiple of ``treatment.decisions`` calls.

        Raises:
            ValueError: If the treatment took no decisions -- there is no rate to match, and
                a control built from one would be a no-op arm.
        """
        return cls(moves=treatment.moves, decisions=treatment.decisions, seed=seed)

    @property
    def target_rate(self) -> float:
        """The rate this control is matched to: ``moves / decisions`` of the treatment run."""
        return self.target_moves / self.block

    def _on_observe(self, rows: Iterable[Mapping[str, float]] | None) -> None:
        """Ignore the batch's features entirely.

        Written as an explicit, documented no-op rather than by leaving the hook unimplemented
        or omitting the parameter, because "this arm does not read the feature" is the arm's
        defining property and it should be a line of code that a reader and a mutation test
        can both find. The parameter is accepted so that the treatment and the control are
        driven through an identical call site.
        """
        return None

    def _next_token(self) -> bool:
        """Draw the next scheduled outcome, reshuffling a fresh deck when one is exhausted.

        Returns:
            True for a scheduled MOVE, False for a scheduled STAY.
        """
        if not self._deck:
            self._deck = [True] * self.target_moves + [False] * (
                self.block - self.target_moves
            )
            self._rng.shuffle(self._deck)
        return self._deck.pop()

    def _decide(
        self, variants: Sequence[HarnessVariant], current: HarnessVariant
    ) -> tuple[HarnessVariant | None, float, str, str]:
        """Serve the next scheduled outcome; on a MOVE, pick a destination at random.

        The statistic recorded is ``nan``: this arm read no feature, and recording a number
        would imply it did.
        """
        if not self._next_token():
            return (
                None,
                math.nan,
                f"rate-matched control drew STAY from a schedule of {self.target_moves} "
                f"moves per {self.block} decisions (seed {self.seed}); no feature was read",
                _NO_MOVE_WANTED,
            )
        others = [v for v in variants if v.name != current.name]
        if not others:
            return (
                None,
                math.nan,
                f"rate-matched control drew MOVE but {current.name!r} is the only configured "
                "variant; this decision is a realised mismatch against the treatment's rate "
                "and is counted as one",
                _NO_VARIANT,
            )
        target = others[self._rng.randrange(len(others))]
        return (
            target,
            math.nan,
            f"rate-matched control drew MOVE from a schedule of {self.target_moves} moves "
            f"per {self.block} decisions (seed {self.seed}) and picked {target.name!r} "
            "uniformly among the other variants; no feature was read",
            "",
        )


# ---------------------------------------------------------------------------------------
# Config surface. Everything above is the audited rule; this block is the part that lets an
# ARM NAME one, which nothing on this branch previously did -- ``build_dispatcher`` took only
# variant names and hardcoded ``round_robin``, so the selectors were reachable from Python
# and from tests but never from a run's configuration.
#
# The registry is deliberately a name -> FACTORY map with keyword arguments passed through
# verbatim, because the control cannot be named without numbers: it is constructed from the
# treatment's REALISED ``moves`` and ``decisions``, read off the treatment run after it
# finished. A registry that took only a name would force the control's rate to be a constant
# in code, which is the nominal-rate mistake ``from_treatment`` exists to prevent.

#: Selectors an arm may name in ``group_routing.harness_selector``.
SELECTORS: dict[str, object] = {
    "truncation_step_limit": TruncationStepLimitSelector,
    "rate_matched_control": RateMatchedControlSelector,
}


def build_selector(name: str, args: dict | None = None):
    """Resolve a configured selector name and its arguments into a selector.

    The single production entry point, so "which selectors exist?" has one answer and the
    trainer does not reimplement the registry lookup.

    Args:
        name: A key of :data:`SELECTORS`.
        args: Keyword arguments for the factory, from
            ``group_routing.harness_selector_args``. Integral values are coerced to ``int``
            so that a config arriving through OmegaConf as ``40.0`` builds the same selector
            as one arriving as ``40``; :class:`RateMatchedControlSelector` takes ``moves``,
            ``decisions`` and ``seed``, all of which are counts.

    Returns:
        The constructed selector.

    Raises:
        ValueError: If the name is unregistered, or the arguments do not fit the factory.
            Falling back to a default rule would report an arm that never ran.
    """
    factory = SELECTORS.get(name)
    if factory is None:
        raise ValueError(
            f"unknown harness selector {name!r}; registered: {sorted(SELECTORS)}. "
            f"Falling back to the default rule would report a selector arm that never ran"
        )
    kwargs = {}
    for key, value in (args or {}).items():
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        kwargs[key] = value
    try:
        return factory(**kwargs)
    except TypeError as exc:
        raise ValueError(
            f"harness_selector_args {sorted(kwargs)} do not fit selector {name!r}: {exc}"
        ) from exc
