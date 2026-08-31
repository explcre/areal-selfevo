"""M9: the rule evolve-policy -- a hand-written cold start, and the baseline to beat.

GOAL.md carries this as "Rule evolve-policy (cold start + baseline)", with the note
*needed before "learned" is falsifiable*. That note is the whole justification for this
module. The learned controller has been measured against the matched RANDOM control and
come out null (2026-08-31: MATH-500 **-0.0020**, OlympiadBench **+0.0000** on 675
problems), which establishes that its per-unit decision buys nothing over a coin at matched
proportions. It says nothing about whether the per-unit decision buys anything over
*thinking*, because there was no written-down rule to compare against. This is that rule.

**Why this is not** :class:`~selfevo.routing.routers.SolveRateRouter`, which the ``rule``
slot on the ``evolve_policy`` axis pointed at before M9:

* It decides from **one** scalar; :class:`~selfevo.routing.contextual.ContextualBanditRouter`
  decides from **seven**. A baseline on a strictly smaller input is not a like-for-like
  comparison -- the arm difference would confound "written vs learned" with "1 feature vs 7".
  This router therefore *consumes* all seven (see :meth:`RulePolicyRouter._features`), and
  documents exactly which of them it reads.
* Its criterion is provably **inert** at this granularity.
  :func:`selfevo.routing.criteria.threshold_is_inert` proves that with ``p_hat = k / G``
  every threshold in ``(0, I_RL(1/G, G)]`` induces the identical partition, and
  SolveRateRouter's default 0.1 is inside that interval at every group size run here
  (``I_RL(1/G, G)`` is 0.68 at G=4, 0.66 at G=8, 0.64 at G=16, and
  ``threshold_is_inert`` returns True at all three). So its one tunable number cannot
  change a decision: the rule is really "was this group unanimous?", re-encoded through a
  plug-in estimate that is additionally biased by Jensen.

This rule asks that question directly, of the quantity that answers it **exactly**.

The rule, one branch at a time, with the measurement each rests on:

``reward_std > 0``  ->  ``RL``
    An identity, not a threshold. GRPO's centred advantage is ``A_i = r_i - rbar``, so a
    group whose raw rewards are not all equal has at least one non-zero advantage and RL
    carries signal (``selfevo.routing.criteria`` module docstring). Keying on ``reward_std``
    rather than on ``solve_rate`` also keeps a group that is unanimous in *outcome* but not
    in *reward* -- e.g. rewards ``[1.0, 0.8, 1.0, 0.8]``, the case
    ``test_reward_std_is_zero_exactly_when_the_group_is_unanimous`` pins -- on the RL branch
    where it belongs, instead of deleting a live gradient because ``solve_rate == 1.0``.

silent and ``solve_rate == 1``  ->  ``SKIP`` (configurable)
    RL is provably dead and the group carries its own SFT target. The branch is still
    skipped by default because the free self-target was **measured inert at 0.5 and harmful
    at 2.0** (EXPERIMENTS.md 2026-08-31; GOAL.md M24 "the solved branch is abandoned"). Those
    are the only two operating points ever measured for it, and neither is a reason to spend
    it. GOAL.md critical-path item 1 is the A/B that would revise this, and
    ``solved_mode="sft"`` is how to run that A/B through this same object.

silent and ``solve_rate == 0``  ->  teacher mode if one exists, else ``SKIP``
    RL is dead and there is no self-target by construction: every sample was wrong, so
    nothing in the rollout can serve as a target (GOAL.md M24). No run in this project wires
    an external teacher, so in practice this branch is ``SKIP`` -- which is the honest state
    of M7/M24, not a design preference.

``truncated_fraction >= 1``, on the unsolved branch  ->  ``HarnessAction.PROPOSE``
    The one place this rule uses a feature no solve-rate router can see, and the concrete
    prediction GOAL.md already stakes on ``truncated_fraction``. Measured on OlympiadBench,
    same checkpoint, cap 8192 -> 16384: truncated went 79 -> 78 for ``ctx`` and 61 -> 64 for
    ``rnd``, and on MATH/AMC/AIME ``n_truncated == n_no_box`` in every row. So a truncated
    sample is a generation that **never terminates usefully**, not one cut off mid-solution;
    more budget does not fix it. That is a scaffold failure, whose consumer is the harness,
    not a knowledge failure, whose consumer would be a teacher.

**Cold start.** Used as the opening policy for a learned router, a rule has one hard
requirement beyond being sensible: it must not hand ``batch_outcomes`` a single-mode batch,
because such a batch is refused as :class:`~selfevo.routing.feedback.ConfoundedUpdate` and
the learner goes blind (GOAL.md, "Feedback for a learned router is only defined when the
batch has mode diversity"; measured as the ``cold_start_rounds=0`` deadlock). This rule
clears that on the measured data rather than by assumption: on the matched GSM8K pair
``silent_group_fraction`` is **0.5906** at G=8 and **0.4553** at G=16 (GOAL.md quotes 57.4%
group-level silence), so both branches are populated in a real batch and a batch routed by
this rule contains both ``rl`` and ``skip``. It is not a
*guarantee* -- an all-informative batch would still be refused -- and
``feedback/confounded_skips`` remains the diagnostic.

**Determinism.** Frozen dataclass, no RNG, no state carried between calls, no dependence on
call order. Same context, same decision, always -- which is what makes it usable as a
baseline at all.

**What this rule is NOT, stated so a comparison is not over-read.** Under the shipped
configuration (no external teacher, ``solved_mode="skip"``) it emits only ``rl`` and
``skip``, while the contextual arm was measured at rl 0.295 / sft 0.353 / skip 0.353. So
rule-vs-learned is **not** a matched-proportion comparison, and neither arm's number means
anything without its own ``router=random`` control at its own measured proportions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from selfevo.observability import FEATURE_NAMES
from selfevo.routing.base import (
    Granularity,
    HarnessAction,
    RoutingContext,
    RoutingDecision,
    TrainingMode,
    known_modes,
)
from selfevo.routing.contextual import MissingFeatures

__all__ = [
    "RulePolicyRouter",
    "InconsistentFeatures",
    "MissingFeatures",
    "READ_FEATURES",
]

# The features this rule actually READS, as opposed to the ones it requires to be present.
# Exposed rather than left implicit because GOAL.md M15's open gap is "no ablation showing
# which features carry the decision" -- for this router the answer is not an experiment, it
# is a constant, and a reader should be able to check that claim without reading `route`.
READ_FEATURES: tuple[str, ...] = ("solve_rate", "reward_std", "truncated_fraction")

# How far ``ctx.extra["solve_rate"]`` may sit from ``ctx.solve_rate`` before the context is
# refused. Both are produced by the same float32 reduction in
# ``observability.group_features`` and copied into the two places by ``actor._route_groups``,
# so the only difference either can legitimately carry is the float32 -> float64 widening,
# which is exact. Any real gap means the features describe a DIFFERENT group than the
# context does -- the mis-attribution ``group_features`` calls "unrecoverable downstream".
_SOLVE_RATE_TOLERANCE: float = 1e-6


class InconsistentFeatures(ValueError):
    """Raised when a context's features contradict the context, or themselves.

    Distinct from :class:`~selfevo.routing.contextual.MissingFeatures` (a feature is
    absent) because the two have different causes and different fixes: absent means the
    producer was never wired, contradictory means it was wired to the wrong unit. Both are
    raised rather than defaulted, because a router that quietly substitutes a value reports
    an arm that never ran.
    """


@dataclass(frozen=True)
class RulePolicyRouter:
    """The M9 hand-written policy: deterministic, feature-consuming, measurement-grounded.

    Args:
        feature_names: Features required to be present in ``ctx.extra``, defaulting to the
            full :data:`selfevo.observability.FEATURE_NAMES`. Required, not merely offered:
            this router is the baseline for a controller that reads all seven, so it must
            fail on exactly the contexts that controller fails on. A run whose observability
            never arrived must not be able to silently produce a "rule" arm.
        solved_mode: Mode for a silent group that solved every sample. Defaults to
            ``SKIP`` because the free self-target was measured **inert at 0.5 and harmful at
            2.0** -- the only two operating points measured. Set to ``sft`` to run GOAL.md's
            critical-path item 1 (SFT on a unit's own correct sample versus SKIP) as an
            ablation of this same router rather than as a different one.
        teacher_mode: Mode for a silent group that solved nothing, when a target exists.
            Must be a mode that actually requires a teacher, or routing the unsolved side to
            it would not supply the target that side is missing.
        truncated_threshold: Fraction of a group's samples that must have hit the token
            budget before the group is proposed to the harness. Defaults to 1.0.
            **This number is not pinned by a measurement and the docstring says so rather
            than inventing one.** 1.0 is the only value at which *every* sample in the group
            failed to terminate, so the group's failure mode is unambiguous -- the same
            guarantee-preserving reasoning :class:`~selfevo.routing.harness.CoHarnessRouter`
            documents for its 1.0/0.0 defaults. What IS measured is the branch's premise:
            doubling the OlympiadBench budget moved truncation 79 -> 78, so truncation is
            non-termination rather than a budget shortfall. Lowering the threshold trades
            the unambiguity for reach and should be justified by a measurement when it is.

    Raises:
        ValueError: If ``feature_names`` omits any of :data:`READ_FEATURES` -- an empty
            tuple included -- because the rule would then read a feature it never checked
            for and take a decision on a value that was never validated. If ``solved_mode`` or
            ``teacher_mode`` is unregistered, if ``teacher_mode`` does not require a teacher,
            if ``solved_mode`` is ``rl`` (whose gradient is identically zero on that branch,
            so it is ``SKIP`` with extra compute under a different name in the logs), or if
            ``truncated_threshold`` is outside [0, 1].
    """

    feature_names: tuple[str, ...] = FEATURE_NAMES
    solved_mode: str = TrainingMode.SKIP
    teacher_mode: str = TrainingMode.SFT
    truncated_threshold: float = 1.0

    def __post_init__(self) -> None:
        # No separate emptiness check: an empty tuple cannot contain READ_FEATURES, so the
        # next guard is the one that fires, and a guard that can never fire reads as
        # protection while providing none.
        missing = [f for f in READ_FEATURES if f not in self.feature_names]
        if missing:
            raise ValueError(
                f"feature_names is missing {missing}, which this rule READS; a feature the "
                "rule reads but never requires would be substituted or absent at decision "
                f"time. Required subset: {list(READ_FEATURES)}"
            )
        modes = known_modes()
        for name, mode in (
            ("solved_mode", self.solved_mode),
            ("teacher_mode", self.teacher_mode),
        ):
            if mode not in modes:
                raise ValueError(f"unknown {name} {mode!r}; register it first")
        if not modes[self.teacher_mode]:
            raise ValueError(
                f"teacher_mode {self.teacher_mode!r} does not require a teacher; routing "
                "unsolved units to it would not supply the missing target"
            )
        if self.solved_mode == TrainingMode.RL:
            raise ValueError(
                "solved_mode='rl' is SKIP with extra compute: the solved branch is reached "
                "only when every advantage in the group is identically zero, so an RL step "
                "there changes no weight while the logs report an rl group"
            )
        if not 0.0 <= self.truncated_threshold <= 1.0:
            raise ValueError(
                f"truncated_threshold must be in [0, 1], got {self.truncated_threshold}"
            )

    # ------------------------------------------------------------------ features ----

    def _features(self, ctx: RoutingContext) -> dict[str, float]:
        """Read and validate ``ctx.extra``, loudly.

        Args:
            ctx: The unit being routed.

        Returns:
            ``{name: value}`` for every name in ``feature_names``, each finite.

        Raises:
            MissingFeatures: If any named feature is absent. Same exception type the
                contextual router raises, so a caller's handling of the two routers is
                genuinely like-for-like rather than like-for-similar.
            InconsistentFeatures: If a feature is not finite (``group_features`` guarantees
                every field finite, so a NaN means the features did not come from it); if
                ``extra["solve_rate"]`` disagrees with ``ctx.solve_rate`` by more than
                :data:`_SOLVE_RATE_TOLERANCE`, which means the two describe different units;
                or if the group is unanimous in reward (``reward_std == 0``) while its solve
                rate is strictly between 0 and 1, which is arithmetically impossible for a
                real group and would send a half-solved unit down a branch whose entire
                justification is that every sample agreed.
        """
        out: dict[str, float] = {}
        for name in self.feature_names:
            if name not in ctx.extra:
                raise MissingFeatures(
                    f"context is missing feature {name!r}; present: {sorted(ctx.extra)}. "
                    "Populate RoutingContext.extra from selfevo.observability."
                    "group_features -- this router is the baseline for a controller that "
                    "reads all of them, so it refuses a context that controller would too."
                )
            value = float(ctx.extra[name])
            if not math.isfinite(value):
                raise InconsistentFeatures(
                    f"feature {name!r} is {value}; group_features guarantees every field "
                    "finite, so a non-finite value means these features came from "
                    "somewhere else"
                )
            out[name] = value

        if abs(out["solve_rate"] - ctx.solve_rate) > _SOLVE_RATE_TOLERANCE:
            raise InconsistentFeatures(
                f"extra['solve_rate']={out['solve_rate']} disagrees with "
                f"ctx.solve_rate={ctx.solve_rate}; the features describe a different unit "
                "than the context does, and one group's decision would be taken on "
                "another group's evidence"
            )
        if out["reward_std"] == 0.0 and out["solve_rate"] not in (0.0, 1.0):
            raise InconsistentFeatures(
                f"reward_std is 0 but solve_rate is {out['solve_rate']}: a group whose raw "
                "rewards are all equal is graded all-correct or all-wrong, never partly. "
                "Both silent branches assume every sample agreed, so this unit belongs to "
                "neither"
            )
        return out

    # -------------------------------------------------------------------- deciding ----

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        """Apply the rule to one unit.

        Args:
            ctx: The unit to route. ``ctx.extra`` must carry ``feature_names``.

        Returns:
            A one-hot :class:`~selfevo.routing.base.RoutingDecision`, with
            ``harness=PROPOSE`` only for a fully-truncated unsolved unit in a run that has a
            harness arm. The decision is a pure function of ``ctx``: no RNG, no state, no
            dependence on how many units were routed before it.

        Raises:
            ValueError: If ``ctx.granularity`` is TOKEN. Checked FIRST, before the features,
                because it is a statement about what kind of unit this is rather than about
                what arrived with it: at TOKEN granularity ``has_self_target`` is forced
                False by :class:`~selfevo.routing.base.RoutingContext`, so a fully-solved
                token would be classified as having no target and silently take the unsolved
                branch. Complaining about its features instead would report the wrong problem.
            MissingFeatures, InconsistentFeatures: See :meth:`_features`.
        """
        if ctx.granularity is Granularity.TOKEN:
            raise ValueError(
                "RulePolicyRouter needs a group mean: at TOKEN granularity has_self_target "
                "is forced False, so a solved unit would silently take the unsolved branch. "
                "Route tokens with selfevo.routing.token_level instead"
            )
        feats = self._features(ctx)

        std = feats["reward_std"]
        if std > 0.0:
            # Identity, not a threshold: A_i = r_i - rbar is non-zero somewhere iff the raw
            # rewards differ. See the module docstring on why this is keyed on reward_std
            # and not on solve_rate.
            return RoutingDecision(
                {TrainingMode.RL: 1.0}, reason=f"informative: reward_std={std:.4f} > 0"
            )

        if ctx.has_self_target:
            # Solved: RL is provably dead and the group's own correct sample is a target.
            # SKIP by default -- measured inert at 0.5, harmful at 2.0.
            return RoutingDecision(
                {self.solved_mode: 1.0},
                reason=f"solved and silent: {self.solved_mode} (self-target available)",
            )

        # Unsolved: RL is dead and no sample can serve as a target.
        trunc = feats["truncated_fraction"]
        propose = trunc >= self.truncated_threshold
        # has_target, not has_teacher: they coincide on this branch because solve_rate == 0
        # admits no self-target, and using has_target keeps the guard identical in shape to
        # every other router here.
        if ctx.has_target:
            mode, why = self.teacher_mode, "unsolved and silent: external target available"
        else:
            mode, why = TrainingMode.SKIP, "unsolved and silent: no target of any kind"

        harness = HarnessAction.NONE
        if propose:
            if ctx.can_evolve_harness:
                harness = HarnessAction.PROPOSE
                why = f"{why}; truncated_fraction={trunc:.2f} -> harness proposal"
            else:
                # Same guard shape as CoHarnessRouter: a run with no harness arm behaves
                # exactly as it would without this axis, and says so in the reason.
                why = (
                    f"{why}; truncated_fraction={trunc:.2f} but harness action dropped "
                    "(no harness arm)"
                )
        return RoutingDecision({mode: 1.0}, reason=why, harness=harness)
