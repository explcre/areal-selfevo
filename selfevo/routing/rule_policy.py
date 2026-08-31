"""M9: the rule evolve-policy -- a hand-written baseline, and what building it revealed.

**RETRACTION, 2026-08-31, after an adversarial audit of the first version.** That version
claimed this router removed a confound: that by deciding from the SAME seven observability
features the learned controller reads, it separated "written vs learned" from
"1 feature vs 7". **The claim is false and is withdrawn.** Under this project's graders this
router is behaviourally IDENTICAL to the :class:`~selfevo.routing.routers.SolveRateRouter`
it was meant to replace.

Measured, not argued. Over every binary group composition -- ``G`` in {2, 4, 8, 16} x ``k``
solved in 0..G x ``truncated_fraction`` in {0, 0.5, 1} -- the two routers agree on
**102/102** contexts with no teacher and **102/102** with one. The mechanism is this
module's own inertness argument turned around: for a BINARY reward, ``reward_std > 0`` and
"the group was not unanimous" are the same predicate, and every reward function in this repo
(``areal/reward/gsm8k.py``, ``boba_grpo.py``) returns exactly 1.0 or 0.0. The equivalence is
pinned by :func:`selfevo.tests.test_rule_policy.test_the_rule_is_equivalent_to_solve_rate_on_binary_rewards`
so that it is a checked invariant rather than a claim, and so that either router drifting
from the other is a test failure rather than a silent second arm.

"Consumes all seven features" therefore survives only in the weak sense of REQUIRING their
presence. Sweeping the other six leaves the mode at ``rl``; only ``reward_std`` moves it.

**Why this was not repaired by adding branches, and why that is the actual finding.** The
obvious repair is a second branch keyed on ``mean_logprob`` or ``len_dispersion``. This
project's standard is that every threshold cites a measurement, and no measurement here
grounds one -- GOAL.md M15 records precisely that gap ("no ablation showing which features
carry the decision"). Inventing a number so the baseline looks richer is the failure this
repo has caught in other guises. So the honest result is:

    A defensible hand-written policy over these seven features collapses to ONE predicate,
    because only one of the seven has a measurement behind it. The "1 feature vs 7" confound
    in a rule-vs-learned comparison is **not removable by writing a better rule**. It is a
    property of the evidence available, and any such comparison has to be reported carrying
    it.

**Where the two routers DO diverge, and why neither divergence is live today.**

* A non-binary grader. Rewards ``[1.0, 0.8, 1.0, 0.8]`` grade all-correct while the
  advantages are +-0.1 and the gradient is real: this router sends them to RL,
  ``SolveRateRouter`` (whose ``I_RL`` is computed from ``solve_rate == 1.0``) skips them.
  No grader in this repo is non-binary, so the divergence is latent.
* The harness axis. ``SolveRateRouter`` has none. This one emits
  :class:`~selfevo.routing.base.HarnessAction.PROPOSE` -- but **that action cannot fire in
  any current run and cannot be observed if it did**: nothing writes
  ``ctx.can_evolve_harness``, and ``actor._route_groups`` keeps only ``.argmax()``, dropping
  ``.harness`` on the floor. This is the same "no consumer" gap GOAL.md's M10 row states,
  and it is stated here rather than left for a reader to discover.

What remains genuinely different, and the reason this ships rather than reverting the slot
to ``SolveRateRouter``, is narrow and worth naming precisely: each branch here is cited to
the measurement that justifies it and is individually mutation-covered, the decision
boundary is the silence condition itself rather than an ``I_RL`` threshold that
:func:`selfevo.routing.criteria.threshold_is_inert` proves cannot change a decision
(0.68 at G=4, 0.66 at G=8, 0.64 at G=16, ``threshold_is_inert`` True at all three), the
context contract is the learned router's (a missing feature raises rather than defaults),
and ``solved_mode`` is the seam for GOAL.md's critical-path item 1. None of that is a
behavioural difference under today's config, and it must not be reported as one.

The rule, one branch at a time, with the measurement each rests on:

silent test: ``reward_std > _UNANIMITY_EPS``
    GRPO's centred advantage is ``A_i = r_i - rbar``, so in exact arithmetic a group whose
    raw rewards are not all equal has a non-zero advantage somewhere. In **float32 it is not
    an identity**, which the first version of this module wrongly claimed it was: a
    unanimous group of 8 rewards of 0.8 reduces to ``reward_std = 5.96e-08``, and a bare
    ``> 0`` test routed it to RL while printing "reward_std=0.0000 > 0".
    :data:`_UNANIMITY_EPS` is the measured fix.

``reward_std`` above that  ->  ``RL``
    The group disagrees; RL carries signal.

silent and ``solve_rate == 1``  ->  ``SKIP`` (configurable)
    RL is dead and the group carries its own SFT target. Still skipped by default because
    the free self-target was **measured inert at 0.5 and harmful at 2.0** (EXPERIMENTS.md
    2026-08-31; GOAL.md M24). Those are the only operating points ever measured for it.
    GOAL.md critical-path item 1 is the A/B that would revise this, and ``solved_mode="sft"``
    runs it through this same object.

silent and ``solve_rate == 0``  ->  teacher mode if one exists, else ``SKIP``
    No self-target by construction: every sample was wrong (GOAL.md M24). No run wires an
    external teacher, so in practice this branch is ``SKIP``.

``truncated_fraction >= 1``, on the unsolved branch  ->  ``HarnessAction.PROPOSE``
    Grounded in a real measurement -- OlympiadBench, same checkpoint, cap 8192 -> 16384 moved
    truncation 79 -> 78 for ``ctx`` and 61 -> 64 for ``rnd``, and ``n_truncated ==
    n_no_box`` in every MATH/AMC/AIME row, so a truncated sample never terminates usefully
    rather than being cut off -- but **unreachable today**, per the harness-axis note above.
    It is a wired prediction awaiting a consumer, not shipped behaviour.

**Cold start.** A rule seeding a learned router must not hand ``batch_outcomes`` a
single-mode batch, which is refused as
:class:`~selfevo.routing.feedback.ConfoundedUpdate` (GOAL.md, "Feedback for a learned router
is only defined when the batch has mode diversity"). On the matched GSM8K pair
``silent_group_fraction`` is 0.5906 at G=8 and 0.4553 at G=16, so both branches are
populated and a real batch carries both ``rl`` and ``skip``. Not a guarantee -- an
all-informative batch is still refused -- and ``feedback/confounded_skips`` is the
diagnostic.

**Determinism.** Frozen dataclass, no RNG, no state between calls, no order dependence. Same
context, same decision, always.

**Not a matched-proportion comparison.** Under the shipped configuration it emits only
``rl`` and ``skip``, against the contextual arm's measured rl 0.295 / sft 0.353 / skip 0.353.
Each arm still needs its own ``router=random`` control at its own measured proportions.
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

# The features this rule READS, as opposed to the ones it requires to be present. Of these
# three, only `reward_std` can change the MODE under the shipped configuration: `solve_rate`
# picks between the two silent branches (both SKIP by default), and `truncated_fraction`
# gates a harness action that has no consumer. Stated as a constant so the claim can be
# checked without reading `route`, and so the audit finding it encodes -- a defensible rule
# here needs one predicate -- is visible at the top of the file rather than buried.
READ_FEATURES: tuple[str, ...] = ("solve_rate", "reward_std", "truncated_fraction")

# Below this, a group's reward dispersion is float32 rounding rather than signal.
#
# MEASURED, both bounds, because "measure the correct implementation's drift against the
# smallest real error" is the only way to choose a tolerance that is not a guess:
#
#   noise floor  -- the largest ``reward_std`` a UNANIMOUS group produces. Swept over G in
#                   {2, 4, 8, 16, 32, 64} and 27 reward values in [0, 1]: **1.192e-07**
#                   (one ULP at 1.0), worst at G=8, value 0.99. G=2 and G=4 leak nothing;
#                   21 of 27 values leak at every G >= 8. So the residue is not exotic --
#                   it is the common case at the production group sizes.
#   real signal  -- the smallest dispersion this project would be wrong to delete. For a
#                   BINARY group that is ``sqrt((1/G)(1-1/G))``: 0.43 at G=4, 0.33 at G=8,
#                   0.24 at G=16. For a partial-credit group the adversarial case found in
#                   audit is ``[1.0, 0.99]`` at **5.0e-03**.
#
# Any value in (1.192e-07, 5.0e-03) therefore decides identically on every group the
# graders in this repo can produce. 1e-6 is chosen inside it with ~8x margin above the
# noise and ~5000x below the signal, and
# ``test_the_unanimity_epsilon_sits_between_the_measured_noise_floor_and_real_signal``
# recomputes both bounds so the constant cannot drift out of the band unnoticed.
_UNANIMITY_EPS: float = 1e-6

# How far ``ctx.extra["solve_rate"]`` may sit from ``ctx.solve_rate`` before the context is
# refused. Both are produced by the same float32 reduction in
# ``observability.group_features`` and copied into the two places by ``actor._route_groups``,
# so the only difference either can legitimately carry is the float32 -> float64 widening,
# which is exact. Any real gap means the features describe a DIFFERENT group than the
# context does -- the mis-attribution ``group_features`` calls "unrecoverable downstream".
_SOLVE_RATE_TOLERANCE: float = 1e-6


def _is_silent(reward_std: float) -> bool:
    """Whether a group's rewards are unanimous to within float32 rounding.

    Args:
        reward_std: The group's population reward standard deviation, from
            :class:`selfevo.observability.GroupFeatures`.

    Returns:
        True when the dispersion is at or below :data:`_UNANIMITY_EPS`, i.e. when RL's
        centred advantage is zero or indistinguishable from zero.

    Why a function rather than an inline comparison: the same predicate decides the routing
    branch AND the impossible-group guard in :meth:`RulePolicyRouter._features`. Written
    twice they would drift, and the drift would put a group on a branch whose precondition
    the guard had already rejected.
    """
    return reward_std <= _UNANIMITY_EPS


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
    """The M9 hand-written policy: deterministic, and grounded branch by branch.

    Behaviourally identical to :class:`~selfevo.routing.routers.SolveRateRouter` under this
    repo's binary graders (102/102 contexts, measured) -- see the module docstring, which
    retracts the "decides from seven features" framing the first version claimed.

    Args:
        feature_names: Features required to be PRESENT in ``ctx.extra``, defaulting to the
            full :data:`selfevo.observability.FEATURE_NAMES`. Presence, not use: only
            :data:`READ_FEATURES` are read, and of those only ``reward_std`` ever changes a
            mode under the shipped configuration. They are required anyway so this router
            fails on exactly the contexts the learned controller fails on -- a run whose
            observability never arrived must not be able to produce a quiet "rule" arm --
            but that is an input-contract guarantee, not evidence of a richer policy.
        solved_mode: Mode for a silent group that solved every sample. Defaults to
            ``SKIP`` because the free self-target was measured **inert at 0.5 and harmful at
            2.0** -- the only two operating points measured. Set to ``sft`` to run GOAL.md's
            critical-path item 1 (SFT on a unit's own correct sample versus SKIP) as an
            ablation of this same router rather than as a different one.
        teacher_mode: Mode for a silent group that solved nothing, when a target exists.
            Must be a mode that actually requires a teacher, or routing the unsolved side to
            it would not supply the target that side is missing.
        truncated_threshold: Fraction of a group's samples that must have hit the token
            budget before the group is proposed to the harness. Defaults to 1.0. Note the
            harness action it gates **cannot fire in any current run**: nothing writes
            ``ctx.can_evolve_harness`` and ``actor._route_groups`` discards ``.harness``.
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
        if _is_silent(out["reward_std"]) and out["solve_rate"] not in (0.0, 1.0):
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
        if not _is_silent(std):
            # In exact arithmetic this is the identity A_i = r_i - rbar != 0; in float32 it
            # is that identity plus a MEASURED rounding tolerance, because a unanimous group
            # of eight 0.8s reduces to 5.96e-08 and a bare `> 0` sent it to RL. See
            # _UNANIMITY_EPS.
            return RoutingDecision(
                {TrainingMode.RL: 1.0},
                reason=f"informative: reward_std={std:.3e} > {_UNANIMITY_EPS:.0e}",
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
