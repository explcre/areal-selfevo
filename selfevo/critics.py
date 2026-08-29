"""Critics: cheap predictions of an item's training value, made before training on it.

Follows BigBang's stated purpose for a critic -- *"a fast proxy for the training value of
synthetic data"* -- but the implementation is ours; their released repository is
evaluation-only and contains no critic code.

**What this predicts.** ``I_RL(p_hat, G)`` is the probability that an item's GRPO group
produces a non-zero gradient. That is *necessary* for training value and *not sufficient*:
an item can yield a large, well-signed gradient and still teach nothing. The score is
therefore a prediction of **gradient informativeness**, not of capability gain, and every
score records its basis so a meta-critic can later tell "the critic was wrong" from "the
critic measured a different quantity".

**Three corrections after an audit of the first version**, each of which had made the score
either meaningless or misleading:

1. *The asymmetry was asserted, not implemented.* The first version classified an item as
   already-solved via ``I_RL < threshold``. But the smallest non-zero attainable
   informativeness is ``I_RL(1/G, G)``, which is >= 0.5 for every ``G >= 2``, so at any
   sane threshold that branch was reachable only at ``p_hat = 1.0`` -- where ``I_RL`` is 0
   and multiplying by a penalty changes nothing. Both ``threshold`` and ``solved_penalty``
   were provably inert, and 6 of 7 interior points at G=8 scored *mirror-symmetrically*: an
   item solved 7/8 scored identically to one solved 1/8. Solved/unsolved is now decided by
   ``p_hat`` directly.
2. *Scores were not comparable across group sizes.* ``I_RL(0.5, G)`` rises from 0.5 at G=2
   to 0.9999 at G=16, so the score was dominated by ``G`` rather than by the item: p=0.5 at
   G=2 (0.500) tied p=0.0 at G=8 and *lost* to p_hat=1/8 at G=8 (0.656). Scores are now
   normalised by the maximum attainable at that ``G``, and ``group_size`` is recorded so a
   ranking caller can refuse a cross-G comparison it cannot justify.
3. *The estimate's uncertainty was invisible.* ``p_hat = k/G`` is coarse, and for an item
   the model mostly solves the score is close to bimodal -- at G=8 and true p=0.9 the audit
   measured E[score]=0.422 with sd=0.379, split between 0.0 (43% of draws) and 0.656 (38%).
   Ranking on that is ranking on noise. :attr:`CriticScore.coarse` now flags it.

**Inherited caveats that apply here and are not repeated in the callers.** ``I_RL(p_hat,G)``
is a *biased* plug-in for ``I_RL(p,G)`` (Jensen; ``I_RL`` is concave in p): measured -24% to
-29% relative at G=4. The floor biases the other way, so the critic's own bias is
non-monotone. And ``criteria.py`` records that ``I_RL`` earns its keep only at task or
cluster granularity; at ``SAMPLE`` granularity it reduces to a unanimity test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .routing.base import Granularity, RoutingContext
from .routing.criteria import SilenceSide, rl_informativeness

__all__ = ["CriticScore", "ScalarCritic"]


@dataclass(frozen=True)
class CriticScore:
    """One critic judgement.

    Args:
        value: Predicted training value in [0, 1], normalised so that scores at different
            group sizes are comparable.
        basis: What the prediction derives from. Load-bearing: a meta-critic compares this
            score against a realised outcome, and without the basis it cannot distinguish
            a wrong prediction from a prediction about a different quantity.
        side: Which side of RL silence the item falls on.
        group_size: The ``G`` this was scored at. Recorded because the normalisation makes
            values comparable but the *precision* still varies with ``G``.
        unit_id: Identifier, carried from the context so the score can be paired with a
            later outcome. This is the whole point of keeping a history.
        coarse: True when the estimate is too noisy to rank on -- a single group of ``G``
            samples resolves ``p`` to +/- ~1/G, and near the extremes the score is close to
            bimodal.
    """

    value: float
    basis: str
    side: SilenceSide
    group_size: int
    unit_id: str | None = None
    coarse: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"value must be in [0, 1], got {self.value}")
        if not self.basis:
            raise ValueError("basis must be non-empty; an unattributed score cannot be calibrated")
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")


@dataclass
class ScalarCritic:
    """Scores an item by normalised gradient informativeness, split by solve rate.

    Args:
        solved_at: ``p_hat`` at or above which an item counts as already solved. Decided on
            the solve rate directly, NOT on an informativeness threshold, because the
            latter is unreachable for interior ``p_hat`` at every group size.
        unsolved_at: ``p_hat`` at or below which an item counts as unlearnable-for-now.
        solved_value: Score for an already-solved item. Defaults to 0: there is nothing left
            to learn, and unlike an unsolved item a teacher does not help.
        unsolved_floor: Score for an unsolved item **when a teacher target exists**;
            0 otherwise. Expressed as a fraction of the maximum attainable informativeness
            so it does not silently outrank informative items at small ``G`` -- at G=2 a
            constant 0.5 equals the maximum, which made the first version constant.
        coarse_below: Group sizes at or below this are flagged coarse. Defaults to 8: the
            audit measured sd=0.379 on a 0-1 score at G=8 for a true p of 0.9.

    Raises:
        ValueError: On out-of-range parameters, or ``unsolved_at >= solved_at``.
    """

    solved_at: float = 1.0
    unsolved_at: float = 0.0
    solved_value: float = 0.0
    unsolved_floor: float = 0.5
    coarse_below: int = 8
    _history: list[CriticScore] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("solved_at", "unsolved_at", "solved_value", "unsolved_floor"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if self.unsolved_at >= self.solved_at:
            raise ValueError(
                f"unsolved_at ({self.unsolved_at}) must be < solved_at ({self.solved_at})"
            )
        if self.coarse_below < 0:
            raise ValueError(f"coarse_below must be >= 0, got {self.coarse_below}")

    def score(self, ctx: RoutingContext, unit_id: str | None = None) -> CriticScore:
        """Predict the training value of the item described by ``ctx``.

        Args:
            ctx: Observed solve rate, group size, teacher availability and granularity.
            unit_id: Overrides ``ctx.unit_id`` when given; otherwise the context's is used.

        Returns:
            A :class:`CriticScore` whose ``value`` is normalised by the maximum attainable
            informativeness at ``ctx.group_size``, so items scored at different group sizes
            are on one scale.

        Raises:
            ValueError: If ``ctx.group_size`` is 1. A singleton group has
                ``A = r - rbar = 0`` identically, so it is degenerate rather than merely
                uninformative -- the same posture ``min_group_size`` takes.
        """
        g = ctx.group_size
        if g < 2:
            raise ValueError(
                "group_size must be >= 2; a singleton group has A = r - rbar identically "
                "zero, so no score over it is meaningful"
            )

        p = ctx.solve_rate
        info = rl_informativeness(p, g)
        ceiling = rl_informativeness(0.5, g)  # max attainable at this G; > 0 for g >= 2
        coarse = g <= self.coarse_below

        if p >= self.solved_at:
            side = SilenceSide.SOLVED
            value = self.solved_value
            basis = f"solved (p_hat={p:.3f} >= {self.solved_at}): no gradient, nothing to learn"
        elif p <= self.unsolved_at:
            side = SilenceSide.UNSOLVED
            if ctx.has_teacher:
                value = self.unsolved_floor
                basis = f"unsolved (p_hat={p:.3f} <= {self.unsolved_at}) but a teacher target exists"
            else:
                value = 0.0
                basis = f"unsolved (p_hat={p:.3f}) and no teacher: unlearnable now"
        else:
            side = SilenceSide.INFORMATIVE
            value = info / ceiling
            basis = (
                f"I_RL={info:.4f}/{ceiling:.4f}={value:.4f} at G={g}; predicts a non-zero "
                "gradient, not capability gain"
            )

        if coarse:
            basis += f" [coarse: G={g} resolves p to about +/-{1.0 / g:.2f}]"
        if ctx.granularity is Granularity.SAMPLE:
            basis += " [sample granularity: I_RL degenerates toward a unanimity test]"

        s = CriticScore(
            value=value, basis=basis, side=side, group_size=g,
            unit_id=unit_id if unit_id is not None else ctx.unit_id, coarse=coarse,
        )
        self._history.append(s)
        return s

    def history(self) -> list[CriticScore]:
        """Every score produced, in order, for later calibration against outcomes."""
        return list(self._history)

    def reset(self) -> None:
        """Clear the recorded history."""
        self._history.clear()
