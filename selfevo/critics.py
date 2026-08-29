"""Critics: cheap predictions of an item's training value, made before training on it.

The design follows BigBang's stated purpose for a critic -- *"a fast proxy for the training
value of synthetic data"*, used because *"repeated training and evaluation are costly and
time-consuming"* -- but the implementation is ours; their released repository is
evaluation-only and contains no critic code.

**What this predicts, stated precisely, because the distinction is the whole point of
having a meta-critic.** :class:`ScalarCritic` scores an item by ``I_RL(p_hat, G)``, which is
the probability that the item's group produces a non-zero gradient. That is *necessary* for
training value and *not sufficient*: an item can yield a large, well-signed gradient and
still teach nothing useful. So the critic's output is a prediction of **gradient
informativeness**, not of downstream capability gain, and treating the two as the same is
the error a meta-critic exists to detect. Every score therefore records what it was based
on, so the gap can later be measured rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .routing.base import RoutingContext
from .routing.criteria import SilenceSide, rl_informativeness, silence_side

__all__ = ["CriticScore", "ScalarCritic"]


@dataclass(frozen=True)
class CriticScore:
    """One critic judgement.

    Args:
        value: Predicted training value in [0, 1]. Higher is better.
        basis: What the prediction is actually derived from. Recorded because the critic
            predicts gradient informativeness while a meta-critic will compare it against
            realised capability gain; without this the comparison would silently conflate
            "the critic was wrong" with "the critic measured a different quantity".
        side: Which side of RL silence the item falls on, or INFORMATIVE.
        unit_id: Optional identifier, for pairing with a later outcome.
    """

    value: float
    basis: str
    side: SilenceSide
    unit_id: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"value must be in [0, 1], got {self.value}")
        if not self.basis:
            raise ValueError("basis must be non-empty; an unattributed score cannot be calibrated")


@dataclass
class ScalarCritic:
    """Scores an item by the probability its group yields a non-zero gradient.

    Args:
        solved_penalty: Multiplier applied to items the model already solves. RL is silent
            there too, but for the opposite reason -- there is nothing left to learn -- so
            they must not receive the same score as unsolved items despite both having
            ``I_RL`` near zero. Defaults to 0.0: already-solved items have no training
            value, whatever a symmetric criterion would say.
        unsolved_floor: Score given to items the model cannot solve. These are worth
            something only if a teacher target exists, so the caller supplies
            ``has_teacher`` in the context and the floor applies only then.
        threshold: Informativeness below which an item counts as silent.

    Raises:
        ValueError: If any parameter is outside [0, 1].
    """

    solved_penalty: float = 0.0
    unsolved_floor: float = 0.5
    threshold: float = 0.1
    _history: list[CriticScore] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("solved_penalty", "unsolved_floor", "threshold"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")

    def score(self, ctx: RoutingContext, unit_id: str | None = None) -> CriticScore:
        """Predict the training value of the item described by ``ctx``.

        Args:
            ctx: The unit's observed solve rate, group size and teacher availability.
            unit_id: Optional identifier, carried into the returned score.

        Returns:
            A :class:`CriticScore`. Items the model already solves score
            ``I_RL * solved_penalty`` (0 by default); items it cannot solve score
            ``unsolved_floor`` when a teacher exists and 0 otherwise; everything else
            scores its informativeness directly.
        """
        info = rl_informativeness(ctx.solve_rate, ctx.group_size)
        side = silence_side(ctx.solve_rate, ctx.group_size, self.threshold)

        if side is SilenceSide.SOLVED:
            value, basis = info * self.solved_penalty, "already solved: no gradient, nothing to learn"
        elif side is SilenceSide.UNSOLVED:
            if ctx.has_teacher:
                value, basis = self.unsolved_floor, "unsolved but a teacher target exists"
            else:
                value, basis = 0.0, "unsolved and no teacher: unlearnable now"
        else:
            value, basis = info, f"I_RL={info:.3f} (predicts non-zero gradient, not capability gain)"

        s = CriticScore(value=value, basis=basis, side=side, unit_id=unit_id)
        self._history.append(s)
        return s

    def history(self) -> list[CriticScore]:
        """Every score produced, in order, for later calibration against outcomes."""
        return list(self._history)

    def reset(self) -> None:
        """Clear the recorded history."""
        self._history.clear()
