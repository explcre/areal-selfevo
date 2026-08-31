"""Produce the outcomes a learned router learns from, with the confound made explicit.

``DecisionOutcome`` and ``observe()`` have existed since the feedback channel was designed,
and nothing has ever produced them: the channel had a consumer and no producer. This is the
producer, and it is deliberately the weakest honest one.

**The confound is the situation, not a flaw in this module.** After an update we observe one
scalar for the whole batch, produced by every decision in it. Attributing that scalar to each
unit is credit assignment with no counterfactual, and no arithmetic here can create the
missing information. What this module can do is refuse the cases where the attribution is
provably vacuous, and record how strong the attribution was so a result can be discounted by
it.

Two refusals, both raising rather than returning a degraded outcome:

* a batch in which every unit got the SAME mode carries no comparative information at all --
  the scalar would credit that mode for whatever the batch did, and repeated, would make the
  first mode tried look best.
* a batch whose scalar is not finite.

A third case is reported rather than refused: when one mode dominates a batch, the attribution
is weak but not empty, and ``AttributionStrength`` says how weak.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from selfevo.routing.base import known_modes
from selfevo.routing.feedback import ConfoundedUpdate, DecisionOutcome

__all__ = ["AttributionStrength", "batch_outcomes"]


@dataclass(frozen=True)
class AttributionStrength:
    """How much comparative information a batch's outcomes actually carry.

    Args:
        n_modes: Distinct modes present in the batch.
        dominant_share: Fraction of units taking the most common mode. At 1.0 the batch is
            single-mode and carries nothing; near 0.5 with two modes it is as informative as
            a batch of this size can be.
        n_units: Units credited.
    """

    n_modes: int
    dominant_share: float
    n_units: int

    @property
    def is_weak(self) -> bool:
        """True when one mode takes more than 90% of the batch."""
        return self.dominant_share > 0.9

    def as_metrics(self) -> dict[str, float]:
        """Flat metrics, so a run records how attributable its own updates were."""
        return {
            "feedback/n_modes": float(self.n_modes),
            "feedback/dominant_share": float(self.dominant_share),
            "feedback/n_units": float(self.n_units),
            "feedback/weak_attribution": 1.0 if self.is_weak else 0.0,
        }


def batch_outcomes(
    unit_modes: dict[str, str],
    *,
    batch_id: str,
    value: float,
    costs: dict[str, float] | None = None,
) -> tuple[dict[str, DecisionOutcome], AttributionStrength]:
    """Attribute one batch-level scalar to every unit in the batch.

    Args:
        unit_modes: ``{unit_id: mode}`` actually applied this batch.
        batch_id: Identifier for this update. Outcomes from different batches are not
            comparable, and ``DecisionOutcome`` requires it for that reason.
        value: The batch-level scalar to credit. Sign convention is the caller's and must be
            consistent: higher must always mean better, or the router learns the negation.
        costs: Optional ``{unit_id: cost}``. Defaults to 1.0. A router that optimises value
            per unit cost needs this, because SKIP is nearly free and would otherwise never
            look attractive.

    Returns:
        ``({unit_id: DecisionOutcome}, AttributionStrength)``.

    Raises:
        ConfoundedUpdate: If the batch is empty or every unit took the same mode -- there is
            nothing to compare, and crediting the single mode would make whichever mode was
            tried first look best.
        ValueError: If a mode is unregistered, or ``value`` is not finite.
    """
    if not unit_modes:
        raise ConfoundedUpdate("no units in this batch; there is nothing to attribute")
    for uid, mode in unit_modes.items():
        if mode not in known_modes():
            raise ValueError(f"unit {uid!r} has unknown mode {mode!r}")
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"value must be finite, got {value}")

    counts = Counter(unit_modes.values())
    if len(counts) < 2:
        only = next(iter(counts))
        raise ConfoundedUpdate(
            f"every unit in batch {batch_id!r} took mode {only!r}; a batch-level scalar "
            "carries no comparative information, and crediting it would make whichever mode "
            "was tried first look best"
        )

    n = len(unit_modes)
    strength = AttributionStrength(
        n_modes=len(counts),
        dominant_share=counts.most_common(1)[0][1] / n,
        n_units=n,
    )
    costs = costs or {}
    out = {
        uid: DecisionOutcome(
            mode=mode,
            value=float(value),
            batch_id=batch_id,
            unit_id=uid,
            cost=float(costs.get(uid, 1.0)),
        )
        for uid, mode in unit_modes.items()
    }
    return out, strength
