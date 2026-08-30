"""Meta-critics: they judge whether a critic's predictions are worth acting on.

The motivating measurement is in ``experiments/EXPERIMENTS.md``. Six checkpoints from one
GRPO run were ranked by their GSM8K training reward and by held-out MATH-500. The training
reward did not merely fail to track capability -- it ordered the checkpoints *wrongly*,
putting the third-best first and the fourth-best last. A signal that mis-orders is worse
than one that is silent, because a policy acting on it moves confidently in the wrong
direction.

So the question a meta-critic must answer is not "is the critic accurate" but "does the
critic's ordering agree with outcomes better than chance". That is the AUC of the critic's
score as a ranker of outcomes, which separates the three cases that matter:

    AUC > 0.5   the ordering carries signal
    AUC = 0.5   the ordering is noise; acting on it is a coin flip
    AUC < 0.5   the ordering is inverted; acting on it is worse than ignoring it

An accuracy- or calibration-based statistic cannot make that last distinction, which is the
one that actually bit this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from .critics import CriticScore


class CalibrationVerdict(Enum):
    """What the meta-critic concluded about a critic's ordering."""

    INFORMATIVE = "informative"
    UNINFORMATIVE = "uninformative"
    MIS_ORDERING = "mis_ordering"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class CalibrationReport:
    """The meta-critic's assessment of one batch of critic predictions.

    Attributes:
        auc: Probability that a randomly chosen successful unit was scored above a randomly
            chosen unsuccessful one, ties counting a half. ``None`` when undefined.
        verdict: See :class:`CalibrationVerdict`.
        n_scored: Critic scores supplied.
        n_paired: Scores that had a matching outcome and were actually used.
        n_dropped_coarse: Scores excluded because the critic flagged them too noisy to rank.
        n_unpaired: Scores with no matching outcome. Reported, never silently ignored --
            a critic that scores units nobody observes is a failure mode of its own.
        basis: Human-readable account of what the verdict rests on.
    """

    auc: float | None
    verdict: CalibrationVerdict
    n_scored: int
    n_paired: int
    n_dropped_coarse: int
    n_unpaired: int
    basis: str

    def __post_init__(self) -> None:
        if self.auc is not None and not 0.0 <= self.auc <= 1.0:
            raise ValueError(f"auc must be in [0, 1] or None, got {self.auc}")
        if self.verdict is CalibrationVerdict.INSUFFICIENT and self.auc is not None:
            raise ValueError("an INSUFFICIENT verdict must not carry an AUC")
        if self.verdict is not CalibrationVerdict.INSUFFICIENT and self.auc is None:
            raise ValueError(f"verdict {self.verdict} requires an AUC")
        if not self.basis:
            raise ValueError("basis must not be empty")


def _auc(scores: Sequence[float], outcomes: Sequence[bool]) -> float:
    """Rank-based AUC with tied scores contributing 0.5, computed via mid-ranks.

    Counting ties as half is what makes a constant scorer come out at exactly 0.5 rather
    than 0.0 or 1.0 depending on comparison direction. A critic that returns the same value
    for everything is uninformative, not perfect and not inverted, and the statistic has to
    say so.
    """
    pos = [s for s, o in zip(scores, outcomes) if o]
    neg = [s for s, o in zip(scores, outcomes) if not o]
    if not pos or not neg:
        raise ValueError("AUC needs at least one successful and one unsuccessful outcome")
    # Mid-ranks over the pooled sample; equivalent to the Mann-Whitney U formulation.
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = mid
        i = j + 1
    rank_sum = sum(r for r, o in zip(ranks, outcomes) if o)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@dataclass
class OutcomeCalibratedMetaCritic:
    """Judges a critic by whether its scores rank observed outcomes better than chance.

    Args:
        min_paired: Fewest score/outcome pairs that will support any verdict. Below this the
            report is INSUFFICIENT. AUC on a handful of pairs is dominated by sampling
            noise, and the whole point of this class is to refuse confident nonsense.
        margin: How far AUC must sit from 0.5 to count as informative or mis-ordering.
            Inside ``0.5 +/- margin`` the verdict is UNINFORMATIVE.
        use_coarse: When False (the default) scores the critic flagged ``coarse`` are
            dropped before scoring. The critic sets that flag when a single group of G
            samples cannot support a ranking, and ranking on them is exactly the error the
            flag exists to prevent.
    """

    min_paired: int = 20
    margin: float = 0.05
    use_coarse: bool = False

    def __post_init__(self) -> None:
        if self.min_paired < 2:
            raise ValueError("min_paired must be at least 2; AUC needs both outcomes")
        if not 0.0 <= self.margin < 0.5:
            raise ValueError(f"margin must be in [0, 0.5), got {self.margin}")

    def assess(
        self,
        scores: Sequence[CriticScore],
        outcomes: Mapping[str, bool],
    ) -> CalibrationReport:
        """Compare the critic's ordering against observed outcomes.

        Args:
            scores: Critic scores. Any with ``unit_id`` None cannot be paired and are
                counted as unpaired rather than dropped silently.
            outcomes: Observed success per ``unit_id``. Should come from a held-out
                measurement; pairing a critic against the same signal that trained it
                measures agreement, not correctness.

        Returns:
            A :class:`CalibrationReport`. Every path that cannot support a verdict returns
            INSUFFICIENT with a basis naming the reason, rather than a default AUC.
        """
        n_scored = len(scores)
        usable, n_coarse, n_unpaired = [], 0, 0
        for s in scores:
            if not self.use_coarse and s.coarse:
                n_coarse += 1
                continue
            if s.unit_id is None or s.unit_id not in outcomes:
                n_unpaired += 1
                continue
            usable.append((s.value, outcomes[s.unit_id]))

        def report(auc, verdict, basis):
            return CalibrationReport(
                auc=auc, verdict=verdict, n_scored=n_scored, n_paired=len(usable),
                n_dropped_coarse=n_coarse, n_unpaired=n_unpaired, basis=basis,
            )

        if len(usable) < self.min_paired:
            return report(None, CalibrationVerdict.INSUFFICIENT,
                          f"only {len(usable)} paired score(s), need {self.min_paired}")
        vals = [v for v, _ in usable]
        outs = [o for _, o in usable]
        if all(outs) or not any(outs):
            return report(None, CalibrationVerdict.INSUFFICIENT,
                          "all outcomes identical; ranking cannot be tested against a "
                          "constant. This is a property of the sample, not of the critic")

        auc = _auc(vals, outs)
        if len(set(vals)) == 1:
            # Guaranteed 0.5 by the tie rule, but say why rather than implying discrimination.
            return report(auc, CalibrationVerdict.UNINFORMATIVE,
                          f"critic returned the identical value {vals[0]:.4f} for all "
                          f"{len(usable)} units; it cannot order anything")
        if auc >= 0.5 + self.margin:
            v, why = CalibrationVerdict.INFORMATIVE, "ranks outcomes better than chance"
        elif auc <= 0.5 - self.margin:
            v, why = (CalibrationVerdict.MIS_ORDERING,
                      "ranks outcomes INVERSELY; acting on this critic is worse than "
                      "ignoring it, the failure mode measured for GSM8K train reward")
        else:
            v, why = (CalibrationVerdict.UNINFORMATIVE,
                      f"within {self.margin} of chance")
        return report(auc, v, f"AUC={auc:.4f} over {len(usable)} pairs: {why}")
