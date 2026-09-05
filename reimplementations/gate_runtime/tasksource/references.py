"""Three reference sets, three different meanings, three different actions.

Collapsing these into one "duplicate rate" loses the only thing that matters about them --
what an overlap COSTS:

* **held-out** -- an overlap is contamination. It destroys the evaluation and cannot be
  noticed afterwards, so a false accept is unbounded and a false reject costs one candidate.
  Action: REJECT, at a deliberately loose threshold.
* **running buffer** -- an overlap is repetition within the run. The loop would train twice on
  the same task and score its novelty reward against itself. Action: REJECT.
* **training pool** -- an overlap is redundancy. The task teaches nothing new and consumes
  budget a genuinely new task would have used. It is not fatal and it is not incorrect.
  Action: FLAG, and the reason is in `RedundancyIndex`.

Every rate reported from these is meaningless without its own floor, because the similarity
measure has a length-dependent false-positive rate: 0.095 overall and 0.142 on statements under
sixty characters, measured against problems known to be unrelated. A rate quoted alone repeats
the mistake that produced two withdrawn findings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .similarity import most_similar


@dataclass
class ReferenceSet:
    """One set to check candidates against, with the action an overlap triggers."""

    name: str
    texts: list[str]
    threshold: float
    action: str          # "reject" or "flag"
    meaning: str = ""

    def __post_init__(self) -> None:
        if self.action not in ("reject", "flag"):
            raise ValueError("action must be reject|flag, got %r" % self.action)

    def check(self, text: str) -> tuple[float, int]:
        """Highest similarity against this set and the index attaining it."""
        if not self.texts:
            return 0.0, -1
        return most_similar(text, self.texts)


@dataclass
class RedundancyIndex:
    """The training pool, checked for redundancy and deliberately NOT used to reject.

    The decision, and the reason, because it is a judgement rather than a default.
    Contamination is rejected because a false accept destroys the held-out set and a false
    reject costs one candidate: wildly asymmetric, so reject on suspicion. Redundancy is
    symmetric -- a false accept wastes one task's budget, a false reject discards one
    genuinely new task -- and the measure's own false-positive floor is around a tenth. So
    rejecting on it would discard roughly one good task in ten to save roughly one wasted task
    in ten, which is not a trade worth making blind.

    It is therefore FLAGGED and reported against its floor, so that a redundancy rate clearly
    above the floor can be acted on later with a better measure.
    """

    texts: list[str]
    threshold: float = 0.60
    flagged: list[tuple[str, float]] = field(default_factory=list)

    def check(self, text: str) -> tuple[bool, float]:
        """Returns ``(is_redundant, similarity)``. Never rejects; the caller records it."""
        if not self.texts:
            return False, 0.0
        sim, _ = most_similar(text, self.texts)
        red = sim >= self.threshold
        if red:
            self.flagged.append((text[:200], sim))
        return red, sim
