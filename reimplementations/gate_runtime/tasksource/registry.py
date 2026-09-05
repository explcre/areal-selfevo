"""The ONE novelty buffer and the contamination filter, shared across every source.

Novelty is scored against a single buffer spanning all sources rather than one buffer per
source. That is the whole reason the layer exists as a layer: a retrieved problem that is a
near-duplicate of an already-generated one has to lose to it, and three buffers with one name
would silently let both through while reporting a novelty rate for each.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .similarity import most_similar, similarity


@dataclass
class SharedNoveltyBuffer:
    """Every accepted task, from every source, in one place.

    Attributes:
        threshold: Similarity at or above which a candidate is rejected as a near-duplicate.
        texts: Accepted statements, in acceptance order.
        owners: The source that contributed each accepted statement, so cross-source
            collisions can be reported rather than merely counted.
    """

    threshold: float = 0.60
    texts: list[str] = field(default_factory=list)
    owners: list[str] = field(default_factory=list)

    def check(self, text: str) -> tuple[bool, float, str | None]:
        """Would this be accepted?

        Returns:
            ``(accept, similarity, owner_of_nearest)``. The owner is returned so a rejection
            can say WHICH source already held the near-duplicate, which is what makes a
            cross-source collision visible instead of just a lower rate.
        """
        if not self.texts:
            return True, 0.0, None
        sim, at = most_similar(text, self.texts)
        return sim < self.threshold, sim, (self.owners[at] if at >= 0 else None)

    def add(self, text: str, owner: str) -> None:
        """Record an accepted task against its source."""
        self.texts.append(text)
        self.owners.append(owner)

    def novelty(self, text: str) -> float:
        """Ornith's `N(q) = 1 - max_j sim(q, q_j)`, over the shared buffer."""
        if not self.texts:
            return 1.0
        sim, _ = most_similar(text, self.texts)
        return 1.0 - sim


@dataclass
class ContaminationFilter:
    """Rejects any candidate that overlaps the HELD-OUT half.

    This is the serious threat from retrieval and it fails in the opposite direction to a
    wrong key: a memorised problem reads as EASY, so it corrupts the difficulty statistic
    downward while a wrong key corrupts it upward, and an item overlapping the report half
    destroys the evaluation rather than merely biasing it.

    The threshold is deliberately lower than the novelty threshold. A false reject costs one
    candidate; a false accept costs the held-out set, and there is no way to notice afterwards.
    """

    held_out: list[str]
    threshold: float = 0.45

    def check(self, text: str) -> tuple[bool, float, int]:
        """Returns ``(accept, similarity, index_of_nearest_held_out_item)``."""
        if not self.held_out:
            return True, 0.0, -1
        sim, at = most_similar(text, self.held_out)
        return sim < self.threshold, sim, at
