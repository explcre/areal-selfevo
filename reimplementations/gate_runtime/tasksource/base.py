"""Task records and the one interface every source emits through.

The point of the layer is that everything downstream is identical whatever produced a task:
the same novelty buffer, the same difficulty measurement, the same verifier. A source that
needed special handling downstream would defeat the comparison the layer exists to support.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Protocol


@dataclass(frozen=True)
class Provenance:
    """Where a task came from, carried with it to the artifact.

    Recorded per item rather than per run because a corpus is not a single provenance: a
    retrieved item has a dataset, a row and a licence, and a distilled one has a teacher
    model and a decoding configuration. Dropping this at the first transformation is how
    scraped content gets laundered, so `TaskRecord` refuses to exist without it.
    """

    source: str
    origin: str
    licence: str
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a provenance that does not identify its origin or its licence."""
        if not self.source or not self.origin or not self.licence:
            raise ValueError(
                "provenance needs source, origin and licence; an item whose licence is "
                "unknown must say 'unknown' explicitly rather than carry an empty string, "
                "so that a filter can act on it."
            )


@dataclass
class TaskRecord:
    """One task, whatever produced it.

    Attributes:
        task_id: Stable digest of the statement, so the same statement from two sources
            collides by construction and can be detected.
        text: The problem statement.
        answer: The asserted answer, i.e. the KEY whose correctness is in question.
        provenance: Required; see :class:`Provenance`.
    """

    text: str
    answer: str
    provenance: Provenance
    task_id: str = ""

    def __post_init__(self) -> None:
        """Fill the id from the statement and refuse an empty task."""
        if not self.text or not self.text.strip():
            raise ValueError("a task needs a statement")
        if self.answer is None or not str(self.answer).strip():
            raise ValueError(
                "a task needs an asserted answer. A task with no key cannot be graded by "
                "`boxed_exact`; every rollout would compare false and it would score "
                "p_hat=0 and look ideally difficult (the loop's guard G8)."
            )
        if not isinstance(self.provenance, Provenance):
            raise TypeError("provenance must be a Provenance, not %r" % type(self.provenance))
        if not self.task_id:
            self.task_id = hashlib.sha256(self.text.strip().encode()).hexdigest()[:16]

    def to_json(self) -> dict:
        """Serialisable form, provenance included."""
        d = asdict(self)
        d["provenance"] = asdict(self.provenance)
        return d


@dataclass
class SourceResult:
    """What a source produced, INCLUDING the case where it produced nothing.

    A source that yields zero tasks must be visible as a failure. Returning an empty list
    and letting the caller average over the survivors is the shape that has bitten this
    project repeatedly: the run completes, the table has a row, and the row is silence.
    """

    name: str
    tasks: list[TaskRecord]
    attempted: int
    ok: bool
    reason: str = ""
    cost_tokens: int = 0
    detail: dict = field(default_factory=dict)

    @classmethod
    def failure(cls, name: str, attempted: int, reason: str, cost_tokens: int = 0
                ) -> "SourceResult":
        """Build an explicit failure, which is not the same as an empty success."""
        return cls(name=name, tasks=[], attempted=attempted, ok=False, reason=reason,
                   cost_tokens=cost_tokens)


class TaskSource(Protocol):
    """Produces candidate tasks. The only thing the pipeline knows about a source."""

    name: str

    def fetch(self, n: int, rng) -> SourceResult:
        """Return up to `n` candidate tasks, or an explicit failure.

        Args:
            n: How many candidates are wanted.
            rng: Seeded random source, so a run is reproducible.

        Returns:
            A :class:`SourceResult`. Producing fewer than `n` is allowed; producing none
            must be reported with ``ok=False`` and a reason.
        """
        ...
