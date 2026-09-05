"""One path every task takes, whatever produced it.

    fetch -> contamination check -> novelty check (ONE shared buffer) -> verifier -> emit

Nothing here branches on the source. The per-source numbers the layer exists to produce are
differences in what survives this path, not differences in the path.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .base import SourceResult, TaskRecord
from .registry import ContaminationFilter, SharedNoveltyBuffer


@dataclass
class SourceStats:
    """Per-source counts. Every candidate is accounted for in exactly one bucket."""

    name: str
    ok: bool = True
    failure_reason: str = ""
    candidates: int = 0
    rejected_contaminated: int = 0
    rejected_duplicate: int = 0
    verified: int = 0
    refuted: int = 0
    unverifiable_mechanical: int = 0
    unverifiable_substantive: int = 0
    accepted: int = 0
    cost_tokens: int = 0
    verify_tokens: int = 0
    max_contamination_sim: float = 0.0
    duplicate_of: dict = field(default_factory=dict)
    detail: dict = field(default_factory=dict)

    @property
    def decided(self) -> int:
        """Verifications that settled the key either way."""
        return self.verified + self.refuted

    @property
    def refuted_rate(self) -> float | None:
        """Refuted share AMONG DECIDED cases, which is the published comparison."""
        return (self.refuted / self.decided) if self.decided else None

    def as_row(self) -> dict:
        """Flat record for the deliverable table."""
        return {
            "source": self.name, "ok": self.ok, "failure_reason": self.failure_reason,
            "candidates": self.candidates,
            "rejected_contaminated": self.rejected_contaminated,
            "rejected_duplicate": self.rejected_duplicate,
            "novelty_rejection_rate": (self.rejected_duplicate / self.candidates
                                       if self.candidates else None),
            "contamination_rejection_rate": (self.rejected_contaminated / self.candidates
                                             if self.candidates else None),
            "decided": self.decided, "refuted": self.refuted,
            "refuted_key_rate": self.refuted_rate,
            "unverifiable_mechanical": self.unverifiable_mechanical,
            "unverifiable_substantive": self.unverifiable_substantive,
            "accepted": self.accepted,
            "cost_tokens": self.cost_tokens + self.verify_tokens,
            "duplicate_of": self.duplicate_of,
            "max_contamination_sim": round(self.max_contamination_sim, 4),
            "detail": self.detail,
        }


class TaskPipeline:
    """Runs sources through the identical path and keeps the per-source accounting."""

    def __init__(self, buffer: SharedNoveltyBuffer, contamination: ContaminationFilter,
                 verifier=None):
        self.buffer = buffer
        self.contamination = contamination
        self.verifier = verifier
        self.stats: dict[str, SourceStats] = {}
        self.accepted: list[TaskRecord] = []

    def run(self, sources, n_per_source: int, rng) -> dict[str, SourceStats]:
        """Fetch from each source and take every candidate through the same stages.

        A source that produces nothing is recorded with ``ok=False`` and its reason, and does
        NOT silently contribute a zero row: the caller can see the difference between "this
        source produced no surviving task" and "this source never ran".
        """
        for src in sources:
            st = SourceStats(name=src.name)
            self.stats[src.name] = st
            res: SourceResult = src.fetch(n_per_source, rng)
            st.cost_tokens = res.cost_tokens
            st.detail = dict(res.detail)
            if not res.ok:
                st.ok, st.failure_reason = False, res.reason
                continue
            st.candidates = len(res.tasks)
            if st.candidates == 0:
                st.ok, st.failure_reason = False, "source returned ok with zero tasks"
                continue
            for task in res.tasks:
                keep, sim, at = self.contamination.check(task.text)
                st.max_contamination_sim = max(st.max_contamination_sim, sim)
                if not keep:
                    st.rejected_contaminated += 1
                    continue
                keep, sim, owner = self.buffer.check(task.text)
                if not keep:
                    st.rejected_duplicate += 1
                    if owner:
                        st.duplicate_of[owner] = st.duplicate_of.get(owner, 0) + 1
                    continue
                if self.verifier is not None:
                    verdict, tokens = self.verifier(task)
                    st.verify_tokens += tokens
                    if verdict == "verified":
                        st.verified += 1
                    elif verdict == "refuted":
                        st.refuted += 1
                        # A refuted key is not accepted: it would train the solver against a
                        # wrong answer and, worse, score p_hat low and look ideally difficult.
                        continue
                    elif verdict == "unverifiable_mechanical":
                        st.unverifiable_mechanical += 1
                    else:
                        st.unverifiable_substantive += 1
                self.buffer.add(task.text, src.name)
                self.accepted.append(task)
                st.accepted += 1
        return self.stats

    def write_artifacts(self, path: str) -> int:
        """Write accepted tasks with their provenance intact.

        Provenance is written from the record rather than reconstructed, so that dropping it
        anywhere upstream shows up here as a missing field rather than as a plausible default.
        """
        with open(path, "w") as fh:
            for t in self.accepted:
                fh.write(json.dumps(t.to_json()) + "\n")
        return len(self.accepted)
