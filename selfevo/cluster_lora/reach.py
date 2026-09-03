"""What the partition actually reached this batch, and how stable it was.

A feature that is built, named, and reaches nothing is the failure mode this repo distrusts
most, and per-cluster adapters have three distinct ways to reach nothing while looking
healthy:

1. **Everything is noise.** HDBSCAN labels every group ``-1``, every group lands on the
   shared adapter, and the run is bit-identical to vanilla LoRA while the config says
   ``partition=meds``. ``noise_fraction`` is the number that catches it, and 1.0 is a
   refusal-worthy value, not a small one.
2. **One cluster swallows the batch.** N adapters exist, N-1 of them never see a group and
   never move, and the merge at inference sums N-1 zeros. ``largest_cluster_fraction``
   catches it.
3. **Labels churn.** Every group changes adapter every step, so each expert receives a
   different subpopulation each time and learns the average anyway -- the exact thing the
   method claims to avoid, arrived at through the mechanism that was supposed to prevent
   it. ``churn`` catches it, and it is why MEDS' kNN stabilisation is implemented rather
   than skipped.

None of the three raises here. They are RECORDED, per batch, with a flat metric namespace
so a run that quietly degenerated is visible on the same panel as one that did not -- a
refusal mid-run would lose the training, whereas a metric at 1.0 for 400 steps is a result.
The refusals that DO raise live in :mod:`selfevo.cluster_lora.partition`, and they are the
cases where no partition exists at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .partition import Partition, label_churn

__all__ = ["ReachReport", "reach_report"]


@dataclass(frozen=True)
class ReachReport:
    """Per-batch reach and stability of a cluster partition.

    Args:
        n_groups: Groups in the batch.
        n_clusters: Non-noise clusters. The shared adapter is not counted, because it is
            where the clustering declined to make a claim.
        sizes: Groups per adapter name, shared included.
        n_noise: Groups on the shared adapter.
        churn: Fraction of groups seen in the previous batch whose adapter changed.
        n_churn_overlap: Groups comparable between the two batches. Zero means churn was
            not measurable, which is different from churn being zero and is reported
            separately so the two cannot be confused.
        basis: The partition's own record of what it rested on.
    """

    n_groups: int
    n_clusters: int
    sizes: Mapping[str, int]
    n_noise: int
    churn: float
    n_churn_overlap: int
    basis: str
    refusals: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.n_groups <= 0:
            raise ValueError(f"n_groups must be positive, got {self.n_groups}")
        if sum(self.sizes.values()) != self.n_groups:
            raise ValueError(
                f"sizes sum to {sum(self.sizes.values())} but {self.n_groups} groups were "
                "partitioned; a group was dropped or double-counted"
            )
        if not 0.0 <= self.churn <= 1.0:
            raise ValueError(f"churn must be a fraction, got {self.churn}")

    @property
    def noise_fraction(self) -> float:
        """Share of the batch on the shared adapter. 1.0 means the method did nothing."""
        return self.n_noise / self.n_groups

    @property
    def largest_cluster_fraction(self) -> float:
        """Share of the batch in the biggest adapter, shared included.

        Includes the shared adapter deliberately: a batch that is 95% noise and a batch
        that is 95% one cluster are the same failure from the gradient's point of view, and
        a metric that excluded noise would report the first as perfectly balanced.
        """
        return max(self.sizes.values()) / self.n_groups if self.sizes else 0.0

    def as_metrics(self) -> dict[str, float]:
        """Flat scalars for the run's metrics namespace.

        Per-cluster sizes are emitted individually and never only as a total. Two clusters
        of 32 and eight clusters of eight give the same total and call for opposite
        readings, and this project has already shipped one metric that summed away the
        distinction it existed to show.
        """
        out = {
            "cluster_lora/n_groups": float(self.n_groups),
            "cluster_lora/n_clusters": float(self.n_clusters),
            "cluster_lora/noise_fraction": float(self.noise_fraction),
            "cluster_lora/largest_cluster_fraction": float(self.largest_cluster_fraction),
            "cluster_lora/churn": float(self.churn),
            "cluster_lora/churn_overlap": float(self.n_churn_overlap),
            "cluster_lora/refusals": float(len(self.refusals)),
        }
        for name, n in self.sizes.items():
            out[f"cluster_lora/size/{name}"] = float(n)
        return out


def reach_report(
    current: Partition,
    previous: Partition | None = None,
    *,
    refusals: tuple[str, ...] = (),
) -> ReachReport:
    """Build the per-batch record from a partition and its predecessor.

    Args:
        current: This batch's partition.
        previous: The previous batch's partition, or ``None`` on the first batch.
        refusals: Typed refusals raised and handled while forming this batch, recorded so a
            run that fell back cannot look like one that did not.

    Returns:
        A :class:`ReachReport`.
    """
    churn, _changed, overlap = label_churn(previous, current)
    return ReachReport(
        n_groups=current.n_groups,
        n_clusters=current.n_clusters,
        sizes=current.sizes,
        n_noise=current.n_noise,
        churn=churn,
        n_churn_overlap=overlap,
        basis=current.basis,
        refusals=tuple(refusals),
    )
