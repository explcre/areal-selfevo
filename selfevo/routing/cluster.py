"""Cluster-granularity routing: one policy per cluster of units, not per unit.

Why a tier between task and sample. SIA (arXiv 2605.27276) selects a training algorithm PER
TASK, so task-level choice is not available to claim. Per-SAMPLE choice is available but was
measured to reach only 1.7% of tokens through the shared-prefix rule, which is too small to
move a benchmark. The cluster tier sits between: units are partitioned inside a task, and
each partition gets its own signal.

The partition is DERIVED, not chosen. ``SilenceSide`` already splits units by where each
estimator is valid, and a live measurement makes the tier worth having: 57.4% of GRPO groups
are RL-silent -- every advantage identically zero -- against the token rule's 1.7%. That is a
34x larger channel, and it is the one an evolve-policy should be deciding about.

The three clusters need OPPOSITE responses, which is why their sum is never reported alone:

* ``INFORMATIVE``  the group disagrees, RL carries signal.
* ``UNSOLVED``     every sample failed. RL has nothing to push on; a teacher target would.
* ``SOLVED``       every sample succeeded. Nothing left to learn; spend the budget elsewhere.

This module deliberately does NOT learn the partition. A learned or embedding-based
clustering is a separate router that must beat this one, and ``MatchedPermutationControl``
exists so "clustering helped" can be told apart from "the mode proportions changed".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from .base import RoutingContext, RoutingDecision, TrainingMode, known_modes
from .criteria import SilenceSide, silence_side, threshold_is_inert

__all__ = ["ClusterAssignment", "ClusterRouter", "silence_cluster_key"]


def silence_cluster_key(ctx: RoutingContext, threshold: float = 0.1) -> str:
    """Derived cluster key: which side of RL silence the unit sits on.

    Uses the estimator's own validity condition rather than an arbitrary partition, which is
    what distinguishes this from k-means over embeddings.
    """
    return silence_side(ctx.solve_rate, ctx.group_size, threshold=threshold).value


@dataclass(frozen=True)
class ClusterAssignment:
    """What the router did to one batch.

    Attributes:
        decisions: One :class:`RoutingDecision` per input context, in input order.
        cluster_of: Cluster key per input context, in input order.
        sizes: Units per cluster.
        fractions: Share of the batch per cluster. Reported per cluster and never summed,
            because the silent clusters call for opposite responses and a combined
            "silent fraction" hides which one dominates.
        basis: What the partition rested on.
    """

    decisions: tuple[RoutingDecision, ...]
    cluster_of: tuple[str, ...]
    sizes: Mapping[str, int]
    fractions: Mapping[str, float]
    basis: str
    mode_counts: Mapping[str, int] = field(default_factory=dict)
    refused_teacher: int = 0
    granularity: str = "cluster"

    def __post_init__(self) -> None:
        if len(self.decisions) != len(self.cluster_of):
            raise ValueError(
                f"{len(self.decisions)} decisions but {len(self.cluster_of)} cluster labels; "
                "every unit must be assigned exactly one cluster"
            )
        if set(self.fractions) != set(self.sizes):
            raise ValueError(
                f"fractions keys {sorted(self.fractions)} != sizes keys {sorted(self.sizes)}"
            )
        if any(v < 0 for v in self.sizes.values()):
            raise ValueError(f"negative cluster size in {self.sizes}")
        if self.fractions and abs(sum(self.fractions.values()) - 1.0) > 1e-9:
            raise ValueError(
                f"fractions sum to {sum(self.fractions.values())}, not 1. They are reported "
                "per cluster precisely so they can be checked; an unchecked share is how a "
                "dropped cluster goes unnoticed."
            )
        if not self.basis:
            raise ValueError("basis must not be empty")
        # No `if self.sizes` guard: empty sizes with non-empty decisions is the bug, not an
        # exemption from the check.
        if sum(self.sizes.values()) != len(self.decisions):
            raise ValueError(
                f"cluster sizes sum to {sum(self.sizes.values())} but {len(self.decisions)} "
                "units were routed; a unit was dropped or double-counted"
            )


@dataclass
class ClusterRouter:
    """Assigns one training signal per cluster of units within a task.

    Args:
        policy: Cluster key -> mode name. Any cluster absent from this mapping falls back to
            ``default_mode``, and the fallback is reported so a silently unrouted cluster
            cannot be mistaken for a deliberate one.
        key_fn: Partition function. Defaults to the derived silence split. Supplying another
            is how a learned or MEDS-style clustering is compared against the derived one.
        threshold: Informativeness below which a unit counts as silent.
        default_mode: Used for clusters the policy does not name.
        require_teacher_for: Modes that need an external target. A unit routed to one of
            these without ``ctx.has_teacher`` is redirected to SKIP rather than being
            trained on a target that does not exist.

    Raises:
        ValueError: If the policy names an unregistered mode.
    """

    policy: Mapping[str, str] = field(
        default_factory=lambda: {
            SilenceSide.INFORMATIVE.value: TrainingMode.RL,
            # SFT, not DISTILL. base.py states hard distillation is deliberately absent
            # and that SFT with a teacher-sourced target is the supported path;
            # BanditRouter excludes DISTILL because its transport is not built, and
            # sim_routing matches neither branch for it, so a unit routed to DISTILL
            # pays full cost and never learns. An audit measured 400/1000 units in
            # exactly that state on the motivating batch.
            SilenceSide.UNSOLVED.value: TrainingMode.SFT,
            SilenceSide.SOLVED.value: TrainingMode.SKIP,
        }
    )
    key_fn: Callable[[RoutingContext], str] | None = None
    threshold: float = 0.1
    default_mode: str = TrainingMode.SKIP

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            # 0.0 is rejected, not merely out of taste. criteria tests `I_RL >= threshold`
            # and I_RL is exactly 0 for a unanimous group, so 0 >= 0 makes EVERY unit
            # "informative": 100% routed to RL while the basis still reads "derived silence
            # split". That is the first value a threshold sweep tries.
            raise ValueError(
                f"threshold must be in (0, 1]; got {self.threshold}. At 0.0 every unanimous "
                "group is classified informative and the split silently disappears."
            )
        modes = known_modes()
        for cluster, mode in self.policy.items():
            if mode not in modes:
                raise ValueError(f"cluster {cluster!r} routed to unknown mode {mode!r}")
        if self.default_mode not in modes:
            raise ValueError(f"unknown default_mode {self.default_mode!r}")

    def _key(self, ctx: RoutingContext) -> str:
        if self.key_fn is not None:
            return self.key_fn(ctx)
        return silence_cluster_key(ctx, self.threshold)

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        """Route a single unit, for compatibility with the ``Router`` protocol."""
        return self.route_batch([ctx]).decisions[0]

    def route_batch(self, contexts: Sequence[RoutingContext]) -> ClusterAssignment:
        """Partition a batch into clusters and give each cluster its own signal.

        Args:
            contexts: Units to route. May be empty, which yields an empty assignment rather
                than an error -- an empty batch is a real state, not a misuse.

        Returns:
            A :class:`ClusterAssignment`.
        """
        keys = [self._key(c) for c in contexts]
        decisions = []
        fell_back = 0
        refused = 0
        for ctx, key in zip(contexts, keys):
            mode = self.policy.get(key)
            if mode is None:
                mode, fell_back = self.default_mode, fell_back + 1
                why = f"cluster {key!r} is not named by the policy; fell back to {mode}"
            else:
                why = f"cluster {key!r} -> {mode}"
            # A teacher mode without a teacher trains on a target that does not exist.
            if known_modes().get(mode) and not ctx.has_teacher:
                why = (f"cluster {key!r} routes to {mode}, which needs a teacher and none is "
                       f"available; skipping instead of training on an absent target")
                mode = TrainingMode.SKIP
                refused += 1
            # RoutingDecision takes weights positionally and carries no granularity field,
            # so the tier is recorded on the assignment rather than on each decision.
            decisions.append(RoutingDecision({mode: 1.0}, reason=why))
        counts = Counter(keys)
        n = len(contexts)
        if self.key_fn is not None:
            basis = "caller-supplied partition"
        else:
            # Report inertness instead of implying the threshold separated anything. At
            # every usable G the smallest non-zero I_RL is 0.64-0.68, so any threshold below
            # that -- including the 0.1 default -- makes the split exactly "was the group
            # unanimous?". Above that band it MISLABELS genuinely informative groups, so the
            # parameter has no safe non-inert setting and the honest thing is to say so.
            gs = {c.group_size for c in contexts}
            inert = all(threshold_is_inert(g, self.threshold) for g in gs) if gs else True
            basis = (f"derived silence split; threshold {self.threshold} is "
                     + ("INERT at these group sizes, so the split is exactly "
                        "'was the group unanimous?'" if inert
                        else "ACTIVE and therefore mislabelling non-unanimous groups"))
        if fell_back:
            basis += f"; {fell_back} unit(s) hit the {self.default_mode} fallback"
        return ClusterAssignment(
            # The mode histogram is what a caller should read: the cluster sizes say how the
            # batch PARTITIONED, not what it was TRAINED on. With no teacher those differ
            # completely -- an audit measured an assignment reporting half the batch routed
            # to the teacher while zero units actually were.
            mode_counts=dict(Counter(next(iter(d.weights)) for d in decisions)),
            refused_teacher=refused,
            decisions=tuple(decisions),
            cluster_of=tuple(keys),
            sizes=dict(counts),
            fractions={k: v / n for k, v in counts.items()} if n else {},
            basis=basis,
        )
