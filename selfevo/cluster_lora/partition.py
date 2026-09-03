"""How a batch of GRPO groups is split into the clusters that own the adapters.

Four partitions of the SAME batch are supported, because a gain from per-cluster adapters
has four different explanations and only one of them is the paper's:

``meds``            HDBSCAN over the model's own behaviour (latter-half layer-wise logits
                    at the final answer token), kNN-stabilised across batches. The method.
``random_matched``  The same number of clusters at the same sizes, labels drawn
                    feature-blind under a seed. **Mandatory.** Without it a gain reads as
                    "more adapters", not "clustering", and no amount of prose fixes that
                    after the fact.
``feature``         Clustering on any supplied per-group vector. Used for the ELREA
                    ablation (arXiv 2502.00089), which clusters on PROMPT-token gradient
                    features and already does cluster -> per-cluster LoRA -> merge in SFT.
                    If prompt-gradient clusters conflict as much as behavioural ones, the
                    rollouts are not buying anything and the method is ELREA in RL clothing.
``task``            The batch's own task labels. The calibration: arXiv 2608.03573 measures
                    cross-TASK RL update cosine at ~1e-5, and a reviewer will say our target
                    is negligible. Reproducing that figure on our own batch is what makes
                    the within-task number mean something.

Every partition is expressed as the SAME :class:`Partition` type, including the degenerate
``none`` one where every group shares a single adapter. That is deliberate: the vanilla
shared-LoRA baseline then runs through the identical code path as the method, so a
difference between the arms cannot come from the plumbing.

HDBSCAN noise (label ``-1``) is not dropped and is not its own cluster per group; it goes to
one SHARED adapter. A group HDBSCAN cannot place is a group we have no behavioural claim
about, and the honest treatment of "no claim" is the average update, not a private expert
fitted to one group.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "DEFAULT_CONTROL_MEMORY",
    "SHARED_CLUSTER",
    "MEDSPartitioner",
    "MatchedControlMemory",
    "Partition",
    "PartitionUnavailable",
    "balanced_assign",
    "cluster_key",
    "feature_partition",
    "label_churn",
    "max_experts_for_roster",
    "meds_partition",
    "no_partition",
    "partition_from_config",
    "random_matched_partition",
    "task_partition",
]

#: Adapter that carries every group the clustering could not place (HDBSCAN label -1).
SHARED_CLUSTER = "shared"

#: The values ``cluster_lora.partition`` accepts. ``feature`` and ``task`` are analysis-only
#: partitions -- they need vectors or labels a training step does not have -- so they are
#: not offered as a training mode.
TRAINING_PARTITIONS = ("meds", "random_matched", "none")
# What partition_from_config actually has a branch for. Separate from the registry above so
# the two can be COMPARED: a name in one and not the other is the arm-mislabelling bug, and
# the refusal at the end of partition_from_config is what makes the gap visible.
DISPATCHED_PARTITIONS = ("none", "meds", "random_matched")


class PartitionUnavailable(RuntimeError):
    """A partition could not be formed, with the reason stated.

    Raised rather than returning a degenerate partition. A partitioner that quietly returns
    "everything in cluster 0" when its inputs are missing produces a run that is
    bit-identical to the shared-adapter baseline while reporting as the method, which is the
    exact failure this codebase keeps hitting and the reason every path here either works or
    raises.
    """


def cluster_key(label: int) -> str:
    """Adapter name for a cluster label.

    Args:
        label: HDBSCAN-style label; ``-1`` is noise.

    Returns:
        ``"shared"`` for noise, else ``"cluster_<label>"``.

    Raises:
        ValueError: If ``label`` is below ``-1``. There is no second noise label, and
            accepting one would give it a private adapter named ``cluster_-2``.
    """
    label = int(label)
    if label < -1:
        raise ValueError(f"cluster label must be >= -1, got {label}")
    return SHARED_CLUSTER if label == -1 else f"cluster_{label}"


@dataclass(frozen=True)
class Partition:
    """One batch's assignment of groups to adapters.

    Args:
        labels: Cluster label per group, in batch order. ``-1`` is noise.
        basis: What the partition rested on, carried into the run record so a partition
            cannot be mistaken for a different one later.
        group_ids: Stable identifier per group (a prompt id, not a batch position), used to
            measure churn between consecutive batches. Batch positions are reshuffled every
            step, so churn measured against them would be noise.

    Raises:
        ValueError: If the fields disagree, or ``basis`` is empty.
    """

    labels: tuple[int, ...]
    basis: str
    group_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(int(v) < -1 for v in self.labels):
            raise ValueError(f"labels must all be >= -1, got {sorted(set(self.labels))}")
        if not self.basis:
            raise ValueError("basis must not be empty")
        if self.group_ids and len(self.group_ids) != len(self.labels):
            raise ValueError(
                f"{len(self.group_ids)} group ids but {len(self.labels)} labels; a partition "
                "whose identities do not line up with its labels cannot be churn-checked"
            )

    @property
    def keys(self) -> tuple[str, ...]:
        """Adapter name per group, in batch order."""
        return tuple(cluster_key(v) for v in self.labels)

    @property
    def n_groups(self) -> int:
        """Groups partitioned."""
        return len(self.labels)

    @property
    def n_clusters(self) -> int:
        """Non-noise clusters. Excludes the shared adapter, which is not a discovery."""
        return len({v for v in self.labels if v != -1})

    @property
    def n_noise(self) -> int:
        """Groups HDBSCAN declined to place, which go to the shared adapter."""
        return sum(1 for v in self.labels if v == -1)

    @property
    def sizes(self) -> Mapping[str, int]:
        """Groups per adapter name, including ``shared``."""
        return dict(Counter(self.keys))

    @property
    def adapters(self) -> tuple[str, ...]:
        """Adapter names this partition needs, sorted, shared last.

        Sorted numerically rather than lexically: ``cluster_10`` must not sort between
        ``cluster_1`` and ``cluster_2``, or the adapter a cluster owns would depend on how
        many clusters were found.
        """
        named = sorted(
            {v for v in self.labels if v != -1}
        )
        out = [f"cluster_{v}" for v in named]
        if self.n_noise:
            out.append(SHARED_CLUSTER)
        return tuple(out)

    def size_multiset(self) -> tuple[int, ...]:
        """Cluster sizes, sorted. What a size-matched control has to reproduce exactly."""
        return tuple(sorted(Counter(self.labels).values()))


def no_partition(n_groups: int, *, group_ids: Sequence[str] = ()) -> Partition:
    """Every group on the shared adapter: the vanilla shared-LoRA arm, in this type.

    Args:
        n_groups: Groups in the batch.
        group_ids: Optional stable identifiers.

    Returns:
        A :class:`Partition` with every label ``-1``.

    Raises:
        ValueError: If ``n_groups`` is not positive. An empty batch reaching a partitioner
            means the caller lost the batch, and returning an empty partition would let the
            step proceed having trained on nothing.
    """
    if n_groups <= 0:
        raise ValueError(f"n_groups must be positive, got {n_groups}")
    return Partition(
        labels=(-1,) * n_groups,
        basis="none: every group shares one adapter (the vanilla LoRA arm)",
        group_ids=tuple(group_ids),
    )


class MatchedControlMemory:
    """Which adapter the size-matched control has already given each group.

    The control must differ from the method on ONE axis -- whether the labels saw the
    behavioural features -- and on no other. Drawing a fresh permutation every batch adds a
    second: MEASURED 2026-09-02 on a partition whose membership never changed at all, the
    method churned 0.0 per step while a freshly-permuted control churned 0.875 and 0.667.

    That second axis is not a nuisance parameter. ``FINDINGS_cluster_lora.md`` section 5.1
    establishes churn as THE mechanism that makes per-cluster experts fail: at churn 1.0
    every expert receives a different subpopulation each step and learns the average anyway,
    which is the exact thing the method claims to avoid and the reason
    :meth:`MEDSPartitioner._resync` exists. A control redrawn every batch is therefore not
    "the method without clustering", it is "the method without clustering AND without stable
    expert identity", and a gain over it cannot be attributed to the clustering.

    This class is the control's half of ``_resync``. The method keeps a cluster's expert by
    matching overlap after each refit; the control keeps a GROUP's expert by remembering the
    label its first, feature-blind draw gave it. The draw is unchanged and stays blind: a
    group is placed by a seeded permutation the first time the control sees it, and by
    nothing about the group ever.

    Args:
        max_entries: Identities to remember; the oldest are dropped first. A run sees one
            identity per prompt per epoch, so the default is far above any realised run and
            exists only so that a long-lived process cannot grow without bound.

    Raises:
        ValueError: If ``max_entries`` is below 1. A memory that remembers nothing is the
            defect this class exists to fix, wearing its name.
    """

    def __init__(self, max_entries: int = 1 << 18) -> None:
        if max_entries < 1:
            raise ValueError(
                f"max_entries must be >= 1, got {max_entries}; a control memory that "
                "remembers nothing re-permutes every batch, which is the defect"
            )
        self.max_entries = int(max_entries)
        self._seen: OrderedDict[str, int] = OrderedDict()

    def __len__(self) -> int:
        """Group identities currently remembered."""
        return len(self._seen)

    def reset(self) -> None:
        """Forget every identity.

        For tests, and for a process that starts a second arm: a control carried over from
        a previous arm would place this arm's groups by the previous arm's draw.
        """
        self._seen.clear()

    def carry_forward(
        self, group_ids: Sequence[str], drawn: Sequence[int]
    ) -> tuple[tuple[int, ...], int]:
        """Re-seat a freshly drawn control so a known group keeps the label it already had.

        Args:
            group_ids: Stable identity per group, in batch order. EMPTY means the caller
                has no identities to key on: the draw is returned untouched and nothing is
                remembered, because there is no carrying forward an assignment for a group
                that cannot be recognised next batch.
            drawn: The seeded permutation, already size-matched to the reference.

        Returns:
            ``(labels, n_carried)``. ``labels`` is a REARRANGEMENT of ``drawn`` -- the same
            labels with the same multiplicities, so the size match is preserved by
            construction rather than by a second permutation that could lose it -- in which
            every group the control has placed before keeps that label, and the rest are
            filled from what is left in the drawn order. ``n_carried`` is how many groups
            kept a previous label, which is what the partition's basis records.
        """
        drawn = [int(v) for v in drawn]
        if not group_ids:
            return tuple(drawn), 0
        capacity = Counter(drawn)
        out: list[int | None] = [None] * len(drawn)
        carried = 0
        for i, gid in enumerate(group_ids):
            remembered = self._seen.get(gid)
            # Capacity, not mere membership. A remembered label this batch has no room for
            # cannot be honoured without breaking the size match, and the size match is the
            # other property the control exists to hold; such a group is redrawn below.
            if remembered is not None and capacity[remembered] > 0:
                out[i] = remembered
                capacity[remembered] -= 1
                carried += 1
        # The leftovers taken in DRAWN order, so a group the control has not seen is placed
        # by the seeded permutation and by nothing else.
        remaining = Counter(capacity)
        leftovers: list[int] = []
        for value in drawn:
            if remaining[value] > 0:
                leftovers.append(value)
                remaining[value] -= 1
        spare = iter(leftovers)
        for i, value in enumerate(out):
            if value is None:
                out[i] = next(spare)
        for gid, label in zip(group_ids, out):
            self._seen.pop(gid, None)
            self._seen[gid] = int(label)
        while len(self._seen) > self.max_entries:
            self._seen.popitem(last=False)
        return tuple(int(v) for v in out), carried


#: The memory :func:`random_matched_partition` uses when the caller supplies none.
#:
#: Module-level rather than per-call because the property being fixed is a property ACROSS
#: calls, and the function is called once per batch from a call site that holds no state of
#: its own. A run that wants its own -- two arms in one process, or a test -- passes
#: ``memory=``; :class:`MEDSPartitioner` holds one so that the control's memory has exactly
#: the method's lifetime.
DEFAULT_CONTROL_MEMORY = MatchedControlMemory()


def random_matched_partition(
    reference: Partition,
    *,
    seed: int,
    tag: str = "random_matched",
    memory: MatchedControlMemory | None = None,
) -> Partition:
    """The mandatory control: same N, same sizes, labels feature-blind.

    Built by PERMUTING the reference's own labels rather than by sampling from the observed
    proportions. The sizes then match EXACTLY rather than in expectation, for any batch and
    any cluster-size distribution, which is the property a control needs -- an arm whose
    cluster sizes drifted from the method's would differ on two axes at once and could not
    settle the question it exists to settle. The repo's
    ``selfevo.routing.proportions.MatchedPermutationControl`` makes the same argument for
    mode proportions and was added after a configured control was measured to be
    mis-matched.

    The noise bucket is permuted along with the rest, so the control also has the same
    number of groups on the shared adapter. A control with fewer shared groups would have
    more per-cluster capacity than the method it controls for.

    STABLE ACROSS BATCHES, which is the third property and was the missing one. The draw is
    carried forward per GROUP IDENTITY through :class:`MatchedControlMemory`, so a group the
    control has already placed keeps that adapter on every later batch instead of being
    re-permuted under the next seed. Without it the control churned 0.875 and 0.667 per step
    where the method churned 0.0, i.e. it differed from the arm in expert STABILITY as well
    as in feature-blindness -- and section 5.1 of the findings makes stability the mechanism
    the whole method rests on, so a gain over such a control is not attributable to the
    clustering. A reference carrying NO group ids has no identity to key on and is drawn
    afresh, exactly as before.

    Args:
        reference: The partition whose N and sizes are to be reproduced. Its ``group_ids``
            are the identities the draw is carried forward on.
        seed: Permutation seed, governing the draw for every group the control has not
            placed before. A private generator, so this neither perturbs nor is perturbed by
            sampling elsewhere in the process.
        tag: Prefix for the recorded basis.
        memory: Where the per-identity draw is carried. Defaults to
            :data:`DEFAULT_CONTROL_MEMORY`; :class:`MEDSPartitioner` holds one so that an
            arm's control forgets exactly when its method does.

    Returns:
        A size-matched, feature-blind, identity-stable :class:`Partition`.

    Raises:
        ValueError: If the reference has no groups.
    """
    if reference.n_groups == 0:
        raise ValueError("cannot match a partition with no groups")
    rng = np.random.default_rng(seed)
    labels = np.array(reference.labels, dtype=np.int64)
    permuted = tuple(int(v) for v in rng.permutation(labels))
    memory = DEFAULT_CONTROL_MEMORY if memory is None else memory
    # A REARRANGEMENT of the draw, never a second draw, so the size match asserted below
    # cannot be lost here.
    permuted, carried = memory.carry_forward(reference.group_ids, permuted)
    out = Partition(
        labels=permuted,
        basis=(
            f"{tag}: labels of {reference.basis!r} permuted under seed {seed}; "
            f"sizes {list(reference.size_multiset())} matched exactly; "
            f"{carried}/{reference.n_groups} groups kept the adapter the control had "
            "already given them"
        ),
        group_ids=reference.group_ids,
    )
    # Not a comment, an assertion. "Matched by construction" is precisely the kind of claim
    # that survives a refactor while stopping being true.
    if out.size_multiset() != reference.size_multiset():
        raise PartitionUnavailable(
            f"the size-matched control is not size-matched: {out.size_multiset()} vs "
            f"{reference.size_multiset()}"
        )
    return out


def task_partition(task_labels: Sequence[str], *, group_ids: Sequence[str] = ()) -> Partition:
    """The cross-task calibration partition.

    arXiv 2608.03573 measures cross-task RL update cosine at about 1e-5. Reproducing that
    on our own batch is what turns the within-task number from a bare figure into a
    comparison, so this partition is not an extra -- it is the scale bar.

    Args:
        task_labels: One task name per group.
        group_ids: Optional stable identifiers.

    Returns:
        A :class:`Partition` with one cluster per distinct task, ordered by first
        appearance so the mapping is reproducible.

    Raises:
        PartitionUnavailable: If the batch spans a single task. Returning one cluster would
            make the calibration silently vacuous -- every pairwise statistic over one
            cluster is empty, and an empty mean reads as 0.0, which is exactly the answer
            the calibration is supposed to produce honestly or not at all.
    """
    distinct: list[str] = []
    for t in task_labels:
        if t not in distinct:
            distinct.append(t)
    if len(distinct) < 2:
        raise PartitionUnavailable(
            f"the batch spans {len(distinct)} task(s) ({distinct}); the cross-task "
            "calibration needs at least two, so it is skipped rather than reported as zero"
        )
    index = {t: i for i, t in enumerate(distinct)}
    return Partition(
        labels=tuple(index[t] for t in task_labels),
        basis=f"task labels: {distinct}",
        group_ids=tuple(group_ids),
    )


def balanced_assign(distances: np.ndarray, capacities: Sequence[int]) -> np.ndarray:
    """Assign rows to columns respecting a per-column capacity, greedily by distance.

    Used to give a feature-based partition the SAME cluster sizes as the partition it is
    being compared against. Without size matching, "the ELREA clusters conflict less" could
    simply mean "the ELREA clusters are more lopsided", since a cluster of one group has no
    within-cluster averaging to do.

    Greedy, and stated as such: it takes the globally closest admissible pair repeatedly,
    which is not the optimal capacitated assignment. Optimal would need a transportation
    solver and a dependency; the approximation is disclosed in the partition's basis so a
    result cannot rest on it silently.

    Args:
        distances: ``(n_rows, n_cols)`` distances.
        capacities: Capacity per column; must sum to ``n_rows``.

    Returns:
        ``(n_rows,)`` column index per row.

    Raises:
        ValueError: If the capacities do not sum to the number of rows -- an under- or
            over-capacity plan silently drops or duplicates groups.
    """
    n_rows, n_cols = distances.shape
    cap = list(int(c) for c in capacities)
    if len(cap) != n_cols:
        raise ValueError(f"{len(cap)} capacities for {n_cols} columns")
    if sum(cap) != n_rows:
        raise ValueError(
            f"capacities sum to {sum(cap)} but there are {n_rows} rows; a plan that does "
            "not partition the batch would drop or duplicate groups"
        )
    order = np.dstack(np.unravel_index(np.argsort(distances, axis=None), distances.shape))[0]
    out = np.full(n_rows, -1, dtype=np.int64)
    remaining = n_rows
    for r, c in order:
        if remaining == 0:
            break
        if out[r] != -1 or cap[c] == 0:
            continue
        out[r] = c
        cap[c] -= 1
        remaining -= 1
    if (out == -1).any():
        raise PartitionUnavailable("greedy assignment left rows unassigned")
    return out


def feature_partition(
    vectors: np.ndarray,
    n_clusters: int,
    *,
    seed: int = 0,
    match_sizes: Sequence[int] | None = None,
    tag: str = "feature",
    group_ids: Sequence[str] = (),
) -> Partition:
    """Cluster groups on any supplied per-group vector. The ELREA-style ablation.

    ELREA (arXiv 2502.00089) clusters instructions on PROMPT-token gradient features and
    trains a LoRA per cluster, then merges -- in SFT. Feeding prompt-gradient sketches here
    reproduces that partition on our batch, and the comparison decides something concrete:
    if prompt-gradient clusters show the same conflict as behavioural ones, nothing about
    this method needs rollouts and the contribution collapses to ELREA-in-RL.

    Args:
        vectors: ``(n_groups, d)`` features. L2-normalised internally, so a group with a
            larger gradient does not dominate the geometry through its magnitude alone.
        n_clusters: Clusters to form. Set from the partition being compared against, so the
            comparison is at matched N.
        seed: k-means seed.
        match_sizes: Optional target cluster sizes; when given, k-means centres are fitted
            first and groups are then assigned by :func:`balanced_assign` so the sizes match
            exactly.
        tag: Prefix for the recorded basis.
        group_ids: Optional stable identifiers.

    Returns:
        A :class:`Partition`. Labels are ``0..n_clusters-1``; this partition has no noise
        bucket, which is a real difference from HDBSCAN and is stated in the basis.

    Raises:
        PartitionUnavailable: If scikit-learn is missing, or there are fewer groups than
            clusters.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
    n = vectors.shape[0]
    if n_clusters < 1 or n_clusters > n:
        raise PartitionUnavailable(
            f"cannot form {n_clusters} clusters from {n} groups"
        )
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:  # pragma: no cover - exercised on boxes without sklearn
        raise PartitionUnavailable(
            f"the feature partition needs scikit-learn ({exc}); run the analysis under "
            "~/venv_probe, which has it, rather than the training venv"
        ) from exc
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise PartitionUnavailable(
            "a group's feature vector is all zeros; its direction is undefined and any "
            "cluster it lands in would be an artefact of the tie-break"
        )
    unit = vectors / norms
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(unit)
    basis = f"{tag}: k-means k={n_clusters} on L2-normalised features, seed {seed}"
    if match_sizes is None:
        labels = tuple(int(v) for v in km.labels_)
    else:
        caps = list(int(c) for c in match_sizes)
        if len(caps) != n_clusters:
            raise PartitionUnavailable(
                f"{len(caps)} target sizes for {n_clusters} clusters"
            )
        d = np.linalg.norm(unit[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
        labels = tuple(int(v) for v in balanced_assign(d, caps))
        basis += f"; sizes forced to {sorted(caps)} by greedy capacitated assignment"
    basis += "; no noise bucket, unlike HDBSCAN"
    return Partition(labels=labels, basis=basis, group_ids=tuple(group_ids))


class MEDSPartitioner:
    """MEDS clustering as a partition, with the kNN stabilisation MEDS uses across batches.

    The two-phase lifecycle is the whole point and is not an optimisation. Re-fitting
    HDBSCAN on each batch independently gives cluster ids that mean something different
    every step: a group that jumps adapters every step contributes one noisy update to each
    of several experts and coherent training to none, which would make the method fail for a
    reason that has nothing to do with its hypothesis. So the current batch is LABELLED by
    kNN against the fitted history (MEDS' ``_classify_with_knn``), and only then added to
    the buffer and re-fitted for the next batch.

    Churn is therefore expected to be small but is never assumed -- :func:`label_churn`
    measures it, and :class:`selfevo.cluster_lora.reach.ReachReport` carries it per batch.

    Args:
        clusterer: A :class:`selfevo.clustering.meds.MEDSClusterer`, or ``None`` to build
            one with MEDS' own defaults.
        warmup_batches: Batches to buffer before the first fit. HDBSCAN on a handful of
            points returns almost all noise, and a first batch that is 100% noise would put
            every group on the shared adapter and look like the method doing nothing.
        max_experts: How many ``cluster_<i>`` adapters exist to be assigned, from
            :func:`max_experts_for_roster`, or ``None`` for the unbounded allocation every
            run before this bound had. The ids are otherwise sparse and unbounded while the
            roster is fixed at process start, so an unbounded partitioner is expected to
            name an adapter the run does not have and die partway through; see
            :func:`max_experts_for_roster` for the measurement and for why the allocation is
            bounded rather than compressed onto the roster.

    Raises:
        ValueError: If ``warmup_batches`` is negative, or if ``max_experts`` is below one --
            refused at CONSTRUCTION, because a roster that cannot seat a single cluster
            cannot run this method at all and the honest place to say so is before the
            accelerators are allocated.
    """

    def __init__(
        self, clusterer=None, *, warmup_batches: int = 1, max_experts: int | None = None
    ) -> None:
        if warmup_batches < 0:
            raise ValueError(f"warmup_batches must be >= 0, got {warmup_batches}")
        if max_experts is not None and int(max_experts) < 1:
            raise ValueError(
                f"max_experts must be >= 1 or None, got {max_experts}; a roster with no "
                "cluster_<i> expert cannot carry a partition, and discovering that at step "
                "300 costs the run"
            )
        if clusterer is None:
            from selfevo.clustering.meds import MEDSClusterer

            clusterer = MEDSClusterer()
        self.clusterer = clusterer
        self.warmup_batches = int(warmup_batches)
        self.max_experts = None if max_experts is None else int(max_experts)
        # Raw labels the roster has no expert left for. A SET of raw labels rather than a
        # count of groups, so it says how many CLUSTERS were turned away however many groups
        # they held, and rebuilt by _resync so it describes the live fit rather than the
        # run's history.
        self._overflow: set[int] = set()
        # The control's half of the identity _resync keeps for the method, held here so the
        # two have the SAME lifetime: a fresh partitioner is a fresh arm, and an arm whose
        # control remembered the previous arm's draw would place this arm's groups by it.
        self.control_memory = MatchedControlMemory()
        self.batches_seen = 0
        self._previous: Partition | None = None
        # Raw HDBSCAN label -> stable expert id, and the stable id of every buffered vector.
        # See _resync for why this indirection is not optional.
        self._raw_to_stable: dict[int, int] = {}
        self._history: list[int | None] = []
        self._next_stable = 0
        self.relabellings = 0

    def _extend(self, features, labels) -> None:
        """Buffer this batch and keep the stable-id history aligned with the buffer.

        The clusterer trims its buffer from the FRONT once it is full, so the history is
        trimmed the same way. A misaligned history would match new clusters against the
        wrong groups and shuffle every expert.
        """
        for v in features:
            self.clusterer.add(v)
        self._history.extend(labels)
        buffered = len(getattr(self.clusterer, "_state", {}).get("vectors", self._history))
        if len(self._history) > buffered:
            del self._history[: len(self._history) - buffered]

    def _stable_id(self, raw: int) -> int:
        """Map a raw HDBSCAN label to the expert that owns it, allocating one if new.

        Allocation is BOUNDED by ``max_experts`` when one was given. ``_next_stable`` is
        monotone by design -- reusing a freed id would give an expert somebody else's
        subpopulation, which is the thing ``_resync`` exists to prevent -- so it climbs past
        the number of live clusters as HDBSCAN fragments and renames across refits, and the
        names emitted stop fitting the roster the run was started with. Once the bound is
        reached a genuinely new cluster is sent to the SHARED adapter and recorded in
        ``overflow_clusters``, which is the treatment this module already gives a group it
        has no behavioural claim about, rather than being given a name the run cannot honour.
        """
        raw = int(raw)
        if raw == -1:
            return -1
        got = self._raw_to_stable.get(raw)
        if got is None:
            if self.max_experts is not None and self._next_stable >= self.max_experts:
                self._overflow.add(raw)
                return -1
            got = self._next_stable
            self._next_stable += 1
            self._raw_to_stable[raw] = got
        return got

    @property
    def overflow_clusters(self) -> tuple[int, ...]:
        """Raw labels the roster has no expert for, sorted; empty when it has room.

        Reported rather than merely handled: a run whose clusters are being folded into the
        shared adapter is training fewer experts than its config names, and the difference
        between that and a run whose clustering found fewer clusters is invisible in
        ``n_clusters`` alone.
        """
        return tuple(sorted(self._overflow))

    def _resync(self) -> None:
        """Re-anchor expert identity after a refit. **Without this the method is a no-op.**

        MEASURED 2026-09-02, and it is the reason this method exists. HDBSCAN is refitted on
        the growing buffer every batch and it RENAMES its clusters each time: on four
        perfectly separated blobs whose membership did not change at all, three consecutive
        batches produced labels 2, then 3, then 0 for the same six groups. Every group
        therefore changed adapter every step -- churn 1.0 -- while the clustering was
        structurally identical.

        MEDS' kNN classify does not prevent this and is not meant to: MEDS uses the label to
        look up a cluster SIZE for reward shaping, where a permutation of names costs
        nothing. Selecting an ADAPTER by that name is a different use, and under it a
        permutation sends every expert somebody else's gradient.

        The fix is to match the new raw labels to the existing expert ids by OVERLAP on the
        buffered groups -- greedily, largest overlap first -- so an expert keeps the
        subpopulation it was fitted to. Raw labels that match nothing get a fresh expert,
        which is the honest outcome when a genuinely new behaviour appears.

        A clusterer that exposes no label state (a test double) falls back to the identity
        mapping, which is stable for any assign() that is itself stable.
        """
        state = getattr(self.clusterer, "_state", None)
        raw_labels = list(state.get("labels", [])) if isinstance(state, dict) else []
        if not raw_labels or len(raw_labels) != len(self._history):
            return
        overlap: dict[tuple[int, int], int] = {}
        for raw, stable in zip(raw_labels, self._history):
            if stable is None or int(raw) == -1 or int(stable) == -1:
                continue
            key = (int(raw), int(stable))
            overlap[key] = overlap.get(key, 0) + 1
        mapping: dict[int, int] = {}
        used: set[int] = set()
        for (raw, stable), _n in sorted(overlap.items(), key=lambda kv: (-kv[1], kv[0])):
            if raw in mapping or stable in used:
                continue
            mapping[raw] = stable
            used.add(stable)
        before = dict(self._raw_to_stable)
        self._raw_to_stable = mapping
        self._next_stable = max([*used, self._next_stable - 1, -1]) + 1
        # Recomputed for the fit that has just happened, so it describes the clusters the
        # roster cannot seat NOW rather than every one it ever turned away.
        self._overflow.clear()
        for raw in {int(r) for r in raw_labels if int(r) != -1}:
            self._stable_id(raw)
        if before and any(before.get(k) != v for k, v in self._raw_to_stable.items()):
            self.relabellings += 1
        self._history = [self._stable_id(int(r)) for r in raw_labels]

    def partition(
        self, features: np.ndarray, *, group_ids: Sequence[str] = ()
    ) -> Partition:
        """Label this batch, then fold it into the history for the next one.

        Args:
            features: ``(n_groups, d)`` behavioural vectors, one per group.
            group_ids: Stable identifiers, carried so churn can be measured.

        Returns:
            The batch's :class:`Partition`.

        Raises:
            PartitionUnavailable: If the clustering dependencies are missing. Wrapping the
                dependency error here rather than letting it escape means the caller sees
                one refusal type for "no partition was formed", whatever the cause.
        """
        from selfevo.clustering.meds import ClusteringUnavailable

        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError(f"features must be (n_groups, d) and non-empty, got {features.shape}")
        try:
            if self.clusterer.fitted:
                # The stabilised path, and the one every batch after the first takes: label
                # against the EXISTING fit, so a group keeps the expert it had, and only
                # then fold this batch into the history and refit for the next one.
                labels = tuple(
                    self._stable_id(int(self.clusterer.assign(v))) for v in features
                )
                basis = (
                    f"meds: kNN(k={self.clusterer.knn_k}, {self.clusterer.knn_metric}) "
                    f"against {self.clusterer.n_clusters} HDBSCAN clusters fitted over "
                    f"{self.batches_seen} previous batch(es), expert ids carried forward "
                    f"by overlap matching"
                )
                self._extend(features, labels)
                self.batches_seen += 1
                self.clusterer.fit()
                self._resync()
            elif self.batches_seen >= self.warmup_batches:
                # Cold start. There is no history to stabilise against, so the only two
                # options are to fit on this batch own features or to send the whole batch
                # to the shared adapter. Fitting is chosen once the warmup is over, because
                # the alternative silently costs the run its first real update -- and with
                # warmup_batches=0 it would cost EVERY batch its labels, since the warmup
                # branch never fits and the fitted branch is therefore never reached.
                self._extend(features, [None] * features.shape[0])
                self.batches_seen += 1
                self.clusterer.fit()
                self._resync()
                labels = tuple(
                    self._stable_id(int(self.clusterer.assign(v))) for v in features
                )
                self._history[-features.shape[0]:] = list(labels)
                basis = (
                    f"meds: COLD START, HDBSCAN fitted on batch {self.batches_seen} "
                    f"({self.clusterer.n_clusters} clusters), then labelled from that fit"
                )
            else:
                labels = (-1,) * features.shape[0]
                basis = (
                    f"meds: WARMUP, batch {self.batches_seen + 1} of {self.warmup_batches}; "
                    "no fit yet, so every group is on the shared adapter"
                )
                self._extend(features, labels)
                self.batches_seen += 1
        except ClusteringUnavailable as exc:
            raise PartitionUnavailable(
                f"MEDS clustering is unavailable ({exc}); the training venv deliberately "
                "does not carry hdbscan/scikit-learn, so a run configured for "
                "partition=meds must refuse rather than fall back to one adapter"
            ) from exc
        if self._overflow:
            # Present tense, and measured after this batch's refit rather than tallied over
            # the run: _resync rebuilds the mapping every batch, so the honest statement is
            # how many clusters the roster cannot seat NOW.
            basis += (
                f"; {len(self._overflow)} cluster(s) have no expert in the "
                f"{self.max_experts}-adapter roster and are on {SHARED_CLUSTER}"
            )
        out = Partition(labels=labels, basis=basis, group_ids=tuple(group_ids))
        self._previous = out
        return out

    @property
    def previous(self) -> Partition | None:
        """The last partition produced, for churn measurement."""
        return self._previous


def meds_partition(
    features: np.ndarray,
    *,
    clusterer=None,
    group_ids: Sequence[str] = (),
) -> Partition:
    """One-shot MEDS partition of a batch, for analysis rather than training.

    Fits HDBSCAN on this batch's own features and labels them directly. Distinct from
    :class:`MEDSPartitioner`, which is what a RUN needs: the run must keep labels stable
    across batches, whereas the interference probe looks at one batch and has no next batch
    for labels to be stable with respect to.

    Args:
        features: ``(n_groups, d)`` behavioural vectors.
        clusterer: Optional :class:`~selfevo.clustering.meds.MEDSClusterer`.
        group_ids: Optional stable identifiers.

    Returns:
        The batch's :class:`Partition`, with HDBSCAN noise as ``-1``.

    Raises:
        PartitionUnavailable: If the clustering dependencies are missing.
    """
    from selfevo.clustering.meds import ClusteringUnavailable, MEDSClusterer

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"features must be (n_groups, d) and non-empty, got {features.shape}")
    clusterer = clusterer if clusterer is not None else MEDSClusterer()
    try:
        for v in features:
            clusterer.add(v)
        clusterer.fit()
    except ClusteringUnavailable as exc:
        raise PartitionUnavailable(f"MEDS clustering is unavailable ({exc})") from exc
    labels = tuple(int(v) for v in clusterer._state["labels"][-features.shape[0]:])
    return Partition(
        labels=labels,
        basis=(
            f"meds: HDBSCAN(min_cluster_size={clusterer.min_cluster_size}, min_samples=1, "
            f"{clusterer.metric} after L2) fitted on this batch, backend "
            f"{clusterer.backend}"
        ),
        group_ids=tuple(group_ids),
    )


def partition_from_config(
    mode: str,
    *,
    n_groups: int,
    features: np.ndarray | None = None,
    partitioner: MEDSPartitioner | None = None,
    seed: int = 0,
    group_ids: Sequence[str] = (),
) -> Partition:
    """Select a TRAINING partition by config value: ``meds | random_matched | none``.

    Args:
        mode: One of :data:`TRAINING_PARTITIONS`.
        n_groups: Groups in the batch.
        features: Behavioural vectors; required for ``meds`` and for ``random_matched``,
            which matches the MEDS partition's sizes and therefore has to compute it.
        partitioner: The stateful MEDS partitioner. Required for ``meds`` and
            ``random_matched`` so labels stay stable across batches; passing ``None`` and
            getting a fresh one per batch is the silent no-op this argument exists to
            prevent.
        seed: Seed for the control, which draws only the groups it has not placed
            before -- the rest are carried forward on the partitioner's own control memory,
            so the control is as stable as the method it controls for.
        group_ids: Stable identifiers.

    Returns:
        The batch's :class:`Partition`.

    Raises:
        ValueError: On an unknown mode, naming the ones that exist, and on a mode that IS
            registered in :data:`TRAINING_PARTITIONS` but has no dispatch branch below.
        PartitionUnavailable: If the mode needs inputs that were not supplied. The control
            is REFUSED rather than approximated: a "size-matched" control whose sizes were
            guessed rather than matched is not a control.
    """
    if mode == "none":
        return no_partition(n_groups, group_ids=group_ids)
    if mode not in TRAINING_PARTITIONS:
        raise ValueError(
            f"unknown cluster_lora.partition={mode!r}; expected one of {list(TRAINING_PARTITIONS)}"
        )
    if features is None or partitioner is None:
        raise PartitionUnavailable(
            f"partition={mode!r} needs both behavioural features and a persistent "
            "MEDSPartitioner; without them the only thing this could return is one "
            "adapter for everything, which is the 'none' arm wearing the method's label"
        )
    meds = partitioner.partition(features, group_ids=group_ids)
    if mode == "meds":
        return meds
    if mode == "random_matched":
        return random_matched_partition(
            meds, seed=seed, memory=partitioner.control_memory
        )
    # No unconditional fallthrough. This used to end in `return random_matched_partition(...)`,
    # so a fourth name added to TRAINING_PARTITIONS without a branch here ran the CONTROL's
    # mechanism under the new arm's label -- and since the returned Partition is bit-identical
    # to the control's, nothing downstream could tell afterwards which one produced the table.
    raise ValueError(
        f"partition={mode!r} is registered in TRAINING_PARTITIONS but partition_from_config "
        f"has no branch for it; dispatched here: {DISPATCHED_PARTITIONS!r}. Add a branch "
        "rather than letting it fall through to the size-matched control, which would report "
        "this arm's label for the control's mechanism."
    )


def max_experts_for_roster(roster: Sequence[str]) -> int | None:
    """How many experts a fixed adapter roster can carry, refusing one that can carry none.

    ``adapter_roster()`` is read once, before FSDP shards the model and before the optimizer
    exists, so the adapters a run has are fixed at process start. :class:`MEDSPartitioner`'s
    expert ids are not: :meth:`MEDSPartitioner._stable_id` allocates a fresh one for every
    raw HDBSCAN label that overlap matching does not claim, and ``_next_stable`` is monotone
    non-decreasing, so the names emitted are both SPARSE and UNBOUNDED. Twelve batches over
    two live clusters were measured emitting ``cluster_1`` and ``cluster_3`` with
    ``_next_stable`` at 4 -- names a ``cluster_0,cluster_1,shared`` roster refuses, although
    the partition has exactly the two clusters that roster was sized for. On the accelerator
    that refusal arrives at whichever step HDBSCAN first splits off a fragment, i.e. it kills
    a long run mid-flight.

    BOUNDING the allocation is chosen over compressing the ids onto the roster. Compression
    would seat two behaviourally distinct clusters on ONE expert while ``n_clusters`` went on
    reporting two, which is the silent mislabelling this module refuses at every other seam.
    A bound is visible instead: the roster IS the capacity, a cluster it cannot seat goes to
    the shared adapter -- the treatment this module already gives a group it has no
    behavioural claim about -- and the partition's basis records how many did.

    Args:
        roster: Adapter names in order, as ``adapter_roster()`` returns them. Empty means no
            roster was configured, and the partitioner is left unbounded exactly as before.

    Returns:
        The number of ``cluster_<i>`` experts, or ``None`` for an empty roster.

    Raises:
        ValueError: If the roster names no ``cluster_<i>`` expert, if its numbers are not
            exactly ``0..N-1``, or if it has no shared adapter. All three are refused HERE,
            at construction, rather than at the step that first needs the missing name: a
            bounded partitioner allocates densely from zero, so a gap leaves an expert no
            cluster can ever be given, and a roster with no shared adapter has nowhere to put
            either HDBSCAN noise or a cluster the bound turns away.
    """
    names = tuple(roster)
    if not names:
        return None
    prefix = "cluster_"
    numbers = sorted(
        int(n[len(prefix):])
        for n in names
        if n.startswith(prefix) and n[len(prefix):].isdigit()
    )
    if not numbers:
        raise ValueError(
            f"the adapter roster {list(names)} names no {prefix}<i> expert, so a partition "
            "would have nothing but the shared adapter to route to and the arm would be the "
            "baseline under the method's name"
        )
    if numbers != list(range(len(numbers))):
        raise ValueError(
            f"the adapter roster {list(names)} numbers its experts {numbers}; a bounded "
            f"partitioner allocates ids densely from 0, so a gap leaves an expert that no "
            f"cluster can ever be given. Name them {prefix}0..{prefix}{len(numbers) - 1}"
        )
    if SHARED_CLUSTER not in names:
        raise ValueError(
            f"the adapter roster {list(names)} has no {SHARED_CLUSTER!r} adapter. HDBSCAN "
            f"noise goes there, and so does any cluster beyond the roster's {len(numbers)} "
            "experts; without it a run dies at whichever step first produces either"
        )
    return len(numbers)


def label_churn(
    previous: Partition | None, current: Partition
) -> tuple[float, int, int]:
    """How many groups changed adapter between two consecutive batches.

    Measured over the groups present in BOTH batches, keyed by ``group_ids`` -- stable
    prompt identity, never batch position, which is reshuffled every step and would make
    churn read as ~1.0 whatever the clustering did.

    Args:
        previous: The previous batch's partition, or ``None`` for the first batch.
        current: This batch's partition.

    Returns:
        ``(churn_fraction, n_changed, n_overlap)``. With no overlap the fraction is 0.0 and
        ``n_overlap`` is 0, so a caller can tell "nothing moved" from "nothing was
        comparable" instead of reading a reassuring zero.

    Raises:
        ValueError: If either partition carries labels but no group ids, since churn
            against positions is not a measurement.
    """
    if previous is None:
        return 0.0, 0, 0
    for p, which in ((previous, "previous"), (current, "current")):
        if p.n_groups and not p.group_ids:
            raise ValueError(
                f"the {which} partition has no group ids; churn measured against batch "
                "positions is noise, because the batch is reshuffled every step"
            )
    before = dict(zip(previous.group_ids, previous.keys))
    after = dict(zip(current.group_ids, current.keys))
    overlap = [g for g in after if g in before]
    if not overlap:
        return 0.0, 0, 0
    changed = sum(1 for g in overlap if before[g] != after[g])
    return changed / len(overlap), changed, len(overlap)
