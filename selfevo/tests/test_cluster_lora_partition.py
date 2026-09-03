"""The partition layer: the control, the calibration, the churn metric, and the refusals.

The control is the reason this file exists. A gain from per-cluster adapters has an obvious
rival explanation -- more adapters, more capacity -- and the only thing that separates them
is a partition with the same N and the same sizes whose labels saw no features. So the
size-match is asserted EXACTLY, on adversarial size distributions, rather than checked on one
convenient batch.

Everything here is pure numpy. The clustering tests that need scikit-learn are in
``test_cluster_lora_clustering.py`` so that this file runs in the training venv, which
deliberately has neither hdbscan nor scikit-learn installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from selfevo.cluster_lora.partition import (
    DEFAULT_CONTROL_MEMORY,
    SHARED_CLUSTER,
    MatchedControlMemory,
    MEDSPartitioner,
    Partition,
    PartitionUnavailable,
    balanced_assign,
    cluster_key,
    label_churn,
    max_experts_for_roster,
    no_partition,
    partition_from_config,
    random_matched_partition,
    task_partition,
)
from selfevo.cluster_lora.reach import reach_report


def part(labels, ids=None):
    """A partition with stable ids, since churn against positions is not a measurement."""
    ids = ids if ids is not None else [f"p{i}" for i in range(len(labels))]
    return Partition(labels=tuple(labels), basis="test", group_ids=tuple(ids))


# ------------------------------------------------------------------- the naming ---------


def test_noise_goes_to_the_shared_adapter_not_to_a_private_one():
    """A group HDBSCAN declined to place has no behavioural claim attached to it.

    Giving it a private expert would fit an adapter to one group, which is the opposite of
    what the shared adapter is for.
    """
    assert cluster_key(-1) == SHARED_CLUSTER
    assert cluster_key(0) == "cluster_0"


def test_there_is_no_second_noise_label():
    with pytest.raises(ValueError, match=">= -1"):
        cluster_key(-2)


def test_adapters_sort_numerically_not_lexically():
    """cluster_10 must not sort between cluster_1 and cluster_2.

    If it did, which adapter a cluster owns would depend on how many clusters were found,
    and an expert would silently change identity between steps.
    """
    p = part([0, 1, 2, 10, 11, -1])
    assert p.adapters == (
        "cluster_0", "cluster_1", "cluster_2", "cluster_10", "cluster_11", SHARED_CLUSTER,
    )


# --------------------------------------------------------------- the mandatory control ---


@pytest.mark.parametrize(
    "labels",
    [
        [0, 0, 1, 1, 2, 2],
        [0] * 15 + [1],                      # one lopsided cluster
        [-1] * 8,                            # all noise
        [0, 1, 2, 3, 4, 5, 6, 7],            # every group its own cluster
        [-1, -1, 0, 0, 0, 1, 2, 2, 2, 2],    # noise plus uneven clusters
    ],
    ids=["even", "lopsided", "all-noise", "singletons", "mixed"],
)
def test_the_control_matches_the_sizes_exactly_on_every_shape(labels):
    """Exactly, not in expectation, and on the shapes that break a sampled control.

    A control that sampled from the observed proportions would match on the even case and
    drift on the others; the drift is largest exactly where the clusters are most unequal,
    which is where the method's claim lives.
    """
    ref = part(labels)
    ctrl = random_matched_partition(ref, seed=7)
    assert ctrl.size_multiset() == ref.size_multiset()
    assert ctrl.n_clusters == ref.n_clusters
    assert ctrl.n_noise == ref.n_noise
    assert ctrl.n_groups == ref.n_groups


def test_the_control_is_actually_feature_blind():
    """It must not reproduce the reference assignment, or it is not a control.

    Asserted over many seeds rather than one: a single seed can permute to the identity by
    chance, and a test that happened to draw it would enshrine a control that controls
    nothing.

    Each seed draws into its OWN memory. Feature-blindness is a property of the DRAW, and
    since the control became identity-stable the draw happens once per group identity; fifty
    seeds sharing one memory would measure the carry rather than the draw, and would pass
    even if the draw had stopped being blind. That the CARRY is blind too is a property of
    where its labels come from -- they are labels this same draw produced -- and is asserted
    against a moving reference in the size-match test below.
    """
    ref = part([0, 0, 0, 0, 1, 1, 1, 1])
    differs = sum(
        random_matched_partition(ref, seed=s, memory=MatchedControlMemory()).labels
        != ref.labels
        for s in range(50)
    )
    assert differs >= 45, f"only {differs}/50 seeds moved any label"


def test_the_control_is_reproducible_under_its_seed():
    """The seed governs the DRAW. What has already been drawn is governed by the memory.

    Against a fresh memory the seed decides everything, which is what this test has always
    asserted and what still has to hold: two seeds that produced the same control would make
    a seed sweep of the control arm one run repeated. Against a memory that has already
    placed these groups the seed deliberately does NOT move them -- that is the churn fix,
    and it is asserted in
    ``test_the_control_keeps_a_group_on_its_adapter_when_the_seed_moves``.
    """

    def draw(seed):
        """One control over the same reference, from an empty memory."""
        return random_matched_partition(
            part([0, 0, 1, 1, -1, -1]), seed=seed, memory=MatchedControlMemory()
        ).labels

    assert draw(3) == draw(3)
    assert draw(3) != draw(4)


# The control's THIRD property, and the one it did not have. Sizes and feature-blindness are
# asserted above; both passed while the control re-permuted every batch, so neither could see
# that the control also destroyed expert identity -- the axis findings 5.1 makes the whole
# method rest on. See MatchedControlMemory.


def test_the_control_keeps_a_group_on_its_adapter_when_the_seed_moves():
    """A group the control has placed stays there, however the per-batch seed advances.

    ``ClusterLoRAKeyFn.begin_batch`` passes ``seed + self.batches``, so the seed moves every
    batch by construction. Before the memory existed that alone re-permuted every group.
    """
    ref = part([0, 0, 0, 1, 1, 1, -1, -1])
    memory = MatchedControlMemory()
    first = random_matched_partition(ref, seed=0, memory=memory)
    for batch in range(1, 6):
        later = random_matched_partition(ref, seed=batch, memory=memory)
        assert later.labels == first.labels, f"batch {batch} re-permuted the control"
    assert len(memory) == ref.n_groups


def test_the_control_churns_no_more_than_the_method_it_controls_for():
    """Churn parity: the mandatory control may not differ from the arm on stability.

    Two references with the same identities. The first never changes, so the method's churn
    is 0.0 and the control's must be too. In the second the METHOD moves every group to a
    different cluster while keeping its size multiset -- the control must not follow it
    (that would make the control feature-sighted) and must not be shaken loose by it either.

    This is the assertion that stops the defect coming back: re-permuting per batch passes
    the size test and the feature-blindness test, and fails only here.
    """
    ids = tuple(f"parity{i}" for i in range(12))
    stable = [part([0, 0, 0, 1, 1, 1, 2, 2, 2, -1, -1, -1], ids=ids) for _ in range(4)]
    memory = MatchedControlMemory()
    control = [random_matched_partition(p, seed=k, memory=memory) for k, p in enumerate(stable)]
    method_churn = [label_churn(stable[k - 1], stable[k])[0] for k in (1, 2, 3)]
    control_churn = [label_churn(control[k - 1], control[k])[0] for k in (1, 2, 3)]
    assert max(method_churn) == 0.0, "the fixture is not stable, so it proves nothing"
    assert control_churn == method_churn, (
        f"the control churned {control_churn} against the method's {method_churn} on a "
        "partition whose membership never changed"
    )

    reshuffled = part([2, 2, 2, -1, -1, -1, 0, 0, 0, 1, 1, 1], ids=ids)
    after = random_matched_partition(reshuffled, seed=99, memory=memory)
    assert label_churn(control[-1], after)[0] == 0.0, (
        "the control moved when the METHOD's labels moved; the control does not see the "
        "method's assignment and must not track it"
    )


def test_the_control_stays_size_matched_when_the_memory_cannot_be_honoured():
    """A carried label the new batch has no room for is redrawn, and the sizes still match.

    The pins are honoured by CAPACITY, not by membership, so a reference whose clusters have
    shrunk, grown or vanished cannot make the control drift off the size multiset -- which is
    the property the control had before the memory existed and must not have lost to it.
    """
    ids = tuple(f"cap{i}" for i in range(9))
    memory = MatchedControlMemory()
    random_matched_partition(part([0] * 3 + [1] * 3 + [2] * 3, ids=ids), seed=1, memory=memory)
    for labels in ([0] * 8 + [1], [-1] * 9, [0, 1, 2, 3, 4, 5, 6, 7, 8], [0] * 4 + [1] * 5):
        ref = part(labels, ids=ids)
        ctrl = random_matched_partition(ref, seed=2, memory=memory)
        assert ctrl.size_multiset() == ref.size_multiset()
        assert ctrl.n_clusters == ref.n_clusters
        assert ctrl.n_noise == ref.n_noise


def test_a_reference_with_no_identities_is_drawn_afresh_every_time():
    """No group ids means nothing to carry, and the memory is left empty rather than keyed
    on batch position -- which is reshuffled every step, so a memory keyed on it would pin
    each POSITION to an adapter and hand the group at that position another group's expert.
    """
    ref = Partition(labels=(0, 0, 1, 1, -1, -1), basis="no identities")
    memory = MatchedControlMemory()
    assert random_matched_partition(ref, seed=3, memory=memory).labels != \
        random_matched_partition(ref, seed=4, memory=memory).labels
    assert len(memory) == 0


def test_the_default_memory_is_shared_and_resettable():
    """The production call passes no memory, so the default is the one that must carry.

    ``partition_from_config`` hands the partitioner's own memory, but
    ``interference_analyze`` and every direct caller do not, and a default that did not
    carry would leave those on the defective control.
    """
    ids = tuple(f"default{i}" for i in range(8))
    ref = part([0, 0, 0, 0, 1, 1, 1, 1], ids=ids)
    DEFAULT_CONTROL_MEMORY.reset()
    first = random_matched_partition(ref, seed=0)
    assert random_matched_partition(ref, seed=1).labels == first.labels
    DEFAULT_CONTROL_MEMORY.reset()
    assert len(DEFAULT_CONTROL_MEMORY) == 0


def test_a_memory_that_remembers_nothing_is_refused():
    """The defect, expressed as a configuration, is not accepted as one."""
    with pytest.raises(ValueError, match="max_entries"):
        MatchedControlMemory(max_entries=0)


def test_the_memory_is_bounded_and_drops_the_oldest_first():
    """A run must not grow a dict for every prompt it has ever seen without a ceiling."""
    memory = MatchedControlMemory(max_entries=4)
    for batch in range(3):
        ids = tuple(f"b{batch}g{i}" for i in range(4))
        random_matched_partition(part([0, 0, 1, 1], ids=ids), seed=batch, memory=memory)
    assert len(memory) == 4


def test_the_control_records_what_it_matched():
    """The basis is the audit record; a control that does not say so is not auditable."""
    ctrl = random_matched_partition(part([0, 0, 1]), seed=1)
    assert "random_matched" in ctrl.basis and "seed 1" in ctrl.basis


def test_an_empty_reference_is_refused():
    with pytest.raises(ValueError, match="no groups"):
        random_matched_partition(Partition(labels=(), basis="empty"), seed=0)


# ----------------------------------------------------------- the cross-task calibration ---


def test_a_single_task_batch_is_refused_rather_than_reported_as_zero():
    """One cluster has no pairs, and an empty mean reads as 0.0.

    That is the most dangerous possible output here: it would look exactly like the
    published ~1e-5 cross-task figure being reproduced, from a batch that could not test it.
    """
    with pytest.raises(PartitionUnavailable, match="spans 1 task"):
        task_partition(["math"] * 6)


def test_task_labels_become_clusters_in_first_appearance_order():
    p = task_partition(["code", "math", "code", "math", "code"])
    assert p.labels == (0, 1, 0, 1, 0)
    assert p.n_clusters == 2 and p.n_noise == 0


# --------------------------------------------------------------------- size matching ----


def test_balanced_assign_respects_capacity_exactly():
    d = np.array([[0.0, 9.0], [0.1, 9.0], [0.2, 9.0], [0.3, 9.0]])
    out = balanced_assign(d, [2, 2])
    assert sorted(np.bincount(out, minlength=2).tolist()) == [2, 2]
    # The two closest rows must still win the cheap column.
    assert out[0] == 0 and out[1] == 0


def test_capacities_that_do_not_partition_the_batch_are_refused():
    with pytest.raises(ValueError, match="capacities sum"):
        balanced_assign(np.zeros((4, 2)), [1, 1])


# ------------------------------------------------------------------------- churn --------


def test_churn_is_measured_against_prompt_identity_not_batch_position():
    """The batch is reshuffled every step, so positional churn would read as noise.

    Here the same prompts keep the same adapters but appear in a different ORDER; churn
    must be 0.
    """
    a = part([0, 1, 0, 1], ids=["p0", "p1", "p2", "p3"])
    b = part([1, 0, 1, 0], ids=["p1", "p0", "p3", "p2"])
    churn, changed, overlap = label_churn(a, b)
    assert (churn, changed, overlap) == (0.0, 0, 4)


def test_churn_sees_a_group_that_changed_adapter():
    a = part([0, 0, 1, 1], ids=list("abcd"))
    b = part([0, 1, 1, 1], ids=list("abcd"))
    churn, changed, overlap = label_churn(a, b)
    assert (changed, overlap) == (1, 4) and churn == 0.25


def test_no_overlap_reports_zero_overlap_not_zero_churn():
    """"Nothing moved" and "nothing was comparable" must not read the same.

    A run whose prompts never recur would otherwise report perfect stability.
    """
    a = part([0, 1], ids=["a", "b"])
    b = part([0, 1], ids=["c", "d"])
    churn, changed, overlap = label_churn(a, b)
    assert overlap == 0 and changed == 0 and churn == 0.0


def test_churn_against_a_partition_with_no_ids_is_refused():
    a = Partition(labels=(0, 1), basis="x")
    with pytest.raises(ValueError, match="no group ids"):
        label_churn(a, part([0, 1]))


def test_the_first_batch_has_no_churn_to_measure():
    assert label_churn(None, part([0, 1])) == (0.0, 0, 0)


# ------------------------------------------------------------------ config selection ----


def test_the_none_arm_puts_every_group_on_one_adapter():
    """The vanilla LoRA baseline, expressed in the same type so it shares the code path."""
    p = no_partition(5)
    assert p.keys == (SHARED_CLUSTER,) * 5
    assert p.n_clusters == 0 and p.n_noise == 5


def test_an_empty_batch_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        no_partition(0)


def test_an_unknown_partition_name_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="unknown cluster_lora.partition"):
        partition_from_config("kmeans", n_groups=4)


def test_a_registered_partition_with_no_dispatch_branch_is_refused(monkeypatch):
    """The arm-mislabelling trap: a name in TRAINING_PARTITIONS with no branch here.

    The dispatch ended in an unconditional ``return random_matched_partition(...)``, so a
    third arm added to the registry ran the CONTROL's mechanism under its own label. The
    partition it returns is bit-identical to ``random_matched`` and every metric downstream
    carries the new arm's name, so nothing afterwards could tell the two apart -- which is
    exactly the failure this module's docstring exists to prevent.
    """
    from selfevo.cluster_lora import partition as pmod

    monkeypatch.setattr(
        pmod, "TRAINING_PARTITIONS", ("meds", "random_matched", "none", "meds_frozen")
    )
    feats = np.arange(12, dtype=float).reshape(6, 2)
    ids = tuple(f"p{i}" for i in range(6))
    with pytest.raises(ValueError, match="no branch"):
        pmod.partition_from_config(
            "meds_frozen",
            n_groups=6,
            features=feats,
            partitioner=pmod.MEDSPartitioner(),
            seed=1,
            group_ids=ids,
        )


@pytest.mark.parametrize("mode", ["meds", "random_matched", "none"])
def test_every_registered_partition_still_dispatches(mode):
    """The guard above must refuse only the undispatched names, not the real arms."""
    from selfevo.cluster_lora import partition as pmod

    feats = np.arange(12, dtype=float).reshape(6, 2)
    ids = tuple(f"p{i}" for i in range(6))
    got = pmod.partition_from_config(
        mode,
        n_groups=6,
        features=feats,
        partitioner=pmod.MEDSPartitioner(),
        seed=1,
        group_ids=ids,
    )
    assert len(got.labels) == 6


@pytest.mark.parametrize("mode", ["meds", "random_matched"])
def test_a_partition_without_features_refuses_instead_of_collapsing_to_one_adapter(mode):
    """The silent no-op this whole design exists to prevent.

    Without features the only thing these modes could return is one adapter for everything
    -- the 'none' arm wearing the method's label, training fine and reporting fine.
    """
    with pytest.raises(PartitionUnavailable, match="wearing the method"):
        partition_from_config(mode, n_groups=4, features=None, partitioner=None)


# -------------------------------------------------------------------- reach report ------


def test_reach_reports_every_cluster_size_separately():
    """Two clusters of 32 and eight of eight give the same total and mean opposite things."""
    r = reach_report(part([0, 0, 0, 1, -1]))
    m = r.as_metrics()
    assert m["cluster_lora/size/cluster_0"] == 3.0
    assert m["cluster_lora/size/cluster_1"] == 1.0
    assert m["cluster_lora/size/shared"] == 1.0
    assert m["cluster_lora/n_clusters"] == 2.0


def test_an_all_noise_batch_reports_the_method_doing_nothing():
    """noise_fraction 1.0 is the signature of a run that is vanilla LoRA in disguise."""
    r = reach_report(part([-1] * 8))
    assert r.noise_fraction == 1.0
    assert r.largest_cluster_fraction == 1.0
    assert r.as_metrics()["cluster_lora/n_clusters"] == 0.0


def test_the_largest_cluster_fraction_counts_the_shared_bucket():
    """95% noise and 95% one cluster are the same failure to the gradient.

    A metric that excluded noise would report the first as perfectly balanced.
    """
    r = reach_report(part([-1] * 19 + [0]))
    assert r.largest_cluster_fraction == pytest.approx(0.95)


def test_a_report_whose_sizes_lost_a_group_is_refused():
    with pytest.raises(ValueError, match="dropped or double-counted"):
        type(reach_report(part([0, 1])))(
            n_groups=3, n_clusters=2, sizes={"cluster_0": 1, "cluster_1": 1},
            n_noise=0, churn=0.0, n_churn_overlap=0, basis="x",
        )


def test_reach_carries_the_churn_between_two_batches():
    a = part([0, 0, 1, 1], ids=list("abcd"))
    b = part([0, 1, 1, 1], ids=list("abcd"))
    r = reach_report(b, a)
    assert r.churn == 0.25 and r.n_churn_overlap == 4
    assert r.as_metrics()["cluster_lora/churn"] == 0.25


def test_the_control_refuses_if_its_own_permutation_stopped_matching(monkeypatch):
    """The last line of defence, which no other test can reach.

    "Matched by construction" is exactly the kind of claim that survives a refactor while
    quietly stopping being true, so the control checks its own sizes. Here the permutation is
    replaced by one that drops a label, and the refusal has to fire.
    """
    class Bad:
        """A generator whose permutation is not a permutation."""

        def permutation(self, labels):
            """Return a same-length array with a different multiset."""
            out = np.array(labels).copy()
            out[0] = 99
            return out

    monkeypatch.setattr(np.random, "default_rng", lambda *_a, **_k: Bad())
    with pytest.raises(PartitionUnavailable, match="not size-matched"):
        random_matched_partition(part([0, 0, 1, 1]), seed=0)


# ------------------------------------------- the roster the emitted names have to fit ---
#
# The expert ids are allocated one per unseen raw HDBSCAN label and _next_stable is monotone,
# while adapter_roster() is fixed at process start. So the names emitted are sparse and
# unbounded, and an unbounded run is expected to name an adapter it does not have partway
# through. Driven here through a scripted clusterer rather than hdbscan: the allocation is
# ours, the labels are the library's, and only the allocation is under test.


class ScriptedClusterer:
    """Emits the raw labels it is handed, so ID ALLOCATION can be driven without hdbscan.

    Carries no ``_state``, so ``_resync`` takes its documented identity fallback -- which is
    exactly the state in which every unseen raw label costs one fresh expert id, the
    allocation these tests are about. The real clusterer is exercised in
    ``test_cluster_lora_clustering.py`` under the probe venv.
    """

    knn_k, knn_metric, min_cluster_size, metric, backend = 3, "cosine", 2, "euclidean", "stub"
    n_clusters = 2

    def __init__(self, labels):
        self.queue = list(labels)
        self.fitted = False

    def add(self, v):
        """Accept a vector, as the real clusterer does; the labels are scripted."""

    def fit(self):
        """Mark fitted, as the real clusterer does."""
        self.fitted = True
        return self.n_clusters

    def assign(self, v):
        """The next scripted raw label."""
        return self.queue.pop(0)


def drive(partitioner, batches, width=4):
    """Run ``batches`` batches of ``width`` groups through a partitioner."""
    ids = tuple(f"g{i}" for i in range(width))
    return [
        partitioner.partition(np.zeros((width, 1)), group_ids=ids) for _ in range(batches)
    ]


def test_an_unbounded_partitioner_names_experts_a_fixed_roster_does_not_have():
    """The hazard, reproduced: two clusters per batch, and by batch two the names have moved.

    This is what kills a long run. ``begin_cluster_batch`` refuses a partition naming an
    adapter outside ``SELFEVO_CLUSTER_LORA_ADAPTERS``, and the roster cannot grow because
    every expert is created before FSDP shards the model and before the optimizer exists.
    """
    p = MEDSPartitioner(ScriptedClusterer([5, 5, 7, 7, 9, 9, 11, 11]), warmup_batches=0)
    second = drive(p, 2)[1]
    assert second.n_clusters == 2
    assert set(second.keys) == {"cluster_2", "cluster_3"}, (
        "the fixture no longer reproduces the drift these tests are about"
    )
    assert not set(second.keys) <= {"cluster_0", "cluster_1", SHARED_CLUSTER}


def test_a_bounded_partitioner_never_names_an_expert_outside_the_roster():
    """The fix: the roster IS the capacity, and a cluster it cannot seat goes to shared."""
    roster = ("cluster_0", "cluster_1", SHARED_CLUSTER)
    p = MEDSPartitioner(
        ScriptedClusterer(range(5, 200)),
        warmup_batches=0,
        max_experts=max_experts_for_roster(roster),
    )
    for part in drive(p, 12):
        assert set(part.keys) <= set(roster), f"emitted {sorted(set(part.keys))}"


def test_the_clusters_the_roster_turned_away_are_counted_and_recorded():
    """Folding a cluster into shared is a real loss of capacity, so it is not silent.

    A run whose clusters are being folded into the shared adapter trains fewer experts than
    its config names, and ``n_clusters`` alone cannot tell that from a clustering that found
    fewer clusters.
    """
    p = MEDSPartitioner(
        ScriptedClusterer(range(5, 200)), warmup_batches=0, max_experts=2
    )
    last = drive(p, 6)[-1]
    assert p.overflow_clusters, "the roster was exhausted and nothing recorded it"
    assert f"have no expert in the 2-adapter roster and are on {SHARED_CLUSTER}" in last.basis


def test_an_unbounded_partitioner_turns_nothing_away():
    """Default off: with no bound the allocation is bit-identical to what it always was."""
    p = MEDSPartitioner(ScriptedClusterer(range(5, 200)), warmup_batches=0)
    drive(p, 6)
    assert p.max_experts is None
    assert p.overflow_clusters == ()


class ResyncingClusterer(ScriptedClusterer):
    """A scripted clusterer that also exposes the label state ``_resync`` reads.

    With ``_state`` present the partitioner takes its overlap-matching path, which rebuilds
    the raw-label-to-expert mapping after every refit. The roster bound has to be recomputed
    with it, or a cluster the roster turned away once stays counted long after HDBSCAN
    stopped producing it, and the basis reports capacity the run is not actually losing.
    """

    def __init__(self, labels, buffer=4):
        super().__init__(labels)
        self.buffer = int(buffer)
        self._state = {"vectors": [], "labels": []}
        self.given = []

    def add(self, v):
        """Buffer the vector, trimming from the FRONT as the real clusterer does."""
        self._state["vectors"].append(np.asarray(v, dtype=float))
        del self._state["vectors"][: max(0, len(self._state["vectors"]) - self.buffer)]

    def assign(self, v):
        """The next scripted label, kept so ``fit`` can publish the same ones."""
        got = super().assign(v)
        self.given.append(got)
        return got

    def fit(self):
        """Publish the labels of everything still buffered, as a real refit does."""
        self._state["labels"] = list(self.given)[-len(self._state["vectors"]):]
        return super().fit()


def test_a_cluster_the_roster_turned_away_stops_being_counted_once_it_is_gone():
    """The overflow set describes the LIVE fit, not the run's history.

    Batch two produces a third cluster the two-expert roster cannot seat. By batch three it
    has aged out of the clusterer's buffer, which is trimmed from the front exactly as MEDS'
    is, so the refit no longer knows about it. A count that survived that refit would report
    capacity the run is no longer losing, and would go on reporting it for the rest of the
    run.
    """
    script = [10, 10, 11, 11] + [10, 10, 11, 99] + [10, 10, 11, 11]
    p = MEDSPartitioner(ResyncingClusterer(script), warmup_batches=0, max_experts=2)
    batches = drive(p, 3)
    assert batches[1].keys == ("cluster_0", "cluster_0", "cluster_1", SHARED_CLUSTER)
    assert "have no expert in the 2-adapter roster" in batches[1].basis
    assert batches[2].keys == ("cluster_0", "cluster_0", "cluster_1", "cluster_1")
    assert p.overflow_clusters == (), "a cluster HDBSCAN no longer produces is still counted"
    assert "have no expert" not in batches[2].basis


def test_a_roster_that_cannot_seat_one_cluster_is_refused_at_construction():
    """Before the accelerators are allocated, not at the step that first needs the name."""
    with pytest.raises(ValueError, match="max_experts"):
        MEDSPartitioner(ScriptedClusterer([]), max_experts=0)


@pytest.mark.parametrize(
    "roster, expected",
    [
        ((), None),
        (("cluster_0", "cluster_1", SHARED_CLUSTER), 2),
        ((SHARED_CLUSTER, "cluster_0"), 1),
        (tuple(f"cluster_{i}" for i in range(8)) + (SHARED_CLUSTER,), 8),
    ],
)
def test_the_roster_bound_counts_the_experts_it_actually_has(roster, expected):
    assert max_experts_for_roster(roster) == expected


@pytest.mark.parametrize(
    "roster, match",
    [
        ((SHARED_CLUSTER,), "names no cluster_<i> expert"),
        (("cluster_0", "cluster_2", SHARED_CLUSTER), "numbers its experts"),
        (("cluster_0", "cluster_1"), f"no {SHARED_CLUSTER!r} adapter"),
    ],
    ids=["no-expert", "sparse-roster", "no-shared"],
)
def test_a_roster_that_cannot_carry_the_partition_is_refused_by_name(roster, match):
    """Each refusal names the fix, because each has a different one.

    A gap in the numbering leaves an expert no cluster can ever be given, since a bounded
    partitioner allocates densely from zero; a roster with no shared adapter has nowhere to
    put HDBSCAN noise OR a cluster the bound turns away, and would die at whichever comes
    first.
    """
    with pytest.raises(ValueError, match=match):
        max_experts_for_roster(roster)


def test_every_mutation_anchor_still_occurs_exactly_once():
    """A refactor that invalidates an anchor turns a mutation into a silent SKIP.

    This happened: rewriting the dump's loss for the 32B OOM left one anchor matching zero
    lines, and the harness reported ``anchor appears 0x`` -- correctly counted as NOT a kill,
    but only because someone read the output. A guard that is never exercised looks exactly
    like a guard that passes, so the anchors are checked here instead of being trusted to a
    line in a log.

    Source-only, so it runs under both interpreters and needs neither torch nor scikit-learn.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    harness = root / "selfevo/tests/mutate_cluster_lora.py"
    names = {
        "ADAPTERS": "adapters.py", "MERGE": "merge.py", "PARTITION": "partition.py",
        "SKETCH": "sketch.py", "DUMP": "interference_dump.py",
        "ANALYZE": "interference_analyze.py",
    }
    sources = {
        k: (root / "selfevo/cluster_lora" / v).read_text() for k, v in names.items()
    }
    bad, checked = [], 0
    for node in ast.walk(ast.parse(harness.read_text())):
        if not (isinstance(node, ast.Tuple) and len(node.elts) == 5):
            continue
        if not isinstance(node.elts[0], ast.Name) or node.elts[0].id not in sources:
            continue
        try:
            label = ast.literal_eval(node.elts[2])
            find = ast.literal_eval(node.elts[3])
        except ValueError:
            continue
        checked += 1
        n = sources[node.elts[0].id].count(find)
        if n != 1:
            bad.append((node.elts[0].id, label, n))
    assert checked > 50, f"only {checked} anchors were parsed; the harness shape changed"
    assert not bad, bad
