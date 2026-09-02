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
    SHARED_CLUSTER,
    Partition,
    PartitionUnavailable,
    balanced_assign,
    cluster_key,
    label_churn,
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
    """
    ref = part([0, 0, 0, 0, 1, 1, 1, 1])
    differs = sum(
        random_matched_partition(ref, seed=s).labels != ref.labels for s in range(50)
    )
    assert differs >= 45, f"only {differs}/50 seeds moved any label"


def test_the_control_is_reproducible_under_its_seed():
    ref = part([0, 0, 1, 1, -1, -1])
    assert random_matched_partition(ref, seed=3).labels == \
        random_matched_partition(ref, seed=3).labels
    assert random_matched_partition(ref, seed=3).labels != \
        random_matched_partition(ref, seed=4).labels


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
