"""The MEDS clustering lifecycle, against the REAL hdbscan/scikit-learn.

``test_cluster_lora_features.py`` exercises the lifecycle over a stub, because the training
venv deliberately carries neither dependency and installing them under a live job is the risk
this project refuses. This file is the other half: the same lifecycle over the vendored
clustering itself, run under ``~/venv_probe``.

Both are needed. The stub proves the part we wrote -- warm up, label against history, only
then add and refit. This file proves the vendored part behaves as that lifecycle assumes,
and it is where two facts about HDBSCAN that the method depends on are pinned:

* the pipeline L2-NORMALISES before euclidean HDBSCAN, so only a feature's DIRECTION matters
  and a feature near the origin is degenerate;
* MEDS' shipped ``min_cluster_size=2`` OVER-FRAGMENTS. Splitting a blob costs MEDS nothing --
  it shapes a reward with the result -- but here every extra cluster is another expert
  trained on fewer groups.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from selfevo.clustering.meds import MEDSClusterer  # noqa: E402
from selfevo.cluster_lora.features import ClusterLoRAKeyFn  # noqa: E402
from selfevo.cluster_lora.partition import (  # noqa: E402
    MEDSPartitioner,
    meds_partition,
)

K, PER = 4, 6
# Directional, never centred on the origin: the vendored path L2-normalises first.
CENTRES = np.array([[5.0, 0.0], [0.0, 5.0], [-5.0, 0.0], [0.0, -5.0]])


def blobs(seed=0, noise=0.05, per=PER):
    """Four well-separated directional blobs, with their true labels."""
    rng = np.random.default_rng(seed)
    labels = [k for k in range(K) for _ in range(per)]
    return np.stack([CENTRES[k] + noise * rng.normal(size=2) for k in labels]), labels


def test_hdbscan_recovers_well_separated_blobs_at_a_sane_min_cluster_size():
    """The premise. If the clustering cannot find planted structure, nothing below matters."""
    X, labels = blobs()
    p = meds_partition(X, clusterer=MEDSClusterer(min_cluster_size=PER))
    assert p.n_clusters == K and p.n_noise == 0
    assert p.size_multiset() == (PER,) * K
    # Same true cluster, same label -- the partition must be the planted one up to renaming.
    for k in range(K):
        got = {p.labels[i] for i in range(len(labels)) if labels[i] == k}
        assert len(got) == 1, f"true cluster {k} was split across {got}"


def test_the_shipped_min_cluster_size_of_two_over_fragments():
    """MEDS' own default, and it is the wrong default for allocating adapters.

    Recorded as a test rather than a note because the number of experts is DECIDED by this
    parameter, and a run that inherited it would spend its capacity on fragments.
    """
    X, _ = blobs()
    shipped = meds_partition(X, clusterer=MEDSClusterer(min_cluster_size=2))
    sane = meds_partition(X, clusterer=MEDSClusterer(min_cluster_size=PER))
    assert shipped.n_clusters > sane.n_clusters == K


def test_a_feature_at_the_origin_degenerates_under_the_l2_normalisation():
    """Only DIRECTION survives the vendored preprocessing, and this is how you find out.

    Two features that differ only in magnitude are the SAME point to this clustering, so a
    behavioural feature whose information is in its scale carries nothing here.
    """
    X = np.array([[5.0, 0.0], [50.0, 0.0], [0.0, 5.0], [0.0, 50.0]] * 3)
    p = meds_partition(X, clusterer=MEDSClusterer(min_cluster_size=3))
    assert p.n_clusters == 2, p.labels


def test_noise_is_labelled_minus_one_and_goes_to_the_shared_adapter():
    """One far-off outlier must not get a private expert fitted to it alone.

    The outlier is placed on a THIRD axis, orthogonal to every blob. A 2-D outlier at 45
    degrees was measured being absorbed into a neighbouring blob -- after L2 normalisation
    the whole feature space is a circle, and on a circle there is nowhere far from
    everything.
    """
    rng = np.random.default_rng(1)
    centres3 = np.hstack([CENTRES, np.zeros((K, 1))])
    labels = [k for k in range(K) for _ in range(4)]
    X = np.stack([centres3[k] + 0.05 * rng.normal(size=3) for k in labels])
    X = np.vstack([X, np.array([[0.0, 0.0, 5.0]])])
    p = meds_partition(X, clusterer=MEDSClusterer(min_cluster_size=4))
    assert p.keys[-1] == "shared", p.labels
    assert p.n_noise >= 1


def _run_batches(kf, X, n=4):
    """Drive several batches of the same population and return the churn seen at each."""
    ids = [f"p{i}" for i in range(len(X))]
    churns = []
    for step in range(n):
        jitter = X + 0.02 * np.random.default_rng(step).normal(size=X.shape)
        kf.begin_batch([f"{step}:{i}" for i in range(len(X))], jitter, group_ids=ids)
        if step:
            churns.append(kf.report().churn)
    return churns


def _keyfn(warmup=0):
    """A key_fn over the real clusterer at a min_cluster_size that recovers the blobs."""
    return ClusterLoRAKeyFn(
        MEDSPartitioner(MEDSClusterer(min_cluster_size=PER), warmup_batches=warmup),
        mode="meds",
    )


def test_the_partitioner_keeps_expert_identity_across_refits():
    """The stabilisation, over the real clusterer. Small churn, not zero, and not 1.0.

    MEASURED 2026-09-02. Without label matching this fixture churns at exactly 1.0 at every
    step -- see the counterfactual below -- because HDBSCAN RENAMES its clusters on each
    refit even when the membership is identical. With matching it churns at 0.0-0.09.

    The residual is real and is not a defect to hide: the buffer grows every batch, HDBSCAN
    finds more structure in more points, and a blob eventually splits off a fragment that
    some groups migrate into. That is the method's own behaviour, so it is bounded and
    reported rather than asserted away.
    """
    X, _ = blobs()
    churns = _run_batches(_keyfn(), X)
    assert churns, "no batch pair was comparable, so nothing was measured"
    assert max(churns) <= 0.15, churns


def test_without_the_label_matching_every_group_changes_expert_every_step():
    """The counterfactual that gives the test above its meaning.

    Disabling ``_resync`` reproduces the naive MEDS behaviour -- kNN classify against raw
    HDBSCAN labels -- and the churn goes to 1.0 with the clusters structurally unchanged.
    Without this comparison the bound above could be satisfied by a partitioner that never
    clustered anything.
    """
    X, _ = blobs()
    kf = _keyfn()
    kf.partitioner._resync = lambda: None  # the naive path
    churns = _run_batches(kf, X)
    assert max(churns) > 0.9, churns


def test_the_stabilisation_is_exercised_and_not_vacuous():
    """HDBSCAN must actually have renamed something, or the matching absorbed nothing."""
    X, _ = blobs()
    kf = _keyfn()
    _run_batches(kf, X)
    assert kf.partitioner.relabellings > 0


def test_churn_is_reported_when_a_group_genuinely_moves():
    """The zero above proves nothing unless a real move is visible."""
    X, _ = blobs()
    ids = [f"p{i}" for i in range(len(X))]
    kf = ClusterLoRAKeyFn(
        MEDSPartitioner(MEDSClusterer(min_cluster_size=PER), warmup_batches=0), mode="meds"
    )
    kf.begin_batch([f"0:{i}" for i in range(len(X))], X, group_ids=ids)
    moved = X.copy()
    moved[0] = CENTRES[2]  # one group changes behaviour to another cluster's direction
    kf.begin_batch([f"1:{i}" for i in range(len(X))], moved, group_ids=ids)
    r = kf.report()
    assert r.n_churn_overlap == len(X) and r.churn > 0.0


def test_the_backend_is_recorded_so_two_libraries_are_not_confused_for_two_results():
    """'the clustering differed' and 'the clustering LIBRARY differed' must not be confusable."""
    X, _ = blobs()
    c = MEDSClusterer(min_cluster_size=PER)
    meds_partition(X, clusterer=c)
    assert c.backend in ("hdbscan", "sklearn")


def test_the_warmup_batch_is_all_shared_over_the_real_clusterer_too():
    X, _ = blobs()
    kf = ClusterLoRAKeyFn(
        MEDSPartitioner(MEDSClusterer(min_cluster_size=PER), warmup_batches=1), mode="meds"
    )
    p = kf.begin_batch([f"0:{i}" for i in range(len(X))], X, group_ids=[f"p{i}" for i in range(len(X))])
    assert set(p.keys) == {"shared"} and "WARMUP" in p.basis
    # And the batch after the warmup does cluster.
    p2 = kf.begin_batch([f"1:{i}" for i in range(len(X))], X, group_ids=[f"p{i}" for i in range(len(X))])
    assert len(set(p2.keys)) > 1, p2.basis


def test_the_size_matched_control_runs_over_the_real_clusters_too():
    """The control has to match what HDBSCAN actually produced, not a tidy assumption."""
    X, _ = blobs()
    ids = [f"p{i}" for i in range(len(X))]
    meds = ClusterLoRAKeyFn(
        MEDSPartitioner(MEDSClusterer(min_cluster_size=PER), warmup_batches=0), mode="meds"
    )
    ctrl = ClusterLoRAKeyFn(
        MEDSPartitioner(MEDSClusterer(min_cluster_size=PER), warmup_batches=0),
        mode="random_matched", seed=3,
    )
    units = [f"0:{i}" for i in range(len(X))]
    meds.begin_batch(units, X, group_ids=ids)
    ctrl.begin_batch(units, X, group_ids=ids)
    assert ctrl.partition.size_multiset() == meds.partition.size_multiset()
    assert ctrl.partition.labels != meds.partition.labels
