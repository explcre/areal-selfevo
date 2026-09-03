"""The behavioural feature, and the key_fn that carries it to the ClusterRouter seam.

Two things are pinned here that a comment cannot pin.

**The hook path equals the hidden-states path.** MEDS reads ``output_hidden_states``, which
retains every layer for every token -- about 6.3 GB per 10240-token microbatch at 32B, the
OOM this project's notes warn about. The extractor's default instead captures one position
per layer through forward hooks. That is only correct if the hooks reproduce
``hidden_states[1:]`` exactly, which depends on transformers building that tuple as
``(embeddings, layer_0_out, ..., layer_{N-2}_out, norm(layer_{N-1}_out))``. The equality is
therefore a test: a transformers upgrade that changes the convention fails here rather than
quietly shifting the feature by one layer.

**A unit with no feature raises.** ``ClusterLoRAKeyFn`` cannot fall back to a default
cluster, because a default cluster is a real adapter that would receive that group's
gradient. The fallback would be indistinguishable from a deliberate assignment.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from selfevo.cluster_lora.features import (  # noqa: E402
    BehaviourFeatureUnavailable,
    ClusterLoRAKeyFn,
    LayerLogitExtractor,
    answer_token_index,
    meds_feature,
)
from selfevo.cluster_lora.partition import MEDSPartitioner, SHARED_CLUSTER  # noqa: E402
from selfevo.routing.base import Granularity, RoutingContext, TrainingMode  # noqa: E402
from selfevo.routing.cluster import ClusterRouter  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    """A four-layer causal LM, small enough that the whole file runs in a second."""
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = AutoConfig.for_model(
        "qwen2", hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
        max_position_embeddings=64, tie_word_embeddings=False,
    )
    return AutoModelForCausalLM.from_config(cfg).eval()


# ---------------------------------------------------------------- the answer token ------


def test_a_boxed_answer_in_the_PROMPT_is_not_mistaken_for_the_model_s():
    """On a few-shot math prompt the exemplars contain \\boxed{ every time.

    Reading the prompt's box would give every rollout of a group the same feature, and the
    whole group would collapse to one point in the clustering.
    """
    boxed = [7, 8]
    seq = [7, 8, 1, 2, 3, 7, 8, 9, 9]
    #      ^ prompt box                ^ response box at index 6
    assert answer_token_index(seq, boxed_ids=boxed, response_start=4) == 6


def test_the_position_is_the_last_token_of_the_box_so_the_next_token_is_the_answer():
    """MEDS' own convention, kept exactly: p+1 is the first character of the answer."""
    seq = [0, 0, 7, 8, 42, 5]
    p = answer_token_index(seq, boxed_ids=[7, 8], response_start=2)
    assert p == 3 and seq[p + 1] == 42


def test_the_last_box_wins_when_a_response_contains_several():
    seq = [0, 7, 8, 1, 9, 7, 8, 2, 3]
    assert answer_token_index(seq, boxed_ids=[7, 8], response_start=1) == 6


def test_a_missing_box_refuses_rather_than_reading_position_zero():
    """Position 0 is the prompt's first token: identical for every rollout of a group."""
    with pytest.raises(BehaviourFeatureUnavailable, match="boxed"):
        answer_token_index([1, 2, 3, 4], boxed_ids=[7, 8])


def test_the_last_position_strategy_is_available_for_tasks_with_no_box():
    seq = [1, 2, 3, 4, 5]
    p = answer_token_index(seq, strategy="last")
    assert p == 3 and p + 1 < len(seq)


def test_boxed_without_boxed_ids_is_a_programming_error_not_a_fallback():
    with pytest.raises(ValueError, match="needs boxed_ids"):
        answer_token_index([1, 2, 3], strategy="boxed")


# ------------------------------------------------------------------ MEDS reduction ------


def test_the_default_keeps_the_LATTER_HALF_of_the_layers():
    """MEDS ships last_n=14 on a 28-layer model; expressed as a rule, not a constant.

    A constant would mean something different on a model of another depth, and every
    comparison across model sizes would be between two different features.
    """
    trace = list(range(28))
    assert meds_feature(trace).tolist() == list(range(14, 28))


def test_an_explicit_last_n_overrides_the_rule():
    assert meds_feature(list(range(10)), last_n_layers=3).tolist() == [7, 8, 9]


def test_layer_diff_is_available_and_off_by_default():
    """MEDS' own run script ships use_layer_diff=False."""
    trace = [0.0, 1.0, 3.0, 6.0]
    assert meds_feature(trace, use_layer_diff=True, last_n_layers=3).tolist() == [1.0, 2.0, 3.0]
    assert meds_feature(trace, last_n_layers=2).tolist() == [3.0, 6.0]


def test_a_non_finite_trace_is_refused():
    """It would cluster as a point rather than fail."""
    with pytest.raises(BehaviourFeatureUnavailable, match="non-finite"):
        meds_feature([1.0, float("nan"), 3.0])


def test_a_trace_too_short_to_reduce_is_refused():
    with pytest.raises(BehaviourFeatureUnavailable, match="cannot be reduced"):
        meds_feature([1.0])


# -------------------------------------------------------------------- the extractor -----


def test_the_hook_path_reproduces_the_hidden_states_path_exactly(tiny):
    """The memory-safe default must be the SAME feature as the MEDS-faithful reference.

    This is what licenses running the cheap path at 32B. It also pins the transformers
    hidden-state convention: hooks on layers 0..N-2 plus the final norm, never every layer.
    """
    ids = torch.randint(0, 128, (1, 12))
    ref = LayerLogitExtractor(mode="hidden_states").trace(tiny, ids, 6, int(ids[0, 7]))
    got = LayerLogitExtractor(mode="hooks").trace(tiny, ids, 6, int(ids[0, 7]))
    assert ref.shape == (4,)
    assert np.allclose(ref, got, atol=1e-5), (ref, got)


def test_the_dot_product_shortcut_is_the_same_number_as_the_full_unembedding(tiny):
    """1/vocab of the arithmetic for the same value; asserted, not assumed."""
    ids = torch.randint(0, 128, (1, 10))
    a = LayerLogitExtractor(mode="hooks", dot_unembed=True).trace(tiny, ids, 5, int(ids[0, 6]))
    b = LayerLogitExtractor(mode="hooks", dot_unembed=False).trace(tiny, ids, 5, int(ids[0, 6]))
    assert np.allclose(a, b, atol=1e-5)


def test_truncating_at_the_answer_token_is_exact_because_attention_is_causal(tiny):
    """Where most of the saving on a long rollout comes from -- and it is not an approximation."""
    ids = torch.randint(0, 128, (1, 30))
    keep = LayerLogitExtractor(truncate=True).trace(tiny, ids, 9, int(ids[0, 10]))
    full = LayerLogitExtractor(truncate=False).trace(tiny, ids, 9, int(ids[0, 10]))
    assert np.allclose(keep, full, atol=1e-5)


def test_two_different_sequences_give_different_traces(tiny):
    """Otherwise the feature is a constant and every clustering of it is an artefact."""
    a = LayerLogitExtractor().trace(tiny, torch.arange(10).unsqueeze(0), 5, 6)
    b = LayerLogitExtractor().trace(tiny, torch.arange(20, 30).unsqueeze(0), 5, 26)
    assert not np.allclose(a, b)


def test_a_position_outside_the_sequence_is_refused(tiny):
    with pytest.raises(BehaviourFeatureUnavailable, match="outside a sequence"):
        LayerLogitExtractor().trace(tiny, torch.arange(5).unsqueeze(0), 9, 1)


def test_an_unknown_extractor_mode_is_refused():
    with pytest.raises(ValueError, match="unknown mode"):
        LayerLogitExtractor(mode="magic")


def test_a_model_with_no_decoder_stack_is_refused(tiny):
    with pytest.raises(BehaviourFeatureUnavailable, match="decoder stack"):
        LayerLogitExtractor().trace(torch.nn.Linear(2, 2), torch.arange(4).unsqueeze(0), 1, 1)


# -------------------------------------------------- the key_fn at the ClusterRouter seam --


class StubClusterer:
    """A MEDSClusterer-shaped stub, so the LIFECYCLE is tested where hdbscan is absent.

    The training venv deliberately carries neither hdbscan nor scikit-learn, and the
    lifecycle -- warm up, label by kNN against history, only then add and refit -- is ours,
    not the vendored library's. Stubbing the library tests the part we wrote; the vendored
    part is exercised for real in ``test_cluster_lora_clustering.py`` under ~/venv_probe.
    """

    knn_k = 3
    knn_metric = "cosine"
    min_cluster_size = 2
    metric = "euclidean"
    backend = "stub"

    def __init__(self):
        self.fitted = False
        self.added = []
        self.fits = 0

    @property
    def n_clusters(self):
        """Two, always, so a partition is formed once fitting has happened."""
        return 2

    def add(self, v):
        """Buffer a vector, as the real clusterer does."""
        self.added.append(np.asarray(v, dtype=float))

    def fit(self):
        """Mark fitted, as the real clusterer does."""
        self.fitted = True
        self.fits += 1
        return self.n_clusters

    def assign(self, v):
        """Split on the first coordinate's sign, which is enough to be a real partition."""
        return 0 if float(np.asarray(v).ravel()[0]) >= 0 else 1


def keyfn(mode="meds", warmup=1, seed=0):
    """A key_fn over the stub, so the seam is exercised without the clustering deps."""
    return ClusterLoRAKeyFn(
        MEDSPartitioner(StubClusterer(), warmup_batches=warmup), mode=mode, seed=seed
    )


def ctxs(n, step=0):
    """Contexts with the unit_id shape the real actor emits: f"{step}:{i}"."""
    return [
        RoutingContext(
            solve_rate=0.5, group_size=4, granularity=Granularity.CLUSTER,
            unit_id=f"{step}:{i}",
        )
        for i in range(n)
    ]


def test_the_key_fn_partitions_a_batch_through_the_real_ClusterRouter():
    """The seam, driven end to end: ClusterRouter.route_batch must see OUR clusters.

    Asserted through ``route_batch`` rather than by calling the key_fn, because a key_fn
    that works in isolation and is never consulted is exactly the dead-code state
    ``ClusterRouter`` was in before it had a caller.
    """
    kf = keyfn(warmup=0)
    feats = np.array([[1.0], [-1.0], [1.0], [-1.0]])
    kf.begin_batch([c.unit_id for c in ctxs(4)], feats)
    router = ClusterRouter(
        key_fn=kf, policy={"cluster_0": TrainingMode.RL, "cluster_1": TrainingMode.RL}
    )
    a = router.route_batch(ctxs(4))
    assert a.cluster_of == ("cluster_0", "cluster_1", "cluster_0", "cluster_1")
    assert a.sizes == {"cluster_0": 2, "cluster_1": 2}
    assert a.basis == "caller-supplied partition"


def test_a_unit_with_no_feature_raises_instead_of_defaulting_to_a_cluster():
    """A default cluster is a real adapter that would receive this group's gradient."""
    kf = keyfn(warmup=0)
    kf.begin_batch(["0:0"], np.array([[1.0]]))
    with pytest.raises(BehaviourFeatureUnavailable, match="no behavioural feature"):
        kf(ctxs(2)[1])


def test_routing_before_any_batch_is_armed_raises():
    with pytest.raises(BehaviourFeatureUnavailable, match="no batch is armed"):
        keyfn()(ctxs(1)[0])


def test_mismatched_feature_and_unit_counts_are_refused():
    """A mismatch would assign one group's adapter from another group's behaviour."""
    with pytest.raises(ValueError, match="would assign one group"):
        keyfn().begin_batch(["0:0", "0:1"], np.array([[1.0]]))


def test_the_warmup_batch_is_all_shared_and_says_so():
    """HDBSCAN on a handful of points is almost all noise.

    A first batch silently clustered would put the experts' first update on labels that mean
    nothing, and every later kNN assignment is anchored to that fit.
    """
    kf = keyfn(warmup=1)
    p = kf.begin_batch(["0:0", "0:1"], np.array([[1.0], [-1.0]]))
    assert p.keys == (SHARED_CLUSTER, SHARED_CLUSTER)
    assert "WARMUP" in p.basis


def test_labels_are_kNN_stabilised_across_batches_and_churn_is_measured():
    """The mechanism that stops a group from changing expert every step.

    A group that jumps adapters every batch gives each expert one noisy update and coherent
    training to none -- the method failing through the very thing meant to prevent it.
    """
    kf = keyfn(warmup=0)
    ids = ["p0", "p1", "p2", "p3"]
    kf.begin_batch([f"0:{i}" for i in range(4)], np.array([[1.0], [-1.0], [1.0], [-1.0]]),
                   group_ids=ids)
    kf.begin_batch([f"1:{i}" for i in range(4)], np.array([[1.0], [-1.0], [1.0], [-1.0]]),
                   group_ids=ids)
    r = kf.report()
    assert r.churn == 0.0 and r.n_churn_overlap == 4
    # And a genuine move IS seen, or the zero above proves nothing.
    kf.begin_batch([f"2:{i}" for i in range(4)], np.array([[-1.0], [-1.0], [1.0], [-1.0]]),
                   group_ids=ids)
    assert kf.report().churn == 0.25


def test_the_control_mode_goes_through_the_same_key_fn():
    """The control must differ from the method in the LABELS only, never in the plumbing."""
    ids = [f"0:{i}" for i in range(6)]
    feats = np.array([[1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0]])
    meds = keyfn("meds", warmup=0)
    meds.begin_batch(ids, feats)
    ctrl = keyfn("random_matched", warmup=0, seed=5)
    ctrl.begin_batch(ids, feats)
    assert sorted(ctrl.partition.sizes.values()) == sorted(meds.partition.sizes.values())
    assert ctrl.partition.labels != meds.partition.labels


def test_the_control_mode_is_as_stable_across_batches_as_the_method():
    """The control carries its assignment forward, and this is the seam that carries it.

    ``begin_batch`` passes ``seed + self.batches``, so the control's seed MOVES every batch
    by construction; before ``MatchedControlMemory`` existed that alone re-permuted every
    group, and the control differed from the method in expert stability as well as in
    feature-blindness -- the one axis findings 5.1 makes the method rest on.

    ``partition_from_config`` hands the control the PARTITIONER's own memory, so a control
    forgets exactly when the method it controls for does. Driven through the key fn rather
    than through ``random_matched_partition`` because the per-batch seed advance lives here.
    """
    ids = [f"p{i}" for i in range(6)]
    feats = np.array([[1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0]])
    ctrl = keyfn("random_matched", warmup=0, seed=5)
    first = None
    for step in range(4):
        ctrl.begin_batch([f"{step}:{i}" for i in range(6)], feats, group_ids=ids)
        if first is None:
            first = ctrl.partition.labels
            continue
        assert ctrl.report().churn == 0.0, f"the control re-permuted at batch {step}"
        assert ctrl.partition.labels == first
    assert len(ctrl.partitioner.control_memory) == len(ids)


def test_the_control_arm_carries_on_the_partitioners_memory_not_the_global_one():
    """A fresh partitioner is a fresh ARM, and two arms in one process must not share.

    ``partition_from_config`` hands the partitioner's own memory rather than letting the
    module-level default carry. Asserted by watching the DEFAULT memory stay empty while the
    arm runs: with the argument dropped the arm still looks stable across its own batches --
    the global carries just as well -- and the only visible difference is that a second arm
    in the same process would inherit the first one's draw. That is a control decided by
    which arm ran first, and it is not observable from inside one arm.
    """
    from selfevo.cluster_lora.partition import DEFAULT_CONTROL_MEMORY

    ids = [f"p{i}" for i in range(6)]
    feats = np.array([[1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0]])
    DEFAULT_CONTROL_MEMORY.reset()
    first = keyfn("random_matched", warmup=0, seed=5)
    first.begin_batch([f"0:{i}" for i in range(6)], feats, group_ids=ids)
    assert len(first.partitioner.control_memory) == len(ids)
    assert len(DEFAULT_CONTROL_MEMORY) == 0, (
        "the control arm carried its draw on the process-global memory, so a second arm in "
        "this process would inherit this arm's control assignment"
    )
    # Over a sweep rather than one further seed: six groups over two clusters admit only
    # twenty assignments, so a single second seed can collide with the first by chance and a
    # test that happened to draw one would pass whatever the arms shared.
    drawn = set()
    for seed in range(6, 16):
        other = keyfn("random_matched", warmup=0, seed=seed)
        other.begin_batch([f"0:{i}" for i in range(6)], feats, group_ids=ids)
        drawn.add(other.partition.labels)
    assert len(drawn) > 1, (
        f"ten arms under ten seeds produced one control ({drawn}); they carried each "
        "other's draw, so which control an arm gets is decided by which arm ran first"
    )


def test_the_none_mode_needs_no_features_and_puts_everything_on_shared():
    kf = ClusterLoRAKeyFn(mode="none")
    kf.begin_batch(["0:0", "0:1"], np.zeros((2, 1)))
    assert kf.partition.keys == (SHARED_CLUSTER, SHARED_CLUSTER)


def test_an_unknown_mode_is_refused_at_construction():
    with pytest.raises(ValueError, match="unknown mode"):
        ClusterLoRAKeyFn(mode="kmeans")


def test_reporting_before_a_batch_raises():
    with pytest.raises(BehaviourFeatureUnavailable, match="no batch"):
        keyfn().report()
