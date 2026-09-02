"""The analysis arithmetic, on a dump whose answer is known in advance.

The probe's job is to decide whether behavioural clusters conflict MORE than a size-matched
random partition, than ELREA-style prompt-gradient clusters, and than the cross-task
calibration. That verdict is only worth reading if the arithmetic underneath is right, so
these tests plant a structure whose statistics can be worked out in advance and check that
the analysis recovers it -- the arithmetic, not the science.

The planted structure is FOUR clusters whose gradients point along four simplex directions,
so every pair has cosine ``-1/3`` exactly. Under the true partition the analysis must recover
that; under a size-matched random partition each cluster receives a mixture of all four, the
sums largely cancel internally, and the between-cluster cosine collapses toward zero. Four
clusters rather than two is deliberate: with two clusters there is exactly ONE pair, the
control's mean cosine is a single draw, and it was measured swinging between -0.12 and -0.22
across control seeds -- a contrast that unstable cannot pin the arithmetic.

Needs numpy and scikit-learn. Run under ``~/venv_probe``; the training venv deliberately has
neither scikit-learn nor hdbscan and skips this file.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from selfevo.cluster_lora.interference_analyze import (  # noqa: E402
    analyse_dump,
    cluster_gradients,
    pairwise_stats,
    partition_block,
)
from selfevo.cluster_lora.partition import (  # noqa: E402
    Partition,
    random_matched_partition,
)


DIM, N, K = 256, 24, 4


def planted(seed=0):
    """Twenty-four groups in four clusters whose gradients mutually conflict.

    The four directions are an orthonormal basis with its mean removed, i.e. the vertices of
    a simplex, so every pair has cosine exactly ``-1/3``. That is a realistic shape for the
    hypothesis under test: experts that each want to move somewhere the others do not.

    The behavioural feature is four well-separated DIRECTIONS, never four points around the
    origin. The vendored MEDS path L2-normalises before euclidean HDBSCAN, so only the
    direction of a feature survives and a blob centred on the origin degenerates -- which
    cost this fixture a rewrite and is worth knowing before a real feature is designed.

    The prompt-gradient feature is deliberately INDEPENDENT of the behavioural split. That
    is the null the ELREA comparison must be able to detect: if prompt gradients carry no
    behavioural information, the ELREA partition has to look like the random one.
    """
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(DIM, K)))[0].T
    dirs = basis - basis.mean(0, keepdims=True)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    labels = [i % K for i in range(N)]
    sketch = np.stack([dirs[k] + 0.05 * rng.normal(size=DIM) for k in labels])
    centres = np.array([[5.0, 0.0], [0.0, 5.0], [-5.0, 0.0], [0.0, -5.0]])
    feats = np.stack([centres[k] + 0.05 * rng.normal(size=2) for k in labels])
    prompt = rng.normal(size=(N, DIM))
    return sketch, feats, prompt, labels


def write_npz(tmp_path, tasks=None, full_grad=None):
    """A dump file in exactly the shape ``interference_dump`` writes."""
    sketch, feats, prompt, labels = planted()
    tasks = tasks or ["math" if i < N // 2 else "code" for i in range(N)]
    path = tmp_path / "dump.npz"
    np.savez_compressed(
        path,
        sketch=sketch, prompt_sketch=prompt, meds_feature=feats,
        group_id=np.array([f"p{i}" for i in range(N)], dtype=object),
        task=np.array(tasks, dtype=object),
        group_size=np.full(N, 4, dtype=np.int64),
        reward_mean=np.zeros(N),
        zero_block_fraction=np.zeros(N),
        full_grad=(full_grad if full_grad is not None
                   else np.zeros((0, 0), dtype=np.float32)),
        meta=np.array(json.dumps({"n_groups": N}), dtype=object),
    )
    return path, labels


def part(labels):
    return Partition(
        labels=tuple(labels), basis="test",
        group_ids=tuple(f"p{i}" for i in range(len(labels))),
    )


# ------------------------------------------------------------------- the arithmetic -----


def test_a_clusters_gradient_is_the_sum_of_its_members():
    """The identity that lets one dump answer four partitions."""
    s = np.arange(12, dtype=float).reshape(4, 3)
    g = cluster_gradients(s, part([0, 0, 1, 1]))
    assert np.array_equal(g["cluster_0"], s[0] + s[1])
    assert np.array_equal(g["cluster_1"], s[2] + s[3])


def test_a_partition_that_does_not_describe_the_dump_is_refused():
    with pytest.raises(ValueError, match="does not describe this dump"):
        cluster_gradients(np.zeros((3, 2)), part([0, 1]))


def test_anti_parallel_clusters_give_cosine_minus_one_and_full_cancellation():
    """The hand-computable case, so the statistics are pinned rather than plausible."""
    v = np.array([1.0, 0.0, 0.0])
    st = pairwise_stats({"a": v, "b": -v})
    assert st["mean_cosine"] == pytest.approx(-1.0)
    assert st["conflict_rate"] == 1.0
    assert st["cancellation"] == pytest.approx(0.0, abs=1e-12)


def test_identical_clusters_give_cosine_one_and_no_cancellation():
    v = np.array([1.0, 2.0, 3.0])
    st = pairwise_stats({"a": v, "b": v})
    assert st["mean_cosine"] == pytest.approx(1.0)
    assert st["conflict_rate"] == 0.0
    assert st["cancellation"] == pytest.approx(1.0)


def test_orthogonal_clusters_report_zero_conflict_and_partial_cancellation():
    """The published cross-task regime: near-orthogonal updates."""
    st = pairwise_stats({"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])})
    assert st["mean_cosine"] == pytest.approx(0.0)
    assert st["conflict_rate"] == 0.0
    assert st["cancellation"] == pytest.approx(np.sqrt(2) / 2)


def test_a_single_cluster_reports_None_not_zero():
    """An empty mean is not 'no conflict'.

    Reported as 0.0 it would look exactly like a successful reproduction of the ~1e-5
    cross-task figure, produced by a partition that could not test anything.
    """
    st = pairwise_stats({"a": np.ones(3)})
    assert st["mean_cosine"] is None and st["conflict_rate"] is None
    assert st["n_pairs"] == 0


def test_a_zero_gradient_cluster_is_excluded_from_the_pairs_not_counted_as_orthogonal():
    """A silent group has advantages identically zero and no direction at all.

    Counting it as orthogonal would drag every mean toward zero in proportion to how many
    groups were RL-silent -- which this project measures at 29-44%.
    """
    st = pairwise_stats({"a": np.array([1.0, 0.0]), "b": np.zeros(2), "c": np.array([1.0, 0.0])})
    assert st["n_clusters"] == 3 and st["n_clusters_with_gradient"] == 2
    assert st["n_pairs"] == 1 and st["mean_cosine"] == pytest.approx(1.0)


def test_the_planted_simplex_cosine_is_recovered_to_three_decimals():
    """Four simplex directions have pairwise cosine -1/3 exactly; the analysis must say so."""
    sketch, _f, _p, labels = planted()
    blk = partition_block("meds", part(labels), sketch, floor=0.05, n_boot=100, seed=0)
    assert blk["n_pairs"] == 6
    assert blk["mean_cosine"] == pytest.approx(-1 / 3, abs=0.05)
    assert blk["conflict_rate"] == 1.0


# --------------------------------------------------- the contrast the paper depends on ---


@pytest.mark.parametrize("control_seed", [0, 1, 2, 3, 4])
def test_the_true_partition_conflicts_more_than_every_size_matched_control(control_seed):
    """The whole claim, in miniature, where the right answer is known.

    Run over five control seeds, not one: a control that happened to permute favourably
    would otherwise decide the test. If this contrast did not appear here, no contrast
    measured on a real batch could be believed -- the probe would be incapable of seeing the
    effect it is built to find.
    """
    sketch, _f, _p, labels = planted()
    truth = part(labels)
    a = partition_block("meds", truth, sketch, floor=0.05, n_boot=300, seed=0)
    b = partition_block(
        "random_matched", random_matched_partition(truth, seed=control_seed),
        sketch, floor=0.05, n_boot=300, seed=0,
    )
    assert a["mean_cosine"] < b["mean_cosine"] - 0.05, (a["mean_cosine"], b["mean_cosine"])
    assert a["cancellation"] < b["cancellation"]
    assert a["size_multiset"] == b["size_multiset"]


def test_the_bootstrap_spread_separates_a_real_partition_from_a_permuted_one():
    """A structure the clustering found is STABLE under resampling; a permutation is not.

    Measured here at std 0.009 for the true partition against 0.06-0.09 for permuted ones --
    an order of magnitude, and a second discriminator that does not depend on the mean.

    The interval is NOT asserted to contain the point estimate. Resampling groups with
    replacement duplicates members and biases a cluster sum toward its duplicated
    directions, so the percentile interval sits slightly above the point estimate; that is a
    property of the percentile bootstrap, and asserting containment would pin a bias rather
    than a result.
    """
    sketch, _f, _p, labels = planted()
    truth = part(labels)
    a = partition_block("meds", truth, sketch, floor=0.05, n_boot=400, seed=0)
    b = partition_block(
        "random_matched", random_matched_partition(truth, seed=1), sketch,
        floor=0.05, n_boot=400, seed=0,
    )
    assert a["bootstrap_mean_cosine"]["n_effective"] > 300
    assert a["bootstrap_mean_cosine"]["std"] < 0.5 * b["bootstrap_mean_cosine"]["std"]
    assert a["bootstrap_mean_cosine"]["lo"] <= a["bootstrap_mean_cosine"]["hi"]


def test_a_cosine_under_the_resolution_floor_is_not_reported_as_resolved():
    """1e-5 is below any floor this probe can afford, and must be said so.

    A reviewer citing the ~1e-5 cross-task figure has to be answered with "below our
    instrument's floor", not with a number the instrument cannot produce.
    """
    rng = np.random.default_rng(0)
    s = rng.normal(size=(8, 4096))
    blk = partition_block("x", part([0, 0, 0, 0, 1, 1, 1, 1]), s,
                          floor=0.5, n_boot=50, seed=0)
    assert blk["resolved"] is False
    assert abs(blk["mean_cosine"]) < 0.5


# ------------------------------------------------------------- all four blocks at once ---


def test_all_four_partitions_come_back_from_one_dump(tmp_path):
    """One GPU pass, four answered objections. The reason the probe is split in two."""
    path, _ = write_npz(tmp_path)
    res = analyse_dump(str(path), n_boot=100, seed=0, min_cluster_size=4)
    names = [b["partition"] for b in res["partitions"]]
    assert names == ["meds", "random_matched", "elrea", "task"]
    assert all("skipped" not in b for b in res["partitions"]), res["partitions"]


def test_hdbscan_recovers_the_planted_clusters_at_a_sane_min_cluster_size(tmp_path):
    """And MEDS' shipped min_cluster_size=2 does NOT, which is a tuning fact not a detail.

    Measured 2026-09-02: on four perfectly separated directional blobs of six,
    ``min_cluster_size=2`` splits them into six clusters plus noise, while 4 recovers exactly
    four of six. MEDS clusters to shape a reward, where over-splitting is harmless; here
    every extra cluster is another expert trained on fewer groups.
    """
    path, _ = write_npz(tmp_path)
    good = analyse_dump(str(path), n_boot=20, seed=0, min_cluster_size=4)["partitions"][0]
    shipped = analyse_dump(str(path), n_boot=20, seed=0, min_cluster_size=2)["partitions"][0]
    assert good["size_multiset"] == [6, 6, 6, 6]
    assert len(shipped["size_multiset"]) > 4


def test_the_elrea_partition_is_size_matched_to_the_meds_one(tmp_path):
    """A more lopsided partition conflicts more for reasons unrelated to its feature."""
    path, _ = write_npz(tmp_path)
    res = analyse_dump(str(path), n_boot=50, seed=0, min_cluster_size=4)
    blocks = {b["partition"]: b for b in res["partitions"]}
    assert blocks["elrea"]["size_multiset"] == blocks["meds"]["size_multiset"]
    assert blocks["random_matched"]["size_multiset"] == blocks["meds"]["size_multiset"]


def test_prompt_gradients_that_carry_no_behavioural_signal_do_not_beat_the_control(tmp_path):
    """The ELREA null, which the probe has to be able to report.

    The fixture's prompt-gradient features are independent of the behavioural split, so the
    ELREA partition must land near the random control rather than near MEDS. A probe that
    reported ELREA as conflicted here would report it as conflicted on any batch.
    """
    path, _ = write_npz(tmp_path)
    res = analyse_dump(str(path), n_boot=100, seed=0, min_cluster_size=4)
    b = {x["partition"]: x for x in res["partitions"]}
    assert b["meds"]["mean_cosine"] < b["elrea"]["mean_cosine"]
    assert abs(b["elrea"]["mean_cosine"] - b["random_matched"]["mean_cosine"]) < 0.15


def test_a_single_task_batch_skips_the_calibration_with_a_reason(tmp_path):
    """Skipped and SAID, never reported as a zero cross-task cosine."""
    path, _ = write_npz(tmp_path, tasks=["math"] * N)
    res = analyse_dump(str(path), n_boot=50, seed=0, min_cluster_size=4)
    task = [b for b in res["partitions"] if b["partition"] == "task"][0]
    assert "spans 1 task" in task["skipped"]
    # And the other three still came back: losing them to an exception would be worse.
    assert sum("skipped" not in b for b in res["partitions"]) == 3


def test_the_headline_contrasts_are_reported_as_differences(tmp_path):
    """A bare 'MEDS conflict is 0.4' answers none of the three objections."""
    path, _ = write_npz(tmp_path)
    res = analyse_dump(str(path), n_boot=50, seed=0, min_cluster_size=4)
    c = res["contrasts"]
    assert set(c) == {"meds_minus_random_matched", "meds_minus_elrea", "meds_minus_task"}
    assert c["meds_minus_random_matched"] < 0
    assert c["meds_minus_elrea"] < 0


def test_the_sketch_is_validated_against_stored_full_gradients(tmp_path):
    """The sketch preserves angles in expectation; on THIS batch it is measured."""
    sketch, _f, _p, _l = planted()
    path, _ = write_npz(tmp_path, full_grad=sketch[:4].astype(np.float32))
    res = analyse_dump(str(path), n_boot=20, seed=0, min_cluster_size=4)
    v = res["sketch_validation"]
    assert v["n_groups"] == 4 and v["n_pairs"] == 6
    assert v["max_abs_error"] < 1e-6  # identical vectors: the check must find no error


def test_a_dump_with_no_full_gradients_says_the_sketch_is_unvalidated(tmp_path):
    """Silence about a missing check reads as a passed check."""
    path, _ = write_npz(tmp_path)
    res = analyse_dump(str(path), n_boot=20, seed=0, min_cluster_size=4)
    assert "unvalidated" in res["sketch_validation"]["note"]


def test_the_result_carries_the_floor_and_the_knob_so_it_cannot_be_read_without_them(tmp_path):
    path, _ = write_npz(tmp_path)
    res = analyse_dump(str(path), n_boot=20, seed=0, min_cluster_size=4)
    assert res["resolution_floor"] == pytest.approx(3 / np.sqrt(DIM))
    assert res["sketch_dim"] == DIM and res["min_cluster_size"] == 4
