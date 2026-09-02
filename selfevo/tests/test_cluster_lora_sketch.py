"""The sketch, whose LINEARITY is the correctness condition for the whole probe.

"A cluster's gradient is the sum of its members' sketches" is what makes four partitions
cost one GPU pass. If the sketch were not linear that sentence would be false and every
cosine in the analysis would describe something that is not a gradient -- with no symptom,
because the numbers would still be numbers in [-1, 1].

The second property tested here is that the projection is SHARED. Two groups sketched under
different hashes are near-orthogonal whatever their gradients did, so a bug that reseeded per
group would report "no conflict anywhere", which is a publishable-looking answer and the most
dangerous failure this file can have.
"""

from __future__ import annotations

import numpy as np
import pytest

from selfevo.cluster_lora.sketch import (
    SketchPlan,
    sketch_dim_resolution,
    sketch_vector,
)

rng = np.random.default_rng(0)


def blocks(seed, shapes=((4, 5), (3,), (2, 2))):
    """A few named arrays standing in for one model's LoRA gradient."""
    r = np.random.default_rng(seed)
    return [(f"layer{i}.lora_A", r.normal(size=s)) for i, s in enumerate(shapes)]


def test_the_sketch_is_linear():
    """sketch(a + b) == sketch(a) + sketch(b). The property the analysis rests on."""
    plan = SketchPlan(dim=64, seed=0)
    a, b = blocks(1), blocks(2)
    summed = [(n, x + y) for (n, x), (_m, y) in zip(a, b)]
    assert np.allclose(
        sketch_vector(summed, plan),
        sketch_vector(a, plan) + sketch_vector(b, plan),
        atol=1e-12,
    )


def test_summing_group_sketches_equals_sketching_the_summed_gradient():
    """The claim in the form the analysis actually uses it: a cluster of several groups."""
    plan = SketchPlan(dim=128, seed=3)
    groups = [blocks(s) for s in range(5)]
    total = [
        (n, sum(g[i][1] for g in groups)) for i, (n, _v) in enumerate(groups[0])
    ]
    assert np.allclose(
        sketch_vector(total, plan),
        sum(sketch_vector(g, plan) for g in groups),
        atol=1e-10,
    )


def test_the_same_plan_gives_the_same_projection_across_calls():
    plan = SketchPlan(dim=64, seed=0)
    g = blocks(1)
    assert np.array_equal(sketch_vector(g, plan), sketch_vector(g, plan))


def test_a_fresh_plan_per_group_would_destroy_the_measurement():
    """The failure this test exists to make impossible to ship unnoticed.

    Sketched under two different seeds, two copies of the SAME gradient are near-orthogonal
    -- so a probe that reseeded per group would report every partition as conflict-free.
    """
    g = blocks(1)
    same = float(
        np.dot(sketch_vector(g, SketchPlan(64, seed=0)), sketch_vector(g, SketchPlan(64, seed=0)))
    )
    crossed = float(
        np.dot(sketch_vector(g, SketchPlan(64, seed=0)), sketch_vector(g, SketchPlan(64, seed=1)))
    )
    assert same > 0
    assert abs(crossed) < 0.5 * same


def test_the_sketch_preserves_the_angle_it_claims_to():
    """An empirical check of the property the theorem promises, at a usable dimension."""
    plan = SketchPlan(dim=4096, seed=0)
    r = np.random.default_rng(5)
    a = r.normal(size=20000)
    b = 0.6 * a + 0.8 * r.normal(size=20000)
    exact = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    sa, sb = sketch_vector([("w", a)], plan), sketch_vector([("w", b)], plan)
    approx = float(np.dot(sa, sb) / (np.linalg.norm(sa) * np.linalg.norm(sb)))
    assert abs(exact - approx) < 5 * sketch_dim_resolution(4096)


def test_the_resolution_floor_is_reported_and_shrinks_with_dimension():
    """A cosine under the floor is 'not resolved', not 'small'.

    The published cross-task figure is ~1e-5; at any dimension this probe can afford, that
    is below the floor, and the analysis has to say so rather than confirm it.
    """
    assert sketch_dim_resolution(8192) == pytest.approx(3 / np.sqrt(8192))
    assert sketch_dim_resolution(65536) < sketch_dim_resolution(8192)
    assert 1e-5 < sketch_dim_resolution(65536)


def test_a_layout_change_between_groups_is_refused():
    """Same parameter name, different length: the sketches stop being comparable."""
    plan = SketchPlan(dim=16, seed=0)
    sketch_vector([("w", np.ones(4))], plan)
    with pytest.raises(ValueError, match="not comparable"):
        sketch_vector([("w", np.ones(5))], plan)


def test_a_non_finite_gradient_is_refused_not_sketched():
    """A NaN becomes a NaN cosine, which would be reported as a measurement."""
    plan = SketchPlan(dim=16, seed=0)
    with pytest.raises(ValueError, match="non-finite"):
        sketch_vector([("w", np.array([1.0, np.nan]))], plan)


def test_an_empty_block_is_refused():
    plan = SketchPlan(dim=16, seed=0)
    with pytest.raises(ValueError, match="empty"):
        sketch_vector([("w", np.zeros(0))], plan)


def test_sketching_nothing_is_refused():
    with pytest.raises(ValueError, match="not a gradient"):
        sketch_vector([], SketchPlan(dim=16, seed=0))


def test_a_zero_dimension_plan_is_refused():
    with pytest.raises(ValueError, match="positive"):
        SketchPlan(dim=0)


def test_the_torch_path_is_the_same_projection_as_the_numpy_one():
    """Identical by construction is exactly the claim that stops being true silently.

    The GPU dump sketches torch tensors; the analysis and every test here use numpy. If the
    two ever diverged, the dump's numbers and the tests' numbers would describe different
    projections and nothing would say so.
    """
    torch = pytest.importorskip("torch")
    from selfevo.cluster_lora.sketch import sketch_torch

    plan = SketchPlan(dim=256, seed=11)
    g = blocks(9)
    np_out = sketch_vector(g, plan)
    t_out = sketch_torch([(n, torch.tensor(v)) for n, v in g], plan)
    assert np.allclose(np_out, t_out, atol=1e-12)


def test_two_parameters_do_not_share_a_hash():
    """The projection is keyed on the parameter NAME, not only on the seed.

    With one shared hash for every block, putting a payload in the first parameter and
    putting it in the second would produce the SAME sketch -- different parts of the model
    would be indistinguishable and every cosine would mix them. Linearity is unaffected, so
    no other test here can see it.
    """
    plan = SketchPlan(dim=64, seed=0)
    x = np.random.default_rng(0).normal(size=10)
    z = np.zeros(10)
    first = sketch_vector([("a", x), ("b", z)], plan)
    second = sketch_vector([("a", z), ("b", x)], plan)
    assert not np.allclose(first, second), "the two parameters share a hash and alias"
