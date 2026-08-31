"""Tests for the signal-routing criterion and routers.

Written to constrain behaviour, not just to execute it: several tests assert the exact
asymmetry the design turns on (the two silent sides need opposite responses), which a
router keyed on informativeness alone would fail.
"""

from __future__ import annotations

import math

import pytest

from selfevo.routing.base import (
    Granularity,
    RoutingContext,
    RoutingDecision,
    Router,
    TrainingMode,
    known_modes,
    register_mode,
)
from selfevo.routing.criteria import (
    SilenceSide,
    expected_nonsilent_groups,
    min_group_size,
    rl_informativeness,
    silence_side,
    silent_group_probability,
)
from selfevo.routing.routers import (
    InvertedRouter,
    RandomRouter,
    SolveRateRouter,
    StaticRouter,
)


# --------------------------------------------------------------------------- criteria


def test_unanimous_groups_are_certain_at_the_extremes():
    assert silent_group_probability(0.0, 4) == 1.0
    assert silent_group_probability(1.0, 4) == 1.0
    assert rl_informativeness(0.0, 4) == 0.0
    assert rl_informativeness(1.0, 4) == 0.0


def test_informativeness_peaks_at_one_half():
    grid = [i / 100 for i in range(101)]
    best = max(grid, key=lambda p: rl_informativeness(p, 4))
    assert best == pytest.approx(0.5)


def test_informativeness_is_symmetric_about_one_half():
    for p in (0.1, 0.25, 0.42):
        assert rl_informativeness(p, 5) == pytest.approx(rl_informativeness(1 - p, 5))


def test_known_informativeness_values():
    # G=4: 1 - 2*0.5^4 = 0.875
    assert rl_informativeness(0.5, 4) == pytest.approx(0.875)
    # G=4, p=0.99: 1 - 0.99^4 - 0.01^4
    assert rl_informativeness(0.99, 4) == pytest.approx(0.03940399, abs=1e-8)


def test_larger_groups_are_more_informative_away_from_the_extremes():
    for p in (0.2, 0.5, 0.8):
        vals = [rl_informativeness(p, g) for g in (2, 4, 8, 16)]
        assert vals == sorted(vals), f"not monotone in G at p={p}: {vals}"


def test_group_size_one_is_always_silent():
    # A single sample has A = r - rbar = 0 identically; no group size 1 can inform.
    for p in (0.0, 0.3, 0.5, 0.9, 1.0):
        assert rl_informativeness(p, 1) == pytest.approx(0.0)


def test_min_group_size_matches_the_published_numbers():
    # p=0.76 measured in step0c
    assert min_group_size(0.76, 0.10) == pytest.approx(6.85, abs=0.02)
    assert min_group_size(0.76, 0.05) == pytest.approx(13.70, abs=0.05)
    # configured n_samples=4 is below both
    assert min_group_size(0.76, 0.10) > 4


def test_min_group_size_is_infinite_where_there_is_no_variance():
    assert math.isinf(min_group_size(0.0, 0.1))
    assert math.isinf(min_group_size(1.0, 0.1))


def test_expected_nonsilent_groups_shrinks_the_effective_batch():
    # 256 prompts at p=0.95, G=4 yields far fewer usable groups than the nominal batch.
    got = expected_nonsilent_groups(0.95, 4, 256)
    assert got == pytest.approx(256 * rl_informativeness(0.95, 4))
    assert got < 60


@pytest.mark.parametrize(
    "fn,args",
    [
        (silent_group_probability, (-0.1, 4)),
        (silent_group_probability, (1.1, 4)),
        (silent_group_probability, (0.5, 0)),
        (rl_informativeness, (0.5, -1)),
        (expected_nonsilent_groups, (0.5, 4, -1)),
    ],
)
def test_invalid_arguments_raise(fn, args):
    with pytest.raises(ValueError):
        fn(*args)


def test_min_group_size_rejects_bad_eps():
    with pytest.raises(ValueError):
        min_group_size(0.5, 0.0)


# ------------------------------------------------------------------- silence side


def test_silence_side_distinguishes_the_two_silent_regimes():
    # This is the property a router keyed on informativeness alone cannot express.
    assert silence_side(0.001, 4) is SilenceSide.UNSOLVED
    assert silence_side(0.999, 4) is SilenceSide.SOLVED
    assert silence_side(0.5, 4) is SilenceSide.INFORMATIVE


def test_silence_side_threshold_is_respected():
    p, g = 0.9, 4
    info = rl_informativeness(p, g)  # ~0.344
    assert silence_side(p, g, threshold=info - 0.01) is SilenceSide.INFORMATIVE
    assert silence_side(p, g, threshold=info + 0.01) is SilenceSide.SOLVED


def test_silence_side_rejects_bad_threshold():
    with pytest.raises(ValueError):
        silence_side(0.5, 4, threshold=1.5)


# ------------------------------------------------------------------------- base types


def test_routing_decision_rejects_empty_and_unknown_and_negative():
    with pytest.raises(ValueError):
        RoutingDecision({})
    with pytest.raises(ValueError):
        RoutingDecision({"no_such_mode": 1.0})
    with pytest.raises(ValueError):
        RoutingDecision({TrainingMode.RL: -1.0})
    with pytest.raises(ValueError):
        RoutingDecision({TrainingMode.RL: 0.0})


def test_routing_decision_normalises_and_argmax_is_deterministic():
    d = RoutingDecision({TrainingMode.RL: 3.0, TrainingMode.SFT: 1.0})
    assert d.normalised() == pytest.approx({TrainingMode.RL: 0.75, TrainingMode.SFT: 0.25})
    assert d.argmax() == TrainingMode.RL
    # ties resolve by name, deterministically, in both construction orders
    a = RoutingDecision({TrainingMode.RL: 1.0, TrainingMode.SFT: 1.0})
    b = RoutingDecision({TrainingMode.SFT: 1.0, TrainingMode.RL: 1.0})
    assert a.argmax() == b.argmax()


def test_routing_context_validates():
    with pytest.raises(ValueError):
        RoutingContext(solve_rate=1.5, group_size=4)
    with pytest.raises(ValueError):
        RoutingContext(solve_rate=0.5, group_size=0)


def test_register_mode_rejects_inconsistent_reregistration():
    register_mode("probe_mode", needs_teacher=True)
    register_mode("probe_mode", needs_teacher=True)  # idempotent
    with pytest.raises(ValueError):
        register_mode("probe_mode", needs_teacher=False)
    with pytest.raises(ValueError):
        register_mode("", needs_teacher=False)


def test_hard_distillation_is_not_a_separate_mode():
    # It is SFT with a teacher-sourced target; a separate mode would imply a gradient
    # difference that does not exist.
    assert "hard_distill" not in known_modes()


# ---------------------------------------------------------------------------- routers


def _ctx(p: float, *, teacher: bool = True, g: int = 4) -> RoutingContext:
    return RoutingContext(
        solve_rate=p, group_size=g, granularity=Granularity.SAMPLE, has_teacher=teacher
    )


def test_all_routers_satisfy_the_protocol():
    for r in (StaticRouter(), SolveRateRouter(), RandomRouter(), InvertedRouter()):
        assert isinstance(r, Router)


def test_static_router_ignores_context():
    r = StaticRouter({TrainingMode.RL: 1.0})
    assert r.route(_ctx(0.0)).argmax() == TrainingMode.RL
    assert r.route(_ctx(1.0)).argmax() == TrainingMode.RL


def test_static_router_validates_eagerly():
    with pytest.raises(ValueError):
        StaticRouter({"nope": 1.0})


def test_solve_rate_router_uses_rl_only_where_the_group_disagrees():
    r = SolveRateRouter()
    assert r.route(_ctx(0.5)).argmax() == TrainingMode.RL


def test_solve_rate_router_asymmetry_is_the_whole_point():
    """Unsolved -> teacher; solved -> skip. Never the reverse."""
    r = SolveRateRouter()
    assert r.route(_ctx(0.0)).argmax() == TrainingMode.SFT
    assert r.route(_ctx(1.0)).argmax() == TrainingMode.SKIP
    # and specifically: a solved unit must NOT be sent to SFT, which would sharpen an
    # already-correct policy and burn entropy
    assert r.route(_ctx(1.0)).argmax() != TrainingMode.SFT


def test_solve_rate_router_never_requires_an_absent_teacher():
    r = SolveRateRouter()
    d = r.route(_ctx(0.0, teacher=False))
    assert d.argmax() == TrainingMode.SKIP
    assert not known_modes()[d.argmax()]


def test_solve_rate_router_rejects_a_teacherless_teacher_mode():
    with pytest.raises(ValueError):
        SolveRateRouter(teacher_mode=TrainingMode.RL)
    with pytest.raises(ValueError):
        SolveRateRouter(teacher_mode="nope")
    with pytest.raises(ValueError):
        SolveRateRouter(threshold=2.0)


def test_blend_mixes_rl_and_teacher_but_not_on_solved_units():
    r = SolveRateRouter(blend=True)
    d = r.route(_ctx(0.3))
    assert set(d.weights) == {TrainingMode.RL, TrainingMode.SFT}
    assert d.weights[TrainingMode.RL] == pytest.approx(rl_informativeness(0.3, 4))
    # a fully solved unit must still be skipped, not blended
    assert r.route(_ctx(1.0)).argmax() == TrainingMode.SKIP


def test_random_router_matches_its_proportions():
    r = RandomRouter({TrainingMode.RL: 0.7, TrainingMode.SFT: 0.3}, seed=1234)
    n = 20000
    counts = {TrainingMode.RL: 0, TrainingMode.SFT: 0}
    for _ in range(n):
        counts[r.route(_ctx(0.5)).argmax()] += 1
    assert counts[TrainingMode.RL] / n == pytest.approx(0.7, abs=0.02)


def test_random_router_is_reproducible_and_isolated_from_global_rng():
    import random as _random

    a = RandomRouter({TrainingMode.RL: 0.5, TrainingMode.SFT: 0.5}, seed=7)
    b = RandomRouter({TrainingMode.RL: 0.5, TrainingMode.SFT: 0.5}, seed=7)
    _random.seed(999)  # perturbing the global RNG must not change routing
    seq_a = [a.route(_ctx(0.5)).argmax() for _ in range(50)]
    _random.seed(1)
    seq_b = [b.route(_ctx(0.5)).argmax() for _ in range(50)]
    assert seq_a == seq_b


def test_random_router_degrades_to_skip_only_when_no_target_exists():
    """SFT is honourable whenever ANY target exists, external or self.

    Changed 2026-08-31 from gating on ``has_teacher`` to gating on ``has_target``. The old
    behaviour degraded EVERY sft draw to skip in every run here, because no run wires an
    external teacher -- so the "matched" control could not emit the mode it was matching and
    its mix collapsed to rl/skip. That is not a control.

    A group with ``solve_rate > 0`` drew at least one correct sample, and that sample is the
    target: rejection-sampling fine-tuning, which is the method's central claim and what
    ``apply_decisions`` already implements without any teacher tensor. The contextual router
    emits sft under exactly these conditions in live training.

    Degrading to skip is still correct when there is NO target at all.
    """
    r = RandomRouter({TrainingMode.SFT: 1.0}, seed=0)
    # No teacher, but the group solved some samples -> its own correct sample IS the target.
    assert r.route(_ctx(0.5, teacher=False)).argmax() == TrainingMode.SFT
    # No teacher and nothing solved -> nothing to train toward, so skip.
    assert r.route(_ctx(0.0, teacher=False)).argmax() == TrainingMode.SKIP
    # An external teacher alone is still sufficient.
    assert r.route(_ctx(0.0, teacher=True)).argmax() == TrainingMode.SFT


def test_inverted_router_is_the_opposite_of_the_criterion():
    fwd, inv = SolveRateRouter(), InvertedRouter()
    for p in (0.0, 0.5, 1.0):
        c = _ctx(p)
        f, i = fwd.route(c).argmax(), inv.route(c).argmax()
        if f == TrainingMode.RL:
            assert i != TrainingMode.RL
        else:
            assert i == TrainingMode.RL
