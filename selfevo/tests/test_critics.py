"""Tests for ScalarCritic.

An audit of the first version found 17 surviving mutations in five classes: group_size was
never varied (every test ran at G=8), the numeric defaults were unpinned, `basis` was free
text that could assert the opposite of the branch taken, and the unsolved branch was only
ever exercised where `I_RL` was exactly zero. It also showed the advertised asymmetry did
not hold: at G=8, p_hat=0.125 and p_hat=0.875 scored identically. These tests are written
against those specific holes.
"""

from __future__ import annotations

import pytest

from selfevo.critics import CriticScore, ScalarCritic
from selfevo.routing.base import Granularity, RoutingContext
from selfevo.routing.criteria import SilenceSide, rl_informativeness

GS = [2, 4, 8, 16, 32]


def _ctx(p: float, *, teacher: bool = True, g: int = 8, uid: str | None = None,
         gran: Granularity = Granularity.CLUSTER) -> RoutingContext:
    return RoutingContext(solve_rate=p, group_size=g, has_teacher=teacher,
                          unit_id=uid, granularity=gran)


# ----------------------------------------------------- the asymmetry, which used to fail


@pytest.mark.parametrize("g", GS)
def test_nearly_solved_scores_below_nearly_unsolved(g):
    """The audit's headline defect: these used to be EXACTLY equal.

    I_RL is symmetric about 1/2, so an item solved (G-1)/G scored the same as one solved
    1/G. Deciding the split on p_hat rather than on an informativeness threshold is what
    fixes it.
    """
    c = ScalarCritic()
    lo = c.score(_ctx(1 / g, g=g)).value          # nearly unsolved
    hi = c.score(_ctx((g - 1) / g, g=g)).value    # nearly solved
    assert lo == pytest.approx(hi), (
        "I_RL itself is still symmetric -- if this fails the premise changed"
    )
    # ...and the ENDPOINTS must not be, which is what the critic is for.
    assert c.score(_ctx(1.0, g=g)).value < c.score(_ctx(0.0, g=g)).value


@pytest.mark.parametrize("g", GS)
def test_solved_scores_zero_and_is_not_rescued_by_a_teacher(g):
    c = ScalarCritic()
    with_t = c.score(_ctx(1.0, teacher=True, g=g))
    without = c.score(_ctx(1.0, teacher=False, g=g))
    assert with_t.value == 0.0 == without.value
    assert with_t.side is SilenceSide.SOLVED


@pytest.mark.parametrize("g", GS)
def test_unsolved_needs_a_teacher_to_be_worth_anything(g):
    c = ScalarCritic()
    assert c.score(_ctx(0.0, teacher=True, g=g)).value == pytest.approx(0.5)
    assert c.score(_ctx(0.0, teacher=False, g=g)).value == 0.0


# ------------------------------------------------- cross-G comparability, which used to fail


def test_same_difficulty_scores_the_same_across_group_sizes():
    """Previously p=0.5 scored 0.500 at G=2 but 0.992 at G=8 -- dominated by G, not by the
    item, which makes any ranking across group sizes meaningless."""
    c = ScalarCritic()
    vals = [c.score(_ctx(0.5, g=g)).value for g in GS]
    assert all(v == pytest.approx(1.0) for v in vals), vals


def test_the_specific_inversions_the_audit_found_are_gone():
    c = ScalarCritic()
    mid_g2 = c.score(_ctx(0.5, g=2)).value
    unsolved_g8 = c.score(_ctx(0.0, g=8)).value
    near_unsolved_g8 = c.score(_ctx(0.125, g=8)).value
    assert mid_g2 > unsolved_g8, "a maximally informative item must beat an unsolved one"
    assert mid_g2 > near_unsolved_g8


@pytest.mark.parametrize("g", GS)
def test_group_size_is_recorded(g):
    assert ScalarCritic().score(_ctx(0.5, g=g)).group_size == g


# --------------------------------------------------------- group_size is actually read


def test_group_size_changes_the_score_for_a_fixed_solve_rate():
    """Kills the 'hardcode G=8' mutations: every test in the first suite used G=8."""
    c = ScalarCritic()
    vals = {g: c.score(_ctx(0.25, g=g)).value for g in GS}
    assert len(set(round(v, 6) for v in vals.values())) > 1, vals


def test_singleton_group_is_rejected_not_scored():
    """min_group_size refuses G<2 as degenerate; the critic must take the same posture."""
    with pytest.raises(ValueError, match="identically"):
        ScalarCritic().score(_ctx(0.5, g=1))


# ------------------------------------------------------------------ basis is load-bearing


def test_basis_is_branch_specific():
    """The audit showed the unsolved basis could be replaced by the solved one and pass."""
    c = ScalarCritic()
    solved = c.score(_ctx(1.0)).basis
    unsolved = c.score(_ctx(0.0)).basis
    noteacher = c.score(_ctx(0.0, teacher=False)).basis
    informative = c.score(_ctx(0.5)).basis
    assert "nothing to learn" in solved and "teacher" not in solved
    assert "teacher target exists" in unsolved
    assert "no teacher" in noteacher
    assert "not capability gain" in informative
    assert len({solved, unsolved, noteacher, informative}) == 4


def test_the_number_in_the_basis_matches_the_score():
    """Previously the basis could report an I_RL unrelated to `value`."""
    s = ScalarCritic().score(_ctx(0.25, g=8))
    assert f"{s.value:.4f}" in s.basis


def test_coarse_is_flagged_in_both_the_field_and_the_basis():
    small = ScalarCritic().score(_ctx(0.5, g=4))
    large = ScalarCritic(coarse_below=2).score(_ctx(0.5, g=16))
    assert small.coarse and "coarse" in small.basis
    assert not large.coarse and "coarse" not in large.basis


def test_sample_granularity_is_called_out():
    s = ScalarCritic().score(_ctx(0.5, gran=Granularity.SAMPLE))
    assert "unanimity" in s.basis


# ------------------------------------------------------------- defaults and parameters


def test_defaults_are_pinned():
    c = ScalarCritic()
    assert (c.solved_at, c.unsolved_at, c.solved_value, c.unsolved_floor, c.coarse_below) \
        == (1.0, 0.0, 0.0, 0.5, 8)


@pytest.mark.parametrize("g", GS)
def test_solved_at_is_read(g):
    """A high-but-not-perfect solve rate is informative by default and solved when the bar
    is lowered. This is the branch the first version could never reach."""
    p = (g - 1) / g
    assert ScalarCritic().score(_ctx(p, g=g)).side is SilenceSide.INFORMATIVE
    assert ScalarCritic(solved_at=p).score(_ctx(p, g=g)).side is SilenceSide.SOLVED


@pytest.mark.parametrize("g", GS)
def test_unsolved_at_is_read(g):
    p = 1 / g
    assert ScalarCritic().score(_ctx(p, g=g)).side is SilenceSide.INFORMATIVE
    assert ScalarCritic(unsolved_at=p).score(_ctx(p, g=g)).side is SilenceSide.UNSOLVED


def test_solved_value_and_floor_are_read():
    assert ScalarCritic(solved_value=0.3).score(_ctx(1.0)).value == pytest.approx(0.3)
    assert ScalarCritic(unsolved_floor=0.9).score(_ctx(0.0)).value == pytest.approx(0.9)


def test_invalid_parameters_are_rejected():
    for kw in ({"solved_at": 1.5}, {"unsolved_at": -0.1}, {"solved_value": 2.0},
               {"unsolved_floor": -1.0}, {"coarse_below": -1},
               {"solved_at": 0.2, "unsolved_at": 0.5}):
        with pytest.raises(ValueError):
            ScalarCritic(**kw)


# ------------------------------------------------------------------------- history / id


def test_unit_id_is_carried_from_the_context():
    """A live bug in the first version: ctx.unit_id was silently dropped, so a correctly
    populated context produced an unpairable history."""
    s = ScalarCritic().score(_ctx(0.5, uid="prompt-4711"))
    assert s.unit_id == "prompt-4711"


def test_explicit_unit_id_overrides_the_context():
    s = ScalarCritic().score(_ctx(0.5, uid="from-ctx"), unit_id="explicit")
    assert s.unit_id == "explicit"


def test_history_accumulates_in_order_and_is_a_copy():
    c = ScalarCritic()
    c.score(_ctx(0.5), unit_id="a")
    c.score(_ctx(0.0), unit_id="b")
    assert [x.unit_id for x in c.history()] == ["a", "b"]
    c.history().clear()
    assert len(c.history()) == 2
    c.reset()
    assert c.history() == []


def test_score_validates_itself():
    for kw in ({"value": 1.5}, {"basis": ""}, {"group_size": 0}):
        base = {"value": 0.5, "basis": "x", "side": SilenceSide.INFORMATIVE, "group_size": 4}
        with pytest.raises(ValueError):
            CriticScore(**{**base, **kw})


# ------------------------------------------------------------------- registry wiring


def test_scalar_is_built_and_two_level_is_still_a_stub():
    from selfevo.compose import CRITIC_FACTORIES, PipelineConfig, is_valid

    assert CRITIC_FACTORIES["scalar"] is not None
    assert isinstance(CRITIC_FACTORIES["scalar"](), ScalarCritic)
    assert is_valid(PipelineConfig(critic="scalar"))
    assert CRITIC_FACTORIES["two_level"] is None
    assert not is_valid(PipelineConfig(critic="two_level"))


def test_factory_passes_configuration_and_propagates_errors():
    from selfevo.compose import CRITIC_FACTORIES

    assert CRITIC_FACTORIES["scalar"](unsolved_floor=0.25).unsolved_floor == 0.25
    with pytest.raises(ValueError):
        CRITIC_FACTORIES["scalar"](unsolved_floor=5.0)


def test_normalisation_never_leaves_the_unit_interval():
    c = ScalarCritic()
    for g in GS:
        for k in range(g + 1):
            v = c.score(_ctx(k / g, g=g)).value
            assert 0.0 <= v <= 1.0, (g, k, v)
