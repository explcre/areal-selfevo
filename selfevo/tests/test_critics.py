"""Tests for ScalarCritic.

The property that matters most is the ASYMMETRY: I_RL is near zero at both p~0 and p~1, so
a critic that scored on informativeness alone would rate an already-solved item the same as
an unlearnable one. They have opposite training value.
"""
from __future__ import annotations

import pytest

from selfevo.critics import CriticScore, ScalarCritic
from selfevo.routing.base import RoutingContext
from selfevo.routing.criteria import SilenceSide, rl_informativeness


def _ctx(p: float, *, teacher: bool = True, g: int = 8) -> RoutingContext:
    return RoutingContext(solve_rate=p, group_size=g, has_teacher=teacher)


def test_already_solved_scores_zero_not_high():
    """The asymmetry. I_RL is ~0 here, but so is the training value -- for the opposite
    reason to an unsolved item, and it must not be rescued by a teacher."""
    s = ScalarCritic().score(_ctx(1.0))
    assert s.value == 0.0
    assert s.side is SilenceSide.SOLVED
    assert "nothing to learn" in s.basis


def test_unsolved_with_a_teacher_beats_already_solved():
    c = ScalarCritic()
    unsolved = c.score(_ctx(0.0, teacher=True))
    solved = c.score(_ctx(1.0, teacher=True))
    assert unsolved.value > solved.value, "both have I_RL~0 but opposite training value"


def test_unsolved_without_a_teacher_is_worthless_now():
    s = ScalarCritic().score(_ctx(0.0, teacher=False))
    assert s.value == 0.0
    assert "no teacher" in s.basis


def test_informative_items_score_their_informativeness():
    s = ScalarCritic().score(_ctx(0.5))
    assert s.side is SilenceSide.INFORMATIVE
    assert s.value == pytest.approx(rl_informativeness(0.5, 8))


def test_the_most_informative_item_outscores_the_extremes():
    c = ScalarCritic()
    mid = c.score(_ctx(0.5)).value
    assert mid > c.score(_ctx(0.0)).value
    assert mid > c.score(_ctx(1.0)).value


def test_basis_states_what_is_actually_predicted():
    """The critic predicts gradient informativeness, not capability gain. Recording which
    is what lets a meta-critic tell 'the critic was wrong' from 'it measured something
    else'."""
    s = ScalarCritic().score(_ctx(0.5))
    assert "not capability gain" in s.basis


def test_history_accumulates_in_order_for_calibration():
    c = ScalarCritic()
    c.score(_ctx(0.5), unit_id="a")
    c.score(_ctx(0.0), unit_id="b")
    h = c.history()
    assert [x.unit_id for x in h] == ["a", "b"]
    c.reset()
    assert c.history() == []


def test_history_is_a_copy_not_the_live_list():
    c = ScalarCritic()
    c.score(_ctx(0.5))
    c.history().clear()
    assert len(c.history()) == 1


def test_solved_penalty_is_configurable_and_read():
    """Tested where I_RL is NON-ZERO but the item is still SOLVED-side.

    At p=1.0 informativeness is exactly 0, so `info * solved_penalty` is 0 for every
    penalty and such a test constrains nothing -- mutation testing caught exactly that.
    At p=7/8 with G=8, I_RL is 0.656, so a threshold of 0.7 classifies the item as SOLVED
    while leaving the multiplication observable.
    """
    p, g = 0.875, 8
    info = rl_informativeness(p, g)
    assert info > 0.5, "precondition: the multiplicand must not be zero"
    ctx = RoutingContext(solve_rate=p, group_size=g, has_teacher=True)

    half = ScalarCritic(threshold=0.7, solved_penalty=0.5).score(ctx)
    full = ScalarCritic(threshold=0.7, solved_penalty=1.0).score(ctx)
    assert half.side is SilenceSide.SOLVED and full.side is SilenceSide.SOLVED
    assert full.value == pytest.approx(info)
    assert half.value == pytest.approx(info * 0.5)
    assert half.value < full.value, "the penalty must actually scale the score"


def test_solved_penalty_default_zeroes_an_already_solved_item():
    p, g = 0.875, 8
    ctx = RoutingContext(solve_rate=p, group_size=g, has_teacher=True)
    assert ScalarCritic(threshold=0.7).score(ctx).value == 0.0


def test_unsolved_floor_is_configurable_and_read():
    assert ScalarCritic(unsolved_floor=0.9).score(_ctx(0.0)).value == pytest.approx(0.9)


def test_threshold_is_read():
    """At G=8, p=1/8 has I_RL well above 0.1; a high threshold reclassifies it as silent."""
    lo = ScalarCritic(threshold=0.01).score(_ctx(0.125))
    hi = ScalarCritic(threshold=0.99).score(_ctx(0.125))
    assert lo.side is SilenceSide.INFORMATIVE
    assert hi.side is not SilenceSide.INFORMATIVE


def test_invalid_parameters_are_rejected():
    for kw in ({"solved_penalty": 1.5}, {"unsolved_floor": -0.1}, {"threshold": 2.0}):
        with pytest.raises(ValueError):
            ScalarCritic(**kw)


def test_score_validates_its_own_range_and_basis():
    with pytest.raises(ValueError):
        CriticScore(value=1.5, basis="x", side=SilenceSide.INFORMATIVE)
    with pytest.raises(ValueError):
        CriticScore(value=0.5, basis="", side=SilenceSide.INFORMATIVE)


# ------------------------------------------------------- registry wiring


def test_scalar_critic_is_no_longer_a_stub():
    """compose must be able to BUILD it, not merely name it."""
    from selfevo.compose import CRITIC_FACTORIES, PipelineConfig, is_valid

    f = CRITIC_FACTORIES["scalar"]
    assert f is not None, "scalar must have a factory, not None"
    obj = f()
    assert isinstance(obj, ScalarCritic)
    # and the stub guard must now ACCEPT it without allow_stubs
    assert is_valid(PipelineConfig(critic="scalar"))


def test_two_level_is_still_correctly_marked_a_stub():
    """Wiring one factory must not silently mark the others as built."""
    from selfevo.compose import CRITIC_FACTORIES, PipelineConfig, is_valid

    assert CRITIC_FACTORIES["two_level"] is None
    assert not is_valid(PipelineConfig(critic="two_level"))


def test_factory_passes_configuration_through():
    from selfevo.compose import CRITIC_FACTORIES

    obj = CRITIC_FACTORIES["scalar"](unsolved_floor=0.25)
    assert obj.unsolved_floor == 0.25
