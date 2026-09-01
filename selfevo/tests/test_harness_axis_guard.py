"""The harness axis is inert end to end, so emitting one must fail loudly.

Nothing reads RoutingDecision.harness: group_apply applies only RL/SFT/SKIP and no production
caller sets can_evolve_harness. A dropped harness action would make a "harness-evolving" arm
train identically to one that is not, and any measured gap would be attributed to a difference
that never happened. These tests pin that the drop is refused, and that the guard does not fire
on the overwhelmingly common case where no harness action is emitted at all.
"""
import pytest

from areal.trainer.ppo.actor import route_all


class _H:
    def __init__(self, name):
        self.name = name


class _D:
    def __init__(self, tag="rl", harness=None):
        self.weights = {tag: 1.0}
        self._tag = tag
        if harness is not None:
            self.harness = _H(harness)

    def argmax(self):
        return self._tag


class _R:
    """Router whose decisions carry the given harness action names."""

    def __init__(self, harnesses):
        self._h = harnesses

    def route_batch(self, contexts):
        return type("A", (), {"decisions": [_D(harness=h) for h in self._h]})()


class _Plain:
    """Router with no route_batch and no harness field at all: the shipped case."""

    def route(self, ctx):
        return _D()


def test_no_harness_action_passes():
    """The default path must not be disturbed: NONE everywhere is the shipped behaviour."""
    out = route_all(_R([None, None, None]), [object()] * 3)
    assert len(out) == 3


def test_explicit_none_passes():
    """A decision that spells NONE explicitly is still the no-op case."""
    out = route_all(_R(["NONE", "NONE"]), [object()] * 2)
    assert len(out) == 2


def test_a_router_without_the_field_passes():
    """Most routers never touch this axis; they must not be penalised for it."""
    out = route_all(_Plain(), [object()] * 4)
    assert len(out) == 4


@pytest.mark.parametrize("harnesses", [
    ["PROPOSE", None, None],
    [None, None, "VALIDATE"],
    ["PROPOSE", "VALIDATE"],
])
def test_any_dropped_harness_action_is_refused(harnesses):
    """One is enough. A single silently dropped action is the whole failure mode."""
    with pytest.raises(ValueError, match="nothing consumes"):
        route_all(_R(harnesses), [object()] * len(harnesses))


def test_the_message_names_how_many_and_where():
    """The message has to be actionable: which decisions, and how many."""
    with pytest.raises(ValueError) as e:
        route_all(_R([None, "PROPOSE", "PROPOSE"]), [object()] * 3)
    assert "2 of 3" in str(e.value)
    assert "index 1" in str(e.value)


def test_a_declared_consumer_is_allowed_through():
    """Once a consumer exists the guard must step aside, or it blocks the feature it guards."""
    out = route_all(_R(["PROPOSE", "VALIDATE"]), [object()] * 2, harness_consumer=True)
    assert len(out) == 2
