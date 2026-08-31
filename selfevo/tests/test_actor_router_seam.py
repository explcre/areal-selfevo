"""The Router -> advantage path, driven through the REAL actor.

``_route_groups`` is the step that turns ``router=contextual`` and ``router=code_policy``
from registry entries into training arms: without it a Router is a component with no caller,
and every test of one proves only that the component works in isolation. The seam had no
test at all, so nothing established that a decision made by a Router reaches the tensor the
loss reads.

Everything here goes through ``PPOActor._compute_advantages`` -- the same entry point
training calls -- rather than calling ``_route_groups`` directly. A test that calls the
helper cannot catch the helper being unreachable, which is exactly the failure mode this
file exists to rule out.

The fixtures are imported from :mod:`selfevo.tests.test_group_routing` rather than copied:
two definitions of "an actor configured like the live runs" drift, and the drift is silent.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from areal.api.cli_args import GroupRoutingConfig
from selfevo import compose
from selfevo.routing.base import RoutingDecision, TrainingMode

from selfevo.tests.test_group_routing import (  # noqa: E402
    G,
    MIXED,
    PROMPT,
    advantages,
    make_actor,
)

STUB = "_seam_stub"


class RecordingRouter:
    """A Router whose decision is fixed and whose calls are observable.

    Fixed on purpose: the point is to prove the ACTOR carries a decision to the tensor, and
    a router with interesting behaviour of its own would make a failure ambiguous between
    the two.

    Args:
        mode: The mode every unit is routed to.
    """

    def __init__(self, mode: str = TrainingMode.SFT) -> None:
        self.mode = mode
        self.seen: list[str] = []
        self.observed: list[dict] = []

    def route(self, ctx) -> RoutingDecision:
        """Record the unit and return the fixed decision."""
        self.seen.append(ctx.unit_id)
        return RoutingDecision(weights={self.mode: 1.0}, reason="stub")

    def observe(self, outcomes) -> None:
        """Record a feedback call so its TIMING can be asserted."""
        self.observed.append(dict(outcomes))


@pytest.fixture
def stub_router():
    """Register a recording router and restore the registry afterwards.

    The registry is module-level state shared with every other test in the process, so a
    test that mutates it without restoring makes an unrelated test fail later, in a
    different file, with no visible connection.
    """
    made: list[RecordingRouter] = []

    def factory(*_a, **_kw):
        r = RecordingRouter()
        made.append(r)
        return r

    previous = compose.ROUTERS.get(STUB, KeyError)
    compose.ROUTERS[STUB] = factory
    try:
        yield made
    finally:
        if previous is KeyError:
            compose.ROUTERS.pop(STUB, None)
        else:
            compose.ROUTERS[STUB] = previous


class AlternatingRouter(RecordingRouter):
    """Routes odd and even units differently so attribution is not vacuous.

    ``batch_outcomes`` refuses a batch whose decisions were all the same, so a router that
    never varies can never generate feedback -- which makes it useless for testing WHEN
    feedback happens.
    """

    def route(self, ctx):
        """Alternate SFT and RL by unit index within the batch."""
        self.seen.append(ctx.unit_id)
        mode = TrainingMode.SFT if len(self.seen) % 2 else TrainingMode.RL
        return RoutingDecision(weights={mode: 1.0}, reason="alternating")


@pytest.fixture
def alternating_router():
    """Same registry discipline as :func:`stub_router`, with a mode-varying router."""
    made: list[AlternatingRouter] = []

    def factory(*_a, **_kw):
        r = AlternatingRouter()
        made.append(r)
        return r

    previous = compose.ROUTERS.get(STUB, KeyError)
    compose.ROUTERS[STUB] = factory
    try:
        yield made
    finally:
        if previous is KeyError:
            compose.ROUTERS.pop(STUB, None)
        else:
            compose.ROUTERS[STUB] = previous


def routed_actor(**kw):
    """An actor whose group routing is driven by the recording router."""
    return make_actor(
        GroupRoutingConfig(enabled=True, solved_advantage=0.5, router=STUB, **kw)
    )


# ------------------------------------------------------- the decision reaches the loss ---


def test_a_router_decision_reaches_the_advantage_tensor(stub_router):
    """The whole point of the seam: an SFT decision must show up in the tensor.

    Asserted on EVERY group, including the informative one. The fixed rule touches only
    silent groups, so a run in which the router path is quietly bypassed would leave the
    informative group's advantages alone -- and that difference is what separates "the
    router decided" from "the hardcoded rule decided".
    """
    adv = advantages(routed_actor(), MIXED)
    response = adv[:, PROMPT:]
    assert torch.allclose(response, torch.full_like(response, 0.5)), response


def test_the_prompt_region_is_left_exactly_as_it_was(stub_router):
    """A routed constant on a prompt token would be gradient on text the model did not choose.

    Asserted against the UNROUTED tensor, not against zero: the actor leaves real GAE values
    on prompt positions (measured at -0.87 for an informative group), so "the prompt region is
    zero" is false before routing ever runs, and a test asserting it would fail for a reason
    that has nothing to do with the seam. What must hold is that routing does not move them.
    """
    torch.manual_seed(0)
    base = advantages(make_actor(None), MIXED)
    torch.manual_seed(0)
    got = advantages(routed_actor(), MIXED)
    assert torch.equal(base[:, :PROMPT], got[:, :PROMPT]), (
        (base[:, :PROMPT] - got[:, :PROMPT]).abs().max()
    )


def test_the_written_magnitude_is_the_configured_weight(stub_router):
    """Scales with config, so a hardcoded constant that happens to match 0.5 is caught."""
    adv = advantages(
        make_actor(GroupRoutingConfig(enabled=True, solved_advantage=0.25, router=STUB)),
        MIXED,
    )
    response = adv[:, PROMPT:]
    assert torch.allclose(response, torch.full_like(response, 0.25)), response


def test_every_group_is_offered_to_the_router(stub_router):
    """One decision per group, not per row and not per batch."""
    advantages(routed_actor(), MIXED)
    assert len(stub_router) == 1
    assert len(stub_router[0].seen) == len(MIXED) // G == 2


# ------------------------------------------------------------------- learned state -----


def test_the_router_is_built_once_and_reused(stub_router):
    """A learned router rebuilt per batch would reset the thing that is supposed to learn.

    This is the failure the caching exists to prevent, and it is invisible in a single-batch
    test: the router still routes, it just never accumulates anything.
    """
    actor = routed_actor()
    advantages(actor, MIXED)
    advantages(actor, MIXED)
    assert len(stub_router) == 1, f"router rebuilt {len(stub_router)} times"
    assert actor._selfevo_router is stub_router[0]


def test_unit_ids_do_not_collide_across_batches(stub_router):
    """Feedback is keyed by unit id, so ids reused across batches would credit the wrong unit."""
    actor = routed_actor()
    advantages(actor, MIXED)
    first = list(stub_router[0].seen)
    advantages(actor, MIXED)
    second = stub_router[0].seen[len(first):]
    assert set(first).isdisjoint(second), (first, second)


# ---------------------------------------------------------------- feedback timing ------


def test_no_feedback_before_an_outcome_exists(stub_router):
    """The first batch has no previous batch to credit; observing then would credit noise."""
    actor = routed_actor()
    advantages(actor, MIXED)
    assert stub_router[0].observed == []


def test_a_uniform_batch_is_refused_rather_than_credited(stub_router):
    """Every group taking the SAME mode makes attribution provably vacuous.

    One scalar reward change cannot be divided among decisions that were all identical, so
    ``batch_outcomes`` raises ``ConfoundedUpdate`` and the router is not updated at all.
    Discovered by this test: the fixed-mode stub NEVER produces feedback, which is correct
    and is the reason the timing test below has to vary the mode.
    """
    actor = routed_actor()
    advantages(actor, MIXED)
    advantages(actor, MIXED)
    assert stub_router[0].observed == [], stub_router[0].observed


def test_feedback_credits_the_previous_batch_not_the_current_one(alternating_router):
    """A decision's outcome is not observable until after the update it took part in.

    Crediting the current batch would score a decision against the reward that PRECEDED it,
    which is not merely noisy -- it is the wrong sign of causality.

    Needs a router that varies its mode WITHIN the batch, or the update is refused as
    confounded before the timing is ever exercised.
    """
    actor = routed_actor()
    advantages(actor, MIXED)
    first_batch_units = set(alternating_router[0].seen)
    advantages(actor, MIXED)
    assert len(alternating_router[0].observed) == 1, alternating_router[0].observed
    credited = set(alternating_router[0].observed[0])
    assert credited <= first_batch_units, (credited, first_batch_units)
    assert credited, "an update that credits nothing is not feedback"


# --------------------------------------------------------------------- refusals --------


def test_an_unregistered_router_name_is_refused(stub_router):
    """Silently falling back to the fixed rule would report an arm that never ran."""
    with pytest.raises(ValueError, match="not a registered router"):
        advantages(
            make_actor(GroupRoutingConfig(enabled=True, solved_advantage=0.5,
                                          router="no_such_router")),
            MIXED,
        )


def test_an_rl_decision_leaves_the_batch_exactly_as_grpo_left_it(stub_router):
    """Rollback INSIDE the router path: routing to RL must be bit-identical to not routing.

    Distinct from the config-level rollback test: this one proves the seam can be entered
    and still change nothing, so a non-zero diff in an RL-only arm is a bug in the seam
    rather than in a mode.
    """
    torch.manual_seed(0)
    base = advantages(make_actor(None), MIXED)

    def rl_factory(*_a, **_kw):
        return RecordingRouter(mode=TrainingMode.RL)

    compose.ROUTERS[STUB] = rl_factory
    torch.manual_seed(0)
    got = advantages(routed_actor(), MIXED)
    assert torch.equal(base, got), (base - got).abs().max()
