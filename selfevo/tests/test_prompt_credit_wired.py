"""The prompt-credit path, driven through the REAL actor.

`credit="prompt"` is the designed fix for the measured null: at `credit="batch"` the router
receives one scalar for all 64 decisions, provably converges every arm to the same parameter
vector, and scored -0.0020 against its matched random control. These tests establish that the
new path (a) is reachable from config, (b) actually credits the PRIOR decision for the SAME
prompt, and (c) leaves the default path untouched.

Driven through `_compute_advantages` rather than `_route_groups`, because a test that calls
the helper cannot catch the helper being unreachable.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import areal.trainer.ppo.actor as actor_mod
from areal.api.cli_args import GroupRoutingConfig
from selfevo import compose
from selfevo.routing.base import RoutingDecision, TrainingMode

from selfevo.tests.test_group_routing import (  # noqa: E402
    G,
    MIXED,
    make_actor,
    make_batch,
    meta,
)

STUB = "_credit_stub"


class Recorder:
    """Router that alternates modes and records every observe() call."""

    def __init__(self) -> None:
        self.seen: list[str] = []
        self.observed: list[dict] = []

    def route(self, ctx):
        self.seen.append(ctx.unit_id)
        mode = TrainingMode.SFT if len(self.seen) % 2 else TrainingMode.RL
        return RoutingDecision({mode: 1.0}, reason="stub")

    def observe(self, outcomes) -> None:
        self.observed.append(dict(outcomes))


@pytest.fixture
def stub():
    made: list[Recorder] = []

    def factory(*_a, **_kw):
        r = Recorder()
        made.append(r)
        return r

    prev = compose.ROUTERS.get(STUB, KeyError)
    compose.ROUTERS[STUB] = factory
    try:
        yield made
    finally:
        if prev is KeyError:
            compose.ROUTERS.pop(STUB, None)
        else:
            compose.ROUTERS[STUB] = prev


def _actor(credit: str):
    return make_actor(
        GroupRoutingConfig(enabled=True, solved_advantage=0.5, router=STUB, credit=credit)
    )


def _run(actor, rewards, ids=None, response_tag=0):
    """One batch through the real advantage path.

    `response_tag` varies the RESPONSE tokens while holding the prompt fixed. That is the
    realistic case and the one that discriminates: two rollouts of the same prompt differ in
    their responses, so a key derived from the whole row would never pair them. An earlier
    version of these tests reused identical rows across sightings, and a mutation replacing
    `prompt_key(...)` with `str(row)` passed every one of them.
    """
    b = make_batch(list(rewards))
    if ids is not None:
        row = ids.clone()
        row[:, 2:] = 100 + response_tag      # response region differs per sighting
        b["input_ids"] = row
    return actor._compute_advantages(b, meta())


def _same_prompts():
    """input_ids where each group's rows share prompt tokens, and the groups differ."""
    ids = torch.zeros(8, 6, dtype=torch.long)
    ids[:G, :2] = torch.tensor([11, 22])      # group 0 prompt
    ids[G:, :2] = torch.tensor([33, 44])      # group 1 prompt
    return ids


def test_the_prompt_path_is_reachable_from_config(stub):
    """credit='prompt' must actually take the new branch, not fall back."""
    a = _actor("prompt")
    _run(a, MIXED, _same_prompts())
    assert getattr(a, "_selfevo_ledger", None) is not None


def test_the_default_path_does_not_build_a_ledger(stub):
    """Rollback must be exact: credit='batch' behaves as before."""
    a = _actor("batch")
    _run(a, MIXED, _same_prompts())
    assert getattr(a, "_selfevo_ledger", None) is None


def test_the_first_batch_credits_nothing(stub):
    """No prompt has been seen twice yet; crediting would invent a delta."""
    a = _actor("prompt")
    _run(a, MIXED, _same_prompts())
    assert stub[0].observed == []


def test_the_second_sighting_credits_the_prior_decision_for_that_prompt(stub):
    """The paired observation, end to end through the actor."""
    a = _actor("prompt")
    ids = _same_prompts()
    _run(a, MIXED, ids, response_tag=0)
    first_units = set(stub[0].seen)
    _run(a, MIXED, ids, response_tag=7)   # same prompts, DIFFERENT responses -> must pair
    assert len(stub[0].observed) == 1, stub[0].observed
    credited = stub[0].observed[0]
    assert credited, "second sighting credited nothing"
    assert set(credited) <= first_units, (set(credited), first_units)


def test_a_different_prompt_does_not_pair(stub):
    """Crediting across prompts would attribute one task's outcome to another's decision."""
    a = _actor("prompt")
    _run(a, MIXED, _same_prompts())
    other = _same_prompts()
    other[:, :2] = torch.tensor([77, 88])      # both groups now a different prompt
    _run(a, MIXED, other)
    assert stub[0].observed == [] or not stub[0].observed[0]


def test_the_credited_value_tracks_the_prompt_solve_rate(stub):
    """A prompt that improved must yield a positive delta, and one that got worse negative."""
    a = _actor("prompt")
    ids = _same_prompts()
    _run(a, [0, 0, 0, 0, 0, 0, 0, 0], ids, response_tag=0)   # solve_rate 0
    _run(a, [1, 1, 1, 1, 1, 1, 1, 1], ids, response_tag=9)   # solve_rate 1, new responses
    assert stub[0].observed, "no credit emitted"
    vals = [o.value for o in stub[0].observed[0].values()]
    assert vals and all(v > 0 for v in vals), vals


def test_an_unknown_credit_mode_is_refused():
    """A silent fallback would report a per-prompt arm that never ran."""
    with pytest.raises(ValueError, match="credit must be"):
        GroupRoutingConfig(enabled=True, credit="sometimes")
