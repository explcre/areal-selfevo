"""Shared fixtures for the selfevo test suite.

Until this file existed the de-facto conftest was a test module: nine test files imported
``make_actor``, ``make_batch``, ``meta``, ``advantages`` and the batch-shape constants from
:mod:`selfevo.tests.test_group_routing`. The instinct was right -- two definitions of "an
actor configured like the live runs" drift, and the drift is silent -- but importing a
fixture from a test module drags in thirteen unrelated tests' module-level state and
couples file ordering under ``pytest -x``. This is the same one-definition idea with none
of the coupling. ``test_group_routing.py`` imports these from here too, so there is still
exactly one such actor and it is still the one every routing test is compared against.

Nothing here imports torch at module scope. Most of the suite gates torch with a
module-level ``pytest.importorskip("torch")`` and the rest runs without it; a conftest that
imported torch eagerly would turn a missing torch from "skip the files that need it" into
"fail to collect the directory". Helpers that need torch or the trainer import them in
their bodies, which costs one dict lookup per call after the first.

What is deliberately NOT here: the eleven local spellings of ``ctx``. They are not the same
function. ``has_teacher`` defaults True in three files and False in four, one adds
``can_evolve_harness``, one fills the seven-feature ``extra`` mapping and one builds unit
ids in the shape the real actor emits. Collapsing them onto a single default would change
what those tests exercise without changing what they assert, which is the failure mode this
suite is built to avoid. :func:`ctx` below is the canonical builder for NEW tests; an
existing local spelling should move here only when the call is behaviourally identical.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING

import pytest

from selfevo import compose
from selfevo.routing.base import Granularity, RoutingContext

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import torch

    from areal.api.cli_args import GroupRoutingConfig
    from areal.trainer.ppo.actor import PPOActor
    from areal.utils.data import TrajBatchMeta


# --------------------------------------------------------------------- batch geometry ---
#
# The shape every group-routing test is written against, and the reason the numbers are
# here rather than in each file: a test that builds its own batch can pass while disagreeing
# with the one the actor is actually driven with.

B, T, G = 8, 6, 4          # two groups of four
PROMPT = 2                 # first PROMPT columns are prompt, loss_mask == 0 there

# group 0: every sample correct  -> silent because SOLVED
# group 1: two of four correct   -> informative, RL has signal
MIXED = [1, 1, 1, 1, 0, 1, 0, 1]
# group 0 solved, group 1 all wrong -> silent because UNSOLVED
SOLVED_AND_UNSOLVED = [1, 1, 1, 1, 0, 0, 0, 0]


# ------------------------------------------------------------------- the routed actor ---


def make_actor(
    group_routing: GroupRoutingConfig | None = None,
    *,
    group_reward_norm: bool = True,
    reward_bias: float = 0.0,
) -> PPOActor:
    """A CPU actor configured like the live runs.

    Args:
        group_routing: Value for ``config.group_routing``. ``None`` is the shipped default
            and must leave the update untouched.
        group_reward_norm: Whether to centre rewards within the group, as the live configs
            do. Set False to get AReaL's own default (``reward_norm=None``), under which a
            unanimous SOLVED group's advantages are NOT zero -- the configuration that
            separates "silent" from "solved".
        reward_bias: Added to every reward before scaling. A non-zero bias is what separates
            "silent" from "unsolved": an all-wrong group scores zero, and zero rewards give
            zero advantages with or without centring, so only a bias makes such a group
            carry gradient while still being unsolved by raw reward.

    Returns:
        A ``PPOActor`` whose ``_compute_advantages`` can be called directly.
    """
    from areal.api.cli_args import NormConfig, PPOActorConfig
    from areal.trainer.ppo.actor import PPOActor

    cfg = PPOActorConfig(
        path="unused-for-advantage-computation",
        kl_ctl=0.0,
        adv_norm=None,
        reward_bias=reward_bias,
        reward_norm=(
            NormConfig(mean_level="group", std_level="group", group_size=G)
            if group_reward_norm
            else None
        ),
    )
    cfg.group_routing = group_routing
    return PPOActor(cfg, engine=None)


def make_batch(rewards: list[float]) -> dict[str, torch.Tensor]:
    """A minimal batch with a prompt region that must never receive a routed constant.

    Args:
        rewards: Per-sample reward, length ``B``.

    Returns:
        The tensor dict ``_compute_advantages`` consumes.
    """
    import torch

    loss_mask = torch.zeros(B, T)
    loss_mask[:, PROMPT:] = 1.0
    return {
        "input_ids": torch.randint(0, 100, (B, T)),
        "loss_mask": loss_mask,
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "old_logp": torch.zeros(B, T),
        "ref_logp": torch.zeros(B, T),
        "logprobs": torch.zeros(B, T),
        "attention_mask": torch.ones(B, T),
    }


def meta() -> TrajBatchMeta:
    """Group structure matching :func:`make_batch`: two groups of ``G`` rows."""
    from areal.utils.data import TrajBatchMeta

    return TrajBatchMeta(n_trajs=B, traj_group_sizes=[G, G], traj_seqlens=[T] * B)


def advantages(actor: PPOActor, rewards: list[float]) -> torch.Tensor:
    """Run the real advantage computation and return the advantage tensor.

    Args:
        actor: An actor from :func:`make_actor`.
        rewards: Per-sample reward, length ``B``.

    Returns:
        The ``advantages`` entry the actor produced, shape ``(B, T)``.
    """
    return actor._compute_advantages(make_batch(rewards), meta())["advantages"]


# ------------------------------------------------------------------ routing contexts ---


def ctx(
    solve_rate: float = 0.5,
    *,
    group_size: int = 4,
    granularity: Granularity = Granularity.SAMPLE,
    has_teacher: bool = False,
    can_evolve_harness: bool = False,
    unit_id: str | None = None,
    extra: dict[str, float] | None = None,
) -> RoutingContext:
    """A routing context, built by keyword so no field order can silently rebind.

    Defaults are the conservative ones: no teacher and no harness evolution, so a test
    exercising a capability has to ask for it and cannot inherit it from a fixture.

    Args:
        solve_rate: Observed fraction of correct samples, in [0, 1].
        group_size: Samples drawn for this unit.
        granularity: Resolution the context describes.
        has_teacher: Whether an external target is available.
        can_evolve_harness: Whether harness evolution is wired for this run.
        unit_id: Optional identifier for logging and reproduction.
        extra: Router-specific features. ``None`` becomes an empty mapping.

    Returns:
        The context.
    """
    return RoutingContext(
        solve_rate=solve_rate,
        group_size=group_size,
        granularity=granularity,
        has_teacher=has_teacher,
        can_evolve_harness=can_evolve_harness,
        unit_id=unit_id,
        extra={} if extra is None else extra,
    )


def mode_of(decision) -> object:
    """The single mode a hard routing decision selected.

    Args:
        decision: A :class:`~selfevo.routing.base.RoutingDecision`.

    Returns:
        The one mode in ``decision.weights``.

    Raises:
        AssertionError: If the decision is a mixture rather than one-hot. A mixture read as
            a mode would silently report whichever key happened to be first.
    """
    assert len(decision.weights) == 1, f"expected a one-hot decision, got {decision.weights}"
    return next(iter(decision.weights))


# ------------------------------------------------------------------------ statistics ---


class Recorder:
    """Captures ``stats_tracker.scalar`` kwargs so the LOGGED values can be asserted.

    Keeps both readings the suite grew independently: ``calls`` is every kwargs dict in
    arrival order, ``seen`` is the flattened last-write-wins view. They describe the same
    stream, so a test can assert on ordering or on a final value without a second class.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.seen: dict = {}

    def scalar(self, **kw) -> None:
        """Record one ``stats_tracker.scalar`` call."""
        self.calls.append(kw)
        self.seen.update(kw)

    def get(self, key: str):
        """Last logged value for ``key``, or None if it was never logged."""
        for call in reversed(self.calls):
            if key in call:
                return call[key]
        return None


@pytest.fixture
def recorder(monkeypatch) -> Recorder:
    """Swap the actor's module-level stats tracker for one that records.

    Returns:
        The :class:`Recorder` the actor will log into for the duration of the test.
    """
    import areal.trainer.ppo.actor as actor_mod

    r = Recorder()
    monkeypatch.setattr(actor_mod, "stats_tracker", r)
    return r


@pytest.fixture
def clear_stats_tracker() -> Iterator[None]:
    """Drop the process-global stats around one test.

    ``grpo_loss_fn`` appends every microbatch's per-token statistics to a module-level
    ``DistributedStatsTracker``, which asserts that a stat and its denominator have the same
    shape. A test that varies the token count therefore fails inside the tracker on its
    second batch, and the failure looks like a loss bug rather than leaked state.

    Not autouse: an autouse fixture in a conftest applies to every test in the directory,
    which would change the state 1,700 unrelated tests run against. A module that wants it
    everywhere declares a two-line autouse alias that requests this one.
    """
    from areal.utils.stats_tracker import DEFAULT_TRACKER

    DEFAULT_TRACKER.stats.clear()
    yield
    DEFAULT_TRACKER.stats.clear()


# -------------------------------------------------------------------- router registry ---


@contextmanager
def registered_router(name: str, factory: Callable[..., object]) -> Iterator[str]:
    """Register ``factory`` in ``compose.ROUTERS`` under ``name`` and restore afterwards.

    The registry is module-level state shared with every other test in the process, so a
    test that mutates it without restoring makes an unrelated test fail later, in a
    different file, with no visible connection. Restoring distinguishes "was absent" from
    "was present": popping a key that existed is as damaging as leaving one that did not.

    Args:
        name: Registry key. Use one unique to the file, so two test modules registering a
            stub cannot disturb each other.
        factory: Called with no arguments by ``_route_groups``, so anything the router
            needs must be closed over.

    Yields:
        ``name``, for use in a ``GroupRoutingConfig(router=...)``.
    """
    absent = object()
    previous = compose.ROUTERS.get(name, absent)
    compose.ROUTERS[name] = factory
    try:
        yield name
    finally:
        if previous is absent:
            compose.ROUTERS.pop(name, None)
        else:
            compose.ROUTERS[name] = previous


@pytest.fixture
def stub_router() -> Iterator[Callable[[str, Callable[..., object]], str]]:
    """Register stub routers for one test and restore the registry afterwards.

    Yields a ``register(name, factory)`` callable; every registration is undone when the
    test ends, in reverse order, whether it passed or raised.

    A module that always wants the same stub defines its own ``stub_router`` fixture built
    on :func:`registered_router`. That shadows this one, which is the intended way to
    specialise it -- the two differ in what they register, not in how they clean up, and
    the cleanup is the part that was worth sharing.
    """
    with ExitStack() as stack:

        def register(name: str, factory: Callable[..., object]) -> str:
            """Register one router for the rest of this test."""
            return stack.enter_context(registered_router(name, factory))

        yield register
