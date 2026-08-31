"""The random control must not silently become the off arm.

Measured 2026-08-31: the `rnd` arm ran with rl_groups=64, sft_groups=0 and skip_groups=0 at
every step. `RandomRouter.proportions` defaults to ``{rl: 1.0}`` and ``_route_groups``
constructs routers with no arguments, so the "matched control" was bit-identical to the off
arm while reporting as a control -- the second instance of the same defect as the contextual
router's zero cold start, in the same registry.

RandomRouter's own contract is that it must run at "the proportions the criterion router
actually produced, measured, not assumed", so there is no correct default: the factory
refuses rather than guessing. These tests go through the REGISTRY, which is the path that was
broken.
"""

from __future__ import annotations

import pytest

from selfevo.compose import ROUTERS, _parse_proportions
from selfevo.routing.base import RoutingContext

SPEC = "rl=0.2946,sft=0.3527,skip=0.3526"


def _ctx(i: int) -> RoutingContext:
    """A unit with a self-target, so sft is usable."""
    return RoutingContext(solve_rate=0.5, group_size=8, has_teacher=False, unit_id=f"u{i}")


def test_the_control_refuses_to_run_without_measured_proportions(monkeypatch):
    """A silent fallback to {rl: 1.0} is a control that controls nothing."""
    monkeypatch.delenv("SELFEVO_RANDOM_PROPORTIONS", raising=False)
    with pytest.raises(ValueError, match="needs proportions"):
        ROUTERS["random"]()


def test_proportions_come_from_the_environment(monkeypatch):
    """The value belongs to the run, not to the code."""
    monkeypatch.setenv("SELFEVO_RANDOM_PROPORTIONS", SPEC)
    r = ROUTERS["random"]()
    assert dict(r.proportions) == {"rl": 0.2946, "sft": 0.3527, "skip": 0.3526}


def test_an_explicit_argument_still_wins(monkeypatch):
    """Ablations and tests must be able to set proportions directly."""
    monkeypatch.setenv("SELFEVO_RANDOM_PROPORTIONS", SPEC)
    r = ROUTERS["random"](proportions={"rl": 1.0, "sft": 1.0})
    assert dict(r.proportions) == {"rl": 1.0, "sft": 1.0}


def test_it_actually_emits_more_than_one_mode(monkeypatch):
    """The behavioural claim, not just the field value.

    Asserted over 64 units, one batch at the live group count. This is the assertion that
    would have caught the inert run.
    """
    monkeypatch.setenv("SELFEVO_RANDOM_PROPORTIONS", SPEC)
    r = ROUTERS["random"]()
    modes = [r.route(_ctx(i)).argmax() for i in range(64)]
    assert len(set(modes)) > 1, f"routed all 64 units to {set(modes)}"


def test_emitted_proportions_track_the_requested_ones(monkeypatch):
    """A control whose mix does not match is not matched.

    Loose bound: 2000 draws, each mode within 0.05 of its target. Tight enough to catch a
    mode being dropped or the weights being ignored, loose enough not to be flaky.
    """
    monkeypatch.setenv("SELFEVO_RANDOM_PROPORTIONS", SPEC)
    r = ROUTERS["random"]()
    n = 2000
    modes = [r.route(_ctx(i)).argmax() for i in range(n)]
    for mode, want in (("rl", 0.2946), ("sft", 0.3527), ("skip", 0.3526)):
        got = modes.count(mode) / n
        assert abs(got - want) < 0.05, f"{mode}: got {got:.3f} want {want:.3f}"


# "rl=0.5,sft" is the case that matters and the one a naive test misses: the specs that
# parse to NOTHING are caught by the empty check regardless, so only a spec mixing a VALID
# pair with a malformed one distinguishes "refuse" from "silently skip". Skipping there
# yields {rl: 0.5} -- a control that never emits sft, which is precisely the defect this
# whole file exists to prevent.
@pytest.mark.parametrize("bad", ["", "rl", "rl=x", "   ", "rl=0.5,sft", "rl=0.5,sft=abc"])
def test_malformed_specs_are_refused(bad, monkeypatch):
    """A dropped mode is a control that never emits it; fail loudly instead."""
    with pytest.raises(ValueError):
        _parse_proportions(bad)
