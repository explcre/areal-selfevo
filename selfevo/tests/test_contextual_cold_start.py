"""The registered contextual router must actually explore.

Measured live on 2026-08-31: with the dataclass default of ``cold_start_rounds=0`` the
contextual arm routed EVERY group to RL at every step, changed nothing
(``route/changed_row_fraction`` 0.0), and had every feedback update refused as confounded.
The learned arm was bit-identical to the off arm while reporting as a learned run.

The loop is closed, which is why a single test on the router in isolation would not have
caught it: no exploration produces a uniform batch, a uniform batch makes attribution vacuous,
a refused update leaves the arm estimates untouched, and untouched estimates tie and resolve
to the same mode again. These tests therefore go through the REGISTRY -- the object a training
run actually builds -- rather than through a directly constructed router.
"""

from __future__ import annotations

import pytest

from selfevo.compose import ROUTERS
from selfevo.routing.base import RoutingContext


def _ctx(i: int) -> RoutingContext:
    """A routable unit with a self-target, so every mode is usable."""
    return RoutingContext(
        solve_rate=0.5,
        group_size=8,
        has_teacher=False,
        unit_id=f"u{i}",
        extra={"solve_rate": 0.5, "reward_std": 0.4, "mean_response_len": 100.0, "len_dispersion": 0.2,
               "mean_logprob": -0.5, "logprob_dispersion": 0.1, "truncated_fraction": 0.0},
    )


def test_the_registered_router_has_a_nonzero_cold_start():
    """_route_groups builds routers with no arguments, so the DEFAULT is what training gets."""
    r = ROUTERS["contextual"]()
    assert getattr(r, "cold_start_rounds", 0) > 0


def test_the_registered_router_does_not_route_a_whole_batch_to_one_mode():
    """The property that actually matters: a batch must contain more than one mode.

    Asserted over 64 units, the live group count for one batch. A router that returns a
    constant here produces exactly the confounded-feedback deadlock measured in the ctx run.
    """
    r = ROUTERS["contextual"]()
    modes = {r.route(_ctx(i)).argmax() for i in range(64)}
    assert len(modes) > 1, f"routed all 64 units to {modes}"


def test_an_explicit_cold_start_still_wins():
    """The default must not become a hardcode; ablations need to set their own."""
    r = ROUTERS["contextual"](cold_start_rounds=5)
    assert r.cold_start_rounds == 5


def test_every_mode_is_actually_tried_during_cold_start():
    """Seeding one arm is not exploration; each usable arm needs observations."""
    r = ROUTERS["contextual"]()
    modes = [r.route(_ctx(i)).argmax() for i in range(r.cold_start_rounds)]
    assert len(set(modes)) >= 2, set(modes)
    counts = {m: modes.count(m) for m in set(modes)}
    assert min(counts.values()) >= 2, counts
