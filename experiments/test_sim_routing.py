"""The routing simulator's cost accounting must not charge for an update it never makes.

``run`` charges ``UPDATE_COSTS[mode]`` for every routed unit and then applies the update in
an ``if RL / elif SFT`` chain with no ``else``. ``distill`` is registered, costs 1.0, and
matches neither branch, so a unit routed to it spent a full rollout plus a full update and
moved ``p`` by nothing -- silently. An audit measured 400 of 1000 units in exactly that
state, and the arm's reported spend was therefore the spend of a mode that never ran.

This is the simulator that decides whether the whole routing idea is worth GPU time, so a
silent no-op here does not crash a run: it publishes a number.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_routing as sim  # noqa: E402

from selfevo.routing.base import RoutingContext, RoutingDecision, TrainingMode  # noqa: E402


class FixedRouter:
    """Emits one mode for every unit, whatever it is asked."""

    def __init__(self, mode):
        self.mode = mode

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        return RoutingDecision({self.mode: 1.0}, reason="fixed")


def units(n=4):
    return sim.make_units("mixed", n, random.Random(0))


def test_a_mode_the_simulator_cannot_update_is_refused_not_charged():
    """The measured defect: full cost, no movement, no complaint."""
    with pytest.raises(ValueError, match="no update branch"):
        sim.run(
            FixedRouter(TrainingMode.DISTILL),
            units(),
            budget=8.0,
            group_size=8,
            lr_rl=0.1,
            lr_sft=0.1,
            seed=0,
        )


@pytest.mark.parametrize("mode", [TrainingMode.RL, TrainingMode.SFT, TrainingMode.SKIP])
def test_the_modes_the_simulator_does_implement_still_run(mode):
    """The refusal must name the missing branch, not every mode."""
    mean_p, counts, made = sim.run(
        FixedRouter(mode),
        units(),
        budget=8.0,
        group_size=8,
        lr_rl=0.1,
        lr_sft=0.1,
        seed=0,
    )
    assert 0.0 <= mean_p <= 1.0
    assert counts.get(mode, 0) > 0
    assert made
