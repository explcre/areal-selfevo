"""Group-level routing, driven through the REAL ``PPOActor._compute_advantages``.

These tests construct a ``PPOActor`` with ``engine=None`` and call the actual method rather
than re-deriving its arithmetic in the test file. That distinction has cost this project
before: a sibling test file isolates the group_ids expression as a local ``derive()`` helper,
which pins a copy of the code and cannot notice the copy drifting from the original.

The configuration mirrors the live runs (``examples/math/gsm8k_grpo.yaml`` with step0l's
overrides): ``reward_norm`` group-level, ``adv_norm`` disabled. Group-level reward
normalisation is what makes a unanimous group's advantages identically zero, so a test
without it would not exercise the silence this feature exists for.

The fixtures this file used to define now live in ``selfevo/tests/conftest.py``. Nine other
test files imported them from here, which made this module a conftest that also ran thirteen
tests: importing one fixture pulled in every constant below it and coupled file ordering
under ``pytest -x``. The names are re-imported here, so this file and anything still
importing them from it both behave exactly as before.
"""

from __future__ import annotations

import logging

import pytest
import torch

from areal.api.cli_args import GroupRoutingConfig

from selfevo.tests.conftest import (  # noqa: F401  (re-exported for older importers)
    B,
    G,
    MIXED,
    PROMPT,
    SOLVED_AND_UNSOLVED,
    T,
    advantages,
    make_actor,
    make_batch,
    meta,
)

logging.disable(logging.INFO)

# --------------------------------------------------------------------------- silence ---


def test_a_unanimous_group_really_is_silent_under_the_live_config():
    """The premise. If this fails, every test below is testing nothing."""
    adv = advantages(make_actor(), MIXED)
    solved_group = adv[:G]
    assert solved_group.abs().max() < 1e-6, solved_group
    # And the mixed group is NOT silent, or "routing only touches silent groups" is vacuous.
    assert adv[G:].abs().max() > 1e-6


# -------------------------------------------------------------------------- rollback ---


@pytest.mark.parametrize(
    "gr",
    [
        None,
        GroupRoutingConfig(),                                    # disabled, zero weights
        GroupRoutingConfig(enabled=True),                        # enabled, zero weights
        GroupRoutingConfig(enabled=False, solved_advantage=1.0),  # weight set but disabled
    ],
    ids=["absent", "disabled", "enabled-but-zero", "weighted-but-disabled"],
)
def test_rollback_is_bit_identical(gr):
    """Every off-configuration must reproduce vanilla exactly, bit for bit."""
    torch.manual_seed(0)
    base = advantages(make_actor(None), MIXED)
    torch.manual_seed(0)
    got = advantages(make_actor(gr), MIXED)
    assert torch.equal(base, got), (base - got).abs().max()


# --------------------------------------------------------------------------- solved ----


def test_solved_groups_receive_exactly_the_configured_constant():
    c = 0.5
    adv = advantages(make_actor(GroupRoutingConfig(enabled=True, solved_advantage=c)), MIXED)
    solved = adv[:G]
    assert torch.allclose(solved[:, PROMPT:], torch.full((G, T - PROMPT), c)), solved
    # SFT is on the response only: the prompt must stay at zero.
    assert torch.equal(solved[:, :PROMPT], torch.zeros(G, PROMPT)), solved


def test_the_informative_group_is_left_alone():
    """Routing must not touch groups RL can already learn from."""
    torch.manual_seed(0)
    base = advantages(make_actor(None), MIXED)
    torch.manual_seed(0)
    got = advantages(make_actor(GroupRoutingConfig(enabled=True, solved_advantage=0.5)), MIXED)
    assert torch.equal(base[G:], got[G:]), (base[G:] - got[G:]).abs().max()


def test_the_constant_scales_with_the_configured_weight():
    """A weight that does not reach the tensor would still pass a presence check."""
    small = advantages(make_actor(GroupRoutingConfig(enabled=True, solved_advantage=0.25)), MIXED)
    large = advantages(make_actor(GroupRoutingConfig(enabled=True, solved_advantage=1.0)), MIXED)
    assert pytest.approx(float(large[0, PROMPT]), rel=1e-6) == 4.0 * float(small[0, PROMPT])


# ------------------------------------------------------------------------- unsolved ----


def test_unsolved_groups_receive_a_negative_constant_and_solved_ones_do_not():
    """The two branches are independent: setting one must not move the other."""
    gr = GroupRoutingConfig(enabled=True, unsolved_advantage=-0.3)
    adv = advantages(make_actor(gr), SOLVED_AND_UNSOLVED)
    assert torch.allclose(adv[G:, PROMPT:], torch.full((G, T - PROMPT), -0.3)), adv[G:]
    assert torch.equal(adv[:G], torch.zeros(G, T)), adv[:G]


def test_both_branches_apply_together_with_their_own_signs():
    gr = GroupRoutingConfig(enabled=True, solved_advantage=0.5, unsolved_advantage=-0.3)
    adv = advantages(make_actor(gr), SOLVED_AND_UNSOLVED)
    assert torch.allclose(adv[:G, PROMPT:], torch.full((G, T - PROMPT), 0.5))
    assert torch.allclose(adv[G:, PROMPT:], torch.full((G, T - PROMPT), -0.3))


# ---------------------------------------------------------------------------- guard ----


@pytest.mark.parametrize(
    "kw, needle",
    [
        (dict(solved_advantage=-0.1), "must be >= 0"),
        (dict(unsolved_advantage=0.1), "must be <= 0"),
    ],
)
def test_wrong_signs_are_refused(kw, needle):
    """A positive weight on an unsolved group would train the model to repeat wrong answers."""
    with pytest.raises(ValueError, match=needle):
        GroupRoutingConfig(**kw)


def test_a_batch_with_no_silent_group_is_untouched_even_when_enabled():
    """Reach is a property of the data, not of the flag."""
    all_mixed = [0, 1, 0, 1, 1, 0, 1, 0]
    torch.manual_seed(0)
    base = advantages(make_actor(None), all_mixed)
    torch.manual_seed(0)
    got = advantages(make_actor(GroupRoutingConfig(enabled=True, solved_advantage=0.5)), all_mixed)
    assert torch.equal(base, got)


# ------------------------------------------------- silent is not the same as solved ----
#
# Under group-level reward normalisation every solved group is also silent, so a rule keyed
# on "solved" and a rule keyed on "silent AND solved" agree on every batch and no test using
# only that config can tell them apart -- two mutations survived until these were added.
# Without group centring the two come apart, and keying on the outcome alone would OVERWRITE
# a live gradient instead of filling an empty one.


def test_without_group_centring_a_solved_group_is_not_silent():
    """The premise for the two tests below."""
    adv = advantages(make_actor(None, group_reward_norm=False), MIXED)
    assert adv[:G].abs().max() > 1e-6, adv[:G]


def test_routing_keys_on_silence_not_on_the_outcome():
    """A solved-but-informative group must be left alone: its gradient is real."""
    torch.manual_seed(0)
    base = advantages(make_actor(None, group_reward_norm=False), MIXED)
    torch.manual_seed(0)
    got = advantages(
        make_actor(GroupRoutingConfig(enabled=True, solved_advantage=0.5),
                   group_reward_norm=False),
        MIXED,
    )
    assert torch.equal(base, got), (base - got).abs().max()


def test_an_unsolved_group_can_carry_gradient_under_a_reward_bias():
    """The premise for the test below.

    An all-wrong group scores zero, and a zero reward gives a zero advantage whether or not
    rewards are centred -- so unlike the solved case, disabling centring does NOT separate
    "unsolved" from "silent". A reward bias does: it lifts the whole group off zero while
    the raw reward, which is what ``unsolved`` is computed from, stays zero.
    """
    adv = advantages(
        make_actor(None, group_reward_norm=False, reward_bias=1.0), SOLVED_AND_UNSOLVED
    )
    assert adv[G:].abs().max() > 1e-6, adv[G:]


def test_the_unsolved_branch_also_keys_on_silence():
    """Same property for the negative branch, which has its own conditional."""
    torch.manual_seed(0)
    base = advantages(
        make_actor(None, group_reward_norm=False, reward_bias=1.0), SOLVED_AND_UNSOLVED
    )
    torch.manual_seed(0)
    got = advantages(
        make_actor(GroupRoutingConfig(enabled=True, unsolved_advantage=-0.3),
                   group_reward_norm=False, reward_bias=1.0),
        SOLVED_AND_UNSOLVED,
    )
    assert torch.equal(base, got), (base - got).abs().max()
