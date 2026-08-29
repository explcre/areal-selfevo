"""Tests for token-level RL-dead gating."""

from __future__ import annotations

import pytest
import torch

from selfevo.routing.token_level import (
    OnPolicyViolation,
    assert_on_policy,
    rl_dead_mask,
    shared_prefix_lengths,
    token_gates,
)

P = 0  # a prompt token id, always masked out of the loss


def _case(rows: list[list[int]], gen: list[list[int]], gids: list[int]):
    return (
        torch.tensor(rows),
        torch.tensor(gen, dtype=torch.bool),
        torch.tensor(gids),
    )


# ------------------------------------------------------------------ on-policy guard


def test_assert_on_policy_accepts_exact_ratio_one():
    assert_on_policy(torch.ones(10), 0.0)


def test_assert_on_policy_rejects_drifted_ratio():
    w = torch.ones(10)
    w[3] = 1.05
    with pytest.raises(OnPolicyViolation, match="not.*on-policy"):
        assert_on_policy(w, 0.0)


def test_mean_one_with_offsetting_deviations_is_still_rejected():
    """The check must be elementwise: offsetting deviations average to exactly 1."""
    w = torch.tensor([0.5, 1.5, 1.0, 1.0])
    assert w.mean().item() == pytest.approx(1.0)
    with pytest.raises(OnPolicyViolation):
        assert_on_policy(w, 0.0)


def test_any_clipping_is_rejected_by_default():
    with pytest.raises(OnPolicyViolation, match="clip"):
        assert_on_policy(torch.ones(4), 1e-6)


def test_empty_ratio_tensor_does_not_crash():
    assert_on_policy(torch.empty(0), 0.0)


# --------------------------------------------------------------- shared prefix length


def test_immediate_divergence_gives_zero_prefix():
    toks, gen, gids = _case(
        [[P, 7, 8, 9], [P, 4, 8, 9]],
        [[0, 1, 1, 1], [0, 1, 1, 1]],
        [0, 0],
    )
    assert shared_prefix_lengths(toks, gen, gids).tolist() == [0, 0]
    assert rl_dead_mask(toks, gen, gids).sum().item() == 0


def test_partial_prefix_is_counted_up_to_first_divergence():
    toks, gen, gids = _case(
        [[P, 7, 8, 1], [P, 7, 8, 2]],
        [[0, 1, 1, 1], [0, 1, 1, 1]],
        [0, 0],
    )
    assert shared_prefix_lengths(toks, gen, gids).tolist() == [2, 2]
    dead = rl_dead_mask(toks, gen, gids)
    assert dead[0].tolist() == [False, True, True, False]


def test_full_agreement_makes_every_generated_token_dead():
    toks, gen, gids = _case(
        [[P, 7, 8, 9], [P, 7, 8, 9]],
        [[0, 1, 1, 1], [0, 1, 1, 1]],
        [0, 0],
    )
    assert shared_prefix_lengths(toks, gen, gids).tolist() == [3, 3]
    dead = rl_dead_mask(toks, gen, gids)
    assert dead.sum().item() == 6  # every generated token in both rows


def test_prompt_tokens_are_never_counted_or_marked_dead():
    """Including the prompt would inflate every prefix; it is shared by construction."""
    toks, gen, gids = _case(
        [[P, P, 7, 1], [P, P, 7, 2]],
        [[0, 0, 1, 1], [0, 0, 1, 1]],
        [0, 0],
    )
    assert shared_prefix_lengths(toks, gen, gids).tolist() == [1, 1]
    dead = rl_dead_mask(toks, gen, gids)
    assert dead[:, 0].sum().item() == 0 and dead[:, 1].sum().item() == 0
    assert dead[0, 2].item() is True


def test_group_of_one_is_entirely_dead_not_entirely_live():
    """A singleton group has A = r - rbar = 0 identically, so every position is dead."""
    toks, gen, gids = _case([[P, 7, 8, 9]], [[0, 1, 1, 1]], [0])
    assert shared_prefix_lengths(toks, gen, gids).tolist() == [3]
    assert rl_dead_mask(toks, gen, gids).sum().item() == 3


def test_multiple_groups_are_independent():
    toks, gen, gids = _case(
        [[P, 7, 1, 0], [P, 7, 2, 0], [P, 5, 6, 0], [P, 9, 6, 0]],
        [[0, 1, 1, 0]] * 4,
        [0, 0, 1, 1],
    )
    # group 0 agrees on one token, group 1 diverges immediately
    assert shared_prefix_lengths(toks, gen, gids).tolist() == [1, 1, 0, 0]


def test_ragged_lengths_stop_at_the_shorter_member():
    toks, gen, gids = _case(
        [[P, 7, 8, 9], [P, 7, 0, 0]],
        [[0, 1, 1, 1], [0, 1, 0, 0]],
        [0, 0],
    )
    # only position 1 is generated in both members
    assert shared_prefix_lengths(toks, gen, gids).tolist() == [1, 1]


# ------------------------------------------------------------------------- gates


def test_gates_are_disjoint_and_cover_generated_tokens_when_teacher_present():
    toks, gen, gids = _case(
        [[P, 7, 8, 1], [P, 7, 8, 2]],
        [[0, 1, 1, 1], [0, 1, 1, 1]],
        [0, 0],
    )
    a_rl, a_t = token_gates(toks, gen, gids, teacher_available=True)
    assert torch.all(a_rl * a_t == 0), "gates must be disjoint"
    assert torch.equal(a_rl + a_t, gen.float()), "together they must cover generated tokens"


def test_no_teacher_means_dead_positions_are_left_unweighted():
    """Not self-imitated: reinforcing the group's own prefix sharpens a collapsing policy."""
    toks, gen, gids = _case(
        [[P, 7, 8, 1], [P, 7, 8, 2]],
        [[0, 1, 1, 1], [0, 1, 1, 1]],
        [0, 0],
    )
    a_rl, a_t = token_gates(toks, gen, gids, teacher_available=False)
    assert a_t.sum().item() == 0
    assert a_rl.sum().item() == 2  # only the two post-divergence tokens


def test_gates_are_zero_outside_generated_positions():
    toks, gen, gids = _case(
        [[P, P, 7, 1], [P, P, 7, 2]],
        [[0, 0, 1, 1], [0, 0, 1, 1]],
        [0, 0],
    )
    a_rl, a_t = token_gates(toks, gen, gids, teacher_available=True)
    assert a_rl[:, :2].sum().item() == 0
    assert a_t[:, :2].sum().item() == 0


# ----------------------------------------------------------------------- validation


def test_shape_mismatches_raise():
    with pytest.raises(ValueError):
        shared_prefix_lengths(torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 4).bool(), torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError):
        shared_prefix_lengths(torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3).bool(), torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError):
        shared_prefix_lengths(torch.zeros(3, dtype=torch.long), torch.zeros(3).bool(), torch.zeros(1, dtype=torch.long))


# ------------------------------------------------- zero-sum advantage guard (MEDS-style)

from selfevo.routing.token_level import assert_zero_sum_advantage  # noqa: E402


def test_centred_advantages_pass():
    adv = torch.tensor([0.5, 0.5, -0.5, -0.5])
    assert_zero_sum_advantage(adv, torch.tensor([0, 0, 0, 0]))


def test_meds_entropy_bonus_breaks_centring_and_is_caught():
    """MEDS dp_actor.py:560 -- advantages += min(0.4*entropy, |A|/2). Entropy >= 0."""
    adv = torch.tensor([0.5, 0.5, -0.5, -0.5])
    entropy = torch.tensor([0.3, 0.3, 0.3, 0.3])
    shaped = adv + torch.min(0.4 * entropy, adv.abs() / 2)
    assert shaped.sum().item() > 0, "precondition: the bonus must break centring"
    with pytest.raises(OnPolicyViolation, match="not 0"):
        assert_zero_sum_advantage(shaped, torch.tensor([0, 0, 0, 0]))


def test_length_normalisation_breaks_centring_and_is_caught():
    adv = torch.tensor([0.5, 0.5, -0.5, -0.5])
    lengths = torch.tensor([10.0, 200.0, 12.0, 15.0])
    with pytest.raises(OnPolicyViolation):
        assert_zero_sum_advantage(adv / lengths, torch.tensor([0, 0, 0, 0]))


def test_groups_are_checked_independently():
    # group 0 centred, group 1 not
    adv = torch.tensor([0.5, -0.5, 1.0, 1.0])
    with pytest.raises(OnPolicyViolation, match="group 1"):
        assert_zero_sum_advantage(adv, torch.tensor([0, 0, 1, 1]))


def test_per_token_advantages_are_reduced_per_row():
    adv = torch.tensor([[0.5, 0.5, 0.0], [-0.5, -0.5, 0.0]])
    assert_zero_sum_advantage(adv, torch.tensor([0, 0]))


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        assert_zero_sum_advantage(torch.zeros(2, 2, 2), torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError):
        assert_zero_sum_advantage(torch.zeros(3), torch.zeros(2, dtype=torch.long))
