"""The group_ids derivation, and that routing can consume what the plumbing emits.

The plumbing sits in _compute_advantages because that is the only place traj_group_sizes is
still in scope. These tests pin the derivation itself and then feed its output through
route_token_weights, so "the key exists" is not mistaken for "the value is usable".
"""
from __future__ import annotations

import pytest
import torch

from selfevo.integration.token_routing import TokenRoutingSpec, route_token_weights


def derive(sizes, bs):
    """The expression used in actor.py, isolated so it can be pinned."""
    # An int is a UNIFORM group size and expands over the number of GROUPS, not the rows.
    # Expanding over rows gives sum = g * bs and silently yields no grouping at all.
    if isinstance(sizes, int):
        sizes = [sizes] * (bs // sizes) if sizes > 0 else []
    else:
        sizes = list(sizes)
    if sum(sizes) != bs:
        return None
    return torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))


def test_group_ids_partition_the_batch_in_row_order():
    g = derive([4, 4], 8)
    assert g.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_uneven_groups_are_handled():
    """Filtered or failed rollouts give groups of differing size."""
    g = derive([3, 1, 4], 8)
    assert g.tolist() == [0, 0, 0, 1, 2, 2, 2, 2]


def test_int_group_size_is_expanded():
    assert derive(2, 6).tolist() == [0, 0, 1, 1, 2, 2]


def test_a_mismatched_total_yields_nothing_rather_than_a_wrong_grouping():
    """Emitting a wrong grouping is worse than emitting none.

    A shared prefix computed over unrelated sequences would route real RL signal away.
    """
    assert derive([4, 4], 9) is None
    assert derive([5], 8) is None


def test_routing_consumes_the_plumbed_values_end_to_end():
    """The point of the plumbing: the router must accept these tensors as produced."""
    bs, T = 4, 5
    sizes = [2, 2]
    gids = derive(sizes, bs)
    tokens = torch.tensor([[1, 2, 9, 0, 0],
                           [1, 2, 8, 0, 0],
                           [3, 4, 5, 7, 0],
                           [3, 4, 5, 6, 0]])
    loss_mask = torch.tensor([[1, 1, 1, 0, 0],
                              [1, 1, 1, 0, 0],
                              [1, 1, 1, 1, 0],
                              [1, 1, 1, 1, 0]], dtype=torch.long)
    gen_mask = loss_mask.bool()                       # exactly what actor.py writes
    adv = torch.tensor([[1.0] * T, [-1.0] * T, [1.0] * T, [-1.0] * T])
    r = route_token_weights(spec=TokenRoutingSpec(enabled=True), rl_loss_weight=1.0,
                            distill_loss_weight=0.005, loss_mask=gen_mask, tokens=tokens,
                            gen_mask=gen_mask, group_ids=gids, advantages=adv)
    # group 0 shares [1,2]; group 1 shares [3,4,5]
    assert r.n_routed == 2 * 2 + 2 * 3
    dead = (r.rl_weight == 0.0)
    assert bool(dead[0, :2].all()) and not bool(dead[0, 2])
    assert bool(dead[2, :3].all()) and not bool(dead[2, 3])


def test_group_ids_length_matches_the_batch():
    """A length mismatch would silently misalign every row's group."""
    for sizes, bs in ([[4, 4], 8], [[1, 7], 8], [[8], 8]):
        assert derive(sizes, bs).numel() == bs
