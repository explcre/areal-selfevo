"""Regression for the column-vs-rank coordinate mismatch (audit F9)."""
from __future__ import annotations

import torch

from selfevo.routing.token_level import rl_dead_mask, shared_prefix_lengths


def _misaligned():
    """Members that start generating at DIFFERENT columns, sharing tokens 11,12,13.

    row 0: cols 2-4 generated -> [11, 12, 13]
    row 1: cols 1-4 generated -> [99, 11, 12, 13]

    In rank space the two disagree at their FIRST generated token (11 vs 99), so the shared
    prefix is 0. The old column-based count reported 3 and then indexed it by rank, routing
    row 1's unique token 99 to distillation while leaving shared token 13 on RL.
    """
    tokens = torch.tensor([[0, 0, 11, 12, 13],
                           [0, 99, 11, 12, 13]])
    gen = torch.tensor([[False, False, True, True, True],
                        [False, True, True, True, True]])
    gids = torch.tensor([0, 0])
    return tokens, gen, gids


def test_prefix_is_measured_in_generated_rank_not_column():
    tokens, gen, gids = _misaligned()
    n = shared_prefix_lengths(tokens, gen, gids)
    assert int(n[0]) == 0, (
        f"expected 0 (the members' FIRST generated tokens are 11 and 99), got {int(n[0])}; "
        "a nonzero value means the count is still being taken over columns"
    )


def test_no_unique_token_is_ever_routed():
    """The invariant that actually matters: a routed token must be shared by every member."""
    tokens, gen, gids = _misaligned()
    dead = rl_dead_mask(tokens, gen, gids)
    for r in range(tokens.shape[0]):
        for c in range(tokens.shape[1]):
            if bool(dead[r, c]):
                tok = int(tokens[r, c])
                others = [int(tokens[o][gen[o]].tolist()[0]) for o in range(tokens.shape[0])]
                assert tok in others or True  # placeholder replaced below
    # Stronger and exact: with a 0-length shared prefix nothing may be routed at all.
    assert int(dead.sum()) == 0, f"routed {int(dead.sum())} tokens despite no shared prefix"


def test_aligned_case_still_routes_the_real_prefix():
    """The fix must not break the ordinary case where members align."""
    tokens = torch.tensor([[5, 5, 7, 1],
                           [5, 5, 7, 2]])
    gen = torch.ones_like(tokens, dtype=torch.bool)
    gids = torch.tensor([0, 0])
    assert int(shared_prefix_lengths(tokens, gen, gids)[0]) == 3
    dead = rl_dead_mask(tokens, gen, gids)
    expected = torch.tensor([[True, True, True, False],
                             [True, True, True, False]])
    assert torch.equal(dead, expected)


def test_shorter_member_bounds_the_prefix():
    """The prefix cannot exceed the shortest member's generated length."""
    tokens = torch.tensor([[4, 4, 4, 0],
                           [4, 4, 0, 0]])
    gen = torch.tensor([[True, True, True, False],
                        [True, True, False, False]])
    gids = torch.tensor([0, 0])
    assert int(shared_prefix_lengths(tokens, gen, gids)[0]) == 2
