"""Tests for packed-microbatch routing (audit F10).

The failure being guarded is not a crash: a packed tensor reshaped to (1, total) reads the
whole microbatch as ONE group and marks every token RL-dead, which looks like a working
router routing everything.
"""
from __future__ import annotations

import pytest
import torch

from selfevo.integration.packed import PackedLayoutError, repack, unpack
from selfevo.integration.token_routing import TokenRoutingSpec, route_token_weights


def _padded_case():
    """Two groups of two, each sharing a 2-token prefix, all length 4."""
    tokens = torch.tensor([[7, 8, 1, 2],
                           [7, 8, 3, 4],
                           [5, 6, 9, 1],
                           [5, 6, 9, 2]])
    gen = torch.ones_like(tokens, dtype=torch.bool)
    gids = torch.tensor([0, 0, 1, 1])
    lm = torch.ones_like(tokens, dtype=torch.bool)
    adv = torch.tensor([[1.0] * 4, [-1.0] * 4, [1.0] * 4, [-1.0] * 4])
    return tokens, gen, gids, lm, adv


def test_packed_matches_the_equivalent_padded_batch():
    """The layout must not change the decision -- only its shape."""
    tokens, gen, gids, lm, adv = _padded_case()
    cu = torch.tensor([0, 4, 8, 12, 16])
    spec = TokenRoutingSpec(enabled=True)
    pad = route_token_weights(spec=spec, rl_loss_weight=1.0, distill_loss_weight=0.005,
                              loss_mask=lm, tokens=tokens, gen_mask=gen, group_ids=gids,
                              advantages=adv)
    pk = route_token_weights(spec=spec, rl_loss_weight=1.0, distill_loss_weight=0.005,
                             loss_mask=repack(lm, cu), tokens=repack(tokens, cu),
                             gen_mask=repack(gen, cu), group_ids=gids,
                             advantages=repack(adv, cu), cu_seqlens=cu)
    assert pk.n_routed == pad.n_routed
    assert torch.equal(pk.rl_weight, repack(pad.rl_weight, cu)), "packed decision differs"


def test_packed_output_keeps_the_packed_layout():
    """A padded weight tensor cannot multiply a packed advantage."""
    tokens, gen, gids, lm, adv = _padded_case()
    cu = torch.tensor([0, 4, 8, 12, 16])
    r = route_token_weights(spec=TokenRoutingSpec(enabled=True), rl_loss_weight=1.0,
                            distill_loss_weight=0.005, loss_mask=repack(lm, cu),
                            tokens=repack(tokens, cu), gen_mask=repack(gen, cu),
                            group_ids=gids, advantages=repack(adv, cu), cu_seqlens=cu)
    assert r.rl_weight.ndim == 1 and r.rl_weight.numel() == 16


def test_a_packed_microbatch_is_not_read_as_one_group():
    """The exact F10 failure: routing everything while looking like it worked."""
    tokens, gen, gids, lm, adv = _padded_case()
    cu = torch.tensor([0, 4, 8, 12, 16])
    r = route_token_weights(spec=TokenRoutingSpec(enabled=True), rl_loss_weight=1.0,
                            distill_loss_weight=0.005, loss_mask=repack(lm, cu),
                            tokens=repack(tokens, cu), gen_mask=repack(gen, cu),
                            group_ids=gids, advantages=repack(adv, cu), cu_seqlens=cu)
    assert r.n_routed < 16, "every token routed: the microbatch was read as a single group"
    # Group 0 rows [7,8,1,2] / [7,8,3,4] share 2 tokens -> 4 routed.
    # Group 1 rows [5,6,9,1] / [5,6,9,2] share 3 (9 as well as 5,6) -> 6 routed.
    # 10, not 8: the first version of this assertion mis-counted group 1.
    assert r.n_routed == 10, f"expected 4 + 6 routed, got {r.n_routed}"
    dead = (r.rl_weight == 0.0)
    expected = torch.tensor([True, True, False, False,
                             True, True, False, False,
                             True, True, True, False,
                             True, True, True, False])
    assert torch.equal(dead, expected), f"wrong packed positions: {dead.int().tolist()}"


def test_one_d_without_cu_seqlens_is_refused():
    """Boundaries cannot be recovered, and guessing marks everything dead."""
    tokens, gen, gids, lm, adv = _padded_case()
    cu = torch.tensor([0, 4, 8, 12, 16])
    with pytest.raises(ValueError, match="cu_seqlens"):
        route_token_weights(spec=TokenRoutingSpec(enabled=True), rl_loss_weight=1.0,
                            distill_loss_weight=0.005, loss_mask=repack(lm, cu),
                            tokens=repack(tokens, cu), gen_mask=repack(gen, cu),
                            group_ids=gids, advantages=repack(adv, cu))


def test_unequal_sequence_lengths_round_trip():
    x = torch.tensor([1, 2, 3, 4, 5, 6])
    cu = torch.tensor([0, 1, 4, 6])
    u = unpack(x, cu)
    assert u.shape == (3, 3)
    assert torch.equal(repack(u, cu), x)


def test_inconsistent_cu_seqlens_is_refused_not_silently_truncated():
    with pytest.raises(PackedLayoutError, match="silently drop or invent"):
        unpack(torch.arange(10), torch.tensor([0, 3, 5, 9]))
    with pytest.raises(PackedLayoutError):
        unpack(torch.arange(9), torch.tensor([1, 3, 5, 9]))     # does not start at 0
    with pytest.raises(PackedLayoutError):
        unpack(torch.arange(9), torch.tensor([0, 5, 3, 9]))     # not non-decreasing
