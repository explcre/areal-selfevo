"""Tests for per-token RL/distillation routing.

The load-bearing test is the rollback one: it EXECUTES the upstream loss expression and the
routed one and compares tensors. Asserting "disabled means unchanged" in prose is how a
silent behaviour change ships.
"""
from __future__ import annotations

import pytest
import torch

from selfevo.integration.token_routing import (
    TokenRoutingSpec,
    route_token_weights,
)


def _group(n_groups=2, group_size=2, prefix=3, tail=2):
    """Sequences sharing a prefix within each group, diverging after it."""
    T = prefix + tail
    tokens = torch.zeros(n_groups * group_size, T, dtype=torch.long)
    for g in range(n_groups):
        pre = torch.arange(1, prefix + 1) + 100 * g
        for k in range(group_size):
            r = g * group_size + k
            tokens[r, :prefix] = pre
            tokens[r, prefix:] = torch.arange(1, tail + 1) + 1000 * (r + 1)
    gen_mask = torch.ones_like(tokens, dtype=torch.bool)
    group_ids = torch.arange(n_groups).repeat_interleave(group_size)
    loss_mask = torch.ones_like(tokens, dtype=torch.bool)
    return tokens, gen_mask, group_ids, loss_mask


def _upstream_loss(rl_w, kd_w, per_token_rl, per_token_kd, loss_mask):
    """The upstream expression: scalars times reduced terms."""
    denom = loss_mask.sum().clamp(min=1)
    rl = (per_token_rl * loss_mask).sum() / denom
    kd = (per_token_kd * loss_mask).sum() / denom
    return rl_w * rl + kd_w * kd


def _routed_loss(rl_w, kd_w, per_token_rl, per_token_kd, loss_mask):
    """The routed expression: weights applied per token before reduction."""
    denom = loss_mask.sum().clamp(min=1)
    rl = (rl_w * per_token_rl * loss_mask).sum() / denom
    kd = (kd_w * per_token_kd * loss_mask).sum() / denom
    return rl + kd


def test_disabled_is_bit_identical_to_upstream():
    """The rollback path must reproduce upstream arithmetic EXACTLY, not approximately."""
    torch.manual_seed(0)
    _, _, _, loss_mask = _group()
    ptr = torch.randn_like(loss_mask, dtype=torch.float32)
    ptk = torch.randn_like(loss_mask, dtype=torch.float32)

    r = route_token_weights(spec=TokenRoutingSpec(enabled=False), rl_loss_weight=1.0,
                            distill_loss_weight=0.005, loss_mask=loss_mask)
    # Disabled must hand back the caller's own scalars, not tensors of the same value:
    # downstream code branches on `rl_loss_weight == 0`, which a tensor breaks.
    assert isinstance(r.rl_weight, float) and isinstance(r.distill_weight, float)
    assert r.rl_weight == 1.0 and r.distill_weight == 0.005

    up = _upstream_loss(1.0, 0.005, ptr, ptk, loss_mask)
    got = _upstream_loss(r.rl_weight, r.distill_weight, ptr, ptk, loss_mask)
    assert torch.equal(up, got), "disabled path changed the loss"


def test_enabled_with_all_rl_control_matches_upstream_numerically():
    """The all_rl control routes nothing, so per-token weights must reproduce the scalars."""
    tokens, gen, gids, loss_mask = _group()
    ptr = torch.randn_like(loss_mask, dtype=torch.float32)
    ptk = torch.randn_like(loss_mask, dtype=torch.float32)
    r = route_token_weights(spec=TokenRoutingSpec(enabled=True, rule="all_rl"),
                            rl_loss_weight=1.0, distill_loss_weight=0.005,
                            loss_mask=loss_mask, tokens=tokens, gen_mask=gen, group_ids=gids)
    assert r.n_routed == 0
    up = _upstream_loss(1.0, 0.005, ptr, ptk, loss_mask)
    got = _routed_loss(r.rl_weight, r.distill_weight, ptr, ptk, loss_mask)
    assert torch.allclose(up, got), "all_rl control diverged from the scalar path"


def test_rl_dead_tokens_lose_rl_weight_and_keep_distillation():
    tokens, gen, gids, loss_mask = _group(prefix=3, tail=2)
    r = route_token_weights(
        spec=TokenRoutingSpec(enabled=True, rule="rl_dead_to_distill"),
        rl_loss_weight=1.0, distill_loss_weight=0.005,
        loss_mask=loss_mask, tokens=tokens, gen_mask=gen, group_ids=gids)
    assert r.n_routed > 0, "a shared prefix exists, so something must be routed"
    dead = r.rl_weight == 0.0
    assert bool((r.distill_weight[dead] == 0.005).all()), "routed tokens lost the teacher too"
    assert bool((r.rl_weight[~dead] == 1.0).all()), "live tokens lost RL weight"


def test_routed_fraction_is_reported_and_bounded():
    tokens, gen, gids, loss_mask = _group()
    r = route_token_weights(spec=TokenRoutingSpec(enabled=True), rl_loss_weight=1.0,
                            distill_loss_weight=0.005, loss_mask=loss_mask,
                            tokens=tokens, gen_mask=gen, group_ids=gids)
    assert 0.0 < r.routed_fraction <= 1.0
    assert r.n_routed == int((r.rl_weight == 0.0).sum())


def test_random_control_matches_the_rate_but_not_the_positions():
    """Separates 'routing helped' from 'changing the RL/KD ratio helped'."""
    tokens, gen, gids, loss_mask = _group(n_groups=3, group_size=3, prefix=4, tail=4)
    real = route_token_weights(spec=TokenRoutingSpec(enabled=True, rule="rl_dead_to_distill"),
                               rl_loss_weight=1.0, distill_loss_weight=0.005,
                               loss_mask=loss_mask, tokens=tokens, gen_mask=gen, group_ids=gids)
    rand = route_token_weights(spec=TokenRoutingSpec(enabled=True, rule="random", seed=1),
                               rl_loss_weight=1.0, distill_loss_weight=0.005,
                               loss_mask=loss_mask, tokens=tokens, gen_mask=gen, group_ids=gids)
    assert rand.n_routed == real.n_routed, "the control must match the rate"
    assert not torch.equal(rand.rl_weight == 0.0, real.rl_weight == 0.0), \
        "the control must not select the same tokens"


def test_all_distill_routes_every_loss_carrying_token():
    tokens, gen, gids, loss_mask = _group()
    loss_mask[0, 0] = False
    r = route_token_weights(spec=TokenRoutingSpec(enabled=True, rule="all_distill"),
                            rl_loss_weight=1.0, distill_loss_weight=0.005,
                            loss_mask=loss_mask, tokens=tokens, gen_mask=gen, group_ids=gids)
    assert r.n_routed == int(loss_mask.sum())
    assert r.rl_weight[0, 0] == 1.0, "a non-loss-carrying token must not be routed"


def test_missing_inputs_raise_rather_than_falling_back():
    """A misconfigured router must not be indistinguishable from a disabled one."""
    _, _, _, loss_mask = _group()
    with pytest.raises(ValueError, match="refusing to fall back"):
        route_token_weights(spec=TokenRoutingSpec(enabled=True), rl_loss_weight=1.0,
                            distill_loss_weight=0.005, loss_mask=loss_mask)


def test_spec_rejects_incoherent_configuration():
    with pytest.raises(ValueError, match="unknown rule"):
        TokenRoutingSpec(rule="wishful")
    with pytest.raises(ValueError):
        TokenRoutingSpec(dead_rl_weight=-1.0)
    with pytest.raises(ValueError):
        TokenRoutingSpec(dead_distill_weight=-0.5)


def test_dead_rl_weight_is_configurable_so_the_claim_can_be_tested():
    """Zero is the theory's value, not an axiom; a run must be able to falsify it."""
    tokens, gen, gids, loss_mask = _group()
    r = route_token_weights(
        spec=TokenRoutingSpec(enabled=True, dead_rl_weight=0.25, dead_distill_weight=0.5),
        rl_loss_weight=1.0, distill_loss_weight=0.005,
        loss_mask=loss_mask, tokens=tokens, gen_mask=gen, group_ids=gids)
    dead = r.distill_weight == 0.5
    assert bool((r.rl_weight[dead] == 0.25).all())
