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


def _valid_adv(group_ids, shape):
    """Advantages summing to zero per group, so the precondition guard passes.

    The guard is not ceremony: batch adv_norm and kl_ctl>0 -- both live in this repo's
    configs -- were measured to break this sum by 87-115% of mean |A|, leaving 9.7% of the
    live gradient at a supposedly dead prefix.
    """
    import torch as _t
    adv = _t.zeros(shape, dtype=_t.float32)
    for g in group_ids.unique():
        rows = (group_ids == g).nonzero().flatten()
        for i, r in enumerate(rows):
            adv[r] = 1.0 if i % 2 == 0 else -1.0
        if len(rows) % 2:
            adv[rows[-1]] = 0.0
    return adv

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
                            loss_mask=loss_mask, tokens=tokens, gen_mask=gen, group_ids=gids,
        advantages=_valid_adv(gids, tokens.shape))
    assert r.n_routed == 0
    up = _upstream_loss(1.0, 0.005, ptr, ptk, loss_mask)
    got = _routed_loss(r.rl_weight, r.distill_weight, ptr, ptk, loss_mask)
    assert torch.allclose(up, got), "all_rl control diverged from the scalar path"


def test_rl_dead_tokens_lose_rl_weight_and_keep_distillation():
    tokens, gen, gids, loss_mask = _group(prefix=3, tail=2)
    r = route_token_weights(
        spec=TokenRoutingSpec(enabled=True, rule="rl_dead_to_distill"),
        rl_loss_weight=1.0, distill_loss_weight=0.005,
        loss_mask=loss_mask, tokens=tokens, gen_mask=gen, group_ids=gids,
        advantages=_valid_adv(gids, tokens.shape))
    assert r.n_routed > 0, "a shared prefix exists, so something must be routed"
    dead = r.rl_weight == 0.0
    # Reversed after the audit: this used to assert the routed tokens KEEP the baseline
    # 0.005, which is precisely the no-op it should have caught -- the default only deleted
    # RL and never gave the teacher anything. A routed token must gain teacher weight.
    assert bool((r.distill_weight[dead] > 0.005).all()), "routed tokens gained no teacher weight"
    assert bool((r.rl_weight[~dead] == 1.0).all()), "live tokens lost RL weight"


def test_routed_fraction_is_reported_and_bounded():
    tokens, gen, gids, loss_mask = _group()
    r = route_token_weights(spec=TokenRoutingSpec(enabled=True), rl_loss_weight=1.0,
                            distill_loss_weight=0.005, loss_mask=loss_mask,
                            tokens=tokens, gen_mask=gen, group_ids=gids,
                            advantages=_valid_adv(gids, tokens.shape))
    assert 0.0 < r.routed_fraction <= 1.0
    assert r.n_routed == int((r.rl_weight == 0.0).sum())


def test_random_control_matches_the_rate_but_not_the_positions():
    """Separates 'routing helped' from 'changing the RL/KD ratio helped'."""
    tokens, gen, gids, loss_mask = _group(n_groups=3, group_size=3, prefix=4, tail=4)
    real = route_token_weights(spec=TokenRoutingSpec(enabled=True, rule="rl_dead_to_distill"),
                               rl_loss_weight=1.0, distill_loss_weight=0.005,
                               loss_mask=loss_mask, tokens=tokens, gen_mask=gen, group_ids=gids,
        advantages=_valid_adv(gids, tokens.shape))
    rand = route_token_weights(spec=TokenRoutingSpec(enabled=True, rule="random", seed=1),
                               rl_loss_weight=1.0, distill_loss_weight=0.005,
                               loss_mask=loss_mask, tokens=tokens, gen_mask=gen, group_ids=gids,
        advantages=_valid_adv(gids, tokens.shape))
    assert rand.n_routed == real.n_routed, "the control must match the rate"
    assert not torch.equal(rand.rl_weight == 0.0, real.rl_weight == 0.0), \
        "the control must not select the same tokens"


def test_all_distill_routes_every_loss_carrying_token():
    tokens, gen, gids, loss_mask = _group()
    loss_mask[0, 0] = False
    r = route_token_weights(spec=TokenRoutingSpec(enabled=True, rule="all_distill"),
                            rl_loss_weight=1.0, distill_loss_weight=0.005,
                            loss_mask=loss_mask, tokens=tokens, gen_mask=gen, group_ids=gids,
        advantages=_valid_adv(gids, tokens.shape))
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
        loss_mask=loss_mask, tokens=tokens, gen_mask=gen, group_ids=gids,
        advantages=_valid_adv(gids, tokens.shape))
    dead = r.distill_weight == 0.5
    assert bool((r.rl_weight[dead] == 0.25).all())


# ---------------------------------------------------------------- audit-driven regressions
#
# An audit mutation-tested this module and 13 of 24 mutants survived. Inverting the dead
# mask -- the module's entire semantic content -- passed all nine tests, because the original
# assertion only checked that the routed and unrouted sets were internally consistent, which
# is symmetric under inversion. These tests pin WHICH tokens are routed, by position.




def test_exactly_the_shared_prefix_is_routed_by_position():
    """Kills the inverted-mask mutant: asserts WHICH positions are dead, not just consistency."""
    tokens, gen, gids, loss_mask = _group(n_groups=1, group_size=2, prefix=3, tail=2)
    r = route_token_weights(
        spec=TokenRoutingSpec(enabled=True, rule="rl_dead_to_distill"),
        rl_loss_weight=1.0, distill_loss_weight=0.005, loss_mask=loss_mask,
        tokens=tokens, gen_mask=gen, group_ids=gids,
        advantages=_valid_adv(gids, tokens.shape))
    dead = (r.rl_weight == 0.0)
    expected = torch.zeros_like(dead)
    expected[:, :3] = True          # the shared prefix, and nothing else
    assert torch.equal(dead, expected), f"routed the wrong positions:\n{dead.int()}"


def test_an_off_by_one_in_the_prefix_boundary_is_caught():
    """Kills the gen_rank +-1 and prefix-length mutants: the boundary column is exact."""
    tokens, gen, gids, loss_mask = _group(n_groups=1, group_size=2, prefix=4, tail=3)
    r = route_token_weights(
        spec=TokenRoutingSpec(enabled=True), rl_loss_weight=1.0, distill_loss_weight=0.005,
        loss_mask=loss_mask, tokens=tokens, gen_mask=gen, group_ids=gids,
        advantages=_valid_adv(gids, tokens.shape))
    dead = (r.rl_weight == 0.0)
    assert bool(dead[:, 3].all()), "last shared column must be routed"
    assert not bool(dead[:, 4].any()), "first diverging column must NOT be routed"
    assert int(dead.sum()) == 2 * 4, f"expected exactly 8 routed, got {int(dead.sum())}"


def test_routing_actually_changes_which_tokens_carry_rl_weight():
    """Kills 'routing is a no-op': the routed and all_rl masks must differ."""
    tokens, gen, gids, loss_mask = _group()
    adv = _valid_adv(gids, tokens.shape)
    real = route_token_weights(spec=TokenRoutingSpec(enabled=True), rl_loss_weight=1.0,
                               distill_loss_weight=0.005, loss_mask=loss_mask, tokens=tokens,
                               gen_mask=gen, group_ids=gids, advantages=adv)
    none = route_token_weights(spec=TokenRoutingSpec(enabled=True, rule="all_rl"),
                               rl_loss_weight=1.0, distill_loss_weight=0.005,
                               loss_mask=loss_mask, tokens=tokens, gen_mask=gen,
                               group_ids=gids, advantages=adv)
    assert not torch.equal(real.rl_weight, none.rl_weight)


def test_rl_loss_weight_magnitude_survives_routing():
    """Kills the dropped-rl_loss_weight mutant: live tokens keep the caller's magnitude."""
    tokens, gen, gids, loss_mask = _group()
    for w in (0.5, 1.0, 2.0):
        r = route_token_weights(spec=TokenRoutingSpec(enabled=True), rl_loss_weight=w,
                                distill_loss_weight=0.005, loss_mask=loss_mask,
                                tokens=tokens, gen_mask=gen, group_ids=gids,
                                advantages=_valid_adv(gids, tokens.shape))
        live = r.rl_weight[r.rl_weight != 0.0]
        assert bool((live == w).all()), f"live tokens lost magnitude at rl_loss_weight={w}"


def test_dead_rl_weight_is_honoured_at_every_value_not_just_zero():
    """Kills the inert-knob mutant: a nonzero dead weight must appear in the tensor."""
    tokens, gen, gids, loss_mask = _group()
    r = route_token_weights(spec=TokenRoutingSpec(enabled=True, dead_rl_weight=0.25),
                            rl_loss_weight=1.0, distill_loss_weight=0.005,
                            loss_mask=loss_mask, tokens=tokens, gen_mask=gen,
                            group_ids=gids, advantages=_valid_adv(gids, tokens.shape))
    assert float(r.rl_weight.min()) == 0.25, "dead_rl_weight was thresholded away"


def test_routed_tokens_gain_teacher_weight_by_default():
    """Kills the no-op-default mutant: the default must route TO distillation, not just off RL."""
    tokens, gen, gids, loss_mask = _group()
    r = route_token_weights(spec=TokenRoutingSpec(enabled=True), rl_loss_weight=1.0,
                            distill_loss_weight=0.005, loss_mask=loss_mask, tokens=tokens,
                            gen_mask=gen, group_ids=gids,
                            advantages=_valid_adv(gids, tokens.shape))
    dead = r.rl_weight == 0.0
    assert float(r.distill_weight[dead].min()) > 0.005, \
        "routed tokens kept the baseline teacher weight: the default is a no-op"
    assert bool((r.distill_weight[~dead] == 0.005).all()), "live tokens gained teacher weight"


def test_violated_zero_sum_precondition_is_refused():
    """The guard exists in the repo and had no callers; that is how the violation shipped."""
    tokens, gen, gids, loss_mask = _group()
    bad = torch.full(tokens.shape, 0.5)      # every advantage positive: sum != 0
    with pytest.raises(Exception):
        route_token_weights(spec=TokenRoutingSpec(enabled=True), rl_loss_weight=1.0,
                            distill_loss_weight=0.005, loss_mask=loss_mask, tokens=tokens,
                            gen_mask=gen, group_ids=gids, advantages=bad)


def test_precondition_check_can_be_waived_only_explicitly():
    """Measuring the rule outside its domain must be a deliberate, visible choice."""
    tokens, gen, gids, loss_mask = _group()
    bad = torch.full(tokens.shape, 0.5)
    r = route_token_weights(
        spec=TokenRoutingSpec(enabled=True, require_valid_preconditions=False),
        rl_loss_weight=1.0, distill_loss_weight=0.005, loss_mask=loss_mask,
        tokens=tokens, gen_mask=gen, group_ids=gids, advantages=bad)
    assert "NOT CHECKED" in r.basis, "waiving the check must be visible in the basis"
