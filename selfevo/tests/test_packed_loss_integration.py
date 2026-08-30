"""End-to-end routing on the tensor shapes grpo_loss_fn actually receives.

step0j died before step 1 because the call site omitted cu_seqlens and advantages. Every
unit test passed: they exercised route_token_weights directly with padded (B, T) tensors,
which is not what the trainer hands it. This test builds a PACKED microbatch the way the
FSDP engine does and drives the real loss function.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch

REPO = pathlib.Path("/home/ubuntu/areal-selfevo")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from areal.trainer.ppo.actor import grpo_loss_fn  # noqa: E402
from selfevo.integration.packed import repack  # noqa: E402
from selfevo.integration.token_routing import TokenRoutingSpec  # noqa: E402


def _packed_batch():
    """Two groups of two, packed 1-D with cu_seqlens, as the engine delivers."""
    tokens = torch.tensor([[11, 12, 31, 41],
                           [11, 12, 32, 42],
                           [21, 22, 23, 51],
                           [21, 22, 23, 52]])
    lm = torch.ones_like(tokens, dtype=torch.bool)
    adv = torch.tensor([[1.0] * 4, [-1.0] * 4, [1.0] * 4, [-1.0] * 4])
    cu = torch.tensor([0, 4, 8, 12, 16])
    n = 16
    data = {
        "logprobs": torch.randn(n),
        "advantages": repack(adv, cu),
        "loss_mask": repack(lm, cu).long(),
        "prox_logp": torch.randn(n),
        "input_ids": repack(tokens, cu),
        "gen_mask": repack(lm, cu),
        # PER TOKEN, exactly as actor.py now writes it: a (B,) tensor does not survive
        # microbatch splitting, which is how the first routed run died.
        "group_ids": repack(
            torch.tensor([0, 0, 1, 1]).unsqueeze(1).expand(4, 4).contiguous(), cu),
        "cu_seqlens": cu,
    }
    return torch.randn(n), torch.rand(n), data


def test_routing_runs_on_a_packed_microbatch():
    """The failure that killed step0j: the call site must pass cu_seqlens and advantages."""
    lp, ent, data = _packed_batch()
    loss = grpo_loss_fn(
        logprobs=lp, entropy=ent, input_data=data,
        eps_clip=0.2, eps_clip_higher=None, c_clip=None,
        token_routing=TokenRoutingSpec(enabled=True, rule="rl_dead_to_distill"),
    )
    assert torch.isfinite(loss), "routed loss is not finite on a packed batch"


def test_routing_changes_the_loss_on_a_packed_batch():
    """If routing runs but changes nothing, the run would look successful and mean nothing."""
    lp, ent, data = _packed_batch()
    torch.manual_seed(0)
    off = grpo_loss_fn(logprobs=lp, entropy=ent, input_data=dict(data),
                       eps_clip=0.2, eps_clip_higher=None, c_clip=None)
    on = grpo_loss_fn(logprobs=lp, entropy=ent, input_data=dict(data),
                      eps_clip=0.2, eps_clip_higher=None, c_clip=None,
                      token_routing=TokenRoutingSpec(enabled=True, rule="rl_dead_to_distill"))
    assert not torch.equal(off, on), "routing had no effect on the loss"


def test_all_rl_control_reproduces_the_unrouted_loss_on_a_packed_batch():
    """The control routes nothing, so it must match exactly even with the hook active."""
    lp, ent, data = _packed_batch()
    off = grpo_loss_fn(logprobs=lp, entropy=ent, input_data=dict(data),
                       eps_clip=0.2, eps_clip_higher=None, c_clip=None)
    ctrl = grpo_loss_fn(logprobs=lp, entropy=ent, input_data=dict(data),
                        eps_clip=0.2, eps_clip_higher=None, c_clip=None,
                        token_routing=TokenRoutingSpec(enabled=True, rule="all_rl"))
    assert torch.allclose(off, ctrl), f"all_rl drifted: {off.item()} vs {ctrl.item()}"


def test_missing_cu_seqlens_still_refuses_rather_than_guessing():
    """The guard that caught step0j must stay armed."""
    lp, ent, data = _packed_batch()
    data.pop("cu_seqlens")
    with pytest.raises(ValueError, match="cu_seqlens"):
        grpo_loss_fn(logprobs=lp, entropy=ent, input_data=data,
                     eps_clip=0.2, eps_clip_higher=None, c_clip=None,
                     token_routing=TokenRoutingSpec(enabled=True))


def test_a_corrupted_per_token_group_tensor_is_refused():
    """If splitting ever mixes groups within a sequence, collapsing would invent a grouping."""
    lp, ent, data = _packed_batch()
    g = data["group_ids"].clone()
    g[1] = 9                      # one token of sequence 0 now claims a different group
    data["group_ids"] = g
    with pytest.raises(ValueError, match="more than one group id"):
        grpo_loss_fn(logprobs=lp, entropy=ent, input_data=data,
                     eps_clip=0.2, eps_clip_higher=None, c_clip=None,
                     token_routing=TokenRoutingSpec(enabled=True))
