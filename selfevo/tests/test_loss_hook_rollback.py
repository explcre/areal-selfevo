"""Proof that the routing hook is a no-op when disabled.

Loads the PRE-PATCH grpo_loss_fn from a saved copy of the file and the patched one from the
repo, runs both on identical inputs, and compares the resulting loss tensors with
torch.equal. Asserting "the default is unchanged" in a docstring is how a silent numerical
change ships; this executes both code paths.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch

REPO = pathlib.Path("/home/ubuntu/areal-selfevo")
# importlib cannot infer a loader from a .bak suffix, so the pre-patch copy is
# kept under a .py name. Verified to contain no rl_mask/token_routing reference.
# Committed to the repo, NOT /tmp. In /tmp it vanished on any fresh machine and every
# rollback test SKIPPED while the suite reported green -- a guarantee that silently
# stops being checked is worse than none. This was "fixed" once before and the edit
# never landed; the print said otherwise and I believed it instead of the file.
BASELINE = pathlib.Path(__file__).parent / "baselines" / "actor_upstream_baseline.py"


def _load(path, name):
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _inputs(B=4, T=12, seed=0, with_teacher=False):
    torch.manual_seed(seed)
    logprobs = torch.randn(B, T, requires_grad=False)
    data = {
        "logprobs": torch.randn(B, T),
        "advantages": torch.randn(B, T),
        "loss_mask": torch.ones(B, T, dtype=torch.long),
        # Required: the default prox_logp_method='recompute' refuses to run without it,
        # which is the harness's own guard against a skipped compute_logp().
        "prox_logp": torch.randn(B, T),
    }
    if with_teacher:
        data["teacher_logp"] = torch.randn(B, T)
    return logprobs, torch.rand(B, T), data


@pytest.fixture(scope="module")
def mods():
    if not BASELINE.exists():
        raise FileNotFoundError(
            f"rollback baseline missing at {BASELINE}. It is committed to the repo; "
            f"restore it from git rather than skipping the guarantee."
        )
    return _load(BASELINE, "actor_baseline"), _load(REPO / "areal/trainer/ppo/actor.py", "actor_patched")


@pytest.mark.parametrize("with_teacher", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_disabled_hook_reproduces_upstream_loss_exactly(mods, seed, with_teacher):
    base, patched = mods
    kw = dict(eps_clip=0.2, eps_clip_higher=None, c_clip=None)

    lp, ent, data = _inputs(seed=seed, with_teacher=with_teacher)
    loss_a = base.grpo_loss_fn(logprobs=lp, entropy=ent, input_data=data, **kw)

    lp2, ent2, data2 = _inputs(seed=seed, with_teacher=with_teacher)
    # token_routing omitted entirely: the rollback is "do not pass the argument".
    loss_b = patched.grpo_loss_fn(logprobs=lp2, entropy=ent2, input_data=data2, **kw)

    assert torch.equal(loss_a, loss_b), (
        f"patched loss differs from upstream with routing disabled "
        f"(seed={seed}, teacher={with_teacher}): {loss_a.item()} vs {loss_b.item()}"
    )


def test_explicit_none_matches_omitted_argument(mods):
    _, patched = mods
    kw = dict(eps_clip=0.2, eps_clip_higher=None, c_clip=None)
    lp, ent, data = _inputs(seed=7)
    a = patched.grpo_loss_fn(logprobs=lp, entropy=ent, input_data=data, **kw)
    lp2, ent2, data2 = _inputs(seed=7)
    b = patched.grpo_loss_fn(logprobs=lp2, entropy=ent2, input_data=data2,
                                token_routing=None, **kw)
    assert torch.equal(a, b)


def test_a_disabled_spec_object_is_also_a_no_op(mods):
    """Passing a spec with enabled=False must be identical to passing nothing."""
    base, patched = mods
    from selfevo.integration.token_routing import TokenRoutingSpec
    kw = dict(eps_clip=0.2, eps_clip_higher=None, c_clip=None)
    lp, ent, data = _inputs(seed=3)
    a = base.grpo_loss_fn(logprobs=lp, entropy=ent, input_data=data, **kw)
    lp2, ent2, data2 = _inputs(seed=3)
    b = patched.grpo_loss_fn(logprobs=lp2, entropy=ent2, input_data=data2,
                                token_routing=TokenRoutingSpec(enabled=False), **kw)
    assert torch.equal(a, b)


def test_enabled_control_all_rl_is_numerically_unchanged(mods):
    """all_rl routes nothing, so it must reproduce upstream even with the hook ACTIVE.

    This is the strongest available check that the hook's machinery -- mask rebuild, weight
    tensors, stats -- introduces no drift of its own.
    """
    base, patched = mods
    from selfevo.integration.token_routing import TokenRoutingSpec
    kw = dict(eps_clip=0.2, eps_clip_higher=None, c_clip=None)
    lp, ent, data = _inputs(seed=5)
    a = base.grpo_loss_fn(logprobs=lp, entropy=ent, input_data=data, **kw)
    lp2, ent2, data2 = _inputs(seed=5)
    b = patched.grpo_loss_fn(logprobs=lp2, entropy=ent2, input_data=data2,
                                token_routing=TokenRoutingSpec(enabled=True, rule="all_rl"),
                                **kw)
    assert torch.equal(a, b), f"active hook drifted with a no-op rule: {a.item()} vs {b.item()}"
