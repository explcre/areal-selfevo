"""Merging the experts: the merged adapter must BE the sum, applied to a real input.

At inference there is no behavioural label -- the cluster is read off the response, and there
is no response yet -- so the experts are summed into one adapter (LSPO, arXiv 2607.27787) and
the deployed model does no routing at all. That makes the merge the thing that actually ships,
and a merge that quietly differed from the sum would change every reported inference number
without changing anything a training metric can see.

The headline test therefore checks the merge where it matters: on a MODULE'S OUTPUT for a
real input, not only on the weight tensors. ``merged(x) - base(x)`` must equal
``(A(x) - base(x)) + (B(x) - base(x))``.

Both adapters have their ``B`` matrices randomised first. At LoRA's initialisation ``B = 0``,
so every adapter's delta is exactly zero and any merge whatsoever reproduces the sum -- a test
that skipped this step would pass on an implementation that returned an empty adapter.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from selfevo.cluster_lora.adapters import ClusterAdapterSet  # noqa: E402
from selfevo.cluster_lora.merge import (  # noqa: E402
    MergeInexact,
    adapter_delta,
    merge_sum,
    summed_delta,
)

NAMES = ("cluster_0", "cluster_1")


def two_experts(seed=0):
    """A model with two adapters whose deltas are both non-zero."""
    from peft import LoraConfig, TaskType
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(seed)
    cfg = AutoConfig.for_model(
        "qwen2", hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
        max_position_embeddings=32, tie_word_embeddings=False,
    )
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
        target_modules=["q_proj", "v_proj"], bias="none",
    )
    a = ClusterAdapterSet.build(AutoModelForCausalLM.from_config(cfg), NAMES, lora)
    # B is zero at init, so every delta would be zero and the merge would be untestable.
    with torch.no_grad():
        for i, name in enumerate(NAMES):
            for _k, p in a.parameters(name):
                p.normal_(0.0, 0.1 * (i + 1), generator=torch.Generator().manual_seed(seed + i))
    return a


def first_lora_module(model, name):
    """The first tuner layer carrying ``name``, used as the input-level probe."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    for mod_name, mod in model.named_modules():
        if isinstance(mod, BaseTunerLayer) and name in getattr(mod, "lora_A", {}):
            return mod_name, mod
    raise AssertionError(f"no module carries adapter {name!r}")


def module_output(model, module, name, x):
    """The module's output for ``x`` with exactly one adapter active."""
    model.set_adapter(name)
    with torch.no_grad():
        return module(x).clone()


# ------------------------------------------------------ the property that matters -------


def test_the_merged_adapter_applied_to_an_input_equals_the_sum_of_the_two_deltas():
    """The headline: merged(x) - base(x) == (A(x) - base(x)) + (B(x) - base(x)).

    Checked on a module's OUTPUT rather than on its weights, because the output is what
    inference produces and a scaling mistake in the merge shows up there even when the
    concatenated tensors look plausible.
    """
    a = two_experts()
    mod_name, module = first_lora_module(a.model, "cluster_0")
    x = torch.randn(3, module.in_features, generator=torch.Generator().manual_seed(7))

    with a.model.disable_adapter():
        with torch.no_grad():
            base = module(x).clone()
    d0 = module_output(a.model, module, "cluster_0", x) - base
    d1 = module_output(a.model, module, "cluster_1", x) - base

    merge_sum(a.model, NAMES, target="merged")
    dm = module_output(a.model, module, "merged", x) - base
    assert torch.allclose(dm, d0 + d1, atol=1e-5), (dm - (d0 + d1)).abs().max()
    # And the sum is not trivially zero, or the assertion above holds for any merge at all.
    assert d0.abs().max() > 1e-4 and d1.abs().max() > 1e-4


def test_the_merged_weight_delta_equals_the_summed_weight_delta_on_every_module():
    """The same claim at the weight level, over all target modules rather than one."""
    a = two_experts()
    ref = summed_delta(a.model, NAMES)
    merge_sum(a.model, NAMES, target="merged")
    got = adapter_delta(a.model, "merged")
    assert set(got) == set(ref)
    for k in ref:
        assert torch.allclose(got[k], ref[k], atol=1e-6), (k, (got[k] - ref[k]).abs().max())


def test_weights_are_honoured_so_a_weighted_merge_is_available():
    """The plain sum is the default; the weighted form is the knob LSPO ablates."""
    a = two_experts()
    ref = summed_delta(a.model, NAMES, weights=[0.25, 0.75])
    merge_sum(a.model, NAMES, target="w", weights=[0.25, 0.75])
    got = adapter_delta(a.model, "w")
    for k in ref:
        assert torch.allclose(got[k], ref[k], atol=1e-6)
    # And it is genuinely different from the unweighted merge.
    plain = summed_delta(a.model, NAMES)
    assert not torch.allclose(got[next(iter(ref))], plain[next(iter(ref))], atol=1e-6)


def test_merging_a_single_expert_reproduces_that_expert():
    """The degenerate case, which is what the 'none' arm merges."""
    a = two_experts()
    ref = adapter_delta(a.model, "cluster_0")
    merge_sum(a.model, ["cluster_0"], target="solo")
    got = adapter_delta(a.model, "solo")
    for k in ref:
        assert torch.allclose(got[k], ref[k], atol=1e-6)


def test_merging_twice_does_not_stack_on_the_first_merge():
    """A second merge later in a run must replace, not accumulate.

    Accumulating would double the deltas silently and every later evaluation would be of a
    model no arm produced.
    """
    a = two_experts()
    merge_sum(a.model, NAMES, target="merged")
    first = {k: v.clone() for k, v in adapter_delta(a.model, "merged").items()}
    merge_sum(a.model, NAMES, target="merged")
    second = adapter_delta(a.model, "merged")
    for k in first:
        assert torch.allclose(first[k], second[k], atol=1e-6)


def test_the_shared_adapter_is_merged_like_any_other_expert():
    """Noise groups still trained something, and dropping it would discard those updates."""
    from peft import LoraConfig, TaskType
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(3)
    cfg = AutoConfig.for_model(
        "qwen2", hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
        max_position_embeddings=32, tie_word_embeddings=False,
    )
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
                      target_modules=["q_proj"], bias="none")
    names = ("cluster_0", "shared")
    a = ClusterAdapterSet.build(AutoModelForCausalLM.from_config(cfg), names, lora)
    with torch.no_grad():
        for _k, p in a.parameters("shared"):
            p.normal_(0.0, 0.1, generator=torch.Generator().manual_seed(1))
    ref = summed_delta(a.model, names)
    merge_sum(a.model, names, target="merged")
    got = adapter_delta(a.model, "merged")
    for k in ref:
        assert torch.allclose(got[k], ref[k], atol=1e-6)


# ------------------------------------------------------------------- the refusals -------


def test_an_adapter_on_no_module_is_refused_rather_than_contributing_nothing():
    """It would merge cleanly, change nothing, and report success."""
    a = two_experts()
    with pytest.raises(MergeInexact, match="present on no LoRA module"):
        adapter_delta(a.model, "not-an-adapter")


def test_merging_nothing_is_refused():
    a = two_experts()
    with pytest.raises(ValueError, match="nothing to merge"):
        summed_delta(a.model, [])


def test_a_weight_per_adapter_is_required():
    a = two_experts()
    with pytest.raises(ValueError, match="weights for"):
        summed_delta(a.model, NAMES, weights=[1.0])


def test_experts_covering_different_modules_are_refused():
    """A ragged merge applies some experts on some layers only, silently."""
    from peft import LoraConfig, TaskType
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = AutoConfig.for_model(
        "qwen2", hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=64,
        max_position_embeddings=32, tie_word_embeddings=False,
    )
    base = AutoModelForCausalLM.from_config(cfg)
    wide = LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
                      target_modules=["q_proj", "v_proj"], bias="none")
    narrow = LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
                        target_modules=["q_proj"], bias="none")
    a = ClusterAdapterSet.build(base, ("cluster_0",), wide)
    a.model.add_adapter("cluster_1", narrow)
    with pytest.raises(MergeInexact, match="ragged module set"):
        summed_delta(a.model, ("cluster_0", "cluster_1"))


def test_the_verification_fires_when_the_merge_is_actually_wrong(monkeypatch):
    """Otherwise the check is decoration: it passes because PEFT happens to be right.

    A silently disabled verification looks identical to a passing one in every other test
    here, since those compare against the reference themselves. This one breaks the merge on
    purpose and requires the refusal.
    """
    a = two_experts()
    real = type(a.model.base_model).add_weighted_adapter

    def wrong(self, adapters, weights, adapter_name, **kw):
        """Merge with the wrong weights, which is what a scaling bug looks like."""
        return real(self, adapters, [w * 2.0 for w in weights], adapter_name, **kw)

    monkeypatch.setattr(type(a.model.base_model), "add_weighted_adapter", wrong)
    with pytest.raises(MergeInexact, match="does not reproduce the sum"):
        merge_sum(a.model, NAMES, target="bad")
