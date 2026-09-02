"""THE GUARD: a cluster's step must move its own adapter and no other, bit for bit.

Everything else in this method is bookkeeping around this one property. If it does not hold,
per-cluster LoRA is vanilla LoRA with more parameters -- and it would still train, still log
a loss per cluster, still merge, and still produce a benchmark number. There is no symptom.
So the assertion is bit-level equality of the other adapters' tensors before and after a real
optimizer step, not ``allclose``: a tolerance would pass an adapter that received a small
leaked update, which is exactly the defect being hunted.

The optimizer is ``AdamW`` with a NON-ZERO weight decay, and that is load-bearing. Decoupled
weight decay moves a parameter on every step it is stepped on, INCLUDING one whose gradient
is exactly zero. So an implementation that zeroed gradients into tensors instead of setting
them to ``None`` would decay every idle expert every step; with ``set_to_none=True`` those
parameters are skipped by the optimizer entirely and keep their exact values. Under a
zero-decay optimizer both implementations pass and the test would constrain nothing.

Two steps, not one, for the same reason: momentum and decay only diverge from the correct
behaviour once state exists.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from selfevo.cluster_lora.adapters import (  # noqa: E402
    AdapterIsolationError,
    ClusterAdapterSet,
    cluster_row_index,
)

NAMES = ("cluster_0", "cluster_1", "shared")


def tiny_peft(names=NAMES, seed=0):
    """A four-layer LM wrapped with one LoRA adapter per cluster, all from one config."""
    from peft import LoraConfig, TaskType
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(seed)
    cfg = AutoConfig.for_model(
        "qwen2", hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
        max_position_embeddings=64, tie_word_embeddings=False,
    )
    base = AutoModelForCausalLM.from_config(cfg)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8,
        target_modules=["q_proj", "v_proj"], bias="none",
    )
    return ClusterAdapterSet.build(base, names, lora)


def lm_loss(model, mb):
    """Plain next-token loss over a microbatch of ids. Enough to produce a real gradient."""
    logits = model(input_ids=mb).logits
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), mb[:, 1:].reshape(-1)
    )


def optimiser(adapters):
    """AdamW with weight decay ON, which is what makes the isolation check bite."""
    params = [p for n in adapters.names for _k, p in adapters.parameters(n)]
    return torch.optim.AdamW(params, lr=1e-2, weight_decay=0.01)


def batch(rows=2, seed=0):
    """A microbatch of token ids."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 128, (rows, 8), generator=g)


# ---------------------------------------------------------------------- the guard -------


def test_an_untouched_adapter_is_bit_identical_after_a_step_for_another_cluster():
    """The property the whole method rests on, asserted at the bit level.

    Two steps: decoupled weight decay and Adam momentum only diverge from correct behaviour
    once optimizer state exists, so a single step would pass on an implementation that
    leaks.
    """
    a = tiny_peft()
    opt = optimiser(a)
    before = {n: a.snapshot(n) for n in NAMES}
    for step in range(2):
        a.step({"cluster_1": batch(seed=step)}, lm_loss, opt)
    assert a.unchanged("cluster_0", before["cluster_0"]), "cluster_0 moved on cluster_1 steps"
    assert a.unchanged("shared", before["shared"]), "the shared adapter moved"


def test_the_adapter_that_did_receive_the_batch_actually_changed():
    """Without this, the test above passes on a step that does nothing at all."""
    a = tiny_peft()
    opt = optimiser(a)
    before = a.snapshot("cluster_1")
    a.step({"cluster_1": batch()}, lm_loss, opt)
    assert not a.unchanged("cluster_1", before)


def test_the_isolation_holds_in_the_other_direction_too():
    """Symmetry, so an implementation that happens to freeze one particular adapter fails."""
    a = tiny_peft()
    opt = optimiser(a)
    before = {n: a.snapshot(n) for n in NAMES}
    for step in range(2):
        a.step({"cluster_0": batch(seed=step)}, lm_loss, opt)
    assert a.unchanged("cluster_1", before["cluster_1"])
    assert a.unchanged("shared", before["shared"])
    assert not a.unchanged("cluster_0", before["cluster_0"])


def test_every_cluster_in_the_batch_is_updated_and_only_those():
    """The realistic step: several clusters present, one absent."""
    a = tiny_peft()
    opt = optimiser(a)
    before = {n: a.snapshot(n) for n in NAMES}
    rec = a.step({"cluster_0": batch(seed=1), "shared": batch(seed=2)}, lm_loss, opt)
    assert not a.unchanged("cluster_0", before["cluster_0"])
    assert not a.unchanged("shared", before["shared"])
    assert a.unchanged("cluster_1", before["cluster_1"])
    assert rec.skipped == ("cluster_1",)
    assert set(rec.losses) == {"cluster_0", "shared"}


def test_each_cluster_gets_its_own_gradient_norm_not_a_shared_one():
    """Two clusters given different data must report different gradient norms.

    Identical norms would be the signature of both microbatches flowing into one adapter.
    """
    a = tiny_peft()
    opt = optimiser(a)
    rec = a.step({"cluster_0": batch(rows=2, seed=1), "cluster_1": batch(rows=4, seed=99)},
                 lm_loss, opt)
    assert rec.grad_norms["cluster_0"] > 0 and rec.grad_norms["cluster_1"] > 0
    assert rec.grad_norms["cluster_0"] != rec.grad_norms["cluster_1"]
    assert rec.rows == {"cluster_0": 2, "cluster_1": 4}


# ------------------------------------------------------------------- the refusals -------


def test_a_loss_that_never_reaches_the_adapter_raises_instead_of_training_nothing():
    """The silent no-op: a cluster reports a loss every step and trains nothing."""
    a = tiny_peft()
    opt = optimiser(a)

    def detached_loss(model, mb):
        """A loss with no path to any parameter, which is what a wiring bug looks like."""
        return (mb.float().sum() * 0.0).requires_grad_(True)

    with pytest.raises(AdapterIsolationError, match="received no gradient"):
        a.step({"cluster_0": batch()}, detached_loss, opt)


def test_a_batch_naming_an_unmanaged_adapter_is_refused():
    """Its gradient would go nowhere, and nothing else would say so."""
    a = tiny_peft()
    with pytest.raises(AdapterIsolationError, match="does not manage"):
        a.step({"cluster_9": batch()}, lm_loss, optimiser(a))


def test_building_a_set_over_a_missing_adapter_is_refused():
    """Creating it lazily would give one cluster a fresh expert mid-run."""
    a = tiny_peft()
    with pytest.raises(AdapterIsolationError, match="not on the model"):
        ClusterAdapterSet(a.model, ("cluster_0", "cluster_7"))


def test_only_leaves_exactly_one_adapter_trainable():
    """Both mechanisms that make isolation work, checked directly.

    PEFT's forward loops over the active adapters, and set_adapter also flips requires_grad.
    Either alone would do; the check is here because the semantics belong to PEFT, not to
    this project, and an upgrade could change them.
    """
    a = tiny_peft()
    with a.only("cluster_1"):
        assert all(p.requires_grad for _k, p in a.parameters("cluster_1"))
        assert not any(p.requires_grad for _k, p in a.parameters("cluster_0"))
        assert not any(p.requires_grad for _k, p in a.parameters("shared"))


def test_only_restores_the_previous_adapter_even_when_the_body_raises():
    """A failed microbatch must not leave the model pointed at the wrong cluster.

    It would send the NEXT cluster's gradient somewhere else entirely, silently.
    """
    a = tiny_peft()
    a.model.set_adapter("cluster_0")
    with pytest.raises(RuntimeError, match="boom"):
        with a.only("cluster_1"):
            raise RuntimeError("boom")
    assert list(a.model.active_adapters) == ["cluster_0"]


def test_an_unmanaged_name_cannot_be_activated():
    a = tiny_peft()
    with pytest.raises(AdapterIsolationError, match="not managed"):
        with a.only("nope"):
            pass


def test_an_empty_adapter_set_is_refused():
    with pytest.raises(ValueError, match="at least one adapter"):
        tiny_peft(names=())


# ------------------------------------------------------- groups to rows to microbatches --


def test_rows_follow_their_group_because_a_group_shares_one_behaviour():
    """The clustering labels GROUPS; a group's samples cannot land on different experts."""
    idx = cluster_row_index(["cluster_0", "shared", "cluster_0"], [2, 3, 1])
    assert idx == {"cluster_0": [0, 1, 5], "shared": [2, 3, 4]}


def test_every_row_is_assigned_exactly_once():
    idx = cluster_row_index(["a", "b", "a", "b"], [2, 2, 2, 2])
    rows = sorted(r for v in idx.values() for r in v)
    assert rows == list(range(8))


def test_a_partition_that_does_not_describe_the_batch_is_refused():
    with pytest.raises(ValueError, match="does not describe this batch"):
        cluster_row_index(["a", "b"], [2])


def test_a_zero_size_group_is_refused():
    """It would count toward a cluster's size while contributing no rows to train on."""
    with pytest.raises(ValueError, match="must be positive"):
        cluster_row_index(["a"], [0])


def test_unchanged_is_bit_equality_and_not_a_tolerance():
    """A leaked update can be tiny, and tiny is what a tolerance is for.

    ``allclose`` with its default rtol would call a perturbation of 1e-9 "unchanged", so the
    isolation guard would pass on an implementation that leaked a small gradient into every
    idle expert -- which is what decoupled weight decay does. Pinned directly, because no
    other test in this file can distinguish the two comparisons.
    """
    a = tiny_peft()
    before = a.snapshot("cluster_0")
    assert a.unchanged("cluster_0", before)
    with torch.no_grad():
        next(iter(a.parameters("cluster_0")))[1].add_(1e-9)
    assert not a.unchanged("cluster_0", before), "a 1e-9 perturbation was called unchanged"
