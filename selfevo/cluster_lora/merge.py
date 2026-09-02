"""Merging the cluster experts into one adapter for inference.

Training routes each group to its own expert. Inference has no such label -- the behavioural
cluster is read off the RESPONSE, and at inference time there is no response yet, so routing
by prompt would be circular. LSPO (arXiv 2607.27787) takes the other exit: keep the experts
separate during training and SUM them afterwards, so the benefit claimed is a training-time
one (less gradient interference per expert) and the deployed model is a single adapter with
no routing at all.

The sum is exact, and not by luck. For LoRA the increment is ``scaling_c * B_c A_c``, which
is linear in the pair, so concatenating the ``A`` blocks (each pre-scaled by
``w_c * scaling_c``) and the ``B`` blocks gives one adapter of rank ``sum(r_c)`` whose
product is exactly ``sum_c w_c * scaling_c * B_c A_c``. That is what PEFT's
``add_weighted_adapter(combination_type="cat")`` does, so it is called rather than
reimplemented -- but the result is CHECKED against the independently computed sum of deltas
before it is returned. A merge that silently differed would change the deployed model
without changing anything a training metric can see.

Routing by a prompt-time classifier is a later flag and is deliberately absent here: it is a
different method with a different failure mode, and shipping it alongside the merge would
make a measured gain unattributable between the two.

THE MERGE SEMANTICS AT SAVE TIME, AND WHY THEY ARE THIS AND NOT SOMETHING ELSE.

Every exit point of a cluster-routed run -- HF checkpoint, weight sync to the rollout engine,
adapter metadata -- carries the PLAIN SUM of the experts, weights all 1.0. ``MERGE_OPERATOR``
records that, and it is one string rather than a configurable knob on purpose.

1. It is what the training denominator already implies. ``_train_batch_by_cluster`` computes
   ``total_loss_weight`` over the WHOLE batch and hands every cluster the same global
   denominator, so expert ``c`` accumulates exactly its cluster's share of the batch gradient.
   Summing the experts reconstitutes the update a single shared adapter would have accumulated
   had there been no routing -- which is precisely the A0 baseline the method is measured
   against. A MEAN would divide the total update by K, so A1 - A0 would be reading a K-fold
   difference in effective learning rate rather than the clustering. Weighting by cluster SIZE
   would count size twice, because the size is already in the gradient through that shared
   denominator.
2. It is exact. ``sum_c scaling_c B_c A_c`` is representable with no approximation as one
   adapter of rank ``sum_c r_c``, by the linearity argument above, so nothing about the
   deployed model is an artefact of the merge.
3. ``experiments/m25/PLAN.md`` fixes it for every arm: "Adapters merged (summed) at eval for
   every multi-adapter arm; the merge operator is held fixed across arms."

That last point is a constraint, not a preference. ``FINDINGS_cluster_lora.md`` records that
cluster-then-merge sits next to task arithmetic (TIES, DARE), whose entire content is the
choice of operator, and that RL task vectors are measured near-orthogonal (2608.03573, cosine
~1e-5). Under near-orthogonality the operator scales the deployed delta almost linearly while
changing nothing a training metric can see, so an arm merged with one operator and its control
merged with another would differ BY THE OPERATOR and the difference would be reported as the
method. The operator is therefore held FIXED across arms, and a second one is not added here:
it would be an uncontrolled second axis in every comparison the paper makes.

WHY THE EXPORT DOES NOT CALL ``merge_sum``. At an exit point the experts are FSDP2 DTensors
sharded across the training group, and PEFT's ``add_weighted_adapter`` builds new unsharded
modules on the live model -- new parameters outside the optimizer and outside FSDP, in the
middle of training. ``merge_expert_tensors`` is the same arithmetic over already-gathered
tensors, and it is not trusted on that claim: ``test_cluster_lora_export.py`` pins it
tensor-for-tensor against what PEFT's own ``add_weighted_adapter(combination_type='cat')``
produces on a real model, so PEFT stays the reference implementation and this stays transport.
"""

from __future__ import annotations

from typing import Mapping, Sequence

__all__ = [
    "ADAPTER_LAYER_NAMES",
    "MERGE_OPERATOR",
    "MergeInexact",
    "MergeSelectionError",
    "adapter_delta",
    "expert_scalings",
    "merge_expert_tensors",
    "merge_sum",
    "merged_lora_config",
    "merged_lora_shapes",
    "split_lora_param_name",
    "summed_delta",
]

#: The one operator every exit point uses. See the module docstring: it is fixed across arms
#: because the arms are compared with each other, and a per-arm operator would be a second axis.
MERGE_OPERATOR = "sum"

#: PEFT's own ``LoraLayer.adapter_layer_names``, as of peft 0.18.1. Named here rather than
#: imported so that a parameter whose layout this module does not understand is REFUSED by
#: :func:`split_lora_param_name` instead of vanishing from an export in silence; the agreement
#: with PEFT is a test, so an upgrade that adds a layer type fails rather than drops it.
ADAPTER_LAYER_NAMES = (
    "lora_A",
    "lora_B",
    "lora_embedding_A",
    "lora_embedding_B",
    "lora_magnitude_vector",
)


class MergeSelectionError(RuntimeError):
    """An export was about to carry the wrong set of adapters.

    Raised, never warned, and raised on the DEFAULT path too. Every selector in this repo's
    save and sync paths picked LoRA tensors by ``requires_grad``, which is true of exactly the
    adapter ``ClusterAdapterSet.only()`` last activated -- so a checkpoint or a weight sync
    from a cluster-routed run carried ONE expert, loaded cleanly, served cleanly, and
    evaluated to a number describing a model no arm produced. The failure has no other
    observable, so the selection refuses rather than reports.
    """


class MergeInexact(RuntimeError):
    """The merged adapter does not reproduce the sum of the deltas it was built from.

    Raised rather than tolerated. The merged adapter IS the deployed model; if it differs
    from the sum of the experts, every number the paper reports at inference describes a
    model that no training arm produced.
    """


def _lora_modules(model, name: str):
    """``(module_name, module)`` for every tuner layer carrying adapter ``name``."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    for mod_name, mod in model.named_modules():
        if isinstance(mod, BaseTunerLayer) and name in getattr(mod, "lora_A", {}):
            yield mod_name, mod


def adapter_delta(model, name: str) -> dict[str, "object"]:
    """The weight increment one adapter contributes, per target module.

    Args:
        model: A ``PeftModel``.
        name: Adapter name.

    Returns:
        ``{module_name: dW}`` with ``dW = scaling * B @ A``, the exact tensor added to the
        base weight. Computed in float64 so the comparison against a merged adapter is
        limited by the merge, not by the check.

    Raises:
        MergeInexact: If the adapter is on no module. An adapter contributing nothing would
            merge cleanly and change nothing, and the merge would still report success.
    """
    import torch

    out: dict[str, object] = {}
    for mod_name, mod in _lora_modules(model, name):
        A = mod.lora_A[name].weight.detach().to(torch.float64)
        B = mod.lora_B[name].weight.detach().to(torch.float64)
        out[mod_name] = float(mod.scaling[name]) * (B @ A)
    if not out:
        raise MergeInexact(
            f"adapter {name!r} is present on no LoRA module, so its contribution to a "
            "merge is nothing at all"
        )
    return out


def summed_delta(
    model, names: Sequence[str], weights: Sequence[float] | None = None
) -> dict[str, "object"]:
    """The reference: the weighted sum of the experts' deltas, computed directly.

    This is what the merged adapter has to equal. It is computed from the individual
    adapters and never from the merge, so the check is independent of the thing being
    checked.

    Args:
        model: A ``PeftModel``.
        names: Adapters to sum.
        weights: Per-adapter weight. ``None`` means 1.0 each, which is the LSPO-style plain
            sum and the default the method uses.

    Returns:
        ``{module_name: dW}``.

    Raises:
        ValueError: If the counts disagree, or ``names`` is empty.
        MergeInexact: If the adapters do not cover the same set of modules. A merge over a
            ragged set would silently apply some experts on some layers only.
    """
    import torch

    names = tuple(names)
    if not names:
        raise ValueError("nothing to merge")
    w = [1.0] * len(names) if weights is None else [float(x) for x in weights]
    if len(w) != len(names):
        raise ValueError(f"{len(w)} weights for {len(names)} adapters")
    total: dict[str, object] = {}
    modules: set[str] | None = None
    for name, wi in zip(names, w):
        d = adapter_delta(model, name)
        here = set(d)
        if modules is None:
            modules = here
        elif here != modules:
            raise MergeInexact(
                f"adapter {name!r} covers modules {sorted(here ^ modules)} that its peers "
                "do not; a merge over a ragged module set applies some experts on some "
                "layers only"
            )
        for mod_name, dW in d.items():
            total[mod_name] = total.get(mod_name, torch.zeros_like(dW)) + wi * dW
    return total


def merge_sum(
    model,
    names: Sequence[str],
    target: str = "merged",
    weights: Sequence[float] | None = None,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> str:
    """Build one adapter equal to the weighted sum of the cluster experts, and verify it.

    Args:
        model: A ``PeftModel`` carrying every adapter in ``names``.
        names: The cluster experts, shared adapter included.
        target: Name for the merged adapter. Replaced if it already exists, so a second
            merge later in a run does not silently stack on the first.
        weights: Per-adapter weights; ``None`` is the plain sum.
        atol: Absolute tolerance of the verification, in float64.
        rtol: Relative tolerance, applied against the magnitude of the reference sum, so a
            large-magnitude layer is not held to an absolute bound it cannot meet in the
            model's own dtype.

    Returns:
        ``target``.

    Raises:
        MergeInexact: If the merged adapter's delta differs from the reference sum by more
            than the tolerance, naming the worst module and the size of the disagreement.
    """
    import torch

    names = tuple(names)
    reference = summed_delta(model, names, weights)
    if target in (getattr(model, "peft_config", {}) or {}):
        model.delete_adapter(target)
    w = [1.0] * len(names) if weights is None else [float(x) for x in weights]
    model.add_weighted_adapter(
        adapters=list(names), weights=w, adapter_name=target, combination_type="cat"
    )
    got = adapter_delta(model, target)
    worst = 0.0
    worst_name = ""
    for mod_name, ref in reference.items():
        if mod_name not in got:
            raise MergeInexact(
                f"the merged adapter is missing module {mod_name!r} that the experts cover"
            )
        err = float((got[mod_name] - ref).abs().max())
        scale = float(ref.abs().max())
        if err > atol + rtol * scale and err > worst:
            worst, worst_name = err, mod_name
    if worst_name:
        raise MergeInexact(
            f"the merged adapter does not reproduce the sum of the experts: worst "
            f"disagreement {worst:.3e} on {worst_name!r}. The merged adapter is what gets "
            "deployed, so this cannot be tolerated"
        )
    return target


def split_lora_param_name(name: str) -> tuple[str, str, str, str]:
    """Take a PEFT adapter parameter name apart, or refuse to.

    Args:
        name: A name as ``Module.named_parameters()`` yields it, for example
            ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.cluster_0.weight``.

    Returns:
        ``(module_path, layer_name, adapter, tail)``. ``tail`` is ``"weight"`` or ``"bias"``
        for a layer PEFT stores as an ``nn.Module``, and ``""`` for one stored as a bare
        ``nn.Parameter`` (the embedding adapters).

    Raises:
        MergeSelectionError: If the name carries no adapter layer this module understands.
            An export that quietly skipped such a tensor would ship an adapter missing a
            module, which loads and serves and is not the model that was trained.
    """
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part in ADAPTER_LAYER_NAMES:
            if i + 1 >= len(parts):
                raise MergeSelectionError(
                    f"{name!r} ends at the adapter layer {part!r} and names no adapter, so "
                    "there is no way to tell which expert it belongs to"
                )
            return (
                ".".join(parts[:i]),
                part,
                parts[i + 1],
                ".".join(parts[i + 2 :]),
            )
    raise MergeSelectionError(
        f"{name!r} carries no adapter layer from {list(ADAPTER_LAYER_NAMES)}; this module "
        "cannot tell which expert it belongs to, and guessing would file one expert's tensor "
        "under another expert's key"
    )


def merge_expert_tensors(
    named_tensors: Mapping[str, "object"],
    names: Sequence[str],
    scalings: Mapping[str, float],
    weights: Sequence[float] | None = None,
) -> dict[str, "object"]:
    """The experts, as ALREADY-GATHERED tensors, summed into one adapter's tensors.

    The arithmetic of ``add_weighted_adapter(combination_type="cat")`` applied to plain
    tensors, so it can run at an exit point where the live model is FSDP-sharded and PEFT
    cannot be called. ``A`` blocks are concatenated along their rank axis after scaling by
    ``weight * scaling``; ``B`` blocks are concatenated along theirs and are NOT scaled. The
    merged adapter has rank ``sum_c r_c`` and ``lora_alpha`` equal to it, so its own scaling
    is 1.0 and its delta is exactly ``sum_c weight_c * scaling_c * B_c A_c``. See
    :func:`merged_lora_config`, and the module docstring for why the operator is the sum.

    Args:
        named_tensors: ``{parameter_name: tensor}`` covering every expert in ``names``, keyed
            as ``named_parameters()`` names them. A tensor belonging to some other adapter is
            refused rather than ignored.
        names: The experts, in the order their blocks are concatenated. The order is part of
            the artifact: ``A`` and ``B`` must be concatenated in the SAME order, or the
            product pairs each expert's ``B`` with another expert's ``A``.
        scalings: ``{adapter: scaling}``, i.e. what ``LoraLayer.scaling[adapter]`` holds.
        weights: Per-expert weight; ``None`` is the plain sum, which is
            :data:`MERGE_OPERATOR`.

    Returns:
        ``{key: tensor}`` where ``key`` is the parameter name with the adapter segment
        removed -- the key format ``PeftModel.save_pretrained`` writes and the rollout engines
        parse.

    Raises:
        ValueError: If ``names`` is empty or the weight count disagrees.
        MergeSelectionError: If a module carries some experts and not others, if a tensor
            belongs to no named expert, if nothing was selected at all, or if a DoRA magnitude
            vector is present. That last one matters: PEFT's own ``cat`` branch SKIPS
            ``lora_magnitude_vector`` and leaves it zeroed, so a DoRA merge there is silently
            not a merge.
    """
    import torch

    names = tuple(names)
    if not names:
        raise ValueError("nothing to merge")
    w = [1.0] * len(names) if weights is None else [float(x) for x in weights]
    if len(w) != len(names):
        raise ValueError(f"{len(w)} weights for {len(names)} adapters")
    weight_of = dict(zip(names, w))

    out: dict[str, object] = {}
    for key, layer_name, per_adapter in group_expert_blocks(named_tensors, names):
        ordered = [per_adapter[n] for n in names]
        if layer_name.endswith("_A"):
            out[key] = torch.cat(
                [t * (weight_of[n] * float(scalings[n])) for n, t in zip(names, ordered)],
                dim=0,
            )
        else:
            out[key] = torch.cat(ordered, dim=1)
    return out


def merged_lora_config(peft_configs: Mapping[str, "object"], names: Sequence[str]):
    """The ``LoraConfig`` describing the merged adapter, exactly as PEFT's ``cat`` builds it.

    The rank and alpha are load-bearing beyond bookkeeping. vLLM and SGLang are handed ``r``
    and ``lora_alpha`` (``WeightUpdateMeta.peft_config`` on the xccl path,
    ``adapter_config.json`` on the disk path) and apply ``alpha / r`` themselves. Shipping a
    merged adapter under one expert's ``r`` would rescale the whole deployed delta -- across
    five experts, by a factor of five -- with nothing anywhere reporting it.

    Args:
        peft_configs: ``{adapter: LoraConfig}``, e.g. ``model.peft_config``.
        names: The experts being merged, in concatenation order.

    Returns:
        A new ``LoraConfig`` whose ``r`` and ``lora_alpha`` are both ``sum_c r_c``, with the
        rank and alpha patterns cleared and the target modules unioned -- PEFT's own choices,
        pinned by a test against ``add_weighted_adapter``.

    Raises:
        MergeSelectionError: If an expert has no config, so its rank cannot be counted.
    """
    import functools
    import operator
    from dataclasses import replace

    names = tuple(names)
    missing = [n for n in names if n not in peft_configs]
    if missing:
        raise MergeSelectionError(
            f"experts {missing} have no peft config; their rank cannot be counted and the "
            "merged adapter would be advertised at the wrong scale"
        )
    ranks = []
    for name in names:
        cfg = peft_configs[name]
        pattern = getattr(cfg, "rank_pattern", None) or {}
        ranks.append(max(cfg.r, *pattern.values()) if pattern else cfg.r)

    targets = [peft_configs[n].target_modules for n in names]
    if isinstance(targets[0], str):
        new_targets = "|".join(f"({t})" for t in targets)
    elif isinstance(targets[0], set):
        new_targets = functools.reduce(operator.or_, targets)
    else:
        new_targets = targets[0]
    return replace(
        peft_configs[names[0]],
        r=int(sum(ranks)),
        lora_alpha=int(sum(ranks)),
        target_modules=new_targets,
        alpha_pattern={},
        rank_pattern={},
    )


def group_expert_blocks(named_items: Mapping[str, "object"], names: Sequence[str]):
    """Every export's selection step: one entry per adapter layer, carrying all its experts.

    Shared by :func:`merge_expert_tensors` and :func:`merged_lora_shapes` so that the tensors
    an export writes and the shapes it advertises are selected by ONE piece of code. Two
    selections would be two chances to disagree, and a shape that disagreed with its tensor is
    exactly the kind of mismatch a rollout engine reports as a load failure at the worst
    moment.

    Args:
        named_items: ``{parameter_name: anything}`` -- tensors at a save point, shapes where
            no collective may be taken.
        names: The experts, in concatenation order.

    Yields:
        ``(export_key, layer_name, {adapter: item})`` in sorted key order, where
        ``export_key`` is the parameter name with the adapter segment removed.

    Raises:
        MergeSelectionError: If a tensor belongs to no named expert, if a module carries some
            experts and not others, if a DoRA magnitude vector is present, or if nothing was
            selected at all. All four are silent otherwise: each one writes a valid adapter
            that is not the model that was trained.
    """
    wanted = set(names)
    blocks: dict[tuple[str, str, str], dict[str, object]] = {}
    for name, item in named_items.items():
        module_path, layer_name, adapter, tail = split_lora_param_name(name)
        if layer_name == "lora_magnitude_vector":
            raise MergeSelectionError(
                f"{name!r} is a DoRA magnitude vector, which a rank-concatenating merge "
                "cannot carry -- PEFT's own 'cat' branch skips it and leaves it zeroed, so "
                "the merged adapter would silently not be the sum of the experts"
            )
        if adapter not in wanted:
            raise MergeSelectionError(
                f"{name!r} belongs to adapter {adapter!r}, which is not one of the experts "
                f"being merged ({list(names)}); an export that dropped it would ship a model "
                "no arm trained"
            )
        blocks.setdefault((module_path, layer_name, tail), {})[adapter] = item

    if not blocks:
        raise MergeSelectionError(
            f"no adapter tensors were selected for experts {list(names)}; an empty export "
            "writes a valid adapter that changes nothing, and every eval of it would report "
            "the base model as the method"
        )
    for (module_path, layer_name, tail), per_adapter in sorted(blocks.items()):
        missing = [n for n in names if n not in per_adapter]
        if missing:
            raise MergeSelectionError(
                f"module {module_path!r} carries {layer_name} for {sorted(per_adapter)} but "
                f"not for {missing}; a merge over a ragged module set applies some experts "
                "on some layers only"
            )
        key = f"{module_path}.{layer_name}" + (f".{tail}" if tail else "")
        yield key, layer_name, per_adapter


def merged_lora_shapes(
    named_shapes: Mapping[str, Sequence[int]], names: Sequence[str]
) -> dict[str, list[int]]:
    """The merged adapter's tensor shapes, without gathering a single tensor.

    Some exit points advertise shapes where no collective may be taken -- ``DTensor.shape`` is
    already the global shape, and an engine method that a controller calls on rank 0 alone
    would DEADLOCK if it started an all-gather. So the shapes are derived arithmetically from
    the same selection :func:`merge_expert_tensors` uses, and the two are asserted to agree.

    Args:
        named_shapes: ``{parameter_name: shape}`` for every expert's tensors.
        names: The experts, in concatenation order.

    Returns:
        ``{export_key: shape}``, with ``A`` ranks summed on axis 0 and ``B`` ranks on axis 1.

    Raises:
        MergeSelectionError: As :func:`group_expert_blocks`, plus a ragged non-rank axis --
            two experts whose ``A`` blocks disagree on the input width do not describe the
            same module and cannot be concatenated.
    """
    out: dict[str, list[int]] = {}
    for key, layer_name, per_adapter in group_expert_blocks(named_shapes, names):
        shapes = [list(per_adapter[n]) for n in names]
        axis = 0 if layer_name.endswith("_A") else 1
        others = {tuple(s[:axis] + s[axis + 1 :]) for s in shapes}
        if len(others) != 1:
            raise MergeSelectionError(
                f"{key!r} has experts of shapes {shapes}, which differ off the rank axis "
                f"{axis}; they do not describe the same module and cannot be concatenated"
            )
        merged = list(shapes[0])
        merged[axis] = int(sum(s[axis] for s in shapes))
        out[key] = merged
    return out


def expert_scalings(model, names: Sequence[str]) -> dict[str, float]:
    """``{adapter: scaling}``, read off the live model and required to be uniform.

    ``merge_expert_tensors`` takes one scaling per expert, which is only meaningful if every
    module agrees. ``ClusterAdapterSet.build`` gives every expert the SAME ``LoraConfig``
    precisely so they do -- experts differing in rank differ in capacity, and a capacity
    difference between clusters would be an uncontrolled second axis. A ``rank_pattern`` or
    ``alpha_pattern`` would break that silently, scaling some layers of some experts wrongly
    in the merged adapter, so it is refused here rather than absorbed.

    Args:
        model: A ``PeftModel``.
        names: The experts.

    Returns:
        ``{adapter: scaling}``.

    Raises:
        MergeSelectionError: If an expert's scaling varies between modules, or if an expert is
            on no module at all.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    out: dict[str, float] = {}
    for mod_name, mod in model.named_modules():
        if not isinstance(mod, BaseTunerLayer):
            continue
        table = getattr(mod, "scaling", None) or {}
        for name in names:
            if name not in table:
                continue
            value = float(table[name])
            if name in out and out[name] != value:
                raise MergeSelectionError(
                    f"adapter {name!r} has scaling {out[name]} on one module and {value} on "
                    f"{mod_name!r}; a per-module scaling cannot be carried by one merged "
                    "adapter, so some layers of this expert would ship at the wrong scale"
                )
            out[name] = value
    missing = [n for n in names if n not in out]
    if missing:
        raise MergeSelectionError(
            f"adapters {missing} are on no LoRA module of this model, so they have no "
            "scaling and contribute nothing to a merge"
        )
    return out
