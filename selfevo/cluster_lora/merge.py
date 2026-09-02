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
"""

from __future__ import annotations

from typing import Mapping, Sequence

__all__ = ["MergeInexact", "adapter_delta", "merge_sum", "summed_delta"]


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
