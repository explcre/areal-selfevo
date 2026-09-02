"""One LoRA adapter per cluster, and the guarantee that a cluster's loss moves only its own.

The mechanism is small and the failure mode is large. Each cluster's microbatch is run with
only that cluster's adapter active, so the gradient it produces lands in that adapter alone;
after every cluster has run, ONE optimizer step applies each adapter's own accumulated
gradient. If that isolation is not real, the method is vanilla LoRA with extra bookkeeping --
and it would still train, still log, and still produce a number, which is exactly why the
isolation is asserted as a bit-level fact in ``test_cluster_lora_routing.py`` rather than
argued for here.

Two independent things make it hold, and both are checked:

1. **PEFT's forward loops over the ACTIVE adapters only** -- ``LoraLayer.forward`` iterates
   ``self.active_adapters`` -- so an inactive adapter is not in the autograd graph at all and
   cannot receive a gradient.
2. **``set_adapter`` also sets ``requires_grad``**, ``True`` for the active adapter and
   ``False`` for the rest, so even a path that did reach an inactive adapter would not
   accumulate into it.

The third leak is the optimizer, not the model: Adam with momentum moves a parameter on a
step even when its current gradient is zero. ``set_to_none=True`` is therefore not a
micro-optimisation here. A parameter whose ``.grad`` is ``None`` is SKIPPED by
``torch.optim``, so its state is never created and its value is bit-identical afterwards;
a parameter whose ``.grad`` is a zero TENSOR gets an Adam state and, on any step after the
first, moves. The guard test would fail on a zero tensor, which is the point.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

__all__ = [
    "AdapterIsolationError",
    "ClusterAdapterSet",
    "ClusterStepRecord",
    "cluster_row_index",
]


class AdapterIsolationError(RuntimeError):
    """A cluster's update did not reach its own adapter, or reached another one.

    Raised, never warned. Both directions are silent in every observable except the weights:
    a cluster whose loss reaches no adapter trains nothing while reporting a loss, and a
    cluster whose loss reaches every adapter is the shared-adapter baseline wearing the
    method's name.
    """


def cluster_row_index(
    keys: Sequence[str], group_sizes: Sequence[int]
) -> dict[str, list[int]]:
    """Rows of the batch belonging to each cluster.

    Groups, not rows, are what the clustering labels -- a GRPO group is one prompt's
    samples and they share a behaviour vector -- so a group's rows all follow its label.
    This is the step that turns a per-group partition into the per-cluster microbatches the
    training step consumes.

    Args:
        keys: Adapter name per group, in batch order.
        group_sizes: Rows per group, in the same order.

    Returns:
        ``{adapter_name: sorted row indices}``. Every row appears exactly once.

    Raises:
        ValueError: If the counts disagree or a size is not positive. A zero-size group
            would give a cluster no rows while still counting toward its size, which makes
            the reach metrics overstate what was trained.
    """
    if len(keys) != len(group_sizes):
        raise ValueError(
            f"{len(keys)} cluster keys but {len(group_sizes)} group sizes; the partition "
            "does not describe this batch"
        )
    out: dict[str, list[int]] = {}
    row = 0
    for key, size in zip(keys, group_sizes):
        size = int(size)
        if size <= 0:
            raise ValueError(f"group size must be positive, got {size}")
        out.setdefault(key, []).extend(range(row, row + size))
        row += size
    return out


@dataclass
class ClusterStepRecord:
    """What one per-cluster step did, for the run record.

    Args:
        losses: Loss per cluster that ran.
        grad_norms: Gradient norm per cluster, over that cluster's adapter only. A cluster
            whose norm is exactly 0.0 produced no update, and the difference between that
            and "the cluster had no rows" is the difference between a bug and a batch.
        rows: Rows per cluster.
        skipped: Clusters with no rows this batch. Recorded because an adapter that is
            skipped for many consecutive batches is stale by the time it is merged, and
            nothing else in the metrics would say so.
    """

    losses: Mapping[str, float] = field(default_factory=dict)
    grad_norms: Mapping[str, float] = field(default_factory=dict)
    rows: Mapping[str, int] = field(default_factory=dict)
    skipped: tuple[str, ...] = ()

    def as_metrics(self) -> dict[str, float]:
        """Flat scalars for the run's metrics namespace."""
        out = {"cluster_lora/clusters_stepped": float(len(self.losses)),
               "cluster_lora/clusters_skipped": float(len(self.skipped))}
        for name, v in self.losses.items():
            out[f"cluster_lora/loss/{name}"] = float(v)
        for name, v in self.grad_norms.items():
            out[f"cluster_lora/grad_norm/{name}"] = float(v)
        for name, v in self.rows.items():
            out[f"cluster_lora/rows/{name}"] = float(v)
        return out


class ClusterAdapterSet:
    """N cluster adapters plus a shared one on a single PEFT model.

    Args:
        model: A ``PeftModel`` that already carries at least one adapter.
        names: Adapter names this set manages, in a fixed order.

    Raises:
        AdapterIsolationError: If a named adapter is not present on the model. Creating it
            silently would give the cluster a freshly-initialised expert half way through a
            run, which trains but is not the run that was configured.
    """

    def __init__(self, model, names: Sequence[str]) -> None:
        self.model = model
        self.names = tuple(names)
        if not self.names:
            raise ValueError("a cluster adapter set needs at least one adapter")
        present = set(getattr(model, "peft_config", {}) or {})
        missing = [n for n in self.names if n not in present]
        if missing:
            raise AdapterIsolationError(
                f"adapters {missing} are not on the model (it has {sorted(present)}); "
                "they must be created up front so every cluster's expert has the same "
                "number of steps behind it"
            )

    @classmethod
    def build(cls, model, names: Sequence[str], peft_config) -> "ClusterAdapterSet":
        """Wrap a base model and create every cluster adapter from one config.

        One config for every adapter, deliberately: experts that differ in rank differ in
        capacity, and a capacity difference between clusters would be an uncontrolled
        second axis in every comparison the paper makes.

        Args:
            model: A base causal LM, or a ``PeftModel`` to extend.
            names: Adapter names to create.
            peft_config: The ``LoraConfig`` shared by all of them.

        Returns:
            The set, with ``self.model`` the PEFT-wrapped model.
        """
        from peft import get_peft_model

        names = tuple(names)
        if not names:
            raise ValueError("a cluster adapter set needs at least one adapter")
        if hasattr(model, "peft_config"):
            wrapped = model
            start = 0
        else:
            wrapped = get_peft_model(
                model, peft_config, adapter_name=names[0], autocast_adapter_dtype=False
            )
            start = 1
        for name in names[start:]:
            if name not in (getattr(wrapped, "peft_config", {}) or {}):
                wrapped.add_adapter(name, peft_config)
        return cls(wrapped, names)

    def _tuner_layers(self):
        """Every PEFT tuner layer on the model, so adapter tensors are read structurally.

        Reading them by parameter-name substring instead would match ``cluster_1`` inside
        ``cluster_11`` on any naming scheme without separators, and the resulting overlap
        would make the isolation guard pass while the isolation was broken.
        """
        from peft.tuners.tuners_utils import BaseTunerLayer

        for mod_name, mod in self.model.named_modules():
            if isinstance(mod, BaseTunerLayer):
                yield mod_name, mod

    def parameters(self, name: str):
        """``(qualified_name, parameter)`` for one adapter's tensors.

        Raises:
            AdapterIsolationError: If the adapter owns no parameters. An adapter with no
                tensors cannot be trained and cannot be merged, and every metric about it
                would read zero rather than missing.
        """
        import torch.nn as nn

        found = False
        for mod_name, mod in self._tuner_layers():
            for layer_name in getattr(mod, "adapter_layer_names", ()):  # lora_A, lora_B, ...
                table = getattr(mod, layer_name, None)
                if table is None or name not in table:
                    continue
                entry = table[name]
                if isinstance(entry, nn.Module):
                    for pname, param in entry.named_parameters():
                        found = True
                        yield f"{mod_name}.{layer_name}.{name}.{pname}", param
                else:
                    found = True
                    yield f"{mod_name}.{layer_name}.{name}", entry
        if not found:
            raise AdapterIsolationError(
                f"adapter {name!r} owns no parameters on this model; it cannot receive an "
                "update and merging it would add nothing"
            )

    def snapshot(self, name: str) -> dict[str, "object"]:
        """Detached clones of one adapter's tensors, for a bit-level before/after check."""
        return {k: p.detach().clone() for k, p in self.parameters(name)}

    def unchanged(self, name: str, before: Mapping[str, "object"]) -> bool:
        """Whether every tensor of ``name`` is BIT-identical to a snapshot.

        Exact equality, not ``allclose``. A tolerance would pass an adapter that received a
        small update, and "a small update leaked into the wrong expert" is precisely the
        defect this is here to detect.
        """
        import torch

        now = dict(self.parameters(name))
        if set(now) != set(before):
            return False
        return all(torch.equal(now[k], before[k]) for k in before)

    @contextmanager
    def only(self, name: str):
        """Run a block with exactly one adapter active and trainable.

        Restores the previously active adapters on the way out, including on an exception,
        so a failed microbatch cannot leave the model configured for the wrong cluster --
        which would send the NEXT cluster's gradient somewhere else entirely and would not
        raise anywhere.

        Args:
            name: The adapter to activate.

        Raises:
            AdapterIsolationError: If, inside the block, any parameter of another managed
                adapter still requires grad. That is the state in which a leak is possible,
                and it is checked rather than assumed because PEFT's activation semantics
                are the one thing here this project does not own.
        """
        if name not in self.names:
            raise AdapterIsolationError(
                f"{name!r} is not managed by this set ({list(self.names)})"
            )
        previous = list(getattr(self.model, "active_adapters", []) or [])
        self.model.set_adapter(name)
        try:
            for other in self.names:
                if other == name:
                    continue
                for pname, param in self.parameters(other):
                    if param.requires_grad:
                        raise AdapterIsolationError(
                            f"activating {name!r} left {pname} trainable; a backward pass "
                            f"here could accumulate into adapter {other!r}"
                        )
            yield self
        finally:
            if previous:
                self.model.set_adapter(previous[0] if len(previous) == 1 else previous)

    def step(
        self,
        batches: Mapping[str, object],
        loss_fn: Callable[[object, object], object],
        optimizer,
        *,
        require_gradient: bool = True,
    ) -> ClusterStepRecord:
        """One training step: each cluster's loss into its own adapter, then one update.

        Args:
            batches: ``{adapter_name: microbatch}``. A cluster with no rows this batch is
                simply absent and is recorded as skipped.
            loss_fn: ``(model, microbatch) -> scalar loss``.
            optimizer: Stepped once, after every cluster has contributed. ``zero_grad`` is
                called with ``set_to_none=True`` first, so an adapter that received no
                gradient is skipped by the optimizer entirely and keeps its exact value.
            require_gradient: Raise if a cluster's backward left its own adapter without a
                gradient. On by default: that is the silent no-op this whole module exists
                to prevent, and a run that hits it should stop rather than train a subset of
                its experts.

        Returns:
            A :class:`ClusterStepRecord`.

        Raises:
            AdapterIsolationError: If ``batches`` names an unmanaged adapter, or a cluster's
                loss failed to reach its own adapter while ``require_gradient`` is set.
        """
        import torch

        unknown = [k for k in batches if k not in self.names]
        if unknown:
            raise AdapterIsolationError(
                f"batches name adapters {unknown} that this set does not manage "
                f"({list(self.names)}); their gradient would go nowhere"
            )
        optimizer.zero_grad(set_to_none=True)
        losses: dict[str, float] = {}
        norms: dict[str, float] = {}
        rows: dict[str, int] = {}
        for name in self.names:
            mb = batches.get(name)
            if mb is None:
                continue
            with self.only(name):
                loss = loss_fn(self.model, mb)
                loss.backward()
            losses[name] = float(loss.detach())
            grads = [p.grad for _k, p in self.parameters(name) if p.grad is not None]
            if not grads:
                if require_gradient:
                    raise AdapterIsolationError(
                        f"cluster {name!r} ran a backward pass and its adapter received no "
                        "gradient at all; the loss is not connected to the adapter, so this "
                        "cluster would train nothing while reporting a loss"
                    )
                norms[name] = 0.0
            else:
                norms[name] = float(
                    torch.sqrt(sum((g.detach().float() ** 2).sum() for g in grads))
                )
            n = getattr(mb, "shape", None)
            rows[name] = int(n[0]) if n is not None else int(len(mb))
        optimizer.step()
        return ClusterStepRecord(
            losses=losses,
            grad_norms=norms,
            rows=rows,
            skipped=tuple(n for n in self.names if n not in batches),
        )
