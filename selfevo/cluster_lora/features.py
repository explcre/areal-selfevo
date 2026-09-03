"""The MEDS behavioural feature, and where it can and cannot be got from.

WHERE THE LOGITS COME FROM -- the honest answer, checked against the code
=======================================================================

MEDS gets its feature for free because verl's actor asks for it in the pass it was already
running: ``recipe/meds/../dp_actor.py`` sets ``extra_args["output_hidden_states"] = True``
inside ``_forward_micro_batch``, so the log-prob pass returns every layer's residual stream
and ``extract_layer_logits_from_output`` reads the answer position out of it.

**AReaL's engine does not expose that, and the seam this project owns cannot reach it.**
Concretely, in ``areal/engine/fsdp_engine.py``:

* ``forward_backward_batch`` calls ``outputs = self.model(**inputs)`` with no
  ``output_hidden_states``, then keeps ``logits = outputs.logits.squeeze(0)`` and lets
  ``outputs`` -- and with it every intermediate hidden state -- fall out of scope at the end
  of the loop iteration.
* The only tensors that cross into selfevo territory are ``logprobs``, ``entropy`` and the
  per-token ``vocab_min/max_logits``. The per-LAYER logits need the residual stream at each
  layer, which is never materialised as an output at all.
* By the time ``PPOActor._compute_advantages`` runs -- where ``ClusterRouter.key_fn`` is
  reached -- the forward is over and the batch holds only ``advantages``, ``loss_mask``,
  ``logprobs`` and the rewards.

So: **the layer-wise logits are NOT reachable at the seam without either an upstream change
to the engine or an extra forward pass.** The upstream change is one line
(``output_hidden_states=True``) in ``areal/engine/fsdp_engine.py``, which this agent does not
own, and it is the expensive option anyway: retaining every layer for every token of a packed
microbatch is ``n_layers x tokens x hidden``, about 6.3 GB per 10240-token microbatch at 32B
in bf16, which is the OOM the project's own notes warn about for per-position caches.

This module therefore implements the EXTRA-FORWARD version, behind a flag, and makes it cheap
in the way the engine change cannot be:

* the forward is ``torch.no_grad`` and is truncated at the answer token, so no work is done
  on tokens after the position we read;
* the residual stream is captured by forward HOOKS at one position per sequence, so the
  retained memory is ``n_layers x hidden`` floats per sequence -- 64 x 5120 x 4 = 1.3 MB at
  32B -- instead of the whole activation stack;
* the per-layer logit is a dot product against one row of the unembedding rather than a full
  ``lm_head`` matmul, which is the same number (verified bit-identical in
  ``test_cluster_lora_features.py``) for 1/vocab of the arithmetic.

``mode="hidden_states"`` keeps the MEDS-faithful path -- ``output_hidden_states=True`` and a
full ``lm_head`` -- as the reference the hook path is checked against, because "the hooks
capture the same thing" is exactly the sort of claim that is true when written and false
after a transformers upgrade. The check is a test, not a comment: transformers builds
``hidden_states`` as ``(embeddings, layer_0_out, ..., layer_{N-2}_out, norm(layer_{N-1}_out))``
and MEDS drops the embedding entry, so the hook path must hook layers ``0..N-2`` AND the final
norm. Verified against transformers 5.3 on 2026-09-02.

MEASURED COST of the extra forward, and it is not free: one additional no-grad forward over
the batch truncated to the answer token. Against a training step (forward + backward, roughly
3x a forward) over the full sequence, that is well under a third of a step; the exact figure
needs a GPU and is recorded in ``selfevo/FINDINGS_cluster_lora.md`` as outstanding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "BehaviourFeatureUnavailable",
    "ClusterLoRAKeyFn",
    "LayerLogitExtractor",
    "answer_token_index",
    "meds_feature",
]


class BehaviourFeatureUnavailable(RuntimeError):
    """The behavioural feature could not be computed, with the reason stated.

    Raised rather than returning zeros. A zero vector is a valid input to HDBSCAN and would
    be clustered into a plausible-looking group, so a silent failure here does not surface
    as an error -- it surfaces as a cluster, which is the worst possible disguise.
    """


def answer_token_index(
    token_ids: Sequence[int],
    *,
    boxed_ids: Sequence[int] | None = None,
    strategy: str = "boxed",
    response_start: int = 0,
) -> int:
    """Position whose next token is the model's final answer token.

    MEDS reads the residual stream at the token emitting the first character inside
    ``\\boxed{``, i.e. the position of the ``{``, and unembeds against the token that
    actually followed. That choice matters: it is the one position where the whole
    trajectory has been committed to a single answer, so two rollouts that reasoned
    differently and concluded the same thing are close there while two that concluded
    differently are far apart -- which is the behaviour the clustering is supposed to see.

    Args:
        token_ids: The full sequence, prompt and response.
        boxed_ids: Token ids of ``"\\boxed{"`` under this tokenizer. Required for the
            ``boxed`` strategy; passed in rather than tokenized here so the tokenizer is
            not a dependency of this module.
        strategy: ``"boxed"`` (MEDS' own) or ``"last"``, which takes the final position and
            is what a task with no boxed answer has to fall back to.
        response_start: First response position. The search is restricted to the response,
            so a ``\\boxed{`` quoted in the PROMPT cannot be mistaken for the model's answer
            -- which on a few-shot math prompt it would be, every time.

    Returns:
        A position ``p`` with ``p + 1 < len(token_ids)``, so the next token exists.

    Raises:
        BehaviourFeatureUnavailable: If no usable position exists. Falling back to position
            0 would read the residual stream over the prompt's first token, which is
            identical for every rollout of a group and would collapse the whole group to one
            point.
        ValueError: On an unknown strategy, or a ``boxed`` strategy with no ``boxed_ids``.
    """
    n = len(token_ids)
    if n < 2:
        raise BehaviourFeatureUnavailable(
            f"sequence of length {n} has no position with a following token"
        )
    lo = max(0, int(response_start))
    if strategy == "last":
        p = n - 2
        if p < lo:
            raise BehaviourFeatureUnavailable(
                f"the response starts at {lo} but the sequence ends at {n}; there is no "
                "response position to read"
            )
        return p
    if strategy != "boxed":
        raise ValueError(f"unknown strategy {strategy!r}; expected 'boxed' or 'last'")
    if not boxed_ids:
        raise ValueError("strategy='boxed' needs boxed_ids for this tokenizer")
    L = len(boxed_ids)
    want = list(boxed_ids)
    ids = list(token_ids)
    for j in range(n - L, lo - 1, -1):
        if ids[j : j + L] == want:
            # MEDS' own convention: the LAST token of the "\boxed{" run, so the token that
            # follows is the first character of the answer itself.
            p = j + L - 1
            if p + 1 < n:
                return p
    raise BehaviourFeatureUnavailable(
        "no '\\boxed{' occurs in the response, so the answer token is undefined; pass "
        "strategy='last' to read the final position instead of guessing one"
    )


def meds_feature(
    layer_logits: Sequence[float],
    *,
    use_layer_diff: bool = False,
    last_n_layers: int | None = None,
) -> np.ndarray:
    """MEDS' post-processing of a per-layer logit trace into the clustering vector.

    Defaults follow the authors' ``run_meds.sh``: ``use_layer_diff=False`` and the last
    fourteen layers of a twenty-eight-layer model, i.e. the LATTER HALF. ``last_n_layers=None``
    reproduces that as a rule rather than a constant, so the same setting means the same
    thing on a model of a different depth.

    Args:
        layer_logits: One logit per layer, shallow to deep.
        use_layer_diff: Take first differences between consecutive layers instead of the
            raw trace. MEDS ships this off.
        last_n_layers: Keep this many trailing entries. ``None`` keeps the latter half.

    Returns:
        The clustering vector.

    Raises:
        BehaviourFeatureUnavailable: If the trace is too short to reduce, or holds a
            non-finite value.
    """
    vec = np.asarray(layer_logits, dtype=np.float64).ravel()
    if vec.size < 2:
        raise BehaviourFeatureUnavailable(
            f"a {vec.size}-layer trace cannot be reduced to a behavioural feature"
        )
    if not np.isfinite(vec).all():
        raise BehaviourFeatureUnavailable(
            "the layer-logit trace holds a non-finite value; it would cluster as a point"
        )
    if use_layer_diff:
        vec = np.diff(vec)
    keep = vec.size // 2 if last_n_layers is None else int(last_n_layers)
    if keep < 1:
        raise BehaviourFeatureUnavailable(
            f"last_n_layers={keep} keeps no layers, so the feature would be empty"
        )
    return vec[-keep:] if vec.size >= keep else vec


def _decoder_stack(model):
    """Find the decoder layer list, the final norm and the unembedding, through any wrapper.

    Written as a search rather than a fixed attribute path because the model arrives wrapped
    differently in every configuration this has to run in -- bare HF, PEFT, FSDP -- and a
    hardcoded ``model.model.layers`` works in exactly one of them and returns an
    AttributeError in the others, at which point the caller's only options are to guess or
    to skip the feature.

    Args:
        model: A causal LM, possibly wrapped.

    Returns:
        ``(layers, norm, unembed_weight)``.

    Raises:
        BehaviourFeatureUnavailable: If any of the three cannot be located.
    """
    unembed = None
    get_out = getattr(model, "get_output_embeddings", None)
    if callable(get_out):
        head = get_out()
        unembed = getattr(head, "weight", None)
    base = model
    for _ in range(6):
        if hasattr(base, "layers") and hasattr(base, "norm"):
            break
        nxt = getattr(base, "model", None) or getattr(base, "base_model", None)
        if nxt is None or nxt is base:
            break
        base = nxt
    layers = getattr(base, "layers", None)
    norm = getattr(base, "norm", None)
    if layers is None or norm is None or unembed is None:
        raise BehaviourFeatureUnavailable(
            f"could not locate the decoder stack on {type(model).__name__} "
            f"(layers={layers is not None}, norm={norm is not None}, "
            f"unembed={unembed is not None}); the layer-logit feature needs all three"
        )
    return layers, norm, unembed


@dataclass
class LayerLogitExtractor:
    """One extra no-grad forward that yields MEDS' per-layer logit trace per sequence.

    Args:
        mode: ``"hooks"`` captures one position per layer through forward hooks and keeps
            ``n_layers x hidden`` floats per sequence. ``"hidden_states"`` is the
            MEDS-faithful reference: it asks the model for every hidden state and gathers,
            which retains the whole activation stack and is there to be compared against,
            not to be run at scale.
        truncate: Cut each sequence at the answer position before the forward. Nothing after
            that position influences the value read at it -- the attention is causal -- so
            this is exact, not an approximation, and it is where most of the saving comes
            from on long rollouts.
        dot_unembed: Compute the per-layer logit as a dot product with one row of the
            unembedding instead of a full vocab matmul. Same number, 1/vocab of the work.

    Raises:
        ValueError: On an unknown mode.
    """

    mode: str = "hooks"
    truncate: bool = True
    dot_unembed: bool = True

    def __post_init__(self) -> None:
        if self.mode not in ("hooks", "hidden_states"):
            raise ValueError(
                f"unknown mode {self.mode!r}; expected 'hooks' or 'hidden_states'"
            )

    def trace(self, model, input_ids, position: int, target_id: int) -> np.ndarray:
        """Per-layer logit of ``target_id`` at ``position``, for one sequence.

        Args:
            model: The causal LM.
            input_ids: ``(1, T)`` or ``(T,)`` token ids.
            position: The position whose residual stream is read.
            target_id: The token whose logit is taken at each layer.

        Returns:
            ``(n_layers,)`` float64, shallow to deep -- the same entries MEDS keeps after
            dropping the embedding.

        Raises:
            BehaviourFeatureUnavailable: If the position is out of range for the sequence.
        """
        import torch

        ids = input_ids if input_ids.ndim == 2 else input_ids.unsqueeze(0)
        if ids.shape[0] != 1:
            raise ValueError(f"trace() takes one sequence, got {ids.shape[0]}")
        if not 0 <= position < ids.shape[1]:
            raise BehaviourFeatureUnavailable(
                f"position {position} is outside a sequence of length {ids.shape[1]}"
            )
        if self.truncate:
            ids = ids[:, : position + 1]
            pos = position
        else:
            pos = position
        layers, norm, unembed = _decoder_stack(model)
        row = unembed[target_id].detach()

        def to_logit(hidden):
            """Layer logit for the target token, as a dot product or a full matmul."""
            h = hidden.to(row.dtype)
            if self.dot_unembed:
                return float(h @ row)
            head = model.get_output_embeddings()
            return float(head(h)[target_id])

        with torch.no_grad():
            if self.mode == "hidden_states":
                out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
                # MEDS drops the embedding entry; what remains is one vector per layer.
                states = out.hidden_states[1:]
                return np.array(
                    [to_logit(s[0, pos, :]) for s in states], dtype=np.float64
                )
            captured: dict[int, "torch.Tensor"] = {}
            handles = []

            def make_hook(i):
                """Keep only the one position, so the rest of the activation is freed."""

                def hook(_mod, _inp, output):
                    t = output[0] if isinstance(output, tuple) else output
                    captured[i] = t[0, pos, :].detach().clone()

                return hook

            try:
                # Layers 0..N-2 only: transformers appends norm(layer_{N-1}_out) as the last
                # hidden state, never the raw output of the last layer, so hooking every
                # layer would produce an unnormalised final entry that MEDS never sees.
                for i, layer in enumerate(layers[:-1]):
                    handles.append(layer.register_forward_hook(make_hook(i)))
                handles.append(norm.register_forward_hook(make_hook(len(layers) - 1)))
                model(input_ids=ids, use_cache=False)
            finally:
                for h in handles:
                    h.remove()
            if len(captured) != len(layers):
                raise BehaviourFeatureUnavailable(
                    f"captured {len(captured)} of {len(layers)} layers; a hook did not fire, "
                    "which silently shortens the feature"
                )
            return np.array(
                [to_logit(captured[i]) for i in range(len(layers))], dtype=np.float64
            )


class ClusterLoRAKeyFn:
    """A ``key_fn`` for the existing ``ClusterRouter`` seam, backed by behavioural features.

    ``ClusterRouter.key_fn`` is ``Callable[[RoutingContext], str]`` and a
    ``RoutingContext.extra`` is ``Mapping[str, float]``, so a whole feature VECTOR cannot be
    carried through the context. The features are therefore supplied per batch through
    :meth:`begin_batch` and looked up by ``ctx.unit_id`` -- which the actor already sets to
    ``f"{step}:{i}"`` and which is unique across batches by construction.

    That makes this callable stateful, and stateful in the way the seam requires: the same
    instance must live across batches, because the MEDS labels are kNN-stabilised against
    the history and a fresh instance per batch would relabel everything every step. The
    router is cached on the actor for exactly this reason, so an instance handed to
    ``ClusterRouter(key_fn=...)`` inherits that lifetime.

    **A unit whose feature was never supplied raises.** It does not fall back to a default
    cluster: a default cluster is a real adapter that would receive that group's gradient,
    so the fallback would be indistinguishable from a deliberate assignment while being
    driven by a missing lookup.

    Args:
        partitioner: A :class:`selfevo.cluster_lora.partition.MEDSPartitioner`, or ``None``
            to build one with MEDS' defaults.
        mode: ``meds``, ``random_matched`` or ``none``, as in ``cluster_lora.partition``.
        seed: Seed for the size-matched control.

    Raises:
        ValueError: On an unknown mode.
    """

    def __init__(self, partitioner=None, *, mode: str = "meds", seed: int = 0) -> None:
        from .partition import TRAINING_PARTITIONS, MEDSPartitioner

        if mode not in TRAINING_PARTITIONS:
            raise ValueError(
                f"unknown mode {mode!r}; expected one of {list(TRAINING_PARTITIONS)}"
            )
        self.mode = mode
        self.seed = int(seed)
        self.partitioner = partitioner if partitioner is not None else MEDSPartitioner()
        self.partition = None
        self.previous = None
        self._keys: dict[str, str] = {}
        self.batches = 0

    def begin_batch(
        self,
        unit_ids: Sequence[str],
        features: np.ndarray,
        *,
        group_ids: Sequence[str] | None = None,
    ):
        """Partition this batch and arm the lookup ``__call__`` will use.

        Args:
            unit_ids: ``ctx.unit_id`` per group, in batch order.
            features: ``(n_groups, d)`` behavioural vectors, in the same order.
            group_ids: Stable prompt identities for churn measurement. Defaults to
                ``unit_ids``, which are unique per batch and therefore make churn
                unmeasurable -- reported as zero OVERLAP rather than as zero churn, so the
                difference is visible.

        Returns:
            The :class:`~selfevo.cluster_lora.partition.Partition` formed.

        Raises:
            ValueError: If the counts disagree, which would silently pair a group with
                another group's feature.
        """
        from .partition import partition_from_config

        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] != len(unit_ids):
            raise ValueError(
                f"{len(unit_ids)} unit ids but features of shape {features.shape}; a "
                "mismatch would assign one group's adapter from another group's behaviour"
            )
        self.previous = self.partition
        self.partition = partition_from_config(
            self.mode,
            n_groups=len(unit_ids),
            features=features,
            partitioner=self.partitioner,
            seed=self.seed + self.batches,
            group_ids=tuple(group_ids) if group_ids is not None else tuple(unit_ids),
        )
        self._keys = dict(zip(unit_ids, self.partition.keys))
        self.batches += 1
        return self.partition

    def __call__(self, ctx) -> str:
        """Adapter name for one routing context.

        Raises:
            BehaviourFeatureUnavailable: If no batch is armed, or this unit's feature was
                never supplied. Both are the same defect from the run's point of view -- a
                group about to be trained on an adapter chosen by accident -- and neither
                has a safe default.
        """
        if not self._keys:
            raise BehaviourFeatureUnavailable(
                "no batch is armed; call begin_batch(unit_ids, features) before routing, "
                "or the partition would be decided by whatever key happened to be cached"
            )
        key = self._keys.get(ctx.unit_id)
        if key is None:
            raise BehaviourFeatureUnavailable(
                f"no behavioural feature was supplied for unit {ctx.unit_id!r}; refusing "
                "rather than defaulting, because a default cluster is a real adapter that "
                "would receive this group's gradient"
            )
        return key

    def report(self):
        """This batch's reach and churn record.

        Returns:
            A :class:`selfevo.cluster_lora.reach.ReachReport`.

        Raises:
            BehaviourFeatureUnavailable: If no batch has been partitioned yet.
        """
        from .reach import reach_report

        if self.partition is None:
            raise BehaviourFeatureUnavailable("no batch has been partitioned yet")
        return reach_report(self.partition, self.previous)
