#!/usr/bin/env python3
"""Dump per-GROUP LoRA gradient sketches from a checkpoint and a batch of rollouts.

This is the GPU half of the interference probe, and it deliberately knows nothing about
clusters. It computes, for each GRPO group in one batch:

* a linear sketch of the gradient of that group's GRPO loss w.r.t. the LoRA parameters;
* a linear sketch of the gradient of the NEGATIVE LOG-LIKELIHOOD of the PROMPT tokens only,
  which is the ELREA (arXiv 2502.00089) feature -- ELREA clusters instructions on
  prompt-token gradients and already does cluster -> per-cluster LoRA -> merge in SFT, so
  "are behavioural clusters different from prompt-gradient clusters?" is the ablation that
  decides whether the rollouts are needed at all;
* the MEDS behavioural vector (latter-half layer-wise logits at the answer token);
* the group's task label, size and mean reward;
* and, for the first few groups, the FULL unprojected gradient, so the sketch can be
  validated instead of assumed.

Clustering happens in ``interference_analyze.py``, on CPU, from this file alone. The split
is not organisational: a cluster's gradient is the SUM of its members' gradients when every
group's loss carries the same denominator, and the sketch is linear, so **every partition is
free once the dump exists**. Four partitions, one pass over the batch. It also puts the
scikit-learn and hdbscan dependency on the analysis side, which matters because the venv that
runs training has neither and must not acquire them.

WHY THE PROMPT-TOKEN MASKING IS DONE HERE. The trainer's loss path cannot produce it: the
GRPO loss is masked to RESPONSE tokens, so its gradient restricted to prompt positions is
identically zero and clustering on it would cluster noise. The ELREA feature is a different
loss -- plain next-token NLL on the prompt -- so it is built in this script rather than by
touching the trainer. Cost: one extra backward per group, i.e. the dump does two backward
passes per group instead of one.

Usage::

    python -m selfevo.cluster_lora.interference_dump \
        --model /path/to/hf/checkpoint --rollouts batch.jsonl --out dump.npz \
        [--adapter /path/to/trained/lora] [--sketch-dim 8192] [--full-grad-groups 8]

Runs under the TRAINING venv (torch, transformers, peft). It needs neither scikit-learn nor
hdbscan, by design.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .features import LayerLogitExtractor, answer_token_index, meds_feature
from .sketch import SketchPlan, sketch_torch

__all__ = [
    "DumpConfig",
    "Group",
    "LOGIT_PEAK_BYTES_PER_ELEMENT",
    "LogitsBudgetExceeded",
    "assert_logits_fit",
    "chunk_tokens_for_budget",
    "group_backward",
    "group_losses",
    "load_rollouts",
    "logits_peak_bytes",
    "run_dump",
]

#: Transient bytes per (token, vocab-entry) inside the chunked unembedding, DERIVED not
#: measured, and deliberately an upper bound.
#:
#: Per chunk the head holds the bf16 matmul output (2), the fp32 log-softmax output (4), and
#: during the recomputed backward the gradient of that output (4) and of the logits (2):
#: 12 bytes per element. It is exposed as ``--logit-peak-bytes`` so a box that measures
#: something different can say so rather than work around the constant.
#:
#: The ORIGINAL unchunked path cost 10 bytes per element over the WHOLE group at once
#: (bf16 logits 2, the ``.float()`` copy 4 with the bf16 node still alive, and the
#: log-softmax output 4), which is 14.20 GB for the largest group of the probe batch at
#: vocab 152,064 -- against 61.02 GB of resident weights on an 80 GiB card. That is the OOM
#: this constant exists to make predictable.
LOGIT_PEAK_BYTES_PER_ELEMENT = 12

#: Default ceiling on the TRUNK's retained activations for one forward.
#:
#: Under gradient checkpointing the decoder keeps one hidden-state tensor per layer, so the
#: cost is ``layers x tokens x hidden x 2`` bytes -- for a 64-layer 5120-wide model that is
#: 655 KB per token, and the probe batch's worst group of 16,712 padded tokens would retain
#: 10.95 GB on top of 61.02 GB of weights. Fixing the unembedding alone leaves that as the new
#: binding constraint, which is why the trunk is sub-batched over SEQUENCES as well.
DEFAULT_ACTIVATION_BUDGET_BYTES = 6 * 1024**3

#: Default ceiling on the chunked head's transient memory. At vocab 152,064 this gives a
#: chunk of about 2,190 tokens, inside the 2,048-4,096 band that the measured length
#: distribution of the probe batch calls for: p90 is 9,320 padded tokens per group and the
#: max is 16,712, so a chunk near p90 would leave roughly the ~10 GB headroom at which the
#: run died, while this leaves the binding constraint on activations instead.
DEFAULT_LOGITS_BUDGET_BYTES = 4 * 1024**3


class LogitsBudgetExceeded(RuntimeError):
    """One group's unembedding would not fit, refused BEFORE the forward runs.

    ``--max-full-grad-gb`` already guards the full-gradient STORE, but the store was never
    the binding constraint: at 32B the per-group LOGITS are an order of magnitude larger,
    and the process died inside the LM head where no guard could see it. This is the same
    discipline applied where it actually binds, and it names the group so a refusal is
    actionable rather than a stack trace.
    """


def logits_peak_bytes(
    chunk_tokens: int, vocab: int, *, peak_bytes: int = LOGIT_PEAK_BYTES_PER_ELEMENT
) -> int:
    """Transient bytes the chunked unembedding needs for one chunk.

    Args:
        chunk_tokens: Positions evaluated per chunk. This is a count of TOKENS, never of
            sequences: the probe batch's worst group is one 1,330-token prompt repeated
            across all eight samples, so a sequence-count chunk of 1 still allocates
            2,089 tokens' worth while a token-count chunk is uniform across groups.
        vocab: Vocabulary size.
        peak_bytes: Bytes per (token, vocab-entry); see
            :data:`LOGIT_PEAK_BYTES_PER_ELEMENT`.

    Returns:
        Estimated peak bytes.

    Raises:
        ValueError: On a non-positive argument.
    """
    if chunk_tokens <= 0 or vocab <= 0 or peak_bytes <= 0:
        raise ValueError(
            f"chunk_tokens, vocab and peak_bytes must all be positive; got "
            f"{chunk_tokens}, {vocab}, {peak_bytes}"
        )
    return int(chunk_tokens) * int(vocab) * int(peak_bytes)


def chunk_tokens_for_budget(
    vocab: int,
    budget_bytes: int = DEFAULT_LOGITS_BUDGET_BYTES,
    *,
    peak_bytes: int = LOGIT_PEAK_BYTES_PER_ELEMENT,
    cap: int | None = None,
) -> int:
    """Largest chunk whose unembedding fits the budget, capped at the work available.

    Derived from the budget rather than fixed, so the same setting means the same MEMORY on
    a model of a different vocabulary -- a token count that is comfortable at vocab 32k is
    five times the intended footprint at 152k.

    Args:
        vocab: Vocabulary size.
        budget_bytes: Ceiling on the chunked head's transient memory.
        peak_bytes: Bytes per (token, vocab-entry).
        cap: Never return more than this; pass the group's token count so a small group is
            done in one chunk instead of padding the plan.

    Returns:
        A chunk size of at least 1.

    Raises:
        LogitsBudgetExceeded: If even a single token exceeds the budget, which no chunking
            can fix and which therefore has to be said rather than silently rounded up to 1.
    """
    per_token = int(vocab) * int(peak_bytes)
    n = int(budget_bytes) // per_token
    if n < 1:
        raise LogitsBudgetExceeded(
            f"a single token needs {per_token / 1e9:.2f} GB at vocab {vocab}, over the "
            f"{budget_bytes / 1e9:.2f} GB logits budget; raise --logits-budget-gb, because "
            "no chunk size can bring this under the limit"
        )
    if cap is not None:
        n = min(n, max(1, int(cap)))
    return int(n)


def forward_tokens_for_budget(
    n_layers: int, hidden: int, budget_bytes: int = DEFAULT_ACTIVATION_BUDGET_BYTES
) -> int:
    """Tokens one trunk forward may carry before its retained activations exceed the budget.

    Args:
        n_layers: Decoder layers.
        hidden: Model width.
        budget_bytes: Ceiling on retained activations.

    Returns:
        A token count of at least 1. Sequences are independent under causal attention with
        right padding, so splitting a group across several forwards changes nothing about
        the gradient -- the loss is a sum over sequences and the sum is reassociated.

    Raises:
        ValueError: On a non-positive argument.
    """
    if n_layers <= 0 or hidden <= 0 or budget_bytes <= 0:
        raise ValueError(
            f"n_layers, hidden and budget_bytes must be positive; got {n_layers}, "
            f"{hidden}, {budget_bytes}"
        )
    per_token = int(n_layers) * int(hidden) * 2
    return max(1, int(budget_bytes) // per_token)


def assert_logits_fit(
    *,
    group_id: str,
    n_tokens: int,
    vocab: int,
    chunk_tokens: int,
    budget_bytes: int = DEFAULT_LOGITS_BUDGET_BYTES,
    free_bytes: int | None = None,
    headroom: float = 1.5,
    peak_bytes: int = LOGIT_PEAK_BYTES_PER_ELEMENT,
) -> dict:
    """Refuse a group whose unembedding will not fit, BEFORE any forward runs.

    Args:
        group_id: Named in the refusal, so the offending group is identified rather than
            inferred from how far the run got. The probe batch's worst group is id 11 of
            128 in file order, so a guard that fires does so in the first minute.
        n_tokens: Padded tokens in this group, i.e. ``B * T``.
        vocab: Vocabulary size.
        chunk_tokens: Planned chunk.
        budget_bytes: Ceiling on the chunked head's transient memory.
        free_bytes: Free device memory, or ``None`` to skip the device check (CPU, or a
            backend that cannot report it). The budget check still runs.
        headroom: Multiple of the estimate that must be free. Above 1 because the estimate
            covers the head only: the decoder's own activations share the same pool, and a
            guard that let the head fit exactly would pass and then die one layer later.

    Returns:
        The budget record, which the dump writes into its metadata so a completed run can be
        read against the plan it ran under.

    Raises:
        LogitsBudgetExceeded: If the chunk exceeds the budget, or the estimate plus headroom
            exceeds free device memory.
    """
    peak = logits_peak_bytes(chunk_tokens, vocab, peak_bytes=peak_bytes)
    record = {
        "group_id": group_id,
        "n_tokens": int(n_tokens),
        "vocab": int(vocab),
        "chunk_tokens": int(chunk_tokens),
        "chunk_peak_bytes": int(peak),
        "unchunked_peak_bytes": int(
            logits_peak_bytes(max(1, n_tokens), vocab, peak_bytes=peak_bytes)
        ),
        "budget_bytes": int(budget_bytes),
        "free_bytes": None if free_bytes is None else int(free_bytes),
    }
    if peak > budget_bytes:
        raise LogitsBudgetExceeded(
            f"group {group_id!r}: a chunk of {chunk_tokens} tokens at vocab {vocab} needs "
            f"{peak / 1e9:.2f} GB, over the {budget_bytes / 1e9:.2f} GB logits budget. "
            "Lower --chunk-tokens or raise --logits-budget-gb"
        )
    if free_bytes is not None and peak * headroom > free_bytes:
        raise LogitsBudgetExceeded(
            f"group {group_id!r} has {n_tokens} padded tokens at vocab {vocab}; the chunked "
            f"unembedding needs {peak / 1e9:.2f} GB and {headroom}x that must be free for "
            f"the decoder's own activations, but only {free_bytes / 1e9:.2f} GB is. "
            f"Unchunked this group would have needed "
            f"{record['unchunked_peak_bytes'] / 1e9:.2f} GB. Lower --chunk-tokens, raise "
            "--logits-budget-gb, or give the probe a card with more free memory"
        )
    return record


class RolloutSchemaError(ValueError):
    """A rollout record is missing something the probe cannot invent.

    Raised per record with the offending key named. Silently skipping malformed rollouts
    would shrink the batch, and a batch that quietly lost its hard groups is exactly the
    batch on which conflict looks small.
    """


@dataclass
class DumpConfig:
    """Everything the dump needs, in one object so a run can be recorded verbatim.

    Args:
        model: HF checkpoint path or hub id.
        rollouts: JSONL of rollout samples.
        out: Output ``.npz``.
        adapter: Optional trained LoRA to load. **Strongly recommended.** At a fresh LoRA
            init ``B = 0``, so ``dL/dA = B^T (...) = 0`` exactly and half of every gradient
            vanishes; the cosines are then taken over the ``B`` blocks alone. That is still
            a real gradient, but it is not the gradient of a model mid-training, and the
            dump records ``zero_block_fraction`` so the degenerate case cannot be mistaken
            for a measurement.
        sketch_dim: CountSketch dimension. The resolution floor is about ``3/sqrt(dim)``;
            at 8192 that is 0.033, so a published cross-task cosine of ~1e-5 cannot be
            confirmed from sketches at this width and is reported as "below the floor"
            instead.
        sketch_seed: Shared across every group, necessarily -- two groups sketched under
            different hashes have cosine ~0 whatever their gradients did.
        full_grad_groups: Groups whose FULL gradient is stored, for validating the sketch.
        max_full_grad_gb: Refuse rather than write a dump larger than this in full
            gradients. The refusal names the size, so the caller lowers the count on
            purpose instead of discovering it after an hour on eight GPUs.
        answer_strategy: ``boxed`` (MEDS' own) or ``last``.
        group_feature_agg: How per-sample behaviour becomes one vector per group. ``mean``
            averages the samples. Recorded because a group with mixed correctness has two
            behaviours in it and their mean is neither -- a limitation of routing at group
            granularity, not of the feature.
        lora_rank / lora_alpha / target_modules: Used only when ``adapter`` is absent.
        dtype: Compute dtype.
        max_len: Truncate sequences to this many tokens.
        device: Torch device.
    """

    model: str
    rollouts: str
    out: str
    adapter: str | None = None
    sketch_dim: int = 8192
    sketch_seed: int = 0
    full_grad_groups: int = 8
    max_full_grad_gb: float = 8.0
    answer_strategy: str = "boxed"
    group_feature_agg: str = "mean"
    lora_rank: int = 16
    lora_alpha: int = 16
    target_modules: tuple[str, ...] = ("all-linear",)
    dtype: str = "bfloat16"
    max_len: int = 4096
    device: str = "cuda"
    use_layer_diff: bool = False
    last_n_layers: int | None = None
    extractor_mode: str = "hooks"
    group_key: str | None = None
    task_key: str | None = None
    reward_key: str | None = None
    chunk_tokens: int | None = None
    seq_chunk: int | None = None
    activation_budget_gb: float = 6.0
    logits_budget_gb: float = 4.0
    logit_peak_bytes: int = LOGIT_PEAK_BYTES_PER_ELEMENT
    use_checkpoint: bool = True
    gradient_checkpointing: bool = True


@dataclass
class Group:
    """One GRPO group: a prompt and the samples drawn for it.

    Args:
        group_id: Stable prompt identity. Used as the row key in the dump and as the churn
            key in training, so it must be the PROMPT's id and never a batch position.
        task: Task label, for the cross-task calibration partition.
        prompt_ids: The prompt's token ids, shared by every sample.
        response_ids: One response per sample.
        rewards: One scalar per sample.
    """

    group_id: str
    task: str
    prompt_ids: list[int]
    response_ids: list[list[int]]
    rewards: list[float]

    @property
    def size(self) -> int:
        """Samples in this group."""
        return len(self.response_ids)

    def advantages(self) -> np.ndarray:
        """GRPO advantages: rewards centred and scaled within the group.

        Matches the live configuration (``reward_norm`` with ``mean_level`` and
        ``std_level`` both ``group``). A group whose samples all score alike therefore has
        advantages identically zero and contributes NO gradient -- which is not a defect to
        paper over here, it is the measured 29-44% of groups this project already tracks,
        and such a group must show up in the dump as a zero-gradient group rather than be
        rescued by a different normalisation.
        """
        r = np.asarray(self.rewards, dtype=np.float64)
        centred = r - r.mean()
        sd = r.std()
        return centred / sd if sd > 0 else centred


def _first_key(rec: dict, names: Sequence[str]):
    """First of ``names`` present in ``rec``, or ``None``.

    Rollout dumps in this project have appeared under several field names, and a loader that
    insisted on one would mean a conversion step -- which is another place rows can be
    dropped silently. Unknown extra fields (lengths, truncation flags) are ignored rather
    than rejected.
    """
    for n in names:
        if n in rec and rec[n] is not None:
            return n
    return None


#: Field names accepted for each role, in priority order.
GROUP_KEYS = ("group_id", "prompt_id", "uid", "qid", "index", "idx")
TASK_KEYS = ("task", "task_name", "dataset", "source", "data_source")
PROMPT_TEXT_KEYS = ("prompt", "question", "query")
PROMPT_ID_KEYS = ("prompt_ids", "prompt_token_ids")
RESPONSES_KEYS = ("responses", "completions", "outputs", "generations")
RESPONSE_ID_LIST_KEYS = ("responses_ids", "response_token_ids", "responses_token_ids")
RESPONSE_TEXT_KEYS = ("response", "completion", "output", "generation")
RESPONSE_ID_KEYS = ("response_ids", "response_token_ids")
REWARDS_KEYS = ("rewards", "scores", "accs", "acc")
REWARD_KEYS = ("reward", "score", "correct")


def load_rollouts(
    path: str,
    *,
    tokenizer,
    max_len: int = 4096,
    group_key: str | None = None,
    task_key: str | None = None,
    reward_key: str | None = None,
) -> list[Group]:
    """Read a JSONL of rollouts into groups, accepting either record shape.

    Two shapes occur, and both are supported because requiring one means a conversion step
    that can itself drop rows:

    * **per sample** -- one line per rollout, with ``response``/``response_ids`` and a scalar
      ``reward``. Lines sharing a group id are gathered into one group.
    * **per group** -- one line per prompt, with ``responses``/``response_ids`` as a LIST and
      ``rewards`` as a list of the same length. This is the shape the harness writes.

    Token ids are accepted in place of text in both. Extra fields -- lengths, truncation
    flags -- are ignored rather than rejected.

    Args:
        path: JSONL file.
        tokenizer: Used only where text has to be encoded.
        max_len: Truncation budget for prompt + response.
        group_key: Field holding prompt identity. ``None`` searches
            :data:`GROUP_KEYS` and falls back to a hash of the prompt tokens. The fallback
            is the one place two genuinely different prompts could merge, so it is used only
            when no id field exists at all.
        task_key: Field holding the task label. ``None`` searches :data:`TASK_KEYS`; absent
            means ``"unknown"``, which makes the cross-task partition REFUSE rather than
            invent a split.
        reward_key: Field holding the reward. ``None`` searches the known names.

    Returns:
        Groups in first-appearance order.

    Raises:
        RolloutSchemaError: On a record missing a prompt, a response or a reward, naming the
            file, the line and the names that were looked for. Skipping malformed rollouts
            would shrink the batch, and a batch that quietly lost its hard groups is exactly
            the batch on which conflict looks small.
    """
    import hashlib

    buckets: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            where = f"{path}:{lineno}"

            k = _first_key(rec, PROMPT_ID_KEYS)
            if k is not None:
                p_ids = list(rec[k])
            else:
                k = _first_key(rec, PROMPT_TEXT_KEYS)
                if k is None:
                    raise RolloutSchemaError(
                        f"{where}: no prompt field. Looked for {list(PROMPT_ID_KEYS)} and "
                        f"{list(PROMPT_TEXT_KEYS)}; the record has {sorted(rec)}"
                    )
                p_ids = tokenizer.encode(rec[k], add_special_tokens=False)

            # Group-shaped first: a list of responses on one line.
            resp_list: list[list[int]] | None = None
            k = _first_key(rec, RESPONSE_ID_LIST_KEYS)
            if k is not None:
                resp_list = [list(r) for r in rec[k]]
            else:
                k = _first_key(rec, RESPONSES_KEYS)
                if k is not None:
                    resp_list = [
                        list(r) if isinstance(r, (list, tuple))
                        else tokenizer.encode(
                            r if isinstance(r, str) else r.get("text", ""),
                            add_special_tokens=False,
                        )
                        for r in rec[k]
                    ]
            if resp_list is None:
                k = _first_key(rec, RESPONSE_ID_KEYS)
                if k is not None:
                    resp_list = [list(rec[k])]
                else:
                    k = _first_key(rec, RESPONSE_TEXT_KEYS)
                    if k is None:
                        raise RolloutSchemaError(
                            f"{where}: no response field. Looked for "
                            f"{list(RESPONSES_KEYS)}, {list(RESPONSE_ID_LIST_KEYS)}, "
                            f"{list(RESPONSE_TEXT_KEYS)} and {list(RESPONSE_ID_KEYS)}; the "
                            f"record has {sorted(rec)}"
                        )
                    resp_list = [tokenizer.encode(rec[k], add_special_tokens=False)]

            rk = reward_key or _first_key(rec, REWARDS_KEYS) or _first_key(rec, REWARD_KEYS)
            if rk is None or rk not in rec:
                raise RolloutSchemaError(
                    f"{where}: no reward field. Looked for {list(REWARDS_KEYS)} and "
                    f"{list(REWARD_KEYS)}; the record has {sorted(rec)}. GRPO advantages "
                    "cannot be formed without one, and a default would fabricate the very "
                    "signal under test"
                )
            raw_reward = rec[rk]
            rewards = (
                [float(x) for x in raw_reward]
                if isinstance(raw_reward, (list, tuple))
                else [float(raw_reward)] * len(resp_list)
            )
            if len(rewards) != len(resp_list):
                raise RolloutSchemaError(
                    f"{where}: {len(rewards)} rewards for {len(resp_list)} responses; a "
                    "mismatch would score one rollout with another rollout's reward"
                )

            gk = group_key or _first_key(rec, GROUP_KEYS)
            gid = (
                str(rec[gk]) if gk is not None and gk in rec
                else hashlib.blake2b(json.dumps(p_ids).encode(), digest_size=8).hexdigest()
            )
            tk = task_key or _first_key(rec, TASK_KEYS)
            task = str(rec[tk]) if tk is not None and tk in rec else "unknown"

            if gid not in buckets:
                buckets[gid] = {"task": task, "prompt": p_ids, "resp": [], "rew": []}
                order.append(gid)
            budget = max(1, max_len - len(p_ids))
            for r, w in zip(resp_list, rewards):
                buckets[gid]["resp"].append(list(r)[:budget])
                buckets[gid]["rew"].append(float(w))

    groups = [
        Group(
            group_id=gid,
            task=buckets[gid]["task"],
            prompt_ids=buckets[gid]["prompt"],
            response_ids=buckets[gid]["resp"],
            rewards=buckets[gid]["rew"],
        )
        for gid in order
    ]
    if not groups:
        raise RolloutSchemaError(f"{path} yielded no groups")
    return groups


def _decoder_and_unembedding(model):
    """The transformer trunk and the unembedding weight, through any wrapper.

    The trunk is called DIRECTLY rather than through the causal-LM wrapper, and that is the
    whole fix: the wrapper's forward ends in a full-vocabulary ``lm_head`` over every
    position at once, which at 32B and vocab 152,064 is 4.73 GB in bf16 for the probe
    batch's worst group and 14.20 GB once the fp32 copy the old code took is counted --
    against 61.02 GB of resident weights on an 80 GiB card. Calling the trunk yields hidden
    states of ``B x T x 5120`` instead, 171 MB for the same group, and the unembedding is
    then applied in chunks that are freed as they go.

    LoRA still applies. PEFT replaces the Linear MODULES in place, so the adapters are
    inside the trunk and are reached whichever forward calls them; only prompt-learning
    methods live on the wrapper, and this method is LoRA. Asserted rather than assumed by
    ``test_the_chunked_path_still_reaches_every_lora_parameter``.

    Args:
        model: A causal LM, possibly PEFT-wrapped.

    Returns:
        ``(decoder_module, unembedding_weight)``.

    Raises:
        RuntimeError: If either cannot be located. Falling back to the wrapper's forward
            would silently restore the allocation this exists to avoid.
    """
    decoder = None
    getter = getattr(model, "get_decoder", None)
    if callable(getter):
        try:
            decoder = getter()
        except Exception:  # pragma: no cover - only on exotic wrappers
            decoder = None
    if decoder is None:
        base = model
        for _ in range(6):
            if hasattr(base, "layers") and hasattr(base, "norm"):
                decoder = base
                break
            nxt = getattr(base, "model", None) or getattr(base, "base_model", None)
            if nxt is None or nxt is base:
                break
            base = nxt
    head = None
    get_out = getattr(model, "get_output_embeddings", None)
    if callable(get_out):
        head = get_out()
    unembed = getattr(head, "weight", None)
    if decoder is None or unembed is None:
        raise RuntimeError(
            f"could not locate the decoder trunk and unembedding on "
            f"{type(model).__name__} (decoder={decoder is not None}, "
            f"unembed={unembed is not None}); the chunked head needs both, and falling back "
            "to the wrapper's forward would restore the full-vocabulary allocation that "
            "OOMs at 32B"
        )
    return decoder, unembed


def _chunk_weighted_logp(hidden, targets, weights, unembed):
    """Weighted sums of the sampled tokens' log-probabilities, for ONE chunk.

    The full-vocabulary tensor exists only inside this function and only for ``chunk`` rows.
    ``log_softmax(..., dtype=torch.float32)`` performs the reduction in fp32 WITHOUT first
    materialising an fp32 copy of the logits, which is what the previous ``.float()`` did --
    that copy kept the bf16 node alive alongside it and was half the peak.

    Args:
        hidden: ``(chunk, H)`` residual stream at the emitting positions.
        targets: ``(chunk,)`` token actually emitted at each position.
        weights: Sequence of ``(chunk,)`` weight vectors, one per loss. Both losses share
            this pass because they read the same log-probabilities; only the weights differ.
        unembed: ``(V, H)`` unembedding.

    Returns:
        A ``(len(weights),)`` tensor of weighted sums.
    """
    import torch
    import torch.nn.functional as F

    logits = F.linear(hidden, unembed)
    logp = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
    sel = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
    return torch.stack([(w * sel).sum() for w in weights])


def _weighted_logp_sums(hidden, targets, weights, unembed, *, chunk_tokens, use_checkpoint):
    """Sum :func:`_chunk_weighted_logp` over chunks of POSITIONS.

    Chunked on tokens rather than on sequences because the probe batch's spread is entirely
    in sequence LENGTH: every group has eight samples, and the worst group is a single
    1,330-token prompt that is prepended to all eight, so a chunk of one SEQUENCE still
    allocates 2,089 tokens' worth there while being three times smaller than the median
    group elsewhere. A token chunk is uniform across groups; a sequence chunk is not.

    The sum over chunks is exactly the sum over positions -- the loss is linear in the
    per-token terms -- so the gradient is unchanged. ``use_checkpoint`` recomputes each
    chunk's logits during the backward instead of retaining them, which trades one extra
    unembedding matmul per chunk for keeping only ONE chunk's logits alive at a time. That
    is exact for a deterministic forward, which :func:`_assert_deterministic_forward`
    establishes rather than assumes.

    Args:
        hidden: ``(N, H)`` flattened emitting positions.
        targets: ``(N,)`` emitted tokens.
        weights: Sequence of ``(N,)`` weight vectors.
        unembed: ``(V, H)`` unembedding.
        chunk_tokens: Positions per chunk.
        use_checkpoint: Recompute chunk logits in the backward pass.

    Returns:
        A ``(len(weights),)`` tensor of weighted sums over every position.
    """
    import torch
    import torch.utils.checkpoint as ckpt

    n = hidden.shape[0]
    chunk = max(1, min(int(chunk_tokens), n))
    # Checkpointing a chunk whose input does not require grad produces no gradient and only a
    # warning, so the flag is honoured only where it can mean something.
    do_ckpt = bool(use_checkpoint) and bool(hidden.requires_grad)
    total = None
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        h, t = hidden[start:stop], targets[start:stop]
        w = [x[start:stop] for x in weights]

        def run(hc, tc, *wc):
            """Bound so ``unembed`` -- which is frozen and needs no grad -- is not a checkpoint input."""
            return _chunk_weighted_logp(hc, tc, wc, unembed)

        out = ckpt.checkpoint(run, h, t, *w, use_reentrant=False) if do_ckpt else run(h, t, *w)
        total = out if total is None else total + out
    if total is None:  # pragma: no cover - an empty group is refused upstream
        raise ValueError("no positions to score; the group has no tokens")
    return total


def group_losses(
    model,
    group: Group,
    *,
    device,
    token_denominator: int,
    prompt_denominator: int,
    chunk_tokens: int | None = None,
    logits_budget_bytes: int = DEFAULT_LOGITS_BUDGET_BYTES,
    peak_bytes_per_element: int = LOGIT_PEAK_BYTES_PER_ELEMENT,
    use_checkpoint: bool = True,
    free_bytes: int | None = None,
    _advantages=None,
):
    """The two per-group losses whose gradients the dump sketches.

    **The denominators are global, not per-group, and that is the correctness condition for
    everything downstream.** With a shared denominator the batch loss is exactly the sum of
    the per-group losses, so the batch gradient is exactly the sum of the per-group
    gradients, so a cluster's gradient is exactly the sum of its members' -- which is what
    lets the analysis form any partition from this dump. Normalising per group would make
    every group's loss carry a different scale and the sums would describe nothing.

    The GRPO surrogate is evaluated ON-POLICY, where the importance ratio is exactly 1 and
    the clip is inert. That is not a simplification of this project's runs; it is what they
    measured -- ``importance_weight`` avg=min=max=1.0 and ``clip_ratio`` 0.0 at every step
    of four runs -- so the surrogate's gradient here is ``-sum_t A_t * dlogp_t``, the same
    one the trainer produces.

    **The unembedding is applied in chunks and the trunk is called directly.** The value and
    the gradient are unchanged -- the loss is a sum over positions and the sum is
    reassociated, nothing more -- but the full-vocabulary tensor now exists for one chunk at
    a time instead of for the whole group. The equivalence is a test
    (``test_the_chunked_path_is_numerically_identical_to_the_unchunked_one``), not a claim,
    because "same maths, less memory" is exactly the sort of refactor that silently is not.

    Args:
        model: The (PEFT-wrapped) causal LM.
        group: The group.
        device: Torch device.
        token_denominator: Total RESPONSE tokens in the batch.
        prompt_denominator: Total PROMPT tokens in the batch.
        chunk_tokens: Positions per unembedding chunk. ``None`` derives it from
            ``logits_budget_bytes`` and the model's vocabulary.
        logits_budget_bytes: Ceiling on the chunked head's transient memory.
        peak_bytes_per_element: Bytes per (token, vocab-entry) for the estimate.
        use_checkpoint: Recompute chunk logits in the backward pass.
        free_bytes: Free device memory for the pre-forward guard, or ``None`` to skip it.

    Returns:
        ``(grpo_loss, prompt_nll_loss)``, both scalars with grad.

    Raises:
        ValueError: If a denominator is zero, which would divide the whole measurement by
            nothing rather than by the batch.
        LogitsBudgetExceeded: If this group will not fit. Raised BEFORE the forward, naming
            the group and the estimate, rather than dying inside the LM head.
    """
    import torch

    if token_denominator <= 0 or prompt_denominator <= 0:
        raise ValueError(
            f"denominators must be positive, got response={token_denominator} "
            f"prompt={prompt_denominator}"
        )
    decoder, unembed = _decoder_and_unembedding(model)
    vocab = int(unembed.shape[0])

    # ``_advantages`` is how :func:`group_backward` hands a sub-batch the advantages its
    # PARENT group defines. Recomputing them from the slice would centre the rewards within
    # the slice, which is a different measurement -- and one that would still run.
    adv = group.advantages() if _advantages is None else _advantages
    n_prompt = len(group.prompt_ids)
    seqs = [group.prompt_ids + r for r in group.response_ids]
    width = max(len(s) for s in seqs)

    # BEFORE the forward, which is the whole point: the old code died inside the LM head,
    # where no guard could see it and no message could name the group.
    plan = chunk_tokens if chunk_tokens is not None else chunk_tokens_for_budget(
        vocab, logits_budget_bytes, peak_bytes=peak_bytes_per_element,
        cap=len(seqs) * max(1, width - 1),
    )
    assert_logits_fit(
        group_id=group.group_id, n_tokens=len(seqs) * width, vocab=vocab,
        chunk_tokens=plan, budget_bytes=logits_budget_bytes, free_bytes=free_bytes,
        peak_bytes=peak_bytes_per_element,
    )

    ids = torch.full((len(seqs), width), 0, dtype=torch.long)
    attn = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, : len(s)] = 1
    ids, attn = ids.to(device), attn.to(device)

    hidden = decoder(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state
    target = ids[:, 1:]
    valid = attn[:, 1:].bool()

    pos = torch.arange(width - 1, device=device).unsqueeze(0)
    # Position t predicts token t+1, so the RESPONSE region in emitter coordinates starts at
    # n_prompt - 1: that position emits the response's first token. Off by one here would
    # put the prompt's last token into the RL loss and drop the response's first, which is
    # the same off-by-one this project has already recorded once in its token router.
    resp_mask = valid & (pos >= n_prompt - 1)
    prompt_mask = valid & (pos < n_prompt - 1)

    a = torch.tensor(adv, dtype=torch.float32, device=device).unsqueeze(1)
    w_grpo = (a * resp_mask).reshape(-1)
    w_nll = prompt_mask.to(torch.float32).reshape(-1)

    sums = _weighted_logp_sums(
        hidden[:, :-1, :].reshape(-1, hidden.shape[-1]),
        target.reshape(-1),
        (w_grpo, w_nll),
        unembed,
        chunk_tokens=plan,
        use_checkpoint=use_checkpoint,
    )
    grpo = -sums[0] / token_denominator
    prompt_nll = -sums[1] / prompt_denominator
    return grpo, prompt_nll


def group_backward(
    model,
    group: Group,
    *,
    which: str,
    device,
    token_denominator: int,
    prompt_denominator: int,
    seq_chunk: int | None = None,
    activation_budget_bytes: int = DEFAULT_ACTIVATION_BUDGET_BYTES,
    **loss_kw,
) -> float:
    """Backward one group's loss, sub-batching the trunk over SEQUENCES.

    The sequences of a GRPO group are independent -- causal attention with right padding
    gives no path between them -- and the loss is a sum over them under a denominator that
    does not depend on the split. So ``sum_s L_s`` has gradient ``sum_s grad L_s``, and
    backwarding each sub-batch immediately accumulates exactly the same ``.grad`` as one
    backward over the whole group while freeing each sub-batch's activations first.

    That is the second half of the OOM fix and it is needed for the same reason as the first.
    Chunking the unembedding removes 14.20 GB, but the trunk still retains one hidden-state
    tensor per layer under gradient checkpointing: 64 x 16,712 x 5120 x 2 = 10.95 GB for the
    probe batch's worst group, on top of 61.02 GB of weights on an 80 GiB card. Sub-batching
    that group into two forwards brings it to about 6.8 GB.

    Args:
        model: The (PEFT-wrapped) causal LM.
        group: The group.
        which: ``"grpo"`` or ``"nll"``; which of the two losses to backward.
        device: Torch device.
        token_denominator: Total RESPONSE tokens in the batch.
        prompt_denominator: Total PROMPT tokens in the batch.
        seq_chunk: Sequences per forward. ``None`` derives it from
            ``activation_budget_bytes`` and the model's depth and width.
        activation_budget_bytes: Ceiling on the trunk's retained activations per forward.
        **loss_kw: Forwarded to :func:`group_losses` (chunking and the logits budget).

    Returns:
        The loss value, summed over sub-batches -- identical to the single-forward value.

    Raises:
        ValueError: On an unknown ``which``.
    """
    if which not in ("grpo", "nll"):
        raise ValueError(f"unknown loss {which!r}; expected 'grpo' or 'nll'")
    width = len(group.prompt_ids) + max(len(r) for r in group.response_ids)
    if seq_chunk is None:
        cfgobj = getattr(model, "config", None)
        n_layers = int(getattr(cfgobj, "num_hidden_layers", 0) or 0)
        hidden = int(getattr(cfgobj, "hidden_size", 0) or 0)
        if n_layers > 0 and hidden > 0:
            per_forward = forward_tokens_for_budget(n_layers, hidden, activation_budget_bytes)
            seq_chunk = max(1, per_forward // max(1, width))
        else:
            seq_chunk = group.size
    seq_chunk = max(1, min(int(seq_chunk), group.size))

    total = 0.0
    for start in range(0, group.size, seq_chunk):
        stop = min(start + seq_chunk, group.size)
        # A sub-batch is a Group in its own right, but its ADVANTAGES must stay the ones the
        # whole group defines: recomputing them from a slice would centre the rewards within
        # the slice and change the measurement. So the slice carries the parent's rewards and
        # the advantages are taken from the parent.
        part = Group(
            group_id=group.group_id, task=group.task, prompt_ids=group.prompt_ids,
            response_ids=group.response_ids[start:stop], rewards=group.rewards[start:stop],
        )
        part_adv = group.advantages()[start:stop]
        grpo, nll = group_losses(
            model, part, device=device, token_denominator=token_denominator,
            prompt_denominator=prompt_denominator, _advantages=part_adv, **loss_kw,
        )
        loss = grpo if which == "grpo" else nll
        loss.backward()
        total += float(loss.detach())
    return total


def _assert_deterministic_forward(model) -> None:
    """Refuse to run if any dropout is active, because two things here recompute the forward.

    Both the per-chunk checkpointing and whole-model gradient checkpointing evaluate the
    forward twice and differentiate the SECOND evaluation. With dropout active the two draw
    different masks, and the resulting gradient is not the gradient of the loss that was
    reported -- it is a plausible number with no error anywhere, which is the failure class
    this project distrusts most and has already recorded once for checkpointing.

    Args:
        model: The model about to be probed.

    Raises:
        RuntimeError: Naming the modules with a non-zero rate, so the fix is to disable them
            rather than to guess.
    """
    import torch.nn as nn

    active = [
        name for name, mod in model.named_modules()
        if isinstance(mod, nn.Dropout) and float(getattr(mod, "p", 0.0)) > 0.0
    ]
    if active:
        raise RuntimeError(
            f"dropout is active on {len(active)} module(s), e.g. {active[:3]}. The probe "
            "recomputes the forward for both chunk and model checkpointing and would "
            "differentiate a different dropout mask than it reported. Disable dropout "
            "(the live configs set disable_dropout and lora_dropout 0) or pass "
            "--no-checkpoint and --no-gradient-checkpointing, which will need far more memory"
        )


def _free_bytes(device) -> int | None:
    """Free memory on ``device``, or ``None`` where it cannot be asked.

    Returned rather than raised on a CPU or a backend without the query, so the budget check
    still runs and only the device check is skipped -- a guard that silently disabled itself
    on an unfamiliar backend would be worse than one that never existed.
    """
    import torch

    try:
        if torch.device(device).type == "cuda" and torch.cuda.is_available():
            return int(torch.cuda.mem_get_info(torch.device(device))[0])
    except Exception:  # pragma: no cover - backend-specific
        return None
    return None


def _lora_blocks(model):
    """``(name, parameter)`` for every trainable LoRA parameter, in a fixed order.

    Order is fixed by ``named_parameters`` and the NAME is what the sketch hashes on, so the
    projection is stable across groups and across processes even if the iteration order ever
    changed.
    """
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" in name:
            yield name, param


def run_dump(cfg: DumpConfig) -> dict[str, Any]:
    """Load, compute and write the dump.

    Returns:
        A summary dict, also written into the ``.npz`` as ``meta``.

    Raises:
        RuntimeError: If the requested full-gradient storage would exceed
            ``max_full_grad_gb``, or if the model carries no LoRA parameters at all.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg.model)
    dtype = getattr(torch, cfg.dtype)
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype)
    if cfg.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, cfg.adapter, is_trainable=True)
    else:
        from peft import LoraConfig, TaskType, get_peft_model

        tm = list(cfg.target_modules)
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                target_modules="all-linear" if tm == ["all-linear"] else tm,
                bias="none",
            ),
            autocast_adapter_dtype=False,
        )
    model.to(cfg.device)
    model.train()
    # Both the chunked head and whole-model checkpointing differentiate a RECOMPUTED
    # forward, so a live dropout would make the measured gradient not the gradient of the
    # reported loss. Checked before anything runs.
    if cfg.use_checkpoint or cfg.gradient_checkpointing:
        _assert_deterministic_forward(model)
    if cfg.gradient_checkpointing:
        # Not optional at 32B. Fixing the unembedding leaves the decoder's own retained
        # activations as the binding constraint -- 64 layers over the worst group's 16,712
        # padded tokens -- and they do not fit either. Exact for a deterministic forward,
        # which the assertion above establishes.
        enable_inputs = getattr(model, "enable_input_require_grads", None)
        if callable(enable_inputs):
            # Without this the trunk's input does not require grad, every layer's
            # checkpoint is a no-op boundary, and no gradient reaches the LoRA parameters.
            enable_inputs()
        enable_ckpt = getattr(model, "gradient_checkpointing_enable", None)
        if callable(enable_ckpt):
            enable_ckpt(gradient_checkpointing_kwargs={"use_reentrant": False})

    blocks = list(_lora_blocks(model))
    if not blocks:
        raise RuntimeError(
            "the model carries no trainable LoRA parameters; the probe would sketch an "
            "empty gradient and report cosines over nothing"
        )
    n_params = sum(int(p.numel()) for _n, p in blocks)

    groups = load_rollouts(
        cfg.rollouts, tokenizer=tok, max_len=cfg.max_len, group_key=cfg.group_key,
        task_key=cfg.task_key, reward_key=cfg.reward_key,
    )
    n_full = min(int(cfg.full_grad_groups), len(groups))
    gb = n_full * n_params * 4 / 1e9
    if gb > cfg.max_full_grad_gb:
        raise RuntimeError(
            f"storing full gradients for {n_full} groups of {n_params:,} LoRA parameters "
            f"needs {gb:.1f} GB, over the {cfg.max_full_grad_gb} GB limit. Lower "
            "--full-grad-groups (the sketch validation needs only a handful of pairs) or "
            "raise --max-full-grad-gb deliberately"
        )

    resp_tokens = sum(len(r) for g in groups for r in g.response_ids)
    prompt_tokens = sum(len(g.prompt_ids) * g.size for g in groups)
    plan = SketchPlan(dim=cfg.sketch_dim, seed=cfg.sketch_seed)
    extractor = LayerLogitExtractor(mode=cfg.extractor_mode)
    boxed = tok.encode("\\boxed{", add_special_tokens=False)

    _decoder, _unembed = _decoder_and_unembedding(model)
    vocab = int(_unembed.shape[0])
    budget_bytes = int(cfg.logits_budget_gb * 1024**3)

    # The whole batch's worst group, priced BEFORE any of it runs. The probe batch's tail is
    # one 1,330-token prompt repeated across eight samples, and that group is 11th of 128 in
    # file order -- so pricing it up front turns a refusal from an hour of wasted work into a
    # message before the first forward.
    def _padded_tokens(g):
        """Tokens one forward materialises for this group: B x T, T padded to its longest."""
        return g.size * (len(g.prompt_ids) + max(len(r) for r in g.response_ids))

    widest = max(groups, key=_padded_tokens)
    widest_tokens = _padded_tokens(widest)
    # Positions, not tokens: chunking iterates over the emitting positions, which is one
    # fewer per sequence. Capping here makes the RECORDED plan the plan that actually runs --
    # without the cap a tiny-vocabulary model records a chunk of millions of tokens and
    # prices it at the full budget while doing kilobytes of work.
    widest_positions = widest.size * max(1, _padded_tokens(widest) // widest.size - 1)
    plan_chunk = cfg.chunk_tokens if cfg.chunk_tokens is not None else chunk_tokens_for_budget(
        vocab, budget_bytes, peak_bytes=cfg.logit_peak_bytes, cap=widest_positions
    )
    loss_kw = dict(
        chunk_tokens=cfg.chunk_tokens, logits_budget_bytes=budget_bytes,
        peak_bytes_per_element=cfg.logit_peak_bytes, use_checkpoint=cfg.use_checkpoint,
    )
    backward_kw = dict(
        seq_chunk=cfg.seq_chunk,
        activation_budget_bytes=int(cfg.activation_budget_gb * 1024**3),
        **loss_kw,
    )
    budget_record = assert_logits_fit(
        group_id=widest.group_id, n_tokens=widest_tokens, vocab=vocab,
        chunk_tokens=plan_chunk, budget_bytes=budget_bytes,
        free_bytes=_free_bytes(cfg.device), peak_bytes=cfg.logit_peak_bytes,
    )

    sketches, prompt_sketches, feats = [], [], []
    full_grads: list[np.ndarray] = []
    gids, tasks, sizes, rewards, zero_frac, feat_ok = [], [], [], [], [], []
    t_grad = t_feat = 0.0

    for gi, g in enumerate(groups):
        # --- the two gradients -------------------------------------------------------
        s = time.time()
        model.zero_grad(set_to_none=True)
        group_backward(
            model, g, which="grpo", device=cfg.device,
            token_denominator=resp_tokens, prompt_denominator=prompt_tokens,
            **backward_kw, free_bytes=_free_bytes(cfg.device),
        )
        grads = [(n, p.grad if p.grad is not None else torch.zeros_like(p)) for n, p in blocks]
        zero_frac.append(
            sum(1 for _n, gr in grads if not bool(gr.any())) / max(1, len(grads))
        )
        sketches.append(sketch_torch(grads, plan))
        if gi < n_full:
            full_grads.append(
                torch.cat([gr.detach().reshape(-1).float().cpu() for _n, gr in grads]).numpy()
            )
        model.zero_grad(set_to_none=True)
        group_backward(
            model, g, which="nll", device=cfg.device,
            token_denominator=resp_tokens, prompt_denominator=prompt_tokens,
            **backward_kw, free_bytes=_free_bytes(cfg.device),
        )
        pgrads = [(n, p.grad if p.grad is not None else torch.zeros_like(p)) for n, p in blocks]
        prompt_sketches.append(sketch_torch(pgrads, plan))
        model.zero_grad(set_to_none=True)
        t_grad += time.time() - s

        # --- the behavioural feature -------------------------------------------------
        s = time.time()
        vecs = []
        for r in g.response_ids:
            seq = g.prompt_ids + r
            try:
                pos = answer_token_index(
                    seq, boxed_ids=boxed, strategy=cfg.answer_strategy,
                    response_start=len(g.prompt_ids),
                )
            except Exception:
                # Fall back to the final position and RECORD it. A group whose answer token
                # could not be located has a feature from a different place than its peers,
                # and a comparison that did not know would be between two things.
                pos = len(seq) - 2
            ids = torch.tensor(seq, dtype=torch.long, device=cfg.device).unsqueeze(0)
            trace = extractor.trace(model, ids, pos, int(seq[pos + 1]))
            vecs.append(
                meds_feature(
                    trace, use_layer_diff=cfg.use_layer_diff,
                    last_n_layers=cfg.last_n_layers,
                )
            )
        feats.append(np.mean(np.stack(vecs, 0), axis=0))
        feat_ok.append(1.0)
        t_feat += time.time() - s

        gids.append(g.group_id)
        tasks.append(g.task)
        sizes.append(g.size)
        rewards.append(float(np.mean(g.rewards)))

    meta = {
        "model": cfg.model,
        "adapter": cfg.adapter or "",
        "n_groups": len(groups),
        "n_lora_params": n_params,
        "sketch_dim": cfg.sketch_dim,
        "sketch_seed": cfg.sketch_seed,
        "response_tokens": resp_tokens,
        "prompt_tokens": prompt_tokens,
        "answer_strategy": cfg.answer_strategy,
        "group_feature_agg": cfg.group_feature_agg,
        "seconds_gradients": round(t_grad, 3),
        "seconds_features": round(t_feat, 3),
        "seconds_total": round(time.time() - t0, 3),
        "full_grad_groups": n_full,
        "denominator": "global: batch response tokens for GRPO, batch prompt tokens for NLL",
        "vocab": vocab,
        "chunk_tokens": plan_chunk,
        "seq_chunk": cfg.seq_chunk,
        "activation_budget_gb": cfg.activation_budget_gb,
        "logits_budget_gb": cfg.logits_budget_gb,
        "logit_peak_bytes": cfg.logit_peak_bytes,
        "use_checkpoint": cfg.use_checkpoint,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        # The worst group priced before the run, so a completed dump can be read against the
        # plan it ran under rather than against the defaults someone assumes it used.
        "logits_budget": budget_record,
    }
    np.savez_compressed(
        cfg.out,
        sketch=np.stack(sketches, 0),
        prompt_sketch=np.stack(prompt_sketches, 0),
        meds_feature=np.stack(feats, 0),
        group_id=np.array(gids, dtype=object),
        task=np.array(tasks, dtype=object),
        group_size=np.array(sizes, dtype=np.int64),
        reward_mean=np.array(rewards, dtype=np.float64),
        zero_block_fraction=np.array(zero_frac, dtype=np.float64),
        full_grad=(np.stack(full_grads, 0) if full_grads else np.zeros((0, 0), dtype=np.float32)),
        meta=np.array(json.dumps(meta), dtype=object),
    )
    return meta


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--rollouts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--adapter", default=None)
    p.add_argument("--sketch-dim", type=int, default=8192)
    p.add_argument("--sketch-seed", type=int, default=0)
    p.add_argument("--full-grad-groups", type=int, default=8)
    p.add_argument("--max-full-grad-gb", type=float, default=8.0)
    p.add_argument("--answer-strategy", default="boxed", choices=["boxed", "last"])
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--last-n-layers", type=int, default=None)
    p.add_argument("--extractor-mode", default="hooks", choices=["hooks", "hidden_states"])
    p.add_argument("--group-key", default=None, help="jsonl field holding prompt identity")
    p.add_argument("--task-key", default=None, help="jsonl field holding the task label")
    p.add_argument("--reward-key", default=None, help="jsonl field holding the reward")
    p.add_argument("--chunk-tokens", type=int, default=None,
                   help="positions per unembedding chunk; default derives from the budget")
    p.add_argument("--seq-chunk", type=int, default=None,
                   help="sequences per trunk forward; default derives from the budget")
    p.add_argument("--activation-budget-gb", type=float, default=6.0,
                   help="ceiling on the trunk's retained activations per forward")
    p.add_argument("--logits-budget-gb", type=float, default=4.0,
                   help="ceiling on the chunked head's transient memory")
    p.add_argument("--logit-peak-bytes", type=int, default=LOGIT_PEAK_BYTES_PER_ELEMENT,
                   help="bytes per (token, vocab-entry) used by the estimate")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="retain every chunk's logits instead of recomputing them")
    p.add_argument("--no-gradient-checkpointing", action="store_true",
                   help="retain the decoder's activations; needs far more memory at 32B")
    a = p.parse_args(argv)
    meta = run_dump(
        DumpConfig(
            model=a.model, rollouts=a.rollouts, out=a.out, adapter=a.adapter,
            sketch_dim=a.sketch_dim, sketch_seed=a.sketch_seed,
            full_grad_groups=a.full_grad_groups, max_full_grad_gb=a.max_full_grad_gb,
            answer_strategy=a.answer_strategy, lora_rank=a.lora_rank,
            lora_alpha=a.lora_alpha, dtype=a.dtype, max_len=a.max_len, device=a.device,
            last_n_layers=a.last_n_layers, extractor_mode=a.extractor_mode,
            group_key=a.group_key, task_key=a.task_key, reward_key=a.reward_key,
            chunk_tokens=a.chunk_tokens, seq_chunk=a.seq_chunk,
            activation_budget_gb=a.activation_budget_gb,
            logits_budget_gb=a.logits_budget_gb,
            logit_peak_bytes=a.logit_peak_bytes,
            use_checkpoint=not a.no_checkpoint,
            gradient_checkpointing=not a.no_gradient_checkpointing,
        )
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
