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

__all__ = ["DumpConfig", "Group", "load_rollouts", "run_dump", "group_losses"]


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


def group_losses(model, group: Group, *, device, token_denominator: int, prompt_denominator: int):
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

    Args:
        model: The (PEFT-wrapped) causal LM.
        group: The group.
        device: Torch device.
        token_denominator: Total RESPONSE tokens in the batch.
        prompt_denominator: Total PROMPT tokens in the batch.

    Returns:
        ``(grpo_loss, prompt_nll_loss)``, both scalars with grad.

    Raises:
        ValueError: If a denominator is zero, which would divide the whole measurement by
            nothing rather than by the batch.
    """
    import torch
    import torch.nn.functional as F

    if token_denominator <= 0 or prompt_denominator <= 0:
        raise ValueError(
            f"denominators must be positive, got response={token_denominator} "
            f"prompt={prompt_denominator}"
        )
    adv = group.advantages()
    n_prompt = len(group.prompt_ids)
    seqs = [group.prompt_ids + r for r in group.response_ids]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), 0, dtype=torch.long)
    attn = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, : len(s)] = 1
    ids, attn = ids.to(device), attn.to(device)
    logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits.float()
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target = ids[:, 1:]
    picked = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    valid = attn[:, 1:].bool()

    pos = torch.arange(width - 1, device=device).unsqueeze(0)
    # Position t predicts token t+1, so the RESPONSE region in emitter coordinates starts at
    # n_prompt - 1: that position emits the response's first token. Off by one here would
    # put the prompt's last token into the RL loss and drop the response's first, which is
    # the same off-by-one this project has already recorded once in its token router.
    resp_mask = valid & (pos >= n_prompt - 1)
    prompt_mask = valid & (pos < n_prompt - 1)

    a = torch.tensor(adv, dtype=torch.float32, device=device).unsqueeze(1)
    grpo = -(a * picked * resp_mask).sum() / token_denominator
    prompt_nll = -(picked * prompt_mask).sum() / prompt_denominator
    return grpo, prompt_nll


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

    sketches, prompt_sketches, feats = [], [], []
    full_grads: list[np.ndarray] = []
    gids, tasks, sizes, rewards, zero_frac, feat_ok = [], [], [], [], [], []
    t_grad = t_feat = 0.0

    for gi, g in enumerate(groups):
        # --- the two gradients -------------------------------------------------------
        s = time.time()
        model.zero_grad(set_to_none=True)
        grpo, _ = group_losses(
            model, g, device=cfg.device,
            token_denominator=resp_tokens, prompt_denominator=prompt_tokens,
        )
        grpo.backward()
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
        _, pnll = group_losses(
            model, g, device=cfg.device,
            token_denominator=resp_tokens, prompt_denominator=prompt_tokens,
        )
        pnll.backward()
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
        )
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
