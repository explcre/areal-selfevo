#!/usr/bin/env python3
"""GRPO + LoRA on one arm of the gate-outcome experiment, one trainer GPU + one rollout GPU.

The loop is deliberately small: draw prompts with the arm's selector, roll them out through
the arm's OWN adapter on a colocated sglang server, grade, form group-relative advantages,
take exactly one gradient step on the data that produced them, write the adapter back to the
server. One step per batch means the importance ratio is exactly 1 where the loss is
evaluated, so no clipping term is needed and none is present -- a PPO ratio here would be a
decoration that never fires.

Three things this file refuses to do silently, each because the failure is invisible:

* **Adapter coverage.** ``target_modules`` inherited from an older Qwen matches
  ``self_attn.{q,k,v,o}_proj``, present on 16 of this model's 64 layers; the other 48 are
  ``linear_attn``. A wrong and a right config give identical forward loss, so coverage is
  asserted by COUNTING LAYERS REACHED, not by comparing losses.
* **Adapter routing.** An adapter applies only through ``/generate``'s ``lora_path``. The
  server is asked, before the first step and after every reload, to produce text under the
  adapter that differs from base.
* **Truncated rollouts.** They are UNKNOWN, not wrong. Scoring them zero would make the token
  budget look like difficulty, which is the failure that made the default operating point
  unusable in the first place.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import sys
import time

import aiohttp
import requests
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arms  # noqa: E402
import gate_lib  # noqa: E402
from gate_lib import build_prompt  # noqa: E402

# Modules the serving backend can route a LoRA through for this architecture, chosen so the
# adapter reaches ALL 64 decoder layers: q/k/v/o_proj exist on the 16 full-attention layers,
# linear_attn.out_proj on the other 48, and the MLP triple on every layer. The GDN input
# projections (in_proj_qkv / in_proj_z) are excluded because sglang fuses them into a single
# `in_proj_qkvz` buffer that an unfused PEFT adapter cannot be loaded into; that exclusion is
# a serving-backend limit and is reported rather than hidden.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
                  "gate_proj", "up_proj", "down_proj"]
EXCLUDE = r".*(visual|mtp)\..*"
EXPECTED_LAYERS = 64


def assert_coverage(model, expected_layers: int = EXPECTED_LAYERS) -> dict:
    """Count the decoder layers and module families the attached adapter actually reaches.

    Raises:
        AssertionError: If any decoder layer has no adapter on it, or if the linear-attention
            or full-attention families are missed entirely -- the 16-of-64 trap.

    Returns:
        A coverage census, recorded in the run's metadata.
    """
    import re
    fams = {"self_attn": set(), "linear_attn": set(), "mlp": set()}
    n_lora = 0
    for name, _ in model.named_modules():
        if not name.endswith("lora_A.default"):
            continue
        n_lora += 1
        # Two naming schemes reach the same weights: the vision-language checkpoint
        # stores `model.language_model.layers.N`, while `AutoModelForCausalLM` resolves to
        # the text-only class and exposes `model.layers.N`. Match either, or the assertion
        # reports zero coverage for a perfectly good adapter.
        m = re.search(r"(?:language_model\.)?layers\.(\d+)\.(self_attn|linear_attn|mlp)\.",
                      name)
        if m:
            fams[m.group(2)].add(int(m.group(1)))
        if re.search(r"(visual|mtp)\.", name):
            raise AssertionError("adapter attached OUTSIDE the language model: %s" % name)
    depth = set().union(*fams.values())
    cov = {"lora_modules": n_lora, "layers_reached": len(depth),
           **{k: len(v) for k, v in fams.items()}}
    if len(depth) != expected_layers:
        raise AssertionError(
            "adapter reaches %d of %d decoder layers; a config that misses layers gives the "
            "SAME forward loss as one that does not, so this cannot be caught downstream. %s"
            % (len(depth), expected_layers, cov))
    if cov["self_attn"] == 0 or cov["linear_attn"] == 0:
        raise AssertionError("adapter misses an attention family entirely: %s" % cov)
    return cov


def save_adapter(model, path: str) -> None:
    """Write the adapter to ``path`` atomically, so the server never loads a half file."""
    tmp = path + ".tmp"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    model.save_pretrained(tmp, safe_serialization=True)
    cfg_path = os.path.join(tmp, "adapter_config.json")
    cfg = json.load(open(cfg_path))
    # sglang accepts only a list (or 'all'/'all-linear') here; PEFT may have stored a regex.
    cfg["target_modules"] = sorted(TARGET_MODULES)
    cfg.pop("exclude_modules", None)
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.rename(tmp, path)


def adapter_fingerprint(model) -> float:
    """Sum of |LoRA-B| over the whole adapter -- an OBSERVABLE that is exactly 0 at init.

    Used instead of "save_pretrained returned" as evidence that a step changed anything: a
    save can succeed on an adapter that never moved, and a zero LoRA-B adapter is
    arithmetically identical to the base model however many steps have "run".
    """
    tot = 0.0
    for n, p in model.named_parameters():
        if "lora_B" in n:
            tot += float(p.detach().abs().sum())
    return tot


def push_adapter(url: str, name: str, path: str, loaded: bool = True) -> None:
    """Replace the adapter the server serves under ``name`` with the one at ``path``.

    The unload is best-effort because the server may or may not already hold the name (it is
    given one at boot so that ``--enable-lora`` has something to size its buffers from); the
    LOAD is not best-effort, because a silently skipped load would leave every later step
    generating from a stale adapter and the reward curve would be of a model that stopped
    changing.
    """
    path = os.path.abspath(path)
    # Bounded, not best-effort-forever. MEASURED: on a clean server the pair takes 0.0 s and
    # 0.2 s, but a server left holding orphaned requests (a trainer killed mid-rollout) logs
    # "Start unload Lora adapter" and never returns, and an unbounded timeout there stalls the
    # arm at 0% GPU looking exactly like a slow step. A load that does not land must stop the
    # run: continuing would roll out from a stale adapter while the trainer moved on, and the
    # reward curve would be of a policy that no longer exists.
    if loaded:
        try:
            requests.post(url + "/unload_lora_adapter", json={"lora_name": name}, timeout=300)
        except Exception as exc:
            raise RuntimeError(
                "unload_lora_adapter(%s) did not return within 300 s (%r). The server is "
                "wedged; every later rollout would come from a stale adapter." % (name, exc))
    r = requests.post(url + "/load_lora_adapter",
                      json={"lora_name": name, "lora_path": path}, timeout=600)
    if r.status_code != 200:
        raise RuntimeError("load_lora_adapter failed: %s %s" % (r.status_code, r.text[:400]))


async def rollout(url: str, tasks, tok, effort: str, cap: int, group: int, lora: str,
                  concurrency: int) -> list[dict]:
    """Sample ``group`` completions for each task and return them with exact token ids.

    ``input_ids`` are sent rather than text, and the generated token ids are read back from
    ``meta_info.output_token_logprobs``, so the sequence the trainer differentiates is exactly
    the sequence the server produced. Re-tokenising the returned text would drift.
    """
    sem = asyncio.Semaphore(concurrency)
    out: list[dict] = []
    lock = asyncio.Lock()

    async def one(session, t, prompt_ids, rep):
        payload = {"input_ids": prompt_ids,
                   "sampling_params": {"max_new_tokens": cap, "temperature": 1.0,
                                       "top_p": 0.95, "skip_special_tokens": True},
                   "return_logprob": True}
        if lora:
            payload["lora_path"] = lora
        async with sem:
            for attempt in range(3):
                try:
                    async with session.post(url + "/generate", json=payload,
                                            timeout=aiohttp.ClientTimeout(total=3600)) as r:
                        body = await r.text()
                        if r.status != 200:
                            raise RuntimeError("HTTP %d %s" % (r.status, body[:300]))
                        d = json.loads(body)
                    break
                except Exception as exc:
                    if attempt == 2:
                        async with lock:
                            out.append({"idx": t.idx, "rep": rep, "error": repr(exc)[:200],
                                        "finish": "error"})
                        return
                    await asyncio.sleep(2.0 * (attempt + 1))
        mi = d["meta_info"]
        ids = [tid for _, tid, _ in mi.get("output_token_logprobs") or []]
        fin = mi.get("finish_reason") or {}
        async with lock:
            out.append({"idx": t.idx, "rep": rep, "prompt_ids": prompt_ids,
                        "output_ids": ids, "text": d.get("text", ""),
                        "finish": fin.get("type") if isinstance(fin, dict) else str(fin),
                        "behaviour_logprob": sum(lp for lp, _, _ in
                                                 (mi.get("output_token_logprobs") or [])),
                        "answer": t.answer})

    conn = aiohttp.TCPConnector(limit=concurrency + 8)
    async with aiohttp.ClientSession(connector=conn) as session:
        jobs = []
        for t in tasks:
            pid = tok(build_prompt(tok, t.problem, effort), add_special_tokens=False)["input_ids"]
            for rep in range(group):
                jobs.append(one(session, t, pid, rep))
        await asyncio.gather(*jobs)
    return out


def group_advantages(rows: list[dict], random_reward: bool, rng: random.Random,
                     truncated: str = "wrong") -> tuple[list[dict], dict]:
    """Grade, form group-relative advantages, and drop the groups that carry none.

    ``truncated`` MUST match the convention the pool was built under, and the caller asserts
    that it does. Mixing them is not a subtle error: with the pool defined by "a rollout that
    hit the cap did not answer, so it is wrong" and the advantage computed by "drop truncated
    rollouts", a task selected precisely because it sometimes fails to terminate has its
    failures deleted, its two surviving rollouts are both correct, the group is unanimous and
    carries no gradient. MEASURED on the first smoke step: accuracy 1.000, informative
    fraction 0.00, no gradient at all, on a pool every member of which is mixed by
    construction.

    Errored rollouts (no response after retries) are dropped under either convention: they are
    a property of the harness, not of the model.

    Args:
        rows: Rollouts from :func:`rollout`.
        random_reward: When true the reward is an independent fair coin, so the C3 arm shares
            the rollouts and the gradient machinery and differs only in what it is told.
        rng: Source for the C3 coin.
        truncated: ``wrong`` (a capped rollout scores 0 and stays in its group) or ``unknown``
            (it is dropped).

    Returns:
        ``(trainable_rows, stats)``. ``stats`` carries the k-histogram, without which a
        gradient statistic over a mostly-unanimous batch reads as a null.
    """
    by_task: dict[int, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["idx"], []).append(r)
    keep: list[dict] = []
    khist: dict[int, int] = {}
    n_dead = n_thin = n_trunc = n_err = 0
    n_groups = 0
    acc_num = acc_den = 0
    for idx, g in by_task.items():
        n_groups += 1
        live = []
        for r in g:
            if r.get("finish") == "error":
                n_err += 1
                continue
            if r.get("finish") == "length":
                n_trunc += 1
                if truncated == "unknown":
                    continue
                r["correct"] = False
            else:
                r["correct"] = bool(gate_lib.math_bench.grade(r["text"], r["answer"]))
            acc_num += int(r["correct"])
            acc_den += 1
            r["reward"] = float(rng.random() < 0.5) if random_reward else float(r["correct"])
            live.append(r)
        if len(live) < 2:
            n_thin += 1
            continue
        k = sum(int(r["reward"] > 0.5) for r in live)
        khist[k] = khist.get(k, 0) + 1
        mean = sum(r["reward"] for r in live) / len(live)
        var = sum((r["reward"] - mean) ** 2 for r in live) / len(live)
        if var <= 0:
            n_dead += 1          # unanimous: every advantage is exactly zero
            continue
        std = var ** 0.5
        for r in live:
            r["adv"] = (r["reward"] - mean) / (std + 1e-6)
        keep.extend(live)
    stats = {"groups": n_groups, "dead_groups": n_dead, "thin_groups": n_thin,
             "truncation_rate": n_trunc / max(len(rows), 1),
             "truncated_rollouts": n_trunc, "error_rollouts": n_err,
             "k_histogram": khist, "trainable_rollouts": len(keep),
             "rollout_accuracy": (acc_num / acc_den) if acc_den else None,
             "informative_group_fraction": (n_groups - n_dead - n_thin) / max(n_groups, 1)}
    return keep, stats


def assert_padding_is_a_noop(model, tok, device, pad_id: int, ratio: float = 1.6,
                             slack: float = 0.02) -> dict:
    """Prove right padding adds nothing beyond the drift that any reshaping already causes.

    Micro-batching is a performance rewrite of the thing being measured, so it has to be shown
    to be a no-op rather than assumed to be one -- 48 of this model's 64 layers are
    gated-delta-net recurrences, and a recurrent layer that consumed padding would silently
    change every logprob the gradient is built from.

    A fixed absolute tolerance cannot do this job, and trying one first is what showed why.
    MEASURED on this model: scoring a sequence alone, then in a batch of two COPIES of itself
    (equal lengths, so no padding at all), already moves a token's logprob by 0.103, while two
    identical calls agree to 0.0 exactly. So the drift is a shape-dependent kernel/reduction
    change in bf16, and a 0.02 tolerance rejects it whatever padding does.

    The discriminating question is whether padding adds anything ON TOP of that, and whether
    what it adds GROWS with the amount of padding -- which a recurrence that consumed pad
    tokens would. Both are checked against the no-padding control.

    Raises:
        AssertionError: If padding drifts more than ``ratio`` times the batching-only control,
            or if drift grows with padding length.
    """
    a = tok("The capital of France is Paris, and the capital of Japan is Tokyo.",
            add_special_tokens=False)["input_ids"]

    def score(seqs, pad_to=None):
        L = pad_to or max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, sq in enumerate(seqs):
            ids[i, : len(sq)] = torch.tensor(sq)
            att[i, : len(sq)] = 1
        ids, att = ids.to(device), att.to(device)
        with torch.no_grad():
            lg = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1, :]
            lp = torch.log_softmax(lg.float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
        return [lp[i, : len(sq) - 1].cpu() for i, sq in enumerate(seqs)]

    d = lambda x, y: float((x - y).abs().max())
    ref = score([a])[0]
    out = {
        "determinism_floor": d(ref, score([a])[0]),          # same shape twice: must be 0
        "batching_only_no_padding": d(ref, score([a, a])[0]),  # reshape without padding
        "padding_only_to_64": d(ref, score([a], pad_to=64)[0]),
        "padding_only_to_256": d(ref, score([a], pad_to=256)[0]),
        "batching_and_padding": d(ref, score([a, a + a])[0]),
        "mean_abs_logprob": float(ref.abs().mean()),
    }
    floor = out["batching_only_no_padding"]
    budget = max(ratio * floor, floor + slack)
    out["budget"] = budget
    out["grows_with_padding"] = out["padding_only_to_256"] > 1.5 * out["padding_only_to_64"] + slack
    if out["determinism_floor"] > 1e-6:
        raise AssertionError("the model is not deterministic at a fixed shape (%g); nothing "
                             "below can be attributed to padding. %s"
                             % (out["determinism_floor"], out))
    if out["padding_only_to_256"] > budget or out["batching_and_padding"] > budget:
        raise AssertionError(
            "right padding drifts MORE than reshaping alone (%.4g vs a %.4g budget from a "
            "%.4g no-padding control), so padding is entering the computation. Micro-batching "
            "would change the gradient it is supposed to compute faster; re-run with "
            "--mb-tokens 1. %s" % (out["padding_only_to_256"], budget, floor, out))
    if out["grows_with_padding"]:
        raise AssertionError(
            "drift GROWS with the amount of padding (%.4g at 64 -> %.4g at 256), which is what "
            "a recurrence consuming pad tokens looks like. %s"
            % (out["padding_only_to_64"], out["padding_only_to_256"], out))
    return out


def calibrate_microbatch(model, optimizer, device, pad_id: int, mb_tokens: int,
                        grad_ckpt: int, headroom_gib: float = 20.0) -> dict:
    """Run ONE worst-case forward+backward before training and measure what it costs.

    A memory setting must be checked by measuring, not by reasoning: gradient checkpointing
    that silently did not switch on gives the same loss and the same logs as one that did, and
    announces itself only as an out-of-memory error partway into the first real step, after
    the model has been loaded and the first rollouts have been paid for.

    This runs the largest micro-batch the trainer can construct (a single sequence filling
    ``mb_tokens``, which is the shape a long rollout produces) and refuses to start if the peak
    leaves less than ``headroom_gib`` free.

    Raises:
        AssertionError: If gradient checkpointing was requested but is not active, or if the
            measured peak leaves too little headroom.
    """
    if grad_ckpt:
        flag = getattr(getattr(model.base_model.model, "model", None),
                       "gradient_checkpointing", None)
        if not model.training:
            raise AssertionError("the model is in eval mode, so gradient checkpointing is "
                                 "inert whatever was requested")
        if flag is not True:
            raise AssertionError("gradient checkpointing was requested but the decoder's flag "
                                 "is %r" % flag)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ids = torch.full((1, mb_tokens), pad_id, dtype=torch.long, device=device)
    att = torch.ones_like(ids)
    out = model(input_ids=ids, attention_mask=att, use_cache=False)
    logits = out.logits[:, :-1, :]
    lp = torch.log_softmax(logits.float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    (lp.sum() / lp.numel()).backward()
    optimizer.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated() / 2**30
    total = torch.cuda.get_device_properties(device).total_memory / 2**30
    del out, logits, lp
    torch.cuda.empty_cache()
    res = {"mb_tokens": mb_tokens, "grad_ckpt": bool(grad_ckpt),
           "peak_gib": round(peak, 1), "device_gib": round(total, 1),
           "free_after_peak_gib": round(total - peak, 1)}
    if total - peak < headroom_gib:
        raise AssertionError(
            "a single %d-token micro-batch peaks at %.1f GiB of %.1f GiB, leaving %.1f GiB. "
            "Lower --mb-tokens or enable --grad-ckpt. %s"
            % (mb_tokens, peak, total, total - peak, res))
    return res


def _microbatches(rows, max_len: int, mb_tokens: int):
    """Group rollouts into padded micro-batches of roughly ``mb_tokens`` tokens each.

    Sequences are sorted by length first, so a batch pads to nearly its own longest member
    instead of to the longest in the step. With a median completion of ~1.2k tokens and a tail
    past 16k, one-sequence-per-forward leaves the device almost idle; this is the single
    change that makes a 27B on-policy loop affordable at this scale.

    Right padding is safe for a causal model: a real position never attends to a padded one,
    and the padded positions' outputs are masked out of the loss.
    """
    items = []
    for r in rows:
        oid = r["output_ids"][:max_len]
        if oid:
            items.append((len(r["prompt_ids"]) + len(oid), r, oid))
    items.sort(key=lambda x: x[0])
    batch, longest = [], 0
    for L, r, oid in items:
        longest_next = max(longest, L)
        if batch and longest_next * (len(batch) + 1) > mb_tokens:
            yield batch
            batch, longest = [], 0
            longest_next = L
        batch.append((r, oid))
        longest = longest_next
    if batch:
        yield batch


def _trunk_and_head(model):
    """The transformer trunk and the (frozen) output head, through any PEFT wrapper.

    Returns:
        ``(trunk, lm_head)`` where ``trunk(input_ids=..., attention_mask=...)[0]`` is the
        hidden-state sequence and ``lm_head`` maps one hidden state to vocabulary logits.
    """
    base = getattr(model, "base_model", None)
    inner = getattr(base, "model", model) if base is not None else model
    return inner.model, inner.lm_head


def loss_backward(model, ids, att, wgt, chunk: int) -> float:
    """Backward one microbatch, materialising logits at most `chunk` positions at a time.

    The loss is a SUM over positions of ``-w_t * log p(x_t)``, so it decomposes exactly over
    any partition of the positions. What does not decompose is the transformer forward, whose
    activations every position depends on. So the trunk runs once, its output is detached into
    a leaf, the per-chunk losses backpropagate into that leaf, and a single backward carries
    the accumulated hidden-state gradient through the trunk. The result is arithmetically the
    same gradient as materialising every logit at once, at a peak set by `chunk` rather than by
    the longest sequence in the batch.

    Positions whose weight is zero -- the prompt and the padding -- are skipped entirely
    rather than multiplied by zero, which is where most of the saving on a short-prompt,
    long-generation row comes from.

    Args:
        model: The PEFT-wrapped causal LM, in train mode.
        ids: ``[B, L]`` token ids, right-padded.
        att: ``[B, L]`` attention mask.
        wgt: ``[B, L]`` per-position advantage weight, already normalised; zero where the loss
            must not act.
        chunk: Positions per logits materialisation. ``0`` materialises them all, which is the
            original behaviour and is kept so the two can be compared.

    Returns:
        The scalar loss value.
    """
    if chunk <= 0:
        out = model(input_ids=ids, attention_mask=att, use_cache=False)
        logits = out.logits[:, :-1, :]
        logp = torch.log_softmax(logits.float(), dim=-1).gather(
            -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        loss = -(wgt[:, 1:] * logp).sum()
        loss.backward()
        val = float(loss.detach())
        del out, logits, logp, loss
        return val

    trunk, lm_head = _trunk_and_head(model)
    h = trunk(input_ids=ids, attention_mask=att, use_cache=False)[0]
    hd = h.detach().requires_grad_(True)
    H = hd.shape[-1]
    hf = hd[:, :-1, :].reshape(-1, H)
    tgt = ids[:, 1:].reshape(-1)
    w = wgt[:, 1:].reshape(-1)
    idx = (w != 0).nonzero(as_tuple=True)[0]
    total = 0.0
    if idx.numel() == 0:
        return 0.0
    for k in range(0, idx.numel(), chunk):
        sel = idx[k: k + chunk]
        lg = lm_head(hf[sel]).float()
        lp = torch.log_softmax(lg, dim=-1).gather(-1, tgt[sel].unsqueeze(-1)).squeeze(-1)
        part = -(w[sel] * lp).sum()
        part.backward(retain_graph=True)
        total += float(part.detach())
        del lg, lp, part
    h.backward(gradient=hd.grad)
    del h, hd, hf
    return total


def policy_step(model, rows, optimizer, device, max_len: int, grad_clip: float,
                mb_tokens: int, pad_id: int, loss_norm: str = "sequence",
                logit_chunk: int = 2048) -> dict:
    """One policy-gradient step over the rollouts that carry advantage.

    ``loss_norm`` decides how a rollout's tokens are weighted, and on this task it decides
    whether the run survives:

    * ``token`` (DAPO) divides by the batch's total completion tokens, so a rollout
      contributes in proportion to its length. Here a truncated rollout is 8192 tokens with a
      NEGATIVE advantage while a correct one is ~1000 tokens with a positive one, so the
      negative mass outweighs the positive by roughly eight to one and the policy is pushed
      hard away from long continuations. MEASURED: with this setting all four arms collapsed
      by step 8-10 -- truncation fell from ~0.60 to 0.02 and accuracy with it (T from 0.583 to
      0.177, C2 from 0.724 to 0.240). The reward is not what did that; the length weighting is,
      and the random-reward arm collapsed the same way.
    * ``sequence`` (vanilla GRPO) averages within a rollout first, so every rollout counts
      once whatever its length. The signal "terminate and be right" survives; the eight-to-one
      artefact does not.

    Exactly one update is taken on the data that produced it, so the importance ratio is 1
    where the loss is evaluated and no clipping term is needed; adding one would be a
    decoration that never fires.
    """
    total_tokens = sum(min(len(r["output_ids"]), max_len) for r in rows)
    if total_tokens == 0:
        return {"loss": None, "tokens": 0, "microbatches": 0}
    n_rows = len(rows)
    optimizer.zero_grad(set_to_none=True)
    loss_sum, nmb = 0.0, 0
    for batch in _microbatches(rows, max_len, mb_tokens):
        seqs = [r["prompt_ids"] + oid for r, oid in batch]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        # advantage per POSITION, zero everywhere the loss must not act (prompt and padding)
        wgt = torch.zeros((len(seqs), L), dtype=torch.float32)
        for i, ((r, oid), sq) in enumerate(zip(batch, seqs)):
            ids[i, : len(sq)] = torch.tensor(sq, dtype=torch.long)
            att[i, : len(sq)] = 1
            # sequence norm: the rollout's own length is folded into its per-token weight
            # here, so summing over tokens gives the rollout's MEAN log-prob times its
            # advantage, and every rollout then counts once.
            scale = (1.0 / len(oid) / n_rows) if loss_norm == "sequence" else (1.0 / total_tokens)
            wgt[i, len(r["prompt_ids"]): len(sq)] = r["adv"] * scale
        ids, att, wgt = ids.to(device), att.to(device), wgt.to(device)
        # the normaliser is already inside wgt; see loss_backward for why this is chunked
        loss_sum += loss_backward(model, ids, att, wgt, logit_chunk)
        nmb += 1
    gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                        grad_clip)
    optimizer.step()
    return {"loss": loss_sum, "tokens": total_tokens, "grad_norm": float(gn),
            "microbatches": nmb, "loss_norm": loss_norm,
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1)}


def main() -> int:
    """Run one arm to a fixed generated-token budget, checkpointing as it goes."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["T", "C1", "C2", "C3"])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="/mnt/localssd/gate/models/Qwen3.8-27B")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gcs", default="")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--loss-norm", default="sequence", choices=["sequence", "token"],
                    help="see policy_step; token-level weighting collapsed every arm here")
    ap.add_argument("--collapse-window", type=int, default=6,
                    help="halt the arm when the mean completion length over this many steps "
                         "falls below --collapse-frac of the first window's, which is what a "
                         "length collapse looks like and is not worth further GPU-hours")
    ap.add_argument("--collapse-frac", type=float, default=0.25)
    ap.add_argument("--grad-ckpt", type=int, default=0,
                    help="1 to enable gradient checkpointing; off by default because this "
                         "device has the memory and the 33% compute is better spent on steps")
    ap.add_argument("--max-len", type=int, default=16384,
                    help="completion tokens trained on per rollout")
    ap.add_argument("--mb-tokens", type=int, default=8192,
                    help="padded tokens per trainer forward; the logits tensor is "
                         "mb_tokens x 248320, so this is the memory knob")
    ap.add_argument("--token-budget", type=int, default=0,
                    help="stop when this many completion tokens have been GENERATED; the "
                         "budget the arms are matched on (pre-registration rule 3)")
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--max-hours", type=float, default=1e9)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--truncated", default="wrong", choices=["wrong", "unknown"],
                    help="how a rollout that hit the cap is scored; must equal the convention "
                         "the pool file was built under, which is asserted below")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--smoke", type=int, default=0,
                    help="run this many steps with tiny settings and exit non-zero on any "
                         "stage that did not fire")
    a = ap.parse_args()

    a.run_dir = os.path.abspath(a.run_dir)
    os.makedirs(a.run_dir, exist_ok=True)
    meta_path = os.path.join(a.run_dir, "meta.json")
    log_path = os.path.join(a.run_dir, "steps.jsonl")
    adapter_dir = os.path.join(a.run_dir, "adapter")

    pool_blob = json.load(open(a.pool))
    pool_conv = pool_blob.get("truncated_convention")
    if pool_conv != a.truncated:
        raise SystemExit(
            "FATAL: the pool was built with truncated=%r and the trainer was asked for %r. "
            "Mixing them makes every group in a mixed pool unanimous and the run trains "
            "nothing while reporting an accuracy of 1.000." % (pool_conv, a.truncated))
    pool = [arms.Task(**t) for t in pool_blob["tasks"]]
    selector = arms.make_selector(a.arm, pool, a.seed)
    rng = random.Random(a.seed + 7)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    device = "cuda:0"
    print("[%s] loading base model" % a.arm, flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map=device)
    model.config.use_cache = False
    lcfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM", target_modules=TARGET_MODULES,
                      exclude_modules=EXCLUDE)
    model = get_peft_model(model, lcfg)
    cov = assert_coverage(model)
    print("[%s] adapter coverage: %s" % (a.arm, cov), flush=True)
    pad_noop = assert_padding_is_a_noop(model, tok, device, pad_id)
    print("[%s] padding no-op check: %s" % (a.arm, pad_noop), flush=True)
    # Gradient checkpointing trades a 33% compute increase for activation memory. On a 183 GB
    # device with a 27B policy and micro-batches of a few thousand tokens there is memory to
    # spare, so it is OFF by default here and the saving is spent on steps instead. This is
    # exactly the headroom that was missing on the 80 GB boxes.
    model.train()   # NOT optional: transformers skips gradient checkpointing entirely when
                    # `self.training` is False, and `from_pretrained` returns a model in eval
                    # mode. MEASURED: without this the checkpointing call is accepted, no
                    # warning is issued, and the first backward OOMs at 175.9 GiB allocated on
                    # a 183 GB device -- the opt silently did not happen.
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95))
    mem = calibrate_microbatch(model, opt, device, pad_id, a.mb_tokens, a.grad_ckpt)
    print("[%s] microbatch calibration: %s" % (a.arm, mem), flush=True)

    meta = {"arm": a.arm, "selector": selector.describe(), "coverage": cov,
            "microbatch_calibration": mem,
            "truncated_convention": a.truncated, "pool_file": os.path.abspath(a.pool),
            "padding_noop": pad_noop,
            "n_trainable_params": n_trainable, "args": vars(a),
            "pool_census": pool_blob.get("census"), "started": time.time()}
    json.dump(meta, open(meta_path, "w"), indent=1, default=str)

    save_adapter(model, adapter_dir)
    push_adapter(a.url, a.arm, adapter_dir)
    # A freshly initialised adapter has LoRA-B = 0 and is arithmetically identical to the base
    # model, so "the adapter changes the output" MUST fail here and its failure is evidence
    # the check is real rather than a guard that cannot fail. The route assertion that has to
    # pass is deferred to the first reload after a real gradient step.
    try:
        asyncio.run(gate_lib.assert_adapter_routes(
            a.url, build_prompt(tok, "What is 17 times 23?", a.effort), a.arm))
        print("[%s] WARNING: a ZERO-initialised adapter changed the output; the route probe "
              "is measuring something other than the adapter" % a.arm, flush=True)
    except AssertionError:
        print("[%s] step-0: null adapter is base-identical, as it must be" % a.arm, flush=True)

    gen_tokens = 0
    t_start = time.time()
    step = 0
    route_verified = False
    logf = open(log_path, "a")
    while step < a.max_steps and (time.time() - t_start) < a.max_hours * 3600:
        if a.token_budget and gen_tokens >= a.token_budget:
            print("[%s] token budget reached: %d" % (a.arm, gen_tokens), flush=True)
            break
        tasks = selector.draw(a.prompts_per_step)
        t0 = time.time()
        rows = asyncio.run(rollout(a.url, tasks, tok, a.effort, a.cap, a.group_size,
                                   a.arm, a.concurrency))
        gen_tokens += sum(len(r.get("output_ids") or []) for r in rows)
        keep, stats = group_advantages(rows, a.arm == "C3", rng, a.truncated)
        t_gen = time.time() - t0
        t0 = time.time()
        upd = policy_step(model, keep, opt, device, a.max_len, a.grad_clip,
                          a.mb_tokens, pad_id, a.loss_norm) if keep else \
            {"loss": None, "tokens": 0}
        t_upd = time.time() - t0
        fp = adapter_fingerprint(model)
        rec = {"step": step, "arm": a.arm, "gen_tokens_cum": gen_tokens,
               "adapter_absB": fp,
               "tasks": [t.idx for t in tasks],
               "sel_p_a": [round(t.p_a, 4) for t in tasks],
               "sel_p_b": [round(t.p_b, 4) for t in tasks],
               "t_gen_s": round(t_gen, 1), "t_update_s": round(t_upd, 1),
               "elapsed_s": round(time.time() - t_start, 1), **stats, **upd}
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        acc_str = ("%.3f" % rec["rollout_accuracy"]) if rec["rollout_accuracy"] is not None \
            else "n/a"
        print("[%s] step %d acc=%s info=%.2f loss=%s tok=%d cum=%d %.0fs+%.0fs"
              % (a.arm, step, acc_str,
                 rec["informative_group_fraction"], rec["loss"], rec["tokens"],
                 gen_tokens, t_gen, t_upd), flush=True)

        if keep:
            save_adapter(model, adapter_dir)
            push_adapter(a.url, a.arm, adapter_dir, loaded=True)
            if not route_verified:
                # NOW the adapter is non-null, so base-identical output would mean the reload
                # is not taking effect and every later checkpoint would be the base model.
                pr = asyncio.run(gate_lib.assert_adapter_routes(
                    a.url, build_prompt(tok, "What is 17 times 23?", a.effort), a.arm))
                json.dump(pr, open(os.path.join(a.run_dir, "route_probe.json"), "w"), indent=1)
                route_verified = True
                print("[%s] ROUTE VERIFIED after first update" % a.arm, flush=True)

        step += 1
        # A length collapse is terminal and cheap to detect: the policy stops emitting and
        # every later step measures a model that no longer solves anything. Halting on it
        # keeps the failure dated and legible instead of burying it in a flat tail.
        recs_so_far = [json.loads(l) for l in open(log_path)]
        lens = [(r["gen_tokens_cum"] - (recs_so_far[i - 1]["gen_tokens_cum"] if i else 0))
                for i, r in enumerate(recs_so_far)]
        if len(lens) >= 2 * a.collapse_window:
            first = sum(lens[: a.collapse_window]) / a.collapse_window
            last = sum(lens[-a.collapse_window:]) / a.collapse_window
            if first > 0 and last < a.collapse_frac * first:
                print("[%s] HALTED: mean generated tokens per step fell from %.0f to %.0f "
                      "(%.2f of the opening window). This is a length collapse, not a "
                      "plateau." % (a.arm, first, last, last / first), flush=True)
                json.dump({"halted": "length_collapse", "first_window_tokens": first,
                           "last_window_tokens": last, "step": step},
                          open(os.path.join(a.run_dir, "halt.json"), "w"), indent=1)
                break
        if a.ckpt_every and step % a.ckpt_every == 0:
            snap = os.path.join(a.run_dir, "ckpt", "step%05d" % step)
            os.makedirs(os.path.dirname(snap), exist_ok=True)
            shutil.copytree(adapter_dir, snap, dirs_exist_ok=True)
        if a.smoke and step >= a.smoke:
            break

    logf.close()
    if a.smoke:
        recs = [json.loads(l) for l in open(log_path)]
        problems = []
        if not recs:
            problems.append("no step completed")
        if not any(r["trainable_rollouts"] for r in recs):
            problems.append("no group ever carried advantage: nothing was trained")
        if not any(r.get("loss") is not None for r in recs):
            problems.append("no gradient step ran")
        if not any((r.get("grad_norm") or 0) > 0 for r in recs):
            problems.append("every gradient step had grad_norm 0: the loss does not depend "
                            "on the adapter parameters")
        if not any((r.get("adapter_absB") or 0) > 0 for r in recs):
            problems.append("the adapter's LoRA-B is still exactly zero after training, so "
                            "the served adapter is arithmetically the base model")
        if not any(r.get("rollout_accuracy") is not None for r in recs):
            problems.append("no rollout was graded")
        if not route_verified:
            problems.append("the adapter route was never verified with a NON-null adapter")
        if problems:
            for p in problems:
                print("SMOKE FAIL: " + p, file=sys.stderr)
            return 1
        print("SMOKE PASS: %d steps, %d trainable rollouts, route verified"
              % (len(recs), sum(r["trainable_rollouts"] for r in recs)))
    if a.gcs:
        os.system("gsutil -mq rsync -r %s %s/%s" % (a.run_dir, a.gcs, a.arm))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
