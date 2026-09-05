#!/usr/bin/env python3
"""Separate three causes of a logprob shift under batching: padding, batching, and bf16.

The trainer's no-op assertion fired at 0.079. That number alone cannot say whether right
padding leaks into the recurrent (gated-delta-net) layers or whether bf16 matmuls simply
reduce in a different order once the batch dimension changes. The two have opposite fixes, so
they are separated here by construction:

  A. batch-only   : score [a] alone, then [a, a]. Same content, equal lengths, NO padding.
                    Any shift here is batching/reduction order, not padding.
  B. padding-only : score [a] alone, then [a] padded to a longer length, batch size 1.
                    Any shift here is padding.
  C. both         : [a] vs [a, b] padded, which is what the trainer actually does.
  D. bf16 floor   : the same call twice, which bounds pure nondeterminism.
"""
from __future__ import annotations

import json
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

M = "/mnt/localssd/gate/models/Qwen3.8-27B"
dev = "cuda:0"
tok = AutoTokenizer.from_pretrained(M)
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map=dev)
model.eval()

a = tok("The capital of France is Paris, and the capital of Japan is Tokyo.",
        add_special_tokens=False)["input_ids"]
b = tok("Two plus two is four. Three plus three is six. Four plus four is eight. Indeed.",
        add_special_tokens=False)["input_ids"]


def score(seqs, pad_to=None):
    """Token logprobs for each sequence, under right padding to a common length."""
    L = pad_to or max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
    ids, att = ids.to(dev), att.to(dev)
    with torch.no_grad():
        lg = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1, :]
        lp = torch.log_softmax(lg.float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return [lp[i, : len(s) - 1].cpu() for i, s in enumerate(seqs)]


def shift(x, y):
    """Largest absolute per-token logprob difference."""
    return float((x - y).abs().max())


ref = score([a])[0]
out = {
    "D_bf16_floor_same_call_twice": shift(ref, score([a])[0]),
    "A_batch_only_[a,a]": shift(ref, score([a, a])[0]),
    "B_padding_only_batch1_padto_64": shift(ref, score([a], pad_to=64)[0]),
    "B_padding_only_batch1_padto_200": shift(ref, score([a], pad_to=200)[0]),
    "C_batch_and_padding_[a,b]": shift(ref, score([a, b])[0]),
    "len_a": len(a), "len_b": len(b),
    "mean_abs_logprob": float(ref.abs().mean()),
}
print(json.dumps(out, indent=1))
json.dump(out, open("/mnt/localssd/gate/out/diag_padding.json", "w"), indent=1)
