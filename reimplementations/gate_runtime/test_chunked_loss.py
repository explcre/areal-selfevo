#!/usr/bin/env python3
"""Chunked loss accumulation: same gradient, lower peak, and a row that used to be untrainable.

Three assertions, because any one alone is passable by a no-op. Equal gradients alone are what
you get when the optimisation did not happen. A lower peak alone is what you get when the
gradient is wrong. And both together on a short row say nothing about the case that motivated
the change, so the third case is a single row long enough to have OOM'd before.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

sys.path.insert(0, "/mnt/localssd/gate/code")

import train_arm  # noqa: E402


def grads_of(model):
    """Snapshot every trainable gradient, detached on CPU."""
    return {n: p.grad.detach().float().cpu().clone()
            for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}


def run_once(model, rows, device, pad_id, chunk, max_len, mb_tokens):
    """Zero grads, run one backward pass at the given chunk size, return grads and peak GiB."""
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)
    out = train_arm.policy_step(model, rows, opt, device, max_len, 1e9, mb_tokens, pad_id,
                                "sequence", logit_chunk=chunk)
    return grads_of(model), torch.cuda.max_memory_allocated() / 2**30, out


def main() -> int:
    """Compare chunked against unchunked, then train a row that previously did not fit."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/localssd/gate/models/Qwen3.8-27B")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--long-len", type=int, default=41000)
    ap.add_argument("--out", default="/mnt/localssd/gate/out/chunked_loss_test.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    device = "cuda:0"
    print("[chunk] loading base model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 device_map=device)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=a.rank, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=train_arm.TARGET_MODULES, exclude_modules=train_arm.EXCLUDE))
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # LoRA-B starts at zero, so the gradient would be identical for trivial reasons. Give the
    # adapter a real perturbation first, or this test compares two zeros.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0.0, 0.01)
    assert train_arm.adapter_fingerprint(model) > 0, "adapter must be non-trivial for this test"

    g = torch.Generator().manual_seed(7)
    def row(plen, olen):
        v = int(tok.vocab_size if hasattr(tok, "vocab_size") else 248320)
        ids = torch.randint(0, min(v, 200000), (plen + olen,), generator=g).tolist()
        return {"prompt_ids": ids[:plen], "output_ids": ids[plen:],
                "adv": float(torch.randn(1, generator=g))}

    res = {}

    # --- case 1: identical gradients on a batch small enough for both paths ---------
    rows = [row(64, 700), row(48, 500), row(96, 900)]
    gc, pc, oc = run_once(model, rows, device, pad_id, a.chunk, 4096, 8192)
    gf, pf, of = run_once(model, rows, device, pad_id, 0, 4096, 8192)
    gf2, _, of2 = run_once(model, rows, device, pad_id, 0, 4096, 8192)
    keys = sorted(set(gc) & set(gf) & set(gf2))

    def reldiff(x, y):
        """Largest elementwise difference, as a fraction of the largest gradient."""
        num = max(float((x[k] - y[k]).abs().max()) for k in keys)
        den = max(float(y[k].abs().max()) for k in keys)
        return num / max(den, 1e-12)

    noise = reldiff(gf, gf2)          # SAME path twice: the floor this must be read against
    rel = reldiff(gc, gf)
    res["equivalence"] = {"n_params": len(keys), "max_rel_diff_chunked_vs_full": rel,
                          "max_rel_diff_full_vs_full": noise,
                          "ratio_to_noise": rel / max(noise, 1e-12),
                          "loss_chunked": oc["loss"], "loss_full": of["loss"],
                          "grad_norm_chunked": oc["grad_norm"],
                          "grad_norm_full": of["grad_norm"],
                          "grad_norm_full_again": of2["grad_norm"]}
    print("[chunk] chunked vs full  : %.3e relative, over %d tensors" % (rel, len(keys)),
          flush=True)
    print("[chunk] full vs full     : %.3e relative  <- same-path noise floor" % noise,
          flush=True)
    print("[chunk] grad norms       : chunked %.6f, full %.6f, full again %.6f"
          % (oc["grad_norm"], of["grad_norm"], of2["grad_norm"]), flush=True)

    # --- case 2: the peak must actually fall on a long row --------------------------
    long_rows = [row(200, a.long_len - 200)]
    gcl, pcl, ocl = run_once(model, long_rows, device, pad_id, a.chunk, a.long_len + 8, 65536)
    res["long_row_chunked"] = {"tokens": a.long_len, "peak_gib": round(pcl, 1),
                               "grad_norm": ocl["grad_norm"], "loss": ocl["loss"]}
    print("[chunk] long row (%d tokens) chunked: peak %.1f GiB, grad_norm %.6f"
          % (a.long_len, pcl, ocl["grad_norm"]), flush=True)

    full_ok, full_peak, full_err = True, None, None
    try:
        _, full_peak, _ = run_once(model, long_rows, device, pad_id, 0, a.long_len + 8, 65536)
    except torch.OutOfMemoryError as exc:
        full_ok, full_err = False, str(exc)[:160]
    res["long_row_unchunked"] = {"fits": full_ok, "peak_gib": full_peak, "error": full_err}
    print("[chunk] long row unchunked: %s" % ("peak %.1f GiB" % full_peak if full_ok
                                              else "OOM as expected -- " + str(full_err)[:90]),
          flush=True)

    res["peaks"] = {"small_chunked_gib": round(pc, 1), "small_full_gib": round(pf, 1)}
    json.dump(res, open(a.out, "w"), indent=1)

    probs = []
    # Judged against the same-path floor, not against a tolerance chosen to pass. A factor of
    # three over run-to-run noise would say chunking changed the arithmetic; equal to it says
    # the difference IS run-to-run noise.
    if rel > max(3.0 * noise, 1e-3):
        probs.append("chunked gradients differ by %.3e, more than 3x the same-path noise "
                     "floor of %.3e" % (rel, noise))
    if not (ocl["grad_norm"] > 0):
        probs.append("the long row produced no gradient")
    if full_ok and full_peak is not None and pcl >= full_peak:
        probs.append("chunking did not lower the peak on the long row (%.1f vs %.1f GiB)"
                     % (pcl, full_peak))
    if probs:
        for p in probs:
            print("FAIL: " + p)
        return 1
    print("\nPASS: chunked-vs-full %.1e relative against a same-path noise floor of %.1e "
          "(ratio %.2f), and a %d-token row that %s now trains at %.1f GiB."
          % (rel, noise, rel / max(noise, 1e-12), a.long_len,
                         "needed %.1f GiB" % full_peak if full_ok else "did not fit at all",
                         pcl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
