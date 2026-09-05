#!/usr/bin/env python3
"""Verify a served adapter is live, calibrated against a NULL adapter on the same server.

Neither naive version of this check works, and both failure modes were observed here:

* comparing greedy TEXT is too coarse. After one gradient step a rank-32 adapter does not move
  an argmax over 32 tokens, so the check failed on a live adapter (arm C1) and passed on
  another (C2) only because its first update happened to be larger.
* comparing logprobs for EQUALITY is too fine. Routing through the LoRA path adds
  ``B(A x) * s`` which is exactly zero for a freshly initialised adapter, but the extra GEMM
  changes the bf16 reduction order, so a provably null adapter still shifts logprobs.

So the null adapter is not a nuisance, it is the CALIBRATION: it measures what routing through
the LoRA path costs numerically when the adapter contributes nothing. A live adapter has to
beat that floor by a wide margin.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys

import aiohttp
import requests
from transformers import AutoTokenizer

sys.path.insert(0, "/mnt/localssd/gate/code")
from gate_lib import GenSpec, build_prompt, one_generation  # noqa: E402

NULL_DIR = "/mnt/localssd/gate/adapters/route_null"
PATTERN = (r"model\.language_model\.layers\.[0-9]+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
           r"|linear_attn\.out_proj|mlp\.(gate_proj|up_proj|down_proj))")
TARGETS = "q_proj,k_proj,v_proj,o_proj,out_proj,gate_proj,up_proj,down_proj"


async def logprobs(url, prompt, lora, n_tokens):
    """Greedy logprobs for a fixed prompt, through ``lora`` or through the base model."""
    async with aiohttp.ClientSession() as s:
        return await one_generation(s, url, prompt, GenSpec(
            max_new_tokens=n_tokens, temperature=0.0, top_p=1.0,
            lora_path=lora, want_logprob=True))


def dmax(x, y):
    """Largest absolute logprob difference over the positions both produced."""
    m = min(len(x["logprobs"]), len(y["logprobs"]))
    return max(abs(a - b) for a, b in zip(x["logprobs"][:m], y["logprobs"][:m])) if m else None


def dmean(x, y):
    """Mean absolute logprob difference; less hostage to a single token than the max."""
    m = min(len(x["logprobs"]), len(y["logprobs"]))
    return (sum(abs(a - b) for a, b in zip(x["logprobs"][:m], y["logprobs"][:m])) / m
            if m else None)


def main() -> int:
    """Compare each named adapter against base, with the null adapter as the floor."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", required=True, help="the arm's adapter name on this server")
    ap.add_argument("--model", default="/mnt/localssd/gate/models/Qwen3.8-27B")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--margin", type=float, default=10.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    subprocess.run([sys.executable, "/mnt/localssd/gate/code/mk_probe_adapter.py",
                    "--out", NULL_DIR, "--scale", "0", "--pattern", PATTERN,
                    "--target-list", TARGETS], capture_output=True, text=True, check=True)
    requests.post(a.url + "/unload_lora_adapter", json={"lora_name": "_null"}, timeout=300)
    r = requests.post(a.url + "/load_lora_adapter",
                      json={"lora_name": "_null", "lora_path": NULL_DIR}, timeout=600)
    if r.status_code != 200:
        print("could not load the calibration adapter: %s" % r.text[:200], file=sys.stderr)
        return 3

    tok = AutoTokenizer.from_pretrained(a.model)
    prompt = build_prompt(tok, "Find the sum of all positive integers n below 50 such that "
                               "n^2 + 3n + 2 is divisible by 7.", "low")
    # Warm-up. MEASURED: the FIRST request after a server has been idle returns logprobs that
    # differ from every later identical request by ~0.05, because the batch it lands in has a
    # different shape and bf16 reduces in a different order. Without discarding it the control
    # ("base repeated equals base") fails and the whole check reports a false alarm.
    for _ in range(a.warmup):
        asyncio.run(logprobs(a.url, prompt, "", a.tokens))
    base = asyncio.run(logprobs(a.url, prompt, "", a.tokens))
    base2 = asyncio.run(logprobs(a.url, prompt, "", a.tokens))
    null = asyncio.run(logprobs(a.url, prompt, "_null", a.tokens))
    live = asyncio.run(logprobs(a.url, prompt, a.name, a.tokens))
    requests.post(a.url + "/unload_lora_adapter", json={"lora_name": "_null"}, timeout=300)

    res = {
        "adapter": a.name,
        "base_repeat_delta": dmax(base, base2),
        "null_adapter_delta": dmax(null, base),
        "live_adapter_delta": dmax(live, base),
        "null_adapter_mean_delta": dmean(null, base),
        "live_adapter_mean_delta": dmean(live, base),
        "text_changed": live["text"] != base["text"],
    }
    floor = res["null_adapter_delta"] or 0.0
    res["ratio_live_over_null"] = (res["live_adapter_delta"] / floor) if floor else None
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    if res["base_repeat_delta"] != 0.0:
        print("FAIL: greedy decoding is not reproducible; nothing below is attributable",
              file=sys.stderr)
        return 1
    if res["live_adapter_delta"] <= max(a.margin * floor, 1e-6):
        print("FAIL: the served adapter %r moves logprobs by %.3g, against a null-adapter "
              "floor of %.3g. That is inside the numerical cost of routing through the LoRA "
              "path at all, so the adapter is not demonstrably reaching the forward pass."
              % (a.name, res["live_adapter_delta"], floor), file=sys.stderr)
        return 1
    print("PASS: %s moves logprobs %.1fx more than a provably null adapter on the same server."
          % (a.name, res["ratio_live_over_null"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
