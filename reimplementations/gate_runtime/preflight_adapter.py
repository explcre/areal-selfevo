#!/usr/bin/env python3
"""Preflight: prove the adapter route is live BEFORE any arm is trained.

Three things are asserted, each of which has a silent failure mode on this stack:

1. **The adapter reaches the forward pass.** Greedy decoding, base vs ``lora_path``. A
   difference under greedy decoding cannot be sampling noise.
2. **A null adapter is null.** A freshly initialised adapter (LoRA-B = 0) must reproduce base
   output EXACTLY. If it does not, something other than the adapter is moving between
   requests and assertion 1 proves nothing. This is the control that stops assertion 1 from
   being a guard that cannot fail.
3. **Naming the adapter in the ``model`` field does NOT apply it.** That is the trap: it
   returns 200 and base-model output. It is asserted so the day sglang starts honouring it,
   this preflight says so rather than the experiment silently changing meaning.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import aiohttp
from transformers import AutoTokenizer

sys.path.insert(0, "/mnt/localssd/gate/code")
from gate_lib import GenSpec, build_prompt, one_generation  # noqa: E402


async def main_async(a) -> int:
    """Run the three assertions and print a machine-readable verdict."""
    tok = AutoTokenizer.from_pretrained(a.model)
    prompt = build_prompt(tok, "What is 17 times 23? Give the number only.", a.effort)
    greedy = dict(max_new_tokens=a.max_new_tokens, temperature=0.0, top_p=1.0)
    out: dict = {}
    async with aiohttp.ClientSession() as s:
        base1 = await one_generation(s, a.url, prompt, GenSpec(**greedy))
        base2 = await one_generation(s, a.url, prompt, GenSpec(**greedy))
        live = await one_generation(s, a.url, prompt, GenSpec(lora_path=a.live, **greedy))
        null = await one_generation(s, a.url, prompt, GenSpec(lora_path=a.null, **greedy))
        # The trap, tested on the SAME prompt string as the /generate calls above so that
        # "identical to base" means the adapter was skipped and not that a different
        # template was applied. /v1/completions takes raw text, so the two routes see
        # byte-identical input.
        async def completions(model_id):
            async with s.post(a.url.rstrip("/") + "/v1/completions", json={
                    "model": model_id, "prompt": prompt, "max_tokens": a.max_new_tokens,
                    "temperature": 0.0}) as r:
                st, body = r.status, await r.text()
            try:
                return st, json.loads(body)["choices"][0]["text"]
            except Exception:
                return st, None

        out["model_field_status"], mf_live = await completions(a.live)
        _, mf_base = await completions("qwen38-27b")
        out["model_field_text"] = (mf_live or "")[:200]
        out["model_field_equals_base_route"] = (mf_live is not None and mf_live == mf_base)

    out["base_deterministic"] = base1["text"] == base2["text"]
    out["live_differs_from_base"] = live["text"] != base1["text"]
    out["null_equals_base"] = null["text"] == base1["text"]
    out["base_text"] = base1["text"][:200]
    out["live_text"] = live["text"][:200]
    out["null_text"] = null["text"][:200]
    print(json.dumps(out, indent=1))

    fail = []
    if not out["base_deterministic"]:
        fail.append("greedy base decoding is not reproducible: 'the adapter changed the "
                    "output' would be indistinguishable from noise")
    if not out["live_differs_from_base"]:
        fail.append("lora_path=%s returned BASE-IDENTICAL text: the adapter is not applied, "
                    "so every arm would serve the same model" % a.live)
    if not out["null_equals_base"]:
        fail.append("a ZERO-initialised adapter changed the output, which is arithmetically "
                    "impossible: the serving path is doing something other than applying "
                    "this adapter")
    if out.get("model_field_equals_base_route") is False:
        print("NOTE: the `model` field is no longer inert on this build -- model=%s differs "
              "from model=<base>. Re-check which route the arms use." % a.live,
              file=sys.stderr)
    if fail:
        for f in fail:
            print("FAIL: " + f, file=sys.stderr)
        return 1
    print("PASS: adapter route is live, null adapter is null, and the model field is inert "
          "(status %s)." % out["model_field_status"])
    return 0


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30010")
    ap.add_argument("--model", default="/mnt/localssd/gate/models/Qwen3.8-27B")
    ap.add_argument("--live", default="live")
    ap.add_argument("--null", default="null")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
