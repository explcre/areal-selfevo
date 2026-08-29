#!/usr/bin/env python3
"""Verify gconfig.min_new_tokens reaches the sglang payload on the LIVE code path.

Two earlier attempts at this fix failed because each verified the wrong layer:
  - the first set AReaL's gconfig.min_new_tokens, which was read nowhere;
  - the second sent extra_body={"min_tokens": 1}, which sglang's protocol does honour,
    but ArealOpenAI.create() rebuilds a fresh GenerationHyperparameters from named
    parameters (client.py:1134) and drops it long before the request is built.

So this exercises the function the running engine actually calls:
remote_inf_engine.py:973 does `self.backend.build_generation_request(req, with_lora,
version)` on an SGLangBackend, and we assert on the emitted sampling_params.
"""
import json

from areal.api.cli_args import GenerationHyperparameters
from areal.api.io_struct import ModelRequest
from areal.engine.sglang_remote import SGLangBackend

fails = []


def sampling_params_for(min_new_tokens: int) -> dict:
    g = GenerationHyperparameters(
        max_new_tokens=1024, min_new_tokens=min_new_tokens, temperature=1.0
    )
    req = ModelRequest(input_ids=[1, 2, 3], gconfig=g, rid="test")
    http = SGLangBackend.build_generation_request(None, req, False, 0)
    body = http.payload if hasattr(http, "payload") else http.json
    if isinstance(body, (str, bytes)):
        body = json.loads(body)
    return body["sampling_params"]


sp1 = sampling_params_for(1)
got1 = sp1.get("min_new_tokens")
print(f"gconfig.min_new_tokens=1 -> sampling_params['min_new_tokens'] = {got1!r}")
if got1 != 1:
    fails.append(f"expected 1 in the payload, got {got1!r}")

sp0 = sampling_params_for(0)
got0 = sp0.get("min_new_tokens")
print(f"gconfig.min_new_tokens=0 -> sampling_params['min_new_tokens'] = {got0!r}")
if got0 != 0:
    fails.append(f"expected 0 in the payload, got {got0!r}")

print(f"max_new_tokens still present: {sp1.get('max_new_tokens')!r}")
if sp1.get("max_new_tokens") != 1024:
    fails.append("max_new_tokens was disturbed")

print()
if fails:
    print("FAILED:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("PASS: min_new_tokens reaches the sglang sampling_params on the live path.")
