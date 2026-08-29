#!/usr/bin/env python3
"""Verify that `min_tokens` on a ChatCompletionRequest reaches sglang's sampling params.

The fix adds extra_body={"min_tokens": 1} to the GSM8K workflow kwargs. Two things must
hold for that to do anything:
  1. `min_tokens` is an accepted field on sglang's ChatCompletionRequest (not silently
     dropped as an unknown extra), and
  2. to_sampling_params() maps it onto `min_new_tokens`.

If either fails the fix is inert -- the same trap as AReaL's dead gconfig.min_new_tokens.
This checks both directly, and also confirms the WRONG name is silently ignored, which is
the failure mode being guarded against.
"""
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

msgs = [{"role": "user", "content": "hi"}]
fails = []

# 1. correct field name is accepted and mapped
req = ChatCompletionRequest(model="default", messages=msgs, min_tokens=1)
sp = req.to_sampling_params(stop=[], model_generation_config={})
got = sp.get("min_new_tokens")
print(f"min_tokens=1        -> sampling_params['min_new_tokens'] = {got!r}")
if got != 1:
    fails.append(f"expected min_new_tokens=1, got {got!r}")

# 2. default really is 0, so the run so far had no floor at all
req0 = ChatCompletionRequest(model="default", messages=msgs)
got0 = req0.to_sampling_params(stop=[], model_generation_config={}).get("min_new_tokens")
print(f"(default)           -> sampling_params['min_new_tokens'] = {got0!r}")
if got0 != 0:
    fails.append(f"expected default 0, got {got0!r}")

# 3. the WRONG name must be shown to be inert, so the choice of name is justified
try:
    reqw = ChatCompletionRequest(model="default", messages=msgs, min_new_tokens=1)
    gotw = reqw.to_sampling_params(stop=[], model_generation_config={}).get("min_new_tokens")
    print(f"min_new_tokens=1    -> sampling_params['min_new_tokens'] = {gotw!r}"
          f"   ({'IGNORED as expected' if gotw == 0 else 'unexpectedly honoured'})")
    if gotw != 0:
        fails.append("min_new_tokens was honoured; the comment in gsm8k_rl.py is wrong")
except Exception as e:
    print(f"min_new_tokens=1    -> rejected by the model: {type(e).__name__}")

# 4. max_new_tokens still comes from max_completion_tokens, so we did not disturb it
reqm = ChatCompletionRequest(model="default", messages=msgs, max_completion_tokens=1024)
gotm = reqm.to_sampling_params(stop=[], model_generation_config={}).get("max_new_tokens")
print(f"max_completion=1024 -> sampling_params['max_new_tokens']  = {gotm!r}")
if gotm != 1024:
    fails.append(f"expected max_new_tokens=1024, got {gotm!r}")

print()
if fails:
    print("FAILED:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print("PASS: min_tokens=1 reaches sglang as min_new_tokens=1; default was 0.")
