#!/usr/bin/env python3
"""Verify min_new_tokens traverses EVERY boundary, not just the endpoints.

Three earlier attempts at this fix each confirmed the value at one layer and asserted it
about the whole path. This checks each hop explicitly and names the hop that fails.
"""
import inspect
import sys

from areal.api.cli_args import AgentConfig, GenerationHyperparameters
from areal.api.io_struct import ModelRequest
from areal.engine.sglang_remote import SGLangBackend
from areal.experimental.openai import client as C
from areal.experimental.openai.proxy import proxy_rollout_server as PR

fails = []


def check(n: int, desc: str, ok: bool) -> None:
    print(f"{n}. {desc}: {'OK' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"hop {n}: {desc}")


check(1, "AgentConfig declares min_new_tokens",
      "min_new_tokens" in AgentConfig.__dataclass_fields__)

check(2, "proxy passes AgentConfig.min_new_tokens into ArealOpenAI",
      "min_new_tokens=agent_cfg.min_new_tokens" in inspect.getsource(PR))

check(3, "ArealOpenAI accepts it",
      "min_new_tokens" in inspect.signature(C.ArealOpenAI.__init__).parameters)

check(4, "ArealOpenAI forwards it to its sub-clients",
      "min_new_tokens=min_new_tokens" in inspect.getsource(C.ArealOpenAI.__init__))

# ArealOpenAI forwards to BOTH sub-clients, so BOTH must accept it. Checking only the
# one that was patched is how step0d died at startup with
#   TypeError: AsyncResponsesWithReward.__init__() got an unexpected keyword argument
# Every construction site on the path is a hop, not just the one being fixed.
for _cls in (C.AsyncCompletionsWithReward, C.AsyncResponsesWithReward):
    check(5, f"{_cls.__name__} accepts it",
          "min_new_tokens" in inspect.signature(_cls.__init__).parameters)

check(6, "AsyncCompletionsWithReward stores it",
      "self.min_new_tokens = min_new_tokens"
      in inspect.getsource(C.AsyncCompletionsWithReward.__init__))

check(7, "its create() writes it into the gconfig",
      "min_new_tokens=self.min_new_tokens"
      in inspect.getsource(C.AsyncCompletionsWithReward.create))

check(8, "SGLangBackend forwards gconfig.min_new_tokens",
      "gconfig.min_new_tokens" in inspect.getsource(SGLangBackend.build_generation_request))

# Hop 9 is the only one that is a behavioural test rather than a source check.
g = GenerationHyperparameters(max_new_tokens=1024, min_new_tokens=1)
req = ModelRequest(input_ids=[1, 2, 3], gconfig=g, rid="t")
http = SGLangBackend.build_generation_request(None, req, False, 0)
body = http.payload if hasattr(http, "payload") else http.json
got = body["sampling_params"].get("min_new_tokens")
check(9, f"sglang payload carries it (got {got!r})", got == 1)

print()
if fails:
    print("CHAIN BROKEN at:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("CHAIN COMPLETE: AgentConfig -> proxy -> ArealOpenAI -> AsyncCompletionsWithReward")
print("                -> gconfig -> SGLangBackend -> sglang sampling_params")
