#!/usr/bin/env python3
"""Shared pieces of the gate-outcome experiment: prompting, generation, grading, selectors.

Grading is NOT reimplemented here. ``math_bench.py`` is imported and its ``load``,
``extract_boxed``, ``grade`` and ``wilson`` are used unchanged, so every success rate in this
experiment is produced by the same grader as every other number in the paper. A second
grader would silently move every p-hat, every dead-group fraction and every accuracy.

Two properties of this stack shape the generation path:

* an adapter is reached ONLY through ``/generate``'s ``lora_path``. The OpenAI-compatible
  route takes a ``model`` field, and naming an adapter there returns base-model output with
  HTTP 200 and no warning -- so the whole experiment would silently compare four identical
  checkpoints. ``assert_adapter_routes`` is the check that this cannot happen unnoticed.
* the thinking budget is a chat-template argument (``reasoning_effort``), not a sampling
  parameter, so it is applied client-side by the tokenizer and travels in the prompt text.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MATH_EVAL_DATA", "/mnt/localssd/gate/data/azr/evaluation/math_eval/eval/data")
import math_bench  # noqa: E402  (path must be set first)

# One instruction string is used for the difficulty measurement, for training rollouts and
# for the held-out evaluation, because p-hat and the outcome have to be measured on the same
# distribution. It differs from ``math_bench.PROMPT`` by one word ("problem" rather than "math
# problem"); that is recorded rather than silently fixed, because the discovery run was already
# generated with it and re-generating to recover a word would cost more than it buys. Nothing
# here is compared to a published OlympiadBench number: those are greedy at a different
# thinking budget and a different cap, so they are not comparable for other reasons anyway.
INSTRUCTION = (
    "Solve the following problem. Reason step by step, and put your final answer within "
    "\\boxed{}.\n\n"
)


def build_prompt(tok, problem: str, effort: str = "low") -> str:
    """Render one problem through the model's chat template at a chosen thinking budget.

    Args:
        tok: A ``transformers`` tokenizer for the base model.
        problem: The problem statement.
        effort: ``reasoning_effort``; the template accepts ``xhigh`` (its default),
            ``medium`` and ``low`` and raises on anything else.

    Returns:
        The prompt string to send to ``/generate``.
    """
    return tok.apply_chat_template(
        [{"role": "user", "content": INSTRUCTION + problem}],
        tokenize=False, add_generation_prompt=True,
        chat_template_kwargs={"reasoning_effort": effort},
    )


@dataclass
class GenSpec:
    """Everything that changes what the server produces, recorded with every generation."""

    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 0.95
    effort: str = "low"
    lora_path: str = ""
    stop: tuple = ()
    want_logprob: bool = False

    def sampling_params(self) -> dict:
        """The sglang ``sampling_params`` block this spec implies."""
        return {"max_new_tokens": self.max_new_tokens, "temperature": self.temperature,
                "top_p": self.top_p, "skip_special_tokens": True}


async def one_generation(session, base_url: str, prompt: str, spec: GenSpec,
                         timeout: float = 3600.0) -> dict:
    """POST a single completion to sglang's native ``/generate``.

    Returns:
        ``{"text", "finish", "completion_tokens", "prompt_tokens"}``. ``finish`` is
        ``"length"`` when the generation hit the cap, which callers must treat as UNKNOWN
        rather than as a wrong answer.
    """
    payload: dict = {"text": prompt, "sampling_params": spec.sampling_params()}
    if spec.lora_path:
        payload["lora_path"] = spec.lora_path
    if getattr(spec, "want_logprob", False):
        payload["return_logprob"] = True
    async with session.post(base_url.rstrip("/") + "/generate", json=payload,
                            timeout=aiohttp.ClientTimeout(total=timeout)) as r:
        body = await r.text()
        if r.status != 200:
            raise RuntimeError("generate HTTP %d: %s" % (r.status, body[:400]))
        d = json.loads(body)
    mi = d.get("meta_info", {})
    fin = mi.get("finish_reason", {})
    return {"text": d.get("text", ""),
            "finish": fin.get("type") if isinstance(fin, dict) else str(fin),
            "completion_tokens": mi.get("completion_tokens"),
            "prompt_tokens": mi.get("prompt_tokens"),
            "logprobs": [lp for lp, _, _ in (mi.get("output_token_logprobs") or [])]}


async def generate_many(base_url: str, jobs: list[dict], spec: GenSpec, concurrency: int,
                        out_path: str, progress_every: float = 60.0) -> int:
    """Run a list of generation jobs, appending each result as it lands.

    Each job is a dict carrying at least ``prompt``; every other key is copied verbatim into
    the output row, which is how a partial file stays interpretable.

    A caution that has already cost this project a wrong number: results arrive in COMPLETION
    order, and short generations finish first, so ANY rate computed from a partial file is
    biased. Rows carry their job keys precisely so a partial file can be detected and
    rejected rather than averaged.

    Args:
        base_url: e.g. ``http://127.0.0.1:30010``.
        jobs: Job dicts; ``prompt`` is consumed, the rest is echoed into the row.
        spec: Sampling specification, recorded once in the sidecar file.
        concurrency: Maximum simultaneous requests.
        out_path: JSONL to append to; existing rows are treated as already done and skipped
            by ``(idx, rep)``.
        progress_every: Seconds between progress lines.

    Returns:
        Number of NEW rows written.
    """
    done: set = set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done.add((r.get("idx"), r.get("rep")))
                except Exception:
                    pass
    todo = [j for j in jobs if (j.get("idx"), j.get("rep")) not in done]
    json.dump({"spec": spec.__dict__, "n_jobs": len(jobs), "n_todo": len(todo),
               "started": time.time()},
              open(out_path + ".spec.json", "w"), indent=1, default=str)
    if not todo:
        return 0

    sem = asyncio.Semaphore(concurrency)
    fh = open(out_path, "a")
    written = 0
    t0 = time.time()
    last = [t0]
    lock = asyncio.Lock()

    async def run(job):
        nonlocal written
        async with sem:
            prompt = job["prompt"]
            for attempt in range(3):
                try:
                    res = await one_generation(session, base_url, prompt, spec)
                    break
                except Exception as exc:  # a spot box can drop a request; retry, then record
                    if attempt == 2:
                        res = {"text": "", "finish": "error", "error": repr(exc)[:300],
                               "completion_tokens": None, "prompt_tokens": None}
                    else:
                        await asyncio.sleep(2.0 * (attempt + 1))
        row = {k: v for k, v in job.items() if k != "prompt"}
        row.update(res)
        async with lock:
            fh.write(json.dumps(row) + "\n")
            written += 1
            if written % 64 == 0:
                fh.flush()
            now = time.time()
            if now - last[0] > progress_every:
                last[0] = now
                print("[gen] %d/%d in %.0fs" % (written, len(todo), now - t0), flush=True)

    conn = aiohttp.TCPConnector(limit=concurrency + 8)
    async with aiohttp.ClientSession(connector=conn) as session:
        await asyncio.gather(*(run(j) for j in todo))
    fh.close()
    print("[gen] DONE %d new rows in %.0fs -> %s" % (written, time.time() - t0, out_path),
          flush=True)
    return written


async def assert_adapter_routes(base_url: str, prompt: str, lora_name: str,
                                n: int = 8, max_new_tokens: int = 32) -> dict:
    """Prove an adapter is applied, on LOGPROBS rather than on the text it happens to produce.

    The first version of this compared greedy TEXT and it was the wrong instrument: after one
    gradient step a rank-32 adapter perturbs the policy far too little to change an argmax over
    32 tokens, so the check failed on a perfectly live adapter (arm C1, step 0) while passing on
    another (C2) whose first update happened to be larger. A guard whose verdict depends on the
    size of the first update is not measuring the route.

    Logprobs are the sensitive version of the same question. Greedy decoding on this server is
    exactly reproducible -- two identical requests return identical logprobs to the last bit --
    so ANY non-zero difference under ``lora_path`` is the adapter, and the base-vs-base repeat
    is carried as the control that says so.

    Two assertions:

    1. base repeated == base, to the bit. Without this, "different" means nothing.
    2. ``lora_path=<name>`` differs from base. If it does not, the adapter is not reaching the
       forward pass and every arm would serve the same model.

    Raises:
        AssertionError: If either fails.
    """
    g = dict(max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0, want_logprob=True)
    out: dict = {}
    async with aiohttp.ClientSession() as session:
        # Warm-up, discarded. MEASURED: the FIRST request to an idle server returns logprobs
        # that differ from every later identical one by ~0.05-0.07, because it lands in a
        # differently shaped batch and bf16 reduces in a different order. Without discarding
        # it the control below ("base repeated equals base") fails and the check reports a
        # false alarm on a perfectly live adapter.
        for _ in range(3):
            await one_generation(session, base_url, prompt, GenSpec(**g))
        base = await one_generation(session, base_url, prompt, GenSpec(**g))
        base2 = await one_generation(session, base_url, prompt, GenSpec(**g))
        lora = await one_generation(session, base_url, prompt, GenSpec(lora_path=lora_name, **g))

    def delta(x, y):
        """Largest absolute logprob difference over the positions both produced."""
        m = min(len(x.get("logprobs") or []), len(y.get("logprobs") or []))
        if m == 0:
            return None
        return max(abs(a - b) for a, b in zip(x["logprobs"][:m], y["logprobs"][:m]))

    out["base_repeat_delta"] = delta(base, base2)
    out["adapter_delta"] = delta(lora, base)
    out["text_also_changed"] = lora["text"] != base["text"]
    out["base_text"] = base["text"][:160]
    out["lora_text"] = lora["text"][:160]
    if out["base_repeat_delta"] is None or out["adapter_delta"] is None:
        raise AssertionError("the server returned no logprobs, so the route cannot be checked "
                             "this way: %s" % out)
    if out["base_repeat_delta"] != 0.0:
        raise AssertionError(
            "greedy decoding is not reproducible on this server (base repeat differs by %g), "
            "so a difference under lora_path cannot be attributed to the adapter"
            % out["base_repeat_delta"])
    if out["adapter_delta"] == 0.0:
        raise AssertionError(
            "lora_path=%s produced logprobs IDENTICAL to the base model. The adapter is not "
            "reaching the forward pass: every arm would train a different adapter and serve "
            "the same model." % lora_name)
    return out
