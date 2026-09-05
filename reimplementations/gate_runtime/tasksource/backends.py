"""Generation backends, provider-agnostic, with paid providers barred by default.

The teacher is designed so a hosted model could serve it, and is TESTED against the local
one. `HostedTeacher` exists so the interface is real rather than hypothetical, and it refuses
to make a call unless spending has been authorised explicitly, because this project's standing
rule is that paid API use needs permission and "the PI raised the possibility" is not that.
"""
from __future__ import annotations

import asyncio
import os
import sys


class SGLangBackend:
    """Local model behind an sglang server. The default for every source."""

    def __init__(self, url: str, tok, cap: int = 4096, concurrency: int = 24,
                 lora: str = "", name: str = "local", effort: str = "low"):
        self.url, self.tok, self.cap = url, tok, cap
        self.concurrency, self.lora, self.name, self.effort = concurrency, lora, name, effort

    def generate(self, prompts) -> tuple[list[str], int]:
        """Return `(texts, generated_tokens)` for a batch of prompts."""
        sys.path.insert(0, "/mnt/localssd/gate/code")
        import ornith_train as ot
        recs = asyncio.run(ot.gen_batch(self.url, self.tok, prompts, self.cap,
                                        min(len(prompts), self.concurrency), self.lora))
        return ([r["text"] for r in recs],
                sum(len(r["output_ids"]) for r in recs))


class HostedTeacher:
    """A hosted, PAID teacher. Never called unless spending is explicitly authorised.

    The guard is the point. `estimate_cost` is always available so a run can be PRICED
    without spending anything, which is what a request for authorisation needs.
    """

    #: Set this environment variable to the literal string below to permit a paid call.
    AUTH_ENV = "SELFEVO_PAID_API_AUTHORISED"
    AUTH_VALUE = "yes-i-have-permission"

    def __init__(self, model: str, price_in_per_mtok: float, price_out_per_mtok: float,
                 name: str = "hosted"):
        self.model, self.name = model, name
        self.price_in, self.price_out = price_in_per_mtok, price_out_per_mtok

    def estimate_cost(self, n_prompts: int, prompt_tokens: int, output_tokens: int) -> dict:
        """Price a run from MEASURED token counts, without contacting anyone."""
        cin = n_prompts * prompt_tokens / 1e6 * self.price_in
        cout = n_prompts * output_tokens / 1e6 * self.price_out
        return {"model": self.model, "n_prompts": n_prompts,
                "prompt_tokens_each": prompt_tokens, "output_tokens_each": output_tokens,
                "usd_input": round(cin, 4), "usd_output": round(cout, 4),
                "usd_total": round(cin + cout, 4)}

    def generate(self, prompts):
        """Refuse unless spending has been authorised for this process."""
        if os.environ.get(self.AUTH_ENV) != self.AUTH_VALUE:
            raise PermissionError(
                "HostedTeacher would spend money on %r and paid API use is not authorised. "
                "Nothing was sent. Price the run with estimate_cost() and obtain explicit "
                "permission; then set %s=%s."
                % (self.model, self.AUTH_ENV, self.AUTH_VALUE))
        raise NotImplementedError(
            "no hosted provider is wired: this class exists so the teacher interface is "
            "provider-agnostic and priceable, and so that a paid call cannot happen by "
            "accident. Wire a provider here only after permission is on record.")
