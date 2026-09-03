"""Model clients. The base model is a parameter, never a constant.

Two implementations:

* `StubClient` -- deterministic, CPU-only, no network. Every CPU-side assertion in this
  directory runs against it, so none of them depends on any GPU or on whether a
  particular base model loads in our stack.
* `OpenAICompatClient` -- an OpenAI-compatible endpoint (SGLang/vLLM). It verifies at
  construction time that the requested model id is actually served, because an
  unregistered id is answered by the base model silently with a 200 and no warning.

Target base model is `Qwen/Qwen3.8-27B`; fallback `Qwen/Qwen3-32B`; last resort
`Qwen2.5-32B-Instruct`. Whether Qwen3.8-27B loads, serves and trains in this stack is an
open question being answered elsewhere, and nothing here assumes an answer.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from typing import Protocol

from .guards import assert_model_served, assert_token_budget_fits


class LLMClient(Protocol):
    """Minimal interface the loop needs from a model."""

    model_id: str

    def generate(self, prompt: str, max_new_tokens: int, seed: int) -> tuple[str, bool]:
        """Generate a completion.

        Args:
            prompt: The prompt.
            max_new_tokens: Generation cap.
            seed: Seed for reproducibility.

        Returns:
            `(text, truncated)` where `truncated` is True when the generation hit the cap
            or was otherwise cut off. The caller maps `truncated` to
            `RolloutOutcome.ABORTED`; it must never be silently graded as a wrong answer.
        """
        ...


class StubClient:
    """Deterministic offline client used for every CPU-side assertion.

    It is *not* a mock that returns a fixed string: it derives its output from a hash of
    the prompt and seed, so different prompts give different outputs and a stage that
    failed to pass its inputs through is detectable. It can also be told to abort a
    given fraction of generations, so the abort-handling paths are exercised.

    Args:
        model_id: Recorded in artifacts as the model that answered.
        abort_rate: Fraction of generations returned with `truncated=True`.
        success_rate: Probability that a generated rollout is written so that the
            scaffold's grader will mark it a success. Lets a test drive `p_hat` to a
            chosen value.
    """

    def __init__(
        self,
        model_id: str = "stub/deterministic-v1",
        abort_rate: float = 0.0,
        success_rate: float = 0.5,
    ) -> None:
        if not (0.0 <= abort_rate <= 1.0):
            raise ValueError(f"abort_rate must be in [0,1], got {abort_rate}")
        if not (0.0 <= success_rate <= 1.0):
            raise ValueError(f"success_rate must be in [0,1], got {success_rate}")
        self.model_id = model_id
        self.abort_rate = abort_rate
        self.success_rate = success_rate
        self.calls: list[tuple[str, int]] = []

    def generate(self, prompt: str, max_new_tokens: int, seed: int) -> tuple[str, bool]:
        """Return a deterministic pseudo-completion derived from prompt and seed."""
        self.calls.append((prompt, seed))
        h = hashlib.sha256(f"{prompt}|{seed}".encode()).digest()
        rng = random.Random(int.from_bytes(h[:8], "big"))
        if rng.random() < self.abort_rate:
            return "", True
        marker = "SOLVED" if rng.random() < self.success_rate else "ATTEMPT"
        return f"{marker} {h[:4].hex()}", False


class OpenAICompatClient:
    """Client for an OpenAI-compatible endpoint, with the served-model check enforced.

    Args:
        base_url: Endpoint ending in `/v1`.
        model_id: The model id to request.
        served_models: Ids reported by `GET /v1/models`. Passed in rather than fetched so
            that the check is testable offline and so the caller records what it saw.
        served_context_len: Context length the backend reports, used for the token-budget
            guard. Must come from the backend, not from our config.
        api_key: Bearer token; "EMPTY" for unauthenticated local services.

    Raises:
        GuardViolation: if `model_id` is not among `served_models`.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        served_models: Sequence[str],
        served_context_len: int,
        api_key: str = "EMPTY",
    ) -> None:
        assert_model_served(model_id, served_models)
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.served_models = list(served_models)
        self.served_context_len = served_context_len
        self.api_key = api_key

    def generate(self, prompt: str, max_new_tokens: int, seed: int) -> tuple[str, bool]:
        """Generate against the endpoint, refusing an over-budget request first.

        Raises:
            GuardViolation: if the token budget does not fit the served context.
            RuntimeError: if `httpx` is unavailable or the endpoint errors.
        """
        approx_prompt_tokens = max(1, len(prompt) // 4)
        assert_token_budget_fits(
            approx_prompt_tokens, max_new_tokens, self.served_context_len
        )
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("httpx is required for OpenAICompatClient") from exc

        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "seed": seed,
            },
            timeout=600.0,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        text = choice["message"]["content"] or ""
        # length -> hit the cap; content_filter/None -> did not finish normally.
        truncated = choice.get("finish_reason") != "stop"
        return text, truncated
