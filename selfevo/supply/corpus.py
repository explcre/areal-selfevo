"""A correct row drawn from an offline pool of solved examples.

THE ONE DESIGN DECISION THAT MATTERS HERE, AND IT IS A REFUSAL. A corpus row is only a target
if it answers THIS prompt. The writer keeps the row's own prompt and splices the payload after
it (``_write_gold_row``: "the gold is a solution TO THAT PROMPT, and a target detached from
its question teaches the model to emit the derivation unconditionally"), so serving some other
prompt's solved example produces a row whose question and answer do not match -- a wrong row,
substituted with full confidence, indistinguishable downstream from a correct one. This
supplier therefore keys its pool on prompt identity and REFUSES with
:attr:`~selfevo.supply.base.Refusal.NO_MATCH` when the prompt is not in it. It is the cheapest
possible fabrication and the easiest to write by accident, so it is also a mutation the test
file kills.

The pool is keyed with :func:`selfevo.supply.base.key_for_prompt`, which routes to the
prompt-credit ledger's own hash, so an offline file built from prompt tokens and a live rollout
row of the same prompt land on the same digest by construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from selfevo.supply.base import (
    Refusal,
    SupplierRefused,
    SupplyConfigError,
    SupplyOffer,
    SupplyRequest,
    key_for_prompt,
)

__all__ = ["CorpusSupplier", "load_corpus_jsonl"]


def load_corpus_jsonl(
    path: str | Path, *, dtype: torch.dtype = torch.long
) -> dict[str, torch.Tensor]:
    """Read an offline pool of solved examples into a prompt-keyed mapping.

    Args:
        path: A JSON Lines file. Each line is an object with ``prompt_ids`` and
            ``response_ids`` (lists of integers) and optionally ``correct`` (bool).
        dtype: Dtype for the response tensors.

    Returns:
        ``{prompt digest: response token ids}``. Later lines win for a repeated prompt.

    Raises:
        SupplyConfigError: On a line missing either id list, or carrying an empty one, or
            marked ``correct: false``. A pool is a set of CORRECT rows; silently dropping a
            malformed line would make the pool's size a number nobody can reproduce, and
            silently keeping an incorrect one puts a wrong target one refactor away from a
            training batch.
    """
    pool: dict[str, torch.Tensor] = {}
    for lineno, line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        prompt = row.get("prompt_ids")
        resp = row.get("response_ids")
        if not prompt or not resp:
            raise SupplyConfigError(
                f"{path}:{lineno} has prompt_ids={prompt!r} response_ids={resp!r}; both must "
                "be non-empty, because a row that cannot be keyed or cannot be spliced is not "
                "a pool entry and must not be dropped in silence"
            )
        if row.get("correct") is False:
            raise SupplyConfigError(
                f"{path}:{lineno} is marked correct=false; a corpus holds SOLVED examples, "
                "and keeping an incorrect one puts a wrong target one refactor from a batch"
            )
        pool[key_for_prompt(prompt)] = torch.tensor([int(t) for t in resp], dtype=dtype)
    return pool


class CorpusSupplier:
    """Serves a solved example for this prompt from an offline pool.

    Args:
        pool: ``{prompt digest: response token ids}``, e.g. from :func:`load_corpus_jsonl`.
            Values may be tensors or sequences of ints.
        dtype: Dtype used when a value has to be converted.

    Raises:
        SupplyConfigError: On an entry whose response is empty.
    """

    name = "corpus"
    required_keys: tuple[str, ...] = ()

    def __init__(
        self,
        pool: Mapping[str, Sequence[int] | torch.Tensor],
        *,
        dtype: torch.dtype = torch.long,
    ) -> None:
        self._pool: dict[str, torch.Tensor] = {}
        for key, value in pool.items():
            t = value if torch.is_tensor(value) else torch.tensor(
                [int(v) for v in value], dtype=dtype
            )
            if t.ndim != 1 or t.numel() == 0:
                raise SupplyConfigError(
                    f"corpus entry {key!r} is {getattr(t, 'shape', None)}; a pool entry must "
                    "be a 1-D non-empty sequence of token ids"
                )
            self._pool[key] = t

    def __len__(self) -> int:
        """Number of prompts the pool covers."""
        return len(self._pool)

    def has_supply(self, batch: Mapping[str, Any]) -> bool:
        """Whether the pool holds anything at all.

        Args:
            batch: Unread; coverage is a property of the pool.

        Returns:
            True when the pool is non-empty. An empty pool behind a configured corpus arm is
            the silent-no-op case and is refused at batch level, not per group.
        """
        return bool(self._pool)

    def supply(self, request: SupplyRequest) -> SupplyOffer:
        """The pool's solved example for this prompt.

        Args:
            request: The row to serve.

        Returns:
            The pooled response token ids.

        Raises:
            SupplierRefused: :attr:`Refusal.NO_IDENTITY` for a row with no prompt region,
                :attr:`Refusal.NO_MATCH` when the pool does not cover this prompt -- it never
                substitutes some other prompt's row -- and :attr:`Refusal.NO_FIT` when the
                pooled response does not fit after the prompt.
        """
        key = request.identity()
        tokens = self._pool.get(key)
        if tokens is None:
            raise SupplierRefused(
                Refusal.NO_MATCH,
                self.name,
                f"the pool covers {len(self._pool)} prompts and not {key}; refusing to serve "
                "another prompt's row, which would splice an answer onto a question it does "
                "not answer",
            )
        n = int(tokens.numel())
        if n > request.capacity:
            raise SupplierRefused(
                Refusal.NO_FIT,
                self.name,
                f"prompt {request.prompt_len} + response {n} exceeds width {request.width}",
            )
        return SupplyOffer(tokens, self.name, f"corpus hit for prompt {key}")
