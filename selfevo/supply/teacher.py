"""A rollout from a stronger model, behind an interface and with NO model served.

WHAT IS BUILT HERE AND WHAT IS NOT. The interface is real: :class:`TeacherClient` is the
contract a served model would satisfy, and :class:`TeacherSupplier` is the supplier that turns
a teacher completion into a batch row through the same splice, the same counters and the same
off-policy treatment as every other source. The only implementation is
:class:`RecordedTeacherClient`, which replays completions from a file or a dict. Nothing in
this module opens a socket, loads weights or imports an engine, and that is deliberate rather
than incidental: this path is developed on a box whose GPUs are running a baseline arm and a
validation, and a supplier that quietly booked one would be undiscoverable until a run.

THE VERIFIER IS MANDATORY, AND THAT IS THE DESIGN DECISION IN THIS FILE. A teacher is stronger,
not correct. On the branch this whole axis exists to serve -- groups where every rollout is
wrong -- an unverified teacher completion is a row that is confidently substituted and may be
wrong, which is precisely what ``substitute_gold_rows`` refuses to do with a truncated gold: a
wrong target that still looks like a target is worse than no target. So :class:`TeacherSupplier`
requires a ``verify`` callable at construction and refuses with
:attr:`~selfevo.supply.base.Refusal.NOT_VERIFIED` when it says no. A run that genuinely wants
unverified teacher rows passes an explicitly-always-true verifier, which leaves a greppable
record in the launcher instead of a default nobody can find afterwards.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import torch

from selfevo.supply.base import (
    Refusal,
    SupplierRefused,
    SupplyConfigError,
    SupplyOffer,
    SupplyRequest,
    key_for_prompt,
)

__all__ = ["TeacherClient", "RecordedTeacherClient", "TeacherSupplier", "Verifier"]

# Given the request and the teacher's candidate response, is this a CORRECT answer to this
# prompt? Typed so the requirement is a signature rather than a convention.
Verifier = Callable[[SupplyRequest, "torch.Tensor"], bool]


@runtime_checkable
class TeacherClient(Protocol):
    """What a stronger model must offer to be usable as a supplier.

    One method, synchronous, token ids in and token ids out. Synchronous because the seam it
    is called from is a pure batch transform that runs before ``compute_logp``; anything
    asynchronous belongs in the rollout worker, on the other side of the batch.
    """

    def complete(self, prompt_ids: Sequence[int]) -> Sequence[int] | None:
        """The teacher's response to a prompt, or None when it has none."""


class RecordedTeacherClient:
    """Replays teacher completions recorded offline. The only implementation in this repo.

    Args:
        responses: ``{prompt digest: response token ids}``, keyed by
            :func:`selfevo.supply.base.key_for_prompt`.
        dtype: Dtype used when a value has to be converted.
    """

    def __init__(
        self,
        responses: Mapping[str, Sequence[int] | torch.Tensor],
        *,
        dtype: torch.dtype = torch.long,
    ) -> None:
        self._responses = dict(responses)
        self._dtype = dtype

    def __len__(self) -> int:
        """Number of prompts this recording covers."""
        return len(self._responses)

    @classmethod
    def from_jsonl(
        cls, path: str | Path, *, dtype: torch.dtype = torch.long
    ) -> "RecordedTeacherClient":
        """Load recorded completions from a JSON Lines file.

        Args:
            path: Each line an object with ``prompt_ids`` and ``response_ids``.
            dtype: Dtype for the response tensors.

        Returns:
            A client over those recordings.

        Raises:
            SupplyConfigError: On a line missing either id list or carrying an empty one.
        """
        out: dict[str, torch.Tensor] = {}
        for lineno, line in enumerate(Path(path).read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt, resp = row.get("prompt_ids"), row.get("response_ids")
            if not prompt or not resp:
                raise SupplyConfigError(
                    f"{path}:{lineno} has prompt_ids={prompt!r} response_ids={resp!r}; both "
                    "must be non-empty"
                )
            out[key_for_prompt(prompt)] = torch.tensor(
                [int(t) for t in resp], dtype=dtype
            )
        return cls(out, dtype=dtype)

    def complete(self, prompt_ids: Sequence[int]) -> torch.Tensor | None:
        """The recorded response for this prompt, or None.

        Args:
            prompt_ids: The prompt's token ids.

        Returns:
            A 1-D tensor, or None when the recording does not cover this prompt.
        """
        return self._responses.get(key_for_prompt(prompt_ids))


class TeacherSupplier:
    """Serves a stronger model's VERIFIED response for this prompt.

    Args:
        client: Anything satisfying :class:`TeacherClient`, or None for an arm configured
            without a teacher -- which refuses every request with
            :attr:`~selfevo.supply.base.Refusal.UNAVAILABLE` rather than pretending.
        verify: Decides whether a candidate is a correct answer to this prompt. REQUIRED; see
            the module docstring.

    Raises:
        SupplyConfigError: If ``verify`` is None. A teacher is stronger, not correct, and an
            unverified completion substituted into an all-wrong group is a wrong target that
            still looks like a target.
    """

    name = "teacher"
    required_keys: tuple[str, ...] = ()

    def __init__(self, client: TeacherClient | None, *, verify: Verifier | None) -> None:
        if verify is None:
            raise SupplyConfigError(
                "TeacherSupplier needs verify=<callable>. A teacher completion is not known "
                "to be correct, and on the unsolved branch a wrong row is substituted with "
                "full confidence and is indistinguishable downstream from a right one. Pass "
                "an explicitly-always-true verifier if that is genuinely intended, so the "
                "choice is greppable in the launcher."
            )
        self.client = client
        self.verify = verify

    def has_supply(self, batch: Mapping[str, Any]) -> bool:
        """Whether a teacher is wired at all.

        Args:
            batch: Unread; a teacher's coverage is a property of the client.

        Returns:
            True when a client exists. ``actor.py``'s literal ``has_teacher=False`` is the
            standing reason no teacher arm has ever been reachable; this is the honest form of
            that question, answered by the object rather than by a constant.
        """
        return self.client is not None

    def supply(self, request: SupplyRequest) -> SupplyOffer:
        """The teacher's verified response for this prompt.

        Args:
            request: The row to serve.

        Returns:
            The teacher's response token ids.

        Raises:
            SupplierRefused: :attr:`Refusal.UNAVAILABLE` with no client,
                :attr:`Refusal.NO_IDENTITY` for a row with no prompt region,
                :attr:`Refusal.NO_MATCH` when the teacher has nothing for this prompt,
                :attr:`Refusal.EMPTY` for a zero-token completion,
                :attr:`Refusal.NOT_VERIFIED` when the verifier rejects it, and
                :attr:`Refusal.NO_FIT` when it does not fit after the prompt.
        """
        if self.client is None:
            raise SupplierRefused(
                Refusal.UNAVAILABLE, self.name, "no teacher client is wired for this run"
            )
        key = request.identity()
        candidate = self.client.complete(request.prompt_ids())
        if candidate is None:
            raise SupplierRefused(
                Refusal.NO_MATCH, self.name, f"the teacher has no response for prompt {key}"
            )
        tokens = (
            candidate
            if torch.is_tensor(candidate)
            else torch.tensor([int(t) for t in candidate], dtype=torch.long)
        )
        if tokens.ndim != 1:
            raise SupplierRefused(
                Refusal.NO_MATCH,
                self.name,
                f"the teacher returned shape {tuple(tokens.shape)}, not a 1-D token sequence",
            )
        if tokens.numel() == 0:
            raise SupplierRefused(
                Refusal.EMPTY, self.name, f"the teacher returned nothing for prompt {key}"
            )
        if not self.verify(request, tokens):
            raise SupplierRefused(
                Refusal.NOT_VERIFIED,
                self.name,
                f"the teacher's response to prompt {key} did not verify as correct",
            )
        if int(tokens.numel()) > request.capacity:
            raise SupplierRefused(
                Refusal.NO_FIT,
                self.name,
                f"prompt {request.prompt_len} + response {int(tokens.numel())} exceeds width "
                f"{request.width}",
            )
        return SupplyOffer(tokens, self.name, f"teacher hit for prompt {key}")
