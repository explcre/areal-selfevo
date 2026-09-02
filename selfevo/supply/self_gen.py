"""A correct rollout of THIS prompt, obtained from somewhere other than this group.

The premise, and the only reason this supplier can exist at all: a group in which every
rollout is wrong this step is not a prompt the policy has never solved. On the live
configuration a prompt recurs roughly every 29 steps, so a prompt that is unsolved now may
have been solved at step k-29, and that earlier rollout is a correct target the model itself
produced. Two sources of such a row, one seam:

* AN EARLIER STEP. :class:`selfevo.supply.store.SolvedRolloutStore`, filled by the trainer
  calling ``record_batch`` on each rollout batch before substitution. Pure bookkeeping, runs
  on CPU, and is what this file is mostly about.
* A HIGHER-TEMPERATURE RESAMPLE. Drawing more samples at a higher temperature turns some
  all-wrong groups into groups with a correct member. That needs an inference engine, so it is
  a SEAM here (:attr:`SelfGeneratedSupplier.resampler`) and not an implementation: this whole
  path is developed and tested on CPU, and a supplier that quietly booked a GPU would be
  undiscoverable until a run.

WHAT MAKES THIS OFF-POLICY AND WHY THAT IS ALREADY HANDLED. A rollout from step k-29 was
emitted by a different policy version than the one being updated, so it has no valid
importance ratio under the current policy -- exactly the condition the gold path already
treats, with a finite sentinel written at substitution and ``reconcile_gold_logprobs``
replacing it with the trainer's own recomputed ``prox_logp`` after the forward pass. There is
no second treatment here, deliberately: two arms that differ in how they weight an off-policy
row, with neither reporting it, is the failure ``FINDINGS_loss_weighting.md`` was written
about.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch

from selfevo.supply.base import (
    Refusal,
    SupplierRefused,
    SupplyOffer,
    SupplyRequest,
)
from selfevo.supply.store import SolvedRolloutStore

__all__ = ["SelfGeneratedSupplier"]

# A resampler is given the request and returns response token ids, or None when the resample
# produced nothing correct. Typed here so the seam is a declared contract rather than a
# convention, and so a fake one in a test has the same signature as a real one.
Resampler = Callable[[SupplyRequest], "torch.Tensor | None"]


class SelfGeneratedSupplier:
    """Serves the model's own earlier correct rollout for this prompt.

    Args:
        store: Where earlier correct rollouts were kept. Keyed by prompt identity, never by
            ``unit_id``, which is batch-local.
        resampler: Optional callable for a higher-temperature resample. NOT implemented in
            this repo and never called by any test that touches a GPU; it exists so the second
            source of a self-generated row has a declared place to land.
    """

    name = "self"
    required_keys: tuple[str, ...] = ()

    def __init__(
        self, store: SolvedRolloutStore, *, resampler: Resampler | None = None
    ) -> None:
        self.store = store
        self.resampler = resampler

    def has_supply(self, batch: Mapping[str, Any]) -> bool:
        """Whether this supplier could serve anything in this batch.

        Args:
            batch: The rollout batch. Unread: what this supplier holds is a property of the
                store and of the resampler, not of the batch in front of it.

        Returns:
            True when the store holds at least one prompt, or a resampler is wired. An empty
            store with no resampler is the reach-guard case: an arm configured with a
            self-generated source and nothing to serve from must refuse loudly on the first
            batch rather than decline every group in silence for 900 steps.
        """
        return len(self.store) > 0 or self.resampler is not None

    def supply(self, request: SupplyRequest) -> SupplyOffer:
        """This prompt's own earlier correct response.

        Args:
            request: The row to serve.

        Returns:
            The stored response token ids.

        Raises:
            SupplierRefused: :attr:`Refusal.NO_IDENTITY` when the row has no prompt region to
                key on, :attr:`Refusal.NO_MATCH` when this prompt has never been solved,
                :attr:`Refusal.EMPTY` for a zero-length stored response, and
                :attr:`Refusal.NO_FIT` when the response does not fit after the prompt.
                Never a truncation and never another prompt's row: this supplier's whole claim
                is that the target is a correct answer TO THIS QUESTION.
        """
        key = request.identity()
        tokens = self.store.get(key)
        source_detail = f"store hit for prompt {key}"
        if tokens is None and self.resampler is not None:
            tokens = self.resampler(request)
            source_detail = f"resample for prompt {key}"
        if tokens is None:
            raise SupplierRefused(
                Refusal.NO_MATCH,
                self.name,
                f"prompt {key} has no recorded correct rollout",
            )
        if not torch.is_tensor(tokens) or tokens.ndim != 1:
            raise SupplierRefused(
                Refusal.NO_MATCH,
                self.name,
                f"prompt {key} yielded {type(tokens).__name__}, not a 1-D tensor",
            )
        n = int(tokens.numel())
        if n == 0:
            raise SupplierRefused(
                Refusal.EMPTY, self.name, f"prompt {key} yielded a zero-token response"
            )
        if n > request.capacity:
            raise SupplierRefused(
                Refusal.NO_FIT,
                self.name,
                f"prompt {request.prompt_len} + response {n} exceeds width {request.width}",
            )
        return SupplyOffer(tokens, self.name, source_detail)
