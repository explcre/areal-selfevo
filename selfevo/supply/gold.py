"""The dataset's own gold solution, as the first implementation of the supplier interface.

This is a REFACTOR and not a rewrite. Every line of arithmetic here was previously inline in
``selfevo/gold/substitute.py::substitute_gold_rows``: read the row's ``gold_mask`` to get the
gold's true length, refuse when it is zero, refuse when prompt + gold exceeds the row width,
otherwise hand back ``gold_ids[row, :n]``. It is moved rather than reimplemented so that gold
is the first supplier rather than a special case with three other suppliers bolted beside it,
and ``test_supply_sources.py`` pins the move by digesting the default path's output against a
constant measured before the refactor existed.

The two refusals map exactly onto the two counters ``GoldStats`` already carried:
:attr:`~selfevo.supply.base.Refusal.NO_GOLD` is ``groups_no_gold`` and
:attr:`~selfevo.supply.base.Refusal.NO_FIT` is ``groups_no_fit``. They stayed separate there
because a missing solution is a dataset problem and a gold that does not fit is a
sequence-length problem, and they stay separate here for the same reason.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from selfevo.gold.attach import GOLD_KEYS
from selfevo.supply.base import Refusal, SupplierRefused, SupplyOffer, SupplyRequest

__all__ = ["GoldSupplier"]


class GoldSupplier:
    """Serves the gold solution the dataset adapter tokenised and the workflow attached.

    Stateless: everything it needs is already in the batch, put there by
    ``areal/dataset/competition_math.py`` (behind ``keep_solution``) and
    ``selfevo.gold.attach``. That is why it is the cheapest supplier and why the shipped
    policy tries it first.
    """

    name = "gold"
    required_keys = GOLD_KEYS

    def has_supply(self, batch: Mapping[str, Any]) -> bool:
        """Whether any row of this batch carries gold text at all.

        Args:
            batch: The rollout batch.

        Returns:
            False when every ``gold_mask`` is empty, which is the state the reach guard turns
            into a loud refusal: a gold arm that trains on no gold must not look like a gold
            arm that ran.
        """
        mask = batch.get("gold_mask")
        return torch.is_tensor(mask) and int(mask.sum()) != 0

    def supply(self, request: SupplyRequest) -> SupplyOffer:
        """The row's own gold solution.

        Args:
            request: The row to serve.

        Returns:
            The gold token ids, exactly ``gold_mask.sum()`` of them.

        Raises:
            SupplierRefused: :attr:`Refusal.NO_GOLD` when this dataset row carried no usable
                gold text, :attr:`Refusal.NO_FIT` when prompt plus gold exceeds the row width.
                Never a truncation: a cut-off derivation is a wrong target that still looks
                like a target, and training on it is worse than not training on it.
        """
        mask = request.batch["gold_mask"]
        n_gold = int(mask[request.row].sum())
        if n_gold == 0:
            raise SupplierRefused(
                Refusal.NO_GOLD,
                self.name,
                f"row {request.row} carries an empty gold_mask, so its dataset row had no "
                "usable gold text",
            )
        if request.prompt_len + n_gold > request.width:
            raise SupplierRefused(
                Refusal.NO_FIT,
                self.name,
                f"prompt {request.prompt_len} + gold {n_gold} exceeds width {request.width}",
            )
        return SupplyOffer(request.batch["gold_ids"][request.row, :n_gold], self.name)

    def missing_key_hint(self, key: str) -> str:
        """The message the batch-level required-key refusal carries for gold.

        Preserved word for word from the inline version so that the refusal a run sees, and
        the test that pins it, are unchanged by the refactor.
        """
        return (
            f"the batch carries no {key}. The dataset adapter keeps the gold only when asked "
            "(keep_solution=True) and the workflow attaches it only when the dataset row "
            "carries it, so this means the gold arm is configured at one end and not the "
            "other."
        )

    def no_supply_message(self, rule_value: str) -> str:
        """The message the batch-level reach guard carries for gold, preserved verbatim."""
        return (
            f"rule={rule_value} but every gold_mask in this batch is empty, so there is no "
            "gold to substitute anywhere. A gold arm that trains on no gold is the silent "
            "no-op this guard exists to prevent."
        )
