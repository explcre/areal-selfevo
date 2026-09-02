"""A bounded store of the model's OWN correct rollouts, keyed by prompt identity.

WHY A STORE IS NEEDED AT ALL. The self-generated supplier's premise is that a group in which
every rollout is wrong THIS step may have been solved by the same policy at an earlier step,
or by a resample outside this group. Either way the correct row is not in the batch in front
of the seam, so something has to have kept it.

WHY IT IS KEYED ON THE PROMPT AND NOT ON ``unit_id``. ``unit_id`` is batch-local by
construction -- it is built from the row's position -- so a store keyed on it would never find
a prompt again on a later step, which is the only thing this supplier is for. The prompt-credit
ledger hit exactly this and solved it by hashing the tokens before the first response token
(``selfevo/routing/prompt_credit.py``, "Identity without plumbing"), measuring on the live
config that a prompt recurs about every 29 steps. This store calls that same function through
:meth:`selfevo.supply.base.SupplyRequest.identity` rather than growing a second scheme.

WHAT IT REFUSES TO RECORD, and both of these are the difference between a self-generated
supplier and a laundering one:

* A row whose reward is not CORRECT. The store exists to hold correct targets; an incorrect
  rollout put in it becomes a wrong row substituted with full confidence later.
* A row that was itself SUBSTITUTED, i.e. one whose ``is_gold`` is set. Such a row's tokens
  are the dataset's gold, or a corpus row, or a teacher's answer -- not the model's own
  output. Recording it would report a self-generated arm that is actually replaying its own
  gold arm, and no counter downstream could tell the difference.

BOUNDED, for the reason ``PriorDecision`` gives for keeping two floats instead of a list: over
10 epochs of a ~7.4k-prompt dataset an unbounded per-prompt memory is the one thing a
long-running trainer must not carry. Eviction is least-recently-used and counted.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Sequence

import torch

from selfevo.routing.prompt_credit import prompt_key
from selfevo.supply.base import SupplyConfigError

__all__ = ["SolvedRolloutStore"]

# Matches DyME's ``acc_rewards > 0.5`` and the actor's own solved/unsolved split, which
# thresholds the RAW reward at the same 0.5. Named once rather than repeated so this store and
# the rule that selects groups cannot drift apart on what "correct" means.
_CORRECT = 0.5


class SolvedRolloutStore:
    """Correct responses the policy itself produced, kept across steps and keyed by prompt.

    Args:
        capacity: Maximum number of distinct prompts held. Least-recently-used prompts are
            evicted past it, and the evictions are counted rather than silent.
        keep_per_prompt: How many correct responses to keep for one prompt. 1 by default: the
            supplier serves one row per group, and holding more multiplies the memory by the
            solve rate for no reach.

    Raises:
        SupplyConfigError: On a non-positive capacity or ``keep_per_prompt``, which would make
            every write a no-op and the arm a silent off arm.
    """

    def __init__(self, capacity: int = 4096, keep_per_prompt: int = 1) -> None:
        if capacity <= 0 or keep_per_prompt <= 0:
            raise SupplyConfigError(
                f"capacity={capacity} keep_per_prompt={keep_per_prompt}: both must be "
                "positive, or every write is a no-op and the arm is silently off"
            )
        self.capacity = int(capacity)
        self.keep_per_prompt = int(keep_per_prompt)
        self._rows: OrderedDict[str, list[torch.Tensor]] = OrderedDict()
        self.recorded = 0
        self.evicted = 0
        self.skipped_incorrect = 0
        self.skipped_substituted = 0
        self.skipped_no_response = 0
        self.skipped_no_identity = 0

    def __len__(self) -> int:
        """Number of distinct prompts held."""
        return len(self._rows)

    def record(self, key: str, tokens: torch.Tensor) -> None:
        """Keep one correct response under a prompt identity.

        Args:
            key: A prompt digest from :func:`selfevo.supply.base.key_for_prompt` or
                :meth:`SupplyRequest.identity`.
            tokens: 1-D integer tensor of response token ids.

        Raises:
            SupplyConfigError: If ``tokens`` is not a 1-D non-empty integer tensor. A
                zero-token "correct response" would be written into a batch as a row with no
                loss mass at all, which reports as a served group and trains on nothing.
        """
        if not torch.is_tensor(tokens) or tokens.ndim != 1 or tokens.numel() == 0:
            raise SupplyConfigError(
                f"a stored response must be a 1-D non-empty tensor; got "
                f"{getattr(tokens, 'shape', type(tokens).__name__)}"
            )
        held = self._rows.pop(key, [])
        held.append(tokens.detach().clone())
        self._rows[key] = held[-self.keep_per_prompt :]
        self.recorded += 1
        while len(self._rows) > self.capacity:
            self._rows.popitem(last=False)
            self.evicted += 1

    def get(self, key: str) -> torch.Tensor | None:
        """The most recent correct response for a prompt, or None.

        Args:
            key: The prompt digest.

        Returns:
            A 1-D tensor, or None when this prompt has never been solved. Looking a prompt up
            marks it recently used, so the prompts a run keeps asking about are the ones the
            bound keeps.
        """
        held = self._rows.get(key)
        if not held:
            return None
        self._rows.move_to_end(key)
        return held[-1]

    def record_batch(
        self,
        batch: Mapping[str, Any],
        *,
        correct_above: float = _CORRECT,
    ) -> int:
        """Harvest every correct, self-generated row of a rollout batch.

        Args:
            batch: A rollout batch, in the TOKEN coordinates a workflow emits. Read only.
            correct_above: Raw-reward threshold for "correct".

        Returns:
            How many rows were recorded.

        The four skip counters are the artifact behind this function's zero. A run whose
        ``recorded`` stays at 0 while ``skipped_substituted`` climbs is replaying its own gold
        arm into the self store, and that is visible in one line instead of being
        reconstructed from a solve rate.
        """
        ids = batch.get("input_ids")
        loss_mask = batch.get("loss_mask")
        rewards = batch.get("rewards")
        if not (
            torch.is_tensor(ids) and torch.is_tensor(loss_mask) and torch.is_tensor(rewards)
        ):
            return 0
        raw = batch.get("original_rewards")
        raw = rewards if not torch.is_tensor(raw) else raw
        is_gold = batch.get("is_gold")
        n_rows = int(ids.shape[0])

        recorded = 0
        for row in range(n_rows):
            if float(raw[row]) <= correct_above:
                self.skipped_incorrect += 1
                continue
            if torch.is_tensor(is_gold) and int(is_gold[row].sum()) != 0:
                # A substituted row's tokens came from a supplier, not from the model.
                self.skipped_substituted += 1
                continue
            resp = ids[row][loss_mask[row].bool()]
            if resp.numel() == 0:
                self.skipped_no_response += 1
                continue
            try:
                key = prompt_key(
                    [int(v) for v in ids[row].tolist()],
                    [float(v) for v in loss_mask[row].tolist()],
                )
            except ValueError:
                self.skipped_no_identity += 1
                continue
            self.record(key, resp)
            recorded += 1
        return recorded

    def as_metrics(self) -> dict[str, float]:
        """Flat metrics under a ``supply/store/`` prefix.

        Emitted even when zero, so a store-backed arm and an arm without one produce the same
        key set and stay readable on one panel.
        """
        return {
            "supply/store/prompts": float(len(self._rows)),
            "supply/store/recorded": float(self.recorded),
            "supply/store/evicted": float(self.evicted),
            "supply/store/skipped_incorrect": float(self.skipped_incorrect),
            "supply/store/skipped_substituted": float(self.skipped_substituted),
            "supply/store/skipped_no_response": float(self.skipped_no_response),
            "supply/store/skipped_no_identity": float(self.skipped_no_identity),
        }

    def keys(self) -> Sequence[str]:
        """The prompt digests currently held, oldest first."""
        return tuple(self._rows)
