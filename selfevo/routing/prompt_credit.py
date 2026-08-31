"""Per-prompt credit assignment across time.

Why this exists. `batch_outcomes` credits ONE scalar -- the change in mean reward between
consecutive batches -- to every decision in the batch. Measured on 2026-08-31 over 129 live
steps: the learned router received 128 clean updates and developed no preference, its mode
mix staying at uniform thirds and getting flatter. The cause is not the bandit. With
``b_m += r x`` and a shared ``r``, every arm converges to the same ``theta`` and the per-arm
counts cancel, so the signal carries nothing that separates the arms. Verified by measurement
in ``selfevo/tests/test_credit_assignment.py``: the same router separates arms and picks the
rewarded mode as soon as credit depends on the mode.

The fix is to hold the PROMPT fixed and vary the mode across time. A prompt routed to SFT in
epoch 3 and seen again in epoch 4 supplies a paired observation: same task, same difficulty,
different point in training. That is a far better-controlled signal than a batch mean over
different prompts.

Identity without plumbing. No prompt id reaches the trainer -- the batch at
``_compute_advantages`` is tensors only, and ``task_id`` elsewhere in AReaL belongs to the
performance tracer. Rather than thread an id through dataset, rollout and batching, this
module derives identity from the prompt TOKENS, which are the prompt. Measured on the live
config: 10 epochs over ~7.4k GSM8K prompts at 29 steps per epoch, so each prompt recurs about
every 29 steps and a paired observation is available roughly ten times per prompt.

What this does NOT fix. The delta for a prompt still spans many updates driven by other
prompts, so it is not a clean causal estimate of the mode's effect -- only a far tighter one
than a batch mean over different tasks. Report it as such.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field

__all__ = ["PromptCreditLedger", "PriorDecision", "prompt_key"]


def prompt_key(input_ids: list[int], loss_mask: list[float]) -> str:
    """Stable identity for a prompt, derived from its own tokens.

    The prompt is the region BEFORE the first response token, i.e. the leading run where
    ``loss_mask`` is zero. Two rollouts of the same prompt share those tokens exactly, and
    two different prompts share them only on a hash collision.

    Args:
        input_ids: Token ids for one sequence.
        loss_mask: Same length; 1 on response tokens, 0 on prompt tokens.

    Returns:
        A 16-hex-character digest of the prompt tokens.

    Raises:
        ValueError: If the lengths differ, or if the row has no prompt region. A row that is
            all response has no identity to key on, and silently returning a constant would
            merge every such row into one prompt -- crediting decisions to each other.
    """
    if len(input_ids) != len(loss_mask):
        raise ValueError(
            f"input_ids has {len(input_ids)} tokens and loss_mask has {len(loss_mask)}; "
            "they index the same positions and a mismatch means the row was mis-sliced"
        )
    n_prompt = 0
    for m in loss_mask:
        if m:
            break
        n_prompt += 1
    if n_prompt == 0:
        raise ValueError(
            "row has no prompt region (loss_mask starts at 1), so it carries no prompt "
            "identity; keying on it would merge unrelated rows into one prompt"
        )
    payload = ",".join(str(int(t)) for t in input_ids[:n_prompt]).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class PriorDecision:
    """A decision awaiting a later observation of the same prompt.

    Args:
        unit_id: The routing unit the decision belonged to, so the router credits the arm it
            actually chose.
        mode: The mode applied.
        value: The prompt's observed value (e.g. its solve rate) AT the time of the decision.
        step: Batch index the decision was made in, for reporting the gap.
    """

    unit_id: str
    mode: str
    value: float
    step: int


@dataclass
class PromptCreditLedger:
    """Pairs each prompt's decision with the next observation of that same prompt.

    Args:
        capacity: Maximum prompts retained. An epoch of the live config is ~7.4k prompts, so
            the default holds rather more than one epoch and a prompt survives to its next
            appearance. Oldest-first eviction.

    Attributes:
        evicted: Prompts dropped for capacity before they could be credited. Counted rather
            than silent: an eviction rate near the record rate means the ledger is too small
            and the router is being starved without anything in the log to say so.
        credited: Successful pairings.
    """

    capacity: int = 20000
    evicted: int = 0
    credited: int = 0
    _pending: OrderedDict[str, PriorDecision] = field(
        default_factory=OrderedDict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {self.capacity}")

    def observe_and_record(
        self, key: str, unit_id: str, mode: str, value: float, step: int
    ) -> tuple[PriorDecision, float] | None:
        """Credit any prior decision for this prompt, then record the current one.

        Both halves happen in one call because they must use the SAME observation: the value
        that closes the previous decision is the value the new decision starts from. Splitting
        them invites a caller to record first and then credit against its own value, which is
        a guaranteed zero delta.

        Args:
            key: From :func:`prompt_key`.
            unit_id: Current routing unit.
            mode: Mode applied now.
            value: The prompt's observed value now.
            step: Current batch index.

        Returns:
            ``(prior, delta)`` if this prompt had an uncredited decision, else ``None``.
            ``delta`` is ``value - prior.value``: positive means the prompt improved since
            the prior decision was applied.
        """
        out: tuple[PriorDecision, float] | None = None
        prior = self._pending.pop(key, None)
        if prior is not None:
            self.credited += 1
            out = (prior, value - prior.value)
        # The key was popped above whenever it was present, so this always inserts at the
        # end and no move_to_end is needed -- eviction order is therefore genuinely
        # oldest-first by last appearance.
        self._pending[key] = PriorDecision(
            unit_id=unit_id, mode=mode, value=value, step=step
        )
        while len(self._pending) > self.capacity:
            self._pending.popitem(last=False)
            self.evicted += 1
        return out

    def as_metrics(self) -> dict[str, float]:
        """Flat metrics, prefixed so they do not collide with the actor's own keys."""
        return {
            "prompt_credit/credited": float(self.credited),
            "prompt_credit/evicted": float(self.evicted),
            "prompt_credit/pending": float(len(self._pending)),
        }
