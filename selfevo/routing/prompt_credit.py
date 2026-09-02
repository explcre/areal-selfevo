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

The baseline the delta is measured against. A raw delta ``v_k - v_{k-1}`` still contains a
component common to every prompt -- the policy improves over the ~29 steps between a prompt's
appearances whatever mode was applied -- so every arm is credited positively and the arm used
most during an improving window accumulates the most evidence. The wiring's first answer was
to subtract the BATCH's mean delta, which is a batch-level aggregate shared by every arm and
therefore the same class of quantity the whole module exists to get away from. Measured over
20 paired seeds on :mod:`selfevo.routing.credit_sim` it is never better than no centring at
all (-0.064 +- 0.016 on subset targeting in the gain-dominated regime, -0.007 +- 0.017 in the
trend-dominated one). ``baseline="self_mean"`` instead subtracts the mean of THAT prompt's own
earlier deltas, which is a per-prompt quantity and cancels the part of the trend that prompt
actually experienced; it beats the raw delta in both regimes (+0.028 +- 0.012 and
+0.052 +- 0.018) and beats batch centring by 4.7 and 3.8 sigma. ``"last"`` stays the default so
every arm run before 2026-09-01 is reproduced bit-for-bit.

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
        n_deltas: How many deltas this prompt has already contributed, i.e. how many times it
            has been seen after its first appearance. Carried on the record rather than in a
            second table so a prompt's whole history is one entry that one eviction removes;
            a split table would let the value survive its own decision and credit a later
            decision against a baseline whose deltas had been evicted.
        mean_delta: Running mean of those deltas. Two floats rather than a list of them: the
            baseline is only ever used as a mean, and a list would make a prompt's memory grow
            with the length of the run, which over 10 epochs x 7.4k prompts is the one thing
            this ledger must not do.
    """

    unit_id: str
    mode: str
    value: float
    step: int
    n_deltas: int = 0
    mean_delta: float = 0.0


@dataclass
class PromptCreditLedger:
    """Pairs each prompt's decision with the next observation of that same prompt.

    Args:
        capacity: Maximum prompts retained. An epoch of the live config is ~7.4k prompts, so
            the default holds rather more than one epoch and a prompt survives to its next
            appearance. Oldest-first eviction.
        baseline: What the prompt's change is measured against.

            ``"last"`` (default, and what every arm before 2026-09-01 ran) credits the raw
            delta ``value - prior.value``. The prompt is its own control for difficulty, which
            is most of the confound, but not for TIME: the policy improves between the two
            appearances whatever was applied, so every arm is credited positively.

            ``"self_mean"`` subtracts the mean of that prompt's own EARLIER deltas, so what
            reaches the router is how this decision fared against what this prompt usually
            does -- the same within-group centring GRPO applies across a rollout group, here
            applied across a prompt's appearances in time. It is deliberately not the batch's
            mean delta: that is one number shared by every arm, which is the quantity this
            module exists to get away from, and it measures worse (see the module docstring).

            The cost is one pairing per prompt. A prompt's FIRST delta has no earlier delta to
            centre against, and the two ways to fill that gap are both worse than withholding:
            a zero baseline hands the first-credited mode the whole common trend, which is the
            bias that made the live arm abandon RL at the exact step credit began flowing, and
            seeding from the batch reintroduces the aggregate. Withheld pairings are counted
            in ``cold_baseline_skips``, not silent.

    Attributes:
        evicted: Prompts dropped for capacity before they could be credited. Counted rather
            than silent: an eviction rate near the record rate means the ledger is too small
            and the router is being starved without anything in the log to say so.
        credited: Successful pairings.
        same_batch_skips: Sightings refused because the prior decision was made in the SAME
            batch. Counted rather than silent: a large value means the batch repeatedly
            contains duplicate prompts, which halves the pairing rate.
        cold_baseline_skips: Pairings withheld under ``baseline="self_mean"`` because the
            prompt had no earlier delta to centre against. Always 0 under ``"last"``. Counted
            because it is the price of that baseline, and a run where it stays close to the
            pairing rate is one where prompts are not recurring often enough for the baseline
            to be worth its cost.
    """

    capacity: int = 20000
    baseline: str = "last"
    evicted: int = 0
    credited: int = 0
    same_batch_skips: int = 0
    cold_baseline_skips: int = 0
    _pending: OrderedDict[str, PriorDecision] = field(
        default_factory=OrderedDict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {self.capacity}")
        if self.baseline not in ("last", "self_mean"):
            raise ValueError(
                f"baseline must be 'last' or 'self_mean', got {self.baseline!r}. An unknown "
                "value falling back to 'last' would report a self-baselined arm that never ran."
            )

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
            ``(prior, credit)`` if this prompt had an uncredited decision that could be
            scored, else ``None``. Under ``baseline="last"`` the credit is
            ``value - prior.value``: positive means the prompt improved since the prior
            decision was applied. Under ``"self_mean"`` it is that delta minus the mean of the
            prompt's earlier deltas, so positive means the prompt improved MORE than it
            usually does -- and ``None`` is also returned for a prompt whose first delta this
            is, since there is nothing yet to centre against.
        """
        out: tuple[PriorDecision, float] | None = None
        prior = self._pending.get(key)
        if prior is not None and prior.step == step:
            # SAME BATCH. Two groups can carry the same prompt within one batch, and pairing
            # them would credit one group's decision with another group's solve rate at the
            # IDENTICAL policy -- a delta that measures sampling noise between two rollout
            # groups, not the effect of a mode. The whole point of this ledger is that the
            # two observations are separated in training time.
            #
            # The earlier record is kept rather than overwritten, so the decision that pairs
            # with the prompt's NEXT appearance is the one applied first in this batch.
            self.same_batch_skips += 1
            return None
        n_deltas, mean_delta = 0, 0.0
        if prior is not None:
            self._pending.pop(key, None)
            delta = value - prior.value
            n_deltas, mean_delta = prior.n_deltas, prior.mean_delta
            if self.baseline == "last":
                self.credited += 1
                out = (prior, delta)
            elif n_deltas > 0:
                self.credited += 1
                out = (prior, delta - mean_delta)
            else:
                self.cold_baseline_skips += 1
            # Leave-current-out, and it is not a detail. The baseline scored above is the mean
            # of STRICTLY EARLIER deltas; this delta joins the history only afterwards.
            # Folding it in first would make every credit its own control, shrinking each one
            # by 1/n towards zero and shrinking the largest -- the most informative --
            # observations the most.
            n_deltas += 1
            mean_delta += (delta - mean_delta) / n_deltas
        # The key was popped above whenever it was present, so this always inserts at the
        # end and no move_to_end is needed -- eviction order is therefore genuinely
        # oldest-first by last appearance.
        self._pending[key] = PriorDecision(
            unit_id=unit_id, mode=mode, value=value, step=step,
            n_deltas=n_deltas, mean_delta=mean_delta,
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
            "prompt_credit/same_batch_skips": float(self.same_batch_skips),
            "prompt_credit/cold_baseline_skips": float(self.cold_baseline_skips),
        }
