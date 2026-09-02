"""Substitute a gold row into the groups that carry no learnable signal.

WHY THIS IS A BATCH OPERATION AND NOT AN ADVANTAGE ONE. An advantage is a per-token
coefficient on the log-probability of a token the model ACTUALLY EMITTED. In an all-wrong
group every emitted token belongs to a wrong derivation, so no value written into the
advantage tensor is a step toward the gold -- a positive constant there is a step toward the
WRONG answer, which is why ``GroupRoutingConfig.unsolved_advantage`` is required to be <= 0
and why ``test_gold_target_reachability.py`` can show that no router in the registry reaches
these groups. The only altitude at which a gold can receive gradient is the batch: put the
gold tokens in as a ROW, and the estimator that already runs on every other row runs on it
too. That is LUFFY's construction (arXiv 2504.14945) and DyME's (ICLR 2026), and it is the
one thing both mandatory baselines need from this repo.

WHAT EACH BASELINE'S RULE BECOMES HERE.

``dyme``
    ``DyMETrainer.py:655-698``: ``has_correct = (acc_rewards > 0.5).sum(1)``; a group with
    ``has_correct == 0`` has its FIRST rollout (``i % num_generations == 0``) replaced by the
    gold and pinned to advantage 1, an all-correct group is zeroed, everything else is
    ordinary GRPO. :attr:`GoldRule.DYME` is the first of those three -- the batch half of the
    rule -- and the other two are advantage-level operations that belong at the actor seam.
    They are NOT silently included here, and :attr:`GoldStats.qualifying_groups` carries what
    that seam needs to apply them. ``FINDINGS_gold_path.md`` records what the estimator
    actually gives the sibling rows once a gold has been substituted, which is NOT what DyME
    gives them.

``lspo_cliff``
    arXiv 2607.27787 defines the cliff set ``C = {x in B : sum_k R(x, y^(k)) = 0}``, and its
    SFT step updates only a LoRA adapter while the RL step updates only the base. The adapter
    routing is a separate axis owned elsewhere; what LSPO needs from here is the gold row plus
    a per-row flag saying which rows are gold, which is ``is_gold``. The predicate is written
    as the paper writes it -- the group's rewards SUM to zero -- and not aliased to ``dyme``,
    because the two coincide only for rewards in {0, 1}: a group scoring ``[-1, +1]`` sums to
    zero and is a cliff by LSPO's definition while DyME sees a correct sample and declines.
    That divergence is tested rather than assumed.

THE OFF-POLICY PROBLEM, AND WHAT MEASUREMENT DECIDED IT. Gold tokens were never sampled, so
there is no behaviour policy for them and ``logprobs`` -- which the actor reads as pi_behave
-- has no honest value. ``selfevo/FINDINGS_loss_weighting.md`` (2026-09-01) settles what may
be written there, and it rules out both of the obvious answers:

* NaN is FATAL, and it is not even a tripwire. ``_compute_advantages`` reads
  ``data["logprobs"]`` for the KL reward at ``actor.py:741``, and ``kl_ctl = 0.0`` does not
  protect it because ``-0.0 * NaN`` is NaN. Under the live ``adv_norm: mean_level=batch``
  (``gsm8k_grpo_lora.yaml:85-87``) the audit measured all 8 of 8 rows coming out NaN from one
  poisoned row. In the loss itself NaN is silent rather than loud: ``functional.py:233``
  rewrites a non-finite log-ratio to 0.0, so the row is scored as perfectly on-policy with
  ``filtered_fraction = 0.0``.
* 0.0 is a silent shrink. With the live rejection sampling,
  ``behave_imp_weight = exp(prox_logp) < 1`` multiplies the surrogate at ``functional.py:568``
  -- 0.368 of the row's weight at ``prox_logp = -1`` -- and no metric reports it.

The rule that audit states is: write a FINITE ``logprobs`` for every gold row, equal to the
trainer's own recomputed ``prox_logp`` for those tokens, which is the only value giving
``behave_imp_weight = exp(0) = 1``. That value does not exist until a forward pass has run,
and this function is pure and runs before it. So the default policy writes a finite SENTINEL
which the caller is required to fill by calling :func:`reconcile_gold_logprobs` after
``compute_logp``, and :func:`assert_gold_logprobs_filled` refuses to let an unfilled gold row
reach the loss. The sentinel is ``+1.0``: finite, so it cannot turn a batch's advantages NaN
the way the audit measured even if every guard is bypassed, and strictly positive, so it is
not a possible value of a log-probability and cannot be mistaken for a filled one.

TOKEN MASS, NOT ROW COUNT, IS THE SIZE OF THIS INTERVENTION. The same audit measures the
objective as a single per-token mean over the global batch (``functional.py:506,571``, whose
per-microbatch division cancels the FSDP rescale at ``fsdp_engine.py:2216``), so a row's
share of the update is proportional to its TOKEN count: an SFT row of 4, 8 and 16 tokens
against 4-token RL rows measured 0.5, 1.0 and 2.0 times their gradient magnitude. Two arms
matched on gold-ROW count can therefore differ several-fold in the quantity the loss actually
reads, and no existing ``route/*`` key reports it. :meth:`GoldStats.as_metrics` emits
``gold/token_mass``.

WHERE THE ROW COMES FROM IS A SEPARATE QUESTION FROM WHERE IT GOES, and only the second one
lives here. The splice, the masks, the off-policy value, the counters and the two reach guards
are properties of putting a correct row into a batch, and they do not depend on who supplied
it. So the gold-reading arithmetic that used to sit inline in the per-group loop has moved to
:class:`selfevo.supply.gold.GoldSupplier` -- the FIRST implementation of
:mod:`selfevo.supply.base`'s interface rather than a special case -- and a self-generated
rollout, an offline corpus row or a verified teacher completion enter through exactly the same
lines. ``selfevo/tests/test_supply_sources.py`` pins the refactor against a digest of this
module's output measured before it happened.

The generalisation is gated by argument and not by a config default: ``suppliers=None`` and
``source_policy=None``, the defaults, mean the gold-only path, whose output -- including its
KEY SET -- is byte-identical to the pre-supplier version, and ``GoldRule.NONE`` above that is
still a true no-op.

Everything in this module is a pure tensor function with no actor dependency: it takes a
batch dict, returns a new batch dict, and never imports the trainer or an engine.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

# GOLD_KEYS is no longer read here -- which keys a supplier needs is now the supplier's own
# declaration -- but it is re-exported from this module's namespace, so it stays imported.
from selfevo.gold.attach import GOLD_KEYS, GoldError, prompt_lengths  # noqa: F401

if TYPE_CHECKING:  # pragma: no cover - typing only
    from selfevo.supply.base import Supplier
    from selfevo.supply.policy import SourcePolicy

__all__ = [
    "GOLD_LOGP_SENTINEL",
    "GoldRule",
    "GoldLogprobPolicy",
    "GoldStats",
    "GoldMissingError",
    "GoldShapeError",
    "GoldPolicyError",
    "GoldOrderingError",
    "substitute_gold_rows",
    "substitute_in_place",
    "reconcile_gold_logprobs",
    "assert_gold_logprobs_filled",
]

# What a gold token's behaviour log-probability holds until the caller fills it.
#
# +1.0, and not NaN, 0.0 or -inf; every part of that is forced by
# selfevo/FINDINGS_loss_weighting.md. It must be FINITE, because a non-finite value in
# data["logprobs"] turns every advantage in a batch-normalised batch to NaN one stage before
# the loss. And it must be IMPOSSIBLE as a log-probability, because a plausible value cannot
# be told apart from a filled one, and an unfilled gold row that reaches the loss is silently
# reweighted by exp(prox_logp - 1) with nothing reporting it. A log-probability is <= 0, so
# any strictly positive number is unmistakable.
GOLD_LOGP_SENTINEL = 1.0

# Tensors that describe the rows as they were BEFORE substitution and cannot survive it.
# Their presence means the batch has already been through compute_logp, i.e. substitution was
# attempted one stage too late; see GoldOrderingError.
_STALE_AFTER_SUBSTITUTION = ("prox_logp", "ref_logp", "teacher_logp")

# Keys a rollout batch must carry for a row to be rewritable at all.
_REQUIRED = ("input_ids", "loss_mask", "attention_mask", "rewards", "logprobs")

# Rewards are graded correct above this, matching DyME's ``acc_rewards > 0.5`` and the
# actor's own solved/unsolved split, which thresholds the RAW reward at the same 0.5.
_CORRECT = 0.5


class GoldMissingError(GoldError):
    """Gold was asked for and none of it reached the update.

    The silent no-op is the failure this repo distrusts most: an arm that is configured,
    reports itself as a gold arm and applies nothing. Raised when the batch carries no gold at
    all, and when groups qualified but not one of them could be given a gold.

    Args:
        message: What went wrong.
        stats: The counts as they stood when the refusal was raised, or None when the batch
            was too malformed to count. Carried on the exception rather than discarded
            because :func:`substitute_in_place` catches this per group and must still be able
            to report that group's loss of reach: without it, a prompt whose gold was missing
            vanished from ``groups_qualifying`` and ``groups_no_gold`` entirely, and the batch
            report understated exactly the quantity these counters exist to expose.
    """

    def __init__(self, message: str, stats: "GoldStats | None" = None):
        super().__init__(message)
        self.stats = stats


class GoldShapeError(GoldError):
    """The batch is not shaped the way a rollout batch is."""


class GoldPolicyError(GoldError):
    """An unknown rule or log-probability policy, or one used without its other half."""


class GoldOrderingError(GoldError):
    """The gold path's two halves were run in the wrong order.

    Substituting after ``compute_logp`` leaves ``prox_logp``/``ref_logp``/``teacher_logp``
    describing tokens that are no longer in the batch -- same shape, wrong content, nothing
    downstream to notice. Reaching the loss without reconciling leaves a gold row carrying
    :data:`GOLD_LOGP_SENTINEL`, which is silently laundered into a fractional weight on the
    one row the arm exists to train. Both are refused.
    """


class GoldRule(Enum):
    """Which groups get a gold row.

    ``NONE`` is the default everywhere and is a true no-op: the batch is returned unchanged
    and no guard fires, so a run with the gold path compiled in but switched off is
    bit-identical to one built before it existed.
    """

    NONE = "none"
    DYME = "dyme"
    LSPO_CLIFF = "lspo_cliff"


class GoldLogprobPolicy(Enum):
    """What is written into ``logprobs`` for tokens that were never sampled.

    ``PROX_RECOMPUTE`` is the default and is the option
    ``selfevo/FINDINGS_loss_weighting.md`` section 3 requires: a finite value equal to the
    trainer's recomputed ``prox_logp``, supplied in two phases because that value does not
    exist before the forward pass.

    ``RATIO_ONE`` writes a plain 0.0 and marks the row, leaving a consumer to force the ratio
    to 1 inside the loss. It is kept so the axis stays swappable if the loss ever grows an
    ``is_gold`` branch, and it is NOT the default because as shipped it is incomplete: nothing
    in ``grpo_loss_fn`` reads ``is_gold``, so 0.0 is taken at face value and the audit
    measured such a row keeping 0.368 of its weight at ``prox_logp = -1``, unreported.
    :func:`reconcile_gold_logprobs` refuses it rather than pretending to fix it.
    """

    PROX_RECOMPUTE = "prox_recompute"
    RATIO_ONE = "ratio_one"


@dataclass(frozen=True)
class GoldStats:
    """What the substitution reached, for the run's panel and for the reach guard.

    Every count is here rather than left for the caller to derive, because the interesting
    numbers are the ones that say a gold did NOT land: a run whose ``rows_substituted`` is 0
    while ``groups_qualifying`` is 40 is a gold arm that trained on no gold, and that has to
    be visible in one line rather than reconstructed from two.

    Args:
        n_groups: Groups in the batch.
        n_rows: Rows in the batch.
        groups_qualifying: Groups the rule selected.
        rows_substituted: Rows actually replaced by a gold. One per served group.
        gold_tokens: Gold tokens now carrying loss mass.
        loss_tokens: ALL masked tokens in the batch after substitution -- the loss's own
            denominator, ``loss_mask.count_nonzero()`` at ``functional.py:506``. Carried so
            that ``gold_tokens`` can be reported as a FRACTION of what the objective averages
            over, which is the only form in which two arms can be said to be matched: the
            audit measures a row's share of the update as proportional to its token count, so
            arms matched on gold-ROW count can still differ several-fold here.
        groups_no_gold: Qualifying groups skipped because their dataset row had no usable gold
            text.
        groups_no_fit: Qualifying groups skipped because prompt + gold exceeded the batch
            width. Counted apart from ``groups_no_gold`` because the two have different fixes
            -- a missing solution is a dataset problem, a gold that does not fit is a
            sequence-length problem -- and one combined "skipped" count would hide which.
        qualifying_groups: Indices of the groups the rule selected, in batch order. Carried so
            a caller can apply the parts of a baseline's rule that are advantage-level (DyME
            zeroes an all-correct group, and the non-gold rows of an all-wrong one) without
            re-deriving the predicate from the rewards a second time and drifting from it.
        substituted_rows: Row indices that now hold a gold, in batch order.
        served_by: ``(source, count)`` pairs -- how many rows each supplier served. The gold
            path had one supplier and needed no such breakdown; with four, "40 rows were
            substituted" does not say whether the arm ran on gold or on a teacher, and the two
            are different experiments.
        refusals: ``(reason, count)`` pairs over EVERY refusal, including one a later link of
            the chain recovered from. This is the attempt-level view: a chain whose first
            supplier refuses on most groups is paying for a lookup that never lands, which is
            invisible in the group-level counts.
        unserved_groups: ``(reason, count)`` pairs over qualifying groups that ended with NO
            source, keyed on the last refusal they saw. The group-level view, and the one that
            says how much reach was lost. ``selfevo/CONTRIBUTING.md`` states the rule these two
            exist for: unserved is a mapping of REASONS, not a count, because every zero in this
            repo has an open artifact behind it.
        decisions: The realised source per QUALIFYING group, in batch order, with an empty
            string where nothing served it. This is the vector
            :class:`selfevo.supply.policy.MatchedSourceControl` permutes, and it keeps the
            unserved groups' slots on purpose: dropping them would match the two arms on served
            groups only, which is the quantity under test.
    """

    n_groups: int
    n_rows: int
    groups_qualifying: int = 0
    rows_substituted: int = 0
    gold_tokens: int = 0
    loss_tokens: int = 0
    groups_no_gold: int = 0
    groups_no_fit: int = 0
    qualifying_groups: tuple[int, ...] = field(default_factory=tuple)
    substituted_rows: tuple[int, ...] = field(default_factory=tuple)
    served_by: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    refusals: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    unserved_groups: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    decisions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def token_mass(self) -> float:
        """Fraction of the loss's denominator that is gold.

        This, and not ``rows_substituted``, is the size of the intervention: the objective is
        one per-token mean over the global batch, so a gold row's share of the update is
        proportional to its token count. Measured in the audit at 0.5 / 1.0 / 2.0 relative
        gradient magnitude for SFT rows of 4 / 8 / 16 tokens against 4-token RL rows.
        """
        return self.gold_tokens / self.loss_tokens if self.loss_tokens else 0.0

    def as_metrics(self) -> dict[str, float]:
        """Flat metrics under a ``gold/`` prefix.

        Prefixed like ``route/`` for the reason that namespace exists: these have to sit on
        one panel beside the routing keys without colliding with the actor's own, and a gold
        arm and a non-gold arm have to emit the SAME key set so the two stay readable together
        -- which they do, because a ``none`` run emits these as zeros rather than omitting
        them.

        ``gold/token_mass`` is the key the loss-weighting audit asks for by name, and that no
        existing ``route/*`` key supplies.
        """
        from selfevo.supply.base import SUPPLY_SOURCES, Refusal

        served = dict(self.served_by)
        refused = dict(self.refusals)
        unserved = dict(self.unserved_groups)
        out = {
            "gold/groups_qualifying": float(self.groups_qualifying),
            "gold/rows_substituted": float(self.rows_substituted),
            "gold/tokens": float(self.gold_tokens),
            "gold/loss_tokens": float(self.loss_tokens),
            "gold/token_mass": self.token_mass,
            "gold/groups_no_gold": float(self.groups_no_gold),
            "gold/groups_no_fit": float(self.groups_no_fit),
            "gold/qualifying_group_fraction": (
                self.groups_qualifying / self.n_groups if self.n_groups else 0.0
            ),
            "supply/sources_used": float(sum(1 for v in served.values() if v)),
        }
        # Every source and every reason, ALWAYS, including the zeros. A one-source arm and a
        # four-source arm then emit the same key set and stay readable on one panel, which is
        # the same reason a `none` run emits the gold keys as zeros rather than omitting them.
        for name in SUPPLY_SOURCES:
            out[f"supply/served/{name}"] = float(served.get(name, 0))
        for reason in Refusal:
            out[f"supply/refused/{reason.value}"] = float(refused.get(reason.value, 0))
            out[f"supply/unserved/{reason.value}"] = float(unserved.get(reason.value, 0))
        return out


def _normalise_group_sizes(
    group_sizes: Sequence[int] | int | None, n_rows: int
) -> list[int]:
    """Row counts per group, validated against the batch.

    Args:
        group_sizes: A per-group list, a uniform int, or None meaning "the whole batch is one
            group" -- which is what a per-trajectory dict out of ``prepare_batch`` is, since
            ``GroupedRolloutWorkflow`` concatenates a prompt's ``n_samples`` rollouts into
            exactly one dict.
        n_rows: Rows in the batch.

    Returns:
        The sizes as a list.

    Raises:
        GoldShapeError: If the sizes do not partition the batch. Guessing a grouping is worse
            than refusing: a wrong partition makes the rule read rewards across unrelated
            prompts, and the "all-wrong group" it then finds is an artifact.
    """
    if group_sizes is None:
        sizes = [n_rows]
    elif isinstance(group_sizes, int):
        if group_sizes <= 0 or n_rows % group_sizes:
            raise GoldShapeError(
                f"uniform group size {group_sizes} does not divide {n_rows} rows"
            )
        sizes = [group_sizes] * (n_rows // group_sizes)
    else:
        sizes = [int(s) for s in group_sizes]
    if not sizes or any(s <= 0 for s in sizes) or sum(sizes) != n_rows:
        raise GoldShapeError(
            f"group sizes {sizes} do not partition {n_rows} rows; refusing to guess a "
            "grouping, because a wrong one makes the rule read rewards across prompts"
        )
    return sizes


def _qualifies(rule: GoldRule, rewards: torch.Tensor) -> bool:
    """Whether one group's rewards select it for a gold row.

    Args:
        rule: The baseline's rule.
        rewards: ``(g,)`` RAW per-rollout rewards for one group. Raw, not normalised: the
            actor's own solved/unsolved split records that thresholding a group-normalised
            reward "reported 0% solved at an 82% solve rate", and both papers' predicates are
            statements about correctness.

    Returns:
        True if the group qualifies.

    Raises:
        GoldPolicyError: For a rule with no predicate, which can only mean a member was added
            to :class:`GoldRule` without teaching this function what it means.
    """
    if rule is GoldRule.DYME:
        # DyMETrainer.py:655 -- has_correct == 0 over (acc_rewards > 0.5).
        return bool((rewards > _CORRECT).sum() == 0)
    if rule is GoldRule.LSPO_CLIFF:
        # arXiv 2607.27787 -- C = {x : sum_k R(x, y_k) = 0}. Written as the paper writes it,
        # which is NOT the same predicate as DyME's once a reward can be negative.
        return bool(torch.isclose(rewards.sum(), torch.zeros((), dtype=rewards.dtype)))
    raise GoldPolicyError(f"no predicate for rule {rule!r}")


def substitute_gold_rows(
    batch: Mapping[str, Any],
    rule: GoldRule | str,
    *,
    group_sizes: Sequence[int] | int | None = None,
    logprob_policy: GoldLogprobPolicy | str = GoldLogprobPolicy.PROX_RECOMPUTE,
    gold_reward: float = 1.0,
    pad_value: float = 0.0,
    suppliers: Mapping[str, "Supplier"] | None = None,
    source_policy: "SourcePolicy | None" = None,
    group_offset: int = 0,
    qualifying_offset: int = 0,
) -> tuple[dict[str, Any], GoldStats]:
    """Replace one rollout of each qualifying group with the group's gold solution.

    Pure: the input batch and its tensors are never mutated, and nothing here imports the
    actor, the trainer or an engine. The seam a caller adds is this call, plus
    :func:`reconcile_gold_logprobs` once the log-probabilities exist.

    Args:
        batch: A ROLLOUT batch, i.e. before ``compute_logp`` and before
            ``_compute_advantages``. Tensors are ``(B, T)`` except ``rewards``, which is
            ``(B,)``. ``loss_mask`` must be in the TOKEN coordinates a workflow emits, not the
            left-rolled emitter coordinates ``_compute_advantages`` writes back, because the
            prompt boundary is read from it.
        rule: ``"dyme"``, ``"lspo_cliff"`` or ``"none"``.
        group_sizes: Rows per GRPO group. ``None`` means the whole batch is one group, which
            is what one element of ``prepare_batch``'s list is.
        logprob_policy: What to write for tokens that were never sampled. See
            :class:`GoldLogprobPolicy`.
        gold_reward: The reward given to the gold row. 1.0 -- the same value a correct rollout
            gets -- so the ORDINARY estimator gives it a positive advantage, rather than a
            special case being carved into the advantage tensor for it. That is the entire
            point of doing this by batch construction.
        pad_value: Value written past the end of a rewritten row, matching
            ``concat_padded_tensors``' own padding so a substituted row is indistinguishable
            in its padding from a collated one.
        suppliers: ``{name: supplier}`` for this arm, e.g. from
            ``selfevo.supply.build_suppliers``. ``None`` -- the default -- means the gold-only
            mapping, and is the OFF state of the supplier axis: the output, its key set
            included, is byte-identical to the version of this function that had no suppliers.
        source_policy: Which supplier serves which group, as a
            :class:`selfevo.supply.policy.SourcePolicy`. ``None`` means try every configured
            supplier in :data:`selfevo.supply.base.SUPPLY_SOURCES` order. The choice of source
            is a routing decision and is expressible per group here, but nothing in this repo
            LEARNS it: the shipped policy is fixed and its mandatory control is
            :class:`selfevo.supply.policy.MatchedSourceControl`.
        group_offset: Index of this batch's first group within a larger batch, so a policy
            replaying a batch-wide assignment sees batch-global group indices.
        qualifying_offset: How many qualifying groups precede this batch, for the same reason.
            :func:`substitute_in_place` threads both, which is the only way a forced or
            permuted assignment can span a list of per-prompt dicts without misaligning.

    Returns:
        ``(new_batch, stats)``. ``new_batch`` additionally carries ``is_gold``, a ``(B, T)``
        int32 tensor that is 1 on every token of a SUBSTITUTED row -- the name predates the
        other suppliers and is kept because LSPO's adapter router and
        :func:`reconcile_gold_logprobs` both read it, and what it means to them ("this row was
        not emitted by the current policy") is true of every source. When, and only when, the
        supplier axis is engaged -- ``suppliers`` or ``source_policy`` was passed -- the batch
        also carries ``source_ids``, a ``(B, T)`` int32 tensor holding
        :func:`selfevo.supply.base.source_code` of the supplier that served the row and 0
        elsewhere. Conditional so the off state adds no tensor to the pipeline;
        :func:`substitute_in_place` applies the same condition to every element of the list, so
        no two collated dicts can disagree on the key. Per TOKEN and not per row for the
        reason ``_compute_advantages`` gives for ``group_ids``: a ``(B,)`` tensor does not
        survive microbatch splitting and packing, and arrives at the loss with the wrong
        length. LSPO's adapter router reads exactly this tensor to send gold rows to the
        adapter and everything else to the base.

    Raises:
        GoldOrderingError: If the batch already carries ``prox_logp``/``ref_logp``/
            ``teacher_logp``, i.e. this was called after ``compute_logp``.
        GoldShapeError: If required keys are missing or shapes disagree.
        GoldMissingError: If a rule is on and no gold reached the update -- either the batch
            carries no gold at all, or every qualifying group's gold was unusable. This is the
            reach guard, and it is why this function refuses rather than returning a batch
            that is quietly identical to its input.
        GoldPolicyError: For an unknown rule or policy name, an empty supplier mapping, or a
            source policy naming a supplier this arm did not build.
    """
    try:
        rule = GoldRule(rule)
    except ValueError as exc:
        raise GoldPolicyError(
            f"unknown gold rule {rule!r}; expected one of {[r.value for r in GoldRule]}"
        ) from exc
    try:
        policy = GoldLogprobPolicy(logprob_policy)
    except ValueError as exc:
        raise GoldPolicyError(
            f"unknown gold logprob policy {logprob_policy!r}; expected one of "
            f"{[p.value for p in GoldLogprobPolicy]}"
        ) from exc

    n_rows = _batch_rows(batch)
    if rule is GoldRule.NONE:
        # A true no-op: no copy, no guard, no is_gold key. A batch that never asked for gold
        # must come back the way it went in, so the off arm is bit-identical.
        return dict(batch), GoldStats(
            n_groups=len(_safe_sizes(group_sizes, n_rows)),
            n_rows=n_rows,
            loss_tokens=_loss_tokens(batch),
        )

    # Resolved AFTER the `none` early return, so a misconfigured supplier cannot fail a run
    # that asked for no substitution at all -- the same rule `_safe_sizes` follows for a bad
    # grouping. The off arm stays inert even when its neighbours are wrong.
    supply_map, src_policy, engaged = _resolve_supply(suppliers, source_policy)

    stale = [k for k in _STALE_AFTER_SUBSTITUTION if k in batch]
    if stale:
        raise GoldOrderingError(
            f"batch already carries {stale}, so it has been through compute_logp. Those "
            "tensors describe the tokens that were in the batch when they were computed; "
            "replacing a row now leaves them the right shape and the wrong content, with "
            "nothing downstream to notice. Substitute BEFORE compute_logp."
        )
    missing = [k for k in _REQUIRED if k not in batch]
    if missing:
        raise GoldShapeError(f"rollout batch is missing {missing}")
    from selfevo.supply.base import missing_key_hint

    used = _used_suppliers(supply_map, src_policy)
    supplied_keys: list[str] = []
    for sup in used:
        for key in sup.required_keys:
            if key not in batch:
                raise GoldMissingError(
                    f"rule={rule.value} but " + missing_key_hint(sup, key)
                )
            if key not in supplied_keys:
                supplied_keys.append(key)

    sizes = _normalise_group_sizes(group_sizes, n_rows)
    width = int(batch["input_ids"].shape[1])
    for key in ("loss_mask", "attention_mask", "logprobs", *supplied_keys):
        t = batch[key]
        if not torch.is_tensor(t) or t.ndim != 2 or tuple(t.shape) != (n_rows, width):
            raise GoldShapeError(
                f"{key} has shape {getattr(t, 'shape', None)}, expected {(n_rows, width)}. "
                "A gold tensor of its own natural length survives collation and breaks at "
                "packing instead, so it is checked here."
            )

    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    is_gold = torch.zeros((n_rows, width), dtype=torch.int32)
    source_ids = torch.zeros((n_rows, width), dtype=torch.int32)
    rewards = batch["rewards"].detach().float()
    raw_rewards = batch.get("original_rewards")
    raw_rewards = rewards if raw_rewards is None else raw_rewards.detach().float()
    prompt_len = prompt_lengths(batch["loss_mask"], batch["attention_mask"])

    from selfevo.supply.base import (
        NO_SOURCE,
        Refusal,
        SupplierRefused,
        SupplyRequest,
        source_code,
    )

    qualifying: list[int] = []
    substituted: list[int] = []
    decisions: list[str] = []
    served_by: Counter = Counter()
    refusals: Counter = Counter()
    unserved_groups: Counter = Counter()
    gold_tokens = 0
    no_gold = 0
    no_fit = 0

    start = 0
    q = 0
    for gi, size in enumerate(sizes):
        rows = slice(start, start + size)
        start += size
        if not _qualifies(rule, raw_rewards[rows]):
            continue
        qualifying.append(gi)
        # DyME replaces the group's FIRST rollout (`i % num_generations == 0`). Kept identical
        # so the baseline is a reproduction rather than a variant; which row is sacrificed
        # cannot matter to the estimator, since every row of a qualifying group scored alike
        # by construction.
        victim = rows.start
        p_len = int(prompt_len[victim])
        request = SupplyRequest(
            batch=batch,
            row=victim,
            group=group_offset + gi,
            prompt_len=p_len,
            width=width,
        )
        chain = tuple(src_policy.chain_for(group_offset + gi, qualifying_offset + q))
        q += 1

        offer = None
        last_reason = Refusal.NO_SOURCE
        gold_reason: Refusal | None = None
        for name in chain:
            try:
                offer = supply_map[name].supply(request)
                break
            except SupplierRefused as exc:
                refusals[exc.reason.value] += 1
                last_reason = exc.reason
                if name == "gold" and gold_reason is None:
                    gold_reason = exc.reason
        if offer is None:
            # NEVER a silent pass: the row is left exactly as the rollout produced it, the
            # group is recorded as unserved with the reason that ended it, and the two
            # gold-specific counters keep the meaning they had when gold was the only
            # supplier -- groups that ended UNSERVED because the gold was missing, or because
            # it did not fit.
            decisions.append(NO_SOURCE)
            unserved_groups[last_reason.value] += 1
            if gold_reason is Refusal.NO_GOLD:
                no_gold += 1
            elif gold_reason is Refusal.NO_FIT:
                no_fit += 1
            continue
        _write_gold_row(
            out,
            row=victim,
            prompt_len=p_len,
            gold=offer.token_ids,
            policy=policy,
            gold_reward=gold_reward,
            pad_value=pad_value,
        )
        is_gold[victim, :] = 1
        source_ids[victim, :] = source_code(offer.source)
        substituted.append(victim)
        decisions.append(offer.source)
        served_by[offer.source] += 1
        gold_tokens += offer.n_tokens

    out["is_gold"] = is_gold
    if engaged:
        out["source_ids"] = source_ids
    stats = GoldStats(
        n_groups=len(sizes),
        n_rows=n_rows,
        groups_qualifying=len(qualifying),
        rows_substituted=len(substituted),
        gold_tokens=gold_tokens,
        loss_tokens=_loss_tokens(out),
        groups_no_gold=no_gold,
        groups_no_fit=no_fit,
        qualifying_groups=tuple(qualifying),
        substituted_rows=tuple(substituted),
        served_by=_counts(served_by),
        refusals=_counts(refusals),
        unserved_groups=_counts(unserved_groups),
        decisions=tuple(decisions),
    )

    # The two reach guards, raised AFTER the counting so each refusal carries the numbers
    # that explain it. The empty-supply one is checked first because it is the more diagnostic
    # of the two: it says the supply failed, not the match. Generalised from "every gold_mask
    # is empty" to "not one configured supplier has anything for this batch", which is the same
    # statement when gold is the only supplier -- and with gold alone the message is unchanged,
    # word for word, because the supplier owns it.
    from selfevo.supply.base import no_supply_message

    if not any(sup.has_supply(batch) for sup in used):
        raise GoldMissingError(
            "; ".join(no_supply_message(sup, rule.value) for sup in used), stats
        )
    if qualifying and not substituted:
        raise GoldMissingError(
            f"rule={rule.value} selected {len(qualifying)} groups and gave a gold to none of "
            f"them ({no_gold} had no gold text, {no_fit} did not fit in {width} tokens). The "
            "arm would report itself as gold-grounded and train on nothing.",
            stats,
        )
    return out, stats


def _batch_rows(batch: Mapping[str, Any]) -> int:
    """Rows in the batch, read from ``input_ids``.

    Raises:
        GoldShapeError: If ``input_ids`` is absent or not 2-D, which every other check here
            would otherwise report as some more confusing symptom.
    """
    ids = batch.get("input_ids")
    if not torch.is_tensor(ids) or ids.ndim != 2:
        raise GoldShapeError(
            "batch needs a 2-D input_ids tensor; got "
            f"{type(ids).__name__} with shape {getattr(ids, 'shape', None)}"
        )
    return int(ids.shape[0])


def _loss_tokens(batch: Mapping[str, Any]) -> int:
    """The loss's own denominator for this batch: masked tokens, counted the way it counts.

    ``functional.py:506`` takes ``loss_mask.count_nonzero()`` and divides the summed surrogate
    by it, so this is the unit in which one row's share of the update is measured. Read here
    rather than by the caller so ``gold_tokens`` and its denominator can never be computed
    against different masks.
    """
    lm = batch.get("loss_mask")
    return int(lm.count_nonzero()) if torch.is_tensor(lm) else 0


def _counts(counter: Counter) -> tuple[tuple[str, int], ...]:
    """A counter as sorted ``(key, count)`` pairs.

    Sorted so two runs of the same batch produce byte-identical stats, and pairs rather than a
    dict so :class:`GoldStats` stays a frozen dataclass whose fields can be compared and
    summed without a caller mutating one arm's counts through the other's reference.
    """
    return tuple(sorted(counter.items()))


def _resolve_supply(
    suppliers: Mapping[str, "Supplier"] | None,
    source_policy: "SourcePolicy | None",
) -> tuple[dict[str, "Supplier"], "SourcePolicy", bool]:
    """Resolve the supplier mapping and the source policy, and say whether the axis is on.

    Args:
        suppliers: ``{name: supplier}`` or None for the gold-only default.
        source_policy: A policy or None for "try every configured supplier in registry order".

    Returns:
        ``(supply_map, policy, engaged)``. ``engaged`` is False exactly when the caller passed
        neither argument, which is the OFF state of the supplier axis: the seam then behaves,
        and emits, precisely what it did before suppliers existed.

    Raises:
        GoldPolicyError: For an empty mapping -- an arm that configured no supplier at all is
            an off arm wearing an on arm's label -- or for a policy naming a supplier this arm
            did not build, which is the typo that would otherwise die after model load.
    """
    from selfevo.supply import default_suppliers
    from selfevo.supply.base import SUPPLY_SOURCES
    from selfevo.supply.policy import FixedSourcePolicy

    engaged = suppliers is not None or source_policy is not None
    supply_map = dict(default_suppliers()) if suppliers is None else dict(suppliers)
    if not supply_map:
        raise GoldPolicyError(
            "suppliers={} names no source at all, so every qualifying group would be refused "
            "while the arm reported itself as grounded. Pass suppliers=None for the gold-only "
            "default, or name at least one source."
        )
    ordered = tuple(n for n in SUPPLY_SOURCES if n in supply_map)
    policy = FixedSourcePolicy(ordered) if source_policy is None else source_policy
    unknown = sorted(set(policy.sources()) - set(supply_map))
    if unknown:
        raise GoldPolicyError(
            f"source policy {getattr(policy, 'name', policy)!r} names {unknown}, which this "
            f"arm did not build (it has {sorted(supply_map)}). Refusing rather than skipping: "
            "a silently-skipped source is an arm that reports a mixture it never ran."
        )
    return supply_map, policy, engaged


def _used_suppliers(
    supply_map: Mapping[str, "Supplier"], policy: "SourcePolicy"
) -> list["Supplier"]:
    """The suppliers a policy could name, in registry order.

    Registry order rather than policy order so the batch-level key check and the batch-level
    reach guard report their causes in the same sequence whatever a policy's chain happens to
    be -- and so the gold-only case reports exactly what it reported before.
    """
    from selfevo.supply.base import SUPPLY_SOURCES

    names = set(policy.sources())
    return [supply_map[n] for n in SUPPLY_SOURCES if n in names and n in supply_map]


def _safe_sizes(group_sizes: Sequence[int] | int | None, n_rows: int) -> list[int]:
    """Group sizes for the ``none`` path, where a bad grouping must not raise.

    The off arm has to stay inert even when its neighbours are misconfigured, so a grouping
    that does not partition the batch degrades to "one group" for the purpose of a count
    nobody acts on, rather than failing a run that asked for no gold at all.
    """
    try:
        return _normalise_group_sizes(group_sizes, n_rows)
    except GoldShapeError:
        return [n_rows]


def _write_gold_row(
    out: dict[str, Any],
    *,
    row: int,
    prompt_len: int,
    gold: torch.Tensor,
    policy: GoldLogprobPolicy,
    gold_reward: float,
    pad_value: float,
) -> None:
    """Rewrite one row in place inside the already-copied output batch.

    The row becomes ``prompt ++ gold ++ padding``: the prompt is kept because the gold is a
    solution TO THAT PROMPT, and a target detached from its question teaches the model to emit
    the derivation unconditionally.

    Every tensor the pipeline carries per token is rewritten, not only the obvious three,
    because a stale one is invisible:

    * ``input_ids`` -- prompt then gold.
    * ``attention_mask`` -- true over prompt + gold, false after, so ``pack_tensor_dict`` packs
      exactly the real tokens and ``seqlens`` is right.
    * ``loss_mask`` -- 0 on the prompt, 1 on the gold. This is what makes the update an SFT
      step on the gold and nothing else.
    * ``logprobs`` -- 0.0 on the prompt, as a rollout writes for prompt positions, and the
      policy's value on the gold. See :class:`GoldLogprobPolicy`.
    * ``versions`` -- -1 everywhere, the sentinel a prompt token already carries, because a
      gold token came from no policy version. A run using ``prox_logp_method`` other than
      ``recompute`` extrapolates from versions and would read these as maximally stale; that
      is the honest reading, and the live configuration recomputes.
    * ``turn_ids`` -- -1 on the prompt, 0 on the gold, matching a single-turn response.
    * ``rewards`` and ``original_rewards`` -- ``gold_reward``.
    """
    ids = out["input_ids"]
    width = int(ids.shape[1])
    n_gold = int(gold.shape[0])
    end = prompt_len + n_gold

    ids[row, prompt_len:end] = gold.to(ids.dtype)
    if end < width:
        ids[row, end:] = int(pad_value)

    am = out["attention_mask"]
    am[row, :end] = torch.ones((), dtype=am.dtype)
    if end < width:
        am[row, end:] = torch.zeros((), dtype=am.dtype)

    lm = out["loss_mask"]
    lm[row, :] = 0
    lm[row, prompt_len:end] = 1

    lp = out["logprobs"]
    fill = GOLD_LOGP_SENTINEL if policy is GoldLogprobPolicy.PROX_RECOMPUTE else 0.0
    lp[row, :] = 0.0
    lp[row, prompt_len:end] = fill

    if "versions" in out and torch.is_tensor(out["versions"]):
        out["versions"][row, :] = -1
    if "turn_ids" in out and torch.is_tensor(out["turn_ids"]):
        out["turn_ids"][row, :] = -1
        out["turn_ids"][row, prompt_len:end] = 0

    out["rewards"][row] = gold_reward
    if "original_rewards" in out and torch.is_tensor(out["original_rewards"]):
        out["original_rewards"][row] = gold_reward


def substitute_in_place(
    trajectories: list[dict[str, Any]],
    rule: GoldRule | str,
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], GoldStats]:
    """Apply :func:`substitute_gold_rows` to a list of per-group trajectory dicts.

    This is the shape the trainer actually holds: ``prepare_batch`` returns ``list[dict]``
    with one dict per PROMPT, each already concatenated over that prompt's ``n_samples``
    rollouts by ``GroupedRolloutWorkflow`` -- which is why ``concat_batch`` reads
    ``traj_group_sizes`` off each dict's first dimension. So each element is exactly one group
    and no grouping has to be reconstructed.

    Args:
        trajectories: One dict per group. Not mutated; new dicts are returned.
        rule: As :func:`substitute_gold_rows`.
        **kwargs: Forwarded, except ``group_sizes``, which is fixed per element, and
            ``group_offset``/``qualifying_offset``, which this function OWNS. It threads
            batch-global indices through the list so a forced or permuted source assignment --
            which is indexed by position among qualifying groups across the whole batch --
            cannot silently restart at 0 on every prompt and reuse the first few decisions for
            all of them.

    Returns:
        ``(new_trajectories, stats)`` with the stats summed over the list, so a caller logs one
        set of numbers for the batch rather than one per prompt. ``token_mass`` is then the
        batch-wide gold fraction, which is the quantity the loss's single per-token mean reads.

    Raises:
        GoldMissingError: If a rule is on and nothing anywhere in the list received a gold.
            Checked ACROSS the list and not per element, because a single prompt whose group
            happens to be solved is the ordinary case and must not fail a step.
    """
    if not trajectories:
        return list(trajectories), GoldStats(n_groups=0, n_rows=0)

    kwargs.pop("group_sizes", None)
    kwargs.pop("group_offset", None)
    kwargs.pop("qualifying_offset", None)
    out: list[dict[str, Any]] = []
    totals = dict(
        n_groups=0, n_rows=0, groups_qualifying=0, rows_substituted=0,
        gold_tokens=0, loss_tokens=0, groups_no_gold=0, groups_no_fit=0,
    )
    served_by: Counter = Counter()
    refusals: Counter = Counter()
    unserved_groups: Counter = Counter()
    decisions: list[str] = []
    qualifying: list[int] = []
    substituted: list[int] = []
    row_offset = 0
    qualifying_offset = 0
    deferred: GoldMissingError | None = None
    for gi, traj in enumerate(trajectories):
        try:
            new, st = substitute_gold_rows(
                traj,
                rule,
                group_sizes=None,
                group_offset=gi,
                qualifying_offset=qualifying_offset,
                **kwargs,
            )
        except GoldMissingError as exc:
            # A group that qualified and could not be served is not by itself a failed step;
            # the batch-level guard below decides. The first such refusal is kept so the
            # message the caller finally sees names a real cause rather than a summary, and
            # the refusal's own counts are folded in rather than dropped -- otherwise the
            # unserved group disappears from groups_qualifying and groups_no_gold, and the
            # batch report understates exactly the loss of reach it exists to show.
            deferred = deferred or exc
            new = dict(traj)
            st = exc.stats or GoldStats(
                n_groups=1, n_rows=_batch_rows(traj), loss_tokens=_loss_tokens(traj)
            )
        out.append(new)
        totals["n_groups"] += st.n_groups
        totals["n_rows"] += st.n_rows
        totals["groups_qualifying"] += st.groups_qualifying
        totals["rows_substituted"] += st.rows_substituted
        totals["gold_tokens"] += st.gold_tokens
        totals["loss_tokens"] += st.loss_tokens
        totals["groups_no_gold"] += st.groups_no_gold
        totals["groups_no_fit"] += st.groups_no_fit
        served_by.update(dict(st.served_by))
        refusals.update(dict(st.refusals))
        unserved_groups.update(dict(st.unserved_groups))
        decisions.extend(st.decisions)
        if st.groups_qualifying:
            qualifying.append(gi)
        substituted.extend(row_offset + r for r in st.substituted_rows)
        row_offset += st.n_rows
        qualifying_offset += st.groups_qualifying

    stats = GoldStats(
        **totals,
        qualifying_groups=tuple(qualifying),
        substituted_rows=tuple(substituted),
        served_by=_counts(served_by),
        refusals=_counts(refusals),
        unserved_groups=_counts(unserved_groups),
        decisions=tuple(decisions),
    )
    if GoldRule(rule) is not GoldRule.NONE and not stats.rows_substituted:
        if deferred is not None:
            raise deferred
        if stats.groups_qualifying:
            raise GoldMissingError(
                f"{stats.groups_qualifying} groups qualified across the batch and none "
                "received a gold row"
            )
    return out, stats


def reconcile_gold_logprobs(
    data: Mapping[str, Any],
    *,
    logprob_policy: GoldLogprobPolicy | str = GoldLogprobPolicy.PROX_RECOMPUTE,
) -> tuple[dict[str, Any], int]:
    """Give the gold tokens their behaviour log-probability, AFTER ``compute_logp``.

    The second half of the two-phase protocol :class:`GoldLogprobPolicy` describes. It
    replaces :data:`GOLD_LOGP_SENTINEL` with the trainer's own recomputed ``prox_logp``, which
    ``selfevo/FINDINGS_loss_weighting.md`` section 3 identifies as the only value leaving the
    surrogate exactly as the gold row's advantage intends: ``log_ratio = prox - old = 0``, so
    the live rejection sampling (level=token, metric=ratio, upper=5.0) computes
    ``behave_imp_weight = 1`` and multiplies the row by 1 rather than by ``exp(prox_logp)``.

    THE COORDINATE SHIFT IS THE WHOLE DIFFICULTY, and getting it wrong is silent. ``logprobs``
    from inference is in TOKEN coordinates -- ``[0.0] * input_len + output_logprobs``, entry
    ``t`` belonging to the token AT position ``t`` -- while ``prox_logp`` comes from
    ``gather_logprobs(logits, roll(input_ids, -1))`` (``fsdp_engine.py:2116-2121``) and is in
    EMITTER coordinates, entry ``t`` being ``log p(token t+1 | <= t)``.
    ``_compute_advantages`` reconciles them by rolling ``logprobs`` LEFT by one, so what must
    be written here is ``prox_logp`` rolled RIGHT by one. Writing it unrolled shifts every gold
    ratio by one position, which changes no shape and raises nothing.

    Column 0 is never written: rolling right wraps the last position onto it, and a token at
    position 0 has no predecessor that could have emitted it. It is a prompt position in every
    trajectory this pipeline produces, so it is masked out of the loss regardless.

    Args:
        data: The batch after ``compute_logp``, carrying ``is_gold``, ``logprobs`` and
            ``prox_logp``.
        logprob_policy: The policy used at substitution.

    Returns:
        ``(new_data, n_rows)`` -- a new dict, and the number of gold rows reconciled.

    Raises:
        GoldPolicyError: Under ``ratio_one``, always: that policy asserts an importance ratio
            of 1 which only the loss can enforce, and silently doing nothing here would let a
            run believe its gold rows had been handled.
        GoldOrderingError: If ``prox_logp`` is absent, which is the one state that cannot be
            repaired -- the gold tokens then have no defined log-probability to adopt, and
            their sentinel would reach the loss.
    """
    policy = GoldLogprobPolicy(logprob_policy)
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
    is_gold = data.get("is_gold")
    if not torch.is_tensor(is_gold) or int(is_gold.sum()) == 0:
        return out, 0

    if policy is GoldLogprobPolicy.RATIO_ONE:
        raise GoldPolicyError(
            "logprob_policy='ratio_one' cannot be reconciled from the batch: it asserts an "
            "importance ratio of 1 that only the loss can enforce, and nothing in "
            "grpo_loss_fn reads is_gold today. Use 'prox_recompute', or land the loss-side "
            "change first."
        )

    prox = data.get("prox_logp")
    if not torch.is_tensor(prox):
        raise GoldOrderingError(
            "prox_logp is absent, so the gold tokens have no defined log-probability to adopt "
            "and would reach the loss carrying the sentinel, where the audit measured them "
            "silently reweighted by exp(prox_logp - 1). Run compute_logp first."
        )

    gold_tok = (is_gold.bool() & data["loss_mask"].bool()).clone()
    gold_tok[:, 0] = False
    shifted = torch.roll(prox, shifts=1, dims=-1).to(data["logprobs"].dtype)
    out["logprobs"] = torch.where(gold_tok, shifted, out["logprobs"])
    assert_gold_logprobs_filled(out)
    return out, int(is_gold.any(dim=-1).sum())


def assert_gold_logprobs_filled(data: Mapping[str, Any]) -> None:
    """Refuse to let a gold row reach the loss still carrying the sentinel.

    A log-probability is at most 0, so any positive entry on a gold token is an unfilled
    :data:`GOLD_LOGP_SENTINEL`. That state is not survivable-but-suboptimal, it is
    unreportable: the audit measured an unfilled row being multiplied by
    ``exp(prox_logp - 1) < 1`` at ``functional.py:568`` with ``filtered_fraction`` still 0.0
    and no metric changing. Non-finite entries are refused by the same check, because
    ``-0.0 * NaN`` turns every advantage in a batch-normalised batch to NaN one stage before
    the loss.

    Args:
        data: A batch carrying ``is_gold`` and ``logprobs``. A batch with no gold passes
            trivially, so this is safe to call unconditionally.

    Raises:
        GoldOrderingError: If any gold token's behaviour log-probability is positive or
            non-finite.
    """
    is_gold = data.get("is_gold")
    lp = data.get("logprobs")
    if not torch.is_tensor(is_gold) or not torch.is_tensor(lp):
        return
    gold_tok = is_gold.bool()
    if not bool(gold_tok.any()):
        return
    vals = lp[gold_tok]
    bad = (~torch.isfinite(vals)) | (vals > 0)
    if bool(bad.any()):
        raise GoldOrderingError(
            f"{int(bad.sum())} of {vals.numel()} gold-row logprobs are positive or "
            "non-finite, so they were never reconciled against prox_logp. Call "
            "reconcile_gold_logprobs after compute_logp: an unfilled gold row is silently "
            "down-weighted by the behavioural importance weight and reports nothing."
        )
