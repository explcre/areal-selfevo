"""Turn per-group routing decisions into an advantage tensor.

The actor currently hardcodes the rule -- solved groups get a constant, unsolved groups get
another -- which means the audited routers in :mod:`selfevo.routing` decide nothing during
training. This is the seam that lets a Router actually drive the update, so
``router=contextual`` and ``router=code_policy`` become real arms rather than registry
entries.

Kept as a pure tensor function, outside the actor, for one reason: the actor is imported by
live training processes, so logic that lives there can only be tested by running a job.
Everything here is testable on CPU in milliseconds, and the actor's diff is a call.

Mode semantics, chosen so a decision means the same thing whether or not the group was
already silent:

    RL    leave the advantages alone -- ordinary GRPO for this unit.
    SFT   REPLACE the response-token advantages with ``sft_weight``. For a silent group its
          advantages are already zero so replace and add coincide; for an informative group
          they do not, and replace is what "train this unit by SFT instead of RL" means.
          Adding would leave the RL gradient in place and superimpose a supervised one.
    SKIP  zero the advantages. For a silent group this is a no-op; for an informative one it
          is the decision to spend nothing on this unit.

A decision does not have to pick one of them. ``RoutingDecision.weights`` has always been a
``Mapping[str, float]``, so a router can say "60% SFT, 40% RL"; :func:`apply_mixtures`
carries that statement to the tensor instead of collapsing it. Its rule is the linear
interpolation of the three above -- for normalised ``{rl: a, sft: b, skip: c}`` the
response-token advantage becomes ``a * original + b * sft_weight + c * 0`` -- and it is
implemented by CALLING :func:`apply_decisions` for the extremes, so a pure mixture is
bit-identical to the corresponding hard decision by construction rather than by agreement
between two copies of the same arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from selfevo.routing.base import TrainingMode

__all__ = ["ApplyStats", "apply_decisions", "apply_mixtures"]

_APPLIED = (TrainingMode.RL, TrainingMode.SFT, TrainingMode.SKIP)


@dataclass(frozen=True)
class ApplyStats:
    """What a batch's decisions did, for logging next to the run.

    Args:
        counts: ``{mode: mode MASS over the batch}`` -- the sum over groups of that
            mode's normalised weight. For hard decisions every weight is 1.0, so this is
            exactly the number of groups taking each mode and the metric it feeds is
            unchanged; for mixtures it is the fractional equivalent, which is the only
            reading under which "0.6 of this group was RL" can be reported at all.
        changed_rows: Rows whose advantages the decision actually altered, counted where the
            loss reads them. Distinct from the count of non-RL groups: a SKIP on an
            already-silent group changes nothing, and reporting it as an intervention would
            overstate the method's reach.
        n_groups: Groups seen.
        n_rows: Rows seen, i.e. the batch size. Carried because ``counts`` is a per-GROUP
            quantity -- a group count for hard decisions, the same thing measured in
            group-equivalents of mass for mixtures -- and a count of rows divided by it is
            not a fraction: at the live group size of 8 it read eight times too high, and
            above 1.0.
        mixed_groups: Groups whose decision was a genuine mixture, i.e. no single mode held
            all the weight. Always 0 on the hard-decision path. It is reported separately
            from ``counts`` because a mixture run in which every decision happened to come
            out one-hot is an ARGMAX run wearing a mixture label, and mass alone cannot
            distinguish the two: eight one-hot RL groups and eight half-RL/half-SFT groups
            both put mass on more than one mode across the batch.
    """

    counts: dict[str, float]
    changed_rows: int
    n_groups: int
    n_rows: int
    mixed_groups: int = 0

    def as_metrics(self) -> dict[str, float]:
        """Flat metrics, prefixed so they do not collide with the actor's own keys.

        ``route/{mode}_groups`` reports :attr:`counts`, which is mode MASS. For hard
        decisions that is exactly the number of groups taking each mode -- what the key has
        always meant and what every existing panel reads -- and for mixtures it is the
        fractional equivalent. The key is deliberately NOT renamed: the two arms have to
        stay readable on one panel, and renaming would silently orphan the history.

        :attr:`mixed_groups` is NOT emitted here. Adding a key would make an argmax run's
        key set differ from a mixture run's, so the actor logs it separately and logs it on
        BOTH branches.
        """
        out = {f"route/{m}_groups": float(n) for m, n in self.counts.items()}
        out["route/changed_row_fraction"] = self.changed_rows / max(self.n_rows, 1)
        out["route/n_groups"] = float(self.n_groups)
        return out


def apply_decisions(
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    group_sizes: list[int],
    modes: list[str],
    *,
    sft_weight: float,
) -> tuple[torch.Tensor, ApplyStats]:
    """Apply one mode per group to the advantage tensor.

    Args:
        advantages: ``(B, T)``, floating point.
        loss_mask: ``(B, T)``, 1 on response tokens. The mask bounds every write -- outside
            it the incoming values survive unchanged -- so a decision can neither put
            gradient on a prompt nor quietly erase what the actor left there.
        group_sizes: Row counts per group; each >= 1, and they must sum to ``B``.
        modes: One mode per group, same order.
        sft_weight: Magnitude written for an SFT group. Must be finite and >= 0: SFT is
            training toward a target believed correct, and a negative weight would train
            away from it.

    Returns:
        ``(advantages, stats)``. The input tensor is not modified in place -- the caller may
        still hold a reference to it, and silently mutating it has already caused one bug in
        this pipeline.

    Raises:
        ValueError: On a shape or rank mismatch, a non-floating-point advantage tensor, a
            grouping that does not partition the batch, an unknown or unsupported mode, or a
            negative or non-finite ``sft_weight``.
    """
    if advantages.shape != loss_mask.shape:
        raise ValueError(
            f"advantages {tuple(advantages.shape)} and loss_mask {tuple(loss_mask.shape)} "
            "must have the same shape"
        )
    if advantages.dim() != 2:
        raise ValueError(
            f"advantages must be (B, T), got {tuple(advantages.shape)}: the row is the unit "
            "this seam slices and counts, and any other rank is sliced and counted along "
            "the wrong axis without raising"
        )
    if not torch.is_floating_point(advantages):
        raise ValueError(
            f"advantages must be floating point, got {advantages.dtype}: an integer tensor "
            "truncates sft_weight (0.7 -> 0), which is a SKIP wearing an SFT label"
        )
    b = advantages.shape[0]
    if any(g < 1 for g in group_sizes):
        raise ValueError(
            f"every group size must be >= 1, got {list(group_sizes)}: a negative size still "
            "passes the sum check below, and the slices it produces silently apply one "
            "group's decision to another group's rows"
        )
    if sum(group_sizes) != b:
        raise ValueError(f"group_sizes sums to {sum(group_sizes)}, batch has {b} rows")
    if len(modes) != len(group_sizes):
        raise ValueError(
            f"{len(modes)} modes for {len(group_sizes)} groups; one decision per group"
        )
    if not math.isfinite(sft_weight):
        raise ValueError(
            f"sft_weight must be finite, got {sft_weight}: every comparison with NaN is "
            "False, so it would pass the sign check below and write NaN into the advantages"
        )
    if sft_weight < 0:
        raise ValueError(
            f"sft_weight must be >= 0, got {sft_weight}: SFT trains TOWARD a target believed "
            "correct, so a negative weight trains away from it"
        )
    unknown = sorted(set(modes) - set(_APPLIED))
    if unknown:
        raise ValueError(
            f"cannot apply modes {unknown}; this seam implements {list(_APPLIED)}. A "
            "teacher-requiring mode needs a target tensor that does not exist here, and "
            "silently treating it as SKIP would report a distillation arm that never ran."
        )

    out = advantages.clone()
    mask = loss_mask.to(out.dtype)
    counts = {m: 0 for m in _APPLIED}
    changed = 0
    start = 0
    for g, mode in zip(group_sizes, modes):
        sl = slice(start, start + g)
        start += g
        counts[mode] += 1
        if mode == TrainingMode.RL:
            continue
        before = out[sl]
        m = mask[sl]
        written = (
            torch.full_like(before, float(sft_weight)) * m
            if mode == TrainingMode.SFT
            else torch.zeros_like(before)
        )
        # The mask bounds the WRITE, not merely its non-zero part. Overwriting a prompt
        # position with zero is invisible to the loss, which masks those positions anyway,
        # but it is visible twice over: it would let changed_rows report a row as reached
        # whose gradient did not move, and it erases the GAE values the actor leaves on
        # prompt positions (measured: -0.87 there for an informative group) in a tensor the
        # caller still holds.
        new = torch.where(m != 0, written, before)
        changed += int((before != new).any(dim=-1).sum())
        out[sl] = new
    return out, ApplyStats(
        counts=counts, changed_rows=changed, n_groups=len(group_sizes), n_rows=b
    )


def _normalised_mixture(mixture: Mapping[str, float], index: int) -> dict[str, float]:
    """Validate one group's mode weights and scale them to sum to 1.

    Args:
        mixture: ``{mode_name: weight}`` for a single group. Weights need not be normalised
            -- ``{rl: 6, sft: 4}`` and ``{rl: 0.6, sft: 0.4}`` are the same mixture --
            because a router that emits scores rather than probabilities should not have to
            know that, and :meth:`RoutingDecision.normalised` makes the same promise.
        index: Position of this mixture in the batch, quoted in every message below. A
            batch carries hundreds of them and "some group was invalid" is not a diagnosis.

    Returns:
        ``{mode_name: weight}`` summing to 1.0. A one-hot input comes back with its single
        weight EXACTLY 1.0 -- ``w / w`` is exact in IEEE754 for any finite non-zero ``w`` --
        which is what lets :func:`apply_mixtures` detect the pure cases by equality and
        reduce to :func:`apply_decisions` bit for bit rather than to within a rounding.

    Raises:
        ValueError: If the mixture is empty, names a mode this seam cannot apply, or
            carries a non-finite, negative, all-zero, or OVERFLOWING set of weights. Each is
            refused rather than repaired. Clamping a negative weight, dropping an unknown
            mode, or treating an all-zero or inf-summing mixture as SKIP would each let a run
            report a mixture arm whose decisions were not the ones it logged -- the same
            failure the unknown-mode guard in :func:`apply_decisions` exists to prevent.
            Per-weight finiteness is not enough for the last of these: two weights of 1e308
            are each finite and sum to inf.
    """
    if not mixture:
        raise ValueError(
            f"group {index} has an empty mixture; spell 'no training signal' as "
            f"{{{TrainingMode.SKIP!r}: 1.0}} so that nothing downstream has to guess what "
            "an absent decision meant"
        )
    unknown_modes = sorted(set(mixture) - set(_APPLIED))
    if unknown_modes:
        raise ValueError(
            f"group {index} mixes modes {unknown_modes}; this seam implements "
            f"{list(_APPLIED)}. A teacher-requiring mode needs a target tensor that does "
            "not exist here, and silently giving it zero weight would report a "
            "distillation component that contributed nothing."
        )
    total = 0.0
    for mode, w in mixture.items():
        value = float(w)
        # Finiteness FIRST. Every comparison with NaN is False, so a NaN weight would pass
        # the sign check below and then propagate through the blend into the advantages,
        # where it destroys the whole update rather than this one group.
        if not math.isfinite(value):
            raise ValueError(
                f"group {index} weight for {mode!r} must be finite, got {w}"
            )
        if value < 0:
            raise ValueError(
                f"group {index} weight for {mode!r} must be >= 0, got {w}: a negative "
                "component subtracts a training signal instead of mixing one in, which is "
                "a fourth mode nobody asked for rather than a proportion"
            )
        total += value
    if total <= 0 or not math.isfinite(total):
        raise ValueError(
            f"group {index} mixture {dict(mixture)} sums to {total}, which cannot be "
            f"normalised. An all-zero mixture is SKIP, so spell it "
            f"{{{TrainingMode.SKIP!r}: 1.0}} and let the log say so. A sum that OVERFLOWS "
            f"to inf is refused for the same reason and not a different one: every "
            f"w / inf is 0.0, so the group would silently train as SKIP while its weights "
            f"said otherwise -- a mixture arm reporting decisions it did not apply."
        )
    return {m: float(w) / total for m, w in mixture.items()}


def apply_mixtures(
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    group_sizes: list[int],
    mixtures: Sequence[Mapping[str, float]],
    *,
    sft_weight: float,
) -> tuple[torch.Tensor, ApplyStats]:
    """Apply a WEIGHTED MIXTURE of modes per group to the advantage tensor.

    The soft counterpart of :func:`apply_decisions`. It reads the whole weight mapping a
    Router returns rather than its ``argmax``, so "train this group 60% by SFT and 40% by
    RL" reaches the tensor as stated instead of arriving as "SFT".

    Semantics, on RESPONSE tokens only, for normalised weights ``{rl: a, sft: b, skip: c}``
    with ``a + b + c == 1``::

        new = a * original_advantage + b * (sft_weight * loss_mask) + c * 0

    Linear, so this is a generalisation and not a second rule: ``a = 1`` is the identity,
    ``b = 1`` is the SFT write, ``c = 1`` is the SKIP write. Those reductions hold BY
    CONSTRUCTION and not by arithmetic coincidence -- the two extreme tensors are produced
    by calling :func:`apply_decisions` itself, so the pure cases cannot drift away from the
    hard path even if someone edits one of them. That matters because rollback to vanilla
    GRPO runs through ``a = 1``, and a rollback claim that depends on two copies of an
    expression agreeing is a rollback claim with a hole in it.

    ``sft_weight * loss_mask`` rather than a bare ``sft_weight``: the mask is 0/1 today, but
    :func:`apply_decisions` already scales its SFT write by the mask VALUE, and the two
    entry points have to stay interchangeable if a weighted mask ever arrives instead of
    silently disagreeing about the magnitude of every SFT write on the day it does.

    Args:
        advantages: ``(B, T)``, floating point.
        loss_mask: ``(B, T)``, non-zero on response tokens. The mask bounds every write --
            outside it the incoming values survive unchanged. The reason is the one spelled
            out in :func:`apply_decisions` and it is not cosmetic: the actor leaves real GAE
            values on prompt positions, and overwriting them with a blend would both erase
            them in a tensor the caller still holds and inflate ``changed_rows``.
        group_sizes: Row counts per group; each >= 1, and they must sum to ``B``.
        mixtures: One ``{mode: weight}`` mapping per group, in the same order. Need not be
            normalised.
        sft_weight: Magnitude of the SFT component at full weight, i.e. the value written
            when ``b == 1``. Must be finite and >= 0, for the reason given in
            :func:`apply_decisions`.

    Returns:
        ``(advantages, stats)``. The caller's tensor is not modified in place. In the
        returned :class:`ApplyStats`, ``counts`` holds mode MASS rather than a group count,
        and ``mixed_groups`` counts the decisions that were genuinely mixed. The reduction
        claim covers the STATS as well as the tensor: for a one-hot mixture every field
        matches what :func:`apply_decisions` returns for the corresponding label, including
        on a batch containing NaN, which is why ``changed_rows`` is accumulated per group
        rather than diffed once at the end.

    Raises:
        ValueError: On everything :func:`apply_decisions` refuses -- shape, rank, dtype,
            partition and ``sft_weight`` are checked by delegating to it, so the two entry
            points cannot disagree about what a valid batch is -- plus a mixture count that
            does not match the group count, and any mixture rejected by
            :func:`_normalised_mixture`. The SFT extreme is built only when some group has
            SFT mass, so a mixture that never asks for it cannot fail on it; that is not
            merely an optimisation, it is what keeps this function from raising on inputs
            :func:`apply_decisions` accepts.
    """
    if len(mixtures) != len(group_sizes):
        raise ValueError(
            f"{len(mixtures)} mixtures for {len(group_sizes)} groups; one mixture per group"
        )
    weights = [_normalised_mixture(mix, i) for i, mix in enumerate(mixtures)]

    # The RL extreme, produced by the hard seam. This is what makes the pure cases exact,
    # and it also routes every shape/dtype/partition/sft_weight guard through the single
    # place those are written.
    n_decisions = len(group_sizes)
    base, _ = apply_decisions(
        advantages, loss_mask, group_sizes,
        [TrainingMode.RL] * n_decisions, sft_weight=sft_weight,
    )
    # The SFT extreme is built LAZILY, on the first group that actually carries SFT mass.
    # Building it eagerly made this function RAISE where apply_decisions succeeds: an all-SFT
    # write of sft_weight > 65504 into a float16 batch overflows inside ``full_like``, so a
    # pure-RL mixture died on a tensor it would never have read.
    sft_only = None

    # `base` is apply_decisions' own freshly cloned, RL-routed copy: equal to `advantages`
    # value for value and reachable by nobody else, so writing into it is safe and saves a
    # redundant clone. The caller's tensor is never touched.
    out = base
    mask = loss_mask.to(out.dtype)
    mass = {m: 0.0 for m in _APPLIED}
    mixed = 0
    changed = 0
    start = 0
    for g, w in zip(group_sizes, weights):
        rows = slice(start, start + g)
        start = start + g
        for mode, weight in w.items():
            mass[mode] = mass[mode] + weight
        # A normalised one-hot weight is EXACTLY 1.0, so this is a test for "some mode took
        # everything" and not a tolerance.
        if max(w.values()) < 1.0:
            mixed = mixed + 1
        a = w.get(TrainingMode.RL, 0.0)
        b = w.get(TrainingMode.SFT, 0.0)
        if a == 1.0:
            # Pure RL: not written at all, exactly as apply_decisions does not write an RL
            # group. This is load-bearing for the STATISTIC, not for the tensor. Writing
            # `1.0 * block` back would be bit-preserving, including for -0.0 -- an earlier
            # version of this comment claimed otherwise and was simply wrong, and the claim
            # is corrected here rather than deleted because it was quoted in a commit
            # message. What the skip buys is `changed_rows`: that is counted by comparing a
            # group's block before and after its write, `NaN != NaN` is True, so a
            # written-but-unchanged group holding a NaN advantage would count itself as
            # reached while the argmax path counts it as zero.
            continue
        block = out[rows]
        m = mask[rows]
        # A zero-weighted term is ABSENT, not `0.0 * x`, and the reason differs by term.
        # For RL it is load-bearing twice: `advantages` is caller data, so `0.0 * x` is NaN
        # where x is non-finite and `+0.0` where x is `-0.0`. For SFT only the second reason
        # applies -- `sft_only` inside the mask is `sft_weight * loss_mask` with sft_weight
        # already validated finite, so it can never introduce a NaN -- but `-0.0` is enough:
        # adding `+0.0` to a `-0.0` RL term flips its sign bit.
        terms = []
        if a != 0.0:
            terms.append(a * block)
        if b != 0.0:
            if sft_only is None:
                sft_only, _ = apply_decisions(
                    advantages, loss_mask, group_sizes,
                    [TrainingMode.SFT] * n_decisions, sft_weight=sft_weight,
                )
            # No `b == 1.0` special case: `1.0 * x` is bit-preserving for every value a
            # float can hold, so a shortcut here would be code no test could distinguish.
            terms.append(b * sft_only[rows])
        if not terms:
            blended = torch.zeros_like(block)          # pure SKIP: c * 0
        elif len(terms) == 1:
            blended = terms[0]
        else:
            blended = terms[0] + terms[1]
        # Same masking rule as apply_decisions, for the same reason: the mask bounds the
        # WRITE, not merely its non-zero part.
        new_block = torch.where(m != 0, blended, block)
        # Counted per group and BEFORE the write. `block` is a VIEW into `out`, so after the
        # assignment it holds the new values and the comparison is vacuous. Counting instead
        # by diffing the whole output against the input at the end is not equivalent either:
        # it diverges from apply_decisions on a NaN batch, where an untouched row compares
        # unequal to itself.
        changed = changed + int((block != new_block).any(dim=-1).sum())
        out[rows] = new_block
    return out, ApplyStats(
        counts=mass,
        changed_rows=changed,
        n_groups=len(group_sizes),
        n_rows=int(advantages.shape[0]),
        mixed_groups=mixed,
    )
