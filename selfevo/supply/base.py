"""The supplier interface behind the batch-construction seam.

WHY A SUPPLIER AXIS AT ALL, AND WHY IT IS THE ONLY BRANCH LEFT. ``selfevo/gold/__init__.py``
states the argument in full and it is not repeated here: an advantage is a coefficient on
tokens the model actually EMITTED, so a group in which every rollout is wrong has nothing
correct to reinforce, ``selfevo/tests/test_gold_target_reachability.py`` pins that no router
and no fixed rule can reach that branch through the advantage, and the only altitude that
works is BATCH CONSTRUCTION. What this module adds is that the batch-construction seam does
not care WHERE the correct row came from. ``selfevo/gold/substitute.py`` implements it for
exactly one supplier, the dataset's own gold solution; every other correct row -- an earlier
step's own solved rollout, an offline corpus row, a stronger model's verified answer -- enters
the update through the same splice, carries the same off-policy treatment and is counted by
the same counters.

This axis survived the two nulls of 2026-09-02 (MEDS separates neither gradients nor solve
rates, against size-matched controls, in both cases with the point estimate favouring the
control) precisely because it needs no clusters. It acts on the branch that has no target at
all, whether or not that branch can be subdivided.

THREE RULES, EACH OF WHICH HAS COST THIS PROJECT A DEFECT BEFORE.

1. A supplier that has nothing to offer REFUSES, with a typed reason, and the reason is
   counted. It never fabricates, never substitutes a row belonging to a different prompt, and
   never returns the row unchanged while reporting success. The last of those is the silent
   no-op this whole path distrusts most: an arm that is configured, logs as a grounded arm,
   and applies nothing. Every zero in this repo has an open artifact behind it, so
   :class:`Refusal` is a mapping of reasons and not a count.
2. Prompt identity comes from the PROMPT, never from ``unit_id``, which is batch-local by
   construction. ``selfevo/routing/prompt_credit.py::prompt_key`` already solved this by
   hashing the tokens before the first response token, and :meth:`SupplyRequest.identity` and
   :func:`key_for_prompt` both call it rather than growing a second scheme. Two schemes that
   agree today diverge silently on the first tokenizer change.
3. A row the model did not emit has no valid importance ratio under the current policy. That
   is handled once, in ``selfevo/gold/substitute.py`` (finite sentinel, then
   ``reconcile_gold_logprobs`` against the trainer's own recomputed ``prox_logp``), and this
   module does not touch ``logprobs`` at all. A second treatment of off-policy weighting is
   how two arms end up differing in a quantity neither reports.

A supplier is deliberately a small, synchronous, pure object: given one row of one batch it
either returns token ids or raises. It holds no tensors of its own beyond its store, imports
neither the trainer nor an engine, and is driven entirely on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch

from selfevo.routing.prompt_credit import prompt_key

__all__ = [
    "SUPPLY_SOURCES",
    "NO_SOURCE",
    "Refusal",
    "SupplyError",
    "SupplyConfigError",
    "SupplierRefused",
    "SupplyOffer",
    "SupplyRequest",
    "Supplier",
    "key_for_prompt",
    "source_code",
]

# The closed set of supplier names. Closed, and checked at construction, for the reason
# `partition_from_config` records: a name that falls through to a default silently runs one
# mechanism under another arm's label, and the artifact afterwards cannot say which ran.
SUPPLY_SOURCES = ("gold", "self", "corpus", "teacher")

# What a policy assigns to a group it wants no supplier to touch, and what a served-source
# vector holds for a group nothing could serve. Empty rather than None so the realised
# assignment is a vector of strings that a permutation control can shuffle without special
# cases.
NO_SOURCE = ""


class Refusal(Enum):
    """Why a supplier declined to supply a row.

    One member per distinct cause, because the counts are the artifact. A single "skipped"
    total sends the reader to the wrong fix: a missing dataset solution, a prompt that has
    never been solved, and a derivation too long for the row have three different remedies,
    and ``GoldStats`` already learned this once by splitting ``groups_no_gold`` from
    ``groups_no_fit``.
    """

    NO_GOLD = "no_gold"
    NO_FIT = "no_fit"
    NO_MATCH = "no_match"
    NO_IDENTITY = "no_identity"
    NOT_VERIFIED = "not_verified"
    UNAVAILABLE = "unavailable"
    EMPTY = "empty"
    NO_SOURCE = "no_source"


class SupplyError(Exception):
    """Base for every refusal on the supplier path."""


class SupplyConfigError(SupplyError):
    """A supplier or policy was built in a state that cannot be honest.

    Raised at CONSTRUCTION, never at supply time, for the things a run must not discover at
    step 3 of 900: an unknown source name, a control with nothing to replay, a teacher with no
    verifier. ``GroupRoutingConfig.__post_init__`` validates ``harness_variants`` against its
    registry for the same reason -- a typo that survives config parse dies after model load,
    on GPU, at the first batch.
    """


class SupplierRefused(SupplyError):
    """This supplier has no correct row for this request.

    An ordinary, expected outcome and not a failure of the run: most groups do not qualify,
    and of those that do, some have no supply. It is an exception rather than a ``None``
    return so that a caller cannot forget to look, and it carries the reason rather than a
    message so the counter is keyed on a value and not on a string.

    Args:
        reason: Which :class:`Refusal`.
        source: The supplier that refused, for the message and for per-source reporting.
        detail: Free text for the log. Never parsed.
    """

    def __init__(self, reason: Refusal, source: str, detail: str = "") -> None:
        super().__init__(f"{source} refused ({reason.value}): {detail}")
        self.reason = reason
        self.source = source
        self.detail = detail


def key_for_prompt(prompt_ids: Sequence[int]) -> str:
    """Prompt identity for a bare prompt, in the prompt-credit ledger's own scheme.

    A store built offline holds prompts, not rollout rows, so it has no ``loss_mask`` to hand
    to :func:`selfevo.routing.prompt_credit.prompt_key`. Rather than hash the tokens here --
    which would be a second identity scheme, agreeing with the first exactly until one of them
    changes -- this appends one sentinel response position so the ledger's own function sees a
    well-formed row whose prompt region is precisely ``prompt_ids``. The digest is therefore
    equal, by construction and not by coincidence, to the one a rollout of this prompt gets
    from :meth:`SupplyRequest.identity`, and ``test_supply_sources.py`` asserts that equality
    rather than assuming it.

    Args:
        prompt_ids: The prompt's token ids.

    Returns:
        The same 16-hex-character digest the ledger uses.

    Raises:
        ValueError: On an empty prompt, which has no identity to key on.
    """
    ids = [int(t) for t in prompt_ids]
    if not ids:
        raise ValueError(
            "an empty prompt has no identity; keying on it would merge unrelated rows"
        )
    return prompt_key(ids + [0], [0] * len(ids) + [1])


def source_code(name: str) -> int:
    """Integer code for a source name, for the per-token ``source_ids`` tensor.

    Args:
        name: A member of :data:`SUPPLY_SOURCES`, or :data:`NO_SOURCE`.

    Returns:
        0 for :data:`NO_SOURCE`, else the 1-based index in :data:`SUPPLY_SOURCES`. 1-based so
        that 0 unambiguously means "this row was not substituted", which is what every row of
        an off arm holds.

    Raises:
        SupplyConfigError: For a name outside the closed set.
    """
    if name == NO_SOURCE:
        return 0
    if name not in SUPPLY_SOURCES:
        raise SupplyConfigError(
            f"unknown supply source {name!r}; expected one of {list(SUPPLY_SOURCES)}"
        )
    return SUPPLY_SOURCES.index(name) + 1


# eq=False: the batch field holds tensors, and a generated __eq__ would compare them
# elementwise and raise on the bool() of the result. Requests are passed, never compared.
@dataclass(frozen=True, eq=False)
class SupplyRequest:
    """One row of one batch, and everything a supplier may look at.

    Deliberately a view over the batch rather than a copy: a supplier reads, it never writes,
    and the single writer is ``selfevo/gold/substitute.py::_write_gold_row``. Passing the
    batch itself also means a supplier that needs a key the gold path does not use -- a future
    difficulty feature, say -- needs no change here.

    Args:
        batch: The ROLLOUT batch, before ``compute_logp``.
        row: Index of the row that will be overwritten if this request is served.
        group: The row's group index within the batch.
        prompt_len: Index of the first RESPONSE token in ``row``, in TOKEN coordinates, as
            ``selfevo.gold.attach.prompt_lengths`` computes it. Passed in rather than
            recomputed so that the boundary the supplier reasons about and the boundary the
            writer splices at cannot drift.
        width: The batch's padded width, i.e. ``input_ids.shape[1]``.
    """

    batch: Mapping[str, Any]
    row: int
    group: int
    prompt_len: int
    width: int

    @property
    def capacity(self) -> int:
        """How many payload tokens fit after the prompt.

        A supplier must refuse with :attr:`Refusal.NO_FIT` rather than truncate: a cut-off
        derivation is a wrong target that still looks like a target, which is the reasoning
        ``attach_gold`` already records for the dataset gold.
        """
        return self.width - self.prompt_len

    def prompt_ids(self) -> list[int]:
        """The row's prompt token ids, as a list.

        Returns:
            ``input_ids[row, :prompt_len]``.
        """
        ids = self.batch["input_ids"]
        return [int(v) for v in ids[self.row, : self.prompt_len].tolist()]

    def identity(self) -> str:
        """Stable identity of this row's prompt.

        Delegates to ``selfevo.routing.prompt_credit.prompt_key``, which hashes the tokens
        before the first response token. ``unit_id`` is NOT usable here: it is batch-local by
        construction, so a store keyed on it would never find a prompt again on a later step,
        which is the whole point of a self-generated supplier.

        Returns:
            The ledger's 16-hex-character digest.

        Raises:
            SupplierRefused: With :attr:`Refusal.NO_IDENTITY` for a row with no prompt region.
                A row that is all response carries no identity, and silently returning a
                constant would merge every such row into one prompt.
        """
        ids = [int(v) for v in self.batch["input_ids"][self.row].tolist()]
        mask = [float(v) for v in self.batch["loss_mask"][self.row].tolist()]
        try:
            return prompt_key(ids, mask)
        except ValueError as exc:
            raise SupplierRefused(Refusal.NO_IDENTITY, "request", str(exc)) from exc


@dataclass(frozen=True)
class SupplyOffer:
    """A correct response, ready to be spliced after the row's prompt.

    Args:
        token_ids: 1-D integer tensor. The RESPONSE only: the writer keeps the row's own
            prompt, because a target detached from its question teaches the derivation
            unconditionally.
        source: The supplier's name, which must be in :data:`SUPPLY_SOURCES`.
        detail: Free text for the log, e.g. which step the rollout came from.

    Raises:
        SupplyConfigError: If the payload is not a 1-D non-empty integer tensor, or the source
            is not a known name. Validated here, on the offer, so a broken supplier is caught
            at the seam it crosses rather than as a shape error inside the writer -- one stage
            later, wearing a message that names neither the supplier nor the batch.
    """

    token_ids: torch.Tensor
    source: str
    detail: str = ""

    def __post_init__(self) -> None:
        """Refuse a malformed payload at the seam rather than inside the writer."""
        t = self.token_ids
        if not torch.is_tensor(t) or t.ndim != 1:
            raise SupplyConfigError(
                f"{self.source} offered {type(t).__name__} with shape "
                f"{getattr(t, 'shape', None)}; a payload must be a 1-D tensor"
            )
        if t.numel() == 0:
            raise SupplyConfigError(
                f"{self.source} offered an empty payload; refuse with Refusal.EMPTY instead, "
                "so the loss of reach is counted rather than written as a zero-token row"
            )
        if t.dtype.is_floating_point or t.dtype == torch.bool:
            raise SupplyConfigError(
                f"{self.source} offered a {t.dtype} payload; token ids are integers"
            )
        source_code(self.source)

    @property
    def n_tokens(self) -> int:
        """Payload length in tokens, which is this offer's share of the loss denominator."""
        return int(self.token_ids.shape[0])


@runtime_checkable
class Supplier(Protocol):
    """What the batch-construction seam requires of a source of correct rows.

    Four members, and each exists because something downstream would otherwise fail silently:

    * ``name`` -- keys the per-source counters and the ``source_ids`` tensor.
    * ``required_keys`` -- batch keys this supplier cannot work without, checked ONCE per
      batch before any group is processed, so "the arm is configured at one end and not the
      other" is one refusal rather than one per group.
    * ``has_supply`` -- whether this supplier has anything at all for this batch. The reach
      guard: a supplier with an empty store must refuse the batch loudly, not decline every
      group quietly.
    * ``supply`` -- the row, or a typed refusal.
    """

    name: str
    required_keys: tuple[str, ...]

    def has_supply(self, batch: Mapping[str, Any]) -> bool:
        """Whether anything in this batch could be served by this supplier at all."""

    def supply(self, request: SupplyRequest) -> SupplyOffer:
        """A correct response for ``request``, or a typed refusal.

        Raises:
            SupplierRefused: Always, when there is nothing correct to offer.
        """


def missing_key_hint(supplier: Any, key: str) -> str:
    """The explanation attached to a missing-required-key refusal.

    Kept as a function taking the supplier rather than a required Protocol member so that a
    minimal supplier -- a test double, say -- needs only ``name``, ``required_keys``,
    ``has_supply`` and ``supply``.

    Args:
        supplier: The supplier whose key is missing.
        key: The absent batch key.

    Returns:
        The supplier's own hint if it defines ``missing_key_hint``, else a generic one.
    """
    hint = getattr(supplier, "missing_key_hint", None)
    if callable(hint):
        return str(hint(key))
    return (
        f"The {getattr(supplier, 'name', '?')} supplier declares {key} as required and the "
        "batch does not carry it, so the arm is configured at one end and not the other."
    )


def no_supply_message(supplier: Any, rule_value: str) -> str:
    """The refusal raised when a supplier has nothing anywhere in the batch.

    Args:
        supplier: The supplier with no supply.
        rule_value: The active rule's name, for the message.

    Returns:
        The supplier's own message if it defines ``no_supply_message``, else a generic one.
    """
    msg = getattr(supplier, "no_supply_message", None)
    if callable(msg):
        return str(msg(rule_value))
    return (
        f"rule={rule_value} but the {getattr(supplier, 'name', '?')} supplier has nothing for "
        "this batch. An arm that trains on nothing is the silent no-op this guard prevents."
    )
