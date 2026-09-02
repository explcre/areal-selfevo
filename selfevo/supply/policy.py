"""Which supplier serves which group, and the control that makes the answer falsifiable.

CHOOSING A SOURCE IS A ROUTING DECISION, AND IT IS NOT LEARNED HERE. It is expressible per
group -- every policy in this module is asked about one group at a time -- but nothing in this
file fits a model, reads a reward or looks at a feature. That is a deliberate stopping point.
This project has now recorded, three separate times, a learned mechanism matching a
feature-blind control once the control was finally run: the contextual router against
``MatchedPermutationControl``, MEDS clusters against a size-matched random partition on
gradient separation, and MEDS clusters against the same control on solve-rate variance
(p between 0.45 and 0.67, point estimate favouring the control in both). A learned source
router wired before the fixed one has been measured against its control would be the fourth.

THE CONTROL, AND WHY IT IS SHAPED THE WAY IT IS. The question a source router would have to
answer is not "does supplying rows help" -- that is the gold arm's question -- but "does
choosing WHICH source per group carry information beyond the MIXTURE of sources it produces".
:class:`MatchedSourceControl` answers it the way ``selfevo/routing/proportions.py`` answers the
mode version: replay the treatment's own realised assignment, shuffled across groups.
Proportions then match exactly rather than in expectation, for any batch, with no probability
to mis-specify.

Three details are load-bearing, and each is asserted rather than assumed in
``selfevo/tests/test_supply_sources.py``:

* THE CONTROL IS FORCED AND SO IS THE TREATMENT IT IS COMPARED TO. :class:`FixedSourcePolicy`
  offers an ordered CHAIN and falls back, so it gets several attempts per group; a control
  granted one attempt would lose for a reason that has nothing to do with targeting. The fair
  comparison replays the treatment's realised source vector through
  :class:`ForcedSourcePolicy` -- one attempt per group -- and the control is a permutation of
  that same vector through the same class. Same mechanism, one difference.
* GROUPS THE TREATMENT COULD NOT SERVE KEEP THEIR SLOT, as :data:`NO_SOURCE`. Dropping them
  would match the arms on served groups only, which is the quantity under test.
* THE CONTROL CANNOT SEE FEATURES, structurally: its constructor takes a vector of names and a
  seed and nothing else. Blindness that depends on a caller not passing something is not
  blindness.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Protocol, Sequence, runtime_checkable

from selfevo.supply.base import NO_SOURCE, SUPPLY_SOURCES, SupplyConfigError

__all__ = [
    "SourcePolicy",
    "FixedSourcePolicy",
    "ForcedSourcePolicy",
    "MatchedSourceControl",
    "source_proportions",
]


def source_proportions(names: Sequence[str]) -> dict[str, float]:
    """Realised proportions of a source assignment.

    ``selfevo.routing.proportions.measure_proportions`` answers the same question one axis
    over, but it is typed on ``Router`` and ``RoutingContext`` and consumes a stream of
    contexts, neither of which exists at this seam; this is the same idea over a vector of
    names that has already been realised.

    Args:
        names: One source name per group, :data:`NO_SOURCE` for a group nothing served.

    Returns:
        ``{name: proportion}`` summing to 1, including a :data:`NO_SOURCE` entry when there is
        one. Empty dict for an empty vector -- which callers must not read as uniform.
    """
    counts = Counter(names)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {name: count / total for name, count in counts.items()}


@runtime_checkable
class SourcePolicy(Protocol):
    """Assigns an ordered chain of suppliers to a group.

    ``chain_for`` is given both indices because the two answer different questions: ``group``
    is where the group sits in the batch, and ``qualifying_index`` is its position among the
    groups the rule SELECTED, which is the sequence a realised assignment and its permutation
    are indexed by. A policy that used the batch index to replay a per-qualifying-group vector
    would silently misalign the moment any group failed to qualify.
    """

    name: str

    def chain_for(self, group: int, qualifying_index: int) -> tuple[str, ...]:
        """Suppliers to try for this group, in order. Empty means "serve this group nothing"."""

    def sources(self) -> frozenset[str]:
        """Every supplier name this policy could ever name. Used to validate the batch once."""


def _validate(chain: Sequence[str], what: str) -> tuple[str, ...]:
    """Check names against the closed source registry.

    Args:
        chain: Names to validate. :data:`NO_SOURCE` is allowed and means "no supplier".
        what: What is being validated, for the message.

    Returns:
        The names as a tuple.

    Raises:
        SupplyConfigError: On an unknown name, or a repeated one within a chain. A typo that
            survives construction dies after model load, on GPU, at the first batch --
            the failure ``GroupRoutingConfig.__post_init__`` validates ``harness_variants``
            to avoid. A repeat is refused because a chain that tries the same supplier twice
            gets the same refusal twice and reports two losses of reach for one cause.
    """
    out = tuple(str(c) for c in chain)
    for name in out:
        if name != NO_SOURCE and name not in SUPPLY_SOURCES:
            raise SupplyConfigError(
                f"{what} names {name!r}, which is not a supply source; expected one of "
                f"{list(SUPPLY_SOURCES)}"
            )
    real = [n for n in out if n != NO_SOURCE]
    if len(set(real)) != len(real):
        raise SupplyConfigError(
            f"{what} repeats a supplier ({out}); the second attempt gets the same refusal and "
            "reports two losses of reach for one cause"
        )
    return out


class FixedSourcePolicy:
    """The shipped, explicit policy: one ordered chain, tried in order, for every group.

    The order is cheapest-and-most-trusted first. ``gold`` is free -- it is already in the
    batch -- and is correct by construction. ``self`` is the model's own verified output, so it
    is on-distribution and needs no external resource. ``corpus`` is correct but off
    distribution. ``teacher`` is last because it is the only source whose correctness depends
    on a verifier that can itself be wrong, and, in this repo, the only one that would need a
    served model.

    The chain is the same for every group, which makes this policy feature-blind in exactly the
    way the brief requires at this stage: the per-group expressiveness is in the interface, not
    in a criterion nobody has measured yet. What varies per group is which link in the chain
    actually serves, and THAT vector is what :class:`MatchedSourceControl` permutes.

    Args:
        chain: Supplier names in the order to try them.

    Raises:
        SupplyConfigError: On an unknown or repeated name, or an empty chain -- a policy that
            names no supplier is an off arm wearing an on arm's label.
    """

    name = "fixed"
    DEFAULT_CHAIN = ("gold", "self", "corpus", "teacher")

    def __init__(self, chain: Sequence[str] = DEFAULT_CHAIN) -> None:
        validated = _validate(chain, "FixedSourcePolicy chain")
        if not [c for c in validated if c != NO_SOURCE]:
            raise SupplyConfigError(
                "FixedSourcePolicy needs at least one supplier; a policy that names none is "
                "an off arm reporting as an on arm"
            )
        self.chain = validated

    def chain_for(self, group: int, qualifying_index: int) -> tuple[str, ...]:
        """The chain, which does not depend on the group.

        Args:
            group: Batch group index. Unread.
            qualifying_index: Position among qualifying groups. Unread.

        Returns:
            The configured chain.
        """
        return self.chain

    def sources(self) -> frozenset[str]:
        """Every supplier this policy can name."""
        return frozenset(c for c in self.chain if c != NO_SOURCE)


class ForcedSourcePolicy:
    """Exactly one supplier per qualifying group, in order, with no fallback.

    The shape both halves of the control comparison take: the treatment replayed at one attempt
    per group, and the permutation of it. Indexed by ``qualifying_index``, because the vector it
    replays has one entry per group the rule selected.

    Args:
        assignment: One supplier name per qualifying group. :data:`NO_SOURCE` for a group to be
            served by nothing, which is how the groups the treatment could not serve keep their
            slot in the multiset.

    Raises:
        SupplyConfigError: On an unknown name or an empty vector. Empty is refused for
            ``MatchedPermutationControl``'s reason: a control with nothing to replay is
            silently a no-op arm.
    """

    name = "forced"

    def __init__(self, assignment: Sequence[str]) -> None:
        if len(assignment) == 0:
            raise SupplyConfigError(
                "ForcedSourcePolicy needs a non-empty assignment; a control with nothing to "
                "replay is silently a no-op arm"
            )
        self.assignment = tuple(
            _validate([a], "ForcedSourcePolicy assignment")[0] for a in assignment
        )

    def chain_for(self, group: int, qualifying_index: int) -> tuple[str, ...]:
        """The one supplier assigned to this qualifying group.

        Args:
            group: Batch group index. Unread: the assignment is indexed by qualifying position.
            qualifying_index: Position among qualifying groups.

        Returns:
            A one-element chain, or an empty chain for :data:`NO_SOURCE`.

        Raises:
            SupplyConfigError: If asked about more qualifying groups than the assignment covers.
                Wrapping would silently reuse decisions and quietly change the realised
                proportions the whole construction exists to match.
        """
        if qualifying_index >= len(self.assignment):
            raise SupplyConfigError(
                f"{self.name} policy holds {len(self.assignment)} decisions and was asked for "
                f"index {qualifying_index}; refusing to wrap, which would reuse decisions and "
                "change the realised proportions this construction exists to match"
            )
        chosen = self.assignment[qualifying_index]
        return () if chosen == NO_SOURCE else (chosen,)

    def sources(self) -> frozenset[str]:
        """Every supplier named anywhere in the assignment."""
        return frozenset(a for a in self.assignment if a != NO_SOURCE)


class MatchedSourceControl(ForcedSourcePolicy):
    """The mandatory control: the treatment's own source multiset, assigned feature-blind.

    A :class:`ForcedSourcePolicy` whose assignment is a permutation of the treatment's. Same
    class, same mechanism, same number of attempts per group; the only difference is which
    group gets which source. If a fixed order does not beat this, its effect came from the
    mixture of sources and not from choosing between them -- which is exactly what
    ``MatchedPermutationControl`` says one axis over, and exactly what the two MEDS nulls of
    2026-09-02 found when the control was finally run.

    The constructor admits no features, no batch and no rewards. Blindness that depends on a
    caller not passing something is not blindness.

    Args:
        realised: The treatment's realised source per qualifying group, :data:`NO_SOURCE` where
            nothing served it.
        seed: Shuffle seed. A private ``random.Random``, so the control neither perturbs nor is
            perturbed by sampling elsewhere.

    Raises:
        SupplyConfigError: On an empty vector or an unknown name.
    """

    name = "matched_random"

    def __init__(self, realised: Sequence[str], seed: int = 0) -> None:
        shuffled = list(realised)
        random.Random(seed).shuffle(shuffled)
        super().__init__(shuffled)
        self.realised = tuple(realised)
        self.seed = int(seed)

    def moved_fraction(self) -> float:
        """Fraction of groups whose assigned source differs from the treatment's.

        0.0 means the permutation reproduced the treatment exactly, which is possible by chance
        on a short or nearly-uniform vector and makes that seed's control uninformative rather
        than wrong. Reported so a vacuous draw is visible instead of being read as a null.
        """
        if not self.realised:
            return 0.0
        moved = sum(1 for a, b in zip(self.assignment, self.realised) if a != b)
        return moved / len(self.realised)
