"""The seam for USING all three sources rather than declaring a winner.

The three have complementary failure modes, so the question a system should ask is not "which
source is best" but "which source is right for this need". Retrieval has reliable keys and a
fixed pool with unknown pretraining contamination; generation is unlimited and targetable but
writes some of its own keys wrong; a teacher may be better than either but costs money and
inherits its own errors, and one from the solver's own family colludes with it.

Profiles are DATA, filled from the measured table, not judgements written into code. A profile
field that has not been measured is None, and routing refuses to rank on a None rather than
treating it as zero -- ranking on an unmeasured field is how a winner gets declared by accident.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SourceProfile:
    """What a source is measured to be good and bad at.

    Attributes:
        supply: "bounded" for a fixed corpus, "unbounded" for a generator.
        key_refuted_rate: Measured share of keys refuted among DECIDED cases, or None.
        key_coverage: Share of keys the verifier could decide at all, or None. A source whose
            keys are mostly unparseable is not a source whose keys are mostly right, so the
            two travel together.
        targetable: Whether difficulty can be aimed at a target by conditioning.
        tokens_per_accepted: Measured cost to PRODUCE one accepted task, 0 for a corpus
            already on disk. Production and verification are separate purchases and must not
            be added: production is a property of the source, whereas verification cost is
            driven by how long the resulting problems are, so a corpus that costs nothing to
            read can still be the most expensive source to trust.
        verify_tokens_per_accepted: Measured cost to VERIFY one accepted task, or None.
        usd_per_accepted: Measured or estimated money cost.
        collusion: Whether the writer shares a family with the solver being trained.
        licence_constrained: Whether redistribution is limited by the item's licence.
    """

    name: str
    supply: str
    targetable: bool
    collusion: bool
    licence_constrained: bool
    key_refuted_rate: float | None = None
    key_coverage: float | None = None
    tokens_per_accepted: int = 0
    verify_tokens_per_accepted: int | None = None
    usd_per_accepted: float | None = 0.0
    notes: str = ""


#: What each need ranks on, and which direction is better. Every key names a profile field, so
#: a need cannot be satisfied by a property nobody measured.
NEEDS = {
    "reliable_keys": ("key_refuted_rate", "lower"),
    "decidable_keys": ("key_coverage", "higher"),
    "cheap_tokens": ("tokens_per_accepted", "lower"),
    "cheap_verification": ("verify_tokens_per_accepted", "lower"),
    "cheap_money": ("usd_per_accepted", "lower"),
}


def route(profiles, need: str, exclude_collusion: bool = False) -> list[tuple[str, str]]:
    """Rank sources for one need, refusing to rank on unmeasured fields.

    Args:
        profiles: The :class:`SourceProfile` list.
        need: A key of :data:`NEEDS`, or ``"unbounded_supply"`` / ``"targetable"`` which are
            categorical rather than ranked.
        exclude_collusion: Drop sources whose writer shares a family with the solver.

    Returns:
        ``[(source_name, reason)]``, best first. Sources whose deciding field is unmeasured are
        returned last with the reason saying so, never silently ranked as zero.

    Raises:
        ValueError: for an unknown need.
    """
    cand = [p for p in profiles if not (exclude_collusion and p.collusion)]
    if need == "unbounded_supply":
        yes = [(p.name, "supply is %s" % p.supply) for p in cand if p.supply == "unbounded"]
        no = [(p.name, "supply is %s" % p.supply) for p in cand if p.supply != "unbounded"]
        return yes + no
    if need == "targetable":
        return ([(p.name, "difficulty can be conditioned") for p in cand if p.targetable]
                + [(p.name, "fixed pool, cannot be aimed") for p in cand if not p.targetable])
    if need not in NEEDS:
        raise ValueError("unknown need %r; expected one of %s plus unbounded_supply, "
                         "targetable" % (need, sorted(NEEDS)))
    field_name, direction = NEEDS[need]
    measured = [p for p in cand if getattr(p, field_name) is not None]
    unmeasured = [p for p in cand if getattr(p, field_name) is None]
    measured.sort(key=lambda p: getattr(p, field_name), reverse=(direction == "higher"))
    vals = [getattr(p, field_name) for p in measured]
    tied = len(set(vals)) == 1 and len(vals) > 1
    out = [(p.name, "%s=%s%s" % (field_name, getattr(p, field_name),
                                 "  (TIED across all measured sources; this order is "
                                 "arbitrary, not a finding)" if tied else ""))
           for p in measured]
    out += [(p.name, "%s NOT MEASURED; not ranked" % field_name) for p in unmeasured]
    return out


def plan(profiles) -> dict:
    """A routing plan: what each source is for, from what was measured.

    This is the deliverable form of "use all three". It states a role per source with the
    measured reason, and it says explicitly where a role is unsupported by measurement.
    """
    return {
        "bulk supply of reliable keys": route(profiles, "reliable_keys"),
        "keys a verifier can actually decide": route(profiles, "decidable_keys"),
        "unlimited and aimable at a difficulty": route(profiles, "targetable"),
        "cheapest to produce per accepted task": route(profiles, "cheap_tokens"),
        "cheapest to verify per accepted task": route(profiles, "cheap_verification"),
        "when collusion with the solver is disqualifying": route(
            profiles, "reliable_keys", exclude_collusion=True),
    }
