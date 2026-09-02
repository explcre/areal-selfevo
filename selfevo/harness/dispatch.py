"""A consumer for the harness half of a routing decision, so the axis stops being a name.

``RoutingDecision`` has carried two coordinates since the axis was designed -- a mode weight
mapping and a :class:`~selfevo.routing.base.HarnessAction` -- but only the first had a
consumer. Nothing set ``RoutingContext.can_evolve_harness``, so every router dropped its
harness action before emitting it, and ``areal.trainer.ppo.actor._refuse_dropped_harness``
exists precisely because an arm labelled "harness-evolving" that trains identically to one
that is not is a silent failure: the gap between the two runs gets attributed to something
that never happened.

This module is the missing consumer, and it is deliberately the *smallest* one that is
honest. It dispatches over CONFIGURATIONS -- :class:`~selfevo.harness.base.HarnessVariant`
objects -- not over agent executions. That choice is not a shortcut, it is what makes the
axis testable at all: :mod:`selfevo.harness.mini_swe` needs Docker images, a SWE-bench
download and a served model before it can produce a single rollout, so a dispatcher that
depended on it could not be exercised on CPU, in CI, or on a box whose GPUs are busy. The
adapter stays a pluggable, optional attribute (:attr:`HarnessDispatcher.adapter`) for the
day a rollout is actually wanted; nothing on the default path touches it.

**What "evolving the harness" is split into here**, following the correction in ``GOAL.md``:

* the VARIANT SET -- what scaffolds exist. Coarse, configured, changes between runs.
* the DISPATCH RULE -- which variant is active. Fine, changes within a run, and is the
  thing a :class:`HarnessAction` moves.

A set with one member has a dispatch rule with one possible answer, which is not a decision.
:attr:`HarnessDispatcher.can_evolve` is therefore False below two members and that is what
production reads to set ``can_evolve_harness``, so a mislabelled single-variant arm reports
itself instead of quietly training like the control.

**Aggregation, and why a batch moves the harness at most once.** A harness is a SHARED
artefact: it applies to every future rollout, not to the unit that proposed it. Routers
emit ``HarnessAction`` per GROUP, so a batch of 32 groups can carry a dozen PROPOSEs. Walking
them one at a time through :meth:`HarnessDispatcher.apply` would rotate the active variant a
dozen times per step, and with an even number of proposals over a two-variant set it would
land back where it started -- the arm would switch constantly and yet be indistinguishable
from one that never switched at any step boundary, with the log showing plenty of activity.
:meth:`HarnessDispatcher.consume` is the batch entry point for exactly that reason: it counts
every action, and acts once. ``apply`` remains the single-decision primitive underneath it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from selfevo.harness.base import VARIANTS, HarnessAdapter, HarnessRollout, HarnessVariant
from selfevo.routing.base import HarnessAction

__all__ = [
    "DispatchRecord",
    "DispatchBatch",
    "HarnessDispatcher",
    "round_robin",
    "build_dispatcher",
]


def round_robin(
    variants: Sequence[HarnessVariant], current: HarnessVariant
) -> HarnessVariant:
    """Pick the next variant in configured order, wrapping at the end.

    The default proposal rule, chosen over anything cleverer for three reasons. It is
    DETERMINISTIC, so a run is reproducible and two arms that differ only in seed do not
    differ in harness history. It is GUARANTEED to move -- with two or more members the
    successor of any element is a different element -- so a PROPOSE that reaches this rule
    always produces an observable change rather than a no-op the log would report as a
    proposal. And over a run it VISITS every member, which is what a measurement arm needs:
    a dispatch rule can only be compared against "one variant is simply better" if every
    variant has data.

    A feature-driven rule (send a unit with ``truncated_fraction == 1`` to the longer-budget
    variant) is the interesting one and is the reason ``selector`` is a parameter rather than
    hardcoded here. It is a larger claim and needs its own control, so it is not the default.

    Args:
        variants: The configured set, in configured order. Must be non-empty.
        current: The active variant; must be a member.

    Returns:
        The member after ``current``, wrapping around.

    Raises:
        ValueError: If ``variants`` is empty or ``current`` is not one of them. Either would
            mean the caller's idea of the variant set and the dispatcher's have diverged, and
            guessing which is right would silently dispatch to a scaffold nobody configured.
    """
    if not variants:
        raise ValueError("round_robin needs a non-empty variant set")
    names = [v.name for v in variants]
    if current.name not in names:
        raise ValueError(
            f"current variant {current.name!r} is not in the configured set {names}; "
            "the caller and the dispatcher disagree about what exists"
        )
    return variants[(names.index(current.name) + 1) % len(variants)]


@dataclass(frozen=True)
class DispatchRecord:
    """What one :class:`~selfevo.routing.base.HarnessAction` did to the active variant.

    The audit record for a single decision. Every field is here so that a switch can be
    reconstructed after the fact from the log alone -- which action arrived, what was active
    before, what is active now, and, when nothing moved, WHY nothing moved. A dispatcher that
    reported only "propose count" could not distinguish an arm whose proposals were acted on
    from one whose proposals were all refused for lack of a second variant.

    Args:
        action: The action consumed.
        before: Name of the variant active before, or ``None`` for an empty variant set.
        after: Name of the variant active after.
        changed: Whether the active variant actually moved. Never inferred from ``action``:
            a PROPOSE that could not move is the failure mode this whole guard exists for.
        reason: Short human-readable justification, in the same spirit as
            ``RoutingDecision.reason``.
    """

    action: HarnessAction
    before: str | None
    after: str | None
    changed: bool
    reason: str


@dataclass(frozen=True)
class DispatchBatch:
    """What one batch of harness actions did, and the metrics that say so.

    Separate from :class:`HarnessDispatcher` because the dispatcher's state is PERSISTENT --
    the active variant survives across steps, being a shared artefact -- while these counts
    are per batch. Folding the counters into the dispatcher would make every logged number
    cumulative, and a per-step panel of a monotonically rising count cannot show that the
    axis went quiet.

    Args:
        records: One record per action consumed, in the order given.
        variant_names: Every configured variant name, so the emitted key set is the same on
            every step and on every arm sharing a variant set. A key that appears only on the
            steps where something happened is a key no dashboard can plot against.
        active: Name of the variant active after the batch, or ``None`` for an empty set.
        can_evolve: Whether the dispatcher could move at all, carried so the metric is
            emitted even by an arm that never proposes.
    """

    records: tuple[DispatchRecord, ...]
    variant_names: tuple[str, ...]
    active: str | None
    can_evolve: bool

    @property
    def switches(self) -> int:
        """How many records actually moved the active variant.

        The number the whole axis rests on. ``propose`` counts what the ROUTER asked for;
        this counts what the harness DID. An arm whose proposals are all refused has a high
        propose count and zero switches, and those two runs must not look alike.
        """
        return sum(1 for r in self.records if r.changed)

    def count(self, action: HarnessAction) -> int:
        """How many records carried ``action``."""
        return sum(1 for r in self.records if r.action is action)

    def as_metrics(self) -> dict[str, float]:
        """Metrics for ``stats_tracker.scalar``, under the actor's ``route/`` namespace.

        Emits the FULL key set every time -- all three action counts, the switch count, and
        one indicator per configured variant -- because two arms that emit different keys
        cannot be put on one panel, and this repo has already shipped a routed run with an
        empty ``route/`` namespace and had to reconstruct what it did from the config.

        The active variant is named in the KEY (``route/harness_active_<name>``) rather than
        encoded as a number in a value, because ``stats_tracker.scalar`` takes floats and an
        ordinal would be meaningless the moment the configured order changed.

        Returns:
            ``{metric name: float}``.
        """
        out: dict[str, float] = {
            "route/harness_none": float(self.count(HarnessAction.NONE)),
            "route/harness_propose": float(self.count(HarnessAction.PROPOSE)),
            "route/harness_validate": float(self.count(HarnessAction.VALIDATE)),
            "route/harness_switches": float(self.switches),
            "route/harness_n_variants": float(len(self.variant_names)),
            "route/harness_can_evolve": float(self.can_evolve),
        }
        for name in self.variant_names:
            out[f"route/harness_active_{name}"] = float(name == self.active)
        return out


class HarnessDispatcher:
    """Owns a harness variant set and the selection among it; consumes ``HarnessAction``.

    The consumer that ``_refuse_dropped_harness`` says must exist before a harness action may
    be emitted. Its whole job is to make two arms that differ only in harness routing differ
    in something an experiment can measure, so every method here is written to fail loudly
    rather than to degrade into the no-op that the guard was built to catch.

    Args:
        variants: The configured set, in the order a proposal rule walks. May be empty or a
            single member -- both are legitimate CONTROL arms ("the harness axis is wired but
            has nowhere to go") and both report :attr:`can_evolve` False.
        selector: Rule that picks the next variant on PROPOSE, given ``(variants, current)``.
            Defaults to :func:`round_robin`. A seam, not a knob: a feature-driven rule is the
            interesting version and it belongs here rather than inside this class.
        adapter: Optional :class:`~selfevo.harness.base.HarnessAdapter` used only by
            :meth:`run`. Nothing on the dispatch path touches it, so this class is fully
            exercisable with no Docker, no benchmark download and no served model -- which is
            the only reason the axis can be tested at all on the boxes this runs on.

    Raises:
        ValueError: If two configured variants share a name, or if every configured variant
            has the same behaviour. Both produce a set that LOOKS evolvable and is not:
            duplicate names make two runs sharing a variant label incomparable, and a set
            whose members differ only in prose dispatches between scaffolds that behave
            identically, which is exactly an arm that trains like its own control while
            reporting switches.
    """

    def __init__(
        self,
        variants: Sequence[HarnessVariant],
        *,
        selector: Callable[[Sequence[HarnessVariant], HarnessVariant], HarnessVariant]
        | None = None,
        adapter: HarnessAdapter | None = None,
    ) -> None:
        variants = tuple(variants)
        names = [v.name for v in variants]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(
                f"duplicate variant names {dupes}: a set that repeats a member looks "
                f"evolvable and is not, and two runs sharing a variant name would be "
                f"incomparable"
            )
        # repr, not the values themselves: settings is adapter-specific and passed through
        # verbatim, so a list or nested dict in there is legal and would be unhashable.
        behaviours = {
            (v.step_limit, repr(sorted(v.settings.items(), key=lambda kv: kv[0])))
            for v in variants
        }
        if len(variants) > 1 and len(behaviours) == 1:
            raise ValueError(
                f"variants {names} all specify the same behaviour "
                f"(step_limit and settings are identical); dispatching between them would "
                f"change nothing, so a 'harness-evolving' arm would train exactly like its "
                f"own control while reporting switches"
            )
        self._variants = variants
        self._selector = selector or round_robin
        self.adapter = adapter
        self._active = variants[0] if variants else None

    @property
    def variants(self) -> tuple[HarnessVariant, ...]:
        """The configured set, in configured order."""
        return self._variants

    @property
    def selector(self):
        """The rule that picks the next variant on PROPOSE.

        Exposed because a feature-driven selector carries the counters an arm is REPORTED
        on -- how many decisions it took, how many it refused -- and a matched control is
        configured from the treatment's realised rates. Reading them off the object that
        made the decisions is the alternative to recomputing them from a log, which is how
        a control comes to be matched to a number the treatment never produced.
        """
        return self._selector

    @property
    def active(self) -> HarnessVariant | None:
        """The variant a rollout would run under now, or ``None`` for an empty set."""
        return self._active

    @property
    def can_evolve(self) -> bool:
        """Whether this dispatcher has somewhere to move, i.e. two or more variants.

        This is the value production writes into ``RoutingContext.can_evolve_harness``, and
        it is the reason that field can finally be True. One variant is not an evolvable
        harness: the dispatch rule has exactly one possible answer, so an arm configured that
        way and labelled "harness-evolving" would be its own control. Enforced here, at the
        one place the answer is computed, rather than left to each caller to remember.
        """
        return len(self._variants) >= 2

    def apply(self, action: HarnessAction) -> DispatchRecord:
        """Consume ONE harness action and report what it did.

        The single-decision primitive. PROPOSE asks for a different variant; VALIDATE keeps
        the current one deliberately (it is the regression case -- "measure what we have" --
        and a rule that quietly moved on VALIDATE would destroy the measurement it was asked
        for); NONE is the default everywhere and must be exactly inert, so that a run that
        never mentions the harness behaves as it did before this module existed.

        Args:
            action: The action to consume.

        Returns:
            A :class:`DispatchRecord`. A PROPOSE that could not move reports
            ``changed=False`` with the reason, and is NEVER reported as a switch.

        Raises:
            ValueError: If ``action`` is not a :class:`~selfevo.routing.base.HarnessAction`;
                if the selector returns something outside the configured set; or if it
                returns the variant that is already active. The last two are the ways a
                custom selector turns dispatch into a no-op or into a scaffold nobody
                configured, and both are invisible downstream -- the run completes, the
                counts look healthy, and the arm is the control.
        """
        if not isinstance(action, HarnessAction):
            raise ValueError(
                f"expected a HarnessAction, got {action!r}; accepting a bare string here "
                f"would let a typo dispatch as NONE and silence the axis"
            )
        before = self._active.name if self._active is not None else None

        if action is HarnessAction.NONE:
            return DispatchRecord(action, before, before, False, "no harness action")
        if action is HarnessAction.VALIDATE:
            return DispatchRecord(
                action, before, before, False, f"validate current variant {before!r}"
            )

        if not self.can_evolve:
            return DispatchRecord(
                action,
                before,
                before,
                False,
                f"proposal refused: {len(self._variants)} variant(s) configured, "
                "nothing to move to",
            )

        chosen = self._selector(self._variants, self._active)
        if chosen not in self._variants:
            raise ValueError(
                f"selector returned {getattr(chosen, 'name', chosen)!r}, which is not in "
                f"the configured set {[v.name for v in self._variants]}; dispatching to an "
                f"unconfigured scaffold would produce rollouts no arm declared"
            )
        if chosen.name == before:
            raise ValueError(
                f"selector returned the already-active variant {before!r} on a PROPOSE; "
                f"a proposal that cannot move is a no-op the log would report as a "
                f"proposal, making a dispatching arm indistinguishable from its control"
            )
        self._active = chosen
        return DispatchRecord(action, before, chosen.name, True, f"proposed {chosen.name!r}")

    def consume(self, actions: Iterable[HarnessAction]) -> DispatchBatch:
        """Consume a batch of harness actions, acting at most ONCE.

        The entry point production uses. Routers emit one action per GROUP, but a harness is
        a shared artefact that applies to every future rollout, so N proposals in one batch
        are N pieces of EVIDENCE for one decision, not N decisions. Applying them one by one
        would rotate the active variant once per proposal, and an even number of proposals
        over a two-variant set would land back on the starting variant every single step:
        the arm would log a dozen switches per batch and be byte-identical to an arm that
        never switched. Aggregating here is what makes the active variant a quantity a step
        boundary can be compared on.

        The first PROPOSE in the batch is the one acted on. Which one is arbitrary in the
        sense that the units are unordered evidence; it is not arbitrary in the sense of
        being random -- it is positional and therefore reproducible.

        Args:
            actions: One action per routed unit, in group order.

        Returns:
            A :class:`DispatchBatch` holding one record per action -- including the refused
            and inert ones, so the counts describe the whole batch and not just its switch.
        """
        records: list[DispatchRecord] = []
        acted = False
        for action in actions:
            if action is HarnessAction.PROPOSE and acted:
                before = self._active.name if self._active is not None else None
                records.append(
                    DispatchRecord(
                        action,
                        before,
                        before,
                        False,
                        "proposal aggregated: the harness already moved this batch",
                    )
                )
                continue
            record = self.apply(action)
            records.append(record)
            if record.changed:
                acted = True
        return DispatchBatch(
            records=tuple(records),
            variant_names=tuple(v.name for v in self._variants),
            active=self._active.name if self._active is not None else None,
            can_evolve=self.can_evolve,
        )

    def run(self, task_id: str) -> HarnessRollout:
        """Run one task under the ACTIVE variant, if an adapter was supplied.

        The optional half of this class, kept behind an explicit adapter so that the
        dispatch path above never depends on it. Every shipped adapter needs something this
        environment does not have -- :class:`~selfevo.harness.mini_swe.MiniSweAdapter` wants
        Docker images, a SWE-bench download and a served model -- and a dispatcher that could
        only be tested where those exist would be a dispatcher that never got tested.

        Args:
            task_id: Instance identifier for the adapter's benchmark.

        Returns:
            The adapter's :class:`~selfevo.harness.base.HarnessRollout`.

        Raises:
            RuntimeError: If no adapter or no variant is configured. Returning a fabricated
                unsolved rollout instead would score an infrastructure failure as reward 0,
                which is a mistake this project has already made and retracted.
        """
        if self.adapter is None:
            raise RuntimeError(
                "no HarnessAdapter configured; this dispatcher selects among variant "
                "CONFIGURATIONS and cannot execute one. Pass adapter=... to run rollouts."
            )
        if self._active is None:
            raise RuntimeError("no variants configured; there is nothing to run under")
        return self.adapter.run(task_id, self._active)


def build_dispatcher(
    names: Sequence[str] | None,
    selector: str | None = None,
    selector_args: dict | None = None,
) -> HarnessDispatcher | None:
    """Resolve configured variant names into a dispatcher, or ``None`` for no harness arm.

    The single production entry point, so that "is there a harness consumer?" has one answer
    and the actor does not have to reimplement the registry lookup. ``None`` in gives
    ``None`` out, which is what every run before this module existed had, and it keeps
    ``_refuse_dropped_harness`` absolute for those runs: no dispatcher means
    ``harness_consumer=False`` and any emitted action still raises.

    Args:
        names: Variant names from config, resolved against
            :data:`selfevo.harness.base.VARIANTS`. ``None`` or empty means no harness arm.
        selector: Name of a rule in :data:`selfevo.harness.selectors.SELECTORS`, or ``None``
            for :func:`round_robin`. Resolved HERE rather than by the caller so that "which
            selectors exist?" has one answer, the same way variant names do; a trainer that
            did its own lookup would be a second place for an unknown name to be dropped.
        selector_args: Keyword arguments for the selector factory. This is how a matched
            CONTROL is configured from the treatment's measured move rate, so it carries
            numbers rather than a mode name.

    Returns:
        A :class:`HarnessDispatcher`, or ``None`` when no variant set was configured.

    Raises:
        ValueError: If a name is not registered. Skipping an unknown name would silently
            shrink a two-variant arm to a one-variant one, which reports ``can_evolve``
            False and trains exactly like the control under a name that says otherwise.
            Also if a selector is named without a set it can walk: a feature-driven rule
            over fewer than two rungs refuses every decision, which is a control arm and
            must be configured as one rather than arrived at by accident.
    """
    if not names:
        return None
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise ValueError(
            f"unknown harness variant(s) {unknown}; registered: {sorted(VARIANTS)}. "
            f"Dropping an unknown name would shrink the variant set and silently turn a "
            f"harness-evolving arm into its own control"
        )
    variants = [VARIANTS[n] for n in names]
    rule = None
    if selector is not None:
        from selfevo.harness.selectors import build_selector, ladder

        rule = build_selector(selector, selector_args)
        # Refuses here, before any GPU is touched, on a set the rule cannot walk: a variant
        # with no generation budget, two rungs of the same length, or fewer than two rungs.
        ladder(variants)
    return HarnessDispatcher(variants, selector=rule)
