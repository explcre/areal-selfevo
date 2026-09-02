"""A CPU simulation of the routing credit loop, with ground truth about which mode is right.

Why this exists. The claim that per-prompt credit fixes the learned router rested on ONE GPU
arm (`ctxpc`, 46 steps) whose mode mix moved. A mix that moves is not a mix that learned:
there is no control in that measurement, no ground truth about which mode was actually better
for which group, and the arm's own log cannot separate "the router found structure" from "the
router's estimates got noisier and its argmax started flapping". Both produce a mode mix away
from uniform. This module supplies the missing half -- an environment where the right answer
is KNOWN by construction, so a router can be scored on whether it finds it, and where the
whole loop runs on a CPU in seconds and can therefore be run with a control and with enough
seeds to put an error bar on a difference.

**The measurement that matters is not the one the goal file tracks.** L1 distance from a
uniform mode mix is what the failing run reported (0.056 -> 0.027 over 129 steps) and it is
what a fixed run is expected to raise. It is not sufficient evidence, and this module was
written partly to show why: with a shared per-batch scalar the router's own choices still feed
back into which contexts each arm sees, so its arms drift apart on noise and L1 rises to
0.11 -> 0.21 here while its targeting stays at chance. So both numbers are reported, and the
one that carries the claim is :meth:`RunTrace.subset_contrast` -- how differently the router
treats the two prompt subsets. A router with a preference that points nowhere scores 0 on it
however far from uniform its mix has wandered.

**What is simulated and what is real.** The world is a simulation: prompts, their solve rates,
and the effect of a mode on them. Everything downstream of a decision is the shipped code --
the same :class:`~selfevo.routing.contextual.ContextualBanditRouter`, the same
:func:`~selfevo.routing.outcomes.batch_outcomes`, the same
:class:`~selfevo.routing.prompt_credit.PromptCreditLedger`. Only the ORDER in which the actor
calls them is restated here, and that restatement is the one real fidelity risk: this project
has twice found a re-derived harness inventing a defect that the real path did not have. It is
bounded deliberately -- the ordering is four lines, it is written to mirror
``areal.trainer.ppo.actor._route_advantages`` (route the batch, observe the prompt's value,
credit the prior decision, record the current one, then let the update land), and the real path
keeps its own end-to-end tests in ``test_prompt_credit_wired.py``. A partitioning router would
NOT be faithfully driven here, because the actor routes through ``route_all`` and this drives
``route`` per unit; that is equivalent for every router without a ``route_batch``, which the
contextual router is, and wrong for :class:`~selfevo.routing.cluster.ClusterRouter`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from selfevo.observability import FEATURE_NAMES
from selfevo.routing.base import RoutingContext, TrainingMode
from selfevo.routing.contextual import ContextualBanditRouter
from selfevo.routing.feedback import ConfoundedUpdate, DecisionOutcome
from selfevo.routing.outcomes import batch_outcomes
from selfevo.routing.prompt_credit import PromptCreditLedger

__all__ = ["ModePreferenceWorld", "RunTrace", "CREDIT_RULES", "simulate"]

# The credit rules a run can be driven with. The first three are exactly the values
# ``GroupRoutingConfig.credit`` accepts, so a result here names a runnable arm rather than a
# hypothetical one; the fourth is the ledger's own per-prompt baseline, which no config value
# reaches yet and which this module exists to justify wiring.
CREDIT_RULES: tuple[str, ...] = ("batch", "prompt", "prompt_centered", "prompt_self_baseline")


@dataclass
class ModePreferenceWorld:
    """Prompts whose right mode is determined by a feature the router can see.

    The point of the design is that the correct policy is CONTEXTUAL, not a constant. Half the
    prompts improve only when routed to ``high_mode`` and the other half only under
    ``low_mode``, and which half a prompt is in is written into one observability feature. So a
    router that merely settles on a favourite mode scores zero on
    :meth:`RunTrace.subset_contrast` no matter how lopsided its mix becomes, and only a router
    that reads the feature can score above chance. That is the discrimination the failing run
    lacked, and a world where one mode were simply best everywhere could not tell the two
    apart.

    A prompt's value also drifts upward whatever is applied to it, scaled by its remaining
    headroom. This is not decoration: it is the confound the live run hit -- a prompt's solve
    rate is compared across ~29 steps over which the policy improves generally, so every arm
    is credited positively and the arm applied most during the improving window looks best.
    Making the drift headroom-scaled makes it NON-STATIONARY, which is the case a lifetime
    baseline has to survive and a fixed offset would not test.

    Args:
        n_prompts: Size of the prompt pool. Sampled without replacement in shuffled epochs, so
            a prompt recurs every ``n_prompts // batch_size`` steps -- the property per-prompt
            credit depends on, and the reason the pool is small enough for a prompt to be seen
            about fifteen times in a run of the default length.
        batch_size: Prompts routed per step.
        gain: Added to a prompt's value when the mode applied is the one it prefers.
        trend: Common upward drift, multiplied by remaining headroom ``1 - value``.
        noise: Standard deviation of the per-application observation noise.
        seed: RNG seed. A private generator, so a run is reproducible independently of any
            other sampling, which matters because the shuffled control has to differ from its
            treatment in exactly one thing.
        marker: The feature that identifies the subset. Must be an observability feature or
            the router would raise ``MissingFeatures`` on every context.
        high_mode: Mode that helps prompts with a high ``marker``.
        low_mode: Mode that helps the rest.

    Raises:
        ValueError: If ``marker`` is not an observability feature, if the two modes are the
            same -- which would delete the discrimination the world exists to pose -- or if
            the pool is smaller than a batch.
    """

    n_prompts: int = 256
    batch_size: int = 32
    gain: float = 0.06
    trend: float = 0.010
    noise: float = 0.02
    seed: int = 0
    marker: str = "truncated_fraction"
    high_mode: str = TrainingMode.SFT
    low_mode: str = TrainingMode.RL

    def __post_init__(self) -> None:
        if self.marker not in FEATURE_NAMES:
            raise ValueError(
                f"marker {self.marker!r} is not an observability feature; the router reads "
                f"only {FEATURE_NAMES} and would raise MissingFeatures on every context"
            )
        if self.high_mode == self.low_mode:
            raise ValueError(
                "high_mode and low_mode must differ: with one mode right everywhere the "
                "world cannot distinguish a router that targets from one with a favourite"
            )
        if self.n_prompts < self.batch_size:
            raise ValueError(
                f"n_prompts ({self.n_prompts}) must be at least batch_size "
                f"({self.batch_size}) so a batch can be drawn without repeats"
            )
        self._rng = np.random.default_rng(self.seed)
        # Alternating rather than random membership so the two subsets are exactly balanced;
        # an imbalance would show up in subset_contrast as sampling noise on the smaller one.
        base = np.where(np.arange(self.n_prompts) % 2 == 0, 0.75, 0.05)
        self._marker = base + self._rng.uniform(-0.05, 0.05, self.n_prompts)
        self._value = self._rng.uniform(0.10, 0.40, self.n_prompts)
        self._order: list[int] = []

    def best_mode(self, prompt: int) -> str:
        """The mode that actually helps this prompt. The ground truth a router is scored on."""
        return self.high_mode if self._marker[prompt] > 0.4 else self.low_mode

    def value(self, prompt: int) -> float:
        """This prompt's current solve rate, the quantity per-prompt credit differences."""
        return float(self._value[prompt])

    def draw(self) -> list[int]:
        """One step's prompts, drawn from a shuffled epoch so recurrence is regular.

        Reshuffling only when the epoch runs out, rather than sampling with replacement each
        step, is what makes a prompt's gap between appearances roughly constant. With
        replacement a large share of prompts would be seen once and never paired, and the
        ledger's pairing rate -- the thing that decides whether per-prompt credit reaches the
        router at all -- would be an artefact of the sampler rather than of the design.
        """
        if len(self._order) < self.batch_size:
            self._order = self._rng.permutation(self.n_prompts).tolist()
        out, self._order = self._order[: self.batch_size], self._order[self.batch_size :]
        return out

    def context(self, prompt: int, step: int, index: int) -> RoutingContext:
        """A routing context for this prompt, carrying every observability feature.

        Every feature the router reads is populated, and all but ``marker`` and ``solve_rate``
        are noise. That is deliberate: a router handed only the informative feature would be
        solving an easier problem than the live one, and a survivor of a mutation test on such
        a world would mean nothing. The router has to find the one feature that matters among
        seven.
        """
        extra = {name: float(self._rng.uniform(0.0, 1.0)) for name in FEATURE_NAMES}
        extra["solve_rate"] = self.value(prompt)
        extra["mean_response_len"] = float(self._rng.uniform(50, 400))
        extra["mean_logprob"] = float(self._rng.uniform(-2, 0))
        extra[self.marker] = float(self._marker[prompt])
        return RoutingContext(
            solve_rate=self.value(prompt),
            group_size=8,
            has_teacher=False,
            unit_id=f"{step}:{index}",
            extra=extra,
        )

    def apply(self, prompt: int, mode: str) -> None:
        """Advance a prompt after a mode was applied to it, clipped to a solve rate."""
        gain = self.gain if mode == self.best_mode(prompt) else 0.0
        drift = self.trend * (1.0 - self.value(prompt))
        self._value[prompt] = float(
            np.clip(
                self._value[prompt] + gain + drift + self._rng.normal(0.0, self.noise),
                0.0,
                1.0,
            )
        )


@dataclass(frozen=True)
class RunTrace:
    """Every decision a run made, paired with the mode that was actually right for it.

    Holding the ground truth alongside the choice is the whole reason this exists: the live
    logs record only the choice, so no amount of re-reading them can say whether a preference
    pointed anywhere. Quarters rather than a per-step curve because the failing signature is
    quoted by quarter and because a per-step mode share over one batch is too noisy to read.

    Args:
        decisions: Per step, per unit, ``(chosen_mode, best_mode)``.
        modes: The modes the router could choose between, so a share is defined for a mode
            that was never chosen -- absent keys would silently drop such a mode out of the L1
            sum and make a router that collapsed onto one mode look CLOSER to uniform.
        updates: Times ``observe`` was actually called with a non-empty set of outcomes. A run
            whose credit never reached the router is a no-op, not a null result, and this is
            the number that tells the two apart.
        metrics: The ledger's own counters at the end of the run, empty for the batch rule.
    """

    decisions: tuple[tuple[tuple[str, str], ...], ...]
    modes: tuple[str, ...]
    updates: int
    metrics: dict[str, float] = field(default_factory=dict)

    def _windows(self, quarters: int) -> list[list[tuple[str, str]]]:
        """Split the run into equal windows, dropping the remainder so they are comparable."""
        if quarters < 1:
            raise ValueError(f"quarters must be >= 1, got {quarters}")
        width = len(self.decisions) // quarters
        if width == 0:
            raise ValueError(
                f"{len(self.decisions)} steps cannot be split into {quarters} windows"
            )
        return [
            [d for step in self.decisions[k * width : (k + 1) * width] for d in step]
            for k in range(quarters)
        ]

    def mode_shares(self, rows: Sequence[tuple[str, str]]) -> dict[str, float]:
        """Fraction of decisions taking each mode, including modes never chosen."""
        n = len(rows)
        return {m: sum(1 for r in rows if r[0] == m) / n for m in self.modes}

    def l1_from_uniform(self, quarters: int = 4) -> list[float]:
        """Per window, the L1 distance of the mode mix from uniform. The tracked number.

        Reported because it is the number the failing run is quoted by, and because a fix that
        did not move it would be suspect. It is NOT the evidence: it rises under a signal that
        teaches nothing, since the router's choices decide which contexts each arm sees and the
        arms drift apart on that feedback alone. Read it beside
        :meth:`subset_contrast`, never instead of it.
        """
        return [
            sum(abs(s - 1.0 / len(self.modes)) for s in self.mode_shares(w).values())
            for w in self._windows(quarters)
        ]

    def subset_contrast(self, quarters: int = 4) -> list[float]:
        """Per window, how differently the two prompt subsets are treated. The evidence.

        Half the total variation distance between the mode distribution on prompts whose best
        mode is ``high_mode`` and on the rest: 0 when the router treats both subsets alike, 1
        when it separates them perfectly. Chance is 0 however lopsided the overall mix is,
        which is exactly the property L1 lacks -- a router that routes everything to SFT scores
        1.33 on L1 and 0 here. It is also blind to a router whose preference points at the
        WRONG mode in the same way for both subsets, so a test that uses it must also check the
        direction with :meth:`targeting_accuracy`.
        """
        out = []
        for w in self._windows(quarters):
            subsets = {b for _, b in w}
            if len(subsets) < 2:
                raise ValueError("a window saw only one prompt subset; nothing to contrast")
            hi, lo = sorted(subsets)
            sh_hi = self.mode_shares([r for r in w if r[1] == hi])
            sh_lo = self.mode_shares([r for r in w if r[1] == lo])
            out.append(0.5 * sum(abs(sh_hi[m] - sh_lo[m]) for m in self.modes))
        return out

    def targeting_accuracy(self, quarters: int = 4) -> list[dict[str, float]]:
        """Per window and per subset, the fraction routed to the mode that was actually right.

        The direction check. ``subset_contrast`` says the router distinguishes the subsets;
        this says it distinguishes them the right way round. A router that had learned the
        structure and inverted it would score high on the first and at chance on this one, and
        reporting only the first would call that a success.
        """
        out = []
        for w in self._windows(quarters):
            acc = {}
            for subset in sorted({b for _, b in w}):
                rows = [c for c, b in w if b == subset]
                acc[subset] = sum(1 for c in rows if c == subset) / len(rows)
            out.append(acc)
        return out


def _credit_prompt_batch(
    ledger: PromptCreditLedger,
    sightings: Sequence[tuple[int, str, str, float]],
    step: int,
    *,
    centred: bool,
    shuffler: np.random.Generator | None,
) -> dict[str, DecisionOutcome]:
    """Turn one step's prompt sightings into outcomes for the decisions they close.

    Args:
        ledger: The shipped ledger. Called once per sighting, in order, because it both credits
            the prior decision and records the current one and the two must see the same value.
        sightings: ``(prompt, unit_id, mode, value)`` in the order they were routed.
        step: Current batch index, which the ledger uses to refuse a within-batch pairing.
        centred: Subtract the batch's mean credit, reproducing ``credit="prompt_centered"``.
        shuffler: When given, permute the credits across the units that earned them. This is
            the absorption control and it is the only thing that differs between the treatment
            and its control: the same ledger, the same pairings, the same multiset of credit
            values, the same number of updates -- only the correspondence between a prompt and
            what happened to it is destroyed. A mechanism that scores the same with it as
            without it was never using the correspondence, and this project has found enough
            "smart" mechanisms matching a random one to make that the first thing to check.
    """
    pairs = []
    for prompt, unit_id, mode, value in sightings:
        closed = ledger.observe_and_record(f"p{prompt}", unit_id, mode, value, step)
        if closed is not None:
            prior, credit = closed
            pairs.append((prior, credit))
    if not pairs:
        return {}
    if centred:
        shift = sum(c for _, c in pairs) / len(pairs)
        pairs = [(prior, c - shift) for prior, c in pairs]
    if shuffler is not None:
        values = [c for _, c in pairs]
        shuffler.shuffle(values)
        pairs = [(prior, v) for (prior, _), v in zip(pairs, values)]
    return {
        prior.unit_id: DecisionOutcome(
            mode=prior.mode, value=credit, batch_id=str(prior.step)
        )
        for prior, credit in pairs
    }


def simulate(
    credit: str,
    *,
    seed: int = 0,
    steps: int = 160,
    world: ModePreferenceWorld | None = None,
    shuffle_credit: bool = False,
    router: ContextualBanditRouter | None = None,
) -> RunTrace:
    """Drive the real router through the real credit machinery against a simulated world.

    The order mirrors the actor: route the whole batch first, so every decision in it is made
    from the same parameters; then observe each prompt's value and let it close whatever
    decision that prompt is carrying; then apply the modes. A decision is therefore never
    credited with an outcome measured before it was applied, which is the mistake that would
    make any of this look like it worked.

    Args:
        credit: One of :data:`CREDIT_RULES`.
        seed: Seeds the world and, separately, the control's shuffling.
        steps: Batches to run. The default gives a prompt about fifteen appearances at the
            default pool size, so the per-prompt baseline has a history to work with.
        world: A world to reuse; built from ``seed`` when omitted.
        shuffle_credit: Run the absorption control. Rejected for the batch rule, where every
            unit gets the same scalar and permuting it is a guaranteed no-op -- reporting that
            as a passing control would be reporting an equivalent mutant as evidence.
        router: A router to drive; the default is a contextual router with one round-robin
            batch of cold start, which it needs or its first batch is single-mode, refused as
            confounded, and the loop never starts.

    Returns:
        A :class:`RunTrace`.

    Raises:
        ValueError: If ``credit`` is not a known rule, or the control is asked for on ``batch``.
    """
    if credit not in CREDIT_RULES:
        raise ValueError(f"credit must be one of {CREDIT_RULES}, got {credit!r}")
    if shuffle_credit and credit == "batch":
        raise ValueError(
            "shuffling a batch-credited run is a no-op -- every unit already holds the same "
            "scalar -- and a control that cannot fail is not a control"
        )
    world = world if world is not None else ModePreferenceWorld(seed=seed)
    router = router if router is not None else ContextualBanditRouter(
        cold_start_rounds=world.batch_size
    )
    shuffler = np.random.default_rng(seed + 9_991) if shuffle_credit else None
    ledger = (
        None
        if credit == "batch"
        else PromptCreditLedger(
            baseline="self_mean" if credit == "prompt_self_baseline" else "last"
        )
    )

    decisions: list[tuple[tuple[str, str], ...]] = []
    pending: tuple[dict[str, str], str, float] | None = None
    updates = 0
    for step in range(steps):
        prompts = world.draw()
        contexts = [world.context(p, step, i) for i, p in enumerate(prompts)]
        modes = [router.route(ctx).argmax() for ctx in contexts]
        decisions.append(tuple((m, world.best_mode(p)) for m, p in zip(modes, prompts)))

        if credit == "batch":
            mean_now = float(np.mean([world.value(p) for p in prompts]))
            if pending is not None:
                prev_modes, prev_id, prev_mean = pending
                try:
                    outcomes, _ = batch_outcomes(
                        prev_modes, batch_id=prev_id, value=mean_now - prev_mean
                    )
                except ConfoundedUpdate:
                    # A single-mode batch. Refused by the shipped producer, so the simulation
                    # must skip it too rather than inventing a credit the real loop never has.
                    outcomes = {}
                if outcomes:
                    router.observe(outcomes)
                    updates += 1
            pending = (
                {ctx.unit_id: m for ctx, m in zip(contexts, modes)},
                str(step),
                mean_now,
            )
        else:
            assert ledger is not None
            outcomes = _credit_prompt_batch(
                ledger,
                [
                    (p, ctx.unit_id, m, world.value(p))
                    for p, ctx, m in zip(prompts, contexts, modes)
                ],
                step,
                centred=credit == "prompt_centered",
                shuffler=shuffler,
            )
            if outcomes:
                router.observe(outcomes)
                updates += 1

        for prompt, mode in zip(prompts, modes):
            world.apply(prompt, mode)

    return RunTrace(
        decisions=tuple(decisions),
        modes=tuple(router.modes),
        updates=updates,
        metrics=ledger.as_metrics() if ledger is not None else {},
    )
