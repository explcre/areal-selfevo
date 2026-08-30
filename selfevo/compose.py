"""Composable experiment axes, and the compatibility rules between them.

The framework is a factorial design: each axis is swappable so methods can be ablated
against each other. The axes are

    shaper        modifies advantages before the loss        none | entropy_bonus
    router        selects a training signal per unit         static_* | solve_rate | bandit
    gate          token-level masking within a sequence      none | prefix_dead
    evolve_target what an evolution step changes             model | harness | both
    evolve_policy what decides that step                     rule | learned_weights |
                                                             learned_code

**Some combinations are invalid, and silently so.** That is the reason this module exists
rather than a config dict. The clearest case, proved in
``experiments/harness/prefix_cancellation.py`` and guarded at runtime by
``token_level.assert_zero_sum_advantage``:

    ``gate=prefix_dead`` requires ``sum_i A_i = 0`` within each group. MEDS's shaper
    (``verl/workers/actor/dp_actor.py:560``) does
    ``advantages += torch.min(0.4*entropy.detach(), advantages.abs()/2)``; entropy is
    non-negative, so the sum becomes strictly positive and the gate's justification is
    void. Nothing raises. The run completes. The numbers are wrong.

So :func:`validate` rejects that pairing statically, and the runtime guards remain as a
second line of defence for the conditions that cannot be known before the run (actual
ratio, actual clipping, actual length normalisation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

__all__ = [
    "PipelineConfig",
    "CRITIC_FACTORIES",
    "Incompatibility",
    "validate",
    "SHAPERS",
    "ROUTERS",
    "GATES",
    "EVOLVE_TARGETS",
    "EVOLVE_POLICIES",
    "CRITICS",
    "META_CRITICS",
    "CADENCES",
    "register_shaper",
    "register_router",
]

# Registries. Values are factories; a new component is added by registering it from
# anywhere, with no edit to this module -- the same extension pattern as routing.base.
SHAPERS: dict[str, Callable[..., object] | None] = {"none": None}
def _static_router(**kw: object) -> object:
    """Factory for :class:`selfevo.routing.routers.StaticRouter`."""
    from .routing.routers import StaticRouter

    return StaticRouter(**kw)  # type: ignore[arg-type]


def _solve_rate_router(**kw: object) -> object:
    """Factory for :class:`selfevo.routing.routers.SolveRateRouter` (SAMPLE granularity)."""
    from .routing.routers import SolveRateRouter

    return SolveRateRouter(**kw)  # type: ignore[arg-type]


def _cluster_router(**kw: object) -> object:
    """Factory for :class:`selfevo.routing.cluster.ClusterRouter` (CLUSTER granularity)."""
    from .routing.cluster import ClusterRouter

    return ClusterRouter(**kw)  # type: ignore[arg-type]


def _random_router(**kw: object) -> object:
    """Factory for :class:`selfevo.routing.routers.RandomRouter`, the matched control."""
    from .routing.routers import RandomRouter

    return RandomRouter(**kw)  # type: ignore[arg-type]


# These were implemented but never registered, so no config could select them and the
# `router` axis was unusable despite four working routers existing. Imported lazily for the
# same reason as the critic factories: `compose` must stay importable for configuration
# validation without pulling in the routing criteria.
ROUTERS: dict[str, Callable[..., object] | None] = {
    "static": _static_router,            # fixed weights, the fixed-mode baseline
    "solve_rate": _solve_rate_router,    # SAMPLE granularity, I_RL silence split
    "cluster": _cluster_router,          # CLUSTER granularity, one signal per cluster
    "random": _random_router,            # matched control: same proportions, shuffled units
}
GATES: dict[str, Callable[..., object] | None] = {"none": None, "prefix_dead": None}
EVOLVE_TARGETS: frozenset[str] = frozenset({"model", "harness", "reward", "both"})
# A critic scores candidates before they are trained on. `two_level` is the BigBang-v1
# design (a fast proxy for training value, then a slower check); their repo ships
# evaluation only -- no critic code -- so this is built from the description, not their
# implementation, and is labelled as such.
CRITICS: frozenset[str] = frozenset({"none", "scalar", "two_level"})
# Factories per critic. None means DECLARED BUT NOT IMPLEMENTED -- validate() rejects a
# config naming one unless allow_stubs=True, so a stub cannot be mistaken for a component.
# The meta-critic (BigBang paper 2.3): compares the critic's assessments against OBSERVED
# training outcomes on held-out real tasks, and uses the discrepancy to refine both the
# evaluation criteria and the generation strategy. Without it an evolving critic has no
# anchor and can drift to whatever it finds easy to score.
# How often an evolving evaluator updates RELATIVE to the policy.
#   simultaneous -- both update every step. This is the configuration the co-evolution
#                   literature reports collapsing; allowed only with nothing to co-evolve.
#   alternating  -- one side is frozen while the other optimises, with an explicit
#                   timescale gap (critic_update_every > 1).
#   frozen       -- the evaluator never updates. Safe, and the degenerate case of
#                   alternating with an infinite period.
CADENCES: frozenset[str] = frozenset({"frozen", "alternating", "simultaneous"})

META_CRITICS: frozenset[str] = frozenset({"none", "outcome_calibrated"})
def _outcome_calibrated_meta_critic(**kw: object) -> object:
    """Factory for :class:`selfevo.meta_critics.OutcomeCalibratedMetaCritic`.

    Imported lazily for the same reason as the critic factories: `compose` must stay
    importable for configuration validation without dragging in the routing criteria.
    """
    from .meta_critics import OutcomeCalibratedMetaCritic

    return OutcomeCalibratedMetaCritic(**kw)  # type: ignore[arg-type]


META_CRITIC_FACTORIES: dict[str, Callable[..., object] | None] = {
    "none": None,
    # IMPLEMENTED: judges a critic by whether its ordering beats chance, which is the only
    # statistic that separates an uninformative critic from an actively inverted one.
    "outcome_calibrated": _outcome_calibrated_meta_critic,
}
def _scalar_critic(**kw: object) -> object:
    """Factory for :class:`selfevo.critics.ScalarCritic`.

    Imported lazily so `compose` stays importable without pulling the critic module (and
    through it the routing criteria) at configuration-validation time.
    """
    from .critics import ScalarCritic

    return ScalarCritic(**kw)  # type: ignore[arg-type]


CRITIC_FACTORIES: dict[str, Callable[..., object] | None] = {
    "none": None,               # "none" is the absence of a critic, so None is correct
    "scalar": _scalar_critic,   # IMPLEMENTED: scores by I_RL with the solved/unsolved split
    "two_level": None,          # not implemented; BigBang ships no critic code to build on
}
# Components whose None factory means "nothing to build", not "not built yet".
_LEGITIMATELY_EMPTY: frozenset[str] = frozenset({"none"})
EVOLVE_POLICIES: frozenset[str] = frozenset({"rule", "learned_weights", "learned_code"})

# Shapers that break the centred-advantage invariant `sum_i A_i = 0`. Membership here is a
# claim about the shaper's arithmetic, not a preference: any shaper adding a non-negative
# per-element term belongs in this set.
_BREAKS_CENTRING: frozenset[str] = frozenset({"entropy_bonus"})


def register_shaper(name: str, factory: Callable[..., object] | None, *, breaks_centring: bool) -> str:
    """Register an advantage shaper.

    Args:
        name: Identifier used in :class:`PipelineConfig`.
        factory: Callable producing the shaper, or None for a declared-but-unbuilt entry.
        breaks_centring: Whether it destroys ``sum_i A_i = 0``. Callers must state this
            explicitly rather than let it be inferred, because getting it wrong silently
            enables an invalid combination.

    Returns:
        The registered name.

    Raises:
        ValueError: On an empty name, or on re-registration with a different
            ``breaks_centring``.
    """
    global _BREAKS_CENTRING
    if not name:
        raise ValueError("shaper name must be non-empty")
    if name in SHAPERS and (name in _BREAKS_CENTRING) != breaks_centring:
        raise ValueError(
            f"shaper {name!r} already registered with breaks_centring="
            f"{name in _BREAKS_CENTRING}"
        )
    SHAPERS[name] = factory
    if breaks_centring:
        _BREAKS_CENTRING = _BREAKS_CENTRING | {name}
    return name


def register_router(name: str, factory: Callable[..., object] | None) -> str:
    """Register a router factory under ``name``."""
    if not name:
        raise ValueError("router name must be non-empty")
    ROUTERS[name] = factory
    return name


@dataclass(frozen=True)
class Incompatibility:
    """One reason a configuration is rejected.

    Args:
        axes: The axes involved.
        reason: What breaks, stated so it can be checked.
        evidence: Where the claim is established -- a file, a line, a measurement.
    """

    axes: tuple[str, ...]
    reason: str
    evidence: str

    def __str__(self) -> str:
        return f"{'+'.join(self.axes)}: {self.reason} [{self.evidence}]"


@dataclass(frozen=True)
class PipelineConfig:
    """One cell of the factorial design.

    Args:
        shaper: Advantage shaper name; must be in :data:`SHAPERS`.
        router: Router name; must be in :data:`ROUTERS` if that registry is populated.
        gate: Token-level gate name; must be in :data:`GATES`.
        evolve_target: What an evolution step changes.
        evolve_policy: What decides the step. ``rule`` is a fixed criterion;
            ``learned_weights`` is an LLM-parameterised policy; ``learned_code`` is a
            policy that emits code. The last two are the ablation the design cares about --
            whether the choice policy needs to be a learned model at all.
        require_feedback: Whether the run will supply outcomes to the router. A learned
            policy that never observes is a fixed policy with extra steps, so this is
            checked rather than assumed.
        cadence: How an evolving evaluator updates relative to the policy, from
            :data:`CADENCES`. Defaults to ``frozen`` because that is the only value safe
            with no further configuration.
        critic_update_every: Policy steps between evaluator updates. Must exceed 1 under
            ``alternating``: a period of 1 IS simultaneous update, whatever the axis is
            labelled, and mislabelling it would defeat the guard rather than satisfy it.
        critic: Candidate scorer, from :data:`CRITICS`. ``two_level`` follows the
            BigBang-v1 description; their released repo is evaluation-only and contains no
            critic implementation, so nothing here is derived from their code.
        frozen_eval_reward: Whether a reward that never changes is retained for
            measurement, alongside any evolving training reward. Required when
            ``evolve_target`` touches the reward: otherwise a rising curve cannot be
            distinguished from a reward that got easier.
        policy_scored_by_frozen_reward: Whether the evolve-policy's own objective is the
            frozen reward rather than the evolving one. Required when a learned policy can
            evolve the reward, because otherwise lowering the bar is the optimum of the
            objective as written, not a failure mode to watch for.
    """

    shaper: str = "none"
    router: str = "solve_rate"
    gate: str = "none"
    evolve_target: str = "model"
    evolve_policy: str = "rule"
    require_feedback: bool = False
    critic: str = "none"
    meta_critic: str = "none"
    cadence: str = "frozen"
    critic_update_every: int = 1
    frozen_eval_reward: bool = False
    policy_scored_by_frozen_reward: bool = False


def _stub_problems(cfg: "PipelineConfig") -> list["Incompatibility"]:
    """Components named by ``cfg`` that are declared but have no implementation."""
    out: list[Incompatibility] = []
    for axis, name, registry in (
        ("shaper", cfg.shaper, SHAPERS),
        ("gate", cfg.gate, GATES),
        ("critic", cfg.critic, CRITIC_FACTORIES),
        ("meta_critic", cfg.meta_critic, META_CRITIC_FACTORIES),
    ):
        if name in _LEGITIMATELY_EMPTY:
            continue
        if name in registry and registry[name] is None:
            out.append(
                Incompatibility(
                    (axis,),
                    f"{axis}={name!r} is declared but has no implementation; running it "
                    "would silently do nothing. Pass allow_stubs=True to explore the "
                    "configuration space before building it.",
                    "registry factory is None",
                )
            )
    return out


def validate(cfg: PipelineConfig, *, allow_stubs: bool = False) -> list[Incompatibility]:
    """Return every reason ``cfg`` is invalid; empty means it can be run.

    Returns a list rather than raising on the first problem, so a sweep can report all
    rejected cells at once instead of one per run.

    Args:
        cfg: The configuration to check.
        allow_stubs: If True, do not reject components that are declared but unimplemented.
            For sweeping the design space before building; a real run must leave it False,
            or a stub silently becomes a no-op component.

    Returns:
        Incompatibilities, possibly empty.
    """
    out: list[Incompatibility] = []
    if not allow_stubs:
        out.extend(_stub_problems(cfg))

    if cfg.shaper not in SHAPERS:
        out.append(Incompatibility(("shaper",), f"unknown shaper {cfg.shaper!r}", "registry"))
    if cfg.gate not in GATES:
        out.append(Incompatibility(("gate",), f"unknown gate {cfg.gate!r}", "registry"))
    if ROUTERS and cfg.router not in ROUTERS:
        out.append(Incompatibility(("router",), f"unknown router {cfg.router!r}", "registry"))
    if cfg.evolve_target not in EVOLVE_TARGETS:
        out.append(
            Incompatibility(
                ("evolve_target",), f"unknown target {cfg.evolve_target!r}", "registry"
            )
        )
    if cfg.evolve_policy not in EVOLVE_POLICIES:
        out.append(
            Incompatibility(
                ("evolve_policy",), f"unknown policy {cfg.evolve_policy!r}", "registry"
            )
        )

    # The rule this module exists for.
    if cfg.gate == "prefix_dead" and cfg.shaper in _BREAKS_CENTRING:
        out.append(
            Incompatibility(
                ("shaper", "gate"),
                f"shaper {cfg.shaper!r} destroys sum_i A_i = 0, which prefix_dead gating "
                "requires; the combination produces wrong gradients without raising",
                "prefix_cancellation.py; MEDS dp_actor.py:560",
            )
        )

    if cfg.cadence not in CADENCES:
        out.append(
            Incompatibility(("cadence",), f"unknown cadence {cfg.cadence!r}", "registry")
        )

    # Anything that makes the evaluator a moving target.
    coevolving = cfg.meta_critic != "none" or cfg.evolve_target in ("reward", "both")

    if coevolving and cfg.cadence == "simultaneous":
        out.append(
            Incompatibility(
                ("cadence", "meta_critic" if cfg.meta_critic != "none" else "evolve_target"),
                "updating the evaluator and the policy on every step is the configuration "
                "the co-evolution literature reports collapsing into shared shortcuts; use "
                "cadence='alternating' with critic_update_every > 1, or 'frozen'",
                "arXiv 2606.07367; 2607.05297; 2510.23595",
            )
        )

    if cfg.cadence == "alternating" and cfg.critic_update_every <= 1:
        out.append(
            Incompatibility(
                ("cadence", "critic_update_every"),
                f"critic_update_every={cfg.critic_update_every} under 'alternating' IS "
                "simultaneous update; alternation needs a timescale gap, so this label "
                "would defeat the guard rather than satisfy it",
                "two-timescale separation",
            )
        )

    if cfg.cadence == "frozen" and cfg.evolve_target in ("reward", "both"):
        out.append(
            Incompatibility(
                ("cadence", "evolve_target"),
                "cadence='frozen' says the evaluator never updates, but evolve_target "
                f"{cfg.evolve_target!r} says it does; one of them is wrong",
                "internal consistency",
            )
        )

    if cfg.meta_critic not in META_CRITICS:
        out.append(
            Incompatibility(
                ("meta_critic",), f"unknown meta_critic {cfg.meta_critic!r}", "registry"
            )
        )

    # A meta-critic exists to compare critic judgements against real outcomes; with no
    # critic there is nothing to calibrate, and with no frozen anchor there is nothing to
    # compare against. Both would run and quietly do nothing.
    if cfg.meta_critic != "none":
        if cfg.critic == "none":
            out.append(
                Incompatibility(
                    ("meta_critic", "critic"),
                    "a meta-critic calibrates a critic's judgements against observed "
                    "outcomes; with critic='none' there is nothing to calibrate",
                    "BigBang paper 2.3",
                )
            )
        if not cfg.frozen_eval_reward:
            out.append(
                Incompatibility(
                    ("meta_critic", "frozen_eval_reward"),
                    "outcome calibration needs a held-out measure that does not move; "
                    "without one the meta-critic compares the critic against a target "
                    "the critic itself influences",
                    "BigBang paper 2.3",
                )
            )

    if cfg.critic not in CRITICS:
        out.append(
            Incompatibility(("critic",), f"unknown critic {cfg.critic!r}", "registry")
        )

    evolves_reward = cfg.evolve_target in ("reward", "both")

    if evolves_reward and not cfg.frozen_eval_reward:
        out.append(
            Incompatibility(
                ("evolve_target", "frozen_eval_reward"),
                "evolving the reward without a frozen held-out reward makes progress "
                "unmeasurable: a rising score can mean the policy improved or the reward "
                "got easier, and the two are not separable after the run",
                "measurement",
            )
        )

    if (
        evolves_reward
        and cfg.evolve_policy.startswith("learned")
        and not cfg.policy_scored_by_frozen_reward
    ):
        out.append(
            Incompatibility(
                ("evolve_target", "evolve_policy"),
                "a learned policy that can evolve the reward AND is scored by that same "
                "reward has an optimum at 'make the reward easier'; score the policy by "
                "the frozen reward instead",
                "degenerate fixed point",
            )
        )

    # A learned policy that never receives outcomes cannot learn.
    if cfg.evolve_policy.startswith("learned") and not cfg.require_feedback:
        out.append(
            Incompatibility(
                ("evolve_policy",),
                f"{cfg.evolve_policy!r} needs require_feedback=True; without an outcome "
                "channel it is a fixed policy with extra machinery, and would be reported "
                "as a learned arm",
                "feedback.LearningRouter",
            )
        )

    # Evolving the harness while attributing results to the model confounds the ablation.
    if cfg.evolve_target == "both" and cfg.evolve_policy == "rule":
        out.append(
            Incompatibility(
                ("evolve_target", "evolve_policy"),
                "evolving model and harness together under a fixed rule leaves no way to "
                "attribute a gain to either; run them as separate arms, or use a policy "
                "that records which target it chose",
                "factorial design",
            )
        )

    return out


def is_valid(cfg: PipelineConfig, *, allow_stubs: bool = False) -> bool:
    """True when :func:`validate` finds nothing."""
    return not validate(cfg, allow_stubs=allow_stubs)
