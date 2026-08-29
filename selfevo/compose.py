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
    "Incompatibility",
    "validate",
    "SHAPERS",
    "ROUTERS",
    "GATES",
    "EVOLVE_TARGETS",
    "EVOLVE_POLICIES",
    "CRITICS",
    "register_shaper",
    "register_router",
]

# Registries. Values are factories; a new component is added by registering it from
# anywhere, with no edit to this module -- the same extension pattern as routing.base.
SHAPERS: dict[str, Callable[..., object] | None] = {"none": None}
ROUTERS: dict[str, Callable[..., object] | None] = {}
GATES: dict[str, Callable[..., object] | None] = {"none": None, "prefix_dead": None}
EVOLVE_TARGETS: frozenset[str] = frozenset({"model", "harness", "reward", "both"})
# A critic scores candidates before they are trained on. `two_level` is the BigBang-v1
# design (a fast proxy for training value, then a slower check); their repo ships
# evaluation only -- no critic code -- so this is built from the description, not their
# implementation, and is labelled as such.
CRITICS: frozenset[str] = frozenset({"none", "scalar", "two_level"})
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
    frozen_eval_reward: bool = False
    policy_scored_by_frozen_reward: bool = False


def validate(cfg: PipelineConfig) -> list[Incompatibility]:
    """Return every reason ``cfg`` is invalid; empty means it can be run.

    Returns a list rather than raising on the first problem, so a sweep can report all
    rejected cells at once instead of one per run.

    Args:
        cfg: The configuration to check.

    Returns:
        Incompatibilities, possibly empty.
    """
    out: list[Incompatibility] = []

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


def is_valid(cfg: PipelineConfig) -> bool:
    """True when :func:`validate` finds nothing."""
    return not validate(cfg)
