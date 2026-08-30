"""Types and protocols for choosing a training signal per unit.

Kept free of torch so routing decisions can be constructed and tested on CPU. Tensors
enter only in :mod:`selfevo.signals`, which turns a :class:`RoutingDecision` into
per-token weights.

Extension points, all Protocol-based so a new mode or router needs no edits here:
``TrainingMode`` is an open registry of names, ``Router`` is a Protocol, and
``RoutingDecision`` carries a mode->weight mapping rather than a single label, so soft
mixtures are expressible and hard routing is just the degenerate one-hot case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Protocol, runtime_checkable

__all__ = [
    "Granularity",
    "TrainingMode",
    "HarnessAction",
    "RoutingContext",
    "RoutingDecision",
    "Router",
    "register_mode",
    "known_modes",
]


class HarnessAction(Enum):
    """What a unit contributes to harness evolution, orthogonal to its training mode.

    A trajectory can feed the gradient and the harness at the same time: the two read
    different things from it -- the estimator reads the tokens, the harness evolver reads
    the failure mode -- so forcing a partition throws one of them away. Co-Harness
    (2607.22688) does partition (success -> model, failure -> harness); keeping this axis
    separate lets that rule be reproduced exactly *and* relaxed, as an ablation rather
    than a rewrite.

    ``NONE`` is the default everywhere, so a config that never mentions harness evolution
    behaves exactly as it did before this axis existed.
    """

    NONE = "none"
    PROPOSE = "propose"
    VALIDATE = "validate"


class Granularity(Enum):
    """Resolution at which a routing decision is made.

    Coarser is cheaper and more stable; finer can exploit within-batch structure but
    needs a better estimate of the solve rate to do so.
    """

    TASK = "task"
    CLUSTER = "cluster"
    SAMPLE = "sample"
    TOKEN = "token"


# Modes are an open set: a name plus whether it needs an external target. Registering is
# how a new mode becomes routable without touching this module's logic.
_MODES: dict[str, bool] = {}


def register_mode(name: str, *, needs_teacher: bool) -> str:
    """Register a training mode.

    Args:
        name: Mode identifier, e.g. ``"rl"``, ``"sft"``, ``"distill"``.
        needs_teacher: Whether the mode requires an external target (gold text or teacher
            distribution). Routers use this to avoid selecting a mode that cannot be
            supplied for a given unit.

    Returns:
        The registered name, so this can be used at class definition sites.

    Raises:
        ValueError: If the name is empty, or is re-registered with a different
            ``needs_teacher`` (silently changing it would make routing decisions
            inconsistent across call sites).
    """
    if not name:
        raise ValueError("mode name must be non-empty")
    if name in _MODES and _MODES[name] != needs_teacher:
        raise ValueError(
            f"mode {name!r} already registered with needs_teacher={_MODES[name]}"
        )
    _MODES[name] = needs_teacher
    return name


def known_modes() -> Mapping[str, bool]:
    """Registered modes as ``{name: needs_teacher}``."""
    return dict(_MODES)


class TrainingMode:
    """Canonical names for the built-in modes.

    Hard distillation is deliberately absent: it is SFT on teacher-generated text -- the
    same estimator with a different proposal distribution -- and giving it a separate mode
    would imply a difference in the gradient that does not exist. Use ``SFT`` with a
    teacher-sourced target.
    """

    RL = register_mode("rl", needs_teacher=False)
    SFT = register_mode("sft", needs_teacher=True)
    DISTILL = register_mode("distill", needs_teacher=True)
    SKIP = register_mode("skip", needs_teacher=False)


@dataclass(frozen=True)
class RoutingContext:
    """What a router is allowed to look at when deciding.

    Args:
        solve_rate: Observed fraction of correct samples for this unit, in [0, 1]. For a
            prompt this is the group's mean binary reward, which GRPO already computes.
        group_size: Samples drawn for this unit, >= 1.
        granularity: Resolution this context describes.
        has_teacher: Whether an *external* target is available. A router must not select
            a teacher-requiring mode when this is False and the unit has no self-target
            (see :attr:`has_target`).
        can_evolve_harness: Whether harness evolution is wired for this run. A router must
            not emit a non-NONE :class:`HarnessAction` when this is False -- the same
            guard shape as ``has_teacher``, so neither can be forgotten independently.
        unit_id: Optional identifier (prompt id, cluster id) for logging and for
            reproducing a decision.
        extra: Escape hatch for router-specific features (cluster stats, difficulty
            estimates) without widening this dataclass for every experiment.
    """

    solve_rate: float
    group_size: int
    granularity: Granularity = Granularity.SAMPLE
    has_teacher: bool = False
    # Keyword-only: kept next to has_teacher because it is the same guard shape, but
    # appended to __init__ so it cannot capture the positional slot unit_id already had.
    can_evolve_harness: bool = field(default=False, kw_only=True)
    unit_id: str | None = None
    extra: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.solve_rate <= 1.0:
            raise ValueError(f"solve_rate must be in [0, 1], got {self.solve_rate}")
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")

    @property
    def has_self_target(self) -> bool:
        """Whether this unit's own rollout already contains a usable SFT target.

        A group with ``solve_rate > 0`` drew at least one correct sample, and that sample
        is a target -- this is rejection-sampling fine-tuning, and it needs no external
        teacher. It matters because the fully-solved groups that GRPO cannot learn from
        (their advantages are identically zero) are exactly the groups where a free target
        is guaranteed to exist.

        Only meaningful when ``solve_rate`` is a group mean over sampled answers, i.e. at
        TASK/CLUSTER/SAMPLE granularity. At TOKEN granularity there is no per-token notion
        of a correct sibling sample, so this is forced to False.
        """
        if self.granularity is Granularity.TOKEN:
            return False
        return self.solve_rate > 0.0

    @property
    def has_target(self) -> bool:
        """Whether *any* target exists for a teacher-requiring mode: external or self."""
        return self.has_teacher or self.has_self_target


@dataclass(frozen=True)
class RoutingDecision:
    """Mode weights for one unit, plus why.

    Args:
        weights: ``{mode_name: weight}``. Weights are non-negative and need not be
            normalised by the caller -- :meth:`normalised` does that. An empty mapping is
            rejected: "no decision" must be spelled as ``{SKIP: 1.0}`` so that downstream
            code never has to guess what an absent decision meant.
        reason: Short human-readable justification, carried into logs so a decision can be
            audited after the fact.
        harness: What this unit contributes to harness evolution. Defaults to
            ``HarnessAction.NONE``, which is what every decision constructed before this
            field existed meant, so adding it changed no behaviour.

    Raises:
        ValueError: If ``weights`` is empty, contains a negative weight, sums to zero, or
            names an unregistered mode.
    """

    weights: Mapping[str, float]
    reason: str = ""
    harness: HarnessAction = HarnessAction.NONE

    def __post_init__(self) -> None:
        # Order matters: a mixture like {RL: 2.0, SFT: -1.0} sums to 1.0, so a sum check
        # alone would accept a negative weight. Each condition is checked on its own so a
        # test for one cannot pass because of another.
        if not self.weights:
            raise ValueError(
                "weights must be non-empty; spell 'no decision' as {TrainingMode.SKIP: 1.0}"
            )
        for mode, w in self.weights.items():
            if mode not in _MODES:
                raise ValueError(f"unknown mode {mode!r}; register it first")
            if w < 0:
                raise ValueError(f"weight for {mode!r} must be >= 0, got {w}")
        if sum(self.weights.values()) <= 0:
            raise ValueError("weights must sum to a positive value")

    def normalised(self) -> dict[str, float]:
        """Weights scaled to sum to 1."""
        total = sum(self.weights.values())
        return {m: w / total for m, w in self.weights.items()}

    def argmax(self) -> str:
        """The single highest-weighted mode.

        Ties resolve by mode name so the result is deterministic across runs -- an
        arbitrary tie-break would make a routing ablation irreproducible.
        """
        return max(sorted(self.weights), key=lambda m: self.weights[m])


@runtime_checkable
class Router(Protocol):
    """Chooses training-signal weights for a unit.

    A learned meta-policy implements this same Protocol, so swapping a fixed criterion for
    a learned one requires no changes anywhere else.
    """

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        """Return mode weights for the unit described by ``ctx``."""
        ...
