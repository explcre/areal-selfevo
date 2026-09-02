"""Agent harnesses as swappable VARIANTS, so the harness axis has something to dispatch over.

The routing design has two coordinates -- an estimator and a harness action -- and the harness
coordinate has been inert because there was nothing to dispatch between. A harness that exists
in exactly one configuration cannot be routed to; "evolve the harness" then means editing a
singleton, which is a schedule, not a decision.

This module defines the seam. A :class:`HarnessVariant` is one named configuration; a
:class:`HarnessAdapter` runs a task under a variant and returns a :class:`HarnessRollout`.
Concrete adapters live beside this file so a second or third harness can be added without
touching anything that consumes them.

**What this is FOR, concretely.** The whole method rests on a measured quantity -- the solve
rate, and the composition of the RL-silent channel it induces. On math that comes free from
the reward function. On agentic software tasks it needs a harness to produce trajectories at
all. So the first job of an adapter is not training: it is *measuring whether the same
composition law holds on SDE tasks*, where solve rates are low by construction and the theory
predicts the unsolved branch dominates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = [
    "HarnessVariant",
    "HarnessRollout",
    "HarnessAdapter",
    "VARIANTS",
    "register_variant",
]


@dataclass(frozen=True)
class HarnessVariant:
    """One named harness configuration -- the unit a dispatcher chooses between.

    Args:
        name: Identifier used in configs and logs.
        description: What makes this variant different, in one line. Required, because a
            variant set whose members are not distinguishable in words will not be
            distinguishable in results either.
        step_limit: Maximum agent steps. The cheapest real axis of variation, and the one
            ``truncated_fraction`` is expected to select on.
        settings: Adapter-specific overrides, passed through verbatim.
    """

    name: str
    description: str
    step_limit: int = 40
    settings: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("variant name must be non-empty")
        if not self.description:
            raise ValueError(
                f"variant {self.name!r} needs a description: a variant set whose members "
                "cannot be told apart in words cannot be told apart in results"
            )
        if self.step_limit < 1:
            raise ValueError(f"step_limit must be >= 1, got {self.step_limit}")


@dataclass(frozen=True)
class HarnessRollout:
    """One attempt at one task under one variant.

    Args:
        task_id: Instance identifier, e.g. a SWE-bench instance id.
        variant: Name of the variant that produced it.
        solved: Whether the task's own checker passed. This is the reward.
        steps: Agent steps consumed.
        truncated: Whether the step limit was hit. Distinguishing "failed" from "ran out of
            budget" is the distinction the whole feature set is built around, and on agentic
            tasks it is far more common than on math.
        error: Exception text when the rollout could not complete at all. An errored rollout
            is NOT a failed one and must not be scored as reward 0 -- infrastructure failures
            laundered into the training signal is a mistake this project has already made.
        cost: Whatever the caller meters (tokens, seconds, dollars).
    """

    task_id: str
    variant: str
    solved: bool
    steps: int
    truncated: bool = False
    error: str | None = None
    cost: float = 0.0

    @property
    def usable(self) -> bool:
        """False when the rollout errored, so callers cannot silently score it as a failure."""
        return self.error is None


@runtime_checkable
class HarnessAdapter(Protocol):
    """Runs one task under one variant. The only thing a dispatcher needs from a harness."""

    def variants(self) -> tuple[HarnessVariant, ...]:
        """Variants this adapter can run."""
        ...

    def run(self, task_id: str, variant: HarnessVariant) -> HarnessRollout:
        """Attempt ``task_id`` under ``variant``."""
        ...


VARIANTS: dict[str, HarnessVariant] = {}


def register_variant(v: HarnessVariant) -> HarnessVariant:
    """Add a variant to the shared registry.

    Args:
        v: The variant.

    Returns:
        The same variant, so this can be used at definition sites.

    Raises:
        ValueError: If the name is already registered with different settings. Silently
            replacing one would make two runs with the same config name incomparable.
    """
    existing = VARIANTS.get(v.name)
    if existing is not None and existing != v:
        raise ValueError(
            f"variant {v.name!r} is already registered with different settings; "
            f"re-registering would make two runs sharing this name incomparable"
        )
    VARIANTS[v.name] = v
    return v


# The initial variant set. Deliberately small and differing along ONE axis each, so a
# dispatch result can be attributed. `truncated_fraction` is already a routing feature, which
# makes short/long the first pairing worth testing.
register_variant(HarnessVariant("plain", "default agent, standard step budget", step_limit=40))
register_variant(HarnessVariant("long", "same agent, 2.5x the step budget", step_limit=100))
register_variant(HarnessVariant("short", "same agent, reduced budget; the cheap arm", step_limit=15))

# A second variant set, for single-turn RLVR, differing on GENERATION BUDGET.
#
# The three above vary ``step_limit``, and on a single-turn math run nothing reads it: the
# workflow issues exactly one completion request per rollout, so dispatching between them
# would change no bit of the rollout while the log reported switches -- an arm that is its
# own control, and the one failure ``HarnessDispatcher`` cannot catch, because the field it
# compares for uniqueness is the field nothing reads. ``step_limit`` is therefore 1 here,
# which is the truth (one agent step), and the axis of variation moves into ``settings``,
# where ``selfevo.harness.selectors.GENERATION_BUDGET_KEY`` names the field the rollout
# genuinely reads. It reaches the engine as ``max_completion_tokens`` on the OpenAI proxy
# path and lands in ``GenerationHyperparameters.max_new_tokens``.
#
# The three budgets are MEASURED, not guessed. Qwen2.5-32B-Instruct on GSM8K at temperature
# 1.0, 60 rollouts sampled at a 1024-token cap (``~/runs/probe1024``; nothing reached that
# cap, so the distribution is uncensored; median response 155 tokens) truncates at:
#
#     cap  64   96   128  160  192  224  256  288  320
#     frac 0.97 0.83 0.62 0.42 0.20 0.13 0.07 0.02 0.00
#
# so 96 / 160 / 256 put one rung above ``TruncationStepLimitSelector``'s upper threshold
# (0.5), one inside its dead band, and one at its lower threshold (0.05). A ladder whose
# rungs all sat above 320 would differ only in a number nothing could respond to: every rung
# reports truncation 0.0, the selector ratchets to the bottom and stays, and that looks like
# a working controller while measuring nothing.
register_variant(
    HarnessVariant(
        "gen96",
        "single-turn; 96-token generation budget, where ~83% of rollouts never terminate",
        step_limit=1,
        settings={"max_new_tokens": 96},
    )
)
register_variant(
    HarnessVariant(
        "gen160",
        "single-turn; 160-token generation budget, about the median response length",
        step_limit=1,
        settings={"max_new_tokens": 160},
    )
)
register_variant(
    HarnessVariant(
        "gen256",
        "single-turn; 256-token generation budget, where ~7% of rollouts never terminate",
        step_limit=1,
        settings={"max_new_tokens": 256},
    )
)
