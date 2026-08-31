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
