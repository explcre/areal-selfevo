"""Guards against the silent-zero paths that have bitten this project before.

Every guard here has a test in tests/test_guards_fire.py that constructs the input the
guard must refuse and asserts that it *fires*. A guard that has never been observed to
fail is not evidence; three guards in one day in this project passed exactly what they
were written to refuse.

Each guard raises `GuardViolation`. None of them returns a bool, because an assertion on
a function returning True is not an assertion on observable state.
"""

from __future__ import annotations

from collections.abc import Sequence

from .types import GroupAdvantages, Task


class GuardViolation(RuntimeError):
    """Raised when a guard detects a condition that would produce a meaningless number."""


def assert_token_budget_fits(
    prompt_tokens: int,
    max_new_tokens: int,
    served_context_len: int,
) -> None:
    """G2: refuse a token budget that exceeds the context the server actually serves.

    Args:
        prompt_tokens: Tokens in the prompt.
        max_new_tokens: Requested generation cap.
        served_context_len: Context length reported by the *backend*, not a constant in
            our config. The distinction is the whole point: a cap validated against a
            hardcoded number stays valid when the server is started with a smaller one,
            and the resulting refusals get graded as wrong answers.

    Raises:
        GuardViolation: if the budget cannot fit, or if any argument is non-positive.
    """
    for name, v in (
        ("prompt_tokens", prompt_tokens),
        ("max_new_tokens", max_new_tokens),
        ("served_context_len", served_context_len),
    ):
        if v <= 0:
            raise GuardViolation(f"{name} must be positive, got {v}")
    total = prompt_tokens + max_new_tokens
    if total > served_context_len:
        raise GuardViolation(
            f"token budget {prompt_tokens}+{max_new_tokens}={total} exceeds the served "
            f"context length {served_context_len}. Requests would be refused and the "
            f"refusals graded as failures (guard G2)."
        )


def assert_group_nonempty(rewards: Sequence[float], what: str = "group") -> None:
    """G3: refuse an empty or singleton reward batch instead of scoring it 0.

    Args:
        rewards: The batch of rewards.
        what: Label used in the error message.

    Raises:
        GuardViolation: if fewer than two rewards are present.
    """
    if len(rewards) < 2:
        raise GuardViolation(
            f"{what} has {len(rewards)} member(s); an empty or singleton batch must be "
            f"refused, not averaged to 0.0 (guard G3)."
        )


def assert_batch_not_all_degenerate(groups: Sequence[GroupAdvantages]) -> None:
    """G4: refuse a batch in which every group carries zero reward-directed gradient.

    Args:
        groups: Per-group advantage results.

    Raises:
        GuardViolation: if `groups` is empty, or if every group is degenerate.

    Note:
        This is the false-negative guard. Gradient statistics computed over an all-
        degenerate batch say nothing about the method; they say the batch was unanimous.
        Reporting "the gradient is ~0" from such a batch is the error this prevents.
    """
    if not groups:
        raise GuardViolation("batch contains no groups (guard G4).")
    if all(g.degenerate for g in groups):
        raise GuardViolation(
            f"all {len(groups)} groups in this batch are degenerate (constant reward), "
            f"so the batch carries exactly zero reward-directed gradient. Any gradient "
            f"statistic computed here is a false negative (guard G4)."
        )


def assert_task_not_in_buffer(task: Task, buffer_texts: Sequence[str]) -> None:
    """G5: refuse to score novelty for a task already present in the buffer.

    Args:
        task: The task about to be scored.
        buffer_texts: Current buffer contents.

    Raises:
        GuardViolation: if the task text is already a buffer member.

    Note:
        If the task is inserted into the buffer before it is scored, it becomes its own
        nearest neighbour, `sim = 1`, and `N = 0` annihilates R_task for every task in
        the run. The loop must insert after scoring (ambiguity A8).
    """
    if task.text in set(buffer_texts):
        raise GuardViolation(
            f"task {task.task_id!r} is already in the buffer, so it is its own nearest "
            f"neighbour and N would be forced to 0, annihilating R_task. Insert into the "
            f"buffer only after scoring (guard G5)."
        )


def assert_model_served(requested: str, served: Sequence[str]) -> None:
    """Refuse a run whose requested base model is not among the ids the backend serves.

    Args:
        requested: The model id we asked for.
        served: The ids returned by the backend's `/v1/models`.

    Raises:
        GuardViolation: if `served` is empty or does not contain `requested`.

    Note:
        An unregistered model id is served as the *base* model silently, with a 200 and
        no warning. Without this check a whole run can be attributed to a model that was
        never loaded.
    """
    if not served:
        raise GuardViolation(
            "backend reported no served models; cannot verify which model will answer."
        )
    if requested not in served:
        raise GuardViolation(
            f"requested model {requested!r} is not served; backend reports {list(served)}. "
            f"An unregistered id is answered by the base model with a 200 and no warning."
        )


def assert_proportions_match(
    treatment: dict[str, float],
    control: dict[str, float],
    tol: float = 0.02,
) -> None:
    """Refuse a control whose covariate proportions do not match the treatment's.

    Args:
        treatment: Measured proportions of the treatment arm, keyed by covariate value.
        control: Proportions of the control arm.
        tol: Maximum absolute deviation permitted per key.

    Raises:
        GuardViolation: if either mapping is empty, if the key sets differ, or if any key
            deviates by more than `tol`.

    Note:
        The control must match the treatment's *measured* proportions, not uniform and
        not the proportions we hoped for. A control drawn at uniform when the treatment
        drifted is not size-matched, and comparing against it is the easiest way to
        manufacture an effect.
    """
    if not treatment or not control:
        raise GuardViolation("proportion mapping is empty; nothing was matched.")
    if set(treatment) != set(control):
        raise GuardViolation(
            f"covariate supports differ: treatment has {sorted(treatment)}, control has "
            f"{sorted(control)}. The control is not matched."
        )
    for k in treatment:
        if abs(treatment[k] - control[k]) > tol:
            raise GuardViolation(
                f"covariate {k!r} deviates by {abs(treatment[k]-control[k]):.4f} > tol="
                f"{tol}: treatment {treatment[k]:.4f} vs control {control[k]:.4f}."
            )
