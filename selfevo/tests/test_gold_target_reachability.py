"""Can a gold-grounded SFT target reach the seam where a group's advantage is decided?

On MATH the UNSOLVED groups are 60.9% of the RL-silent channel and 25.5% of all groups, and
by construction they carry no self-target: every rollout was wrong, so there is nothing
correct of the model's own to reinforce. The cheapest conceivable supplier is the dataset's
own gold solution -- no teacher model, no extra rollout, no GPU inference. This file exists
because that plan rests on a premise nobody had checked, and the premise is FALSE at the one
place the plan needs it to be true.

MEASURED 2026-09-01 on this checkout, by the tests below and by the probes they were
distilled from:

* The gold SOLUTION TEXT is present in 7500/7500 raw ``MATH-lighteval`` training rows, and
  7496/7500 carry a brace-balanced ``\boxed{}``. Tokenised with the live tokenizer it is a
  median of 162 tokens, p90 482, max 2494, and fits in 1024 tokens for 99.15% of rows. So
  supply and length are NOT the obstacle.
* ``areal/dataset/competition_math.py`` throws it away. Its ``keep = {"messages", "answer"}``
  drops every other column, so what reaches the dataset the workflow reads is the prompt and
  a bare ``\boxed{...}`` FINAL ANSWER -- a handful of characters, not a derivation.
* ``RLVRWorkflow.arun_episode`` builds a trajectory out of seven TENSORS, and neither the
  gold solution nor the gold answer is among them. The answer is handed to the reward
  function as ``**task_data`` inside the workflow and then discarded.
* Therefore, by the time ``PPOActor._compute_advantages`` runs -- the only place group
  membership and the advantage tensor coexist, and so the only place a group-level decision
  can be applied at all -- there is no gold of any kind in ``data``.

That is a structural finding and not a plumbing detail. An advantage is a per-token
coefficient on the log-probability of the token the model ACTUALLY EMITTED. On an unsolved
group every emitted token belongs to a wrong derivation, so no value written into the
advantage tensor is a step toward the gold; a positive constant there is a step toward the
WRONG answer, which is exactly why ``GroupRoutingConfig.unsolved_advantage`` is required to
be <= 0. Gold grounding is a batch-CONSTRUCTION intervention -- put the gold tokens in the
batch as a row and let the ordinary estimator act on them -- and the advantage seam is the
wrong altitude for it whether or not the text is ever plumbed this far.

What these tests pin:

1. The premise on the OTHER side, that unsolved groups reach no update, driven through the
   REAL ``_compute_advantages`` rather than re-derived. This holds under the fixed rule at
   its shipped defaults, under the fixed rule with a solved constant set, under every
   constructible registered router, and even under a control router that draws SFT for
   every single group -- because every router in the registry gates a teacher-requiring mode
   on ``RoutingContext.has_target``, which an unsolved group cannot satisfy.
2. That the two ends agree. A gold field appearing in the trajectory schema while the apply
   seam still has no parameter that could carry a target -- or the reverse -- is a
   half-built gold arm, which in this repo is worse than an absent one: it is how a mode
   that nothing applies gets reported as a distillation arm that never ran. That is the
   reason ``_APPLIED`` exists, and this is the same guard one level up.

What they do NOT establish. They say nothing about whether a plumbed gold arm would help;
that needs a GPU run. They also cannot observe a LIVE batch -- producing one needs an
inference engine -- so the trajectory schema is read from the source of the function that
constructs it, and the dataset half is measured by running the real adapter.
"""

from __future__ import annotations

import ast
import inspect
import logging
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from areal.api.cli_args import GroupRoutingConfig  # noqa: E402
from selfevo import compose  # noqa: E402
from selfevo.integration import group_apply  # noqa: E402
from selfevo.tests.test_group_routing import (  # noqa: E402
    G,
    SOLVED_AND_UNSOLVED,
    advantages,
    make_actor,
)

logging.disable(logging.INFO)

REPO = Path(group_apply.__file__).resolve().parents[2]
WORKFLOW = REPO / "areal" / "workflow" / "rlvr.py"

# The keys RLVRWorkflow has always emitted. Asserted as a SUBSET rather than an equality so
# that an unrelated upstream addition does not fail this file, while the gold check below
# still catches anything gold-shaped.
KNOWN_TRAJECTORY_KEYS = frozenset(
    {
        "input_ids",
        "loss_mask",
        "logprobs",
        "versions",
        "turn_ids",
        "attention_mask",
        "rewards",
    }
)

# Substrings a gold-carrying field would plausibly contain. Matched as substrings, not by
# equality, because the name a future plumbing picks is not knowable in advance and the
# point of the check is to notice one arriving at all.
GOLD_NAME_PARTS = ("gold", "solution", "answer", "target", "reference")

MATH_DATASET = "DigitalLearningGmbH/MATH-lighteval"


def _gold_shaped(names) -> set[str]:
    """Names that look like they carry a supervision target.

    Args:
        names: Field or key names to screen.

    Returns:
        The subset whose name contains any of :data:`GOLD_NAME_PARTS`.
    """
    return {n for n in names if any(part in n for part in GOLD_NAME_PARTS)}


def _trajectory_schema() -> set[str]:
    """The keys ``RLVRWorkflow.arun_episode`` puts into a trajectory.

    Read from the source of the function that builds the dict rather than from a live
    batch, because building a live batch needs an inference engine and this file must run on
    CPU. The union of every string-keyed dict literal inside the function is taken, not the
    largest one: a careless edit that adds a second dict would otherwise be invisible, and
    an invisible addition is precisely the failure this check exists to catch.

    Returns:
        The set of literal string keys.

    Raises:
        AssertionError: If the function cannot be found, which would mean this check had
            silently stopped reading anything.
    """
    tree = ast.parse(WORKFLOW.read_text())
    found = False
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef) or node.name != "arun_episode":
            continue
        found = True
        for inner in ast.walk(node):
            if isinstance(inner, ast.Dict):
                for k in inner.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        keys.add(k.value)
    assert found, f"no arun_episode found in {WORKFLOW}; this check reads nothing"
    return keys


def _seam_can_carry_a_target() -> bool:
    """Whether the apply seam has any parameter that could carry a supervision target.

    The seam writes CONSTANTS into an advantage tensor; a gold target is a token sequence,
    which no constant can express. So the honest test for "this seam could consume a gold"
    is whether either entry point has grown a target-shaped argument, and today neither has.

    Returns:
        True if ``apply_decisions`` or ``apply_mixtures`` takes a target-shaped parameter.
    """
    names = set(inspect.signature(group_apply.apply_decisions).parameters)
    names |= set(inspect.signature(group_apply.apply_mixtures).parameters)
    return bool(_gold_shaped(names))


def _constructible_routers() -> list[str]:
    """Registered routers that can be built with no arguments in this process.

    ``random`` and ``code_policy`` refuse a default on purpose -- one would become a control
    that is bit-identical to the off arm, the other has no policy source -- so they are
    excluded here and ``random`` is exercised explicitly below at the proportions that make
    the point hardest.

    Returns:
        Sorted router names.
    """
    out = []
    for name, factory in sorted(compose.ROUTERS.items()):
        if factory is None:
            continue
        try:
            factory()
        except Exception:
            continue
        out.append(name)
    return out


ROUTER_NAMES = _constructible_routers()


# ------------------------------------------------------------------ the premise holds ---


def test_a_solved_group_does_reach_the_update():
    """Anti-vacuity for every unsolved assertion below.

    A test asserting that some group's advantages are zero passes just as well on an actor
    that writes nothing at all. This pins that the SAME call, on the SAME batch, does move
    the solved half -- so "the unsolved half is untouched" is a statement about the branch
    and not about a dead code path.
    """
    gr = GroupRoutingConfig(enabled=True, solved_advantage=0.5)
    adv = advantages(make_actor(gr), SOLVED_AND_UNSOLVED)
    assert float(adv[:G].abs().max()) == pytest.approx(0.5), adv[:G]


@pytest.mark.parametrize(
    "gr",
    [
        GroupRoutingConfig(enabled=True),
        GroupRoutingConfig(enabled=True, solved_advantage=0.5),
        GroupRoutingConfig(enabled=True, solved_advantage=2.0),
    ],
    ids=["shipped-defaults", "solved-0.5", "solved-2.0"],
)
def test_an_unsolved_group_reaches_no_update_under_the_fixed_rule(gr):
    """Every rollout failed, so nothing correct exists to reinforce and nothing is written.

    ``unsolved_advantage`` defaults to 0.0, and raising ``solved_advantage`` does not leak
    across: the two branches are independent by construction. This is the 25.5% of MATH
    groups that a gold supplier would exist to serve.
    """
    adv = advantages(make_actor(gr), SOLVED_AND_UNSOLVED)
    assert float(adv[G:].abs().max()) == 0.0, adv[G:]


@pytest.mark.parametrize("router", ROUTER_NAMES)
def test_an_unsolved_group_reaches_no_update_under_any_registered_router(router):
    """Not a property of the fixed rule: no router in the registry can reach these groups.

    Every router gates a teacher-requiring mode on ``RoutingContext.has_target``, and an
    unsolved group satisfies neither half of it -- ``has_teacher`` is written False by
    ``_route_groups`` because no run wires a teacher, and ``has_self_target`` is False
    because ``solve_rate`` is 0. So the branch falls to SKIP or RL, both of which leave a
    silent group exactly as it arrived.
    """
    gr = GroupRoutingConfig(enabled=True, solved_advantage=0.5, router=router)
    adv = advantages(make_actor(gr), SOLVED_AND_UNSOLVED)
    assert float(adv[G:].abs().max()) == 0.0, (router, adv[G:])


def test_even_an_all_sft_control_cannot_reach_an_unsolved_group(monkeypatch):
    """The hardest case: a control that draws SFT for EVERY group still writes nothing here.

    ``router=random`` at ``sft=1.0`` is the strongest available probe of the claim, because
    it removes any question of the criterion declining to act. The SFT constant lands on the
    solved group and not on the unsolved one, which shows the block is the missing TARGET
    and not a missing decision.
    """
    monkeypatch.setenv("SELFEVO_RANDOM_PROPORTIONS", "sft=1.0")
    gr = GroupRoutingConfig(enabled=True, solved_advantage=0.5, router="random")
    adv = advantages(make_actor(gr), SOLVED_AND_UNSOLVED)
    assert float(adv[:G].abs().max()) == pytest.approx(0.5), adv[:G]
    assert float(adv[G:].abs().max()) == 0.0, adv[G:]


# ------------------------------------------------------------- the gold does not arrive ---


def test_the_trajectory_schema_carries_no_gold_field():
    """Nothing gold-shaped is in the dict a rollout hands to training.

    The seven known keys are asserted present so that a schema this check failed to read
    would fail loudly rather than pass on an empty set.
    """
    schema = _trajectory_schema()
    assert KNOWN_TRAJECTORY_KEYS <= schema, sorted(KNOWN_TRAJECTORY_KEYS - schema)
    assert _gold_shaped(schema) == set(), sorted(_gold_shaped(schema))


@pytest.fixture(scope="module")
def math_columns():
    """Raw and adapted column names for the MATH training set, or a skip.

    Loaded once for the module: the adapter maps 7500 rows, and doing it per test would
    make a data assertion cost more than the rest of this file combined.
    """
    datasets = pytest.importorskip("datasets")
    from areal.dataset.competition_math import get_math_rl_dataset

    try:
        raw = datasets.load_dataset(path=MATH_DATASET, split="train")
    except Exception as exc:  # pragma: no cover - box without the dataset cached
        pytest.skip(f"{MATH_DATASET} not available here: {exc}")
    # DELIBERATELY OUTSIDE the try. The skip above answers one question only -- is the
    # dataset on this box -- and a broader guard around the adapter would turn an adapter
    # that RAISES into a skip, which reads as "not measured here" rather than "broken". That
    # is not hypothetical: mutating ``keep`` to drop the answer column makes this call die on
    # a KeyError, and the first version of this fixture swallowed it and let the mutant live.
    adapted = get_math_rl_dataset(
        path=MATH_DATASET, split="train", tokenizer=None, max_length=None
    )
    return raw, adapted


def test_the_raw_dataset_does_carry_a_gold_solution(math_columns):
    """Anti-vacuity for the drop test: the gold really is there to begin with.

    If this ever fails, the finding is not "the adapter drops the gold" but "this dataset
    has no gold", which is a different problem with a different fix.
    """
    raw, _ = math_columns
    assert "solution" in raw.column_names, raw.column_names
    non_empty = sum(1 for s in raw["solution"] if s and str(s).strip())
    assert non_empty == len(raw), f"{non_empty}/{len(raw)} rows carry a solution"


def test_the_math_adapter_drops_the_gold_solution_text(math_columns):
    """The gold derivation does not survive the adapter, so it never reaches a rollout.

    ``keep = {"messages", "answer"}`` removes it. What survives is the bare boxed FINAL
    ANSWER, which is what the grader needs and is not a supervision target: training toward
    ``\boxed{7}`` teaches the model to emit the token 7, not to derive it.
    """
    _, adapted = math_columns
    assert set(adapted.column_names) == {"messages", "answer"}, adapted.column_names
    assert "solution" not in adapted.column_names


def test_gold_is_absent_at_both_ends_or_present_at_both():
    """A half-built gold arm is worse than an absent one, and this is the guard for it.

    Two ends have to move together: the batch must carry the gold tokens, and the seam that
    spends them must be able to accept a target. Today neither does, which is a consistent
    state and is what this asserts. If a later change adds a gold field to the trajectory
    while leaving the seam unable to consume one, the arm would be configurable, would log
    as a gold arm, and would apply nothing -- the exact failure ``_APPLIED`` exists to
    prevent, one level up. The reverse is checked by the same equality because a seam
    advertising a target it can never be given is the same defect wearing the other face.
    """
    supplied = bool(_gold_shaped(_trajectory_schema()))
    consumable = _seam_can_carry_a_target()
    assert supplied == consumable, (
        f"gold in the trajectory schema: {supplied}; the apply seam can carry a target: "
        f"{consumable}. These must change together. Plumbing gold into the batch without "
        "giving group_apply a way to spend it produces an arm that logs as gold-grounded "
        "and applies nothing; the reverse advertises a target that can never arrive."
    )
