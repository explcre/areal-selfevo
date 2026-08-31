"""DAPO dynamic sampling, driven through the REAL rollout path it runs in.

Two things can silently ruin this baseline, and both are checked here rather than reasoned
about. First, the rule itself: verl keeps a prompt when

    std > 0 or len(prompt_uid2metric_vals[uid]) == 1        # recipe/dapo, np.std, ddof=0

so the tests below compare our decision against that expression evaluated with ``np.std``,
alongside literal expected values so a drift in the comparison helper cannot pass unnoticed.
Second, and worse: ``should_accept_fn`` is handed whatever ``arun_episode`` returned, and if
that were ONE sample rather than a whole group, every call would hit the singleton carve-out,
nothing would ever be rejected, and the arm would quietly be vanilla GRPO. That premise is
established by running the real ``GroupedRolloutWorkflow`` and the real
``WorkflowExecutor._create_workflow_task``, not by asserting it in a comment.

Nothing here re-derives ``dapo.py``'s arithmetic in the test file; the one formula written
out is verl's, which lives in another repository and cannot be imported.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from areal.api import RolloutWorkflow
from areal.api.cli_args import GRPOConfig, InferenceEngineConfig
from areal.api.io_struct import ModelResponse
from areal.experimental.openai.types import (
    InteractionWithTokenLogpReward,
    normalize_group_rewards,
)
from areal.infra.controller.rollout_controller import RolloutController
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow, RemoteInfEngine
from areal.infra.staleness_manager import StalenessManager
from areal.infra.workflow_executor import (
    BatchTaskDispatcher,
    WorkflowExecutor,
    _RolloutTaskInput,
)
from areal.utils import stats_tracker
from selfevo.baselines.dapo import dapo_dynamic_sampling, group_reward_std

logging.disable(logging.INFO)

# The import path the config help text and the launch scripts point at. A rename that left
# this string behind would disable the arm without failing anything else.
DAPO_PATH = "selfevo.baselines.dapo.dapo_dynamic_sampling"

G = 4  # gconfig.n_samples in examples/math/gsm8k_grpo.yaml
SEQLEN = 5

ALL_CORRECT = [1.0] * G
ALL_WRONG = [0.0] * G
MIXED = [0.0, 1.0, 0.0, 1.0]


# --------------------------------------------------------------------------- fixtures ---


def sample_dict(reward: float, seqlen: int = SEQLEN) -> dict[str, torch.Tensor]:
    """One sample with batch dim 1, shaped exactly as ``areal/workflow/rlvr.py`` builds it.

    Args:
        reward: The scalar reward for this sample.
        seqlen: Sequence length; irrelevant to the filter but required by the concatenation.

    Returns:
        A tensor dict whose ``rewards`` entry has shape ``(1,)``.
    """
    return {
        "input_ids": torch.arange(seqlen, dtype=torch.int32).unsqueeze(0),
        "loss_mask": torch.ones(1, seqlen, dtype=torch.int32),
        "logprobs": torch.zeros(1, seqlen),
        "versions": torch.zeros(1, seqlen, dtype=torch.int32),
        "turn_ids": torch.zeros(1, seqlen, dtype=torch.int32),
        "attention_mask": torch.ones(1, seqlen, dtype=torch.bool),
        "rewards": torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
    }


def interaction(reward: float, seqlen: int = SEQLEN) -> InteractionWithTokenLogpReward:
    """One completion as the OpenAI-proxy path produces it, with a real ModelResponse.

    ``examples/math/gsm8k_rl.py`` runs ``MathAgent``, whose rollouts arrive as these objects
    and are turned into tensors by the executor before the filter sees them, so this is the
    shape the live arm actually filters on.

    Args:
        reward: The scalar reward for this completion.
        seqlen: Number of prompt and of output tokens.

    Returns:
        An interaction whose real ``to_tensor_dict`` yields ``rewards`` of shape ``(1,)``.
    """
    return InteractionWithTokenLogpReward(
        model_response=ModelResponse(
            input_tokens=list(range(seqlen)),
            output_tokens=list(range(seqlen)),
            output_logprobs=[0.0] * seqlen,
            output_versions=[0] * seqlen,
        ),
        reward=float(reward),
    )


class _Scripted(RolloutWorkflow):
    """Emits one sample per call from a fixed per-sample reward list."""

    def __init__(self, rewards: list[float], as_interaction: bool = False):
        self.rewards = list(rewards)
        self.as_interaction = as_interaction
        self.calls = 0

    async def arun_episode(self, engine, data):
        """Return the next scripted sample, cycling if asked for more than were scripted."""
        r = self.rewards[self.calls % len(self.rewards)]
        self.calls += 1
        if self.as_interaction:
            return {f"cmpl-{self.calls}": interaction(r)}
        return sample_dict(r)


def grouped(rewards: list[float], as_interaction: bool = False) -> GroupedRolloutWorkflow:
    """The real group wrapper the trainer builds, over a scripted per-sample workflow."""
    return GroupedRolloutWorkflow(
        _Scripted(rewards, as_interaction), len(rewards), MagicMock()
    )


def executor(staleness_manager=None, consumer_batch_size: int = 1) -> WorkflowExecutor:
    """A real WorkflowExecutor with rollout side effects stubbed out.

    Args:
        staleness_manager: The real manager when the collector loop is under test, ``None``
            for a stub when only the accept/reject decision is.
        consumer_batch_size: Batch the executor is configured for.

    Returns:
        A ``WorkflowExecutor`` whose ``_create_workflow_task`` can be driven directly.
    """
    ex = WorkflowExecutor(
        config=InferenceEngineConfig(
            backend="sglang:d1",
            consumer_batch_size=consumer_batch_size,
            dump_to_file=False,
        ),
        inference_engine=MagicMock(),
        staleness_manager=staleness_manager or MagicMock(),
    )
    ex.logger = MagicMock()
    return ex


def run_through_executor(
    rewards: list[float], as_interaction: bool = False, filter_fn=dapo_dynamic_sampling
) -> tuple[bool, list[dict]]:
    """Run one grouped rollout through the real executor task.

    Args:
        rewards: One reward per sample of the group.
        as_interaction: Take the OpenAI-proxy path instead of the tensor path.
        filter_fn: The ``should_accept_fn`` to install; ``None`` is the shipped default.

    Returns:
        ``(accepted, seen)`` where ``seen`` holds every trajectory the filter was called on.
    """
    seen: list[dict] = []

    def spy(traj):
        seen.append(traj)
        return filter_fn(traj)

    task = executor()._create_workflow_task(
        _RolloutTaskInput(
            task_id=1,
            data={},
            workflow=grouped(rewards, as_interaction),
            should_accept_fn=None if filter_fn is None else spy,
        )
    )
    return asyncio.run(task()) is not None, seen


# ------------------------------------------------------------------- the reference rule ---


def reference_keep(vals: list[float]) -> bool:
    """verl's ``recipe/dapo`` keep-condition, written out because it cannot be imported.

    Args:
        vals: The per-sample rewards of one prompt, as verl collects them per uid.

    Returns:
        Whether verl would keep this prompt.
    """
    with np.errstate(invalid="ignore"):
        std = np.std(vals) if len(vals) else np.float64("nan")
    return bool(std > 0 or len(vals) == 1)


NAMED_GROUPS = [
    ("all-correct", ALL_CORRECT, False),
    ("all-wrong", ALL_WRONG, False),
    ("mixed", MIXED, True),
    ("one-of-four-correct", [0.0, 0.0, 0.0, 1.0], True),
    ("singleton-correct", [1.0], True),
    ("singleton-wrong", [0.0], True),
    ("empty", [], False),
    ("unanimous-nonbinary", [0.7] * G, False),
    ("nan-inside-a-mixed-group", [0.0, 1.0, float("nan"), 1.0], False),
    ("all-nan", [float("nan")] * G, False),
]


@pytest.mark.parametrize(
    "vals, expected", [(v, e) for _, v, e in NAMED_GROUPS],
    ids=[n for n, _, _ in NAMED_GROUPS],
)
def test_the_decision_matches_verls_rule_and_the_expected_literal(vals, expected):
    """Both halves matter: the literal pins the case, the reference pins the rule."""
    got = dapo_dynamic_sampling({"rewards": torch.tensor(vals, dtype=torch.float32)})
    assert got is expected
    assert got == reference_keep(vals)


@pytest.mark.parametrize(
    "vals", [ALL_CORRECT, ALL_WRONG, MIXED, [1.0], [], [0.25, 0.75, 0.75, 0.25]],
    ids=["all-correct", "all-wrong", "mixed", "singleton", "empty", "fractional"],
)
def test_group_reward_std_is_exactly_np_std(vals):
    """The population estimator, to the last bit.

    For any group of two or more, sample and population std are positive together, so the
    estimator does not change the accept decision and only this value test can catch a switch
    to ``unbiased=True``.
    """
    got = group_reward_std({"rewards": torch.tensor(vals, dtype=torch.float64)})
    with np.errstate(invalid="ignore"):
        want = float(np.std(vals)) if vals else float("nan")
    if np.isnan(want):
        assert np.isnan(got)
    else:
        assert got == want


def test_group_reward_std_matches_np_std_on_a_random_sweep():
    """A single hand-picked group cannot separate the two estimators; a sweep can."""
    rng = np.random.default_rng(0)
    for n in (2, 3, 4, 8, 16):
        for _ in range(20):
            vals = rng.normal(size=n).astype(np.float64)
            got = group_reward_std({"rewards": torch.tensor(vals)})
            assert got == pytest.approx(float(np.std(vals)), rel=0, abs=1e-12)


def test_the_singleton_carve_out_keeps_a_group_a_bare_std_test_would_drop():
    """A one-sample group has zero std, so ``std > 0`` alone would reject it."""
    one = {"rewards": torch.tensor([1.0])}
    assert group_reward_std(one) == 0.0
    assert dapo_dynamic_sampling(one) is True


def test_an_empty_group_is_not_treated_as_a_singleton():
    """``np.std([])`` is NaN and ``len([]) == 1`` is False, so verl drops it -- so must we."""
    empty = {"rewards": torch.tensor([])}
    assert np.isnan(group_reward_std(empty))
    assert dapo_dynamic_sampling(empty) is False


# ------------------------------------------------------------------- degenerate inputs ---


def test_a_trajectory_without_rewards_raises_instead_of_being_accepted():
    """Accepting it would turn the arm into vanilla GRPO with nothing in the log to show it."""
    for fn in (dapo_dynamic_sampling, group_reward_std):
        with pytest.raises(KeyError, match="vanilla GRPO"):
            fn({"input_ids": torch.zeros(G, SEQLEN)})


@pytest.mark.parametrize(
    "rewards",
    [[0.0, 1.0, 0.0, 1.0], (0.0, 1.0, 0.0, 1.0), np.array([0.0, 1.0, 0.0, 1.0])],
    ids=["list", "tuple", "ndarray"],
)
def test_non_tensor_rewards_decide_exactly_as_the_tensor_form_does(rewards):
    """Not every workflow hands back tensors; the rule must not depend on the container."""
    assert dapo_dynamic_sampling({"rewards": rewards}) is True
    assert group_reward_std({"rewards": rewards}) == 0.5
    unanimous = type(rewards)([1.0, 1.0, 1.0, 1.0]) if not isinstance(
        rewards, np.ndarray
    ) else np.array([1.0, 1.0, 1.0, 1.0])
    assert dapo_dynamic_sampling({"rewards": unanimous}) is False


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool, torch.float16,
                                   torch.float32, torch.float64])
def test_integer_and_float_reward_dtypes_agree(dtype):
    """``torch.std`` refuses integer dtypes outright, so the cast is load-bearing."""
    mixed = {"rewards": torch.tensor([0, 1, 0, 1], dtype=dtype)}
    unanimous = {"rewards": torch.tensor([1, 1, 1, 1], dtype=dtype)}
    assert dapo_dynamic_sampling(mixed) is True
    assert group_reward_std(mixed) == 0.5
    assert dapo_dynamic_sampling(unanimous) is False
    assert group_reward_std(unanimous) == 0.0


def test_a_scalar_reward_tensor_is_a_singleton_group():
    """``torch.tensor(1.0)`` has no batch dimension but is still exactly one sample."""
    assert dapo_dynamic_sampling({"rewards": torch.tensor(1.0)}) is True


def test_a_group_that_varies_only_slightly_is_still_kept():
    """The threshold is ``> 0``, not a tolerance: verl keeps anything with any spread."""
    traj = {"rewards": torch.tensor([0.5, 0.5, 0.5, 0.5000001], dtype=torch.float64)}
    assert group_reward_std(traj) > 0.0
    assert dapo_dynamic_sampling(traj) is True


# --------------------------------------- the premise: the filter sees a whole group ---


@pytest.mark.parametrize("as_interaction", [False, True], ids=["tensor", "openai-proxy"])
def test_the_filter_receives_the_whole_group_not_a_single_sample(as_interaction):
    """If this fails, every rejection test below is vacuous and the arm is vanilla GRPO."""
    _, seen = run_through_executor(MIXED, as_interaction)
    assert len(seen) == 1, "the filter must be called once per group, not once per sample"
    rewards = seen[0]["rewards"]
    assert tuple(rewards.shape) == (G,), rewards
    assert torch.equal(rewards.to(torch.float32), torch.tensor(MIXED))


@pytest.mark.parametrize("as_interaction", [False, True], ids=["tensor", "openai-proxy"])
@pytest.mark.parametrize(
    "rewards, accepted",
    [(ALL_CORRECT, False), (ALL_WRONG, False), (MIXED, True)],
    ids=["all-correct", "all-wrong", "mixed"],
)
def test_the_executor_drops_unanimous_groups_and_keeps_mixed_ones(
    rewards, accepted, as_interaction
):
    """The end-to-end decision, through the code that runs in training."""
    got, seen = run_through_executor(rewards, as_interaction)
    assert got is accepted
    assert tuple(seen[0]["rewards"].shape) == (G,)


def test_group_size_one_makes_the_filter_a_no_op():
    """gconfig.n_samples=1 leaves nothing for DAPO to reject -- a config trap, not a bug."""
    got, seen = run_through_executor([1.0])
    assert tuple(seen[0]["rewards"].shape) == (1,)
    assert got is True


def test_reward_normalization_does_not_change_any_decision():
    """``rewards`` may already be group-normalised; the decision must survive that.

    Driven through the real ``normalize_group_rewards``, because the whole reason a
    normalised group is safe to filter on is that the transform is affine per group.
    """
    for rewards, expected in ((ALL_CORRECT, False), (ALL_WRONG, False), (MIXED, True)):
        results = [{f"c{i}": interaction(r)} for i, r in enumerate(rewards)]
        for result in results:
            for it in result.values():
                it.to_tensor_dict()  # populate the cache normalize_group_rewards rewrites
        assert normalize_group_rewards(results) is True
        merged = {k: v for result in results for k, v in result.items()}
        traj = {
            "rewards": torch.cat([v.to_tensor_dict()["rewards"] for v in merged.values()])
        }
        assert dapo_dynamic_sampling(traj) is expected, rewards


# ---------------------------------------------------------------- rejection accounting ---


def test_accepted_and_rejected_are_counted_under_the_count_suffix():
    """The extra generation cost is read off these keys, so they have to be real.

    The bare ``rollout/rejected`` is a mean of ones and is always 1.0; only the ``__count``
    key carries the number of rejections.
    """
    stats_tracker.export_all()  # clear whatever earlier tests recorded
    for rewards in (ALL_CORRECT, ALL_WRONG, MIXED, MIXED):
        run_through_executor(rewards)
    stats = stats_tracker.export_all()
    assert stats["rollout/rejected__count"] == 2
    assert stats["rollout/accepted__count"] == 2
    assert stats["rollout/rejected"] == 1.0


def test_the_controller_forwards_the_rejection_count_to_the_trainer():
    """The cost of this baseline is read off the logs, so the count must survive the trip.

    Under the default single-controller layout the filter runs inside a rollout WORKER, and
    the trainer only ever sees what ``RolloutController.export_stats`` aggregates. That
    aggregator averages every key by its ``__count``, and ``rollout/rejected`` is a mean of
    ones -- 1.0 no matter how many groups DAPO threw away. Only the count is the measurement.
    """
    worker = {
        "rollout/accepted": 1.0,
        "rollout/accepted__count": 3,
        "rollout/rejected": 1.0,
        "rollout/rejected__count": 5,
    }

    class _TwoWorkers(RolloutController):
        def __init__(self):
            """Skip the real controller setup: only the stats aggregation is under test."""

        def _collective_rpc(self, method, http_timeout=None, **kwargs):
            """Stand in for the RPC that collects each worker's exported stats."""
            return [dict(worker), dict(worker)]

    stats = _TwoWorkers().export_stats()
    assert stats["rollout/rejected__count"] == 10
    assert stats["rollout/accepted__count"] == 6
    assert stats["rollout/rejected"] == 1.0, "the bare key is a mean and says nothing"


# ------------------------------------------------------------------------ regeneration ---


class _Version:
    def get_version(self) -> int:
        """Stay at version 0 so staleness never throttles the collector."""
        return 0


def collect(batch_size: int, dynamic_bs: bool) -> tuple[int, int, int]:
    """Run the real batch collector over an alternating stream of groups.

    Every other prompt is unanimous, so exactly half the groups are rejected.

    Args:
        batch_size: The batch the collector is asked for.
        dynamic_bs: The ``dynamic_bs`` flag under test.

    Returns:
        ``(n_returned, accepted, rejected)``.
    """
    stale = StalenessManager(_Version(), 64, batch_size, 8)
    ex = executor(staleness_manager=stale, consumer_batch_size=batch_size)
    disp = BatchTaskDispatcher(
        max_queue_size=4096,
        task_factory=ex._create_workflow_task,
        staleness_manager=stale,
        deterministic_order=True,
    )
    disp.initialize(logger=MagicMock())
    ids = itertools.count()

    def gen():
        for i in itertools.count():
            yield _RolloutTaskInput(
                task_id=next(ids),
                data={},
                workflow=grouped(ALL_CORRECT if i % 2 == 0 else MIXED),
                should_accept_fn=dapo_dynamic_sampling,
            )

    try:
        results = disp.active_submit_and_wait(gen(), batch_size, dynamic_bs=dynamic_bs)
    finally:
        disp.destroy()
    stats = stale.get_stats()
    return len(results), stats.accepted, stats.rejected


def test_rejected_groups_are_regenerated_under_the_shipped_dynamic_bs():
    """DAPO's oversampling. Half the groups are dropped, yet the batch still comes back full.

    ``GRPOConfig.dynamic_bs`` defaults to False, and that default is what makes this arm DAPO
    rather than GRPO-on-a-smaller-batch.
    """
    assert GRPOConfig().dynamic_bs is False
    n, accepted, rejected = collect(batch_size=8, dynamic_bs=False)
    assert n == 8, "a rejected group must be replaced, not merely skipped"
    assert accepted == 8
    assert rejected == 8, "half the stream is unanimous, so a full batch costs 16 rollouts"


def test_dynamic_bs_true_shrinks_the_batch_instead_of_refilling_it():
    """The opposite behaviour, pinned so nobody enables the flag believing it helps."""
    n, accepted, rejected = collect(batch_size=8, dynamic_bs=True)
    assert n == 4
    assert (accepted, rejected) == (4, 4)


# ------------------------------------------------------------------------- the default ---


def test_the_shipped_config_default_is_no_filter():
    """Every other experiment in the repo must be untouched by this baseline existing."""
    assert GRPOConfig().dynamic_filter_fn is None
    assert RemoteInfEngine._resolve_should_accept_fn(
        object.__new__(RemoteInfEngine), None
    ) is None


def test_the_default_accepts_a_group_this_baseline_would_reject():
    """``dynamic_filter_fn=None`` must leave the rollout path exactly as it was."""
    accepted, seen = run_through_executor(ALL_CORRECT, filter_fn=None)
    assert accepted is True
    assert seen == [], "no filter must be invoked at all"


def test_the_documented_import_path_resolves_to_this_function():
    """The config help text and launch scripts carry this string; a rename would strand it."""
    resolved = RemoteInfEngine._resolve_should_accept_fn(
        object.__new__(RemoteInfEngine), DAPO_PATH
    )
    assert resolved is dapo_dynamic_sampling
