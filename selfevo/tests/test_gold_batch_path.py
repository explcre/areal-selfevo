"""The gold batch-construction path, end to end, on CPU.

``test_gold_target_reachability.py`` established that an UNSOLVED group -- 25.5% of MATH
groups -- reaches no update under any router, that the MATH adapter throws the gold
derivation away, and that a gold cannot enter through the advantage tensor at all, because an
advantage multiplies tokens the model emitted and in such a group every one of them is wrong.
This file tests the path built in response: keep the gold, carry it to the batch, and
substitute it in as a ROW so the ordinary estimator acts on it.

Four things here are testable only because they were nearly missed:

1. **The collation trap.** ``check_trajectory_format`` only WARNS on a mismatched second dim
   and ``concat_padded_tensors`` pads per KEY, so an unpadded gold survives collation and
   breaks inside the engine one stage later. Both halves are asserted -- the padded gold
   packs, and the unpadded one is shown to reach packing before failing -- so the test is not
   passing on a trap that does not exist.
2. **The off-policy value.** ``selfevo/FINDINGS_loss_weighting.md`` measured that NaN in
   ``logprobs`` turns every advantage in a batch-normalised batch NaN, and that 0.0 silently
   shrinks the gold row by ``exp(prox_logp)``. The sentinel-and-reconcile protocol is driven
   through the REAL ``ppo_actor_loss_fn`` here, including the one-position coordinate shift
   between the two log-probability conventions, which is wrong in a way nothing raises.
3. **What the estimator sees afterwards.** A substituted group is no longer all-wrong, so the
   M20 ``unsolved_advantage`` path stops applying to it. Driven through the real
   ``PPOActor._compute_advantages`` rather than argued.
4. **Rollback.** With the flag off the adapter's output is byte-identical to a digest recorded
   before the change, and the workflow emits exactly its seven tensors.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging

import pytest

torch = pytest.importorskip("torch")

from areal.api.cli_args import (  # noqa: E402
    GenerationHyperparameters,
    GroupRoutingConfig,
    RejectionSamplingConfig,
)
from areal.utils.data import (  # noqa: E402
    TrajBatchMeta,
    concat_padded_tensors,
    pack_tensor_dict,
    split_padded_tensor_dict_into_mb_list,
)
from areal.utils.data import MicroBatchSpec  # noqa: E402
from areal.utils.functional import ppo_actor_loss_fn  # noqa: E402
from selfevo.gold import (  # noqa: E402
    GOLD_LOGP_SENTINEL,
    GoldAttachError,
    GoldLogprobPolicy,
    GoldMissingError,
    GoldOrderingError,
    GoldPolicyError,
    GoldRule,
    GoldShapeError,
    GoldStats,
    assert_gold_logprobs_filled,
    attach_gold,
    attach_gold_from_data,
    prompt_lengths,
    reconcile_gold_logprobs,
    substitute_gold_rows,
    substitute_in_place,
)
from selfevo.tests.test_group_routing import G, make_actor  # noqa: E402

logging.disable(logging.INFO)

MATH_DATASET = "DigitalLearningGmbH/MATH-lighteval"

# sha256 over (messages, answer) of every row of the adapted MATH train split, measured on
# 2026-09-01 with the adapter as it stood BEFORE keep_solution existed. The point of a
# recorded digest rather than a column-name check is that a column-name check passes on an
# adapter that has quietly changed what it puts IN the columns.
MATH_TRAIN_DIGEST = "dbe5c602f6ca651b71ea8367e49d96ac8f9f103681ab16acfc9dc10dc2255033"

# The keys RLVRWorkflow has always emitted, and must still emit alone when no gold is asked
# for.
SEVEN = {
    "input_ids",
    "loss_mask",
    "logprobs",
    "versions",
    "turn_ids",
    "attention_mask",
    "rewards",
}

T = 12
PROMPT = 4
GOLD = [901, 902, 903]


def make_row(
    reward: float,
    *,
    prompt_len: int = PROMPT,
    resp_len: int = 5,
    width: int = T,
) -> dict[str, torch.Tensor]:
    """One rollout row shaped exactly as a collated batch's row is.

    Args:
        reward: The row's raw reward.
        prompt_len: Tokens before the response.
        resp_len: Response tokens.
        width: Padded width.

    Returns:
        A dict of ``(1, width)`` tensors plus a ``(1,)`` reward, in the TOKEN coordinates a
        workflow emits.
    """
    n = prompt_len + resp_len
    loss_mask = torch.zeros(1, width, dtype=torch.int32)
    loss_mask[0, prompt_len:n] = 1
    attn = torch.zeros(1, width, dtype=torch.bool)
    attn[0, :n] = True
    logp = torch.zeros(1, width, dtype=torch.float32)
    logp[0, prompt_len:n] = -0.5
    versions = torch.full((1, width), -1, dtype=torch.int32)
    versions[0, prompt_len:n] = 0
    turn_ids = torch.full((1, width), -1, dtype=torch.int32)
    turn_ids[0, prompt_len:n] = 0
    return {
        "input_ids": torch.arange(1, width + 1, dtype=torch.int32).unsqueeze(0),
        "loss_mask": loss_mask,
        "logprobs": logp,
        "versions": versions,
        "turn_ids": turn_ids,
        "attention_mask": attn,
        "rewards": torch.tensor([reward], dtype=torch.float32),
    }


def make_group(
    rewards: list[float], gold: list[int] | None = GOLD, **row_kwargs
) -> dict[str, torch.Tensor]:
    """One GRPO group, collated the way ``GroupedRolloutWorkflow`` collates one.

    Args:
        rewards: One raw reward per rollout.
        gold: Gold token ids shared by the group, or None for a row with no gold.
        **row_kwargs: Forwarded to :func:`make_row`.

    Returns:
        A ``(len(rewards), T)`` batch dict carrying the gold keys.
    """
    rows = [attach_gold(make_row(r, **row_kwargs), gold) for r in rewards]
    return concat_padded_tensors(rows)


def make_batch(groups: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Several groups collated into one batch, as ``concat_batch`` does."""
    return concat_padded_tensors(groups)


# ------------------------------------------------------------------------- the dataset ---


@pytest.fixture(scope="module")
def math_tokenizer():
    """A real tokenizer, or a skip. Any tokenizer will do; the ids are never compared."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as exc:  # pragma: no cover - box without the model cached
        pytest.skip(f"no tokenizer available here: {exc}")


def _math_dataset(**kwargs):
    """The adapted MATH train split, or a skip if the dataset is not on this box."""
    pytest.importorskip("datasets")
    import datasets

    from areal.dataset.competition_math import get_math_rl_dataset

    try:
        datasets.load_dataset(path=MATH_DATASET, split="train[:1]")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"{MATH_DATASET} not available here: {exc}")
    return get_math_rl_dataset(
        path=MATH_DATASET, split="train", tokenizer=None, max_length=None, **kwargs
    )


def test_the_adapter_default_is_byte_identical_to_before_the_flag_existed():
    """Rollback, asserted against a digest and not against a column list.

    A column-name check passes on an adapter that kept its columns and changed their
    contents. This hashes every ``messages``/``answer`` pair of all 7500 rows against the
    value measured before ``keep_solution`` was added, so any change to what the default path
    produces fails here rather than in a run.
    """
    adapted = _math_dataset()
    assert set(adapted.column_names) == {"messages", "answer"}, adapted.column_names
    h = hashlib.sha256()
    for row in adapted:
        h.update(
            json.dumps(
                [row["messages"], row["answer"]], sort_keys=True, ensure_ascii=False
            ).encode()
        )
    assert h.hexdigest() == MATH_TRAIN_DIGEST


def test_keep_solution_adds_a_tokenised_gold_and_changes_nothing_else(math_tokenizer):
    """The flag adds exactly one column and leaves the other two alone.

    Measured over the full split when this was written: median gold length 163 tokens
    (162 + EOS), max 2495, none empty, 99.13% at or under 1024. A row's gold ends with the
    tokenizer's EOS, without which the gold row would be the only row in the batch that never
    terminates.
    """
    pytest.importorskip("datasets")
    from areal.dataset.competition_math import get_math_rl_dataset

    try:
        gold_ds = get_math_rl_dataset(
            path=MATH_DATASET,
            split="train[:64]",
            tokenizer=math_tokenizer,
            max_length=None,
            keep_solution=True,
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"{MATH_DATASET} not available here: {exc}")
    assert set(gold_ds.column_names) == {"messages", "answer", "gold_ids"}
    lens = [len(g) for g in gold_ds["gold_ids"]]
    assert min(lens) > 1, lens
    assert all(g[-1] == math_tokenizer.eos_token_id for g in gold_ds["gold_ids"])

    plain = get_math_rl_dataset(
        path=MATH_DATASET, split="train[:64]", tokenizer=math_tokenizer, max_length=None
    )
    assert plain["answer"] == gold_ds["answer"]
    assert plain["messages"] == gold_ds["messages"]


def test_keep_solution_without_a_tokenizer_is_refused():
    """A gold column of empty lists is a gold arm that trains on nothing while reporting one.

    So the adapter refuses rather than producing it. Uses no dataset: the guard is checked
    before anything is loaded, which is the point -- it fails at configuration time, not
    after a download.
    """
    from areal.dataset.competition_math import get_math_rl_dataset

    with pytest.raises(ValueError, match="keep_solution=True needs a tokenizer"):
        get_math_rl_dataset(
            path=MATH_DATASET, split="train", tokenizer=None, keep_solution=True
        )


def test_the_gold_template_is_applied_and_is_not_a_constant(math_tokenizer):
    """The template seam exists because no default is right for every chat template.

    The live 30B model's prompt ends inside an open ``<think>`` block, so a gold spliced in
    without closing it trains the model to answer somewhere it never emits. This asserts the
    template actually reaches the tokeniser, which is the difference between a configurable
    seam and a documented intention.
    """
    pytest.importorskip("datasets")
    from areal.dataset.competition_math import get_math_rl_dataset

    try:
        plain = get_math_rl_dataset(
            path=MATH_DATASET,
            split="train[:8]",
            tokenizer=math_tokenizer,
            keep_solution=True,
        )
        templated = get_math_rl_dataset(
            path=MATH_DATASET,
            split="train[:8]",
            tokenizer=math_tokenizer,
            keep_solution=True,
            gold_template="\n</think>\n\n{solution}",
        )
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"{MATH_DATASET} not available here: {exc}")
    for a, b in zip(plain["gold_ids"], templated["gold_ids"]):
        assert len(b) > len(a), (len(a), len(b))
        assert b[-1] == a[-1]


# -------------------------------------------------------------------------- attachment ---


def test_the_gold_is_padded_to_the_trajectory_width_and_left_aligned():
    """Shape, alignment and mask, which together are what makes collation safe."""
    traj = attach_gold(make_row(0.0), GOLD)
    assert traj["gold_ids"].shape == traj["input_ids"].shape
    assert traj["gold_mask"].shape == traj["input_ids"].shape
    assert traj["gold_ids"][0, : len(GOLD)].tolist() == GOLD
    assert traj["gold_ids"][0, len(GOLD) :].tolist() == [0] * (T - len(GOLD))
    assert int(traj["gold_mask"].sum()) == len(GOLD)


def test_a_row_with_no_gold_still_carries_the_keys():
    """An absent column is not a degradation here, it is a crash.

    ``concat_padded_tensors`` raises when the dicts in a batch disagree on their key set, so a
    row whose dataset entry has no solution must carry empty gold rather than none -- and the
    loss of reach is then counted at the substitution seam instead of killing the step.
    """
    traj = attach_gold(make_row(0.0), None)
    assert int(traj["gold_mask"].sum()) == 0
    assert set(traj) - SEVEN == {"gold_ids", "gold_mask"}
    batch = concat_padded_tensors([traj, attach_gold(make_row(1.0), GOLD)])
    assert batch["gold_mask"].shape == batch["input_ids"].shape


def test_a_gold_longer_than_the_row_is_refused_rather_than_truncated():
    """A truncated derivation is a wrong target that still looks like a target."""
    with pytest.raises(GoldAttachError, match="Refusing to truncate"):
        attach_gold(make_row(0.0), list(range(T + 1)))


def test_the_workflow_side_helper_downgrades_a_too_long_gold_to_an_empty_one():
    """The rollout worker must not die of a long solution meeting a short rollout.

    ``attach_gold`` refuses; ``attach_gold_from_data`` converts that one refusal into an empty
    gold, because it runs inside a rollout and the honest response there is to lose the reach
    and count it, not to fail the episode. Every other refusal still propagates.
    """
    traj = attach_gold_from_data(make_row(0.0), {"gold_ids": list(range(T + 1))})
    assert int(traj["gold_mask"].sum()) == 0
    assert attach_gold_from_data(make_row(0.0), {}) .keys() == make_row(0.0).keys()
    assert attach_gold_from_data(None, {"gold_ids": GOLD}) is None


def test_a_row_with_no_response_reports_its_real_length_as_the_prompt_boundary():
    """A rollout that emitted nothing has no first response token, and 0 is the wrong answer.

    ``loss_mask.argmax`` returns 0 on an all-zero row, so without the fallback the gold would
    be spliced at position 0 and overwrite the prompt -- a target detached from its question,
    written into the one row least able to survive it. The case is reachable: sglang's abort
    path returns ``output_tokens=[]`` (``sglang_remote.py``, "Abort before prefill"), and such
    a row reaches collation like any other.

    Found by mutation: replacing the fallback with a bare ``argmax`` survived the suite until
    this test existed.
    """
    empty = make_row(0.0, resp_len=0)
    assert int(empty["loss_mask"].sum()) == 0
    assert int(prompt_lengths(empty["loss_mask"], empty["attention_mask"])[0]) == PROMPT
    # And with no attention_mask to fall back on, the honest answer is the full width rather
    # than a position inside the prompt.
    assert int(prompt_lengths(empty["loss_mask"])[0]) == T

    group = concat_padded_tensors(
        [attach_gold(make_row(0.0, resp_len=0), GOLD)]
        + [attach_gold(make_row(0.0), GOLD) for _ in range(3)]
    )
    out, stats = substitute_gold_rows(group, "dyme")
    assert stats.rows_substituted == 1
    assert torch.equal(out["input_ids"][0, :PROMPT], group["input_ids"][0, :PROMPT])
    assert out["input_ids"][0, PROMPT : PROMPT + len(GOLD)].tolist() == GOLD


def test_a_list_of_group_sizes_that_does_not_sum_to_the_batch_is_refused():
    """The uniform-int path and the explicit-list path are two different checks.

    ``group_sizes=3`` on four rows is caught by the divisibility test; a LIST whose entries do
    not sum to the batch reaches the final guard instead, and until this test existed that
    guard was unconstrained -- mutation replaced it with ``if False`` and the suite passed. A
    wrong partition is not cosmetic: it makes the rule read rewards across unrelated prompts,
    so the all-wrong groups it then finds are an artifact of the grouping.
    """
    batch = make_batch([make_group([0.0] * 4), make_group([0.0] * 4)])
    with pytest.raises(GoldShapeError, match="do not partition"):
        substitute_gold_rows(batch, "dyme", group_sizes=[4, 3])
    with pytest.raises(GoldShapeError, match="do not partition"):
        substitute_gold_rows(batch, "dyme", group_sizes=[8, 0])
    # The valid list still works, so this is not a blanket refusal of the list form.
    _, stats = substitute_gold_rows(batch, "dyme", group_sizes=[4, 4])
    assert stats.n_groups == 2 and stats.rows_substituted == 2


# ------------------------------------------------------------------ the collation trap ---


def _pack(batch: dict) -> dict:
    """Collate-then-pack, i.e. what the engine does to a batch before the loss."""
    return pack_tensor_dict(batch)


def test_a_gold_carrying_batch_survives_collation_packing_and_microbatching():
    """The trap, closed. Every gold tensor is packed like its neighbours.

    ``pack_tensor_dict`` packs a tensor only when ``shape[1] == seq_len``, so the assertion
    that matters is that ``gold_ids`` comes out 1-D with the same total length as
    ``input_ids`` -- if it were still 2-D it would have silently skipped packing.
    """
    batch = make_batch([make_group([0.0] * 4), make_group([1.0, 0.0, 1.0, 0.0])])
    sub, _ = substitute_gold_rows(batch, GoldRule.DYME, group_sizes=4)
    packed = _pack(sub)
    for key in ("input_ids", "gold_ids", "gold_mask", "is_gold", "loss_mask"):
        assert packed[key].ndim == 1, (key, packed[key].shape)
        assert packed[key].shape == packed["input_ids"].shape, key
    mbs = split_padded_tensor_dict_into_mb_list(sub, mb_spec=MicroBatchSpec(n_mbs=2))
    assert len(mbs.mbs) == 2
    for mb in mbs.mbs:
        p = pack_tensor_dict(mb)
        assert p["is_gold"].shape == p["input_ids"].shape


def test_an_unpadded_gold_would_have_survived_collation_and_broken_at_packing():
    """Anti-vacuity for the test above: the trap is real, and it is silent at collation.

    Without this, "the padded gold packs" says nothing -- it would pass just as well if
    collation rejected mismatched widths, in which case nothing needed padding at
    construction. Here the mismatched tensor passes ``concat_padded_tensors`` unremarked and
    arrives at packing as a 2-D tensor while every sibling is 1-D, which is the shape error
    that lands inside the engine, one stage after the mistake.
    """
    rows = []
    for r in (0.0, 1.0):
        row = make_row(r)
        row["gold_ids"] = torch.tensor(GOLD, dtype=torch.int32).unsqueeze(0)
        rows.append(row)
    batch = concat_padded_tensors(rows)
    assert batch["gold_ids"].shape == (2, len(GOLD))
    assert batch["input_ids"].shape == (2, T)
    packed = pack_tensor_dict(batch)
    assert packed["gold_ids"].ndim == 2, "the unpadded gold silently skipped packing"
    assert packed["input_ids"].ndim == 1


def test_check_trajectory_format_only_warns_about_the_mismatch():
    """The reason the trap is silent: the format check does not refuse it.

    Pinned because the whole padding decision rests on it. If this ever becomes an exception,
    padding at construction stops being the only defence and this file should say so.
    """
    from areal.infra.workflow_executor import check_trajectory_format

    traj = make_row(0.0)
    traj["gold_ids"] = torch.tensor(GOLD, dtype=torch.int32).unsqueeze(0)
    assert check_trajectory_format(traj) is True


# --------------------------------------------------------------------- the workflow off ---


def _one(*args, **kwargs) -> float:
    """A reward of 1.0, defined at module level because it is pickled.

    ``AsyncRewardWrapper`` hands the reward function to a ``ProcessPoolExecutor``, so a
    lambda or a closure fails to pickle and the workflow retries four times and returns 0.0 --
    a failure that looks like a scoring result rather than a broken test.
    """
    return 1.0


def _prompt_ids(data, tokenizer, enable_thinking) -> list[int]:
    """A fixed prompt, so two episodes are comparable token for token."""
    return [5, 6, 7]


def _identity(data):
    """The workflow's prompt extractor, bypassing the chat template."""
    return data


class _StubTokenizer:
    """The smallest tokenizer ``RLVRWorkflow`` actually uses.

    A real one would be a second-long import for a test that never inspects a token, and
    would tie this file to a model being cached on the box.
    """

    pad_token_id = 0
    eos_token_id = 1

    def decode(self, ids, **kwargs):
        """Text for the reward function, which this file's reward ignores."""
        return " ".join(str(int(i)) for i in ids)


class _StubEngine:
    """An inference engine that returns one fixed response."""

    async def agenerate(self, req):
        """The same three output tokens every time, so trajectories are comparable."""
        from areal.api.io_struct import ModelResponse

        return ModelResponse(
            input_tokens=list(req.input_ids),
            output_tokens=[11, 12, 13],
            output_logprobs=[-0.1, -0.2, -0.3],
            output_versions=[0, 0, 0],
            stop_reason="stop",
        )


def _run_episode(data):
    """Drive the real ``RLVRWorkflow.arun_episode`` on CPU.

    Args:
        data: The dataset row, with or without ``gold_ids``.

    Returns:
        The trajectory dict.
    """
    from areal.workflow.rlvr import RLVRWorkflow

    wf = RLVRWorkflow(
        reward_fn=_one,
        gconfig=GenerationHyperparameters(max_new_tokens=4),
        tokenizer=_StubTokenizer(),
        get_input_ids_fn=_prompt_ids,
        data_extract_prompt_fn=_identity,
    )
    return asyncio.run(wf.arun_episode(_StubEngine(), data))


def test_the_seven_tensor_path_is_unchanged_when_no_gold_is_asked_for():
    """Rollback at the workflow: same keys, same bytes.

    Two assertions, because either alone is weak. The key set catches a gold tensor appearing
    unconditionally; the tensor comparison catches the seven being built differently on the
    way to adding an eighth.
    """
    plain = _run_episode({"messages": [{"role": "user", "content": "hi"}]})
    assert set(plain) == SEVEN, sorted(set(plain) ^ SEVEN)
    again = _run_episode({"messages": [{"role": "user", "content": "hi"}]})
    for key in SEVEN:
        assert torch.equal(plain[key], again[key]), key


def test_the_workflow_emits_the_gold_only_when_the_dataset_row_carries_it():
    """And the seven are bit-identical to the no-gold run, so gold is purely additive."""
    plain = _run_episode({"messages": [{"role": "user", "content": "hi"}]})
    withgold = _run_episode(
        {"messages": [{"role": "user", "content": "hi"}], "gold_ids": GOLD}
    )
    assert set(withgold) - SEVEN == {"gold_ids", "gold_mask"}
    for key in SEVEN:
        assert torch.equal(plain[key], withgold[key]), key
    assert withgold["gold_ids"].shape == withgold["input_ids"].shape
    assert int(withgold["gold_mask"].sum()) == len(GOLD)


def test_the_old_reachability_guard_no_longer_covers_this_supplier():
    """A blind spot recorded rather than left to be discovered.

    ``test_gold_target_reachability.py::test_gold_is_absent_at_both_ends_or_present_at_both``
    reads the string keys of dict LITERALS inside ``arun_episode`` and requires that a gold in
    the schema coexist with an apply seam able to accept a target. The gold now reaches the
    trajectory through a function call rather than a literal, so that guard still passes --
    and it should, because its "consumable" half asks about ``group_apply``, which is
    precisely the seam this path establishes gold CANNOT use. This test states both halves so
    the guard's reduced scope is a recorded fact and not an accident.
    """
    import ast
    from pathlib import Path

    import areal.workflow.rlvr as rlvr_mod

    tree = ast.parse(Path(rlvr_mod.__file__).read_text())
    literal_keys = {
        k.value
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "arun_episode"
        for inner in ast.walk(node)
        if isinstance(inner, ast.Dict)
        for k in inner.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
    assert not any("gold" in k for k in literal_keys), sorted(literal_keys)
    traj = _run_episode(
        {"messages": [{"role": "user", "content": "hi"}], "gold_ids": GOLD}
    )
    assert "gold_ids" in traj


def test_the_executor_attaches_the_gold_for_every_workflow():
    """The seam that serves the OpenAI-proxy path, checked at the source.

    The live MATH runs do not use ``RLVRWorkflow``; they use ``MathAgent`` behind
    ``OpenAIProxyWorkflow``, whose tensors are assembled in
    ``InteractionWithTokenLogpReward.to_tensor_dict``. ``WorkflowExecutor`` is where the two
    ways of building a trajectory converge, so the attach call lives there and both are served
    by one tested function.

    Read from the source rather than driven, because driving it needs an inference engine and
    this file runs on CPU; the LOGIC it calls is driven directly by the tests above, so what is
    checked here is only that the call exists and is guarded. That is a weaker statement than
    the rest of this file makes, and it is stated as such: a defect INSIDE the guard would not
    be caught here.
    """
    import ast
    from pathlib import Path

    import areal.infra.workflow_executor as exec_mod

    src = Path(exec_mod.__file__).read_text()
    tree = ast.parse(src)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "attach_gold_from_data"
    ]
    assert len(calls) == 1, f"{len(calls)} attach calls in {exec_mod.__file__}"
    assert '"gold_ids" in pending_task.data' in src, "the attach call is unguarded"


# ------------------------------------------------------------------------ substitution ---


def test_the_none_rule_returns_the_batch_untouched():
    """The off arm is a true no-op: same tensors, no ``is_gold``, no guard.

    ``is`` and not ``torch.equal``: a copy that happens to be equal would still be a change to
    what the off arm hands the trainer, and this path must not even allocate.
    """
    batch = make_batch([make_group([0.0] * 4)])
    out, stats = substitute_gold_rows(batch, "none", group_sizes=4)
    assert "is_gold" not in out
    for key, value in batch.items():
        assert out[key] is value, key
    assert stats.rows_substituted == 0 and stats.groups_qualifying == 0
    assert stats.loss_tokens == int(batch["loss_mask"].count_nonzero())


def test_dyme_substitutes_the_first_row_of_an_all_wrong_group():
    """The batch half of DyME's rule, one assertion per tensor it rewrites."""
    batch = make_batch([make_group([0.0] * 4), make_group([1.0, 0.0, 1.0, 0.0])])
    out, stats = substitute_gold_rows(batch, GoldRule.DYME, group_sizes=4)

    assert stats.groups_qualifying == 1
    assert stats.rows_substituted == 1
    assert stats.substituted_rows == (0,)
    assert stats.qualifying_groups == (0,)
    assert stats.gold_tokens == len(GOLD)

    assert out["input_ids"][0, PROMPT : PROMPT + len(GOLD)].tolist() == GOLD
    assert out["loss_mask"][0].tolist() == [0] * PROMPT + [1] * len(GOLD) + [0] * (
        T - PROMPT - len(GOLD)
    )
    assert out["attention_mask"][0].tolist() == [True] * (PROMPT + len(GOLD)) + [
        False
    ] * (T - PROMPT - len(GOLD))
    assert float(out["rewards"][0]) == 1.0
    assert out["is_gold"][0].tolist() == [1] * T
    # Untouched rows stay bit-identical, including the other group.
    for row in range(1, 8):
        assert torch.equal(out["input_ids"][row], batch["input_ids"][row]), row
        assert out["is_gold"][row].sum() == 0, row


def test_the_prompt_is_kept_and_only_the_response_is_replaced():
    """A target detached from its question teaches the derivation unconditionally."""
    batch = make_group([0.0] * 4)
    out, _ = substitute_gold_rows(batch, GoldRule.DYME)
    assert torch.equal(out["input_ids"][0, :PROMPT], batch["input_ids"][0, :PROMPT])
    assert int(out["loss_mask"][0, :PROMPT].sum()) == 0


def test_is_gold_is_per_token_so_it_survives_packing():
    """A ``(B,)`` tensor does not survive this pipeline, which is how the first routed run died.

    Asserted through the real packing rather than by reading the shape, because the shape
    alone does not say whether the pipeline keeps it aligned with ``input_ids``.
    """
    batch = make_group([0.0] * 4)
    out, _ = substitute_gold_rows(batch, GoldRule.DYME)
    assert out["is_gold"].shape == out["input_ids"].shape
    packed = pack_tensor_dict(out)
    assert packed["is_gold"].shape == packed["input_ids"].shape
    lens = out["attention_mask"].sum(-1).tolist()
    assert int(packed["is_gold"][: lens[0]].sum()) == lens[0]


def test_dyme_and_lspo_agree_on_binary_rewards_and_diverge_on_signed_ones():
    """The two predicates are not aliases, and the difference is the paper's own definition.

    LSPO's cliff is ``sum_k R = 0``; DyME's is ``no rollout scored above 0.5``. On rewards in
    {0, 1} those select the same groups, which is why aliasing them would go unnoticed. Give
    the group a ``[-1, +1]`` and they part company: LSPO calls it a cliff, DyME sees a correct
    sample and declines. Whichever is right is the paper's business; conflating them is not.
    """
    binary = make_batch([make_group([0.0] * 4), make_group([1.0, 1.0, 0.0, 0.0])])
    d_out, d_stats = substitute_gold_rows(binary, "dyme", group_sizes=4)
    l_out, l_stats = substitute_gold_rows(binary, "lspo_cliff", group_sizes=4)
    assert d_stats.qualifying_groups == l_stats.qualifying_groups == (0,)
    assert torch.equal(d_out["input_ids"], l_out["input_ids"])

    signed = make_group([-1.0, 1.0, -1.0, 1.0])
    _, l_signed = substitute_gold_rows(signed, "lspo_cliff")
    assert l_signed.rows_substituted == 1
    _, d_signed = substitute_gold_rows(signed, "dyme")
    assert d_signed.groups_qualifying == 0 and d_signed.rows_substituted == 0


def test_a_group_with_a_correct_sample_is_left_alone_by_both_rules():
    """Anti-vacuity: the predicates say no to something, and saying no is not a refusal.

    A batch in which no group qualifies is a batch that needed no gold, so the reach guard
    stays quiet and the rows come back untouched. Distinguishing that from "gold was needed
    and did not land" is the entire job of the guard, and an over-eager version of it would
    fail every step whose prompts were all solved.
    """
    batch = make_group([1.0, 0.0, 0.0, 0.0])
    for rule in ("dyme", "lspo_cliff"):
        out, stats = substitute_gold_rows(batch, rule)
        assert stats.groups_qualifying == 0 and stats.rows_substituted == 0, rule
        assert int(out["is_gold"].sum()) == 0, rule
        assert torch.equal(out["input_ids"], batch["input_ids"]), rule


# ------------------------------------------------------------------------- reach guard ---


def test_a_batch_with_no_gold_keys_is_refused():
    """The flag on at one end and off at the other is a configuration error, not a no-op."""
    batch = concat_padded_tensors([make_row(0.0), make_row(0.0)])
    with pytest.raises(GoldMissingError, match="carries no gold_ids"):
        substitute_gold_rows(batch, "dyme")


def test_a_batch_whose_gold_is_all_empty_is_refused():
    """A gold arm that trains on no gold must not look like a gold arm that ran."""
    batch = make_group([0.0] * 4, gold=None)
    with pytest.raises(GoldMissingError, match="every gold_mask in this batch is empty"):
        substitute_gold_rows(batch, "dyme")


def test_a_qualifying_group_whose_gold_does_not_fit_is_counted_and_refused():
    """Length is the one reason a well-formed gold arm can still reach nothing.

    The refusal names both causes separately, because a missing solution is a dataset problem
    and a gold that will not fit is a sequence-length one, and a single "skipped" count would
    send the reader to the wrong fix.
    """
    long_gold = list(range(901, 901 + T - 1))
    batch = make_group([0.0] * 4, gold=long_gold)
    with pytest.raises(GoldMissingError, match="did not fit"):
        substitute_gold_rows(batch, "dyme")


def test_reach_counts_on_a_fixture_batch():
    """The numbers a run would log, on a batch built to exercise every counter at once.

    Four groups of four: one all-wrong with a usable gold, one all-wrong whose gold is empty,
    one all-wrong whose gold is too long, and one solved. So 3 qualify, 1 is served, and the
    two loss-of-reach counts are 1 each -- which is the shape of report that distinguishes "no
    gold was needed" from "gold was needed three times and landed once".
    """
    batch = make_batch(
        [
            make_group([0.0] * 4),
            make_group([0.0] * 4, gold=None),
            make_group([0.0] * 4, gold=list(range(901, 901 + T - 1))),
            make_group([1.0] * 4),
        ]
    )
    out, stats = substitute_gold_rows(batch, "dyme", group_sizes=4)
    assert (stats.n_groups, stats.n_rows) == (4, 16)
    assert stats.groups_qualifying == 3
    assert stats.rows_substituted == 1
    assert stats.groups_no_gold == 1
    assert stats.groups_no_fit == 1
    assert stats.gold_tokens == len(GOLD)
    assert stats.substituted_rows == (0,)
    m = stats.as_metrics()
    assert m["gold/groups_qualifying"] == 3.0
    assert m["gold/rows_substituted"] == 1.0
    assert m["gold/tokens"] == float(len(GOLD))
    assert m["gold/qualifying_group_fraction"] == pytest.approx(0.75)


def test_token_mass_is_reported_and_is_not_the_row_fraction():
    """The quantity the loss actually reads, which no ``route/*`` key supplies.

    The objective is one per-token mean over the global batch, so a gold row's share of the
    update is proportional to its TOKEN count, not to its being one row of sixteen. Here the
    gold is 3 tokens of a batch whose masked total is 16 rows' worth of response, and the two
    fractions differ -- which is the whole reason the key exists.
    """
    batch = make_batch([make_group([0.0] * 4), make_group([1.0, 0.0, 1.0, 0.0])])
    out, stats = substitute_gold_rows(batch, "dyme", group_sizes=4)
    expected_total = int(out["loss_mask"].count_nonzero())
    assert stats.loss_tokens == expected_total
    assert stats.token_mass == pytest.approx(len(GOLD) / expected_total)
    row_fraction = stats.rows_substituted / stats.n_rows
    assert stats.token_mass != pytest.approx(row_fraction)
    assert stats.as_metrics()["gold/token_mass"] == pytest.approx(stats.token_mass)


def test_token_mass_tracks_the_gold_length():
    """Doubling the gold doubles the mass at fixed row count, which is the audit's finding."""
    short = make_group([0.0] * 4, gold=[901, 902])
    long = make_group([0.0] * 4, gold=[901, 902, 903, 904])
    _, s_short = substitute_gold_rows(short, "dyme")
    _, s_long = substitute_gold_rows(long, "dyme")
    assert s_short.rows_substituted == s_long.rows_substituted == 1
    assert s_long.gold_tokens == 2 * s_short.gold_tokens


def test_substituting_after_compute_logp_is_refused():
    """The quietest corruption available on this path, so it is a typed refusal.

    ``prox_logp`` describes the tokens that were in the batch when it was computed. Replace a
    row afterwards and it keeps its shape and loses its meaning, with nothing downstream to
    notice.
    """
    batch = make_group([0.0] * 4)
    batch["prox_logp"] = torch.zeros_like(batch["logprobs"])
    with pytest.raises(GoldOrderingError, match="Substitute BEFORE compute_logp"):
        substitute_gold_rows(batch, "dyme")


def test_a_grouping_that_does_not_partition_the_batch_is_refused():
    """A wrong partition invents all-wrong groups out of unrelated prompts."""
    batch = make_batch([make_group([0.0] * 4)])
    with pytest.raises(GoldShapeError, match="does not divide"):
        substitute_gold_rows(batch, "dyme", group_sizes=3)


def test_unknown_rules_and_policies_are_refused_by_name():
    """A typo in a config must not silently select the off arm."""
    batch = make_group([0.0] * 4)
    with pytest.raises(GoldPolicyError, match="unknown gold rule"):
        substitute_gold_rows(batch, "dime")
    with pytest.raises(GoldPolicyError, match="unknown gold logprob policy"):
        substitute_gold_rows(batch, "dyme", logprob_policy="nan")


def test_the_input_batch_is_never_mutated():
    """Purity, asserted rather than intended."""
    batch = make_group([0.0] * 4)
    before = {k: v.clone() for k, v in batch.items()}
    substitute_gold_rows(batch, "dyme")
    for key, value in before.items():
        assert torch.equal(batch[key], value), key


def test_the_list_form_sums_the_counts_across_groups():
    """The shape the trainer actually holds: one dict per prompt out of ``prepare_batch``."""
    trajs = [make_group([0.0] * 4), make_group([1.0] * 4), make_group([0.0] * 4)]
    out, stats = substitute_in_place(trajs, "dyme")
    assert stats.n_groups == 3 and stats.n_rows == 12
    assert stats.groups_qualifying == 2 and stats.rows_substituted == 2
    assert stats.substituted_rows == (0, 8)
    assert stats.gold_tokens == 2 * len(GOLD)
    assert all(torch.equal(a["input_ids"], b["input_ids"]) for a, b in zip(trajs, out)) is False


def test_the_list_form_tolerates_a_solved_prompt_but_not_a_whole_dead_batch():
    """A solved prompt is the ordinary case; a batch that needed gold and got none is not.

    Three cases, and the middle one is why the guard is applied across the LIST rather than
    per element: one prompt whose group qualifies and cannot be served must not fail a step
    on its own, but a batch in which every qualifying group went unserved must.
    """
    ok, stats = substitute_in_place(
        [make_group([1.0] * 4), make_group([0.0] * 4)], "dyme"
    )
    assert int(ok[1]["is_gold"].sum()) > 0
    assert stats.groups_qualifying == 1 and stats.rows_substituted == 1

    served, mixed = substitute_in_place(
        [make_group([0.0] * 4, gold=None), make_group([0.0] * 4)], "dyme"
    )
    assert mixed.groups_qualifying == 2 and mixed.rows_substituted == 1
    assert mixed.groups_no_gold == 1

    with pytest.raises(GoldMissingError):
        substitute_in_place(
            [make_group([0.0] * 4, gold=None), make_group([0.0] * 4, gold=None)], "dyme"
        )

    nothing_needed, quiet = substitute_in_place(
        [make_group([1.0] * 4), make_group([1.0] * 4)], "dyme"
    )
    assert quiet.groups_qualifying == 0 and quiet.rows_substituted == 0


# ------------------------------------------------------------- the off-policy log-probs ---


def test_the_gold_logprobs_are_finite_and_impossible_rather_than_nan():
    """The audit's rule, in the two properties that make the sentinel safe and detectable.

    Finite, because ``_compute_advantages`` reads this tensor for the KL reward at
    ``actor.py:741`` and ``-0.0 * NaN`` is NaN, which under the live batch-level ``adv_norm``
    was measured turning all 8 of 8 rows NaN. Positive, because a log-probability never is, so
    an unfilled row cannot be mistaken for a filled one.
    """
    batch = make_group([0.0] * 4)
    out, _ = substitute_gold_rows(batch, "dyme")
    gold = out["logprobs"][0, PROMPT : PROMPT + len(GOLD)]
    assert torch.isfinite(out["logprobs"]).all()
    assert (gold == GOLD_LOGP_SENTINEL).all()
    assert GOLD_LOGP_SENTINEL > 0


def test_an_unreconciled_gold_row_is_refused_before_the_loss():
    """The silent path the audit measured: ``exp(prox_logp - 1)`` on the one row that matters."""
    batch = make_group([0.0] * 4)
    out, _ = substitute_gold_rows(batch, "dyme")
    with pytest.raises(GoldOrderingError, match="never reconciled"):
        assert_gold_logprobs_filled(out)


def test_a_batch_with_no_gold_passes_the_guard_trivially():
    """So a caller can call it unconditionally, which is the only way it gets called."""
    assert_gold_logprobs_filled(make_group([0.0] * 4))
    assert_gold_logprobs_filled({})


def test_reconcile_sets_the_behaviour_logprob_to_the_proximal_one_with_the_right_shift():
    """The coordinate shift, which is wrong in a way that raises nothing.

    ``logprobs`` is in TOKEN coordinates and ``prox_logp`` in EMITTER coordinates, and
    ``_compute_advantages`` rolls the former LEFT by one to compare them. So what is written
    here must be ``prox_logp`` rolled RIGHT by one, and the check is the round trip: after the
    actor's own roll, the two tensors must agree exactly on the gold tokens.
    """
    batch = make_group([0.0] * 4)
    out, _ = substitute_gold_rows(batch, "dyme")
    prox = torch.linspace(-3.0, -0.1, T).unsqueeze(0).repeat(4, 1)
    out["prox_logp"] = prox
    fixed, n_rows = reconcile_gold_logprobs(out)
    assert n_rows == 1
    assert_gold_logprobs_filled(fixed)

    # The actor's own reconciliation, verbatim: loss_mask and logprobs are both rolled LEFT
    # by one into emitter coordinates, and the loss compares them there. So the positions to
    # check are the rolled ones -- checking at the token-coordinate gold positions compares
    # the last gold token against a padding entry, which is exactly the off-by-one this
    # function exists to get right.
    old_logp = torch.roll(fixed["logprobs"], shifts=-1, dims=-1)
    emitter_gold = torch.roll(fixed["is_gold"].bool() & fixed["loss_mask"].bool(),
                              shifts=-1, dims=-1)
    assert int(emitter_gold.sum()) == len(GOLD)
    assert torch.equal(old_logp[emitter_gold], prox[emitter_gold])
    # And the non-gold rows are untouched, so this cannot be a blanket overwrite.
    assert torch.equal(fixed["logprobs"][1:], out["logprobs"][1:])
    # Anti-vacuity: an UNSHIFTED write would disagree here, so the assertion above is a
    # statement about the shift and not about the two tensors happening to be equal.
    unshifted = torch.where(
        fixed["is_gold"].bool() & fixed["loss_mask"].bool(), prox, out["logprobs"]
    )
    assert not torch.equal(
        torch.roll(unshifted, shifts=-1, dims=-1)[emitter_gold], prox[emitter_gold]
    )


def test_reconcile_without_prox_logp_is_refused():
    """The one state that cannot be repaired, so it must not be run."""
    batch = make_group([0.0] * 4)
    out, _ = substitute_gold_rows(batch, "dyme")
    with pytest.raises(GoldOrderingError, match="prox_logp is absent"):
        reconcile_gold_logprobs(out)


def test_ratio_one_is_offered_and_refuses_to_pretend_it_worked():
    """The swappable alternative, kept honest.

    ``ratio_one`` writes a plain 0.0 and asserts a ratio only the loss can enforce; nothing in
    ``grpo_loss_fn`` reads ``is_gold`` today, so reconciling it would be a no-op that looks
    like a fix. It refuses instead.
    """
    batch = make_group([0.0] * 4)
    out, _ = substitute_gold_rows(batch, "dyme", logprob_policy="ratio_one")
    assert float(out["logprobs"][0, PROMPT]) == 0.0
    assert_gold_logprobs_filled(out)  # 0.0 is a legal log-probability, so the guard passes
    out["prox_logp"] = torch.zeros_like(out["logprobs"])
    with pytest.raises(GoldPolicyError, match="only the loss can enforce"):
        reconcile_gold_logprobs(out, logprob_policy="ratio_one")


def test_a_reconciled_gold_row_gets_importance_weight_exactly_one_in_the_real_loss():
    """Driven through ``ppo_actor_loss_fn`` with the live rejection sampling configuration.

    This is the property the whole two-phase protocol exists for: with
    ``old_logprobs == proximal_logprobs`` on the gold tokens the behavioural importance weight
    is ``exp(0) = 1``, so the surrogate is exactly the gold row's advantage and nothing is
    rejected. The contrast row is the same call with a 0.0 placeholder, which the audit
    measured being silently shrunk -- reproduced here so the difference is a number in this
    file and not a citation.
    """
    n, width = 2, 6
    loss_mask = torch.ones(n, width, dtype=torch.bool)
    advantages = torch.ones(n, width)
    logprobs = torch.full((n, width), -1.0)
    prox = torch.full((n, width), -1.0)
    rs = RejectionSamplingConfig(level="token", action="mask", metric="ratio", upper=5.0)

    matched, stat = ppo_actor_loss_fn(
        logprobs=logprobs,
        old_logprobs=prox.clone(),
        proximal_logprobs=prox,
        advantages=advantages,
        eps_clip=0.2,
        eps_clip_higher=None,
        loss_mask=loss_mask,
        rejection_sampling=rs,
    )
    placeholder, _ = ppo_actor_loss_fn(
        logprobs=logprobs,
        old_logprobs=torch.zeros(n, width),
        proximal_logprobs=prox,
        advantages=advantages,
        eps_clip=0.2,
        eps_clip_higher=None,
        loss_mask=loss_mask,
        rejection_sampling=rs,
    )
    assert float(matched) == pytest.approx(-1.0)
    assert float(placeholder) == pytest.approx(-torch.exp(torch.tensor(-1.0)).item(), abs=1e-6)
    assert float(placeholder) > float(matched)


# ------------------------------------------ what the solved/unsolved routing sees after ---


def _meta(n_groups: int, size: int, width: int) -> TrajBatchMeta:
    """Group structure for ``_compute_advantages``."""
    return TrajBatchMeta(
        n_trajs=n_groups * size,
        traj_group_sizes=[size] * n_groups,
        traj_seqlens=[width] * (n_groups * size),
    )


def _advantages(actor, batch):
    """Run the REAL advantage computation on a copy of the batch."""
    data = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    n_groups = data["input_ids"].shape[0] // G
    return actor._compute_advantages(data, _meta(n_groups, G, data["input_ids"].shape[1]))


def test_before_substitution_an_all_wrong_group_takes_the_unsolved_constant():
    """Anti-vacuity for the test below: M20 does reach this group as the batch arrives."""
    gr = GroupRoutingConfig(enabled=True, unsolved_advantage=-0.5)
    batch = make_batch([make_group([1.0] * 4), make_group([0.0] * 4)])
    adv = _advantages(make_actor(gr), batch)["advantages"]
    assert float(adv[G:].min()) == pytest.approx(-0.5)


def test_after_substitution_the_group_is_no_longer_unsolved_and_m20_stops_applying():
    """The interaction M19/M20 must not silently change, stated as a measurement.

    A gold row scores 1.0, so the group's raw rewards go from ``[0,0,0,0]`` to ``[1,0,0,0]``:
    ``unsolved = (max <= 0.5)`` becomes False and ``solved = (min > 0.5)`` stays False, so the
    group is neither, and ``unsolved_advantage`` -- unlikelihood on known-wrong samples --
    correctly stops applying to a group that now contains a correct one. It also stops being
    SILENT: group-level reward normalisation gives the gold row a positive advantage and the
    three wrong rollouts negative ones, which is the ordinary GRPO signal the substitution
    exists to create.

    The consequence for reporting, and it is not cosmetic: a gold arm's
    ``unsolved_group_fraction`` is lower than an ungrounded arm's by exactly the number of
    groups that received a gold, so the two arms' silence panels are NOT comparable at face
    value.
    """
    gr = GroupRoutingConfig(enabled=True, unsolved_advantage=-0.5)
    batch = make_batch([make_group([1.0] * 4), make_group([0.0] * 4)])
    sub, stats = substitute_gold_rows(batch, "dyme", group_sizes=G)
    assert stats.substituted_rows == (G,)

    adv = _advantages(make_actor(gr), sub)["advantages"]
    gold_row, sibling_rows = adv[G], adv[G + 1 : 2 * G]
    assert float(gold_row.max()) > 0.0, gold_row
    assert float(sibling_rows.max()) <= 0.0, sibling_rows
    # Not the M20 constant: every wrong sibling would then sit at exactly -0.5.
    assert float(sibling_rows.min()) != pytest.approx(-0.5)


def test_a_solved_group_is_untouched_by_substitution_so_m19_is_unaffected():
    """The other half of the interaction: M19's reach does not move."""
    gr = GroupRoutingConfig(enabled=True, solved_advantage=0.5)
    batch = make_batch([make_group([1.0] * 4), make_group([0.0] * 4)])
    plain = _advantages(make_actor(gr), batch)["advantages"]
    sub, _ = substitute_gold_rows(batch, "dyme", group_sizes=G)
    routed = _advantages(make_actor(gr), sub)["advantages"]
    assert torch.equal(plain[:G], routed[:G])


def test_the_finite_sentinel_does_not_poison_the_advantages():
    """Why the sentinel is finite, checked on the path that would have been poisoned.

    ``_compute_advantages`` reads ``logprobs`` for the KL reward with ``kl_ctl = 0.0``, and a
    NaN there would survive that multiplication. With the sentinel every advantage in the
    batch stays finite, which is the property the audit says the choice of value has to buy.
    """
    batch = make_batch([make_group([1.0] * 4), make_group([0.0] * 4)])
    sub, _ = substitute_gold_rows(batch, "dyme", group_sizes=G)
    adv = _advantages(make_actor(GroupRoutingConfig(enabled=True)), sub)["advantages"]
    assert torch.isfinite(adv).all()


def test_stats_are_frozen_so_a_caller_cannot_edit_the_record():
    """The counts are the audit trail of an arm; they are not a scratch dict."""
    stats = GoldStats(n_groups=1, n_rows=4)
    with pytest.raises(Exception):
        stats.rows_substituted = 5
