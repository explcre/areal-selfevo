"""The DeepMath-103K adapter, its registration, and the proof that MATH is unchanged.

The corpus switch exists because a routing method can only route what the data produces.
MATH-lighteval's test counterpart is saturated at 0.966 on 27B, so most groups come back
all-correct and the UNSOLVED branch -- the one with no self-target, which a teacher or the gold
path is the only thing that can serve -- is rare. DeepMath-103K (arXiv:2504.11456, MIT) is the
replacement, chosen because LSPO (arXiv:2607.27787) trained on it and LSPO is arm A4 in
``experiments/m25/PLAN.md``.

Three things here are tested because they were nearly missed, and each fails silently:

1. **The dispatch ORDERING.** ``zwhe99/DeepMath-103K`` satisfies the MATH branch's own
   ``"math" in path.lower()`` predicate. The new branch therefore has to come FIRST, and the
   test that asserts it is paired with an anti-vacuity test showing the id really does match the
   MATH predicate -- without that pair, the ordering test would pass just as well on a
   dispatcher that had no ordering hazard at all.

2. **The gold template is INVERTED relative to MATH.** DeepMath derivations are R1 traces that
   already close their own thinking block, so the template MATH needs
   (``"\\n</think>\\n\\n{solution}"``) produces two closes here and trains a shape the model never
   emits. Nothing downstream reads gold text, so this would run to completion.

3. **Rollback.** The MATH adapter's output is hashed through the DISPATCHER against the digest
   recorded before any of this existed, so a registration change that rerouted MATH, and a
   content change that kept the column names, both fail here rather than in a run.
"""

from __future__ import annotations

import hashlib
import json

import pytest

DEEPMATH = "zwhe99/DeepMath-103K"
MATH_DATASET = "DigitalLearningGmbH/MATH-lighteval"

# The digest recorded in selfevo/tests/test_gold_batch_path.py, measured over the adapted MATH
# train split before ``keep_solution`` existed. Repeated rather than imported so this file states
# what it is asserting; if the two ever disagree, both tests fail and the disagreement is the
# signal. Asserted here through ``_get_custom_dataset`` rather than through the adapter directly,
# which is the part that catches a registration change rerouting MATH.
MATH_TRAIN_DIGEST = "dbe5c602f6ca651b71ea8367e49d96ac8f9f103681ab16acfc9dc10dc2255033"

# Rows measured on this box over all 103,022: exactly these two have an empty ``final_answer``.
# A slice containing both is how ``drop_unanswerable`` is tested on real data rather than on a
# fixture that assumes the defect exists.
EMPTY_ANSWER_ROWS = (99592, 99696)
EMPTY_ANSWER_SLICE = "train[99560:99720]"

# A DeepMath derivation, abbreviated: no opening tag, exactly one close, then the summary. This
# is the shape measured over all 103,022 rows (99.965% carry no opening tag; 99.999% carry
# exactly one close).
R1_SHAPED = "Okay, so I need to evaluate this limit. Let me try substitution.\n</think>\n\nThe limit is $\\boxed{0}$."
MATH_TEMPLATE = "\n</think>\n\n{solution}"


# --------------------------------------------------------------------------------------------
# The close-tag guard. Pure function, so these run everywhere and need no corpus.
# --------------------------------------------------------------------------------------------


def test_the_guard_refuses_the_math_template_on_r1_shaped_golds():
    """MATH's template is correct for MATH and wrong here, and nothing else would say so."""
    from areal.dataset.deepmath import _assert_one_think_close

    probe = [MATH_TEMPLATE.format(solution=R1_SHAPED)] * 10
    with pytest.raises(ValueError, match="more than one"):
        _assert_one_think_close(probe, MATH_TEMPLATE)


def test_the_guard_accepts_the_default_template_on_r1_shaped_golds():
    """The default must not be refused, or the corpus would be unusable for gold at all."""
    from areal.dataset.deepmath import _assert_one_think_close

    _assert_one_think_close(["{solution}".format(solution=R1_SHAPED)] * 10, "{solution}")


def test_the_guard_is_a_majority_test_and_not_an_any_row_test():
    """One malformed row must not veto a correct configuration.

    Exactly one row of the real corpus already carries two closes on its own. An any-row guard
    would refuse the DEFAULT template on the real data, which is the configuration that is
    right -- a guard that fires on the correct setting gets deleted rather than heeded.
    """
    from areal.dataset.deepmath import _assert_one_think_close

    probe = [R1_SHAPED] * 9 + [R1_SHAPED + "\n</think>\n"]
    _assert_one_think_close(probe, "{solution}")


def test_the_guard_fires_once_the_bad_rows_are_the_majority():
    """The complement of the test above: the threshold is real, not an unreachable branch."""
    from areal.dataset.deepmath import _assert_one_think_close

    probe = [R1_SHAPED] * 4 + [R1_SHAPED + "\n</think>\n"] * 6
    with pytest.raises(ValueError, match="more than one"):
        _assert_one_think_close(probe, "{solution}")


def test_the_guard_ignores_an_empty_probe():
    """An empty probe means nothing was measured, and a guard must not invent a verdict."""
    from areal.dataset.deepmath import _assert_one_think_close

    _assert_one_think_close([], "{solution}")


# --------------------------------------------------------------------------------------------
# Argument guards. No corpus needed: these must raise before any load is attempted.
# --------------------------------------------------------------------------------------------


def test_keep_solution_without_a_tokenizer_raises():
    """Otherwise every gold_ids is empty and the gold arm trains on nothing while reporting
    itself as a gold arm."""
    from areal.dataset.deepmath import get_deepmath_rl_dataset

    with pytest.raises(ValueError, match="needs a tokenizer"):
        get_deepmath_rl_dataset(
            path=DEEPMATH, split="train", tokenizer=None, keep_solution=True
        )


def test_an_unknown_solution_field_raises():
    """A field name this corpus does not ship would silently yield an empty gold for every row."""
    from areal.dataset.deepmath import get_deepmath_rl_dataset

    with pytest.raises(ValueError, match="solution_field"):
        get_deepmath_rl_dataset(
            path=DEEPMATH,
            split="train",
            tokenizer=object(),
            keep_solution=True,
            solution_field="solution",
        )


# --------------------------------------------------------------------------------------------
# Registration and the ordering hazard.
# --------------------------------------------------------------------------------------------


def test_the_deepmath_id_matches_the_math_branch_predicate():
    """Anti-vacuity for the ordering test below.

    The MATH branch fires on ``("MATH" in path or "math" in path.lower())``. If that were false
    for this id there would be no hazard, and the ordering test would be asserting nothing.
    """
    path = DEEPMATH
    assert "math" in path.lower()
    assert ("MATH" in path or "math" in path.lower()) and "gsm8k" not in path


def test_a_deepmath_path_routes_to_the_deepmath_adapter(monkeypatch):
    """The branch order is what makes this true; see the anti-vacuity test above."""
    import areal.dataset as ds
    import areal.dataset.competition_math as cm
    import areal.dataset.deepmath as dm

    monkeypatch.setattr(dm, "get_deepmath_rl_dataset", lambda **kw: ("deepmath", kw))
    monkeypatch.setattr(cm, "get_math_rl_dataset", lambda **kw: ("math", kw))
    got = ds._get_custom_dataset(path=DEEPMATH, type="rl", split="train", tokenizer=None)
    assert got[0] == "deepmath"


def test_a_math_path_still_routes_to_the_math_adapter(monkeypatch):
    """The new branch must not capture the corpus every existing config names."""
    import areal.dataset as ds
    import areal.dataset.competition_math as cm
    import areal.dataset.deepmath as dm

    monkeypatch.setattr(dm, "get_deepmath_rl_dataset", lambda **kw: ("deepmath", kw))
    monkeypatch.setattr(cm, "get_math_rl_dataset", lambda **kw: ("math", kw))
    got = ds._get_custom_dataset(path=MATH_DATASET, type="rl", split="train", tokenizer=None)
    assert got[0] == "math"


def test_a_deepmath_path_of_the_wrong_type_does_not_reach_the_rl_adapter(monkeypatch):
    """The branch is gated on type == 'rl'; an sft request must not silently get RL rows."""
    import areal.dataset as ds
    import areal.dataset.deepmath as dm

    monkeypatch.setattr(dm, "get_deepmath_rl_dataset", lambda **kw: ("deepmath", kw))
    with pytest.raises(ValueError):
        ds._get_custom_dataset(path=DEEPMATH, type="sft", split="train", tokenizer=None)


def test_deepmath_is_advertised_in_valid_datasets():
    """VALID_DATASETS is what the unsupported-dataset error prints; an omission sends the next
    reader looking for an adapter that is already here."""
    from areal.dataset import VALID_DATASETS

    assert "deepmath" in VALID_DATASETS


# --------------------------------------------------------------------------------------------
# Rollback: the MATH default is byte-identical, asserted through the dispatcher.
# --------------------------------------------------------------------------------------------


def _digest(adapted) -> str:
    """sha256 over every (messages, answer) pair, the convention used by the gold-path test."""
    h = hashlib.sha256()
    for row in adapted:
        h.update(
            json.dumps(
                [row["messages"], row["answer"]], sort_keys=True, ensure_ascii=False
            ).encode()
        )
    return h.hexdigest()


def test_the_math_default_is_byte_identical_through_the_dispatcher():
    """Registration changed the dispatcher, so the proof is taken THROUGH the dispatcher.

    A column-name check passes on an adapter that kept its columns and changed their contents,
    and an adapter-level check passes on a dispatcher that stopped routing MATH to it. Hashing
    the dispatcher's own output over all 7500 rows catches both.
    """
    pytest.importorskip("datasets")
    import datasets

    import areal.dataset as ds

    try:
        datasets.load_dataset(path=MATH_DATASET, split="train[:1]")
    except Exception as exc:  # pragma: no cover - box without the corpus
        pytest.skip(f"{MATH_DATASET} not available here: {exc}")
    adapted = ds._get_custom_dataset(
        path=MATH_DATASET, type="rl", split="train", tokenizer=None, max_length=None
    )
    assert set(adapted.column_names) == {"messages", "answer"}, adapted.column_names
    assert _digest(adapted) == MATH_TRAIN_DIGEST


# --------------------------------------------------------------------------------------------
# The adapter against the real corpus.
# --------------------------------------------------------------------------------------------


def _deepmath(split="train[:64]", **kwargs):
    """The adapted DeepMath split, or a skip if the corpus is not on this box."""
    pytest.importorskip("datasets")
    import datasets

    from areal.dataset.deepmath import get_deepmath_rl_dataset

    try:
        datasets.load_dataset(path=DEEPMATH, split="train[:1]")
    except Exception as exc:  # pragma: no cover - box without the corpus
        pytest.skip(f"{DEEPMATH} not available here: {exc}")
    kwargs.setdefault("tokenizer", None)
    kwargs.setdefault("max_length", None)
    return get_deepmath_rl_dataset(path=DEEPMATH, split=split, **kwargs)


@pytest.fixture(scope="module")
def tok():
    """A real tokenizer, or a skip. Any tokenizer will do; the ids are never compared."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"no tokenizer available here: {exc}")


def test_the_default_schema_is_messages_and_answer_only():
    """Extra columns do not degrade -- concat_padded_tensors refuses a batch whose trajectory
    dicts disagree on their key set, so a passed-through difficulty column breaks collation."""
    ds = _deepmath()
    assert set(ds.column_names) == {"messages", "answer"}


def test_every_answer_is_boxed_and_non_empty():
    """final_answer is bare for 103,022 of 103,022 rows. A bare gold self-verifies on only
    83.8% of MATH and the 16% it loses are the structured answers, so the wrap is the
    difference between grading the hard half and zeroing it."""
    ds = _deepmath()
    answers = list(ds["answer"])
    assert all(a.startswith("\\boxed{") and a.endswith("}") for a in answers)
    assert all(len(a) > len("\\boxed{}") for a in answers)


def test_the_prompt_is_one_user_turn_carrying_the_boxed_instruction():
    """MathAgent reads messages; the instruction is what makes the response parseable."""
    ds = _deepmath()
    for row in ds:
        assert [m["role"] for m in row["messages"]] == ["user"]
        assert row["messages"][0]["content"].endswith(
            "\nPlease put your final answer within \\boxed{}."
        )


def test_keep_solution_adds_a_tokenised_gold_and_nothing_else(tok):
    """The flag adds exactly one column, and the gold terminates.

    Without the EOS the gold row would be the only row in the batch that never terminates, and
    training on it teaches the model not to stop after a derivation.
    """
    ds = _deepmath(tokenizer=tok, keep_solution=True)
    assert set(ds.column_names) == {"messages", "answer", "gold_ids"}
    golds = list(ds["gold_ids"])
    assert all(len(g) > 0 for g in golds)
    assert all(g[-1] == tok.eos_token_id for g in golds)


def test_append_eos_false_leaves_the_gold_unterminated(tok):
    """The complement, so the assertion above is about append_eos and not about a gold that
    happened to end in the EOS id anyway."""
    ds = _deepmath(tokenizer=tok, keep_solution=True, append_eos=False)
    assert all(g[-1] != tok.eos_token_id for g in ds["gold_ids"])


def test_the_gold_is_long_enough_that_fit_is_the_binding_constraint(tok):
    """The measurement arms A4/A5 have to plan against, pinned so it cannot drift unnoticed.

    Measured over the whole corpus: median gold 4,369 tokens, p90 10,160, only 0.34% at or
    under 1024 -- against MATH's 162 median and 99.15% under 1024. selfevo/gold/attach.py pads
    the gold to the trajectory's own width, so on this corpus the gold path's reach is bounded
    by the generation cap and NOT by how many groups come back unsolved. Asserted loosely (a
    median above 1024) so the test pins the regime rather than a tokenizer-specific number.
    """
    ds = _deepmath(tokenizer=tok, keep_solution=True)
    lens = sorted(len(g) for g in ds["gold_ids"])
    assert lens[len(lens) // 2] > 1024


def test_the_solution_field_selects_which_derivation_becomes_the_gold(tok):
    """Three independent R1 paths ship per row; which one is used is a seam, so it must
    actually change the output."""
    a = _deepmath(tokenizer=tok, keep_solution=True, solution_field="r1_solution_1")
    b = _deepmath(tokenizer=tok, keep_solution=True, solution_field="r1_solution_2")
    assert list(a["gold_ids"]) != list(b["gold_ids"])
    assert list(a["answer"]) == list(b["answer"])


def test_the_math_gold_template_is_refused_on_the_real_corpus(tok):
    """The guard, driven through the adapter on real rows rather than on a fixture."""
    with pytest.raises(ValueError, match="more than one"):
        _deepmath(tokenizer=tok, keep_solution=True, gold_template=MATH_TEMPLATE)


def test_drop_unanswerable_removes_the_two_rows_with_no_final_answer():
    """A row with no answer grades every rollout wrong forever, which in the logs is
    indistinguishable from a problem the model cannot solve -- a fake permanent member of the
    UNSOLVED branch this project measures."""
    kept = _deepmath(split=EMPTY_ANSWER_SLICE, drop_unanswerable=True)
    raw = _deepmath(split=EMPTY_ANSWER_SLICE, drop_unanswerable=False)
    assert len(raw) - len(kept) == len(EMPTY_ANSWER_ROWS)
    assert all(a for a in kept["answer"])
    assert sum(1 for a in raw["answer"] if not a) == len(EMPTY_ANSWER_ROWS)


def test_the_difficulty_filter_selects_on_the_corpus_own_field():
    """Difficulty is the whole reason for the switch, so it has to be a usable knob."""
    pytest.importorskip("datasets")
    import datasets

    try:
        raw = datasets.load_dataset(path=DEEPMATH, split="train[:512]")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"{DEEPMATH} not available here: {exc}")
    expected = sum(1 for d in raw["difficulty"] if d >= 7.0)
    got = _deepmath(split="train[:512]", min_difficulty=7.0)
    assert len(got) == expected
    assert 0 < expected < 512, expected


def test_an_impossible_difficulty_filter_raises_rather_than_training_on_nothing():
    """An empty training set would otherwise surface as a dataloader error naming no filter."""
    with pytest.raises(ValueError, match="difficulty filter"):
        _deepmath(split="train[:512]", min_difficulty=99.0)


def test_the_corpus_is_the_verified_row_count():
    """Verified by row count and not by the path a download returned.

    ``hf download`` on this box returned exit 0 and a snapshot path while fetching 48 KB and
    zero data files, because the extra --include patterns were parsed as positional filenames
    and the flag was silently ignored. The count below is the sum of ``metadata.num_rows`` over
    the ten parquet shards, and matches the dataset card.
    """
    pytest.importorskip("datasets")
    import datasets

    try:
        builder = datasets.load_dataset_builder(DEEPMATH)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"{DEEPMATH} not available here: {exc}")
    assert builder.info.splits["train"].num_examples == 103022


# --------------------------------------------------------------------------------------------
# Refusals on a synthetic corpus. The real corpus has only 2 unanswerable rows and they are not
# adjacent, so there is no real slice on which the schema guard can fire -- and a guard with no
# test is a guard that gets deleted by the next person who reads it.
# --------------------------------------------------------------------------------------------


def _fake_corpus(monkeypatch, **cols):
    """Point the adapter at an in-memory dataset instead of the Hub."""
    datasets = pytest.importorskip("datasets")
    import areal.dataset.deepmath as dm

    ds = datasets.Dataset.from_dict(cols)
    monkeypatch.setattr(dm, "load_dataset", lambda **kw: ds)


ANSWERLESS = dict(
    question=["what is 1+1?"] * 4,
    final_answer=["", "  ", "", ""],
    difficulty=[5.0] * 4,
    topic=["Mathematics"] * 4,
    r1_solution_1=[R1_SHAPED] * 4,
    r1_solution_2=[R1_SHAPED] * 4,
    r1_solution_3=[R1_SHAPED] * 4,
)


def test_a_corpus_whose_answers_are_all_empty_is_refused(monkeypatch):
    """Training would otherwise run with every reward zero, which in the logs looks exactly
    like a model that cannot solve anything -- the most expensive way to learn about a schema
    change."""
    from areal.dataset.deepmath import get_deepmath_rl_dataset

    _fake_corpus(monkeypatch, **ANSWERLESS)
    with pytest.raises(ValueError, match="no answer found"):
        get_deepmath_rl_dataset(
            path=DEEPMATH, split="train", tokenizer=None, drop_unanswerable=False
        )


def test_dropping_every_row_as_unanswerable_is_refused(monkeypatch):
    """The same defect reached by the other route: a filter that removes everything must say
    so, not hand back an empty dataset."""
    from areal.dataset.deepmath import get_deepmath_rl_dataset

    _fake_corpus(monkeypatch, **ANSWERLESS)
    with pytest.raises(ValueError, match="empty final_answer"):
        get_deepmath_rl_dataset(
            path=DEEPMATH, split="train", tokenizer=None, drop_unanswerable=True
        )


def test_a_synthetic_corpus_with_answers_adapts_cleanly(monkeypatch):
    """Anti-vacuity for the two refusals above: the same shape WITH answers must succeed, or
    they could be passing because the fixture is malformed in some unrelated way."""
    from areal.dataset.deepmath import get_deepmath_rl_dataset

    cols = dict(ANSWERLESS, final_answer=["2", "2", "2", "2"])
    _fake_corpus(monkeypatch, **cols)
    ds = get_deepmath_rl_dataset(path=DEEPMATH, split="train", tokenizer=None)
    assert set(ds.column_names) == {"messages", "answer"}
    assert list(ds["answer"]) == ["\\boxed{2}"] * 4
