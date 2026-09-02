"""DeepMath-103K as an RL training set, for the operating point where routing has something to route.

MATH-lighteval is the inherited training corpus and it is too easy for the question this
project asks. Its test counterpart MATH-500 is saturated at 0.966 on 27B, and a routing method
can only route what the data produces: on near-saturated data most groups come back all-correct,
the UNSOLVED branch is rare, and both the router and any teacher supplier have almost nothing to
reach. Measured on this repo's own Level 4-5 filtered probe batch, 48.4% of groups were still
fully correct.

DeepMath-103K (arXiv:2504.11456, MIT) is the replacement, and it was chosen for a reason that is
not "it is harder": LSPO (arXiv:2607.27787) trained on it, and LSPO is arm A4 in
``experiments/m25/PLAN.md``. Matching its corpus makes that baseline comparison like-for-like
rather than confounded by data.

MEASURED ON THIS BOX over all 103,022 rows, against MATH-lighteval's 7,500 through the same
tokenizer -- the MATH figures below reproduce the ones already recorded in ``competition_math``,
so this is one comparison and not two unrelated measurements:

===========================  ====================  =========================
quantity                     MATH-lighteval train  DeepMath-103K train
===========================  ====================  =========================
rows                         7,500                 103,022
stated difficulty            Level 1-5, 5 buckets  1.0-10.0 float, median 6.0
fraction at the top rung     Level 5: 30.7%        >= 7.0: 31.4%
problem tokens median/p90    49 / 149              59 / 120
solution tokens median/p90   162 / 482             4,369 / 10,160
solution <= 1024 tokens      99.15%                0.34%
===========================  ====================  =========================

Two consequences follow from that table and both are load-bearing.

**The gold derivation is a REASONING TRACE, not a worked solution.** ``r1_solution_1`` is a
DeepSeek-R1 completion with the opening ``<think>`` stripped and the closing ``</think>``
retained: measured over all 103,022 rows, 99.999% contain exactly one close tag and 99.965%
contain no opening tag at all, with the close sitting at 88.5% of the way through the string
(median). That is precisely the shape a prompt ending in ``<|im_start|>assistant\\n<think>\\n``
expects as its continuation, so for a thinking model the correct ``gold_template`` here is the
DEFAULT ``"{solution}"``. This is the OPPOSITE of MATH, where the same model needs
``"\\n</think>\\n\\n{solution}"`` because a MATH solution closes no block. Carrying MATH's
template over to this corpus splices a SECOND close tag into the middle of an assistant turn and
trains a shape the model never emits -- silently, since nothing downstream inspects gold text.
``_assert_one_think_close`` refuses that combination rather than letting it train.

**A gold row at a 4,369-token median will usually not FIT.** ``selfevo/gold/attach.py`` pads the
gold to the trajectory's own width and ``GoldStats.groups_no_fit`` counts the ones too long for
it, so on this corpus the gold path's reach is bounded by the generation cap rather than by how
many groups come back unsolved. Whoever runs arms A4/A5 must read ``groups_no_fit`` first and set
the cap against the number above, or truncate deliberately through ``gold_template``.

The reward function is unchanged: ``math_reward_fn`` is ``math_verify.parse`` + ``verify``
against ``data["answer"]``. So, exactly as for MATH, the job here is to hand it ``messages`` and
``answer`` with the gold in a form ``parse`` can read.

FIELD CHOICES CROSS-CHECKED AGAINST THE AUTHORS' OWN LOADER rather than guessed. DeepMath ships
a verl preprocessor at ``zwhe99/verl:examples/data_preprocess/deepmath_103k.py`` (Apache-2.0, so
reusable) which takes ``question`` as the prompt, ``final_answer`` as the rule-based
``ground_truth``, and carries ``r1_solution_1`` alongside as the worked derivation. This adapter
makes the same three choices, so an arm run here and an arm run on the authors' own pipeline
differ in framework rather than in what they read. LSPO needs exactly that derivation column --
it fits its scaffold by supervised learning on the worked solutions, not on final answers -- and
supplying it is what ``keep_solution`` is for.

WHAT THIS ADAPTER DOES NOT DO: it does not add columns. The output schema is ``messages`` and
``answer``, plus ``gold_ids`` under ``keep_solution``, and nothing else. ``difficulty`` and
``topic`` are exposed as FILTERS rather than passed through, because ``concat_padded_tensors``
refuses a batch whose trajectory dicts disagree on their key set, so an extra column would break
collation rather than degrade.
"""

from __future__ import annotations

from datasets import load_dataset

# Reused rather than reimplemented. The measurement that justifies \boxed{}-wrapping a bare gold
# lives in competition_math and applies unchanged here: over 400 MATH training examples a bare
# gold self-verifies on 83.8% while \boxed{gold} reaches 100%, and the 16% it loses are the
# tuples, intervals, surds and mixed numbers -- the structured half of the answer space.
# DeepMath's final_answer is bare for 103,022 of 103,022 rows (zero of them contain a box), so
# every row here is in exactly that regime.
from .competition_math import _boxed_gold

__all__ = ["get_deepmath_rl_dataset", "SOLUTION_FIELDS"]

#: The three independent R1 derivations each row ships. A gold arm picks one; they are distinct
#: reasoning paths for the same problem, so which one is used is a seam and not a formality.
SOLUTION_FIELDS = ("r1_solution_1", "r1_solution_2", "r1_solution_3")

#: The closing tag an R1 trace carries. Counted, never stripped -- see _assert_one_think_close.
_THINK_CLOSE = "</think>"

#: How many rows the schema and template probes read. Large enough that a corpus whose schema
#: does not match cannot pass by luck, small enough to cost nothing.
_PROBE_ROWS = 200


def _assert_one_think_close(probe_texts: list[str], gold_template: str) -> None:
    """Refuse a gold_template that duplicates the reasoning-trace close tag.

    The failure this exists for is silent. The close tag appears exactly once in 99.999% of this
    corpus's derivations, so a template written for MATH -- ``"\\n</think>\\n\\n{solution}"``,
    which is CORRECT there because a MATH solution closes no block -- yields two closes here. The
    resulting gold row trains the model to emit a second close tag mid-answer, and nothing
    downstream reads gold text, so the run completes and reports nothing wrong.

    The test is on the ASSEMBLED text rather than on the template alone, so it also catches the
    reverse case: a corpus revision that starts shipping the opening tag as well. It is a
    MAJORITY test rather than an any-row test because exactly one row in the corpus already
    carries two closes on its own, and one malformed row must not veto a correct configuration.

    Args:
        probe_texts: Assembled gold texts, i.e. with gold_template already applied.
        gold_template: The template used, quoted back in the error so the fix is obvious.

    Raises:
        ValueError: If more than half the probed golds carry more than one close tag.
    """
    if not probe_texts:
        return
    dup = sum(1 for t in probe_texts if t.count(_THINK_CLOSE) > 1)
    if dup > len(probe_texts) // 2:
        raise ValueError(
            f"gold_template={gold_template!r} produces {dup}/{len(probe_texts)} probed gold "
            f"rows carrying more than one {_THINK_CLOSE!r}. DeepMath derivations are R1 traces "
            f"that already close their own thinking block (99.999% of rows carry exactly one), "
            f"so a template that adds another splices a second close into the middle of an "
            f"assistant turn and trains a shape the model never emits. For a thinking model "
            f"whose prompt ends at the opening tag, the correct template for THIS corpus is the "
            f"default '{{solution}}'; the MATH form that prepends a close is correct for "
            f"MATH-lighteval and wrong here."
        )


def get_deepmath_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    keep_solution: bool = False,
    gold_template: str = "{solution}",
    append_eos: bool = True,
    solution_field: str = "r1_solution_1",
    min_difficulty: float | None = None,
    max_difficulty: float | None = None,
    drop_unanswerable: bool = True,
    **kwargs,
):
    """DeepMath-103K, shaped exactly like the MATH and GSM8K RL datasets.

    Args:
        path: HF dataset id, i.e. ``zwhe99/DeepMath-103K``.
        split: A ``datasets`` split expression; the corpus ships only ``train``.
        tokenizer: Used for the length filter, and required when ``keep_solution`` is set.
        max_length: Drop prompts longer than this many tokens.
        keep_solution: Keep the gold DERIVATION as a tokenised ``gold_ids`` column. Default
            False, matching the MATH adapter, so a config that does not ask for gold gets the
            same two columns any other RL adapter here would give it.
        gold_template: How the gold text is assembled before tokenising, with ``{solution}``
            substituted. Default ``"{solution}"`` -- the raw trace, which is already the right
            continuation for a prompt that ends at an opening think tag. See
            ``_assert_one_think_close``.
        append_eos: Append the tokenizer's EOS. Default True, for the reason recorded in
            ``competition_math``: a rollout's output tokens end with the stop token, so a gold
            row without one would be the only row in the batch that never terminates.
        solution_field: Which of the three R1 derivations to use as the gold. One of
            ``SOLUTION_FIELDS``.
        min_difficulty: Keep rows whose ``difficulty`` is at least this. The corpus's own field,
            1.0-10.0. Also the way to drop the 4 rows carrying the -1.0 sentinel.
        max_difficulty: Keep rows whose ``difficulty`` is at most this.
        drop_unanswerable: Drop rows with an empty ``final_answer``. Default True. There are
            exactly 2 in 103,022, and each would otherwise become a group that grades every
            rollout wrong forever -- indistinguishable in the logs from a problem the model
            cannot solve, and so a fake permanent member of the UNSOLVED branch this project
            measures. Set False to keep them and see them.

    Returns:
        A dataset with ``messages`` and ``answer``, matching what ``MathAgent`` reads, plus
        ``gold_ids`` when ``keep_solution`` is set. ``gold_ids`` is an empty list for a row whose
        solution is missing or empty; the column is present for EVERY row either way, because
        ``concat_padded_tensors`` refuses a batch whose trajectory dicts disagree on their key
        set, so an absent-for-some column would break collation rather than degrade.

    Raises:
        ValueError: If ``keep_solution`` is set without a tokenizer, which would otherwise
            produce a gold column of empty lists and a gold arm that trains on nothing; if
            ``solution_field`` is not one of the three the corpus ships; if a filter leaves no
            rows; if no example yields an answer, which means the schema is not what this adapter
            expects and training would proceed with every reward zero, indistinguishable from a
            model that cannot solve anything; or if ``gold_template`` duplicates the close tag.
    """
    if keep_solution and tokenizer is None:
        raise ValueError(
            "keep_solution=True needs a tokenizer: the gold is tokenised here, once, rather "
            "than per rollout. Without one every gold_ids would be empty and the gold arm "
            "would train on nothing while still reporting itself as a gold arm."
        )
    if solution_field not in SOLUTION_FIELDS:
        raise ValueError(
            f"solution_field={solution_field!r} is not one of {SOLUTION_FIELDS}. A field name "
            f"this corpus does not ship would silently yield an empty gold for every row."
        )
    dataset = load_dataset(path=path, split=split)

    # Filters run BEFORE the map, so the probes below see the rows that will actually train.
    if min_difficulty is not None or max_difficulty is not None:
        lo = -float("inf") if min_difficulty is None else float(min_difficulty)
        hi = float("inf") if max_difficulty is None else float(max_difficulty)
        dataset = dataset.filter(lambda s: lo <= float(s["difficulty"]) <= hi)
        if len(dataset) == 0:
            raise ValueError(
                f"difficulty filter [{min_difficulty}, {max_difficulty}] kept 0 of the rows in "
                f"{path}:{split}. The corpus's own field runs 1.0-10.0 with a median of 6.0, "
                f"plus 4 rows at a -1.0 sentinel; an empty training set would otherwise surface "
                f"as a dataloader error with no mention of the filter."
            )
    if drop_unanswerable:
        dataset = dataset.filter(lambda s: bool((s["final_answer"] or "").strip()))
        if len(dataset) == 0:
            raise ValueError(
                f"every row of {path}:{split} has an empty final_answer; the schema is not what "
                "this adapter expects."
            )

    def process(sample):
        # The answer comes from the corpus's own curated final_answer, NOT from extracting the
        # box out of the derivation. Measured over all 103,022 rows the two agree on 97.47% by
        # string equality and every derivation does carry a brace-balanced box, so either would
        # work -- but final_answer is the field this dataset verifies and publishes, and the
        # 2.53% they disagree on is exactly where a formatting variant inside a trace would
        # otherwise become the gold.
        answer = (sample.get("final_answer") or "").strip()
        out = {
            "messages": [
                {
                    "role": "user",
                    "content": sample["question"]
                    + "\nPlease put your final answer within \\boxed{}.",
                }
            ],
            "answer": _boxed_gold(answer),
        }
        if keep_solution:
            # add_special_tokens=False: this is a CONTINUATION of a prompt that has already been
            # through the chat template, so a second BOS or turn header would land in the middle
            # of an assistant turn.
            solution = sample.get(solution_field) or ""
            text = gold_template.format(solution=solution) if solution else ""
            ids = tokenizer.encode(text, add_special_tokens=False) if text else []
            if ids and append_eos and tokenizer.eos_token_id is not None:
                ids = list(ids) + [int(tokenizer.eos_token_id)]
            out["gold_ids"] = ids
        return out

    if keep_solution:
        # .select() rather than dataset[col][:n]: on several datasets versions the latter
        # materialises the ENTIRE column, which here is 103k reasoning traces of ~18 KB each,
        # i.e. ~1.9 GB read in order to look at 200 rows -- paid on every run, silently.
        head = dataset.select(range(min(_PROBE_ROWS, len(dataset))))
        probe = [gold_template.format(solution=s) for s in head[solution_field] if s]
        _assert_one_think_close(probe, gold_template)

    keep = {"messages", "answer"} | ({"gold_ids"} if keep_solution else set())
    dataset = dataset.map(process)
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep])

    n_probe = min(_PROBE_ROWS, len(dataset))
    n_gold = sum(1 for a in dataset.select(range(n_probe))["answer"] if a)
    if n_gold == 0:
        raise ValueError(
            f"no answer found in the first {_PROBE_ROWS} examples of {path}:{split}; the schema "
            "is not what this adapter expects. Training would run with every reward zero, which "
            "looks exactly like a model that cannot solve anything."
        )

    if max_length is not None:

        def filter_length(sample):
            return len(tokenizer.encode(sample["messages"][0]["content"])) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
