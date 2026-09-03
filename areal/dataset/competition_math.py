"""Competition-MATH as an RL training set, for the harder operating point.

GSM8K is easy for the models used here: the solve rate is high, so the RL-silent channel is
~87.5% SOLVED and the free self-target dominates. That branch was measured inert, which makes
the remaining claim a conditional one -- that on a harder task the same channel becomes
UNSOLVED-dominated, where no self-target exists and a teacher or the harness is the only
consumer. Testing that needs a harder training set, and this is it.

The reward function is unchanged: ``math_reward_fn`` is ``math_verify.parse`` + ``verify``
against ``data["answer"]``, which is answer-format agnostic. So the only job here is to hand
it the same two fields GSM8K does -- ``messages`` and ``answer`` -- with the gold in a form
``parse`` can read.

``keep_solution`` adds a third field, ``gold_ids``: the gold DERIVATION, tokenised. It is off
by default and every existing run reproduces bit for bit with it off, which is asserted by
``selfevo/tests/test_gold_batch_path.py`` against a recorded digest of the adapter's output.
It exists because the bare ``\\boxed{}`` answer is not a supervision target -- training
toward ``\\boxed{7}`` teaches the model to emit the token 7, not to derive it -- and an
UNSOLVED group has no self-target of any kind, so the derivation is the only correct target
available without a teacher model or a second rollout.

TOKENISED HERE, not in the workflow, and the reason is cost measured on this box: all 7500
training solutions tokenise in 3.29s once and are then cached by ``datasets`` fingerprinting,
whereas the workflow encodes per ROLLOUT -- 80 times per prompt at the live ``n_samples=8``
over 10 epochs -- inside the async rollout loop of every rollout worker. Tokenising once also
means the length distribution is knowable before a GPU is booked: median 162 tokens, p90 482,
max 2494, and 99.15% under 1024.
"""

from __future__ import annotations

import re

from datasets import load_dataset

__all__ = ["get_math_rl_dataset", "extract_boxed_answer"]

_BOXED = re.compile(r"\\boxed\s*{")


def extract_boxed_answer(solution: str) -> str | None:
    """The contents of the LAST ``\\boxed{...}`` in a solution, brace-balanced.

    Brace-balanced rather than regex-greedy because MATH solutions nest braces freely
    (``\\boxed{\\frac{1}{2}}``), and a non-balanced match truncates at the first ``}`` and
    silently yields a wrong gold answer -- which would grade every correct response as wrong
    and look like a model failure.

    Args:
        solution: A MATH solution string.

    Returns:
        The boxed expression, or None when there is no balanced box.
    """
    last = None
    for m in _BOXED.finditer(solution):
        i = m.end()
        depth, start = 1, i
        while i < len(solution) and depth:
            if solution[i] == "{":
                depth += 1
            elif solution[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last = solution[start : i - 1]
    return last


def _boxed_gold(inner: str | None) -> str:
    """Wrap an extracted answer so the verifier can parse it, or return "" if there was none.

    Args:
        inner: The contents of the solution's last ``\\boxed{...}``, or None.

    Returns:
        ``\\boxed{inner}``, matching the form predictions arrive in, or "".
    """
    return f"\\boxed{{{inner}}}" if inner else ""


def get_math_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    keep_solution: bool = False,
    gold_template: str = "{solution}",
    append_eos: bool = True,
    **kwargs,
):
    """Competition MATH, shaped exactly like the GSM8K RL dataset.

    Args:
        path: HF dataset id, e.g. ``DigitalLearningGmbH/MATH-lighteval``.
        split: ``train`` or ``test``.
        tokenizer: Used for the length filter, and required when ``keep_solution`` is set.
        max_length: Drop prompts longer than this many tokens.
        keep_solution: Keep the gold DERIVATION as a tokenised ``gold_ids`` column. Default
            False, which is the shipped behaviour and the only one any prior run has seen.
        gold_template: How the gold text is assembled before tokenising, with ``{solution}``
            substituted. Default ``"{solution}"`` -- the raw derivation. It is a parameter
            rather than a constant because a gold row is spliced in after a PROMPT, and the
            prompt's chat template decides what a valid continuation looks like: the live 30B
            model's template ends the prompt at ``<|im_start|>assistant\\n<think>\\n``, so a
            gold that does not close that block with ``\\n</think>\\n\\n`` first would train
            the model to answer inside a thinking block it never leaves. Set it per model;
            there is no default that is right for every template, and guessing one silently
            is how a gold arm trains a shape the model never emits.
        append_eos: Append the tokenizer's EOS. Default True. A rollout's ``output_tokens``
            end with the stop token (``ModelResponse.output_tokens_without_stop`` strips it,
            and ``multi_turn.py:115`` re-adds it when it is absent), so a gold row without one
            would be the only row in the batch that never terminates, and training on it
            teaches the model not to stop after a derivation.

    Returns:
        A dataset with ``messages`` and ``answer``, matching what ``MathAgent`` reads, plus
        ``gold_ids`` when ``keep_solution`` is set. ``gold_ids`` is an empty list for a row
        whose solution is missing or empty; the column is present for EVERY row either way,
        because ``concat_padded_tensors`` refuses a batch whose trajectory dicts disagree on
        their key set, so an absent-for-some column would break collation rather than degrade.

    Raises:
        ValueError: If no example yields a boxed answer -- that means the schema is not what
            this adapter expects, and training would proceed with every reward zero, which is
            indistinguishable from a model that cannot solve anything. Also if
            ``keep_solution`` is set without a tokenizer, which would otherwise produce a
            gold column of empty lists and a gold arm that trains on nothing.
    """
    if keep_solution and tokenizer is None:
        raise ValueError(
            "keep_solution=True needs a tokenizer: the gold is tokenised here, once, rather "
            "than per rollout. Without one every gold_ids would be empty and the gold arm "
            "would train on nothing while still reporting itself as a gold arm."
        )
    dataset = load_dataset(path=path, split=split)

    def process(sample):
        # The gold is handed back \boxed{}-wrapped, not bare. math_verify.parse needs LaTeX
        # delimiters to read a structured answer: measured over 400 training examples, a bare
        # gold self-verifies on only 83.8% while \boxed{gold} reaches 100%. The 16% it loses
        # are not random -- they are the tuples, intervals, surds and mixed numbers, i.e.
        # exactly the harder answer types -- so training on bare golds would have silently
        # zeroed the reward on the structured half of MATH and biased the task toward simple
        # scalars. See _boxed_gold below.
        solution = sample.get("solution", "")
        gold = _boxed_gold(extract_boxed_answer(solution))
        out = {
            "messages": [
                {
                    "role": "user",
                    "content": sample["problem"]
                    + "\nPlease put your final answer within \\boxed{}.",
                }
            ],
            "answer": gold,
        }
        if keep_solution:
            # add_special_tokens=False: this is a CONTINUATION of a prompt that has already
            # been through the chat template, so a second BOS or a second turn header would
            # be spliced into the middle of an assistant turn. Measured on the live
            # tokenizer, add_special_tokens=True prepends nothing for this model, but that
            # is a property of one tokenizer and not a guarantee.
            text = gold_template.format(solution=solution) if solution else ""
            ids = tokenizer.encode(text, add_special_tokens=False) if text else []
            if ids and append_eos and tokenizer.eos_token_id is not None:
                ids = list(ids) + [int(tokenizer.eos_token_id)]
            out["gold_ids"] = ids
        return out

    keep = {"messages", "answer"} | ({"gold_ids"} if keep_solution else set())
    dataset = dataset.map(process)
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep])

    n_gold = sum(1 for a in dataset["answer"][:200] if a)
    if n_gold == 0:
        raise ValueError(
            f"no boxed answer found in the first 200 examples of {path}:{split}; the schema "
            "is not what this adapter expects. Training would run with every reward zero, "
            "which looks exactly like a model that cannot solve anything."
        )

    if max_length is not None:

        def filter_length(sample):
            return len(tokenizer.encode(sample["messages"][0]["content"])) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
