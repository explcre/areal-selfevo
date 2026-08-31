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
    **kwargs,
):
    """Competition MATH, shaped exactly like the GSM8K RL dataset.

    Args:
        path: HF dataset id, e.g. ``DigitalLearningGmbH/MATH-lighteval``.
        split: ``train`` or ``test``.
        tokenizer: Used only for the length filter.
        max_length: Drop prompts longer than this many tokens.

    Returns:
        A dataset with ``messages`` and ``answer``, matching what ``MathAgent`` reads.

    Raises:
        ValueError: If no example yields a boxed answer -- that means the schema is not what
            this adapter expects, and training would proceed with every reward zero, which is
            indistinguishable from a model that cannot solve anything.
    """
    dataset = load_dataset(path=path, split=split)

    def process(sample):
        # The gold is handed back \boxed{}-wrapped, not bare. math_verify.parse needs LaTeX
        # delimiters to read a structured answer: measured over 400 training examples, a bare
        # gold self-verifies on only 83.8% while \boxed{gold} reaches 100%. The 16% it loses
        # are not random -- they are the tuples, intervals, surds and mixed numbers, i.e.
        # exactly the harder answer types -- so training on bare golds would have silently
        # zeroed the reward on the structured half of MATH and biased the task toward simple
        # scalars. See _boxed_gold below.
        gold = _boxed_gold(extract_boxed_answer(sample.get("solution", "")))
        return {
            "messages": [
                {
                    "role": "user",
                    "content": sample["problem"]
                    + "\nPlease put your final answer within \\boxed{}.",
                }
            ],
            "answer": gold,
        }

    keep = {"messages", "answer"}
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
