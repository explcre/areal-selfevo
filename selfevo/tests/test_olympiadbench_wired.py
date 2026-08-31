"""OlympiadBench must stay scoreable by the suite that scores our arms.

It is the frontier math target: MATH-500 saturates at 27B (0.966) and AIME is unusable at
1.5B (0.000), so OlympiadBench's 675 problems and 7-point CI are where a method claim on math
can actually be made. It was excluded from SUITE on a premise that had stopped being true --
the schema adapter already existed in `load` -- so these tests pin BOTH that it is wired and
that the grader still accepts its answer format.

The self-verification test is the one that matters. A grader that cannot match a benchmark's
own gold answers reports a low score for every model, which reads as "the model is bad"
rather than "the harness is broken". This project has already been bitten by exactly that on
MATH, where bare golds self-verified at only 83.8%.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[2] / "experiments" / "bench"
sys.path.insert(0, str(BENCH))

mb = pytest.importorskip("math_bench")

try:
    _ROWS = mb.load("olympiadbench")
except FileNotFoundError:  # pragma: no cover - box without the AZR data clone
    _ROWS = None

needs_data = pytest.mark.skipif(_ROWS is None, reason="olympiadbench data not on this box")


def test_olympiadbench_is_in_the_suite():
    """A benchmark absent from SUITE is not run, however well its loader works."""
    assert "olympiadbench" in mb.SUITE


@needs_data
def test_it_loads_the_full_675_problems():
    """A truncated load scores fewer problems and silently narrows the CI."""
    assert len(_ROWS) == 675


@needs_data
def test_every_problem_has_a_nonempty_question_and_answer():
    """final_answer is a one-element list; an empty extraction would grade to zero forever."""
    bad = [r["idx"] for r in _ROWS if not str(r["problem"]).strip() or not str(r["answer"]).strip()]
    assert not bad, f"{len(bad)} rows with an empty field, first: {bad[:5]}"


@needs_data
def test_the_grader_accepts_the_benchmarks_own_gold_answers():
    """100% or the score floor is the harness, not the model.

    OlympiadBench answers arrive wrapped in $...$ and include fractions and surds, which is
    precisely the shape that broke MATH grading before.
    """
    ok = sum(1 for r in _ROWS
             if mb.grade(f"The answer is \\boxed{{{r['answer']}}}", r["answer"]))
    assert ok == len(_ROWS), f"gold self-verify {ok}/{len(_ROWS)}"
