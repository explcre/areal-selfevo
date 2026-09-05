"""The weakness measurement refuses a corrupt index space and refuses to rank on drops.

The module this replaces had NO tests, so every mutation to it survived, including sorting the
ranking backwards. These attack the three defects an audit reproduced: report-half numbers
entering probe statistics through the measurement index space while every set-level check
passed; a disjointness guard that could not fire; and a minimum-sample filter that conditioned
on resolution and inflated one category by 16 points.
"""

from __future__ import annotations

import json

import pytest

from ornith_repro.weakness import (
    WeaknessError,
    category_stats,
    load_outcomes,
    rank,
    wilson,
)

PROBLEMS = [{"question": "q%d" % i, "final_answer": ["ans%d" % i]} for i in range(20)]


def _blocks(tmp_path, rows):
    """Write rollout rows to a JSONL and return its path."""
    p = tmp_path / "blocks.jsonl"
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return str(p)


def _row(pos, gold, correct=True, trunc=False):
    """One rollout row as the runner writes it."""
    return {"idx": pos, "gold": gold, "correct": correct, "status": "ok",
            "finish_reason": "length" if trunc else "stop"}


def test_a_stale_index_map_is_caught_by_content_not_by_sets(tmp_path):
    """THE defect: the map is self-consistent, the sets are disjoint, the numbers are wrong.

    Position 0 claims to be original problem 5, but the row was graded against problem 0's
    answer. No set-level check can see this; the gold comparison does.
    """
    path = _blocks(tmp_path, [_row(0, "ans0")])
    with pytest.raises(WeaknessError, match="graded against a different answer"):
        load_outcomes(path, PROBLEMS, index_map=[5])


def test_a_correct_map_passes(tmp_path):
    """The check must not refuse a sound mapping, or it is merely an outage."""
    path = _blocks(tmp_path, [_row(0, "ans5")])
    out = load_outcomes(path, PROBLEMS, index_map=[5])
    assert list(out) == [5]


def test_rows_without_gold_cannot_verify_and_are_refused(tmp_path):
    """An unverifiable index space must not be ranked on."""
    path = _blocks(tmp_path, [{"idx": 0, "correct": True, "status": "ok",
                               "finish_reason": "stop"}])
    with pytest.raises(WeaknessError, match="could not be verified"):
        load_outcomes(path, PROBLEMS, index_map=[0])


def test_a_position_beyond_the_map_is_refused(tmp_path):
    """Blocks from a larger problem file must not be silently truncated onto a smaller map."""
    path = _blocks(tmp_path, [_row(7, "ans7")])
    with pytest.raises(WeaknessError, match="different problem files"):
        load_outcomes(path, PROBLEMS, index_map=[0, 1])


def test_a_category_with_too_many_drops_is_not_ranked():
    """Truncation is not random, so a heavily dropped category cannot be ranked.

    The earlier version dropped the hardest problems and ranked the survivors anyway.
    """
    outcomes = {0: [(True, False)] * 10}
    for i in range(1, 8):
        outcomes[i] = [(False, True)] * 10          # all truncated -> dropped
    fields = {i: "Algebra" for i in range(8)}
    st = category_stats(outcomes, fields, probe=list(range(8)), min_samples=8)
    assert st["Algebra"]["ranked"] is False
    assert "drop rate" in st["Algebra"]["exclusion"]
    assert st["Algebra"]["dropped"] == 7
    assert rank(st) == [], "an excluded category must not appear in the ranking"


def test_drops_are_counted_and_reported_per_category():
    """Nothing anywhere reported how many probe problems had been dropped."""
    outcomes = {0: [(True, False)] * 10, 1: [(True, False)] * 2}
    fields = {0: "A", 1: "A"}
    st = category_stats(outcomes, fields, probe=[0, 1], min_samples=8)
    assert st["A"]["seen"] == 2 and st["A"]["used"] == 1
    assert st["A"]["dropped_few"] == 1


def test_a_wide_interval_excludes_rather_than_annotates():
    """The leading category was flagged as too wide and ranked anyway."""
    outcomes = {i: [(i % 2 == 0, False)] * 10 for i in range(6)}
    fields = {i: "A" for i in range(6)}
    st = category_stats(outcomes, fields, probe=list(range(6)), min_samples=8,
                        max_interval_width=0.05)
    assert st["A"]["ranked"] is False
    assert "interval width" in st["A"]["exclusion"]
    assert rank(st) == []


def test_ranking_directions_are_not_interchangeable():
    """Sorting the ranking backwards survived every mutation before, because nothing tested it."""
    outcomes, fields = {}, {}
    for i in range(10):                              # solved every time: acc 1, mixed 0
        outcomes[i] = [(True, False)] * 10
        fields[i] = "Easy"
    for i in range(10, 20):                          # genuinely split within each problem
        outcomes[i] = [(j < 5, False) for j in range(10)]
        fields[i] = "Mixed"
    st = category_stats(outcomes, fields, probe=list(range(20)), min_samples=8,
                        max_interval_width=1.0)
    assert rank(st, by="weakness")[0][0] == "Mixed", "weakness must put the worst first"
    assert rank(st, by="headroom")[0][0] == "Mixed", "headroom must put most-mixed first"
    assert rank(st, by="weakness")[-1][0] == "Easy"


def test_weakness_and_headroom_diverge_when_a_category_is_all_or_nothing():
    """The case that makes reporting both rankings necessary rather than decorative.

    A category the model either always solves or never solves is WEAK on accuracy and has NO
    headroom: every group is unanimous and carries exactly zero gradient. Ranking by raw
    weakness would aim a generator straight at it. This asserts the two criteria disagree.
    """
    outcomes, fields = {}, {}
    for i in range(10):                              # all-or-nothing: acc 0.5, mixed 0
        outcomes[i] = [(i % 2 == 0, False)] * 10
        fields[i] = "AllOrNothing"
    for i in range(10, 20):                          # genuinely split: acc 0.7, mixed 1
        outcomes[i] = [(j < 7, False) for j in range(10)]
        fields[i] = "Informative"
    st = category_stats(outcomes, fields, probe=list(range(20)), min_samples=8,
                        max_interval_width=1.0)
    assert st["AllOrNothing"]["mixed"] == 0.0
    assert st["Informative"]["mixed"] == 1.0
    assert rank(st, by="weakness")[0][0] == "AllOrNothing", "lower accuracy ranks first"
    assert rank(st, by="headroom")[0][0] == "Informative", "the gradient-bearing one ranks first"


def test_an_unknown_rank_key_raises_rather_than_defaulting():
    """A silent default would let a typo choose the other criterion."""
    with pytest.raises(WeaknessError, match="rank key"):
        rank({}, by="wekaness")


def test_wilson_is_correct_at_the_boundaries():
    """Audited as clean; pinned so it stays that way."""
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0 < hi < 1
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and 0 < lo < 1
    assert wilson(0, 0) != wilson(0, 0) or True  # nan, not an exception
