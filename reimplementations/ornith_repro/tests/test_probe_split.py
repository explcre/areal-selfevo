"""The weakness probe cannot overlap the evaluation set, and the guard proves it by failing.

If the set used to RANK categories by weakness shares problems with the set used to REPORT a
gain, the ranking selects on the outcome and the result is void rather than weak. Nothing in
the output would reveal it. So every relation is asserted on construction, and these tests
make each assertion FAIL on purpose -- a guard only ever observed to pass is not evidence.
"""

from __future__ import annotations

import pytest

from ornith_repro.probe_split import SplitError, assert_probe_disjoint, make_probe_split

SEARCH = list(range(0, 60))
REPORT = list(range(60, 120))
FIELDS = {i: ("Algebra" if i % 3 == 0 else "Geometry" if i % 3 == 1 else "NumberTheory")
          for i in range(120)}


def test_probe_and_train_partition_the_search_half():
    """Nothing may be silently dropped between the two."""
    probe, train = make_probe_split(SEARCH, REPORT, FIELDS)
    assert set(probe) | set(train) == set(SEARCH)
    assert not (set(probe) & set(train))
    assert probe, "probe must not be empty"


def test_probe_never_touches_the_report_half():
    """The relation whose violation would void a capability claim."""
    probe, _ = make_probe_split(SEARCH, REPORT, FIELDS)
    assert not (set(probe) & set(REPORT))


def test_the_guard_fires_when_probe_meets_report():
    """Proven by making it fail: a guard only seen to pass is not evidence."""
    with pytest.raises(SplitError, match="circular"):
        assert_probe_disjoint(probe=[1, 2, 61], train=[3], report=[60, 61], search=[1, 2, 3, 61])


def test_the_guard_fires_when_train_meets_report():
    """Training on the evaluation set is a different disaster, also refused."""
    with pytest.raises(SplitError, match="train and report overlap"):
        assert_probe_disjoint(probe=[1], train=[60], report=[60], search=[1, 60])


def test_the_guard_fires_when_the_partition_is_incomplete():
    """A dropped problem would silently shrink the training pool."""
    with pytest.raises(SplitError, match="partition"):
        assert_probe_disjoint(probe=[1], train=[2], report=[9], search=[1, 2, 3])


def test_the_guard_fires_on_an_empty_probe():
    """An empty probe would rank every category on no evidence at all."""
    with pytest.raises(SplitError, match="empty"):
        assert_probe_disjoint(probe=[], train=[1, 2], report=[9], search=[1, 2])


def test_every_subfield_is_represented_in_the_probe():
    """Weakness is per-category, so a category absent from the probe cannot be ranked."""
    probe, _ = make_probe_split(SEARCH, REPORT, FIELDS)
    fields = {FIELDS[i] for i in probe}
    assert fields == {"Algebra", "Geometry", "NumberTheory"}, fields


def test_the_split_is_reproducible_from_its_seed():
    """A split that moved between runs would make the ranking unauditable."""
    a, _ = make_probe_split(SEARCH, REPORT, FIELDS, seed=7)
    b, _ = make_probe_split(SEARCH, REPORT, FIELDS, seed=7)
    c, _ = make_probe_split(SEARCH, REPORT, FIELDS, seed=8)
    assert a == b
    assert a != c, "different seeds must give different splits, or the seed is ignored"
