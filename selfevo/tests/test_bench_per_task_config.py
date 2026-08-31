"""Per-benchmark generation config: resolution, provenance, and the cap-limited flag.

Generation parameters are a property of the BENCHMARK, not the run. Before this the suite
applied one global `--max-tokens` to every benchmark and recorded only `seed` and
`temperature` in the results row, so two runs generated at different budgets looked
comparable. Measured consequence: truncation at a uniform 3072 ran 7.8% on MATH-500, 15.3% on
OlympiadBench and 36.7% on AIME24, and every truncated generation is graded wrong.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "bench"))
mb = pytest.importorskip("math_bench")


def _args(**kw):
    """A CLI namespace with the shipped defaults, overridable per test."""
    d = dict(max_tokens=3072, temperature=0.0, top_p=1.0, n=1,
             concurrency=32, timeout=600, seed=0)
    d.update(kw)
    return argparse.Namespace(**d)


def test_a_benchmark_without_overrides_gets_the_cli_values():
    """The table states only what DIFFERS; everything else must fall through."""
    got = mb.resolve_params("math500", _args())
    assert got == {"max_tokens": 3072, "temperature": 0.0, "top_p": 1.0, "n": 1,
                   "concurrency": 32, "timeout": 600, "seed": 0}


def test_an_override_wins_over_the_cli_default():
    """Otherwise the table is decorative and every benchmark runs at the global value."""
    assert mb.resolve_params("aime24", _args())["max_tokens"] == 8192
    assert mb.resolve_params("olympiadbench", _args())["max_tokens"] == 8192


def test_unoverridden_keys_still_follow_the_cli_on_an_overridden_benchmark():
    """A partial override must not reset the rest to defaults."""
    got = mb.resolve_params("aime24", _args(temperature=0.7, concurrency=8))
    assert got["temperature"] == 0.7 and got["concurrency"] == 8
    assert got["max_tokens"] == 8192


def test_every_generation_key_is_resolvable():
    """A key missing from the resolved dict would read as None at request time."""
    got = mb.resolve_params("math500", _args())
    assert set(got) == set(mb.GEN_KEYS)


def test_a_typo_in_the_override_table_is_refused(monkeypatch):
    """Silently ignoring it would run the default while the row claimed the override."""
    monkeypatch.setitem(mb.BENCH_OVERRIDES, "math500", {"max_token": 9999})
    with pytest.raises(ValueError, match="unknown generation parameter"):
        mb.resolve_params("math500", _args())


def test_the_override_table_only_names_real_parameters():
    """Guards the shipped table itself, not just the resolver."""
    for bench, over in mb.BENCH_OVERRIDES.items():
        unknown = set(over) - set(mb.GEN_KEYS)
        assert not unknown, f"{bench} names {sorted(unknown)}"


def test_overrides_only_target_benchmarks_in_the_suite():
    """An override for a benchmark nobody runs is dead config that reads as coverage."""
    for bench in mb.BENCH_OVERRIDES:
        assert bench in mb.SUITE, f"{bench} is not in SUITE"


# --------------------------------------------------------------- cap-limited flag ------


def test_cap_limited_threshold_is_a_rate_not_a_count():
    """A count would flag a 500-problem benchmark and miss a 30-problem one."""
    assert 0.0 < mb.CAP_LIMITED_RATE < 1.0


@pytest.mark.parametrize("n_trunc,n,expect", [(0, 100, False), (9, 100, False),
                                              (11, 100, True), (11, 30, True)])
def test_the_flag_fires_on_the_measured_rate(n_trunc, n, expect):
    """36.7% on AIME24 must flag; 7.8% on MATH-500 must not."""
    assert (n_trunc / n > mb.CAP_LIMITED_RATE) is expect
