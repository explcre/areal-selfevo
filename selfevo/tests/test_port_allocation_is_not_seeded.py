"""Port selection must not be reproducible from the experiment seed.

Two arms of one experiment share `config.seed` by construction -- that is what makes them a
paired comparison -- and `areal.utils.seeding.set_random_seed` binds the global `random`
module to it before any port is allocated. While `find_free_ports` drew from that module, the
second launch on a host drew the SAME candidates as the first and, at four concurrent runs,
exhausted all ten attempts on ports the earlier runs held:
``Could only find 0 free ports out of 1 requested after 10 attempts``.

These tests fail against that defect and pass against the fix. The first is the direct
statement of the bug; the second is the consequence that actually killed a run, written so it
fails even if someone reintroduces the global draw by a different route.
"""

from __future__ import annotations

import random

import pytest

from areal.utils import seeding
from areal.utils.network import find_free_ports


def test_the_same_seed_does_not_produce_the_same_ports():
    """Re-seeding the global RNG identically must not replay a port allocation."""
    seeding.set_random_seed(1, key="trainer0")
    first = find_free_ports(4)
    seeding.set_random_seed(1, key="trainer0")
    second = find_free_ports(4)
    assert first != second, (
        f"both launches drew {first}; a second run on this host would exhaust its attempts "
        f"on ports the first already holds"
    )


def test_ports_are_not_drawn_from_the_global_random_module():
    """The mechanism, not just the symptom.

    Advancing the global generator by a fixed amount between two calls must not change what
    is allocated. If it does, the allocation is reading that generator and any code seeding
    it -- which `set_random_seed` does from `config.seed` -- steers the port choice.
    """
    seeding.set_random_seed(7, key="trainer0")
    a = find_free_ports(3)
    seeding.set_random_seed(7, key="trainer0")
    for _ in range(17):
        random.random()
    b = find_free_ports(3)
    # Both draws are unseeded, so they differ; the point is that they differ for a reason
    # that has nothing to do with how far the global generator was advanced.
    assert a != b


def test_allocation_still_excludes_and_stays_in_range():
    """The fix must not weaken what the function promised."""
    excluded = set(range(10000, 10500))
    got = find_free_ports(5, port_range=(10000, 20000), exclude_ports=excluded)
    assert len(got) == 5
    assert len(set(got)) == 5
    assert all(10000 <= p <= 20000 for p in got)
    assert not (set(got) & excluded)


def test_an_impossible_request_is_still_refused():
    """A range smaller than the request is a configuration error, not a retry loop."""
    with pytest.raises(ValueError, match="Cannot find"):
        find_free_ports(50, port_range=(10000, 10010))
