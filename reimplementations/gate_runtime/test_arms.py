#!/usr/bin/env python3
"""Behavioural tests for the four selectors, on a synthetic pool with known difficulties.

The property that matters is not "C1 is random" but that C1 is a DIFFERENT RULE from T which
nonetheless lands on the same true-difficulty profile. If C1 reproduced T's selecting-block
difficulty it would be a relabelling of T and the control would tie by construction -- the
vacuity the pre-registration warns about. That is asserted here rather than assumed.
"""
import collections
import random
import statistics
import sys

sys.path.insert(0, "/mnt/localssd/gate/code")
import arms


def build_pool(n=120, seed=0):
    """A synthetic pool whose two blocks are independent draws at a known true difficulty."""
    rng = random.Random(seed)
    pool = []
    for i in range(n):
        p = rng.random()
        pool.append(arms.Task(idx=i, answer="x", problem="p%d" % i, n_a=8,
                              c_a=sum(rng.random() < p for _ in range(8)), n_b=8,
                              c_b=sum(rng.random() < p for _ in range(8))))
    return [t for t in pool if 0 < t.c_a + t.c_b < t.n_a + t.n_b]


def test_selectors():
    """T targets p*, C1 matches T's FRESH-block profile without copying its selection."""
    pool = build_pool()
    by = {t.idx: t for t in pool}
    draws = {}
    for arm in ("T", "C1", "C2", "C3"):
        sel = arms.make_selector(arm, pool, seed=7)
        got = []
        for _ in range(4000):
            got += [t.idx for t in sel.draw(8)]
        draws[arm] = got

    m = lambda arm, f: statistics.fmean(f(by[i]) for i in draws[arm])
    t_pa, t_pb = m("T", lambda t: t.p_a), m("T", lambda t: t.p_b)
    c_pa, c_pb = m("C1", lambda t: t.p_a), m("C1", lambda t: t.p_b)
    assert abs(t_pa - arms.P_STAR) < 0.15, "T does not concentrate near p*: %.3f" % t_pa
    assert abs(t_pb - c_pb) < 0.02, "C1 is not matched on the fresh block: %.3f vs %.3f" % (
        t_pb, c_pb)
    assert abs(c_pa - t_pa) > 0.02, (
        "C1 reproduced T's SELECTING-block difficulty, so it is a relabelling of T and the "
        "control ties by construction")
    fT, fC = collections.Counter(draws["T"]), collections.Counter(draws["C1"])
    assert sum(1 for i in fC if fT[i] < fC[i] / 4) > 0, "C1 reaches no task T avoids"
    assert all(arms.BAND[0] <= by[i].p_a <= arms.BAND[1] for i in set(draws["C2"])), \
        "C2 drew outside its band"
    assert draws["C3"] == draws["T"], "C3 must select exactly as T; only its reward differs"


if __name__ == "__main__":
    test_selectors()
    print("ALL SELECTOR ASSERTIONS PASS")
