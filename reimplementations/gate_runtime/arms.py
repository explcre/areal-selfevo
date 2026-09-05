#!/usr/bin/env python3
"""The task pool, the four selectors, and the two-block difficulty estimate they read.

The pool is the MIXED SUBSET, established here from this box's own rollouts rather than
inherited: a problem is in it when, over its 16 samples, at least one is correct and at least
one is not. That is the only place a group-relative advantage can be non-zero, and an
always-solved problem cannot disagree with itself at any group size, so the informative
fraction of a pool is capped by its mixed fraction and no rollout budget can lift it.

Every arm draws from that same pool, so the pool is not what is under test; the selection rule
inside it is.

Two blocks, and which one each arm may read, follow the pre-registration's matching rule 2:

* ``p_a`` -- reps 0-7, the SELECTING block. This is what a difficulty gate can actually see,
  so T and C2 read it.
* ``p_b`` -- reps 8-15, an independent FRESH block. C1 is matched to the fresh-block difficulty
  of the tasks T actually selected, never to the inflated selecting-block value. Matching on
  the selecting block would hand C1 T's own noise and guarantee a tie by construction.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

P_STAR = 0.2      # the one hyperparameter Ornith-1.5 publishes
SIGMA = 0.15      # not published (AMBIGUITIES A2); the value this project has used throughout
BAND = (0.1, 0.9)  # C2's one-line filter
BLOCK_SPLIT = 8   # reps [0, 8) are block A, [8, 16) are block B


@dataclass
class Task:
    """One pool problem with its two independent difficulty estimates."""

    idx: int
    answer: str
    problem: str
    n_a: int          # resolved samples in the selecting block
    c_a: int          # correct among them
    n_b: int
    c_b: int

    @property
    def p_a(self) -> float:
        """Selecting-block success rate; what a gate is allowed to read."""
        return self.c_a / self.n_a

    @property
    def p_b(self) -> float:
        """Fresh-block success rate; the reference C1 is matched on."""
        return self.c_b / self.n_b


def difficulty_reward(p_hat: float, p_star: float = P_STAR, sigma: float = SIGMA) -> float:
    """Ornith-1.5's difficulty kernel, ``exp(-(p-p*)^2 / (2 sigma^2))``.

    Args:
        p_hat: Measured success rate of the task.
        p_star: Target success rate.
        sigma: Kernel width.

    Returns:
        A weight in (0, 1], peaking at ``p_star``.
    """
    return math.exp(-((p_hat - p_star) ** 2) / (2.0 * sigma * sigma))


def build_pool(graded_path: str, problems: list[dict], min_per_block: int = 3,
               truncated: str = "wrong") -> tuple[list[Task], dict]:
    """Read a graded 16-sample file and return the mixed subset plus a census.

    The treatment of a truncated generation is the decision this whole pool turns on, and the
    two answers are not close, so it is a parameter and both are reported:

    * ``wrong`` -- a rollout that hit the token cap did not produce an answer and scores zero.
      This is the benchmark convention, it is what the held-out outcome measures, and it makes
      "sometimes finishes in budget, sometimes does not" a source of gradient. Its cost is that
      p-hat then carries the token budget as well as the mathematics, which is the
      contamination that made the DEFAULT thinking budget unusable; the mixed-subset filter
      removes the worst of it, because a problem that never terminates is not mixed and is
      excluded.
    * ``unknown`` -- truncated samples are dropped, so p-hat is mathematics only. This is the
      convention the difficulty measurement uses, and on a model that terminates two thirds of
      the time it leaves very little mixed.

    An errored request is dropped under both: it is a property of the harness, not the model.

    Args:
        graded_path: JSONL from ``gen_pool.py``, one row per (problem, repetition).
        problems: The benchmark rows, for statements and gold answers.
        min_per_block: Minimum usable samples required in each block.
        truncated: ``wrong`` or ``unknown``.

    Returns:
        ``(pool, census)`` -- the mixed subset, and counts for every category so the
        selection can be reported rather than asserted.
    """
    by_idx: dict[int, dict] = {}
    for line in open(graded_path):
        r = json.loads(line)
        d = by_idx.setdefault(r["idx"], {"a": [], "b": [], "trunc": 0, "err": 0, "n": 0})
        d["n"] += 1
        if r.get("error"):
            d["err"] += 1
            continue
        if r.get("truncated"):
            d["trunc"] += 1
            if truncated == "unknown":
                continue
            (d["a"] if r["rep"] < BLOCK_SPLIT else d["b"]).append(0)
            continue
        (d["a"] if r["rep"] < BLOCK_SPLIT else d["b"]).append(1 if r["correct"] else 0)

    stmt = {p["idx"]: p for p in problems}
    census = {"truncated_convention": truncated,
              "problems_seen": len(by_idx), "n_rows": sum(d["n"] for d in by_idx.values()),
              "truncated_rows": sum(d["trunc"] for d in by_idx.values()),
              "error_rows": sum(d["err"] for d in by_idx.values()),
              "dropped_thin_block": 0, "always_solved": 0, "never_solved": 0, "mixed": 0}
    pool: list[Task] = []
    for idx, d in sorted(by_idx.items()):
        if len(d["a"]) < min_per_block or len(d["b"]) < min_per_block:
            census["dropped_thin_block"] += 1
            continue
        n = len(d["a"]) + len(d["b"])
        c = sum(d["a"]) + sum(d["b"])
        if c == 0:
            census["never_solved"] += 1
            continue
        if c == n:
            census["always_solved"] += 1
            continue
        census["mixed"] += 1
        pool.append(Task(idx=idx, answer=stmt[idx]["answer"], problem=stmt[idx]["problem"],
                         n_a=len(d["a"]), c_a=sum(d["a"]),
                         n_b=len(d["b"]), c_b=sum(d["b"])))
    census["pool_size"] = len(pool)
    return pool, census


class Selector:
    """Base class: an arm's rule for drawing prompts from the pool."""

    name = "base"

    def __init__(self, pool: list[Task], seed: int):
        self.pool = pool
        self.rng = random.Random(seed)

    def draw(self, n: int) -> list[Task]:
        """Draw ``n`` tasks for one training step (with replacement across steps)."""
        raise NotImplementedError

    def describe(self) -> dict:
        """Arm metadata recorded with every run."""
        return {"selector": self.name, "pool_size": len(self.pool)}


class GateSelector(Selector):
    """T: sample in proportion to Ornith's difficulty kernel over the SELECTING block."""

    name = "T_gate"

    def __init__(self, pool, seed, p_star=P_STAR, sigma=SIGMA):
        super().__init__(pool, seed)
        self.p_star, self.sigma = p_star, sigma
        self.w = [difficulty_reward(t.p_a, p_star, sigma) for t in pool]
        if sum(self.w) <= 0:
            raise ValueError("the gate assigns zero weight to every task in the pool")

    def draw(self, n):
        """Weighted draw without replacement within a step, so a step holds n distinct tasks."""
        return _weighted_sample_without_replacement(self.pool, self.w, n, self.rng)

    def describe(self):
        d = super().describe()
        d.update(p_star=self.p_star, sigma=self.sigma,
                 mean_p_a_selected=_wmean([t.p_a for t in self.pool], self.w),
                 mean_p_b_selected=_wmean([t.p_b for t in self.pool], self.w))
        return d


class MatchedRandomSelector(Selector):
    """C1: uniform-random, resampled to T's realised FRESH-block difficulty histogram.

    The gate reads ``p_a`` and therefore chases a noisy statistic; the tasks it lands on have
    some distribution of TRUE difficulty, measured on the independent block. C1 reproduces
    that distribution without the kernel, which is what isolates "targeting 0.2 helps" from
    "the tasks it happens to reach are the right difficulty".

    Matching is by histogram over ``p_b`` bins: within a bin the draw is uniform, so the only
    property C1 inherits from T is the difficulty profile.
    """

    name = "C1_matched_random"

    def __init__(self, pool, seed, target_weights: list[float], nbins: int = 8):
        super().__init__(pool, seed)
        self.nbins = nbins
        self.bin_of = [min(nbins - 1, int(t.p_b * nbins)) for t in pool]
        tgt = [0.0] * nbins
        for b, w in zip(self.bin_of, target_weights):
            tgt[b] += w
        s = sum(tgt) or 1.0
        self.bin_target = [x / s for x in tgt]
        counts = [0] * nbins
        for b in self.bin_of:
            counts[b] += 1
        # Within-bin uniform: a task's weight is its bin's target mass shared equally.
        self.w = [self.bin_target[b] / counts[b] if counts[b] else 0.0 for b in self.bin_of]
        if sum(self.w) <= 0:
            raise ValueError("C1's matched histogram has no mass on any task")

    def draw(self, n):
        """Weighted draw without replacement, weights = matched histogram."""
        return _weighted_sample_without_replacement(self.pool, self.w, n, self.rng)

    def describe(self):
        d = super().describe()
        d.update(nbins=self.nbins, bin_target=self.bin_target,
                 mean_p_a_selected=_wmean([t.p_a for t in self.pool], self.w),
                 mean_p_b_selected=_wmean([t.p_b for t in self.pool], self.w))
        return d


class BandSelector(Selector):
    """C2: keep 0.1 <= p_hat <= 0.9 on the selecting block, uniform within. No target."""

    name = "C2_band"

    def __init__(self, pool, seed, lo=BAND[0], hi=BAND[1]):
        super().__init__(pool, seed)
        self.lo, self.hi = lo, hi
        self.w = [1.0 if lo <= t.p_a <= hi else 0.0 for t in pool]
        self.kept = int(sum(self.w))
        if self.kept == 0:
            raise ValueError("C2's band keeps no task in the pool")

    def draw(self, n):
        """Uniform draw without replacement from the kept band."""
        return _weighted_sample_without_replacement(self.pool, self.w, n, self.rng)

    def describe(self):
        d = super().describe()
        d.update(band=[self.lo, self.hi], kept=self.kept,
                 mean_p_a_selected=_wmean([t.p_a for t in self.pool], self.w),
                 mean_p_b_selected=_wmean([t.p_b for t in self.pool], self.w))
        return d


class RandomRewardSelector(GateSelector):
    """C3: T's selector exactly; the reward is replaced downstream, not the selection.

    Keeping T's selection rule is what makes C3 a control on the REWARD. If C3 also changed
    which tasks it saw, a difference could be either cause. The reward substitution happens in
    the trainer (``--random-reward``), which is where it is visible in the loss.
    """

    name = "C3_random_reward"


def _wmean(xs, ws):
    """Weighted mean, or None when the weights are all zero."""
    s = sum(ws)
    return sum(x * w for x, w in zip(xs, ws)) / s if s else None


def _weighted_sample_without_replacement(items, weights, n, rng):
    """Draw ``n`` distinct items with probability proportional to ``weights``.

    Efraimidis-Spirakis keys: for each positive-weight item draw ``u^(1/w)`` and take the top
    n. Falls back to everything with positive weight when fewer than n such items exist,
    which keeps a small pool usable instead of failing mid-run.
    """
    live = [(i, w) for i, w in enumerate(weights) if w > 0]
    if len(live) <= n:
        return [items[i] for i, _ in live]
    keys = [(rng.random() ** (1.0 / w), i) for i, w in live]
    keys.sort(reverse=True)
    return [items[i] for _, i in keys[:n]]


def make_selector(arm: str, pool: list[Task], seed: int) -> Selector:
    """Build the selector for an arm name.

    Args:
        arm: One of ``T``, ``C1``, ``C2``, ``C3``.
        pool: The mixed subset.
        seed: Draw seed.

    Returns:
        The arm's :class:`Selector`.
    """
    if arm == "T":
        return GateSelector(pool, seed)
    if arm == "C2":
        return BandSelector(pool, seed)
    if arm == "C3":
        return RandomRewardSelector(pool, seed)
    if arm == "C1":
        gate = GateSelector(pool, seed)
        return MatchedRandomSelector(pool, seed, gate.w)
    raise ValueError("unknown arm %r" % arm)
