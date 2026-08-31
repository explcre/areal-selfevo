#!/usr/bin/env python3
"""Self-consistency (majority-vote) baseline, as a function of sampling budget.

Why this exists: arXiv 2607.12227 shows that reported gains from evolution methods often
fail to beat plain test-time scaling once budgets are matched, and that prior work mostly
omits the comparison. Any claim we make that routing or evolution improved a model must
therefore be placed against maj@k at the same inference budget, or it is unevaluated.

Reads a generations.jsonl written with --n K (temperature > 0) and reports maj@k for every
k <= K, so a claim costing K samples of compute can be compared against the baseline that
spends K samples doing nothing but voting.

pass@k is also reported, clearly labelled: it needs an oracle to pick the right sample and
is therefore an upper bound, NOT a baseline anyone can deploy. It is printed because the
gap between maj@k and pass@k says how much of the headroom voting actually captures.

Usage: selfconsistency.py <generations.jsonl> [--split report] [--trials 200]
"""
from __future__ import annotations
import argparse, collections, json, math, pathlib, random, re, sys

BENCH = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
from math_bench import grade


def norm(ans: str | None) -> str | None:
    """Normalise a boxed answer for VOTE CLUSTERING only, never for grading.

    Grading stays with the repo's `grade()`, which does symbolic comparison. Voting needs
    to group equal predictions cheaply, and a light textual normalisation is the standard
    choice; being slightly conservative here splits a vote rather than merging two
    different answers, which understates the baseline rather than inflating it.
    """
    if ans is None:
        return None
    s = ans.strip().replace(" ", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.rstrip(".").rstrip("$").lstrip("$")
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("generations")
    ap.add_argument("--split", default="", help="restrict to a committed half")
    ap.add_argument("--trials", type=int, default=200,
                    help="random subsets averaged per k (k < K); 0 uses the first k only")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.generations) if l.strip()]
    if not rows:
        sys.exit("empty generations file -- refusing to report a baseline")

    keep = None
    if a.split:
        sf = BENCH / "math500_split.json"
        keep = set(json.loads(sf.read_text())[a.split])

    by_problem: dict[int, list[dict]] = collections.defaultdict(list)
    for r in rows:
        if keep is not None and r["idx"] not in keep:
            continue
        by_problem[r["idx"]].append(r)
    if not by_problem:
        sys.exit("no problems left after the split filter -- refusing to report")

    ks = sorted({len(v) for v in by_problem.values()})
    if len(ks) != 1:
        print(f"WARNING: uneven sample counts per problem {ks}; using the minimum", file=sys.stderr)
    K = min(ks)
    if K < 2:
        sys.exit(f"only {K} sample(s) per problem; self-consistency needs --n > 1 at "
                 f"temperature > 0. A single greedy sample is not a scaling baseline.")

    # Precompute, per problem: the normalised vote of each sample and whether it is correct.
    prepared = {}
    for idx, rs in by_problem.items():
        votes, correct = [], []
        for r in rs[:K]:
            votes.append(norm(r.get("boxed")))
            correct.append(bool(grade(r["text"], r["gold"])))
        prepared[idx] = (votes, correct)

    rng = random.Random(0)
    print(f"problems={len(prepared)}  samples/problem={K}"
          f"{'  split=' + a.split if a.split else ''}\n")
    print(f"{'k':>3} | {'maj@k':>7} | {'pass@k (oracle)':>16}")
    print("-" * 34)
    for k in range(1, K + 1):
        maj_hits, pass_hits, n = 0.0, 0.0, 0
        for idx, (votes, correct) in prepared.items():
            n += 1
            if k == K or a.trials == 0:
                subsets = [list(range(k))]
            else:
                subsets = [rng.sample(range(K), k) for _ in range(a.trials)]
            m = p = 0.0
            for sub in subsets:
                vs = [votes[i] for i in sub if votes[i] is not None]
                if vs:
                    # Plurality; ties broken by first occurrence, which is arbitrary but
                    # not favourable -- it does not consult correctness.
                    top = collections.Counter(vs).most_common(1)[0][0]
                    winner = next(i for i in sub if votes[i] == top)
                    m += 1.0 if correct[winner] else 0.0
                p += 1.0 if any(correct[i] for i in sub) else 0.0
            maj_hits += m / len(subsets)
            pass_hits += p / len(subsets)
        print(f"{k:>3} | {maj_hits/n:7.3f} | {pass_hits/n:16.3f}")
    print("\nmaj@k is the deployable baseline. pass@k needs an oracle and is an upper bound.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
