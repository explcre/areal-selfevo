#!/usr/bin/env python3
"""Generate the committed OlympiadBench search/report split, once.

Same rationale as ``make_split.py`` for MATH-500 (arXiv 2607.12227): searching and reporting
on one task set overstates the gain, so every claim searches on one half and reports on the
other. Committed rather than computed per run, because a split regenerated per run lets anyone
-- including us, without noticing -- reroll until a half flatters the method. Regenerating it
is then a visible diff.

Written BEFORE any OlympiadBench number existed on this box, which is the only moment at which
generating it is above suspicion.

OlympiadBench carries none of the fields MATH-500 stratifies on: ``level`` and ``subject`` are
absent, and the usable strata are ``subfield`` (Algebra 264, Combinatorics 154, Geometry 129,
Number Theory 128) and ``answer_type`` (Numerical 572, Expression 64, Tuple 33, Interval 6).
Both are used, because answer type drives how often the grader can parse an answer at all and
a half skewed on it would read as a method effect.

675 is ODD, so the halves are 338/337 rather than the 250/250 MATH-500 gets. The alternation
carries parity across buckets exactly as the MATH-500 version does, so every bucket splits as
evenly as it can and the global split is off by exactly one.

Run once:  python3 make_olympiadbench_split.py --write
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import random
import sys

DATA = pathlib.Path(os.path.expanduser("~/evaldata/olympiadbench/test.jsonl"))
OUT = pathlib.Path(__file__).resolve().parent / "olympiadbench_split.json"
SEED = 20260830  # the same seed the MATH-500 split used; no reroll was performed


def main() -> int:
    """Build the split and write it, or print what it would be."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write the split file")
    args = ap.parse_args()

    raw = DATA.read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]

    buckets: dict[tuple, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        buckets[(r.get("subfield"), r.get("answer_type"))].append(i)

    rng = random.Random(SEED)
    search: list[int] = []
    report: list[int] = []
    parity = 0
    for key in sorted(buckets, key=lambda k: (str(k[0]), str(k[1]))):
        idx = buckets[key][:]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            (search if (j + parity) % 2 == 0 else report).append(i)
        parity = (parity + len(idx)) % 2

    search.sort()
    report.sort()
    assert not (set(search) & set(report)), "halves overlap"
    assert set(search) | set(report) == set(range(len(rows))), "halves do not cover the set"

    def dist(ix, field):
        c = collections.Counter(str(rows[i].get(field)) for i in ix)
        return {k: c[k] for k in sorted(c)}

    payload = {
        "dataset": "olympiadbench",
        "dataset_md5": md5,
        "n_problems": len(rows),
        "seed": SEED,
        "stratified_by": ["subfield", "answer_type"],
        "search": search,
        "report": report,
        "subfield_distribution": {"search": dist(search, "subfield"),
                                  "report": dist(report, "subfield")},
        "answer_type_distribution": {"search": dist(search, "answer_type"),
                                     "report": dist(report, "answer_type")},
        "note": ("Indices address rows of test.jsonl in file order. If dataset_md5 stops "
                 "matching, the indices address different problems and every number "
                 "computed against this split is wrong; math_bench verifies it on load. "
                 "Generated before any OlympiadBench score existed on this box."),
    }

    print(f"{DATA}\n  md5 {md5}\n  {len(rows)} problems -> "
          f"search {len(search)} / report {len(report)}")
    for field in ("subfield", "answer_type"):
        print(f"  {field}:")
        print(f"    search {dist(search, field)}")
        print(f"    report {dist(report, field)}")
    if args.write:
        OUT.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {OUT}")
    else:
        print("(dry run; pass --write to commit the split)")
    return 0


sys.exit(main())
