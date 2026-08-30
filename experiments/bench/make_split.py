#!/usr/bin/env python3
"""Generate the committed MATH-500 search/report split, once.

Why this exists: arXiv 2607.12227 shows that when an evolution loop searches and reports on
the same benchmark, the reported gain is partly overfitting to that task set. Every claim we
make about routing or evolution must therefore search on one half and report on the other.

Why it is committed rather than computed at run time: a split regenerated per run lets
anyone -- including us, without noticing -- reroll until a half flatters the method. The
split is generated once, written to disk, and read thereafter. Regenerating it is a visible
diff.

The dataset checksum is pinned because the split addresses problems BY INDEX. If the
underlying file changes, the indices silently address different problems, and every number
computed against the split becomes wrong with no error.

Run once:  python3 make_split.py --write
"""
from __future__ import annotations
import argparse, collections, hashlib, json, pathlib, random, sys

DATA = pathlib.Path("/home/ubuntu/baselines/Absolute-Zero-Reasoner/evaluation/math_eval/eval/data/math500/test.jsonl")
OUT = pathlib.Path(__file__).resolve().parent / "math500_split.json"
SEED = 20260830


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write the split file")
    args = ap.parse_args()

    raw = DATA.read_bytes()
    md5 = hashlib.md5(raw).hexdigest()
    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]

    # Stratify by (level, subject) so the halves are comparable in difficulty and topic.
    # A uniform random split of 500 items can leave the halves several points apart on
    # level-5 density alone, which would read as a method effect.
    buckets: dict[tuple, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        buckets[(r.get("level"), r.get("subject"))].append(i)

    rng = random.Random(SEED)
    search, report = [], []
    # Alternate within each bucket, and CARRY the parity across buckets so an odd bucket
    # does not hand the next one the same starting side. Without the carry the halves come
    # out 207/293 with level-2 at 20 against 70 -- a split that would read as a method
    # effect. The carry makes the global assignment strictly alternating, so a 500-problem
    # set splits exactly 250/250 and every bucket splits as evenly as it can.
    parity = 0
    for key in sorted(buckets, key=lambda k: (str(k[0]), str(k[1]))):
        idx = buckets[key][:]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            (search if (j + parity) % 2 == 0 else report).append(i)
        parity = (parity + len(idx)) % 2

    search.sort(); report.sort()
    assert set(search) & set(report) == set(), "halves overlap"
    assert set(search) | set(report) == set(range(len(rows))), "halves do not cover the set"

    def dist(ix):
        c = collections.Counter(rows[i]["level"] for i in ix)
        return {str(k): c[k] for k in sorted(c)}

    payload = {
        "dataset": "math500",
        "dataset_md5": md5,
        "n_problems": len(rows),
        "seed": SEED,
        "stratified_by": ["level", "subject"],
        "search": search,
        "report": report,
        "level_distribution": {"search": dist(search), "report": dist(report)},
        "note": ("Indices address rows of test.jsonl in file order. If dataset_md5 stops "
                 "matching, the indices no longer identify the same problems and every "
                 "number computed against this split is invalid."),
    }
    print(f"md5={md5}  n={len(rows)}  search={len(search)}  report={len(report)}")
    print("  search levels:", payload["level_distribution"]["search"])
    print("  report levels:", payload["level_distribution"]["report"])
    if args.write:
        OUT.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {OUT}")
    else:
        print("(dry run; pass --write to create the file)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
