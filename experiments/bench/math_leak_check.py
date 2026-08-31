#!/usr/bin/env python3
"""Does a MATH training split overlap the MATH-500 problems we evaluate on?

MATH-500 is drawn from MATH's TEST split, so a train/test split should be disjoint -- but
"should be" is exactly the kind of assumption that silently invalidates a result, and several
HF mirrors of MATH differ in how they split. Checked by normalised problem text, not by index.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import sys

from datasets import load_dataset

WS = re.compile(r"\s+")


def norm(s: str) -> str:
    """Whitespace- and case-normalised problem text, for comparing across mirrors."""
    return WS.sub(" ", (s or "").strip().lower())


def main() -> int:
    """Report the overlap between a candidate train split and our MATH-500 eval set."""
    train_id = sys.argv[1] if len(sys.argv) > 1 else "DigitalLearningGmbH/MATH-lighteval"
    tr = load_dataset(train_id, split="train")
    print(f"train: {train_id} n={len(tr)} cols={tr.column_names}")

    eval_path = pathlib.Path(
        os.environ.get("MATH_EVAL_DATA",
                       os.path.expanduser("~/baselines/Absolute-Zero-Reasoner/"
                                          "evaluation/math_eval/eval/data"))
    ) / "math500" / "test.jsonl"
    ev = [json.loads(l) for l in eval_path.open()]
    print(f"eval : {eval_path} n={len(ev)}")

    ev_key = "problem" if "problem" in ev[0] else next(
        k for k in ev[0] if isinstance(ev[0][k], str) and len(ev[0][k]) > 40
    )
    tr_set = {norm(p) for p in tr["problem"]}
    ev_norm = [norm(r[ev_key]) for r in ev]
    overlap = [i for i, p in enumerate(ev_norm) if p in tr_set]

    print(f"\nMATH-500 problems also present in the train split: {len(overlap)} / {len(ev)}")
    if overlap:
        print("LEAKAGE -- training on this split contaminates the eval set.")
        print(f"example eval idx {overlap[0]}: {ev[overlap[0]][ev_key][:110]!r}")
    else:
        print("No overlap: this train split is safe to train on while evaluating on MATH-500.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
