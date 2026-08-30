#!/usr/bin/env python3
"""Mutation-test selfevo/tests/test_group_routing.py against a COPY of the repo.

A copy, not the live checkout: the training supervisor relaunches on process exit, so a
mutated actor.py sitting on disk for even a few seconds could be imported by a real run.
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys

REPO = pathlib.Path(sys.argv[1])
TARGET = REPO / "areal/trainer/ppo/actor.py"
TESTS = "selfevo/tests/test_group_routing.py"

# (label, find, replace) -- each is a single-line defect a careless edit could produce.
MUTATIONS = [
    ("prompt tokens also receive the constant",
     'advantages = advantages + row_adv.unsqueeze(1) * data["loss_mask"]',
     'advantages = advantages + row_adv.unsqueeze(1)'),
    ("solved weight ignored, constant hardcoded",
     ').to(row_adv.dtype) * gr.solved_advantage',
     ').to(row_adv.dtype) * 1.0'),
    ("solved branch keyed on solved alone, not silent-and-solved",
     'silent * solved, sizes_t',
     'solved, sizes_t'),
    ("unsolved branch keyed on unsolved alone",
     'silent * unsolved, sizes_t',
     'unsolved, sizes_t'),
    ("enabled flag ignored",
     'if gr is not None and getattr(gr, "enabled", False):',
     'if gr is not None:'),
    ("solved and unsolved swapped",
     'if gr.solved_advantage != 0.0:',
     'if gr.unsolved_advantage != 0.0:'),
    ("routed constant never written back",
     '                        data["advantages"] = advantages',
     '                        pass'),
]


def run_tests() -> bool:
    """True if the suite passes."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x"],
        cwd=REPO, capture_output=True, text=True, timeout=900,
    )
    return r.returncode == 0


def main() -> int:
    original = TARGET.read_text()
    digest = hashlib.sha256(original.encode()).hexdigest()

    if not run_tests():
        print("BASELINE IS RED -- mutation results would be meaningless"); return 2
    print(f"baseline green; {len(MUTATIONS)} mutations\n")

    survivors = []
    for label, find, repl in MUTATIONS:
        if original.count(find) != 1:
            print(f"SKIP  {label}: anchor appears {original.count(find)}x")
            survivors.append((label, "anchor not unique"))
            continue
        TARGET.write_text(original.replace(find, repl, 1))
        passed = run_tests()
        TARGET.write_text(original)
        assert hashlib.sha256(TARGET.read_text().encode()).hexdigest() == digest, "restore failed"
        if passed:
            print(f"SURVIVED  {label}")
            survivors.append((label, "tests still passed"))
        else:
            print(f"killed    {label}")

    print(f"\n{len(MUTATIONS) - len(survivors)}/{len(MUTATIONS)} killed")
    if survivors:
        print("\nSURVIVORS (the tests do not constrain these):")
        for label, why in survivors:
            print(f"  - {label}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
