#!/usr/bin/env python3
"""Mutation-test selfevo/tests/test_dapo_baseline.py against a COPY of the repo.

A copy, not the live checkout: the training supervisor relaunches on process exit, so a
mutated dapo.py sitting on disk for even a few seconds could be imported by a real run.
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys

REPO = pathlib.Path(sys.argv[1])
TARGET = REPO / "selfevo/baselines/dapo.py"
TESTS = "selfevo/tests/test_dapo_baseline.py"

# (label, find, replace) -- each is a defect a careless edit could produce. A faithful DAPO
# baseline is the point of this file, so every one of these silently weakens the arm.
MUTATIONS = [
    ("accept ties: > becomes >=",
     "return bool(group_reward_std(traj) > 0.0 or n_samples == 1)",
     "return bool(group_reward_std(traj) >= 0.0 or n_samples == 1)"),
    ("singleton carve-out dropped",
     "return bool(group_reward_std(traj) > 0.0 or n_samples == 1)",
     "return bool(group_reward_std(traj) > 0.0)"),
    ("empty group treated as a singleton",
     "return bool(group_reward_std(traj) > 0.0 or n_samples == 1)",
     "return bool(group_reward_std(traj) > 0.0 or n_samples <= 1)"),
    ("every group treated as a singleton",
     "return bool(group_reward_std(traj) > 0.0 or n_samples == 1)",
     "return bool(group_reward_std(traj) > 0.0 or n_samples >= 1)"),
    ("decision inverted",
     "return bool(group_reward_std(traj) > 0.0 or n_samples == 1)",
     "return not bool(group_reward_std(traj) > 0.0 or n_samples == 1)"),
    ("sample estimator instead of population",
     "return float(_group_rewards(traj).std(unbiased=False))",
     "return float(_group_rewards(traj).std(unbiased=True))"),
    ("mean compared instead of std",
     "return float(_group_rewards(traj).std(unbiased=False))",
     "return float(_group_rewards(traj).mean())"),
    ("missing rewards accepted instead of raising",
     '''    if "rewards" not in traj:
        raise KeyError(
            "trajectory has no 'rewards'; DAPO dynamic sampling cannot decide without it, "
            "and defaulting to accept would silently degrade this arm to vanilla GRPO"
        )
    r = traj["rewards"]''',
     '    r = traj.get("rewards", torch.tensor([0.0]))'),
    ("float64 cast dropped",
     "return r.flatten().to(torch.float64)",
     "return r.flatten()"),
]


def run_tests() -> bool:
    """True if the suite passes."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x", "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True, timeout=1800,
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
