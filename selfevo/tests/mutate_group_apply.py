#!/usr/bin/env python3
"""Mutation-test selfevo/tests/test_group_apply.py against a COPY of the repo.

A copy, not the live checkout: ``group_apply.py`` is imported by the actor's routed path, and
the training supervisor relaunches on process exit, so a mutated file sitting on disk for
even a few seconds could be picked up by a real run.

Usage: ``python3 selfevo/tests/mutate_group_apply.py /path/to/repo/copy``
"""
from __future__ import annotations

import hashlib
import pathlib
import subprocess
import sys

REPO = pathlib.Path(sys.argv[1])
TARGET = REPO / "selfevo/integration/group_apply.py"
TESTS = "selfevo/tests/test_group_apply.py"

# (label, find, replace) -- each is a single-line defect a careless edit could produce.
MUTATIONS = [
    # ---- semantics
    ("SFT adds to the RL advantages instead of replacing them",
     "            torch.full_like(before, float(sft_weight)) * m",
     "            before + torch.full_like(before, float(sft_weight)) * m"),
    ("SFT write ignores the loss mask",
     "torch.full_like(before, float(sft_weight)) * m",
     "torch.full_like(before, float(sft_weight))"),
    ("the mask no longer bounds the write, only its non-zero part",
     "        new = torch.where(m != 0, written, before)",
     "        new = written"),
    ("SKIP writes the SFT weight instead of zero",
     "            else torch.zeros_like(before)",
     "            else torch.full_like(before, float(sft_weight))"),
    ("SKIP is the no-op and RL is applied",
     "        if mode == TrainingMode.RL:",
     "        if mode == TrainingMode.SKIP:"),
    ("SFT and SKIP swapped",
     "            if mode == TrainingMode.SFT",
     "            if mode == TrainingMode.SKIP"),
    # ---- aliasing
    ("the caller's tensor is mutated in place",
     "    out = advantages.clone()",
     "    out = advantages"),
    # ---- group slicing
    ("off-by-one: the last row of every group is left out",
     "        sl = slice(start, start + g)",
     "        sl = slice(start, start + g - 1)"),
    ("group cursor advances one row too far",
     "        start += g",
     "        start += g + 1"),
    # ---- counting
    ("changed count inverted",
     "        changed += int((before != new).any(dim=-1).sum())",
     "        changed += int((before == new).any(dim=-1).sum())"),
    ("every row of a routed group counted, changed or not",
     "        changed += int((before != new).any(dim=-1).sum())",
     "        changed += int(before.shape[0])"),
    ("counts increments by rows instead of by group",
     "        counts[mode] += 1",
     "        counts[mode] += g"),
    ("n_rows reports the group count",
     "        counts=counts, changed_rows=changed, n_groups=len(group_sizes), n_rows=b",
     "        counts=counts, changed_rows=changed, n_groups=len(group_sizes), "
     "n_rows=len(group_sizes)"),
    ("changed_row_fraction divided by groups again (the original defect)",
     "        out[\"route/changed_row_fraction\"] = self.changed_rows / max(self.n_rows, 1)",
     "        out[\"route/changed_row_fraction\"] = self.changed_rows / "
     "max(sum(self.counts.values()), 1)"),
    # ---- validation guards
    ("shape guard dropped",
     "    if advantages.shape != loss_mask.shape:",
     "    if False:"),
    ("rank guard dropped",
     "    if advantages.dim() != 2:",
     "    if False:"),
    ("floating-point guard dropped",
     "    if not torch.is_floating_point(advantages):",
     "    if False:"),
    ("group-size guard dropped",
     "    if any(g < 1 for g in group_sizes):",
     "    if False:"),
    ("partition guard dropped",
     "    if sum(group_sizes) != b:",
     "    if False:"),
    ("one-decision-per-group guard dropped",
     "    if len(modes) != len(group_sizes):",
     "    if False:"),
    ("finite-weight guard dropped",
     "    if not math.isfinite(sft_weight):",
     "    if False:"),
    ("sign guard dropped",
     "    if sft_weight < 0:",
     "    if sft_weight < -1e9:"),
    ("teacher-requiring mode silently treated as SKIP",
     "    if unknown:",
     "    if False:"),
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
