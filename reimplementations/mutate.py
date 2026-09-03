"""Mutation harness: prove the test suite CONSTRAINS the code.

A green suite is not evidence. This project has had five real defects survive 350 passing
tests. So each mutation below is a defect we care about; the suite must go RED for every
one. A mutation that survives is a hole in the tests, and it is reported as such.

Every mutation is applied to a fresh copy of the package, so the working tree is never
left mutated (a mutation harness that races with the tree produces false results).

Usage:
    python mutate.py <python-interpreter>
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent / "ornith_repro"

# (id, file, find, replace, what defect this represents)
MUTATIONS: list[tuple[str, str, str, str, str]] = [
    (
        "M1-empty-buffer-novelty-zero",
        "rewards.py",
        "        return 1.0, True",
        "        return 0.0, True",
        "empty-buffer novelty returns 0.0, annihilating R_task for the first task",
    ),
    (
        "M2-abort-graded-as-failure",
        "rewards.py",
        '    abort_policy: str = "exclude",\n) -> tuple[float, int, int]:',
        '    abort_policy: str = "failure",\n) -> tuple[float, int, int]:',
        "aborted generations default to being graded as wrong answers",
    ),
    (
        "M3-product-becomes-sum",
        "rewards.py",
        "    return V * D * N",
        "    return min(1.0, (V + D + N) / 3.0)",
        "R_task uses a mean instead of the published product, so no factor can gate",
    ),
    (
        "M4-wrong-p-star",
        "rewards.py",
        "P_STAR_PUBLISHED = 0.2",
        "P_STAR_PUBLISHED = 0.5",
        "the one published hyperparameter is wrong",
    ),
    (
        "M5-singleton-group-allowed",
        "grpo.py",
        "    if n < 2:",
        "    if n < 0:",
        "a singleton group is scored as zeros instead of being refused",
    ),
    (
        "M6-buffer-insert-before-scoring",
        "loop.py",
        "    assert_task_not_in_buffer(task, buffer.texts())",
        "    buffer.add(task)\n    assert_task_not_in_buffer(task, [])",
        "task enters the buffer before scoring, forcing N=0 for every task",
    ),
    (
        "M7-proportion-guard-never-fires",
        "guards.py",
        "    if not treatment or not control:",
        "    return\n    if not treatment or not control:",
        "the control-matching guard accepts any control (a guard that cannot fail)",
    ),
    (
        "M8-degeneracy-formula-wrong",
        "grpo.py",
        "    return theta**group_size + (1.0 - theta) ** group_size",
        "    return theta**group_size",
        "degeneracy prediction drops the all-failure term",
    ),
    (
        "M9-provenance-constant",
        "types.py",
        '    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")',
        '    blob = b"constant"',
        "provenance ignores its inputs, so a fabricated artifact verifies",
    ),
    (
        "M10-degenerate-batch-guard-never-fires",
        "guards.py",
        "    if all(g.degenerate for g in groups):",
        "    if False:",
        "the all-degenerate-batch guard cannot fire, permitting a false negative",
    ),
    (
        "M11-degenerate-flag-always-false",
        "grpo.py",
        "    degenerate = std == 0.0",
        "    degenerate = False",
        "constant-reward groups are never flagged as carrying zero gradient",
    ),
    (
        "M12-token-budget-guard-off-by-comparison",
        "guards.py",
        "    if total > served_context_len:",
        "    if total > served_context_len * 100:",
        "the context guard effectively never fires",
    ),
]


def run_suite(root: Path, interpreter: str) -> tuple[bool, str]:
    """Run the test suite in `root`; return (passed, tail_of_output)."""
    proc = subprocess.run(
        [interpreter, "-m", "pytest", "ornith_repro/tests", "-q", "--no-header",
         "-p", "no:cacheprovider", "-x"],
        cwd=root, capture_output=True, text=True, timeout=900,
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip().splitlines()[-1:]


def main() -> int:
    interpreter = sys.argv[1] if len(sys.argv) > 1 else sys.executable

    with tempfile.TemporaryDirectory() as td:
        baseline_root = Path(td) / "baseline"
        baseline_root.mkdir()
        shutil.copytree(SRC, baseline_root / "ornith_repro")
        ok, tail = run_suite(baseline_root, interpreter)
        if not ok:
            print(f"BASELINE IS RED -- mutation results would be meaningless. {tail}")
            return 2
        print(f"baseline: GREEN  {tail}")

    survivors: list[str] = []
    print(f"\n{'mutation':<44} {'defect caught?':<16} detail")
    print("-" * 100)
    for mid, fname, find, repl, desc in MUTATIONS:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "m"
            root.mkdir()
            shutil.copytree(SRC, root / "ornith_repro")
            target = root / "ornith_repro" / fname
            text = target.read_text()
            if find not in text:
                print(f"{mid:<44} {'HARNESS ERROR':<16} pattern not found in {fname}")
                survivors.append(f"{mid} (harness error)")
                continue
            target.write_text(text.replace(find, repl, 1))
            ok, tail = run_suite(root, interpreter)
            caught = not ok
            status = "caught (RED)" if caught else "SURVIVED (GREEN)"
            print(f"{mid:<44} {status:<16} {desc}")
            if not caught:
                survivors.append(mid)

    print("-" * 100)
    if survivors:
        print(f"\n{len(survivors)}/{len(MUTATIONS)} MUTATIONS SURVIVED -- the suite does "
              f"not constrain these defects:")
        for s in survivors:
            print(f"  - {s}")
        return 1
    print(f"\nall {len(MUTATIONS)} mutations caught; the suite constrains every defect tested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
