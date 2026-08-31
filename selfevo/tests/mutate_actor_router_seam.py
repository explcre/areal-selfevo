"""Mutation test for the Router -> advantage seam in ``_route_groups``.

A green suite proves the tests run, not that they constrain anything. Each mutation below is
a one-line defect a careless edit could plausibly produce; a mutation that SURVIVES marks a
property the tests claim to check but do not.

    python selfevo/tests/mutate_actor_router_seam.py /path/to/repo

Run it with the interpreter the tests import ``torch`` from, against the SAME tree pytest
collects. A stale checkout once reported two false survivors on a neighbouring harness.
"""

import hashlib
import pathlib
import subprocess
import sys

REPO = pathlib.Path(sys.argv[1])
TARGET = REPO / "areal/trainer/ppo/actor.py"
TESTS = "selfevo/tests/test_actor_router_seam.py"

# (label, find, replace) -- each is a single-line defect a careless edit could produce.
MUTATIONS = [
    ("router rebuilt every batch, so a learned router never accumulates",
     'router = getattr(self, "_selfevo_router", None)',
     'router = None'),
    ("sft weight hardcoded instead of read from config",
     'sft_weight=float(gr.solved_advantage),',
     'sft_weight=0.5,'),
    ("unit ids drop the batch prefix and collide across batches",
     'unit_id=f"{step}:{i}",',
     'unit_id=f"{i}",'),
    ("routed tensor computed then discarded",
     '        return routed',
     '        return advantages'),
    ("feedback never fires because the pending decision is never read",
     'pending = getattr(self, "_selfevo_pending", None)',
     'pending = None'),
    ("an unregistered router name is silently ignored",
     '            if factory is None:',
     '            if False:'),
    ("the router is consulted but every group is forced to one mode",
     'modes.append(router.route(ctx).argmax())',
     'modes.append("rl")'),
]


def run_tests() -> bool:
    """True if the suite passes."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x"],
        cwd=REPO, capture_output=True, text=True, timeout=1800,
    )
    return r.returncode == 0


def main() -> int:
    """Apply each mutation, run the suite, restore, and report survivors."""
    original = TARGET.read_text()
    digest = hashlib.sha256(original.encode()).hexdigest()
    if not run_tests():
        print("BASELINE IS RED -- fix the suite before trusting any survivor count")
        return 2
    print(f"baseline green; {len(MUTATIONS)} mutations\n")

    survivors = []
    for label, find, repl in MUTATIONS:
        if original.count(find) != 1:
            print(f"SKIP  {label}: anchor appears {original.count(find)}x")
            survivors.append((label, "anchor not unique"))
            continue
        TARGET.write_text(original.replace(find, repl, 1))
        try:
            survived = run_tests()
        finally:
            TARGET.write_text(original)
            assert hashlib.sha256(TARGET.read_text().encode()).hexdigest() == digest, \
                "restore failed"
        print(f"{'SURVIVED' if survived else 'killed  '}  {label}")
        if survived:
            survivors.append((label, "tests still passed"))

    print(f"\n{len(MUTATIONS) - len(survivors)}/{len(MUTATIONS)} killed")
    if survivors:
        print("\nSURVIVORS (the tests do not constrain these):")
        for label, why in survivors:
            print(f"  - {label}: {why}")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
