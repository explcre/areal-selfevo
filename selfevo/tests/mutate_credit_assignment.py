"""Mutation test: do the credit-assignment tests actually constrain the bandit?

A characterisation experiment can pass for the wrong reason -- if the router never learned
under ANY signal the "shared scalar" test would still pass, and the finding would be an
artifact. These mutations break learning outright; the informative-signal test must catch it.
"""
import hashlib, pathlib, subprocess, sys

REPO = pathlib.Path(sys.argv[1])
TARGET = REPO / "selfevo/routing/contextual.py"
TESTS = "selfevo/tests/test_credit_assignment.py"

MUTATIONS = [
    ("observe credits nothing, so no signal can ever separate the arms",
     "    def observe(self, outcomes: Mapping[str, DecisionOutcome]) -> None:",
     "    def observe(self, outcomes: Mapping[str, DecisionOutcome]) -> None:\n        return"),
    ("cold start never ends, so selection is round-robin forever",
     "        if n < self.cold_start_rounds:", "        if True:"),
    ("cold start never runs, so arms are never seeded",
     "        if n < self.cold_start_rounds:", "        if False:"),
]


def run_tests() -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x"],
                       cwd=REPO, capture_output=True, text=True, timeout=1200)
    return r.returncode == 0


def main() -> int:
    original = TARGET.read_text()
    digest = hashlib.sha256(original.encode()).hexdigest()
    if not run_tests():
        print("BASELINE IS RED"); return 2
    print(f"baseline green; {len(MUTATIONS)} mutations\n")
    survivors = []
    for label, find, repl in MUTATIONS:
        if original.count(find) != 1:
            print(f"SKIP  {label}: anchor appears {original.count(find)}x")
            survivors.append(label); continue
        TARGET.write_text(original.replace(find, repl, 1))
        try:
            survived = run_tests()
        finally:
            TARGET.write_text(original)
            assert hashlib.sha256(TARGET.read_text().encode()).hexdigest() == digest
        print(f"{'SURVIVED' if survived else 'killed  '}  {label}")
        if survived:
            survivors.append(label)
    print(f"\n{len(MUTATIONS) - len(survivors)}/{len(MUTATIONS)} killed")
    for x in survivors:
        print(f"  SURVIVOR: {x}")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
