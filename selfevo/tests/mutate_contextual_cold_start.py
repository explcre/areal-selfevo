"""Mutation test for the contextual router's cold-start default."""
import hashlib, pathlib, subprocess, sys

REPO = pathlib.Path(sys.argv[1])
TARGET = REPO / "selfevo/compose.py"
TESTS = "selfevo/tests/test_contextual_cold_start.py"

MUTATIONS = [
    ("cold start default removed, so training gets the dataclass 0",
     '    kw.setdefault("cold_start_rounds", _CONTEXTUAL_COLD_START)\n', ''),
    ("cold start set to zero",
     '_CONTEXTUAL_COLD_START = 192', '_CONTEXTUAL_COLD_START = 0'),
    ("default overrides an explicit value instead of deferring to it",
     '    kw.setdefault("cold_start_rounds", _CONTEXTUAL_COLD_START)',
     '    kw["cold_start_rounds"] = _CONTEXTUAL_COLD_START'),
    ("cold start too short to seed every arm",
     '_CONTEXTUAL_COLD_START = 192', '_CONTEXTUAL_COLD_START = 1'),
]


def run_tests() -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x"],
                       cwd=REPO, capture_output=True, text=True, timeout=900)
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
