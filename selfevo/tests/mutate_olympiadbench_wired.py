"""Mutation test for the OlympiadBench wiring."""
import hashlib, pathlib, subprocess, sys

REPO = pathlib.Path(sys.argv[1])
TARGET = REPO / "experiments/bench/math_bench.py"
TESTS = "selfevo/tests/test_olympiadbench_wired.py"

MUTATIONS = [
    ("benchmark dropped from SUITE, so it is never run",
     '         "livemathbench", "olympiadbench"]', '         "livemathbench"]'),
    ("adapter yields an empty answer, grading every problem wrong",
     '            r["answer"] = fa[0] if isinstance(fa, list) and fa else fa',
     '            r["answer"] = ""'),
    ("adapter never fires, so the rows fail the schema check",
     '        if "question" in r and "final_answer" in r and "problem" not in r:',
     '        if "question" in r and "final_answer" in r and "problem" in r:'),
    ("problems silently truncated",
     '    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]',
     '    rows = [json.loads(l) for l in raw.decode().splitlines() if l.strip()][:100]'),
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
