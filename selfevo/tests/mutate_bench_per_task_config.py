"""Mutation test for per-benchmark generation config."""
import hashlib, pathlib, subprocess, sys

REPO = pathlib.Path(sys.argv[1])
TARGET = REPO / "experiments/bench/math_bench.py"
TESTS = "selfevo/tests/test_bench_per_task_config.py"

MUTATIONS = [
    ("overrides ignored, so every benchmark runs at the global value",
     "    out.update(over)", "    pass"),
    ("overrides clobber the CLI for keys they do not name",
     "    out = {k: getattr(args, k) for k in GEN_KEYS}\n    out.update(over)",
     "    out = dict(over)"),
    ("unknown key accepted, so a typo runs the default while the row claims the override",
     "    if unknown:", "    if False:"),
    ("resolver returns only the overridden keys",
     "    out = {k: getattr(args, k) for k in GEN_KEYS}",
     "    out = {k: getattr(args, k) for k in GEN_KEYS if k in over}"),
    ("cap-limited threshold set to a count-like value above 1",
     "CAP_LIMITED_RATE = 0.10", "CAP_LIMITED_RATE = 10"),
    ("an override targets a benchmark not in the suite",
     '    "aime24": {"max_tokens": 8192},',
     '    "aime24": {"max_tokens": 8192},\n    "not_a_benchmark": {"max_tokens": 1},'),
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
