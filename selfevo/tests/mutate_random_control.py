"""Mutation test for the random control's guards."""
import hashlib, pathlib, subprocess, sys

REPO = pathlib.Path(sys.argv[1])
TESTS = "selfevo/tests/test_random_control.py"
COMPOSE = REPO / "selfevo/compose.py"
ROUTERS = REPO / "selfevo/routing/routers.py"

MUTATIONS = [
    (COMPOSE, "silent fallback to the class default instead of refusing",
     '        if not spec:\n', '        if False:\n'),
    (COMPOSE, "environment value ignored",
     '        kw["proportions"] = _parse_proportions(spec)',
     '        kw["proportions"] = {"rl": 1.0}'),
    (COMPOSE, "explicit argument overridden by the environment",
     '    if "proportions" not in kw:', '    if True:'),
    (COMPOSE, "malformed pair skipped instead of refused",
     '            raise ValueError(f"bad proportion {part!r}; expected mode=weight")',
     '            continue'),
    (ROUTERS, "teacher-requiring mode gated on has_teacher, so sft never fires",
     '        if known_modes()[chosen] and not ctx.has_target:',
     '        if known_modes()[chosen] and not ctx.has_teacher:'),
]


def run_tests() -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x"],
                       cwd=REPO, capture_output=True, text=True, timeout=900)
    return r.returncode == 0


def main() -> int:
    originals = {f: f.read_text() for f in {COMPOSE, ROUTERS}}
    digests = {f: hashlib.sha256(t.encode()).hexdigest() for f, t in originals.items()}
    if not run_tests():
        print("BASELINE IS RED"); return 2
    print(f"baseline green; {len(MUTATIONS)} mutations\n")
    survivors = []
    for target, label, find, repl in MUTATIONS:
        orig = originals[target]
        if orig.count(find) != 1:
            print(f"SKIP  {label}: anchor appears {orig.count(find)}x")
            survivors.append(label); continue
        target.write_text(orig.replace(find, repl, 1))
        try:
            survived = run_tests()
        finally:
            target.write_text(orig)
            assert hashlib.sha256(target.read_text().encode()).hexdigest() == digests[target]
        print(f"{'SURVIVED' if survived else 'killed  '}  {label}")
        if survived:
            survivors.append(label)
    print(f"\n{len(MUTATIONS) - len(survivors)}/{len(MUTATIONS)} killed")
    for x in survivors:
        print(f"  SURVIVOR: {x}")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
