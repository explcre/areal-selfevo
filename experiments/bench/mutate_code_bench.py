#!/usr/bin/env python3
"""Mutation test for the LiveCodeBench grading and sandbox guards.

Green tests are not evidence that a guard is constrained; this project has shipped 350
passing tests that let five real defects through. So each guard that matters is broken on
purpose and the suite must go red. The mutants are chosen to be the plausible ways this
grader could silently lie: a failure quietly leaving the denominator, a crash or a hang
being graded as a wrong answer instead of as itself, private tests not being run, a
comparator loose enough to accept a wrong number, a sandbox that stops bounding what it
claims to bound, a VALID submission thrown away because a later block was empty (a live
32B run scored one that way), and a dataset the harness cannot find on a box that has it.

The model-identity guards -- refusing to score a model id the endpoint does not serve, and
recording which one answered -- are mutated separately in mutate_model_attribution.py,
because they live half in math_bench.py and need the math suite to judge them.

A mutant that leaves the file byte-identical, or that does not compile, proves nothing.
Both are detected and reported as SKIP rather than counted, because "the tests still
passed" is not a fact about a mutation that was never really applied.
"""

from __future__ import annotations

import ast
import hashlib
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
TESTS = ["test_code_bench.py", "test_code_sandbox.py"]

# (id, target file, description, find, replace)
MUTANTS = [
    # ---------------------------------------------------------------- the accounting
    ("A01", "code_bench.py",
     "unparseable submissions silently leave the denominator",
     '    n_graded = n_problems - counts[ST_GEN_FAILED]',
     '    n_graded = n_problems - counts[ST_GEN_FAILED] - counts[ST_NO_CODE]'),
    ("A02", "code_bench.py",
     "generation failures are counted as graded, hiding a partial outage",
     '        "n_failed": counts[ST_GEN_FAILED],',
     '        "n_failed": 0,'),
    ("A03", "code_bench.py",
     "accuracy_all repeats accuracy, so exclusions stop being visible",
     '        "accuracy_all": (n_pass / n_problems) if n_problems else float("nan"),',
     '        "accuracy_all": acc,'),
    ("A04", "code_bench.py",
     "harness errors are not counted, so a broken grader reports a clean score",
     '        "n_harness_error": counts[ST_HARNESS],',
     '        "n_harness_error": 0,'),
    ("A05", "code_bench.py",
     "a submission passes once ANY test passes",
     '    if base["n_tests_passed"] == base["n_tests"]:',
     '    if base["n_tests_passed"] >= 1:'),

    # ------------------------------------------------------------------ the verdicts
    ("V01", "code_bench.py",
     "a missing code block is not detected, so prose reaches the interpreter",
     '    if not code or not code.strip():',
     '    if False:'),
    ("V02", "code_bench.py",
     "a timeout is not recognised and degrades into a wrong answer",
     '        if r.status == "timeout":',
     '        if False and r.status == "timeout":'),
    ("V03", "code_bench.py",
     "a crash is not recognised and degrades into a wrong answer",
     '        if r.status == "error":',
     '        if False and r.status == "error":'),
    ("V04", "code_bench.py",
     "a sandbox fault is reported as the submission's fault",
     '        if r.status == "harness_error":',
     '        if False and r.status == "harness_error":'),
    ("V05", "code_bench.py",
     "a truncated output is scored a wrong answer instead of an undecidable comparison",
     '        if r.stdout_truncated or any(r.files_truncated.values()):',
     '        if False:'),
    ("V06", "code_bench.py",
     "a functional solution that never returns is treated as returning null",
     '            if raw is None:',
     '            if False:'),

    # -------------------------------------------------------------------- the tests run
    ("T01", "code_bench.py",
     "only the PUBLIC tests are run, so an overfitted submission passes",
     '    tests = problem["tests"][:max_tests] if max_tests else problem["tests"]',
     '    tests = [t for t in problem["tests"] if t["visibility"] == "public"]'),
    ("T02", "code_bench.py",
     "private tests are dropped at load time",
     '        for vis, key in (("public", "public_test_cases"), ("private", "private_test_cases")):',
     '        for vis, key in (("public", "public_test_cases"),):'),
    ("T03", "code_bench.py",
     "a problem with no test cases loads instead of being fatal",
     '        if not tests:',
     '        if False:'),

    # ------------------------------------------------------------------ the comparator
    ("C01", "code_bench.py",
     "the float tolerance also applies to integer answers, so 2024 matches 2025",
     '    if rel_tol <= 0 or _INT_RE.match(want):',
     '    if rel_tol <= 0:'),
    ("C02", "code_bench.py",
     "output line counts are not compared, so a short answer matches",
     '    if len(g) != len(w):\n        return False',
     '    if False:\n        return False'),
    ("C03", "code_bench.py",
     "booleans compare equal to integers",
     '    if isinstance(want, bool) or isinstance(got, bool):',
     '    if False:'),
    ("C04", "code_bench.py",
     "list length is not compared, so a prefix matches",
     '        if not isinstance(got, (list, tuple)) or len(got) != len(want):',
     '        if not isinstance(got, (list, tuple)):'),

    # ------------------------------------------------------------------- the extraction
    ("E01", "code_bench.py",
     "a completion with no fence is executed as if it were code",
     '    if not blocks:\n        return None',
     '    if not blocks:\n        return text'),
    ("E02", "code_bench.py",
     "the FIRST code block wins, grading exploration rather than the answer",
     "        for body in reversed(level):",
     "        for body in level:"),
    ("E03", "code_bench.py",
     "an empty chosen block ends the search, discarding a valid earlier submission",
     "        for body in reversed(level):",
     "        for body in level[-1:]:"),
    ("E04", "code_bench.py",
     "an empty block is submitted for execution instead of refused",
     "            if body.strip():",
     "            if True:"),
    ("E05", "code_bench.py",
     "an untagged block outranks a python-tagged one, so sample output displaces the answer",
     "    for level in (py, [b for _, b in blocks]):",
     "    for level in ([b for _, b in blocks], py):"),

    # ------------------------------------------------------------------- the data path
    ("H01", "code_bench.py",
     "HF_HOME is ignored, so a box that has the dataset still needs LCB_DATA by hand",
     "        roots.append(Path(os.path.expanduser(home)) / _HF_SUBPATH)",
     "        pass"),
    ("H02", "code_bench.py",
     "the default cache outranks HF_HOME, so the wrong snapshot can be scored",
     '''    roots = []
    home = os.environ.get("HF_HOME")
    if home:
        roots.append(Path(os.path.expanduser(home)) / _HF_SUBPATH)
    default = Path(os.path.expanduser(_HF_DEFAULT_HOME)) / _HF_SUBPATH
    if default not in roots:
        roots.append(default)
    return roots''',
     '''    roots = [Path(os.path.expanduser(_HF_DEFAULT_HOME)) / _HF_SUBPATH]
    home = os.environ.get("HF_HOME")
    if home:
        roots.append(Path(os.path.expanduser(home)) / _HF_SUBPATH)
    return roots'''),
    ("H03", "code_bench.py",
     "the first cache root wins even when it holds no snapshot at all",
     '''        snaps = sorted(d for d in root.iterdir() if d.is_dir())
        if snaps:
            return snaps[-1]''',
     '''        snaps = sorted(d for d in root.iterdir() if d.is_dir())
        return snaps[-1]'''),

    # ---------------------------------------------------------------------- the sandbox
    ("S01", "code_sandbox.py",
     "a hung program is not recorded as a timeout",
     '            killed = True\n            _kill_group(proc)',
     '            killed = False\n            _kill_group(proc)'),
    ("S02", "code_sandbox.py",
     "the kill is SIGTERM, which untrusted code can ignore",
     '        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)',
     '        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)'),
    ("S03", "code_sandbox.py",
     "resource limits are never applied",
     '            resource.setrlimit(which, (cap, cap))',
     '            pass'),
    ("S04", "code_sandbox.py",
     "the network monkeypatch is dropped, leaving the weak tier open",
     '            setattr(socket, name, _denied)',
     '            pass'),
    ("S05", "code_sandbox.py",
     "the child inherits this process's environment, credentials and GPUs",
     '                    env=child_env(),',
     '                    env=None,'),
    ("S06", "code_sandbox.py",
     "the read-back prefix is not clipped, so an unbounded answer reaches the comparator",
     '    return data[:cap].decode("utf-8", "replace"), truncated',
     '    return data.decode("utf-8", "replace"), truncated'),
    ("S07", "code_sandbox.py",
     "a caller may overwrite _guard.py and disable every limit",
     '            if os.path.basename(name) != name or name in RESERVED_NAMES:',
     '            if False:'),
    ("S08", "code_sandbox.py",
     "an unknown forced tier silently falls back to the weakest one",
     '            raise ValueError(f"unknown sandbox tier {forced!r}; expected one of {TIERS}")',
     '            return TIER_SUBPROCESS'),
]

# Mutations deliberately NOT applied, with the reason. Reported as SKIP rather than as a
# survivor, because a mutant that was never really applied is not evidence either way.
DECLARED_SKIPS = [
    ("S11", "code_sandbox.py",
     "fh.read(cap + 1) -> fh.read()",
     "Reported SKIP, not SURVIVED. The mutation applies cleanly and compiles, but it is "
     "observationally equivalent: the very next lines compute truncated = len(data) > cap "
     "and return data[:cap], so the VALUE handed back is byte-identical either way. What "
     "changes is only the grader's own peak memory while reading a huge file, which no "
     "assertion in this suite can see. It survived the first round for exactly that "
     "reason, and calling that a surviving defect would be wrong. The observable half of "
     "the same guard is mutated as S06."),
    ("S09", "code_sandbox.py",
     "start_new_session=True -> False",
     "Applying it makes the timeout path call os.killpg on the GRADER's own process "
     "group, which SIGKILLs the pytest runner and this mutation harness along with it. "
     "The suite would go red, so it would score as killed, but by destroying the "
     "measurement rather than by detecting the defect. Not run on a shared box."),
    ("S10", "code_sandbox.py",
     "proc.wait(timeout=...) -> proc.wait()",
     "Removes the only wall clock, so the first infinite-loop test blocks forever and the "
     "run ends at the harness timeout instead of at an assertion. It would be recorded as "
     "killed for the wrong reason and costs the full timeout to learn nothing; S01 "
     "constrains the same path in bounded time."),
]


def run_tests(timeout: int = 900) -> bool:
    """Run the two suites and report whether they are green.

    Args:
        timeout: Seconds before the run is abandoned; a mutant that hangs the suite is
            treated as killed, since a hung grader is exactly the failure being guarded.

    Returns:
        True when the suite passed.
    """
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header",
                            "-x", "-p", "no:randomly"],
                           cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def main() -> int:
    """Apply each mutant, require the suite to go red, and always restore the sources."""
    targets = sorted({m[1] for m in MUTANTS})
    originals = {f: (ROOT / f).read_text() for f in targets}
    digests = {f: hashlib.sha256(s.encode()).hexdigest() for f, s in originals.items()}

    if not run_tests():
        print("BASELINE IS RED -- fix the suite before reading any mutation result")
        return 2
    print(f"baseline green; {len(MUTANTS)} mutants, {len(DECLARED_SKIPS)} declared skips\n")

    killed, survived, skipped = [], [], []
    try:
        for mid, fn, desc, find, repl in MUTANTS:
            src = originals[fn]
            n = src.count(find)
            if n != 1:
                skipped.append((mid, desc, f"anchor appears {n} times, so the edit is "
                                            f"ambiguous or already gone"))
                print(f"  SKIP      {mid}  {desc}")
                continue
            mutated = src.replace(find, repl, 1)
            if mutated == src:
                skipped.append((mid, desc, "replacement is byte-identical: a no-op"))
                print(f"  SKIP      {mid}  {desc}  [no-op]")
                continue
            try:
                ast.parse(mutated)
            except SyntaxError as exc:
                skipped.append((mid, desc, f"mutant does not compile ({exc.msg}), so a red "
                                           f"suite would prove nothing"))
                print(f"  SKIP      {mid}  {desc}  [does not compile]")
                continue
            (ROOT / fn).write_text(mutated)
            try:
                assert (ROOT / fn).read_text() != src, "mutation did not reach disk"
                green = run_tests()
            finally:
                (ROOT / fn).write_text(src)
                assert hashlib.sha256((ROOT / fn).read_text().encode()).hexdigest() \
                    == digests[fn], f"RESTORE FAILED for {fn}"
            (survived if green else killed).append((mid, desc))
            print(f"  {'SURVIVED' if green else 'killed  '}  {mid}  {desc}")
    finally:
        for fn, src in originals.items():
            (ROOT / fn).write_text(src)
            assert hashlib.sha256((ROOT / fn).read_text().encode()).hexdigest() \
                == digests[fn], f"RESTORE FAILED for {fn}"

    print(f"\nkilled {len(killed)}/{len(killed) + len(survived)} applied; "
          f"{len(skipped)} skipped in-flight; {len(DECLARED_SKIPS)} declared skips")
    for mid, desc, why in skipped:
        print(f"  SKIP {mid}: {desc} -- {why}")
    for mid, fn, what, why in DECLARED_SKIPS:
        print(f"  SKIP {mid}: {what} ({fn}) -- {why}")
    for mid, desc in survived:
        print(f"  SURVIVOR {mid}: {desc}")
    return 1 if survived or skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
