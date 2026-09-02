#!/usr/bin/env python3
"""Verify the LiveCodeBench GRADER, not the model: gold must pass, wrong must fail.

A grader is only worth its output if both directions have been measured. This script runs
four checks and exits non-zero if any deviates, so it can be run as a gate rather than
read as a report.

1. SANDBOX  -- :func:`code_sandbox.selftest` on every tier this machine supports. The
   sandbox is a claim about a machine, so it is measured on the machine.

2. REPLAY ORACLE  -- for all 175 problems, a synthesised submission that returns the
   dataset's own expected output, keyed by the SHA-256 of the exact bytes the harness
   delivered. It is NOT a reference solution and computes nothing; ``code_generation_lite``
   ships no reference solutions, so the OlympiadBench check (run the dataset's gold and
   demand 100%) is not directly available and this is the honest substitute. What it does
   prove, across all 7000 test cases, is that stdin arrives byte-exact, functional
   arguments are splatted in the right order, the right method is dispatched, the return
   value round-trips, and the comparator does not mangle either side. A miss is a harness
   bug by construction, because the submission IS the answer key.

3. REFERENCE SOLUTIONS  -- a small set of genuinely computed solutions, hand-written for
   this repository and stored in ``lcb_reference_solutions.json``, covering both the stdin
   and functional families. These are what show the grader accepts code that SOLVES the
   problem rather than code that replays the key. They cover a handful of problems, not
   175, and that is stated rather than glossed.

4. KNOWN-WRONG BATTERY  -- deliberately broken submissions, each of which must FAIL. A
   grader that passes everything is worse than no grader. Two of them are built by
   restricting the replay oracle to a prefix of the tests, which is the check that the
   PRIVATE tests are really being run: an oracle that knows only the public answers must
   fail, and one that knows all but the last test must also fail.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import code_sandbox  # noqa: E402
from code_bench import (  # noqa: E402
    ST_NO_CODE,
    ST_PASS,
    ST_WRONG,
    SandboxLimits,
    grade_many,
    grade_submission,
    load,
    replay_oracle,
)

REFERENCE_FILE = _HERE / "lcb_reference_solutions.json"


def check_sandbox() -> tuple:
    """Run the sandbox selftest on every tier available here.

    Returns:
        ``(ok, rows)`` where ``rows`` is one dict per tier. The weakest tier is expected to
        FAIL the filesystem-isolation part of its own description, and its result is
        reported rather than hidden: a tier that cannot keep its promise must say so.
    """
    rows, ok = [], True
    for tier in code_sandbox.TIERS:
        if tier != code_sandbox.detect_tier() and not _tier_usable(tier):
            rows.append({"tier": tier, "available": False})
            continue
        r = code_sandbox.selftest(tier)
        r["available"] = True
        # Every tier must bound time and memory and report a crash. Only the two namespace
        # tiers are expected to block the network at the kernel; the weak tier's block is a
        # monkeypatch, which this still exercises but does not claim as containment.
        r["ok"] = all(bool(r[k]) for k in ("runs_ok", "stdin_wired", "timeout_killed",
                                           "memory_bounded", "exception_is_error",
                                           "network_blocked"))
        ok = ok and r["ok"]
        rows.append(r)
    return ok, rows


def _tier_usable(tier: str) -> bool:
    """Whether a non-default tier can run at all on this machine."""
    try:
        r = code_sandbox.run_python("print(1)", tier=tier,
                                    limits=SandboxLimits(wall_seconds=10.0))
        return r.status == "ok"
    except Exception:
        return False


def check_replay_oracle(problems, workers, limits, tier) -> tuple:
    """Grade the replay oracle for every problem.

    Args:
        problems: Problems from :func:`code_bench.load`.
        workers: Concurrent sandboxes.
        limits: Sandbox ceilings.
        tier: Isolation tier.

    Returns:
        ``(n_pass, records)``. Anything short of ``len(problems)`` is a harness defect and
        the offending records carry the reason.
    """
    codes = [replay_oracle(p) for p in problems]
    recs = grade_many(problems, codes, workers=workers, limits=limits, tier=tier)
    return sum(1 for r in recs if r["status"] == ST_PASS), recs


def check_references(problems, workers, limits, tier) -> tuple:
    """Grade the hand-written reference solutions.

    Args:
        problems: All problems, from which the covered ones are selected.
        workers: Concurrent sandboxes.
        limits: Sandbox ceilings.
        tier: Isolation tier.

    Returns:
        ``(n_pass, n_total, records)``. ``n_total`` is 0 when the reference file is absent,
        which is reported as "no coverage" rather than as a pass.
    """
    if not REFERENCE_FILE.exists():
        return 0, 0, []
    table = json.loads(REFERENCE_FILE.read_text())
    by_qid = {p["question_id"]: p for p in problems}
    picked, codes = [], []
    for qid, src in sorted(table.items()):
        if qid in by_qid:
            picked.append(by_qid[qid])
            codes.append(src)
    if not picked:
        return 0, 0, []
    recs = grade_many(picked, codes, workers=workers, limits=limits, tier=tier)
    return sum(1 for r in recs if r["status"] == ST_PASS), len(picked), recs


def _prefix_oracle(problem, n_tests):
    """A replay oracle that knows only the first ``n_tests`` answers.

    Used to prove the private tests are actually executed: an oracle holding only the
    PUBLIC answers must fail, because otherwise a submission overfitted to the visible
    examples would score as correct.

    Args:
        problem: A problem dict.
        n_tests: How many of its tests the oracle is given.

    Returns:
        The oracle source.
    """
    trimmed = dict(problem)
    trimmed["tests"] = problem["tests"][:n_tests]
    return replay_oracle(trimmed)


def _unique_input_index(problem):
    """Index of the last test whose INPUT is repeated nowhere else in the problem.

    Needed because the replay oracle is keyed by the input, and this dataset repeats
    inputs: 70 of the 175 problems contain at least one duplicated test input, 372 of the
    7000 cases in total, and in abc387_b only 26 of 43 inputs are distinct. Withholding
    "the last test" from the oracle therefore does NOT reliably make it ignorant -- the
    answer is still in the table under an earlier identical input, so the oracle passes.
    That is the dataset being redundant, not the grader skipping tests, and the check has
    to be built to tell the two apart rather than to trip over the difference.

    Args:
        problem: A problem dict.

    Returns:
        The index, or ``None`` when every input in the problem is duplicated.
    """
    keys = [hashlib.sha256(t["input"].encode()).hexdigest() for t in problem["tests"]]
    counts = collections.Counter(keys)
    for i in range(len(keys) - 1, -1, -1):
        if counts[keys[i]] == 1:
            return i
    return None


def _oracle_omitting(problem, k):
    """A replay oracle given every answer except the one for test ``k``.

    With ``k`` chosen by :func:`_unique_input_index` the submission is genuinely ignorant
    of exactly one test, so it must FAIL -- and it fails only if the grader really runs
    that test instead of stopping once the visible ones are satisfied.

    Args:
        problem: A problem dict.
        k: Index to withhold.

    Returns:
        The oracle source.
    """
    trimmed = dict(problem)
    trimmed["tests"] = [t for i, t in enumerate(problem["tests"]) if i != k]
    return replay_oracle(trimmed)


def wrong_submissions(problem) -> list:
    """The known-wrong battery for one problem.

    Every entry must FAIL. Where the failure bucket is deterministic it is asserted too,
    because "it failed" is satisfied by a grader that fails everything -- the bucket is what
    shows the grader understood WHY.

    Args:
        problem: A problem dict; both families are handled.

    Returns:
        A list of ``(name, code, expected_statuses_or_None)``.
    """
    n_pub = sum(1 for t in problem["tests"] if t["visibility"] == "public")
    oracle = replay_oracle(problem)
    cases = [
        ("no_code_block", None, {ST_NO_CODE}),
        ("empty_completion", "", {ST_NO_CODE}),
        ("prints_nothing", "pass\n", None),
        ("uncaught_exception", "raise RuntimeError('boom')\n", {"runtime_error"}),
        ("syntax_error", "def (:\n  pass\n", {"runtime_error"}),
        ("infinite_loop", "while True:\n    pass\n", {"timeout"}),
        ("memory_hog", "x = bytearray(8 * 1024 ** 3)\n", {"runtime_error"}),
        ("network_attempt",
         "import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=5)\n",
         {"runtime_error"}),
        ("bounded_fork",
         "import os\nfor _ in range(200):\n"
         "    if os.fork() == 0:\n        os._exit(0)\n",
         None),
        ("write_outside_cwd",
         "import os\nopen(os.path.expanduser('~/lcb_should_not_exist'), 'w').write('x')\n",
         None),
        ("exit_zero_no_output", "import sys\nsys.exit(0)\n", None),
        ("oracle_public_only", _prefix_oracle(problem, n_pub), {ST_WRONG, "runtime_error"}),
        # Withholds ONE test whose input appears nowhere else, so the oracle cannot
        # answer it out of a duplicate. This is the check that the LAST private test is
        # really executed rather than assumed.
        ("oracle_missing_one_unique_test",
         _oracle_omitting(problem, _unique_input_index(problem)),
         {ST_WRONG, "runtime_error"}),

    ]
    if problem["func_name"]:
        cases += [
            ("functional_returns_none",
             f"class Solution:\n    def {problem['func_name']}(self, *a):\n"
             f"        return None\n", {ST_WRONG}),
            ("functional_wrong_method_name",
             "class Solution:\n    def not_the_method(self, *a):\n        return 0\n",
             {"runtime_error"}),
            # Expected to PASS, and listed here rather than elsewhere so the asymmetry is
            # measured rather than assumed: a functional verdict is the RETURNED value, so
            # a submission that also prints debug output is still correct. The same extra
            # print sinks a stdin submission, which the stdin branch below asserts.
            ("functional_stray_stdout_ignored", oracle + "\nprint('EXTRA')\n",
             {ST_PASS}),
        ]
    else:
        cases += [
            ("echoes_input", "import sys\nprint(sys.stdin.read())\n", None),
            ("stdin_oracle_plus_extra_line", oracle + "\nprint('EXTRA')\n", {ST_WRONG}),
        ]
    return cases


def check_wrong(problems, limits, tier, workers) -> tuple:
    """Run the known-wrong battery on one stdin and one functional problem.

    Args:
        problems: All problems.
        limits: Sandbox ceilings.
        tier: Isolation tier.
        workers: Concurrent sandboxes.

    Returns:
        ``(ok, rows)``; ``ok`` is False if any wrong submission PASSED or landed in an
        unexpected bucket.
    """
    picks = []
    for pred in (lambda p: not p["func_name"], lambda p: bool(p["func_name"])):
        chosen = next((p for p in problems if pred(p) and len(p["tests"]) >= 3
                       and _unique_input_index(p) is not None), None)
        if chosen:
            picks.append(chosen)
    rows, ok = [], True
    for problem in picks:
        cases = wrong_submissions(problem)
        recs = grade_many([problem] * len(cases), [c[1] for c in cases], workers=workers,
                          limits=limits, tier=tier)
        for (name, _, expect), rec in zip(cases, recs):
            passed = rec["status"] == ST_PASS
            bucket_ok = expect is None or rec["status"] in expect
            # Almost every entry must FAIL. The one documented exception asserts a PASS,
            # and is written as {ST_PASS} so the rule stays visible in the case table
            # instead of living in a comment.
            expect_pass = expect == {ST_PASS}
            row = {"problem": problem["question_id"], "case": name,
                   "status": rec["status"], "passed": passed,
                   "expected": sorted(expect) if expect else "any failure",
                   "ok": bucket_ok if expect_pass else ((not passed) and bucket_ok),
                   "detail": rec.get("detail", "")[:200]}
            ok = ok and row["ok"]
            rows.append(row)
    return ok, rows


def main() -> int:
    """Run every check and return 0 only if all of them held."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", default="livecodebench_v6")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--difficulty", default="")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--wall-seconds", type=float, default=15.0)
    ap.add_argument("--memory-mb", type=int, default=2048)
    ap.add_argument("--output-mb", type=int, default=16)
    ap.add_argument("--sandbox", default="")
    ap.add_argument("--skip-sandbox", action="store_true")
    ap.add_argument("--out", default="", help="JSON report")
    args = ap.parse_args()

    tier = args.sandbox or code_sandbox.detect_tier()
    limits = SandboxLimits(wall_seconds=args.wall_seconds,
                           memory_bytes=args.memory_mb * 1024 * 1024,
                           output_bytes=args.output_mb * 1024 * 1024)
    report = {"bench": args.bench, "tier": tier,
              "describes": code_sandbox.describe_tier(tier)}
    ok = True

    if not args.skip_sandbox:
        t0 = time.time()
        s_ok, s_rows = check_sandbox()
        report["sandbox"] = s_rows
        ok = ok and s_ok
        print(f"SANDBOX  {'ok' if s_ok else 'FAILED'}  ({time.time() - t0:.0f}s)")
        for r in s_rows:
            print("   ", json.dumps(r))

    problems = load(args.bench, difficulty=args.difficulty, limit=args.limit)
    print(f"loaded {len(problems)} problems, "
          f"{sum(len(p['tests']) for p in problems)} test cases")

    t0 = time.time()
    n_ok, recs = check_replay_oracle(problems, args.workers, limits, tier)
    report["replay_oracle"] = {"pass": n_ok, "total": len(problems),
                               "failures": [r for r in recs if r["status"] != ST_PASS]}
    ok = ok and n_ok == len(problems)
    print(f"REPLAY ORACLE  {n_ok}/{len(problems)}  ({time.time() - t0:.0f}s)")
    for r in recs:
        if r["status"] != ST_PASS:
            print(f"    MISS {r['question_id']} [{r['difficulty']}/{r['platform']}] "
                  f"{r['status']} test {r['first_fail']}: {r['detail'][:300]}")

    t0 = time.time()
    n_ref, n_ref_total, ref_recs = check_references(problems, args.workers, limits, tier)
    report["references"] = {"pass": n_ref, "total": n_ref_total,
                            "records": ref_recs}
    ok = ok and (n_ref == n_ref_total)
    print(f"REFERENCE SOLUTIONS  {n_ref}/{n_ref_total}  ({time.time() - t0:.0f}s)"
          + ("  [no reference solutions on file: this direction is UNVERIFIED]"
             if n_ref_total == 0 else ""))
    for r in ref_recs:
        if r["status"] != ST_PASS:
            print(f"    MISS {r['question_id']} {r['status']} test {r['first_fail']}: "
                  f"{r['detail'][:300]}")

    t0 = time.time()
    w_ok, w_rows = check_wrong(problems, limits, tier, args.workers)
    report["known_wrong"] = w_rows
    ok = ok and w_ok
    print(f"KNOWN-WRONG  {sum(1 for r in w_rows if r['ok'])}/{len(w_rows)}  "
          f"({time.time() - t0:.0f}s)")
    for r in w_rows:
        flag = "ok  " if r["ok"] else "BAD "
        print(f"    {flag} {r['problem']:<12} {r['case']:<26} -> {r['status']}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}")
    print("SELFCHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
