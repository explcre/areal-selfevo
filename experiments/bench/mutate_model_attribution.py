#!/usr/bin/env python3
"""Mutation test for the model-attribution guards in math_bench.py and code_bench.py.

A live run against a served 32B found the exposure these guards close: the request payload
names a MODEL ID and nothing else, an adapter is reachable only because sglang registers
``--lora-paths NAME=path`` as a model id, and an id the server has never heard of is
answered HTTP 200 by the BASE model. The harness recorded no model id at all, and its own
default was such a name, so a run left on the default would have scored base weights with
nothing in results.json to say so.

Green tests are not evidence that the fix is constrained -- this project has shipped 350
passing tests that let five real defects through -- so every part of it is broken on
purpose here and the suite must go red. The mutants are the ways the fix could rot: the
check skipped, the answer assumed instead of fetched, the record dropped, the ORDER
inverted so the endpoint is contacted before the missing flag is noticed, and the default
put back.

MUTATES A COPY, NOT THE LIVE CHECKOUT. Pass the bench directory of a copied tree as
argv[1]; the run refuses unless pytest really imports the harness from there, and every
source file is restored and re-checksummed after each mutant. A mutant that leaves a file
byte-identical, or that does not compile, is reported SKIP rather than counted: "the tests
still passed" is not a fact about a mutation that was never applied.
"""

from __future__ import annotations

import ast
import hashlib
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 \
    else pathlib.Path(__file__).resolve().parent
TESTS = ["test_math_bench.py", "test_code_bench.py"]

_VERIFY_CALL_MATH = ('        params.update(await verify_model(session, args.base_url,\n'
                     '                                         getattr(args, "model", None)))')
_VERIFY_CALL_CODE = ('        params.update(await verify_model(session, args.base_url,\n'
                     '                                         getattr(args, "model", None)))')

# (id, target file, description, find, replace)
MUTANTS = [
    # ------------------------------------------------- the check itself (math_bench)
    ("A01", "math_bench.py",
     "an unserved id is accepted, so the BASE model is scored in silence",
     "    if model not in served:",
     "    if False:"),
    ("A02", "math_bench.py",
     "the served list is assumed rather than asked for, so the check can never fail",
     "        served = await list_served_models(session, base_url, timeout)",
     "        served = [model]"),
    ("A03", "math_bench.py",
     "a missing model id is not itself a refusal",
     "    if not model:",
     "    if False:"),
    # The bare `if r.status != 200:` also appears in generate(), so this anchor carries the
    # line above it: an ambiguous anchor is skipped, and a skip proves nothing.
    ("A04", "math_bench.py",
     "a non-200 model list is treated as a model list",
     "        async with session.get(url, timeout=timeout) as r:\n"
     "            if r.status != 200:",
     "        async with session.get(url, timeout=timeout) as r:\n"
     "            if False:"),
    ("A05", "math_bench.py",
     "a body that is not a model list is not rejected",
     "    if not isinstance(data, list):",
     "    if False:"),
    ("A06", "math_bench.py",
     "an endpoint that lists no ids passes as a verified endpoint",
     "    if not ids:",
     "    if False:"),
    ("A07", "math_bench.py",
     "an unreachable endpoint raises past the refusal instead of stopping the run",
     "    except RuntimeError as exc:",
     "    except ValueError as exc:"),

    # ------------------------------------------------------ the record (math_bench)
    ("A08", "math_bench.py",
     "run_bench verifies but records nothing, so no score is attributable",
     _VERIFY_CALL_MATH,
     '        await verify_model(session, args.base_url, getattr(args, "model", None))'),
    ("A09", "math_bench.py",
     "run_bench records what was REQUESTED, not what the endpoint answered",
     _VERIFY_CALL_MATH,
     '        params.update({"model": args.model, "endpoint": url, "served_models": []})'),

    # ------------------------------------------------------ the default (math_bench)
    ("A10", "math_bench.py",
     "the model flag gets a default back, which IS the silent-wrong-model bug",
     '    ap.add_argument("--model", default=None,',
     '    ap.add_argument("--model", default="evalmodel",'),
    ("A11", "math_bench.py",
     "main() runs with no model id named",
     "    if not args.model:",
     "    if False:"),

    # -------------------------------------------------- the same three (code_bench)
    ("M01", "code_bench.py",
     "the model id is verified but never recorded, so no score is attributable",
     _VERIFY_CALL_CODE,
     '        await verify_model(session, args.base_url, getattr(args, "model", None))'),
    ("M02", "code_bench.py",
     "the id is recorded but never verified: the BASE model can be scored in silence",
     _VERIFY_CALL_CODE,
     '        params.update({"model": args.model, "endpoint": url, "served_models": []})'),
    ("M03", "code_bench.py",
     "the model flag gets a default back",
     '    ap.add_argument("--model", default=None,',
     '    ap.add_argument("--model", default="evalmodel",'),
    ("M04", "code_bench.py",
     "a generating run with no model named reaches the endpoint before it stops",
     "    if not args.from_generations and not args.model:",
     "    if False:"),
    ("M05", "code_bench.py",
     "a regrade claims an endpoint it never called",
     '                params.update({"model": None, "endpoint": None, "served_models": None})',
     '                params.update({"model": args.model, "endpoint": args.base_url,\n'
     '                               "served_models": []})'),
    ("M06", "code_bench.py",
     "a regrade forgets which recorded file it graded",
     '            params["generations_source"] = args.from_generations or None',
     '            params["generations_source"] = None'),
]

# Mutations deliberately NOT applied, with the reason. Reported as SKIP, never as a
# survivor: a mutant that was never really applied is not evidence either way.
DECLARED_SKIPS = [
    ("A12", "math_bench.py",
     "MODELS_TIMEOUT -> params['timeout']",
     "Reported SKIP, not SURVIVED. The mutation compiles and applies, but no assertion in "
     "this suite can see it: the stub endpoint answers instantly and the local server "
     "answers in milliseconds, so a 60-second budget and a 600-second one are "
     "observationally identical. What it changes is how long a WEDGED endpoint holds a "
     "run open before refusing, which is a property of a hang, and testing it means "
     "actually hanging for minutes."),
]


def run_tests(timeout: int = 1200) -> bool:
    """Run the two suites and report whether they are green.

    Args:
        timeout: Seconds before the run is abandoned; a mutant that hangs the suite counts
            as killed, since a harness that hangs on a bad model id is also a failure.

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


def _assert_isolated() -> None:
    """Refuse to run unless pytest would import the harness from ROOT, not the checkout."""
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, '.'); import math_bench, code_bench; "
         "print(math_bench.__file__); print(code_bench.__file__)"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    got = [pathlib.Path(x).resolve() for x in r.stdout.split()]
    want = [ROOT / "math_bench.py", ROOT / "code_bench.py"]
    if got != want:
        raise SystemExit(f"ISOLATION FAILED: imports resolve to {got}, not {want}")
    print(f"isolated: mutating {ROOT}")


def main() -> int:
    """Apply each mutant, require the suite to go red, and always restore the sources."""
    _assert_isolated()
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
