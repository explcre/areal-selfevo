#!/usr/bin/env python3
"""Mutation-test selfevo/tests/test_code_policy.py against a COPY of the repo.

A copy, not the live checkout: this module compiles and runs generated source, so a mutated
code_policy.py sitting on disk -- one with the call allowlist or the teacher guard removed
-- must never be importable by a real run, even for the second a test takes.

Every mutation is a single-line loosening of the kind a careless edit would make: one more
node type in the allowlist, one guard turned off, one counter left un-incremented. A
survivor means the suite would have shipped that defect. The file is restored and its
sha256 re-checked after each one.

Usage: ``python selfevo/tests/mutate_code_policy.py /path/to/repo/copy``
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(sys.argv[1]).resolve()
TARGET = REPO / "selfevo/routing/code_policy.py"
TESTS = "selfevo/tests/test_code_policy.py"

# (label, find, replace) -- each is a single-line defect a careless edit could produce.
MUTATIONS = [
    # -- the allowlist itself: one tuple entry is the whole containment claim -------------
    ("Attribute allowlisted, so __class__.__subclasses__ is reachable",
     "    ast.Subscript, ast.Call, ast.Tuple, ast.List,",
     "    ast.Subscript, ast.Call, ast.Tuple, ast.List, ast.Attribute,"),
    ("Import allowlisted, so a policy can import anything",
     "    ast.Subscript, ast.Call, ast.Tuple, ast.List,",
     "    ast.Subscript, ast.Call, ast.Tuple, ast.List, ast.Import, ast.ImportFrom, ast.alias,"),
    ("Lambda and comprehensions allowlisted, so a policy can loop",
     "    ast.Subscript, ast.Call, ast.Tuple, ast.List,",
     "    ast.Subscript, ast.Call, ast.Tuple, ast.List, ast.Lambda, ast.ListComp,\n"
     "    ast.comprehension,"),
    ("Pow dropped, so ordinary arithmetic a policy needs stops working",
     "    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,",
     "    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,"),
    ("the call allowlist widened to the dangerous builtins",
     '_ALLOWED_CALLS = frozenset({"min", "max", "abs", "float", "len", "round"})',
     '_ALLOWED_CALLS = frozenset({"min", "max", "abs", "float", "len", "round", "eval",\n'
     '                            "getattr", "type", "open", "__import__"})'),
    ("the per-call allowlist dropped, so any name may be called",
     "            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_CALLS:",
     "            if not isinstance(node.func, ast.Name):"),
    ("the dunder name check never fires",
     '        if isinstance(node, ast.Name) and node.id.startswith("__"):',
     '        if isinstance(node, ast.Name) and node.id.startswith("____"):'),

    # -- the shape checks: every one of these was a real escape before it was added -------
    ("the decorator check skipped, so a decorator runs at construction",
     "    if fn.decorator_list:",
     "    if False and fn.decorator_list:"),
    ("the nested-function check skipped, so an inner def's defaults run",
     "        if isinstance(node, ast.FunctionDef) and node is not fn:",
     "        if False and isinstance(node, ast.FunctionDef) and node is not fn:"),
    ("a parser RecursionError escapes as itself instead of a rejection",
     "    except (SyntaxError, RecursionError) as exc:",
     "    except SyntaxError as exc:"),
    ("annotations evaluated again, so an annotation is an expression site",
     '    exec(compile(tree, filename="<policy>", mode="exec",  # noqa: S102 - allowlisted AST\n'
     "                 flags=__future__.annotations.compiler_flag), ns)",
     '    exec(compile(tree, filename="<policy>", mode="exec",  # noqa: S102 - allowlisted AST\n'
     "                 dont_inherit=True), ns)"),
    ("more than one top-level statement accepted",
     "    if len(tree.body) != 1 or len(funcs) != 1:",
     "    if len(funcs) != 1:"),
    ("the function name no longer has to be 'route'",
     '    if fn.name != "route":',
     '    if False and fn.name != "route":'),
    ("default arguments accepted, so a default is an expression site",
     "    if a.defaults or a.kw_defaults:",
     "    if False and (a.defaults or a.kw_defaults):"),

    # -- the guards and the accounting ----------------------------------------------------
    ("the teacher guard skipped for the policy's own choice",
     "        if known_modes()[got] and not ctx.has_target:",
     "        if False and known_modes()[got] and not ctx.has_target:"),
    ("the teacher guard skipped for the fallback the router substitutes",
     "        if known_modes()[self.fallback] and not ctx.has_target:",
     "        if False and known_modes()[self.fallback] and not ctx.has_target:"),
    ("the mode check dropped, so the policy's value is returned as-is",
     "        if not isinstance(got, str) or got not in (self.allowed_modes or ()):",
     "        if False and not isinstance(got, str):"),
    ("allowed_modes ignored in favour of the whole registry",
     "        if not isinstance(got, str) or got not in (self.allowed_modes or ()):",
     "        if not isinstance(got, str) or got not in known_modes():"),
    ("an empty allowed_modes read as every mode again",
     "        modes = tuple(known_modes()) if self.allowed_modes is None "
     "else tuple(self.allowed_modes)\n"
     "        if not modes:\n"
     '            raise ValueError("allowed_modes must name at least one mode, or be None for all")\n',
     "        modes = tuple(self.allowed_modes or known_modes())\n"),
    ("invalid_returns stops incrementing",
     "            self.invalid_returns += 1\n", ""),
    ("errors stops incrementing",
     "            self.errors += 1\n", ""),
    ("teacher_blocked stops incrementing",
     "            self.teacher_blocked += 1\n", ""),
    ("calls stops incrementing",
     "        self.calls += 1\n", ""),

    # -- containment and log hygiene ------------------------------------------------------
    ("the policy is handed the caller's own extra dict",
     "        f = {k: float(v) for k, v in ctx.extra.items()}",
     "        f = ctx.extra if isinstance(ctx.extra, dict) else dict(ctx.extra)"),
    ("one builtins table shared by every policy in the process",
     '    ns: dict[str, object] = {"__builtins__": dict(_SAFE_BUILTINS)}',
     '    ns: dict[str, object] = {"__builtins__": _SAFE_BUILTINS}'),
    ("the whole returned value repr'd into the reason, and so into the log",
     '                ctx, f"code policy returned {reprlib.repr(got)}, not an allowed mode"',
     '                ctx, f"code policy returned {got!r}, not an allowed mode"'),
]


def run_tests() -> bool:
    """True if the suite passes."""
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run(
        [sys.executable, "-m", "pytest", TESTS, "-q", "--no-header", "-x", "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True, timeout=900, env=env,
    )
    return r.returncode == 0


def _assert_isolated() -> None:
    """Refuse to run unless the copy, not the live checkout, is what pytest imports."""
    env = dict(os.environ, PYTHONPATH=str(REPO))
    r = subprocess.run(
        [sys.executable, "-c",
         "import selfevo.routing.code_policy as m; print(m.__file__)"],
        cwd=REPO, capture_output=True, text=True, env=env, timeout=300,
    )
    got = pathlib.Path(r.stdout.strip()).resolve()
    if got != TARGET:
        raise SystemExit(f"ISOLATION FAILED: pytest would import {got}, not {TARGET}")
    print(f"isolated: imports resolve to {got}")


def main() -> int:
    _assert_isolated()
    original = TARGET.read_text()
    digest = hashlib.sha256(original.encode()).hexdigest()

    if not run_tests():
        print("BASELINE IS RED -- mutation results would be meaningless")
        return 2
    print(f"baseline green; {len(MUTATIONS)} mutations\n")

    survivors = []
    for label, find, repl in MUTATIONS:
        if original.count(find) != 1:
            print(f"SKIP  {label}: anchor appears {original.count(find)}x")
            survivors.append((label, "anchor not unique"))
            continue
        TARGET.write_text(original.replace(find, repl, 1))
        passed = run_tests()
        TARGET.write_text(original)
        assert hashlib.sha256(TARGET.read_text().encode()).hexdigest() == digest, "restore failed"
        if passed:
            print(f"SURVIVED  {label}")
            survivors.append((label, "tests still passed"))
        else:
            print(f"killed    {label}")

    print(f"\n{len(MUTATIONS) - len(survivors)}/{len(MUTATIONS)} killed")
    if survivors:
        print("\nSURVIVORS (the tests do not constrain these):")
        for label, why in survivors:
            print(f"  - {label}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
