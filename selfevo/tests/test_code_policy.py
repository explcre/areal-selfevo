"""Tests for a routing policy written as generated source.

This module runs model-generated Python, and its own docstring says plainly that the AST
allowlist is a CORRECTNESS boundary and not a security one. So these tests do not claim an
adversary cannot escape. They pin the three things that *were* claimed, and the one that
was not:

* CONTAINMENT -- a policy sees the dict it was handed and nothing else. No import by any
  route, no attribute, no name beginning ``__``, no builtin outside the numeric handful.
* LOUDNESS -- every way a policy can go wrong lands on its own counter and says so in
  ``reason``. A broken arm must never look like a conservative one, which is the whole
  reason this router counts at all.
* THE GUARDS APPLY TO THE ROUTER'S OWN ANSWER -- not only to the policy's. The teacher
  guard was enforced on the mode a policy returned and skipped on the fallback the router
  substituted, so a ``fallback='sft'`` arm emitted SFT for target-free units on exactly the
  path where it had just logged that the policy was wrong. Swept below.
* TERMINATION -- **not** claimed, and the table says so out loud. ``9 ** 9 ** 9`` and
  ``[0] * 10 ** 9`` are allowlisted arithmetic; removing them means removing ``**`` and
  ``*``. They are pinned as ACCEPTED BY VALIDATION and never executed, because executing
  them is the finding.

The adversarial corpus is a TABLE of observed outcomes, not a list of rejections. Each entry
records whether a construct is refused at validation, raised at runtime and counted,
returned something unusable and counted, or genuinely ran. A loosening -- adding a node
type, dropping the ``__`` check, dropping the call allowlist, forgetting a counter -- moves
an entry from one column to another and fails here. Sweeping rather than spot-checking is
the point: a spot check on ``import os`` says nothing about ``__import__``, ``f"{x}"``,
a decorator, or an annotation, and each of those was a separate hole.
"""

from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path

import pytest

from selfevo.routing.base import (
    Granularity,
    Router,
    RoutingContext,
    TrainingMode,
    known_modes,
)
from selfevo.routing.code_policy import (
    POLICY_SIGNATURE,
    CodePolicyRouter,
    PolicyRejected,
    compile_policy,
    validate_policy_source,
)
from selfevo.routing import code_policy

GRANS = tuple(Granularity)
RATES = (0.0, 0.25, 0.5, 0.75, 1.0)

# The four ways a policy can end up. Named, because "it was rejected" and "it raised and we
# counted it" are different guarantees and a test that conflates them proves neither.
REJECTED = "rejected at validation"
ERROR = "raised at runtime, counted"
INVALID = "returned a non-mode, counted"
BLOCKED = "teacher guard, counted"
RAN = "ran"


def ctx(p=0.5, *, teacher=True, g=4, gran=Granularity.SAMPLE, extra=None):
    """A routing context, built by keyword so no field order can silently rebind."""
    return RoutingContext(
        solve_rate=p, group_size=g, granularity=gran,
        has_teacher=teacher, extra=extra if extra is not None else {},
    )


def mode_of(d):
    """The single mode a hard routing decision selected."""
    assert len(d.weights) == 1, f"expected a one-hot decision, got {d.weights}"
    return next(iter(d.weights))


def body(*lines, sig=POLICY_SIGNATURE):
    """A policy built from its body, so the corpus below reads as the rule, not the syntax."""
    return sig + "\n" + "".join(f"    {ln}\n" for ln in lines)


def outcome(source, *, c=None, **kw):
    """Build a router from ``source``, route one unit, and say which column it landed in.

    Returns:
        ``(outcome, mode_or_message, health)``. The mode is the emitted mode when anything
        ran, and the rejection message when validation refused the source.
    """
    try:
        r = CodePolicyRouter(source=source, **kw)
    except PolicyRejected as exc:
        return REJECTED, str(exc), {}
    d = r.route(c if c is not None else ctx())
    h = r.health()
    got = ERROR if h["errors"] else (
        INVALID if h["invalid_returns"] else (
            BLOCKED if h["teacher_blocked"] else RAN))
    return got, mode_of(d), h


# ------------------------------------------------------------------- the adversarial table

# (name, source, expected outcome, expected mode when it ran). Routed against a unit that
# HAS a teacher, so the teacher guard is not what is under test here -- validity is.
_CORPUS: tuple[tuple[str, str, str, str | None], ...] = (
    # -- policies that are supposed to work, so the table cannot pass by rejecting all ----
    ("plain constant", body('return "rl"'), RAN, TrainingMode.RL),
    ("threshold", body('if features["solve_rate"] > 0.4:', '    return "sft"',
                       'return "rl"'), RAN, TrainingMode.SFT),
    ("arithmetic and calls", body('x = min(1.0, max(0.0, features["solve_rate"] / 2))',
                                  'return "rl" if abs(x) > 0 else "skip"'),
     RAN, TrainingMode.RL),
    ("round with a positional ndigits", body('return "rl" if round(1.234, 2) > 0 else "skip"'),
     RAN, TrainingMode.RL),
    ("len of a literal tuple", body('return "rl" if len((1, 2)) == 2 else "skip"'),
     RAN, TrainingMode.RL),
    ("chained comparison", body('return "rl" if 0 <= features["solve_rate"] <= 1 else "skip"'),
     RAN, TrainingMode.RL),
    ("boolean operators", body('return "rl" if not (0 and 1) or 1 else "skip"'),
     RAN, TrainingMode.RL),
    ("group_size and has_target are exposed",
     body('return "sft" if features["has_target"] and features["group_size"] > 1 else "rl"'),
     RAN, TrainingMode.SFT),
    ("tuple unpacking", body("a, b = 1, 2", 'return "rl" if a < b else "skip"'),
     RAN, TrainingMode.RL),
    ("chained assignment", body("a = b = 1", 'return "rl" if a == b else "skip"'),
     RAN, TrainingMode.RL),
    ("a docstring", body('"what this rule does"', 'return "rl"'), RAN, TrainingMode.RL),
    ("semicolons on one line", "def route(features): x = 1; return 'rl'\n",
     RAN, TrainingMode.RL),
    ("a one-line body", "def route(features): return 'rl'\n", RAN, TrainingMode.RL),
    ("CRLF line endings", "def route(features):\r\n    return 'rl'\r\n", RAN, TrainingMode.RL),
    ("a plain annotation", "def route(features: float) -> str:\n    return 'rl'\n",
     RAN, TrainingMode.RL),
    ("the argument may be named anything", "def route(x):\n    return 'rl'\n",
     RAN, TrainingMode.RL),
    ("mutating the features dict", body('features["solve_rate"] = 1.0',
                                        'features["new"] = 2.0', 'return "rl"'),
     RAN, TrainingMode.RL),
    ("a single leading underscore", body("_x = 1", 'return "rl"'), RAN, TrainingMode.RL),
    ("trailing underscores", body("x__ = 1", 'return "rl"'), RAN, TrainingMode.RL),
    ("'__builtins__' as a string, not a name",
     body('return "rl" if len("__builtins__") else "skip"'), RAN, TrainingMode.RL),
    ("implicit string concatenation", body('return "r" "l"'), RAN, TrainingMode.RL),
    ("a mode assembled at runtime", body('return "rx"[0] + "l"'), RAN, TrainingMode.RL),
    ("float('inf')", body('return "rl" if float("inf") > 0 else "skip"'), RAN, TrainingMode.RL),
    ("float overflow is just inf", body('return "rl" if 1e308 * 10 else "skip"'),
     RAN, TrainingMode.RL),
    ("NaN compares false both ways",
     body('return "rl" if float("nan") > 0 else ("skip" if float("nan") <= 0 else "sft")'),
     RAN, TrainingMode.SFT),
    ("a small exponent still works", body('return "rl" if 2 ** 3 ** 2 == 512 else "skip"'),
     RAN, TrainingMode.RL),

    # -- imports, by every route ----------------------------------------------------------
    ("import statement", body("import os", 'return "rl"'), REJECTED, None),
    ("from-import", body("from os import system", 'return "rl"'), REJECTED, None),
    ("__import__ call", body('return __import__("os")'), REJECTED, None),
    ("__import__ as a name", body("x = __import__", 'return "rl"'), REJECTED, None),
    ("__import__ out of __builtins__", body('return __builtins__["__import__"]("os")'),
     REJECTED, None),

    # -- builtins that were not allowlisted ------------------------------------------------
    ("__builtins__ by name", body("return __builtins__"), REJECTED, None),
    ("globals()", body("return globals()"), REJECTED, None),
    ("locals()", body("return locals()"), REJECTED, None),
    ("vars()", body("return vars()"), REJECTED, None),
    ("dir()", body("return dir(features)"), REJECTED, None),
    ("eval", body('return eval("1")'), REJECTED, None),
    ("exec", body('exec("x=1")', 'return "rl"'), REJECTED, None),
    ("compile", body('return compile("1", "x", "eval")'), REJECTED, None),
    ("open", body('return open("/etc/passwd")'), REJECTED, None),
    ("breakpoint", body("breakpoint()", 'return "rl"'), REJECTED, None),
    ("input", body("return input()"), REJECTED, None),
    ("type", body("return type(features)"), REJECTED, None),
    ("getattr", body('return getattr(features, "keys")'), REJECTED, None),
    ("setattr", body('setattr(features, "x", 1)', 'return "rl"'), REJECTED, None),
    ("print", body('print("hi")', 'return "rl"'), REJECTED, None),
    ("str", body('return str("rl")'), REJECTED, None),
    ("int", body("return int(1)"), REJECTED, None),
    ("tuple, via a generator", body('return "rl" if len(tuple(x for x in (1,))) else "s"'),
     REJECTED, None),

    # -- attributes: the whole __class__ -> __subclasses__ ladder --------------------------
    ("__class__", body("return features.__class__"), REJECTED, None),
    ("__subclasses__ ladder", body("return features.__class__.__base__.__subclasses__()"),
     REJECTED, None),
    ("route.__globals__", body("return route.__globals__"), REJECTED, None),
    ("a plain method call", body("return features.keys()"), REJECTED, None),
    ("an attribute without calling it", body("x = features.items", 'return "rl"'),
     REJECTED, None),
    ("calling the result of a subscript", body('return features["k"]("x")'), REJECTED, None),

    # -- syntax that would smuggle in an expression ----------------------------------------
    ("f-string with no placeholder", body('return f"rl"'), REJECTED, None),
    ("f-string with a placeholder", body('return f"{features}"'), REJECTED, None),
    ("starred call argument", body("return min(*[1, 2])"), REJECTED, None),
    ("starred assignment target", body("a, *b = [1, 2, 3]", 'return "rl"'), REJECTED, None),
    ("walrus", body('if (y := features["solve_rate"]) > 0:', '    return "rl"',
                    'return "skip"'), REJECTED, None),
    ("lambda", body("f = lambda x: x", 'return "rl"'), REJECTED, None),
    ("list comprehension", body('return "rl" if [x for x in (1,)] else "skip"'),
     REJECTED, None),
    ("set comprehension", body('return "rl" if {x for x in (1,)} else "skip"'), REJECTED, None),
    ("dict comprehension", body('return "rl" if {x: x for x in (1,)} else "skip"'),
     REJECTED, None),
    ("a decorator naming a builtin", "@len\n" + body('return "rl"'), REJECTED, None),
    ("a decorator that is a call", "@min(1, 2)\n" + body('return "rl"'), REJECTED, None),
    ("a decorator naming nothing", "@nope\n" + body('return "rl"'), REJECTED, None),
    ("a nested def", body("def inner(x):", "    return x", 'return "rl"'), REJECTED, None),
    ("a nested def with a default", body("def inner(x=9 ** 9 ** 9):", "    return x",
                                         'return "rl"'), REJECTED, None),
    ("global", body("global g", "g = 1", 'return "rl"'), REJECTED, None),
    ("del a local", body("x = 1", "del x", 'return "rl"'), REJECTED, None),
    ("del a features key", body('del features["solve_rate"]', 'return "rl"'), REJECTED, None),
    ("assert", body('assert features["solve_rate"] >= 0', 'return "rl"'), REJECTED, None),
    ("raise", body('raise ValueError("x")'), REJECTED, None),
    ("raise a BaseException", body("raise KeyboardInterrupt"), REJECTED, None),
    ("try/except", body("try:", "    x = 1", "except Exception:", "    x = 2",
                        'return "rl"'), REJECTED, None),
    ("with", body('with open("/etc/passwd") as f:', "    x = 1", 'return "rl"'),
     REJECTED, None),
    ("yield", body('yield "rl"'), REJECTED, None),
    ("await", body("await features", 'return "rl"'), REJECTED, None),
    ("async def", "async def route(features):\n    return 'rl'\n", REJECTED, None),
    ("match", body('match features["solve_rate"]:', "    case 0:", '        return "skip"',
                   'return "rl"'), REJECTED, None),
    ("a nested class", body("class K:", "    pass", 'return "rl"'), REJECTED, None),
    ("a class beside the function", "class K:\n    pass\n" + body('return "rl"'),
     REJECTED, None),
    ("dict literal", body('return "rl" if {"a": 1} else "skip"'), REJECTED, None),
    ("set literal", body('return "rl" if {1, 2} else "skip"'), REJECTED, None),
    ("while", body("while True:", "    pass", 'return "rl"'), REJECTED, None),
    ("for", body("for i in (1, 2):", "    pass", 'return "rl"'), REJECTED, None),
    ("augmented assignment", body("x = 1", "x += 1", 'return "rl"'), REJECTED, None),
    ("annotated assignment", body("x: int = 1", 'return "rl"'), REJECTED, None),
    ("a slice", body('return "rl" if (1, 2, 3)[0:2] else "skip"'), REJECTED, None),
    ("the in operator", body('return "rl" if "solve_rate" in features else "skip"'),
     REJECTED, None),
    ("the is operator", body('return "rl" if features is features else "skip"'),
     REJECTED, None),
    ("bitwise or", body('return "rl" if 1 | 2 else "skip"'), REJECTED, None),
    ("floor division", body('return "rl" if 5 // 2 else "skip"'), REJECTED, None),
    ("bitwise invert", body('return "rl" if ~1 else "skip"'), REJECTED, None),
    ("a keyword argument", body('return "rl" if round(1.2, ndigits=1) else "skip"'),
     REJECTED, None),
    ("**kwargs at a call site", body("return min(**features)"), REJECTED, None),
    ("pass", body("pass", 'return "rl"'), REJECTED, None),

    # -- the wrong shape --------------------------------------------------------------------
    ("a second top-level statement", body('return "rl"') + "x = 1\n", REJECTED, None),
    ("a module docstring", '"""doc"""\n' + body('return "rl"'), REJECTED, None),
    ("the wrong function name", "def decide(features):\n    return 'rl'\n", REJECTED, None),
    ("two arguments", "def route(features, other):\n    return 'rl'\n", REJECTED, None),
    ("*args", "def route(*features):\n    return 'rl'\n", REJECTED, None),
    ("**kwargs", "def route(**features):\n    return 'rl'\n", REJECTED, None),
    ("a keyword-only argument", "def route(features, *, x):\n    return 'rl'\n",
     REJECTED, None),
    ("a positional-only marker", "def route(features, /):\n    return 'rl'\n", REJECTED, None),
    ("a default argument", "def route(features=1):\n    return 'rl'\n", REJECTED, None),
    ("a syntax error", "def route(features:\n", REJECTED, None),
    ("a null byte", body('return "rl"') + "\x00", REJECTED, None),
    ("nesting deeper than the parser's stack",
     "def route(features):\n    return " + "1+" * 20000 + "1\n", REJECTED, None),
    ("nesting deeper than the parser's parens",
     "def route(features):\n    return " + "(" * 20000 + "1" + ")" * 20000 + "\n",
     REJECTED, None),
    ("a name that NFKC-normalises to a dunder",
     "def route(features):\n    return _" + chr(0xFF3F) + "builtins" + chr(0xFF3F) + "_\n",
     REJECTED, None),

    # -- valid source that fails at runtime, and must be counted ----------------------------
    ("a missing feature key", body('return features["nope"]'), ERROR, TrainingMode.SKIP),
    ("division by zero", body('return "rl" if 1 / 0 else "skip"'), ERROR, TrainingMode.SKIP),
    ("round of an overflowing float", body('return "rl" if round(1e308 ** 2) else "s"'),
     ERROR, TrainingMode.SKIP),
    ("min of nothing", body('return "rl" if min() else "skip"'), ERROR, TrainingMode.SKIP),
    ("abs of a string", body('return "rl" if abs("x") else "skip"'), ERROR, TrainingMode.SKIP),
    ("assigning into a string", body("x = 'ab'", "x[0] = 'c'", 'return "rl"'),
     ERROR, TrainingMode.SKIP),

    # -- valid source that returns something unusable, and must be counted ------------------
    ("returns None", body("return None"), INVALID, TrainingMode.SKIP),
    ("returns nothing at all", body("x = 1"), INVALID, TrainingMode.SKIP),
    ("returns an int", body("return 1"), INVALID, TrainingMode.SKIP),
    ("returns a bool", body("return True"), INVALID, TrainingMode.SKIP),
    ("returns Ellipsis", body("return ..."), INVALID, TrainingMode.SKIP),
    ("returns bytes", body('return b"rl"'), INVALID, TrainingMode.SKIP),
    ("returns a tuple of the mode", body('return ("rl",)'), INVALID, TrainingMode.SKIP),
    ("returns a list of the mode", body('return ["rl"]'), INVALID, TrainingMode.SKIP),
    ("returns the empty string", body('return ""'), INVALID, TrainingMode.SKIP),
    ("returns an unregistered mode", body('return "telepathy"'), INVALID, TrainingMode.SKIP),
    ("returns a %-formatted dict", body('return "%s" % features'), INVALID, TrainingMode.SKIP),
)


def test_the_adversarial_corpus_lands_exactly_where_it_is_pinned():
    """One row per attempt. Loosening the allowlist moves a row and fails here."""
    for name, source, expected, expected_mode in _CORPUS:
        got, detail, _h = outcome(source)
        assert got == expected, f"{name}: {got} (not {expected}) -- {detail}"
        if expected_mode is not None:
            assert detail == expected_mode, f"{name}: emitted {detail}, not {expected_mode}"


def test_the_corpus_covers_every_column_so_it_cannot_pass_by_rejecting_everything():
    """A table where every row is REJECTED proves the validator says no, nothing more."""
    seen = {}
    for _n, _s, expected, _m in _CORPUS:
        seen[expected] = seen.get(expected, 0) + 1
    for column in (REJECTED, ERROR, INVALID, RAN):
        assert seen.get(column, 0) >= 5, f"{column}: only {seen.get(column, 0)} rows"


def test_a_rejected_policy_names_the_construct_it_was_rejected_for():
    """A generated policy is debugged from this message; 'no' on its own is not usable."""
    for source, needle in (
        (body("import os", 'return "rl"'), "Import"),
        (body("return features.keys()"), "Attribute"),
        (body("return eval('1')"), "eval"),
        (body("return __builtins__"), "__builtins__"),
        (body('return f"{features}"'), "JoinedStr"),
        (body("for i in (1,):", "    pass", 'return "rl"'), "For"),
        ("@len\n" + body('return "rl"'), "decorated"),
        (body("def inner():", "    return 1", 'return "rl"'), "inner"),
        ("def decide(features):\n    return 'rl'\n", "route"),
        ("async def route(features):\n    return 'rl'\n", "'def' statements"),
    ):
        with pytest.raises(PolicyRejected, match=needle):
            validate_policy_source(source)


# ------------------------------------------------- what validation accepts but cannot bound

# ACCEPTED, and never executed here. Each is ordinary allowlisted arithmetic whose cost is
# unbounded, and none is interruptible from Python: the exponentiation and the sequence
# repetition are single opcodes, so a signal handler never gets a turn. Removing them means
# removing ``**`` and ``*``, which the RAN rows above use. Pinned so that the gap is a fact
# in the test suite rather than a surprise in a training run.
_ACCEPTED_BUT_UNBOUNDED = (
    ("integer exponentiation", body('return "rl" if 9 ** 9 ** 9 else "skip"')),
    ("a smaller but still fatal exponent", body('return "rl" if 2 ** 2 ** 30 else "skip"')),
    ("list repetition", body('return "rl" if len([0] * 10 ** 9) else "skip"')),
    ("tuple repetition", body('return "rl" if len((0,) * 10 ** 9) else "skip"')),
    ("string repetition", body('return "rl" if len("x" * 10 ** 9) else "skip"')),
    ("printf-style padding", body('return "rl" if len("%999999999d" % 1) else "skip"')),
    ("an unbounded annotation", "def route(features: 9 ** 9 ** 9):\n    return 'rl'\n"),
)


@pytest.mark.parametrize("name,source", _ACCEPTED_BUT_UNBOUNDED, ids=[n for n, _ in
                                                                     _ACCEPTED_BUT_UNBOUNDED])
def test_validation_accepts_arithmetic_whose_cost_it_cannot_bound(name, source):
    """NOT an endorsement. Validation only; running any of these is the finding itself."""
    assert isinstance(validate_policy_source(source).body[0].name, str)


def test_a_bounded_version_of_each_unbounded_shape_really_does_run():
    """The row above would also 'pass' if ** and * had been quietly dropped from the list."""
    for source, mode in (
        (body('return "rl" if 2 ** 3 ** 2 == 512 else "skip"'), TrainingMode.RL),
        (body('return "rl" if len([0] * 3) == 3 else "skip"'), TrainingMode.RL),
        (body('return "rl" if len("x" * 3) == 3 else "skip"'), TrainingMode.RL),
        (body('return "rl" if len("%9d" % 1) == 9 else "skip"'), TrainingMode.RL),
    ):
        got, emitted, _h = outcome(source)
        assert (got, emitted) == (RAN, mode), source


def test_an_annotation_is_stringified_and_never_evaluated():
    """``1 / 0`` in an annotation raises at DEF time if annotations are live, which is
    inside the constructor and outside route()'s try. Fails fast rather than hanging, which
    is what the same check written with ``9 ** 9 ** 9`` would do."""
    fn = compile_policy("def route(features: 1 / 0) -> 1 / 0:\n    return 'rl'\n")
    assert fn.__annotations__ == {"features": "1 / 0", "return": "1 / 0"}
    assert fn({}) == TrainingMode.RL


# ------------------------------------------------------------------------- the counters

def test_each_rejection_path_increments_its_own_counter_and_no_other():
    """Three paths, three counters. A mutant that increments the wrong one, or reuses one
    counter for two paths, is invisible to any test that only checks the emitted mode."""
    cases = (
        (body("return features['nope']"), ctx(), "errors"),
        (body('return "telepathy"'), ctx(), "invalid_returns"),
        (body('return "sft"'), ctx(0.0, teacher=False), "teacher_blocked"),
    )
    for source, c, counter in cases:
        r = CodePolicyRouter(source=source)
        for i in range(3):
            r.route(c)
            h = r.health()
            assert h["calls"] == i + 1
            assert h[counter] == i + 1, f"{counter} did not move: {h}"
            assert sum(v for k, v in h.items() if k not in ("calls", counter)) == 0, h


def test_health_accounts_for_every_call_it_was_given():
    """calls == the good ones plus the three rejection paths, over a mixed stream. A
    counter that stops incrementing shows up as a gap, not as a wrong mode."""
    source = body(
        'if features["solve_rate"] > 0.9:', '    return "telepathy"',
        'if features["solve_rate"] > 0.6:', '    return features["nope"]',
        'if features["solve_rate"] > 0.2:', '    return "rl"',
        'return "sft"',
    )
    r = CodePolicyRouter(source=source)
    good = 0
    for p, teacher in itertools.product(RATES, (False, True)):
        d = r.route(ctx(p, teacher=teacher))
        if d.reason == "code policy":
            good += 1
    h = r.health()
    assert h["calls"] == len(RATES) * 2
    assert good + h["errors"] + h["invalid_returns"] + h["teacher_blocked"] == h["calls"]
    assert min(h["errors"], h["invalid_returns"], h["teacher_blocked"]) > 0, \
        f"the stream never exercised one of the paths, so it proved nothing: {h}"


def test_every_rejection_says_in_the_reason_what_happened():
    """A fallback that reads like a decision is the failure this router exists to prevent."""
    for source, c, needle in (
        (body("return features['nope']"), ctx(), "raised"),
        (body('return "telepathy"'), ctx(), "not an allowed mode"),
        (body('return "sft"'), ctx(0.0, teacher=False), "no target"),
        (body('return "rl"'), ctx(), "code policy"),
    ):
        assert needle in CodePolicyRouter(source=source).route(c).reason


def test_the_reason_is_bounded_however_large_the_returned_value_is():
    """``reason`` is carried into logs. A policy can return ten megabytes of string, and
    ``f"...{got!r}"`` would put all of it there -- and build it a second time to do so."""
    r = CodePolicyRouter(source=body('return "x" * 10 ** 7'))
    d = r.route(ctx())
    assert r.health()["invalid_returns"] == 1
    assert len(d.reason) < 500, f"reason was {len(d.reason)} characters"
    assert "not an allowed mode" in d.reason


# --------------------------------------------------------------------------- the guards

def test_a_teacher_requiring_mode_is_never_emitted_without_a_target():
    """SAFETY, swept: every registered mode, every rate, every granularity, both teacher
    settings. The policy does not get to opt out of the guard every other router obeys."""
    checked = 0
    for mode, p, gran, teacher in itertools.product(known_modes(), RATES, GRANS, (False, True)):
        c = ctx(p, teacher=teacher, gran=gran)
        d = CodePolicyRouter(source=body(f'return "{mode}"')).route(c)
        emitted = mode_of(d)
        assert not (known_modes()[emitted] and not c.has_target), \
            f"{mode} at p={p} gran={gran.value} teacher={teacher} -> {emitted}"
        checked += 1
    assert checked == len(known_modes()) * len(RATES) * len(GRANS) * 2


def test_the_guard_reads_has_target_not_has_teacher():
    """A group with a correct sample is its own SFT target; requiring an external teacher
    would throw away exactly the units GRPO cannot learn from."""
    r = CodePolicyRouter(source=body('return "sft"'))
    assert mode_of(r.route(ctx(0.5, teacher=False))) == TrainingMode.SFT
    assert r.health()["teacher_blocked"] == 0
    # ...but a self-target is not defined per token, so TOKEN granularity is blocked
    assert mode_of(r.route(ctx(0.5, teacher=False, gran=Granularity.TOKEN))) == TrainingMode.SKIP
    assert r.health()["teacher_blocked"] == 1


def test_a_teacher_requiring_fallback_is_held_to_the_guard_too():
    """THE FALLBACK IS THE ROUTER'S OWN ANSWER. Enforcing the guard on the policy's choice
    and skipping it on the substitute emits, on the rejection path, the exact decision the
    guard exists to prevent -- and does it while logging that the policy was wrong."""
    paths = (
        (body('return "telepathy"'), "invalid_returns"),
        (body("return features['nope']"), "errors"),
        (body('return "sft"'), "teacher_blocked"),
    )
    for (source, counter), fb in itertools.product(paths,
                                                   (TrainingMode.SFT, TrainingMode.DISTILL)):
        r = CodePolicyRouter(source=source, fallback=fb)
        d = r.route(ctx(0.0, teacher=False))
        assert mode_of(d) == TrainingMode.SKIP, f"{counter} with fallback={fb}: {d.weights}"
        assert r.health()[counter] == 1, r.health()
        assert "needs a target" in d.reason


def test_the_guarded_fallback_is_still_the_fallback_wherever_it_can_be_honoured():
    """The guard degrades a fallback it cannot honour; it must not replace one it can, or
    a ``fallback='sft'`` arm would quietly become a ``fallback='skip'`` arm."""
    for source in (body('return "telepathy"'), body("return features['nope']")):
        for fb in (TrainingMode.SFT, TrainingMode.DISTILL):
            r = CodePolicyRouter(source=source, fallback=fb)
            for c in (ctx(0.5, teacher=False), ctx(0.0, teacher=True)):
                assert mode_of(r.route(c)) == fb, f"fallback={fb} dropped for {c}"


def test_a_teacherless_fallback_is_unaffected_by_that_guard():
    """ROLLBACK: the default fallback is SKIP, and nothing about it changed."""
    for fb in (TrainingMode.SKIP, TrainingMode.RL):
        for p, teacher, gran in itertools.product(RATES, (False, True), GRANS):
            d = CodePolicyRouter(source=body('return "telepathy"'), fallback=fb).route(
                ctx(p, teacher=teacher, gran=gran))
            assert mode_of(d) == fb
            assert "needs a target" not in d.reason


# ------------------------------------------------------------------- construction contract

def test_a_bad_policy_is_rejected_when_the_router_is_built_not_mid_run():
    """A run must not get halfway through a batch before finding out the arm cannot work."""
    for source in (body("import os", 'return "rl"'), "def route(:\n", "@len\n" + body("return 1"),
                   "def route(features):\n    return " + "1+" * 20000 + "1\n"):
        with pytest.raises(PolicyRejected):
            CodePolicyRouter(source=source)


def test_construction_refuses_a_mode_it_could_never_honour():
    with pytest.raises(ValueError, match="unknown fallback mode"):
        CodePolicyRouter(source=body('return "rl"'), fallback="telepathy")
    with pytest.raises(ValueError, match="unknown mode"):
        CodePolicyRouter(source=body('return "rl"'), allowed_modes=("rl", "telepathy"))


def test_an_empty_allowed_modes_is_refused_rather_than_read_as_all_of_them():
    """A caller whose filter came back empty asked for nothing; ``or`` would hand it
    everything, which is the widest silent failure this file can pin."""
    with pytest.raises(ValueError, match="at least one mode"):
        CodePolicyRouter(source=body('return "rl"'), allowed_modes=())
    assert CodePolicyRouter(source=body('return "rl"')).allowed_modes == tuple(known_modes())


def test_allowed_modes_is_honoured_for_every_mode_it_leaves_out():
    """An excluded mode is an invalid return, not a quietly accepted one."""
    for mode in known_modes():
        allowed = tuple(m for m in known_modes() if m != mode)
        r = CodePolicyRouter(source=body(f'return "{mode}"'), allowed_modes=allowed)
        d = r.route(ctx())
        assert mode_of(d) == TrainingMode.SKIP
        assert r.health()["invalid_returns"] == 1
        assert r.health()["teacher_blocked"] == 0, "an excluded mode is not a guard failure"
        r2 = CodePolicyRouter(source=body(f'return "{mode}"'), allowed_modes=(mode,))
        assert mode_of(r2.route(ctx())) == mode


def test_the_router_is_a_router_and_its_signature_constant_is_the_shape_it_accepts():
    r = CodePolicyRouter(source=body('return "rl"'))
    assert isinstance(r, Router)
    assert compile_policy(POLICY_SIGNATURE + '\n    return "rl"\n')({}) == TrainingMode.RL


# ------------------------------------------------------------------------- determinism

def test_the_same_unit_gives_the_same_decision_every_time_and_across_routers():
    """A routing ablation that is not reproducible is not an ablation."""
    source = body('if features["solve_rate"] > 0.6:', '    return "sft"',
                  'if features["solve_rate"] > 0.2:', '    return "rl"',
                  'return "telepathy"')
    a, b = CodePolicyRouter(source=source), CodePolicyRouter(source=source)
    units = [ctx(p, teacher=t, gran=g) for p, t, g in
             itertools.product(RATES, (False, True), GRANS)]
    seq_a = [(mode_of(a.route(c)), a.route(c).reason) for c in units]
    seq_b = [(mode_of(b.route(c)), b.route(c).reason) for c in units]
    assert seq_a == seq_b
    assert len({m for m, _ in seq_a}) > 1, "the sweep produced one mode, so it proved nothing"


# --------------------------------------------------------------------------- containment

def test_a_policy_cannot_reach_the_dict_the_caller_still_holds():
    """``ctx.extra`` is the caller's; the policy is handed a copy, per call."""
    extra = {"a": 1.0}
    c = ctx(0.5, extra=extra)
    r = CodePolicyRouter(source=body('features["a"] = 99.0', 'features["new"] = 1.0',
                                     'return "rl"'))
    r.route(c)
    r.route(c)
    assert dict(extra) == {"a": 1.0}
    assert dict(c.extra) == {"a": 1.0}


def test_a_policy_cannot_carry_state_from_one_unit_to_the_next():
    """A fresh dict per call. A policy that accumulated across units would make routing
    depend on batch order, which no other router does."""
    c = ctx(0.5, extra={"a": 0.0})
    r = CodePolicyRouter(source=body('features["a"] = features["a"] + 1',
                                     'return "sft" if features["a"] > 1.5 else "rl"'))
    assert [mode_of(r.route(c)) for _ in range(3)] == [TrainingMode.RL] * 3


def test_the_features_dict_is_exactly_what_was_documented():
    """The contract a generated policy is written against."""
    r = CodePolicyRouter(source=body('return "rl"'))
    f = r._features(ctx(0.25, teacher=True, g=8, extra={"entropy": 2}))
    assert f == {"entropy": 2.0, "solve_rate": 0.25, "group_size": 8.0, "has_target": 1.0}
    assert all(isinstance(v, float) for v in f.values())


def test_the_policy_globals_hold_the_numeric_handful_and_nothing_else():
    fn = compile_policy(body('return "rl"'))
    assert set(fn.__globals__["__builtins__"]) == {"min", "max", "abs", "float", "len", "round"}
    assert set(fn.__globals__) - {"__builtins__", "route"} == set()


# ------------------------------------------------------------------- the allowlist itself

def test_two_policies_never_share_one_builtins_table():
    """Each compile gets its own copy of the table, and the module's own copy is never the
    one a policy runs against. If they were shared, a single reachable write -- the kind a
    widened allowlist or a new builtin would hand a policy -- would reconfigure every other
    policy in the process, and every one compiled after it."""
    counting = body('return "rl" if len((1,)) else "skip"')
    a, b = compile_policy(counting), compile_policy(counting)
    assert a.__globals__["__builtins__"] is not b.__globals__["__builtins__"]
    try:
        a.__globals__["__builtins__"]["len"] = lambda _x: 0   # poison one policy's table
        assert a({}) == TrainingMode.SKIP, "the poison did not take, so nothing was proved"
        assert b({}) == TrainingMode.RL, "one policy reconfigured another"
        assert code_policy._SAFE_BUILTINS["len"] is len, "a policy reached the module table"
        assert compile_policy(counting)({}) == TrainingMode.RL, "and every later policy"
    finally:
        code_policy._SAFE_BUILTINS["len"] = len


def test_the_allowlist_is_exactly_this_set():
    """Pinned by name. Widening it is a deliberate act with a test to update, not a diff
    line that slips through review -- ``Attribute``, ``Import`` and ``Lambda`` are each one
    tuple entry away from being legal."""
    assert set(code_policy._ALLOWED_CALLS) == {"min", "max", "abs", "float", "len", "round"}
    assert {n.__name__ for n in code_policy._ALLOWED_NODES} == {
        "Module", "FunctionDef", "arguments", "arg", "Return", "Assign",
        "Name", "Store", "Load", "Constant", "Expr",
        "If", "IfExp", "Compare", "BoolOp", "UnaryOp", "BinOp",
        "And", "Or", "Not", "USub", "UAdd",
        "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE",
        "Add", "Sub", "Mult", "Div", "Mod", "Pow",
        "Subscript", "Call", "Tuple", "List",
    }


def test_the_call_allowlist_and_the_builtin_table_cannot_drift_apart():
    """A name callable in a policy but absent from the globals is a NameError at runtime;
    the reverse is a builtin reachable without ever being allowlisted."""
    assert set(code_policy._SAFE_BUILTINS) == set(code_policy._ALLOWED_CALLS)
    assert all(callable(v) for v in code_policy._SAFE_BUILTINS.values())
    for name in code_policy._ALLOWED_CALLS:
        got, emitted, _h = outcome(body(f'return "rl" if {name}((1.0, 2.0)) else "skip"')
                                   if name in ("min", "max", "len") else
                                   body(f'return "rl" if {name}(1.0) else "skip"'))
        assert got == RAN and emitted == TrainingMode.RL, f"{name}: {got}"


def test_safe_builtins_is_built_under_both_import_contexts():
    """``__builtins__`` is a dict in an imported module and the builtins MODULE in
    ``__main__``. The table comprehension branches on that, and only one branch runs
    wherever the suite happens to run -- so the other is exercised in a subprocess.
    Measured, not reasoned about: this is exactly the kind of portability trap that only
    shows up the first time somebody runs the file directly."""
    assert isinstance(code_policy.__dict__["__builtins__"], dict), \
        "an imported module should take the dict branch"

    src = Path(code_policy.__file__)
    root = src.parents[3]
    script = src.parent / "_code_policy_main_probe.py"  # never imported; removed below
    tail = (
        "\n\nif __name__ == '__main__':\n"
        "    assert not isinstance(__builtins__, dict), 'expected the module branch here'\n"
        "    assert sorted(_SAFE_BUILTINS) == sorted(_ALLOWED_CALLS)\n"
        "    print(compile_policy(\"def route(f):\\n    return 'rl'\\n\")({}))\n"
    )
    script.write_text(src.read_text() + tail)
    try:
        p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                           cwd=str(root), timeout=120)
    finally:
        script.unlink()
    assert p.returncode == 0, p.stderr[-2000:]
    assert p.stdout.strip() == TrainingMode.RL, p.stdout
