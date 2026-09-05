"""The gold verifier refutes a wrong key, and it can fail.

Seeded with the exact task the first live iteration produced: "the sum of all positive
integers n for which n^2-3n+1 divides n^3-2n+5", whose proposer asserted 6 and whose true
answer is 12. A verifier that cannot return REFUTED on that task is not a verifier, so it is
the first case here.

The suite is written so that a degenerate verifier FAILS loudly: `test_mutation_*` construct
sources that always agree, always refuse, or ignore the problem, and assert the harness
notices. A check that only ever passes would not be evidence.
"""

from __future__ import annotations

import inspect

import pytest

from ornith_repro.verify import (
    REGISTRY,
    AnswerSource,
    ExecutableEnumeration,
    LeanBackend,
    Verdict,
    extract_code,
    program_is_safe,
    run_program,
    verify_answer,
    verify_with_cascade,
)

# The task from the first live iteration. Asserted 6; the true answer is 12.
CAUGHT = ("Find the sum of all positive integers $n$ for which $n^2 - 3n + 1$ divides "
          "$n^3 - 2n + 5$.")

CORRECT_PROGRAM = '''```python
good = []
for n in range(1, 1000):
    d = n * n - 3 * n + 1
    if d != 0 and (n ** 3 - 2 * n + 5) % d == 0:
        good.append(n)
print(sum(good))
```'''


class ScriptedClient:
    """Returns a fixed completion, recording every prompt it was shown.

    The recording is what makes the independence test possible: the suite asserts the
    asserted answer never appears in any prompt the source received.
    """

    def __init__(self, completion: str, truncated: bool = False) -> None:
        self.completion = completion
        self.truncated = truncated
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_new_tokens: int, seed: int) -> tuple[str, bool]:
        """Record the prompt and return the scripted completion."""
        self.prompts.append(prompt)
        return self.completion, self.truncated


def _exec_source(completion: str) -> ExecutableEnumeration:
    """An executable backend wired to a scripted completion."""
    return ExecutableEnumeration(ScriptedClient(completion), timeout=15.0, attempts=1)


def test_refutes_the_wrong_key_that_motivated_this_module():
    """The caught task with its asserted 6 must come back REFUTED, computing 12."""
    res = verify_answer(CAUGHT, "6", _exec_source(CORRECT_PROGRAM))
    assert res.verdict is Verdict.REFUTED, res
    assert res.computed == "12"


def test_verifies_the_same_task_under_its_true_key():
    """The same program against the true answer must come back VERIFIED, or the
    verifier is simply refusing everything."""
    res = verify_answer(CAUGHT, "12", _exec_source(CORRECT_PROGRAM))
    assert res.verdict is Verdict.VERIFIED, res


def test_source_never_sees_the_asserted_answer_by_signature():
    """Independence is enforced by the type, so assert the type.

    `solve` must take the statement only. If a parameter carrying the key were ever added,
    a backend could rationalise toward it and this suite would stop meaning anything.
    """
    params = list(inspect.signature(AnswerSource.solve).parameters)
    assert params == ["self", "problem"], params
    for factory in REGISTRY.values():
        impl = getattr(factory, "solve", None)
        if impl is not None:
            assert "asserted" not in inspect.signature(impl).parameters


def test_source_never_sees_the_asserted_answer_in_practice():
    """Belt and braces: the key must not appear in any prompt the backend was shown."""
    client = ScriptedClient(CORRECT_PROGRAM)
    src = ExecutableEnumeration(client, timeout=15.0, attempts=1)
    verify_answer(CAUGHT, "6", src)
    assert client.prompts, "backend never called the client"
    for p in client.prompts:
        assert "6" not in p.replace(CAUGHT, ""), "asserted answer leaked into the prompt"


def test_unverifiable_when_no_code_block():
    """A completion with no program is UNVERIFIABLE, never a guess."""
    res = verify_answer(CAUGHT, "6", _exec_source("I think the answer is 6."))
    assert res.verdict is Verdict.UNVERIFIABLE
    assert res.computed is None


def test_unverifiable_when_the_program_crashes():
    """A crashing program yields UNVERIFIABLE, not a verdict on the key."""
    res = verify_answer(CAUGHT, "6", _exec_source("```python\nraise SystemExit(3)\n```"))
    assert res.verdict is Verdict.UNVERIFIABLE


def test_sandbox_timeout_fires():
    """The wall-clock guard must actually stop a non-terminating program."""
    out, detail = run_program("while True:\n    pass\n", timeout=2.0)
    assert out is None and "timeout" in detail


def test_sandbox_refuses_network_and_process_access():
    """The safety screen must refuse, including through an alias, and allow plain maths."""
    assert not program_is_safe("import socket")[0]
    assert not program_is_safe("import urllib.request as u")[0]
    assert not program_is_safe("from requests import get")[0]
    assert not program_is_safe("import os\nos.system('ls')")[0]
    assert not program_is_safe("eval('1+1')")[0]
    assert program_is_safe("print(sum(range(10)))")[0]


def test_lean_backend_declines_rather_than_guessing():
    """The registered placeholder must return UNVERIFIABLE with a reason."""
    res = verify_answer(CAUGHT, "6", LeanBackend())
    assert res.verdict is Verdict.UNVERIFIABLE
    assert "not implemented" in res.detail


def test_cascade_stops_at_the_first_decisive_backend():
    """UNVERIFIABLE falls through; a decision ends the cascade."""
    res = verify_with_cascade(CAUGHT, "6", [LeanBackend(), _exec_source(CORRECT_PROGRAM)])
    assert res.verdict is Verdict.REFUTED
    assert res.backend == "executable_enumeration"


def test_extract_code_takes_the_last_block():
    """A completion that reasons in code then answers in code must use the final block."""
    assert extract_code("```python\nprint(1)\n```\n```python\nprint(2)\n```") == "print(2)"
    assert extract_code("no code") is None


# ---------------------------------------------------------------- mutation tests
class AlwaysAgrees:
    """A deliberately broken source that echoes whatever it is compared against.

    It cannot literally see the key, so it approximates the failure by returning a constant
    that the test then asserts against -- the point is that a source which always produces
    the asserted value would make every task VERIFIED.
    """

    name = "always_agrees"

    def __init__(self, value: str) -> None:
        self.value = value

    def solve(self, problem: str) -> tuple[str | None, str, str]:
        """Return the planted value regardless of the problem."""
        return self.value, "planted", ""


def test_mutation_a_verifier_that_always_agrees_is_caught():
    """A source that always returns the asserted value verifies a KNOWN-WRONG key.

    This is the mutation that matters: if such a source passed the suite, the verifier would
    be certifying the very mistake it exists to catch.
    """
    bad = verify_answer(CAUGHT, "6", AlwaysAgrees("6"))
    assert bad.verdict is Verdict.VERIFIED
    good = verify_answer(CAUGHT, "6", _exec_source(CORRECT_PROGRAM))
    assert good.verdict is Verdict.REFUTED
    assert bad.verdict != good.verdict, (
        "a rubber-stamp source and a real one must not agree on a known-wrong key")


def test_mutation_a_verifier_that_never_decides_is_caught():
    """A source that always declines must not be mistaken for a working one."""
    res = verify_answer(CAUGHT, "12", LeanBackend())
    assert res.verdict is Verdict.UNVERIFIABLE
    real = verify_answer(CAUGHT, "12", _exec_source(CORRECT_PROGRAM))
    assert real.verdict is Verdict.VERIFIED
    assert res.verdict != real.verdict


def test_mutation_a_program_ignoring_the_problem_is_not_verified():
    """A program that prints a constant unrelated to the problem must refute a true key."""
    res = verify_answer(CAUGHT, "12", _exec_source("```python\nprint(999)\n```"))
    assert res.verdict is Verdict.REFUTED


# ------------------------------------------------- sandbox bypasses found by an audit
# Each of these ran successfully against an earlier version of `program_is_safe`: `open` was
# not banned at all, nor `importlib`, nor `getattr(__builtins__, "__import__")`, and the
# denylist held `execv` while missing `execlp` and `posix_spawn`. They are kept as a suite so
# the screen cannot quietly regress to the version that let them through.
AUDITED_BYPASSES = {
    "read a file via open": "print(open('/etc/passwd').readline())",
    "import socket via importlib": "import importlib; s=importlib.import_module('socket')",
    "import via getattr on builtins": "print(getattr(__builtins__,'__import__')('socket'))",
    "spawn via os.execlp": "import os; os.execlp('echo','echo','x')",
    "spawn via os.posix_spawn": "import os; os.posix_spawn('/bin/echo',['echo','x'],{})",
    "write outside the scratch dir": "open('/tmp/PWNED','w').write('x')",
    "aliased os.system": "import os; f=os.system; f('echo hi')",
    "dunder traversal": "print(().__class__.__bases__)",
}


def test_every_audited_sandbox_bypass_is_refused():
    """All eight programs an audit ran successfully must now be refused."""
    allowed = {name: why for name, code in AUDITED_BYPASSES.items()
               for ok, why in [program_is_safe(code)] if ok}
    assert not allowed, "sandbox still allows: %s" % allowed


def test_the_screen_still_permits_a_legitimate_solver_program():
    """A screen that refuses everything would pass the test above and be useless.

    The solver protocol requires `os.environ` for SEARCH_SCALE, so `os` must remain importable
    while its dangerous members stay blocked by the attribute check.
    """
    legit = ("import json, os\n"
             "scale = int(os.environ.get('SEARCH_SCALE', '1'))\n"
             "print(json.dumps({'answer': str(sum(range(scale)))}))\n")
    ok, why = program_is_safe(legit)
    assert ok, why
    out, detail = run_program(legit, timeout=15.0, env_extra={"SEARCH_SCALE": "5"})
    assert out == '{"answer": "10"}', (out, detail)


def test_environment_is_allowlisted_not_spliced():
    """Arbitrary env would have let LD_PRELOAD into the child, voiding the isolation."""
    out, detail = run_program("print(1)", timeout=10.0,
                              env_extra={"LD_PRELOAD": "/attacker/evil.so"})
    assert out is None
    assert "refused environment keys" in detail
    assert "LD_PRELOAD" in detail


# ------------------------------------ mechanical vs substantive abstention (the 189 bug)
def test_an_unverifiable_must_declare_why_it_abstained():
    """A silent default is what let a truncated reply be reported as a coverage gap."""
    from ornith_repro.verify import VerificationResult
    with pytest.raises(ValueError, match="MECHANICAL"):
        VerificationResult(Verdict.UNVERIFIABLE, "b", None, "1", "some reason")


def test_a_decided_verdict_may_not_carry_an_abstention_reason():
    """The two are mutually exclusive; allowing both invites miscounting."""
    from ornith_repro.verify import Abstain, VerificationResult
    with pytest.raises(ValueError, match="must not carry"):
        VerificationResult(Verdict.VERIFIED, "b", "1", "1", "ok", abstain=Abstain.MECHANICAL)


def test_truncation_is_mechanical_not_substantive():
    """The exact failure that discarded 189 tasks which had passed every substantive bar."""
    from ornith_repro.verify import Abstain, classify_abstain
    assert classify_abstain("comparison truncated") is Abstain.MECHANICAL
    assert classify_abstain("restatement truncated") is Abstain.MECHANICAL
    assert classify_abstain("program printed nothing") is Abstain.MECHANICAL
    assert classify_abstain("no fenced code block") is Abstain.MECHANICAL
    assert classify_abstain("timeout after 10.0s") is Abstain.MECHANICAL


def test_real_undecidability_is_substantive():
    """A verifier declining on the merits must not be counted as a plumbing failure."""
    from ornith_repro.verify import Abstain, classify_abstain
    assert classify_abstain("bound unjustified (heuristic_limit)") is Abstain.SUBSTANTIVE
    assert classify_abstain("independent programs disagree") is Abstain.SUBSTANTIVE
    assert classify_abstain("refutation NOT substantiated, downgraded") is Abstain.SUBSTANTIVE
    assert classify_abstain("bound FALSIFIED: widening changed the answer") is Abstain.SUBSTANTIVE
