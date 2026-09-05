"""The Lean loop refuses to certify a mis-formalised statement.

The prover is not what is under test here. Lean proves the theorem that was written, so a
compiled proof only shows a formalisation is internally consistent. The component that decides
whether the loop is worth anything is the SECOND agent, which sees the problem text and the
Lean statement alone and judges whether they ask the same question. These tests hold that
component's wiring: a MISMATCH must block certification, an unparseable reply must abstain
rather than pass, and the asserted answer must never reach any part of the loop.

SCOPE, stated so the numbers here are not over-read. With scripted clients these tests
establish the PLUMBING -- that a rejection is honoured and that nothing leaks. They do not
measure how good a real model is at spotting a flipped quantifier; that is an inference
measurement and needs a GPU budget, which has not been spent.
"""

from __future__ import annotations

from ornith_repro.verify import Verdict
from ornith_repro.verify_lean import (
    LeanBackendLoop,
    LeanLoop,
    NullLeanCompiler,
    claimed_answer,
    extract_lean,
    statement_of,
)

PROBLEM = ("Find the sum of all positive integers $n$ for which $n^2 - 3n + 1$ divides "
           "$n^3 - 2n + 5$.")

GOOD = """```lean
theorem sum_of_n : (Finset.filter (fun n => (n^2 - 3*n + 1) ∣ (n^3 - 2*n + 5))
    (Finset.range 100)).sum id = 12 := by
  decide
```"""

# Internally fine, asks the WRONG question: the maximum, not the sum. This is the exact
# error class that produced the mis-keyed task in the first place.
MAX_NOT_SUM = """```lean
theorem max_of_n : (Finset.filter (fun n => (n^2 - 3*n + 1) ∣ (n^3 - 2*n + 5))
    (Finset.range 100)).max' (by decide) = 6 := by
  decide
```"""

WITH_SORRY = """```lean
theorem sum_of_n : True = True := by
  sorry
```"""


class FakeCompiler:
    """Compiler stub with a fixed verdict, so the loop's control flow can be tested."""

    def __init__(self, ok: bool = True, msg: str = "compiled") -> None:
        self.available = True
        self.ok = ok
        self.msg = msg
        self.sources: list[str] = []

    def check(self, source: str) -> tuple[bool, str]:
        """Record the source and return the fixed verdict, rejecting `sorry`."""
        self.sources.append(source)
        if "sorry" in source:
            return False, "proof contains `sorry`"
        return self.ok, self.msg


class ScriptClient:
    """Client returning a fixed formalisation and a fixed faithfulness verdict."""

    def __init__(self, lean: str = GOOD, faithful: str = "FAITHFUL\nlooks right") -> None:
        self.lean = lean
        self.faithful = faithful
        self.prompts: list[str] = []

    def generate(self, prompt: str, max_new_tokens: int, seed: int) -> tuple[str, bool]:
        """Dispatch on prompt kind, recording everything shown to the model."""
        self.prompts.append(prompt)
        if "Lean 4 theorem" in prompt:
            return self.lean, False
        if "FAITHFUL or MISMATCH" in prompt:
            return self.faithful, False
        return "", False


def test_statement_is_separated_from_its_proof():
    """The checker must see the claim, not the argument that produced it."""
    stmt = statement_of(extract_lean(GOOD))
    assert stmt.startswith("theorem")
    assert "decide" not in stmt, "proof leaked into the statement shown to the checker"


def test_claimed_answer_is_the_final_right_hand_side():
    """The loop's claim is read off the statement, not invented."""
    assert claimed_answer(statement_of(extract_lean(GOOD))) == "12"
    assert claimed_answer(statement_of(extract_lean(MAX_NOT_SUM))) == "6"


def test_mismatched_statement_is_refused_even_though_it_compiles():
    """The headline behaviour: a compiling but WRONG formalisation must not certify.

    The statement computes a maximum while the problem asks for a sum. The compiler is happy.
    Only the independent checker stands between that and a false VERIFIED.
    """
    client = ScriptClient(lean=MAX_NOT_SUM, faithful="MISMATCH\nasks for the maximum")
    loop = LeanLoop(client, compiler=FakeCompiler(ok=True), max_attempts=2)
    res = loop.run(PROBLEM)
    assert res.verdict is Verdict.UNVERIFIABLE
    assert "MISMATCH" in res.detail


def test_faithful_and_compiled_yields_a_claim():
    """A compiled, faithful formalisation produces the answer it states."""
    loop = LeanLoop(ScriptClient(), compiler=FakeCompiler(ok=True), max_attempts=2)
    res = loop.run(PROBLEM)
    assert res.verdict is Verdict.VERIFIED
    assert res.claimed == "12"
    assert res.attempts_used == 1


def test_unparseable_faithfulness_reply_abstains():
    """An unreadable verdict must not be read as approval."""
    client = ScriptClient(faithful="I think it's probably fine?")
    loop = LeanLoop(client, compiler=FakeCompiler(ok=True), max_attempts=1)
    res = loop.run(PROBLEM)
    assert res.verdict is Verdict.UNVERIFIABLE
    assert "unparseable" in res.detail


def test_sorry_is_not_a_proof():
    """A proof containing `sorry` must never reach the faithfulness stage."""
    loop = LeanLoop(ScriptClient(lean=WITH_SORRY), compiler=FakeCompiler(ok=True),
                    max_attempts=1)
    res = loop.run(PROBLEM)
    assert res.verdict is Verdict.UNVERIFIABLE
    assert "sorry" in res.detail


def test_missing_toolchain_declines_rather_than_passing():
    """With no Lean installed the loop must abstain, never certify."""
    loop = LeanLoop(ScriptClient(), compiler=NullLeanCompiler(), max_attempts=1)
    res = loop.run(PROBLEM)
    assert res.verdict is Verdict.UNVERIFIABLE
    assert "no Lean toolchain" in res.detail


def test_retries_are_bounded_and_recorded():
    """Attempts are capped and counted, so answer-shopping is visible rather than hidden."""
    loop = LeanLoop(ScriptClient(), compiler=FakeCompiler(ok=False, msg="type error"),
                    max_attempts=3)
    res = loop.run(PROBLEM)
    assert res.attempts_used == 3
    assert len(res.attempts) == 3
    assert res.detail.count("attempt") == 3


def test_the_asserted_answer_never_reaches_the_loop():
    """No prompt in the loop may contain the key; retries must not become answer-shopping."""
    client = ScriptClient()
    backend = LeanBackendLoop(LeanLoop(client, compiler=FakeCompiler(ok=True)))
    backend.verify(PROBLEM, "99999")
    assert client.prompts, "loop never called the model"
    assert not any("99999" in p for p in client.prompts)


def test_backend_refutes_when_a_faithful_proof_contradicts_the_key():
    """A faithful, compiled claim that disagrees with the key is a refutation."""
    backend = LeanBackendLoop(LeanLoop(ScriptClient(), compiler=FakeCompiler(ok=True)))
    assert backend.verify(PROBLEM, "6").verdict is Verdict.REFUTED
    assert backend.verify(PROBLEM, "12").verdict is Verdict.VERIFIED


def test_nominal_independence_is_disclosed():
    """When formaliser and checker are one client, the result must say so."""
    loop = LeanLoop(ScriptClient(), compiler=FakeCompiler(ok=False), max_attempts=1)
    assert "same client" in loop.run(PROBLEM).detail


class FlakyChecker:
    """Formalises fine, but its faithfulness replies vary across samples.

    Models the measured behaviour that motivated unanimity: on the adversarial set a single
    sample falsely certified 1 mutant in 42, while no mutant survived all three samples.
    """

    def __init__(self, verdicts: list[str]) -> None:
        self.lean = GOOD
        self.verdicts = verdicts
        self.calls = 0

    def generate(self, prompt: str, max_new_tokens: int, seed: int) -> tuple[str, bool]:
        """Return the formalisation, then the scripted verdict sequence."""
        if "Lean 4 theorem" in prompt:
            return self.lean, False
        if "FAITHFUL or MISMATCH" in prompt:
            v = self.verdicts[min(self.calls, len(self.verdicts) - 1)]
            self.calls += 1
            return v, False
        return "", False


def test_unanimity_is_required_across_samples():
    """One dissenting MISMATCH must veto, however many samples agreed before it.

    A majority rule would have certified the mutant that a single sample missed; unanimity is
    what drove false certifications to zero on the measured set.
    """
    from ornith_repro.verify_lean import check_faithful
    flaky = FlakyChecker(["FAITHFUL\nfine", "FAITHFUL\nfine", "MISMATCH\nwrong quantity"])
    ok, detail = check_faithful(flaky, PROBLEM, "theorem t : x = 1", samples=3)
    assert ok is False
    assert "MISMATCH on sample 3" in detail


def test_unanimous_agreement_passes_and_says_how_many():
    """All-FAITHFUL passes and records the sample count, so the bar is visible."""
    from ornith_repro.verify_lean import check_faithful
    steady = FlakyChecker(["FAITHFUL\nfine"])
    ok, detail = check_faithful(steady, PROBLEM, "theorem t : x = 1", samples=3)
    assert ok is True
    assert "unanimously across 3 samples" in detail
    assert steady.calls == 3, "unanimity must actually sample three times"
