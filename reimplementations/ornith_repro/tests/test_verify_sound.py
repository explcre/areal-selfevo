"""Soundness of the gold verifier, measured on an adversarial set with known ground truth.

The claim under test is NOT "the verifier is right". It is the weaker, achievable claim that
when it DECIDES it is right, and that it abstains otherwise. So the headline is the confusion
matrix restricted to decisions: how often a VERIFIED or REFUTED verdict contradicts the known
truth. Target zero. A high UNVERIFIABLE rate is not a failure here and is reported separately.

The scenarios are built to attack the specific ways a verifier of this shape goes wrong:
a wrong key, a correct key, a key that is right while the search bound is unjustified, a
claimed bound that widening falsifies, a program that answers a DIFFERENT question and happens
to match, a witness the predicate rejects, and two programs that disagree.
"""

from __future__ import annotations

import json

from ornith_repro.verify import Verdict
from ornith_repro.verify_sound import SoundVerifier

CAUGHT = ("Find the sum of all positive integers $n$ for which $n^2 - 3n + 1$ divides "
          "$n^3 - 2n + 5$.")

SOLVER_TRUE = '''```python
import json, os
scale = int(os.environ.get("SEARCH_SCALE", "1"))
good = [n for n in range(1, 100 * scale)
        if (n * n - 3 * n + 1) != 0 and (n ** 3 - 2 * n + 5) % (n * n - 3 * n + 1) == 0]
print(json.dumps({"answer": str(sum(good)), "witness": good, "aggregator": "sum",
                  "bound_kind": "proved_bound",
                  "bound_note": "n^2-3n+1 must divide 6n+2, forcing n<=9"}))
```'''

SOLVER_HEURISTIC = SOLVER_TRUE.replace('"proved_bound"', '"heuristic_limit"').replace(
    '"n^2-3n+1 must divide 6n+2, forcing n<=9"', '""')

# Claims a proof but its answer grows when the range widens: the claim is falsified.
SOLVER_BAD_BOUND = '''```python
import json, os
scale = int(os.environ.get("SEARCH_SCALE", "1"))
good = [n for n in range(1, 5 * scale)]
print(json.dumps({"answer": str(sum(good)), "witness": good, "aggregator": "sum",
                  "bound_kind": "proved_bound", "bound_note": "claimed"}))
```'''

# Answers a DIFFERENT question (largest, not sum) but happens to match the asserted key.
SOLVER_WRONG_QUESTION = '''```python
import json, os
scale = int(os.environ.get("SEARCH_SCALE", "1"))
good = [n for n in range(1, 100 * scale)
        if (n * n - 3 * n + 1) != 0 and (n ** 3 - 2 * n + 5) % (n * n - 3 * n + 1) == 0]
print(json.dumps({"answer": str(max(good)), "witness": good, "aggregator": "max",
                  "bound_kind": "proved_bound", "bound_note": "divides 6n+2"}))
```'''

PREDICATE_TRUE = '''```python
import json, os
cands = json.loads(os.environ.get("CANDIDATES", "[]"))
out = []
for n in cands:
    d = n * n - 3 * n + 1
    out.append(bool(d != 0 and (n ** 3 - 2 * n + 5) % d == 0))
print(json.dumps({"results": out}))
```'''

PREDICATE_REJECTS = '''```python
import json, os
cands = json.loads(os.environ.get("CANDIDATES", "[]"))
print(json.dumps({"results": [False for _ in cands]}))
```'''


class ScenarioClient:
    """Scripted client that answers each of the verifier's four prompt kinds.

    Args:
        solver: Completion for the solver-program prompt.
        predicate: Completion for the predicate-program prompt.
        restate: Sentence returned when asked what a program computes.
        compare: SAME or DIFFERENT.
        solver2: Optional distinct completion for the second solver call.
    """

    def __init__(self, solver=SOLVER_TRUE, predicate=PREDICATE_TRUE,
                 restate="the sum of all positive integers n satisfying the divisibility",
                 compare="SAME", solver2=None) -> None:
        self.solver = solver
        self.solver2 = solver2 if solver2 is not None else solver
        self.predicate = predicate
        self.restate = restate
        self.compare = compare
        self.seen: list[str] = []

    def generate(self, prompt: str, max_new_tokens: int, seed: int) -> tuple[str, bool]:
        """Dispatch on the prompt kind and return the scripted completion."""
        self.seen.append(prompt)
        if "print ONE line of JSON with exactly these keys" in prompt:
            return (self.solver if seed == 101 else self.solver2), False
        if "Do not solve the problem" in prompt:
            return self.predicate, False
        if "what question this program answers" in prompt:
            return self.restate, False
        if "ask for the SAME quantity" in prompt:
            return self.compare, False
        return "", False


def _verify(asserted, **kw):
    """Run the sound verifier over one scenario."""
    c = ScenarioClient(**kw)
    return SoundVerifier(c, timeout=20.0).verify(CAUGHT, asserted)


# ------------------------------------------------------------------ refutation is sound
def test_wrong_key_is_refuted_with_a_checked_witness():
    """The motivating case: asserted 6, true 12, witness confirmed by the predicate."""
    res = _verify("6")
    assert res.verdict is Verdict.REFUTED, res.detail
    assert res.computed == "12"
    assert "all 4 witness items confirmed" in res.detail


def test_refutation_names_the_aggregation_error():
    """The diagnosis must identify that 6 is max(witness) while the question asks the sum."""
    res = _verify("6")
    assert "max(witness)=6" in res.detail, res.detail


def test_refutation_is_downgraded_when_the_witness_fails_its_check():
    """If the witness does not check out, we must NOT report a refutation."""
    res = _verify("6", predicate=PREDICATE_REJECTS)
    assert res.verdict is Verdict.UNVERIFIABLE
    assert "NOT substantiated" in res.detail


# ---------------------------------------------------------------- verification is strict
def test_correct_key_is_verified_when_every_bar_is_met():
    """Two agreeing programs, a justified and stable bound, and a SAME round trip."""
    res = _verify("12")
    assert res.verdict is Verdict.VERIFIED, res.detail


def test_correct_key_with_unjustified_bound_is_unverifiable():
    """Right answer, admitted guess for a bound: abstain rather than certify."""
    res = _verify("12", solver=SOLVER_HEURISTIC)
    assert res.verdict is Verdict.UNVERIFIABLE
    assert "bound unjustified" in res.detail


def test_claimed_bound_falsified_by_widening_is_unverifiable():
    """A bound claimed as proved but whose answer moves when widened must not verify."""
    res = _verify("10", solver=SOLVER_BAD_BOUND)
    assert res.verdict is Verdict.UNVERIFIABLE
    assert "bound FALSIFIED" in res.detail


def test_program_answering_a_different_question_is_not_verified():
    """The round trip must stop a coincidental match from being certified.

    The program computes max, the problem asks for the sum, and the asserted key is 6, so the
    numbers agree. Only the restatement check separates them.
    """
    res = _verify("6", solver=SOLVER_WRONG_QUESTION, compare="DIFFERENT",
                  restate="the largest positive integer n satisfying the divisibility")
    assert res.verdict is not Verdict.VERIFIED, res.detail
    assert "round-trip DIFFERENT" in res.detail


def test_disagreeing_independent_programs_are_unverifiable():
    """Two programs that do not agree cannot support a VERIFIED verdict."""
    res = _verify("12", solver2=SOLVER_WRONG_QUESTION)
    assert res.verdict is Verdict.UNVERIFIABLE
    assert "disagree" in res.detail


def test_nominal_independence_is_disclosed():
    """When both programs come from one client, the result must say so."""
    res = _verify("12")
    assert "independence is nominal" in res.detail


# ----------------------------------------------------------------- the confusion matrix
def test_confusion_matrix_has_no_wrong_decisions():
    """The headline: over the adversarial set, no VERIFIED or REFUTED verdict is wrong.

    Each scenario carries its ground truth (`key_ok`). A decision is WRONG when the verdict is
    VERIFIED on a bad key or REFUTED on a good one. UNVERIFIABLE is never counted as wrong;
    it is counted separately, because abstention is the price this design pays for soundness.
    """
    scenarios = [
        ("wrong key, witness checks", "6", {}, False),
        ("correct key, all bars met", "12", {}, True),
        ("correct key, heuristic bound", "12", {"solver": SOLVER_HEURISTIC}, True),
        ("correct key, bound falsified", "10", {"solver": SOLVER_BAD_BOUND}, True),
        ("coincidental match, wrong question", "6",
         {"solver": SOLVER_WRONG_QUESTION, "compare": "DIFFERENT",
          "restate": "the largest such n"}, False),
        ("wrong key, witness rejected", "6", {"predicate": PREDICATE_REJECTS}, False),
        ("wrong key, programs disagree", "6", {"solver2": SOLVER_HEURISTIC}, False),
        ("correct key, programs disagree", "12",
         {"solver2": SOLVER_WRONG_QUESTION}, True),
    ]
    decided = wrong = abstained = 0
    rows = []
    for name, asserted, kw, key_ok in scenarios:
        res = _verify(asserted, **kw)
        if res.verdict is Verdict.UNVERIFIABLE:
            abstained += 1
            rows.append((name, res.verdict.value, "abstain"))
            continue
        decided += 1
        ok = (res.verdict is Verdict.VERIFIED) == key_ok
        if not ok:
            wrong += 1
        rows.append((name, res.verdict.value, "correct" if ok else "WRONG"))

    print("\nadversarial confusion matrix")
    for name, verdict, mark in rows:
        print("  %-38s %-13s %s" % (name, verdict, mark))
    print("  decided %d, wrong %d, abstained %d of %d"
          % (decided, wrong, abstained, len(scenarios)))
    assert wrong == 0, "wrong decisions: %s" % [r for r in rows if r[2] == "WRONG"]
    assert decided >= 2, "a verifier that abstains on everything proves nothing"
