"""A gold verifier optimised for SOUNDNESS, not for coverage.

THE TRADE, stated plainly so nobody later mistakes silence for failure. This verifier is
built so that when it says VERIFIED or REFUTED it is essentially never wrong, and it pays for
that by saying UNVERIFIABLE far more often. A high UNVERIFIABLE rate is the design working,
not the design failing. One hundred percent correctness is not achievable here; near-zero
wrong DECISIONS, with abstention everywhere else, is.

THE ASYMMETRY THE DESIGN IS BUILT AROUND. Refuting a key is far easier to make sound than
verifying one, because exhibiting a counterexample is cheap while proving none was missed is
hard. The two verdicts therefore face different bars:

  REFUTED  needs a WITNESS -- the concrete objects the search found -- and every witness item
           must be re-checked against the STATEMENT by a separately generated predicate
           program that never saw the search. On the task that motivated this, the witness is
           {1, 2, 3, 6}; each substitutes back into "n^2-3n+1 divides n^3-2n+5" and is
           confirmed by direct arithmetic, and their sum is 12 against an asserted 6. The
           verdict then rests on arithmetic a reader can redo, not on trusting the search.
           If the witness does not check out, the verdict is downgraded to UNVERIFIABLE.

  VERIFIED needs (a) two INDEPENDENTLY produced programs to agree on the answer, (b) a
           JUSTIFIED search bound, and (c) a round-trip restatement that still asks the
           original question. A program that searches to 100 and finds four solutions has
           proven nothing unless something rules out solutions beyond 100, so an unjustified
           bound yields UNVERIFIABLE even when the answer happens to be right.

THE FAILURE THAT DEFEATS NOMINAL INDEPENDENCE. Two programs from the same model can misread
the statement the SAME way, agree, and both be wrong -- which is exactly the error class that
motivated this module, where the proposer answered "largest solution" while asking for "sum of
solutions". Signature-level independence does not touch that. The ROUND-TRIP CHECK does: a
model restates the problem from the generated PROGRAM ALONE, without the original text, and
that restatement is compared against the original. If what is being asked differs, the program
is solving a different problem and the verdict is UNVERIFIABLE.

Which model produced which program is recorded on every artifact, so a later reader can see
whether independence was real or nominal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .loop import answers_match
from .symbolic import symbolic_compare
from .verify import (
    Verdict,
    unverifiable,
    VerificationResult,
    extract_code,
    program_is_safe,
    run_program,
)

#: Aggregations a solver may declare. A closed set on purpose: the final step from a witness
#: to an answer is where the "sum versus largest" error lives, so it is auditable rather than
#: free-form code.
AGGREGATORS = {
    "sum": lambda xs: sum(xs),
    "count": lambda xs: len(xs),
    "max": lambda xs: max(xs) if xs else None,
    "min": lambda xs: min(xs) if xs else None,
    "product": lambda xs: _product(xs),
    "none": lambda xs: xs[0] if len(xs) == 1 else None,
}


def _product(xs):
    """Product of a list, or None when empty.

    Args:
        xs: Numbers.

    Returns:
        The product, or None.
    """
    if not xs:
        return None
    out = 1
    for x in xs:
        out *= x
    return out


SOLVER_PROGRAM_PROMPT = """Write a Python program that computes the answer to this problem.

PROBLEM: {problem}

Your program must print ONE line of JSON with exactly these keys:
  "answer"        : the final answer, as a string
  "witness"       : the list of underlying objects your search found (e.g. the values of n
                    that satisfy the condition). Use [] if the problem has no such set.
  "aggregator"    : how "answer" is obtained from "witness" -- one of
                    "sum", "count", "max", "min", "product", "none"
  "bound_kind"    : one of
                    "exhaustive_finite" (the search space is provably finite and you searched
                                         all of it),
                    "proved_bound"      (you can justify that nothing beyond your limit can
                                         qualify -- state the argument in bound_note),
                    "heuristic_limit"   (you picked a limit without proving it suffices)
  "bound_note"    : one sentence justifying the bound, or "" if heuristic

Read the environment variable SEARCH_SCALE (default "1") and multiply your search range by
that integer, so the same program can be re-run over a larger range.

Be honest in "bound_kind". Claiming a proof you do not have is worse than admitting a guess.
Use the standard library only. No network, no file access.

Reply with one ```python fenced block and nothing else.
"""

PREDICATE_PROGRAM_PROMPT = """Write a Python program that CHECKS candidates against a condition.

PROBLEM: {problem}

Do not solve the problem and do not search. Write only a checker:
read a JSON list of candidate values from the environment variable CANDIDATES, and for each
one decide whether it satisfies the condition stated in the problem.

Print ONE line of JSON: {{"results": [true/false, ...]}} in the same order as the input.

Use the standard library only. No network, no file access.

Reply with one ```python fenced block and nothing else.
"""

RESTATE_PROMPT = """Below is a Python program. Without guessing at any wider context, describe
in one sentence what question this program answers.

```python
{code}
```

Reply with one sentence and nothing else.
"""

COMPARE_PROMPT = """Do these two descriptions ask for the SAME quantity?

A: {original}

B: {restated}

Answer SAME if they ask for the same quantity, or DIFFERENT if they ask for different
quantities (for example one asks for a sum and the other for a largest value, or one asks for
a count and the other for the items themselves).

Reply with exactly one word: SAME or DIFFERENT.
"""


@dataclass
class SolveArtifact:
    """One structured solve, with everything the soundness checks need."""

    answer: str
    witness: list = field(default_factory=list)
    aggregator: str = "none"
    bound_kind: str = "heuristic_limit"
    bound_note: str = ""
    code: str = ""
    model: str = "unknown"


def parse_artifact(line: str) -> SolveArtifact | None:
    """Parse the JSON line a solver program prints.

    Args:
        line: The program's last printed line.

    Returns:
        A `SolveArtifact`, or None when the line is not the agreed shape. A malformed line is
        never guessed at.
    """
    try:
        d = json.loads(line)
    except (ValueError, TypeError):
        return None
    if not isinstance(d, dict) or "answer" not in d:
        return None
    agg = d.get("aggregator", "none")
    if agg not in AGGREGATORS:
        agg = "none"
    bk = d.get("bound_kind", "heuristic_limit")
    if bk not in ("exhaustive_finite", "proved_bound", "heuristic_limit"):
        bk = "heuristic_limit"
    wit = d.get("witness", [])
    if not isinstance(wit, list):
        wit = []
    return SolveArtifact(answer=str(d["answer"]), witness=wit, aggregator=agg,
                         bound_kind=bk, bound_note=str(d.get("bound_note", "")))


def solve_structured(client, problem: str, seed: int, model_name: str,
                     max_new_tokens: int = 12288, timeout: float = 15.0,
                     scale: int = 1) -> SolveArtifact | None:
    """Generate and run one structured solver program.

    Args:
        client: Generation client.
        problem: The statement, and nothing else.
        seed: Generation seed.
        model_name: Recorded on the artifact so independence is auditable.
        max_new_tokens: Generation cap.
        timeout: Sandbox limit.
        scale: SEARCH_SCALE handed to the program.

    Returns:
        A `SolveArtifact`, or None if the program was unusable.
    """
    text, truncated = client.generate(
        SOLVER_PROGRAM_PROMPT.format(problem=problem), max_new_tokens, seed)
    if truncated:
        return None
    code = extract_code(text)
    if not code:
        return None
    ok, _ = program_is_safe(code)
    if not ok:
        return None
    out, _ = run_program(code, timeout=timeout, env_extra={"SEARCH_SCALE": str(scale)})
    if out is None:
        return None
    art = parse_artifact(out)
    if art is None:
        return None
    art.code = code
    art.model = model_name
    return art


def check_witness(client, problem: str, items: list, seed: int,
                  max_new_tokens: int = 12288, timeout: float = 15.0) -> tuple[bool, str]:
    """Re-check every witness item against the STATEMENT with a separate predicate program.

    The predicate program is generated from the problem alone and is told not to search, so it
    is a much simpler artifact than the solver and does not inherit the solver's reading of
    the task through code. It never sees the solver's program or its answer.

    Args:
        client: Generation client.
        problem: The statement.
        items: Witness items to check.
        seed: Generation seed, distinct from the solver's.
        max_new_tokens: Generation cap.
        timeout: Sandbox limit.

    Returns:
        `(all_items_confirmed, detail)`.
    """
    if not items:
        return False, "empty witness"
    text, truncated = client.generate(
        PREDICATE_PROGRAM_PROMPT.format(problem=problem), max_new_tokens, seed)
    if truncated:
        return False, "predicate generation truncated"
    code = extract_code(text)
    if not code:
        return False, "predicate had no code block"
    ok, why = program_is_safe(code)
    if not ok:
        return False, "predicate refused: %s" % why
    out, detail = run_program(code, timeout=timeout,
                              env_extra={"CANDIDATES": json.dumps(items)})
    if out is None:
        return False, "predicate did not run: %s" % detail
    try:
        res = json.loads(out).get("results")
    except (ValueError, TypeError):
        return False, "predicate output not JSON"
    if not isinstance(res, list) or len(res) != len(items):
        return False, "predicate returned %r for %d items" % (res, len(items))
    if not all(bool(x) for x in res):
        bad = [items[i] for i, x in enumerate(res) if not x]
        return False, "witness items rejected by predicate: %r" % (bad[:5],)
    return True, "all %d witness items confirmed" % len(items)


def round_trip_ok(client, problem: str, code: str, seed: int,
                  max_new_tokens: int = 4096) -> tuple[bool, str]:
    """Restate the question from the PROGRAM ALONE and compare it with the original.

    This is the guard against the failure that nominal independence cannot catch: two programs
    from one model misreading the statement identically. If the restatement asks for a
    different quantity than the original, the program is answering a different question.

    Args:
        client: Generation client.
        problem: The original statement.
        code: The generated program.
        seed: Generation seed.
        max_new_tokens: Generation cap.

    Returns:
        `(same_question, detail)`.
    """
    text, truncated = client.generate(RESTATE_PROMPT.format(code=code), max_new_tokens, seed)
    if truncated:
        return False, "restatement truncated"
    restated = text.split("</think>")[-1].strip().splitlines()
    restated = restated[-1].strip() if restated else ""
    if not restated:
        return False, "empty restatement"
    # NOT 64. That was the first value and it cost 189 of 634 tasks their verdict: with
    # thinking enabled the model emits reasoning tokens before the one-word answer, so a
    # 64-token cap truncated the reply on every task that had already passed both programs,
    # an exhaustive_finite bound and the stability re-run. The fourth token-budget trap in
    # this project, and the only one that was self-inflicted by the verifier itself.
    verdict, truncated2 = client.generate(
        COMPARE_PROMPT.format(original=problem, restated=restated), 4096, seed + 1)
    if truncated2:
        return False, "comparison truncated"
    word = verdict.split("</think>")[-1].strip().upper()
    if "DIFFERENT" in word:
        return False, "round-trip DIFFERENT: %r" % restated[:160]
    if "SAME" in word:
        return True, "round-trip SAME"
    return False, "comparison inconclusive: %r" % word[:80]


class SoundVerifier:
    """Verifier tuned so that a DECISION is essentially never wrong.

    Args:
        primary: Client producing the first solver program.
        secondary: Client producing the independent second program. May be a different model;
            when it is the same object, independence is only nominal and the result records
            that so a reader is not misled.
        primary_name: Model label recorded on artifacts.
        secondary_name: Model label recorded on artifacts.
        timeout: Sandbox limit per program.
        bound_scale: Factor for the bound-stability re-run.
    """

    def __init__(self, primary, secondary=None, primary_name="primary",
                 secondary_name="secondary", timeout: float = 15.0,
                 bound_scale: int = 10) -> None:
        self.primary = primary
        self.secondary = secondary if secondary is not None else primary
        self.primary_name = primary_name
        self.secondary_name = secondary_name
        self.nominal_independence = secondary is None or secondary is primary
        self.timeout = timeout
        self.bound_scale = bound_scale

    def verify(self, problem: str, asserted: str) -> VerificationResult:
        """Return a verdict that abstains unless its bar is fully met.

        Args:
            problem: The statement. Neither solver ever receives `asserted`.
            asserted: The proposer's key, used only for the final comparison.

        Returns:
            A `VerificationResult` whose detail records every check that ran.
        """
        notes: list[str] = []
        a = solve_structured(self.primary, problem, 101, self.primary_name,
                             timeout=self.timeout)
        if a is None:
            return unverifiable("sound", None, asserted,
                                      "primary solver produced no usable artifact")
        notes.append("primary(%s)=%s bound=%s" % (a.model, a.answer, a.bound_kind))

        # ---- how the computed answer compares with the key -------------------------------
        # Three-valued on purpose. String comparison refuted 49 of 675 PUBLISHED OlympiadBench
        # keys, and reading them showed almost none was a mathematical disagreement:
        # '$\\frac{19}{34}$' against '19/34', '$8$,$4$' against '4, 8', an interval written
        # two ways. A two-valued comparator cannot express "I cannot compare these", so it
        # reports a formatting difference as a refutation.
        state, cmp_why = symbolic_compare(a.answer, asserted)
        notes.append("compare=%s (%s)" % (state, cmp_why[:60]))
        if state == "indeterminate":
            return unverifiable("sound", a.answer, asserted, "; ".join(notes), a.code)

        # ---- refutation path: cheap, and made sound by an independently checked witness ----
        if state == "different" and a.witness:
            ok, detail = check_witness(self.secondary, problem, a.witness, 202,
                                       timeout=self.timeout)
            notes.append("witness: %s" % detail)
            if ok:
                diag = self._diagnose(a, asserted)
                if diag:
                    notes.append(diag)
                return VerificationResult(Verdict.REFUTED, "sound", a.answer, asserted,
                                          "; ".join(notes), a.code)
            notes.append("refutation NOT substantiated, downgraded")
            return unverifiable("sound", a.answer, asserted,
                                      "; ".join(notes), a.code)

        # ---- verification path: three independent bars, all must be met ----
        if state == "different":
            notes.append("disagrees but no witness to substantiate a refutation")
            return unverifiable("sound", a.answer, asserted,
                                      "; ".join(notes), a.code)

        if a.bound_kind == "heuristic_limit":
            notes.append("bound unjustified (heuristic_limit): answer matches but nothing "
                         "rules out solutions beyond the search limit")
            return unverifiable("sound", a.answer, asserted,
                                      "; ".join(notes), a.code)

        wide = solve_structured(self.primary, problem, 101, self.primary_name,
                                timeout=self.timeout, scale=self.bound_scale)
        if wide is None:
            notes.append("bound-stability re-run failed")
            return unverifiable("sound", a.answer, asserted,
                                      "; ".join(notes), a.code)
        if not answers_match(wide.answer, a.answer):
            notes.append("bound FALSIFIED: widening the search changed the answer %s -> %s"
                         % (a.answer, wide.answer))
            return unverifiable("sound", a.answer, asserted,
                                      "; ".join(notes), a.code)
        notes.append("bound stable at scale %d" % self.bound_scale)

        b = solve_structured(self.secondary, problem, 303, self.secondary_name,
                             timeout=self.timeout)
        if b is None:
            notes.append("second independent program produced no usable artifact")
            return unverifiable("sound", a.answer, asserted,
                                      "; ".join(notes), a.code)
        notes.append("secondary(%s)=%s" % (b.model, b.answer))
        if not answers_match(b.answer, a.answer):
            notes.append("independent programs disagree")
            return unverifiable("sound", a.answer, asserted,
                                      "; ".join(notes), a.code)

        ok, detail = round_trip_ok(self.secondary, problem, a.code, 404)
        notes.append(detail)
        if not ok:
            return unverifiable("sound", a.answer, asserted,
                                      "; ".join(notes), a.code)

        if self.nominal_independence:
            notes.append("NOTE: both programs came from the same client; independence is "
                         "nominal, not across models")
        return VerificationResult(Verdict.VERIFIED, "sound", a.answer, asserted,
                                  "; ".join(notes), a.code)

    def _diagnose(self, art: SolveArtifact, asserted: str) -> str:
        """Say whether the asserted key equals a DIFFERENT aggregation of the same witness.

        This names the error class that motivated the module: a proposer that answered
        "largest solution" while asking for "sum of solutions" leaves a witness whose max
        equals the asserted key.

        Args:
            art: The solver artifact carrying the witness.
            asserted: The asserted key.

        Returns:
            A diagnosis string, or "" when nothing matches.
        """
        nums = [x for x in art.witness if isinstance(x, (int, float))]
        if not nums:
            return ""
        for name, fn in AGGREGATORS.items():
            if name == art.aggregator:
                continue
            try:
                val = fn(nums)
            except (TypeError, ValueError):
                continue
            if val is not None and answers_match(str(val), asserted):
                return ("asserted key equals %s(witness)=%s while the question asks for "
                        "%s=%s -- the proposer answered a different aggregation"
                        % (name, val, art.aggregator, art.answer))
        return ""
