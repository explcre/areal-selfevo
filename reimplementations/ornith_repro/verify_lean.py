"""A proof-assistant verification loop, built around the check that actually carries it.

THE DISTINCTION THIS MODULE IS ORGANISED AROUND. Lean proves the theorem you WROTE, not the
theorem you MEANT. A compiler-checked proof is therefore evidence that a formalisation is
internally consistent, and is NOT evidence that the answer key is right. The load-bearing
component is the second agent -- which sees only the original problem text and the Lean
STATEMENT, never the asserted answer, never the proof, never the first agent's reasoning --
and decides whether the statement faithfully encodes the problem. If that agent is weak the
whole loop is decorative, however impressive the proofs are.

COVERAGE AND SOUNDNESS ARE DIFFERENT QUANTITIES AND THIS MODULE TOUCHES BOTH DIFFERENTLY.
Coverage is the share of problems that get a decisive verdict; soundness is the share of
decisive verdicts that are correct. Adding this backend raises COVERAGE, because it can
settle problems enumeration cannot. Only the faithfulness check raises SOUNDNESS. Toward
100% is a coverage goal; soundness can approach but never reach it.

RETRIES ARE BOUNDED AND RECORDED, because looping until something compiles selects for
formalisations that happen to compile rather than for correct ones. `attempts_used` is on
every result so a reader can see that a verdict needed nine tries. The asserted answer is
never visible at any point in the loop, so retrying cannot become answer-shopping: the first
agent must derive its own answer and formalise THAT.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Protocol

from .loop import answers_match
from .verify import Verdict, VerificationResult, unverifiable

FORMALISE_PROMPT = """Solve this problem and write your solution as a Lean 4 theorem with a
complete proof.

PROBLEM: {problem}

Requirements:
- Work out the answer yourself. State the theorem so that the answer appears explicitly on the
  right-hand side of the final equality.
- The theorem statement must capture the problem exactly: the same quantity, the same
  quantifiers, the same constraints.
- Give a complete proof. Do not use `sorry`.

Reply with one ```lean fenced block and nothing else.
"""

FAITHFUL_PROMPT = """You are checking whether a formal statement matches a problem.

PROBLEM (natural language):
{problem}

FORMAL STATEMENT (Lean 4):
{statement}

Does the formal statement express exactly the same question as the problem? Check especially:
- is the same QUANTITY being asked for (a sum is not a maximum, a count is not a list)?
- are the QUANTIFIERS the same (for all versus there exists, all n versus some n)?
- are the CONSTRAINTS the same (positive integers versus all integers, strict versus
  non-strict inequalities)?
- is the DOMAIN the same?

Do not attempt to solve the problem and do not judge whether the stated answer is correct.
Judge only whether the statement asks the problem's question.

Reply with exactly one word on the first line: FAITHFUL or MISMATCH.
On a second line give a short reason.
"""


class LeanCompiler(Protocol):
    """Checks a Lean source file, returning whether it compiles."""

    available: bool

    def check(self, source: str) -> tuple[bool, str]:
        """Compile `source`.

        Args:
            source: Lean 4 source.

        Returns:
            `(compiled_without_error, message)`.
        """
        ...


class NullLeanCompiler:
    """Stands in when no Lean toolchain is installed.

    It reports unavailability rather than pretending to check, so the loop degrades to
    UNVERIFIABLE. A compiler that silently returned success would turn this backend into a
    machine for certifying arbitrary formalisations.
    """

    available = False

    def check(self, source: str) -> tuple[bool, str]:
        """Always decline, with a reason."""
        return False, "no Lean toolchain installed; cannot check the proof"


class SubprocessLeanCompiler:
    """Runs a real `lean` binary over the source, with a hard timeout.

    Args:
        lean_bin: Path to the `lean` executable.
        timeout: Wall-clock seconds.
        extra_args: Additional arguments, e.g. to point at a mathlib environment.
    """

    def __init__(self, lean_bin: str = "lean", timeout: float = 120.0,
                 extra_args: tuple[str, ...] = ()) -> None:
        self.lean_bin = lean_bin
        self.timeout = timeout
        self.extra_args = tuple(extra_args)

    @property
    def available(self) -> bool:
        """Whether the configured binary exists and runs."""
        try:
            subprocess.run([self.lean_bin, "--version"], capture_output=True,
                           timeout=30, check=False)
            return True
        except Exception:  # noqa: BLE001
            return False

    def check(self, source: str) -> tuple[bool, str]:
        """Compile `source`, treating `sorry` as a failure rather than a proof."""
        if "sorry" in source:
            return False, "proof contains `sorry`, which proves nothing"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "Check.lean")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(source)
            try:
                proc = subprocess.run([self.lean_bin, *self.extra_args, path],
                                      capture_output=True, text=True,
                                      timeout=self.timeout, cwd=tmp)
            except subprocess.TimeoutExpired:
                return False, "lean timed out after %.0fs" % self.timeout
            except Exception as exc:  # noqa: BLE001
                return False, "lean failed to run: %r" % (exc,)
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout or "")[-400:]
        return True, "compiled"


def extract_lean(text: str) -> str | None:
    """Pull the last fenced Lean block from a completion.

    Args:
        text: Raw model output.

    Returns:
        The Lean source, or None.
    """
    blocks = re.findall(r"```(?:lean4?)?\s*\n(.*?)```", text, re.S)
    return blocks[-1].strip() if blocks else None


def statement_of(source: str) -> str | None:
    """Return the theorem statement, i.e. everything up to the proof.

    The faithfulness checker must see the statement WITHOUT the proof, so that it judges what
    is being claimed rather than being led by how it was argued.

    Args:
        source: Lean source.

    Returns:
        The statement text, or None when no theorem is present.
    """
    m = re.search(r"(theorem|lemma|example)\b", source)
    if not m:
        return None
    body = source[m.start():]
    cut = body.find(":=")
    return (body[:cut] if cut > 0 else body).strip()


def claimed_answer(statement: str) -> str | None:
    """Extract the answer the statement asserts, i.e. the final right-hand side.

    Args:
        statement: The theorem statement.

    Returns:
        The claimed value, or None when the statement has no final equality.
    """
    s = statement.rstrip()
    for tail in (")", ":"):
        s = s.rstrip(tail).rstrip()
    idx = s.rfind("=")
    if idx < 0 or idx + 1 >= len(s):
        return None
    val = s[idx + 1:].strip().rstrip(")").strip()
    return val or None


@dataclass
class LeanAttempt:
    """One pass through the loop, kept so the retry distribution can be reported."""

    source: str = ""
    statement: str = ""
    compiled: bool = False
    compile_msg: str = ""
    faithful: bool | None = None
    faithful_msg: str = ""


@dataclass
class LeanLoopResult:
    """The loop's outcome plus the accounting a reader needs to trust it."""

    verdict: Verdict
    claimed: str | None = None
    attempts_used: int = 0
    attempts: list = field(default_factory=list)
    detail: str = ""


def check_faithful(checker_client, problem: str, statement: str,
                   max_new_tokens: int = 1024, seed: int = 77,
                   samples: int = 3) -> tuple[bool, str]:
    """Ask a SEPARATE agent whether the Lean statement encodes the problem.

    It is shown the problem text and the statement, and nothing else: not the proof, not the
    asserted key, not the first agent's reasoning. That isolation is the whole point, because
    the failure being guarded against is a formalisation that is internally fine and asks the
    wrong question -- a sum formalised as a maximum, or a quantifier flipped.

    Args:
        checker_client: A generation client, ideally a different model from the formaliser.
        problem: The original natural-language problem.
        statement: The Lean statement, proof stripped.
        max_new_tokens: Generation cap.
        seed: Generation seed.

    UNANIMITY, and it is measured rather than assumed. On an adversarial set of 22 statements
    run 3 times against `qwen38-27b`, a SINGLE sample caught 40 of 42 mutants (0.952) and
    falsely certified 1 (0.024). Requiring all `samples` replies to say FAITHFUL drove false
    certifications to **0 of 14 mutant cases**, while the false-rejection rate on faithful
    statements was unchanged at 0.125 -- the one faithful statement that ever drew a MISMATCH
    already drew one. So unanimity buys soundness here at no measured cost in coverage, for a
    3x inference bill that came to 48 seconds for the whole set.

    Args:
        samples: Replies that must ALL say FAITHFUL. Any MISMATCH or unparseable reply
            rejects, because the safe default is to abstain.

    Returns:
        `(faithful, detail)`. An unparseable reply is treated as NOT faithful.
    """
    verdicts: list[str] = []
    for i in range(max(1, samples)):
        text, truncated = checker_client.generate(
            FAITHFUL_PROMPT.format(problem=problem, statement=statement),
            max_new_tokens, seed + i)
        if truncated:
            return False, "faithfulness check truncated on sample %d" % (i + 1)
        body = text.split("</think>")[-1].strip()
        head = body.splitlines()[0].strip().upper() if body else ""
        if "MISMATCH" in head:
            return False, "MISMATCH on sample %d: %s" % (
                i + 1, " ".join(body.splitlines()[1:])[:200])
        if "FAITHFUL" not in head:
            return False, "faithfulness reply unparseable on sample %d: %r" % (
                i + 1, head[:80])
        verdicts.append("FAITHFUL")
    return True, "FAITHFUL unanimously across %d samples" % len(verdicts)


class LeanLoop:
    """Propose, compile, and independently check the statement -- in that order.

    Args:
        formaliser: Client that writes the Lean theorem and proof.
        checker: Client that judges faithfulness. Should be a DIFFERENT model; when it is the
            same object the result says so, so nominal independence is not mistaken for real.
        compiler: A `LeanCompiler`.
        max_attempts: Hard cap on retries, recorded on every result.
    """

    name = "lean_loop"

    def __init__(self, formaliser, checker=None, compiler: LeanCompiler | None = None,
                 max_attempts: int = 3) -> None:
        self.formaliser = formaliser
        self.checker = checker if checker is not None else formaliser
        self.nominal_independence = checker is None or checker is formaliser
        self.compiler = compiler or NullLeanCompiler()
        self.max_attempts = max_attempts

    def run(self, problem: str) -> LeanLoopResult:
        """Attempt the loop, stopping at the first compiled AND faithful formalisation."""
        attempts: list[LeanAttempt] = []
        for i in range(self.max_attempts):
            att = LeanAttempt()
            attempts.append(att)
            text, truncated = self.formaliser.generate(
                FORMALISE_PROMPT.format(problem=problem), 8192, 8100 + i)
            if truncated:
                att.compile_msg = "formalisation truncated"
                continue
            src = extract_lean(text)
            if not src:
                att.compile_msg = "no lean block"
                continue
            att.source = src
            stmt = statement_of(src)
            if not stmt:
                att.compile_msg = "no theorem statement found"
                continue
            att.statement = stmt
            att.compiled, att.compile_msg = self.compiler.check(src)
            if not att.compiled:
                continue
            # Only now, and only on the statement: the independent faithfulness check.
            att.faithful, att.faithful_msg = check_faithful(self.checker, problem, stmt)
            if att.faithful:
                return LeanLoopResult(Verdict.VERIFIED, claimed_answer(stmt), i + 1,
                                      attempts, "compiled and judged faithful")
        detail = "; ".join(
            "attempt %d: %s%s" % (j + 1, a.compile_msg or "compiled",
                                  "" if a.faithful is None else " / %s" % a.faithful_msg)
            for j, a in enumerate(attempts))
        if self.nominal_independence:
            detail += "; NOTE: formaliser and checker are the same client"
        return LeanLoopResult(Verdict.UNVERIFIABLE, None, len(attempts), attempts, detail)


class LeanBackendLoop:
    """Adapter presenting the loop as a verifier over an asserted key.

    The loop never receives the asserted answer; it is compared only here, after the loop has
    committed to a claim, exactly as the executable backend works.
    """

    name = "lean_loop"

    def __init__(self, loop: LeanLoop) -> None:
        self.loop = loop

    def verify(self, problem: str, asserted: str) -> VerificationResult:
        """Return VERIFIED/REFUTED only when the loop produced a faithful, compiled claim."""
        res = self.loop.run(problem)
        detail = "attempts=%d; %s" % (res.attempts_used, res.detail)
        if res.verdict is not Verdict.VERIFIED or res.claimed is None:
            return unverifiable(self.name, res.claimed,
                                      asserted, detail)
        verdict = (Verdict.VERIFIED if answers_match(res.claimed, asserted)
                   else Verdict.REFUTED)
        return VerificationResult(verdict, self.name, res.claimed, asserted, detail)
