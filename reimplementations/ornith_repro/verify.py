"""Independent verification of a generated task's answer key.

WHY THIS EXISTS. The first live iteration proposed a well-posed, correctly stated, solvable
problem -- "the sum of all positive integers n for which n^2-3n+1 divides n^3-2n+5" -- and
asserted the answer 6. The true answer is 12; the proposer found the largest solution and
reported it instead of the sum. That key alone drove the measured success rate to 0.042,
which sits near Ornith's difficulty target and earned the task a high difficulty reward. A
wrong key does not merely add noise: it MANUFACTURES the signal the gate is built to chase,
so "genuinely hard" and "mis-keyed" are confounded inside the one statistic the gate reads.

WHY EXECUTION RATHER THAN PROOF, first. That failure is caught cheaply by enumeration: the
remainder argument bounds n at 9 and a brute-force search settles it in milliseconds. A proof
assistant would give higher assurance at much lower yield, because the model must formalise
the statement correctly AND close a complete proof, and for "find all n such that" it must
additionally show no other solutions exist. So the executable path is primary and Lean is a
registered backend behind the same interface rather than the main route.

THE CONSTRAINT THAT MATTERS MOST, and it is enforced by the types rather than by discipline.
An `AnswerSource.solve` takes the problem STATEMENT ONLY. It has no parameter through which
the asserted answer could reach it, so a backend cannot rationalise toward the key even by
accident. Comparison happens afterwards, in `verify_answer`, which the backend never sees.
On the task above, a program written from the statement prints 12 and refutes the key; a
program written while looking at "6" very likely would not.

THREE-WAY BY DESIGN. VERIFIED / REFUTED / UNVERIFIABLE. Most competition problems will land
in the third bucket, and that is the honest outcome: a binary verifier is forced to guess,
and a guess here silently poisons the training pool. The three rates are reported as a
headline, the way the gold-disagreement rate is.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol

from .loop import answers_match, extract_boxed


class Verdict(str, Enum):
    """The three outcomes a verifier may return.

    UNVERIFIABLE is a first-class result, not a failure: it says the backend could not
    settle the key, which is different from saying the key is wrong.
    """

    VERIFIED = "verified"
    REFUTED = "refuted"
    UNVERIFIABLE = "unverifiable"


class Abstain(str, Enum):
    """WHY a verifier declined, split by whether the problem or the plumbing defeated it.

    The distinction is not bookkeeping. An earlier run reported 93.4% UNVERIFIABLE and a
    coverage of 6.3%, and 189 of those tasks had passed every substantive bar -- two
    independently produced programs agreeing, an exhaustive_finite bound, and that bound
    stable at ten times the search width -- before dying because a 64-token cap truncated a
    one-word SAME/DIFFERENT reply. A harness failure was being reported as an inability to
    decide the problem, which made the verifier look conservative when it was broken.

    MECHANICAL means we could not READ an answer: a truncation, an unparseable reply, a
    program that failed to appear or to run. It is a bug or a budget, and it must never be
    counted as verification coverage.

    SUBSTANTIVE means we could not DECIDE: the bound was unjustified or falsified, two
    programs disagreed, the witness failed its independent check, the round trip asked a
    different question. That is the verifier working as designed.
    """

    NONE = "none"
    MECHANICAL = "mechanical"
    SUBSTANTIVE = "substantive"


@dataclass
class VerificationResult:
    """One verification attempt, with everything needed to audit the verdict."""

    verdict: Verdict
    backend: str
    computed: str | None = None
    asserted: str | None = None
    detail: str = ""
    artefact: str = ""
    abstain: Abstain = Abstain.NONE

    def __post_init__(self) -> None:
        """Refuse an abstention that does not say which kind it is.

        A silent default here is exactly how the truncation hid: an UNVERIFIABLE with no
        stated cause is indistinguishable from a decision not to decide.
        """
        if self.verdict is Verdict.UNVERIFIABLE and self.abstain is Abstain.NONE:
            raise ValueError(
                "an UNVERIFIABLE result must state whether it abstained for a MECHANICAL "
                "reason (we could not read an answer) or a SUBSTANTIVE one (we could not "
                "decide the problem); defaulting to NONE is what let a truncated reply be "
                "reported as a coverage gap."
            )
        if self.verdict is not Verdict.UNVERIFIABLE and self.abstain is not Abstain.NONE:
            raise ValueError("a decided verdict must not carry an abstention reason")


class AnswerSource(Protocol):
    """Produces an answer from the problem STATEMENT ALONE.

    The signature is the safeguard. There is no parameter carrying the asserted answer, so
    no backend can see the key it is being used to check.
    """

    name: str

    def solve(self, problem: str) -> tuple[str | None, str, str]:
        """Answer the problem independently.

        Args:
            problem: The problem statement, and nothing else.

        Returns:
            `(answer or None, detail, artefact)`. `None` means the source could not
            produce an answer, which becomes UNVERIFIABLE rather than a guess.
        """
        ...


# --------------------------------------------------------------- sandboxed execution
def _kill_group(proc) -> None:
    """Kill a child and everything it spawned.

    `subprocess.run` terminates only the direct child, so a program that spawned a
    background process left it running after a timeout was reported.

    Args:
        proc: A `Popen` started with `start_new_session=True`.
    """
    import signal as _signal
    try:
        os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
    try:
        proc.communicate(timeout=5)
    except Exception:  # noqa: BLE001
        pass


def run_program(code: str, timeout: float = 10.0,
                env_extra: dict | None = None) -> tuple[str | None, str]:
    """Execute a short program in a subprocess with a hard timeout and no network.

    THIS IS NOT A SECURITY BOUNDARY, and an audit proved the point: an earlier version of
    this screen let 7 of 9 hostile programs through -- `open` was not banned at all, nor was
    `importlib`, nor `getattr(__builtins__, "__import__")`, and `os.execlp` and
    `os.posix_spawn` were absent from a denylist that contained `execv`. A spawned process
    also survived a "timeout" verdict, because only the direct child was killed. Those holes
    are closed, but an AST denylist is inherently a list of the bypasses someone thought of.
    Run only code generated by a model you are already trusting with the rest of the loop;
    never a third party's.

    The isolation is deliberate but modest, and its limits are stated rather than implied:
    the child runs with `-I` (isolated mode: no user site-packages, no inherited PYTHONPATH),
    an emptied environment, a scratch working directory, and a wall-clock timeout. It is a
    guard against runaway or accidentally-networked generated code, not a security boundary
    against hostile code; nothing here should execute a program from an untrusted third
    party.

    Args:
        code: Python source. Its final answer must be printed on the last non-empty line.
        timeout: Wall-clock seconds before the child is killed.
        env_extra: Extra environment for the child. Used to re-run the SAME program with a
            larger search range, which is how an unjustified bound is falsified.

    Returns:
        `(last printed line or None, detail)`.
    """
    # Only these keys may be set by a caller. Splicing arbitrary env in let LD_PRELOAD
    # through, which would have voided the one real isolation property the emptied
    # environment provided.
    allowed = {"SEARCH_SCALE", "CANDIDATES"}
    rejected = sorted(set(env_extra or {}) - allowed)
    if rejected:
        return None, "refused environment keys %r (allowed: %s)" % (
            rejected, sorted(allowed))
    with tempfile.TemporaryDirectory() as tmp:
        env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "PYTHONHASHSEED": "0",
               **{k: v for k, v in (env_extra or {}).items() if k in allowed}}
        path = os.path.join(tmp, "prog.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(code)
        try:
            # start_new_session so the child gets its own process group: `subprocess.run`
            # kills only the direct child on timeout, and the audit showed a spawned
            # `sleep 45` outliving a "timeout" verdict by seconds. The group is killed below.
            proc = subprocess.Popen(
                [sys.executable, "-I", "-S", path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmp,
                env=env, start_new_session=True,
            )
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                return None, "timeout after %.1fs" % timeout
            proc = subprocess.CompletedProcess(proc.args, proc.returncode, out, err)
        except subprocess.TimeoutExpired:
            return None, "timeout after %.1fs" % timeout
        except Exception as exc:  # noqa: BLE001
            return None, "exec failed: %r" % (exc,)
    if proc.returncode != 0:
        return None, "exit %d: %s" % (proc.returncode, (proc.stderr or "")[-300:])
    lines = [l.strip() for l in (proc.stdout or "").splitlines() if l.strip()]
    if not lines:
        return None, "program printed nothing"
    return lines[-1], "ok"


def program_is_safe(code: str) -> tuple[bool, str]:
    """Refuse generated code that imports the network or touches the process.

    Checked on the AST rather than by string search, so a comment mentioning `socket` does
    not trip it and an aliased import does not slip past it. This project has already been
    bitten by a text-matching audit that a docstring could silence.

    Args:
        code: Python source.

    Returns:
        `(ok, reason)`.
    """
    banned = {"socket", "urllib", "requests", "httpx", "http", "ftplib", "smtplib",
              "subprocess", "shutil", "ctypes", "multiprocessing", "webbrowser",
              # added after an audit ran 7 of 9 hostile programs straight through:
              # `os` is NOT banned as a module: the solver protocol requires
              # os.environ to read SEARCH_SCALE/CANDIDATES. Its dangerous members are
              # blocked by `banned_attrs`/`banned_names` below instead, which also catches
              # `os.open` and `os.system` through the attribute check.
              "importlib", "sys", "pathlib", "io", "tempfile", "glob", "pickle",
              "runpy", "code", "pty", "signal", "resource", "asyncio", "selectors",
              "telnetlib", "poplib", "imaplib", "xmlrpc", "ssl", "ssh", "paramiko"}
    banned_names = {"open", "eval", "exec", "compile", "__import__", "getattr", "setattr",
                    "globals", "locals", "vars", "breakpoint", "memoryview", "input"}
    banned_attrs = {"system", "popen", "execv", "execve", "execl", "execlp", "execle",
                    "execvp", "execvpe", "fork", "forkpty", "spawnv", "spawnl",
                    "posix_spawn", "posix_spawnp", "kill", "killpg", "remove", "unlink",
                    "rmdir", "chmod", "chown", "rename", "symlink", "link", "truncate",
                    "write_text", "write_bytes", "read_text", "read_bytes"}
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, "syntax error: %s" % exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                if al.name.split(".")[0] in banned:
                    return False, "imports %s" % al.name
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in banned:
                return False, "imports from %s" % node.module
        elif isinstance(node, ast.Name):
            # A LOAD, not only a call: `f = os.system` then `f(...)` was invisible before.
            if node.id in banned_names:
                return False, "references %s" % node.id
        elif isinstance(node, ast.Attribute):
            if node.attr in banned_attrs or node.attr in banned_names:
                return False, "references attribute %s" % node.attr
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return False, "references dunder %s" % node.attr
    return True, "ok"


def extract_code(text: str) -> str | None:
    """Pull the last fenced Python block out of a completion.

    Args:
        text: Raw model output.

    Returns:
        The code, or None when no fenced block is present.
    """
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return blocks[-1].strip() if blocks else None


PROGRAM_PROMPT = """Write a short Python program that computes the answer to this problem.

PROBLEM: {problem}

Rules:
- Solve it by direct computation or exhaustive search over a justified finite range.
- Do not guess, and do not hard-code an answer you have not computed.
- Print ONLY the final answer on the last line, with no label and no explanation.
- Use the standard library only. No network, no file access.

Reply with one ```python fenced block and nothing else.
"""


class ExecutableEnumeration:
    """Writes a program from the statement, runs it sandboxed, and reports what it printed.

    The primary backend, because the failure that motivated this module -- an answer to a
    different question than the one asked -- is exactly what a brute-force search over a
    justified range catches.

    Args:
        client: Anything with `generate(prompt, max_new_tokens, seed) -> (text, truncated)`.
        max_new_tokens: Generation cap for the program.
        timeout: Sandbox wall-clock limit.
        attempts: How many programs to try before giving up.
    """

    name = "executable_enumeration"

    def __init__(self, client, max_new_tokens: int = 4096, timeout: float = 10.0,
                 attempts: int = 2) -> None:
        self.client = client
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.attempts = attempts

    def solve(self, problem: str) -> tuple[str | None, str, str]:
        """Generate and run a program for `problem`, never seeing any asserted answer."""
        last_detail = "no attempt"
        last_code = ""
        for i in range(self.attempts):
            text, truncated = self.client.generate(
                PROGRAM_PROMPT.format(problem=problem), self.max_new_tokens, 991 + i)
            if truncated:
                last_detail = "program generation truncated"
                continue
            code = extract_code(text)
            if not code:
                last_detail = "no fenced code block"
                continue
            last_code = code
            ok, why = program_is_safe(code)
            if not ok:
                last_detail = "refused: %s" % why
                continue
            out, detail = run_program(code, timeout=self.timeout)
            if out is None:
                last_detail = detail
                continue
            return out, "ok", code
        return None, last_detail, last_code


SOLUTION_PROMPT = """Solve this problem. Put your final answer in \\boxed{{}}.

PROBLEM: {problem}
"""


class SolverConsensus:
    """Answers by independent majority vote among several solution attempts.

    A second, weaker source for problems no program settles. It is a consensus, not a proof,
    so it only reports an answer when the majority is decisive; otherwise it returns None and
    the task is UNVERIFIABLE. Note the shared-mode failure it cannot escape: if the model is
    confidently wrong in the same way every time, consensus agrees with itself. That is why
    it ranks below execution and why its threshold is high.

    Args:
        client: A generation client.
        k: Independent attempts.
        threshold: Share of resolved attempts the modal answer must reach.
        max_new_tokens: Generation cap.
    """

    name = "solver_consensus"

    def __init__(self, client, k: int = 5, threshold: float = 0.8,
                 max_new_tokens: int = 8192) -> None:
        self.client = client
        self.k = k
        self.threshold = threshold
        self.max_new_tokens = max_new_tokens

    def solve(self, problem: str) -> tuple[str | None, str, str]:
        """Sample k solutions and return the modal boxed answer if it is decisive."""
        answers: list[str] = []
        for i in range(self.k):
            text, truncated = self.client.generate(
                SOLUTION_PROMPT.format(problem=problem), self.max_new_tokens, 5000 + i)
            if truncated:
                continue
            box = extract_boxed(text)
            if box:
                answers.append(box.strip())
        if not answers:
            return None, "no resolved attempts", ""
        top, n = Counter(answers).most_common(1)[0]
        share = n / len(answers)
        if share < self.threshold:
            return None, "no consensus (%d/%d for %r)" % (n, len(answers), top), ""
        return top, "consensus %d/%d" % (n, len(answers)), ""


class LeanBackend:
    """Registered placeholder for a proof-assistant backend.

    Present so the interface is the one a Lean backend would use, and so choosing it is a
    deliberate act rather than a silent fallback. It returns None, which becomes
    UNVERIFIABLE -- never a guess.
    """

    name = "lean"

    def solve(self, problem: str) -> tuple[str | None, str, str]:
        """Always decline, with a reason."""
        return None, "lean backend not implemented; registered for interface parity", ""


#: Backend registry. Adding a strategy means adding a factory here, not editing callers.
REGISTRY: dict[str, Callable[..., AnswerSource]] = {
    ExecutableEnumeration.name: ExecutableEnumeration,
    SolverConsensus.name: SolverConsensus,
    LeanBackend.name: lambda *a, **k: LeanBackend(),
}


#: Substrings that mark a decline as MECHANICAL -- a failure to obtain or read an answer,
#: rather than a failure to settle the problem. Kept as data so a new failure string has to
#: be classified deliberately instead of silently inheriting SUBSTANTIVE.
MECHANICAL_MARKERS = (
    "truncat", "unparseable", "no fenced code block", "no code block", "timeout",
    "printed nothing", "exit ", "exec failed", "not json", "no usable artifact",
    "did not run", "no attempt", "refused:", "generation truncated", "inconclusive",
    "no sources", "empty restatement",
)


def classify_abstain(detail: str) -> "Abstain":
    """Decide whether a decline was mechanical or substantive.

    Args:
        detail: The verifier's detail string.

    Returns:
        `Abstain.MECHANICAL` when the text matches a known plumbing failure, else
        `Abstain.SUBSTANTIVE`.
    """
    low = (detail or "").lower()
    return (Abstain.MECHANICAL if any(m in low for m in MECHANICAL_MARKERS)
            else Abstain.SUBSTANTIVE)


def unverifiable(backend: str, computed, asserted, detail: str,
                 artefact: str = "") -> VerificationResult:
    """Build an UNVERIFIABLE result with its abstention kind derived from the detail.

    Every decline goes through here so none can omit the classification. Constructing the
    dataclass directly still works and still raises on a missing kind; this is the convenient
    path, not the only enforced one.

    Args:
        backend: Backend name.
        computed: Whatever answer was obtained, if any.
        asserted: The key under test.
        detail: Human-readable reason, also used to classify.
        artefact: Optional code or evidence.

    Returns:
        A `VerificationResult` with `abstain` set.
    """
    return VerificationResult(Verdict.UNVERIFIABLE, backend, computed, asserted, detail,
                              artefact, abstain=classify_abstain(detail))


def verify_answer(problem: str, asserted: str, source: AnswerSource) -> VerificationResult:
    """Compare an independently computed answer against the asserted key.

    The source is asked for an answer to `problem` alone; `asserted` is used only here, after
    the source has committed. Comparison reuses `answers_match`, the same normalisation the
    live grader uses, so a task is not verified under one convention and graded under another.

    Args:
        problem: The problem statement.
        asserted: The key the proposer claimed.
        source: An `AnswerSource`.

    Returns:
        A `VerificationResult`.
    """
    computed, detail, artefact = source.solve(problem)
    if computed is None:
        return VerificationResult(Verdict.UNVERIFIABLE, source.name, None, asserted,
                                  detail, artefact, abstain=classify_abstain(detail))
    verdict = Verdict.VERIFIED if answers_match(computed, asserted) else Verdict.REFUTED
    return VerificationResult(verdict, source.name, computed, asserted, detail, artefact)


def verify_with_cascade(problem: str, asserted: str,
                        sources: list[AnswerSource]) -> VerificationResult:
    """Try sources in order, stopping at the first that reaches a decision.

    A REFUTED or VERIFIED verdict ends the cascade; UNVERIFIABLE falls through to the next
    source. The order encodes assurance: execution first, consensus second.

    Args:
        problem: The problem statement.
        asserted: The asserted key.
        sources: Backends in descending order of assurance.

    Returns:
        The first decisive result, or the last UNVERIFIABLE one.
    """
    last = VerificationResult(Verdict.UNVERIFIABLE, "none", None, asserted, "no sources",
                              abstain=Abstain.MECHANICAL)
    for src in sources:
        last = verify_answer(problem, asserted, src)
        if last.verdict is not Verdict.UNVERIFIABLE:
            return last
    return last
