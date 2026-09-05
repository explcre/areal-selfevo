"""Route a problem to the backend that can settle it, or refuse to route it at all.

WHY ROUTING MATTERS MORE THAN A PROOF ASSISTANT. Of 592 abstentions on the generated set,
212 said the same thing: "I found an answer but cannot prove I searched far enough"
(`bound_falsified` 160, `bound_rerun_failed` 52). That is a limit of SEARCH, not of
mathematics, and a computer algebra system dissolves it by construction -- solving an equation
returns all roots, summing a series in closed form needs no search width, evaluating a limit
exactly needs no stability check. Add the 189 lost to a truncation bug and roughly two thirds
of the abstentions are addressable with no proof assistant at all.

THE CLASSIFIER MUST BE ABLE TO REFUSE. A misrouted problem produces a confident wrong verdict,
which is the single outcome this pipeline cannot tolerate, so `UNCLASSIFIABLE` is a first-class
answer and the default when the labels do not agree. Refusing costs coverage; guessing costs
soundness, and coverage is the cheaper thing to lose.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum


class ProblemClass(str, Enum):
    """What kind of computation could settle a problem's answer.

    The classes are named for the BACKEND that claims them, not for a mathematical taxonomy,
    because their only job is routing.
    """

    #: A finite integer search whose range can be justified: divisibility, digit conditions,
    #: bounded Diophantine questions. Enumeration's home ground.
    BOUNDED_SEARCH = "bounded_search"
    #: Equations, polynomials, algebraic identities, series, limits, derivatives, integrals.
    #: A CAS returns every root or an exact closed form, so no bound needs justifying.
    SYMBOLIC = "symbolic"
    #: Counting and probability over a finite structure, settled by exact enumeration or
    #: exact rational arithmetic.
    COMBINATORIAL = "combinatorial"
    #: Geometry reducible to coordinates or symbolic algebra.
    GEOMETRY = "geometry"
    #: Asks for a proof, a construction, or a characterisation rather than a value. No
    #: executable backend claims these; they are the honest proof-assistant candidates.
    PROOF_OR_CONSTRUCTION = "proof_or_construction"
    #: The classifier declined. Routed nowhere.
    UNCLASSIFIABLE = "unclassifiable"


#: Which backend claims which class. A class absent here has no executable route.
CLASS_BACKENDS: dict[ProblemClass, str] = {
    ProblemClass.BOUNDED_SEARCH: "executable_enumeration",
    ProblemClass.COMBINATORIAL: "executable_enumeration",
    ProblemClass.SYMBOLIC: "symbolic_cas",
    ProblemClass.GEOMETRY: "symbolic_cas",
}

CLASSIFY_PROMPT = """Classify this mathematics problem by HOW its answer could be checked by a
computer. Do not solve it.

PROBLEM: {problem}

Choose exactly one label:

bounded_search        - the answer is found by searching a finite, justifiable range of
                        integers (divisibility, digit properties, bounded Diophantine)
symbolic              - an equation, polynomial, identity, series, limit, derivative or
                        integral, where exact algebra gives a closed form or all roots
combinatorial         - counting or probability over a finite structure
geometry              - geometry that reduces to coordinates or algebra
proof_or_construction - asks for a proof, a construction, or a characterisation rather than
                        a single specific value
unclassifiable        - none of the above fits, or the problem is unclear

Reply with exactly one word from that list on the first line, and nothing else.
"""


def parse_label(text: str) -> ProblemClass:
    """Read a classifier reply, refusing anything that is not exactly one known label.

    Args:
        text: The raw completion.

    Returns:
        The named class, or `UNCLASSIFIABLE` when the reply is not a clean label.
    """
    if not text:
        return ProblemClass.UNCLASSIFIABLE
    body = text.split("</think>")[-1].strip()
    first = body.splitlines()[0].strip().lower().strip(".,`\"'") if body else ""
    for c in ProblemClass:
        if first == c.value:
            return c
    return ProblemClass.UNCLASSIFIABLE


def classify(client, problem: str, samples: int = 3, max_new_tokens: int = 2048,
             seed: int = 4242) -> tuple[ProblemClass, str]:
    """Classify a problem, requiring the samples to agree.

    Unanimity is the same rule the faithfulness checker uses and for the same reason: a
    disagreement means the model is unsure, and an unsure routing decision is worse than no
    routing decision. The majority label is reported alongside so the cost of the strict rule
    can be measured rather than assumed.

    Args:
        client: Generation client.
        problem: The problem statement.
        samples: Replies that must agree.
        max_new_tokens: Generation cap. Sized for reasoning tokens, not for the one-word
            answer -- a budget sized for the visible output is how four separate truncation
            bugs entered this project.
        seed: Base seed.

    Returns:
        `(class, detail)` where the class is `UNCLASSIFIABLE` unless every sample agreed.
    """
    labels: list[ProblemClass] = []
    for i in range(max(1, samples)):
        text, truncated = client.generate(
            CLASSIFY_PROMPT.format(problem=problem), max_new_tokens, seed + i)
        if truncated:
            return ProblemClass.UNCLASSIFIABLE, "classification truncated on sample %d" % (i + 1)
        labels.append(parse_label(text))
    top, n = Counter(labels).most_common(1)[0]
    if n == len(labels):
        return top, "unanimous across %d" % len(labels)
    return (ProblemClass.UNCLASSIFIABLE,
            "no consensus: %s (majority %s %d/%d)"
            % ([x.value for x in labels], top.value, n, len(labels)))


def backend_for(cls: ProblemClass) -> str | None:
    """The backend claiming a class, or None when nothing executable claims it.

    Args:
        cls: A problem class.

    Returns:
        Backend name, or None.
    """
    return CLASS_BACKENDS.get(cls)
