"""Compare two mathematical answers by meaning rather than by spelling.

WHY THIS EXISTS, and it is not the reason a computer algebra system was originally proposed.
The verifier refuted 49 of 675 PUBLISHED OlympiadBench keys, a 22.2% rate among decided tasks
that looked like a false-refutation floor. Reading them showed almost none was a mathematical
disagreement:

    key '$\\frac{19}{34}$'          computed '19/34'
    key '$8$,$4$'                   computed '4, 8'
    key '-6,-8,-10'                 computed '-10, -8, -6'
    key '$[\\frac{1}{2}, 8]$'        computed 'x in [1/2, 8]'
    key '$8 \\%$'                    computed '8.000000000000007%'
    key '$a=2$, $a=-6-4\\sqrt{2}$'   computed '2, -6-4*sqrt(2)'

Every one of those is correct and was marked wrong by string comparison. So the largest gain
from symbolic computation here is not solving more problems: it is COMPARING answers correctly.
An answer is a LaTeX-wrapped rational, an unordered set of roots, an interval, a tuple, or a
value carrying a unit, and a comparator that does not know that manufactures refutations.

FLOATING POINT IS NEVER USED FOR EQUALITY. `8.000000000000007%` is exactly the artefact that
rule exists to prevent; comparison is on exact rationals, and a float is converted to a
rational before it is compared.

SOUNDNESS DIRECTION. Every rule here can only turn a REFUTED into a VERIFIED, never the
reverse, because it can only find two things equal that string comparison called different.
That is the safe direction for a verifier whose refutations were the suspect half, but it does
mean a bug here shows up as a false VERIFIED, so each rule is narrow and tested.
"""

from __future__ import annotations

import re
import sys
from fractions import Fraction

#: Optional wheels live here so the serving venv is never disturbed. Appended AFTER the
#: standard path where possible; prepended only when a newer antlr4 is required by
#: sympy's LaTeX parser, which is the one case a shadowing import is unavoidable.
import os as _os
_EXTRA = _os.environ.get("ORNITH_PYLIBS", "/mnt/localssd/gate/pylibs")
if _os.path.isdir(_EXTRA) and _EXTRA not in sys.path:
    sys.path.insert(0, _EXTRA)

try:  # pragma: no cover - environment dependent
    import sympy
    from sympy import Rational, simplify, sympify
    HAVE_SYMPY = True
except Exception:  # noqa: BLE001  # pragma: no cover
    HAVE_SYMPY = False

try:  # pragma: no cover
    from sympy.parsing.latex import parse_latex
    HAVE_LATEX = True
except Exception:  # noqa: BLE001  # pragma: no cover
    HAVE_LATEX = False

try:  # pragma: no cover
    import flint
    HAVE_FLINT = True
except Exception:  # noqa: BLE001  # pragma: no cover
    HAVE_FLINT = False

try:  # pragma: no cover
    import mpmath
    HAVE_MPMATH = True
except Exception:  # noqa: BLE001  # pragma: no cover
    HAVE_MPMATH = False

#: LaTeX wrappers that carry no mathematical content.
_STRIP = (
    (r"\left", ""), (r"\right", ""), (r"\!", ""), (r"\,", ""), (r"\;", ""),
    (r"\quad", " "), (r"\qquad", " "), (r"^\circ", ""), (r"^{\circ}", ""),
    (r"\%", ""), (r"\$", ""), (r"\cdot", "*"), (r"\times", "*"),
    (r"\infty", "oo"), (r"\pi", "pi"), (r"\ldots", ""), (r"\dots", ""),
)


def strip_latex(s: str) -> str:
    """Remove LaTeX decoration that does not change an answer's value.

    Args:
        s: A raw answer string.

    Returns:
        The same answer with wrappers, units and dollar signs removed.
    """
    if s is None:
        return ""
    out = str(s).strip()
    out = re.sub(r"\\text\s*\{[^}]*\}", " ", out)
    out = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", out)
    out = out.replace("$", " ")
    for a, b in _STRIP:
        out = out.replace(a, b)
    # \frac{a}{b} and \dfrac{a}{b} -> (a)/(b), innermost first.
    for _ in range(6):
        new = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", out)
        if new == out:
            break
        out = new
    out = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", out)
    out = re.sub(r"\\sqrt\s*(\w)", r"sqrt(\1)", out)
    out = out.replace("^", "**").replace("\\", " ")
    # A leading "x =" or "r =" names the unknown and is not part of the value.
    out = re.sub(r"^\s*[A-Za-z]\w*\s*(?:=|\\?in|∈)\s*", "", out.strip())
    out = out.replace("%", "").replace("°", "")
    return " ".join(out.split())


def split_top_level(s: str) -> list[str]:
    """Split on commas or semicolons that are not inside brackets.

    "(1,2),(3,4)" is two items, not four. Getting this wrong is what turned a correct set of
    ordered pairs into a mismatch.

    Args:
        s: A normalised answer string.

    Returns:
        The top-level items.
    """
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch in ",;" and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur).strip())
    return [x for x in out if x]


def to_exact(text: str):
    """Parse one item into an exact sympy object, never a float.

    Args:
        text: One normalised item.

    Returns:
        A sympy object, or None when it cannot be parsed.
    """
    if not HAVE_SYMPY or not text:
        return None
    t = text.strip().strip("$ ").rstrip(".")
    t = re.sub(r"^\s*[A-Za-z]\w*\s*=\s*", "", t)
    if not t:
        return None
    # LaTeX first: the \frac{19}{34} class is a PARSING problem, not a mathematical one, and
    # a real grammar handles it far better than the regex normaliser below.
    if HAVE_LATEX and "\\" in t:
        try:
            return parse_latex(t)
        except Exception:  # noqa: BLE001
            pass
    try:
        expr = sympify(t, rational=True)
    except Exception:  # noqa: BLE001
        try:
            expr = sympify(t.replace(" ", "*"), rational=True)
        except Exception:  # noqa: BLE001
            return None
    try:
        # A float compared by equality is the bug this rule exists to prevent.
        return sympy.nsimplify(expr, rational=True) if expr.free_symbols == set() \
            and expr.is_Float else expr
    except Exception:  # noqa: BLE001
        return expr


def _as_items(s: str):
    """Normalise a raw answer to a list of exact items, or None if unparseable."""
    norm = strip_latex(s)
    if not norm:
        return None
    inner = norm.strip()
    if len(inner) > 1 and inner[0] in "[{(" and inner[-1] in "]})":
        stripped = inner[1:-1]
        if split_top_level(stripped) and inner[0] in "[{":
            inner = stripped
    parts = split_top_level(inner)
    items = [to_exact(p) for p in parts]
    return None if any(i is None for i in items) else items


def has_inexact_float(text: str) -> bool:
    """Whether a normalised answer carries a decimal that is not an exact value.

    `8.000000000000007` is a program's float rounding, not a mathematical claim about 8. Such
    an answer cannot be compared exactly, and forcing it equal would reintroduce the float
    equality this module exists to avoid, while refuting it would blame the model for the
    verifier's own arithmetic. Both are wrong, so the comparison is declared indeterminate.

    Args:
        text: A raw answer string.

    Returns:
        True when a decimal literal with more than a few fractional digits is present.
    """
    for m in re.finditer(r"\d+\.(\d+)", strip_latex(text or "")):
        frac = m.group(1).rstrip("0")
        if len(frac) >= 6:
            return True
    return False


def contains_float(expr) -> bool:
    """Whether a parsed expression carries a floating-point atom.

    THE STRUCTURAL RULE: no float may reach a comparison. A float can only enter by a
    `float()` or an unprecise `evalf()`, and once it exists no rule can undo the damage
    honestly -- `8.000000000000007` is not a claim about 8, it is a program's rounding. So its
    presence is detected and the comparison declines rather than guessing.

    Args:
        expr: A sympy expression, or None.

    Returns:
        True when any atom is a Float.
    """
    if expr is None or not HAVE_SYMPY:
        return False
    try:
        return bool(expr.atoms(sympy.Float))
    except Exception:  # noqa: BLE001
        return False


def exactly_different(x, y) -> bool:
    """Prove two values differ by exact rational arithmetic, with no numerics at all.

    When both sides are rationals or integers, inequality IS a proof: exact arithmetic is
    complete on that domain, so nothing is being approximated and no enclosure is needed. This
    is the refutation route that works without any optional wheel, which matters because a
    comparator that can only refute when `python-flint` happens to be installed would silently
    stop refuting on a machine that lacks it.

    Args:
        x: A sympy expression.
        y: Likewise.

    Returns:
        True when both are exact rationals and they differ.
    """
    if not HAVE_SYMPY:
        return False
    try:
        if x.free_symbols or y.free_symbols:
            return False
        if x.is_Rational and y.is_Rational:
            return bool(sympy.Rational(x) != sympy.Rational(y))
        return False
    except Exception:  # noqa: BLE001
        return False


def proves_different(x, y) -> bool:
    """Whether the difference of two values can be PROVED, by exact arithmetic or by Arb.

    Args:
        x: A sympy expression.
        y: Likewise.

    Returns:
        True when either route establishes the values differ.
    """
    return exactly_different(x, y) or arb_proves_different(x, y)


def arb_proves_different(x, y) -> bool:
    """Prove two EXACT quantities differ using Arb ball arithmetic.

    Ball arithmetic carries a rigorous error bound with every value, so two enclosures that do
    not overlap PROVE the quantities differ. That is the only numeric route that can soundly
    support a refutation; ordinary floating point and even 50-digit mpmath can only ever be
    evidence. It is applied to exact inputs only -- an inexact decimal is refused earlier,
    because Arb would faithfully report `8` and `8.000000000000007` as disjoint when the
    difference is the verifier's own rounding rather than a mathematical fact.

    Args:
        x: A sympy expression with no free symbols.
        y: Likewise.

    Returns:
        True only when the enclosures are provably disjoint.
    """
    if not (HAVE_FLINT and HAVE_SYMPY):
        return False
    try:
        if x.free_symbols or y.free_symbols:
            return False
        with flint.ctx.workprec(256):
            bx = flint.arb(str(sympy.nsimplify(x, rational=True).evalf(40)))
            by = flint.arb(str(sympy.nsimplify(y, rational=True).evalf(40)))
            return not bx.overlaps(by)
    except Exception:  # noqa: BLE001
        return False


def mpmath_agrees(x, y, dps: int = 50) -> bool:
    """High-precision agreement as EVIDENCE for equality, never for difference.

    Fifty digits of agreement is strong evidence two closed forms are equal and is not a
    proof, so this may support a VERIFIED verdict and must never create a REFUTED one.

    Args:
        x: A sympy expression with no free symbols.
        y: Likewise.
        dps: Decimal digits of working precision.

    Returns:
        True when the two agree to nearly `dps` digits.
    """
    if not (HAVE_MPMATH and HAVE_SYMPY):
        return False
    try:
        if x.free_symbols or y.free_symbols:
            return False
        mpmath.mp.dps = dps
        dx = mpmath.mpf(str(x.evalf(dps)))
        dy = mpmath.mpf(str(y.evalf(dps)))
        scale = max(abs(dx), abs(dy), mpmath.mpf(1))
        return abs(dx - dy) < scale * mpmath.mpf(10) ** (-(dps - 5))
    except Exception:  # noqa: BLE001
        return False


def symbolic_compare(a: str, b: str) -> tuple[str, str]:
    """Three-valued comparison: "equal", "different", or "indeterminate".

    A verifier that must not produce wrong REFUTED verdicts cannot use a two-valued
    comparator, because "I cannot compare these" and "these differ" are different claims and
    only the second justifies a refutation. That distinction is what turned 49 formatting
    mismatches into refutations of professionally curated keys.

    THE ASYMMETRY IS DELIBERATE. Equality may rest on evidence: exact simplification, or
    failing that fifty digits of mpmath agreement. **Difference must be PROVED**, by disjoint
    Arb enclosures or by an exact structural mismatch. `simplify(a - b)` failing to reach zero
    is not a proof that the two differ -- simplification is incomplete -- so it alone yields
    indeterminate.

    Args:
        a: First answer.
        b: Second answer.

    Returns:
        `(state, reason)`.
    """
    if a is None or b is None:
        return "indeterminate", "missing"
    if strip_latex(a) == strip_latex(b):
        return "equal", "identical after latex normalisation"
    if not HAVE_SYMPY:
        return "indeterminate", "sympy unavailable"
    # No float may reach a comparison, at the string level or after parsing.
    if has_inexact_float(a) or has_inexact_float(b):
        return "indeterminate", "an answer carries an inexact decimal; refusing to compare"

    ia, ib = _as_items(a), _as_items(b)
    if ia is None or ib is None:
        return "indeterminate", "unparseable"
    if any(contains_float(x) for x in ia + ib):
        return "indeterminate", "a parsed value contains a Float atom"

    if len(ia) != len(ib):
        return "different", "different number of items (%d vs %d)" % (len(ia), len(ib))

    if len(ia) == 1:
        x, y = ia[0], ib[0]
        try:
            if simplify(x - y) == 0:
                return "equal", "symbolically equal"
        except Exception:  # noqa: BLE001
            pass
        if mpmath_agrees(x, y):
            return "equal", "agree to 50 digits (evidence, not proof)"
        if exactly_different(x, y):
            return "different", "exact rationals, provably unequal"
        if arb_proves_different(x, y):
            return "different", "disjoint Arb enclosures prove the values differ"
        return "indeterminate", "simplification did not close and no disjoint enclosure"

    # Unordered multiset match: each item on one side must pair with a distinct item.
    remaining = list(ib)
    for x in ia:
        hit = None
        for j, y in enumerate(remaining):
            try:
                if simplify(x - y) == 0 or mpmath_agrees(x, y):
                    hit = j
                    break
            except Exception:  # noqa: BLE001
                if str(x) == str(y):
                    hit = j
                    break
        if hit is None:
            # Only a PROVED mismatch against every candidate justifies "different".
            proved = all(proves_different(x, y) for y in remaining) if remaining else False
            if proved:
                return "different", "item %s provably matches nothing on the other side" % x
            return "indeterminate", "item %s unmatched but not provably distinct" % x
        remaining.pop(hit)
    return "equal", "equal as an unordered collection of %d items" % len(ia)


def symbolic_equal(a: str, b: str) -> tuple[bool, str]:
    """Decide whether two answers denote the same thing.

    Handles LaTeX wrappers, a leading unknown name, exact rationals against decimals, and
    unordered collections: a set of roots is the same set however it is ordered, and a list of
    ordered pairs must match as a set of pairs rather than element by element.

    Args:
        a: First answer.
        b: Second answer.

    Returns:
        `(equal, reason)`. The reason names the rule that decided it, so a wrong match can be
        traced to a rule rather than to "the comparator".
    """
    if a is None or b is None:
        return False, "missing"
    if strip_latex(a) == strip_latex(b):
        return True, "identical after latex normalisation"
    if not HAVE_SYMPY:
        return False, "sympy unavailable"
    ia, ib = _as_items(a), _as_items(b)
    if ia is None or ib is None:
        return False, "unparseable"
    if len(ia) != len(ib):
        return False, "different number of items (%d vs %d)" % (len(ia), len(ib))
    if len(ia) == 1:
        try:
            return (bool(simplify(ia[0] - ib[0]) == 0), "symbolically equal")
        except Exception:  # noqa: BLE001
            try:
                return (bool(ia[0].equals(ib[0])), "equals()")
            except Exception:  # noqa: BLE001
                return False, "could not compare"
    # Unordered multiset match: every item on one side pairs with a distinct item on the other.
    remaining = list(ib)
    for x in ia:
        hit = None
        for j, y in enumerate(remaining):
            try:
                if simplify(x - y) == 0:
                    hit = j
                    break
            except Exception:  # noqa: BLE001
                if str(x) == str(y):
                    hit = j
                    break
        if hit is None:
            return False, "item %s unmatched" % x
        remaining.pop(hit)
    return True, "equal as an unordered collection of %d items" % len(ia)
