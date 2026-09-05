"""The answer comparator must not manufacture refutations from formatting.

Every case here is a REAL pair from the run that refuted 49 of 675 published OlympiadBench
keys. Almost none was a mathematical disagreement: they were LaTeX wrappers, unordered sets,
a named unknown, and a float that a program printed with rounding error. A comparator that
calls those "different" produces a false-refutation floor that makes every downstream claim
about key quality uninterpretable.

The negative cases matter as much: a comparator that calls everything equal would pass every
positive test here and destroy the verifier, so genuinely different answers must still come
back different.
"""

from __future__ import annotations

from ornith_repro.symbolic import (
    has_inexact_float,
    split_top_level,
    strip_latex,
    symbolic_compare,
)

# Real (published key, verifier output) pairs that were wrongly refuted.
WRONGLY_REFUTED = [
    (r"$\frac{19}{34}$", "19/34"),
    (r"$-\frac{30}{17}$", "-30/17"),
    (r"$8$,$4$", "4, 8"),
    ("-6,-8,-10", "-10, -8, -6"),
    (r"$a=2$, $a=-6-4 \sqrt{2}$", "2, -6-4*sqrt(2)"),
    (r"$100,\frac{1}{100}$", "0.01, 100"),
    (r"$4^{\circ}$", "4"),
    ("1", "r = 1"),
    ("(1,7,103, 105), (3, 5, 101, 107)", "[(1, 7, 103, 105), (3, 5, 101, 107)]"),
    (r"$(3,2),(-3,2),(3,-2),(-3,-2)$", "[(-3, -2), (-3, 2), (3, -2), (3, 2)]"),
]

GENUINELY_DIFFERENT = [
    ("12", "6"),
    ("1,2,3", "1,2,4"),
    ("19/34", "19/35"),
    ("-6,-8,-10", "-6,-8,-11"),
    (r"$\frac{1}{2}$", "2"),
]


def test_formatting_differences_are_not_refutations():
    """Each of these was a real false refutation of a professionally curated key."""
    bad = [(a, b, symbolic_compare(a, b)) for a, b in WRONGLY_REFUTED
           if symbolic_compare(a, b)[0] != "equal"]
    assert not bad, "still not recognised as equal: %s" % bad


def test_genuinely_different_answers_are_still_different():
    """A comparator that equates everything would pass the test above and be useless."""
    bad = [(a, b, symbolic_compare(a, b)) for a, b in GENUINELY_DIFFERENT
           if symbolic_compare(a, b)[0] != "different"]
    assert not bad, "failed to distinguish: %s" % bad


def test_float_rounding_is_indeterminate_not_a_refutation():
    """`8.000000000000007` is the verifier's own arithmetic, not a claim about 8.

    Forcing equality would reintroduce float equality; refuting would blame the model for the
    verifier's rounding. Both are wrong, so the comparison declines.
    """
    state, why = symbolic_compare(r"$8 \%$", "8.000000000000007")
    assert state == "indeterminate", (state, why)
    assert has_inexact_float("8.000000000000007")
    assert not has_inexact_float("0.5")
    assert not has_inexact_float("19/34")


def test_ordered_pairs_are_not_split_into_numbers():
    """'(1,2),(3,4)' is two items. Splitting it into four turned correct sets into mismatches."""
    assert split_top_level("(1,2),(3,4)") == ["(1,2)", "(3,4)"]
    assert split_top_level("1, 2, 3") == ["1", "2", "3"]


def test_order_matters_inside_a_tuple_but_not_between_items():
    """A set of roots is unordered; the coordinates of one point are not."""
    assert symbolic_compare("8, 4", "4, 8")[0] == "equal"
    assert symbolic_compare("(1,2)", "(2,1)")[0] != "equal"


def test_latex_stripping_keeps_the_value():
    """Normalisation must remove decoration without altering the mathematics."""
    assert strip_latex(r"$\frac{1}{2}$") == "((1)/(2))"
    assert strip_latex(r"$4^{\circ}$") == "4"
    assert strip_latex(r"x \in [1/2, 8]") == "[1/2, 8]"
