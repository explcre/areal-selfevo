"""Grading tests. The boxed extractor is the correctness-critical part: a naive
`\\boxed\{([^}]*)\}` truncates `\\boxed{\\frac{1}{2}}` to `\\frac{1`, which grades as WRONG
and shows up as a plausible lower score rather than an error."""
import sys
sys.path.insert(0, ".")
from math_bench import extract_boxed, grade

def test_simple():
    assert extract_boxed(r"so \boxed{42}") == "42"

def test_nested_braces_not_truncated():
    assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_boxed(r"\boxed{\frac{\sqrt{3}}{2}}") == r"\frac{\sqrt{3}}{2}"

def test_last_boxed_wins():
    assert extract_boxed(r"first \boxed{1} then \boxed{2}") == "2"

def test_missing_and_unbalanced():
    assert extract_boxed("no box here") is None
    assert extract_boxed(r"\boxed{1") is None

def test_grade_exact_and_equivalent():
    assert grade(r"\boxed{42}", "42")
    assert not grade(r"\boxed{41}", "42")
    assert not grade("no answer", "42")

def test_grade_symbolic_equivalence():
    # math_verify should treat these as equal
    assert grade(r"\boxed{\frac{1}{2}}", r"\frac{1}{2}")
    assert grade(r"\boxed{0.5}", r"\frac{1}{2}")

def test_grade_tolerates_formatting():
    assert grade(r"\boxed{ 42 }", "42")
    assert grade(r"\boxed{1,000}", "1000") or grade(r"\boxed{1000}", "1,000")
