"""Text similarity used for BOTH novelty and contamination, so they cannot disagree.

Token Jaccard alone is not enough and that is the whole design point here. A paraphrase of a
held-out problem -- same quantities, reordered clauses, synonyms -- can share few enough tokens
to pass a Jaccard threshold while being the same problem. Character n-grams survive reordering
and small substitutions, so the two are combined and the MAXIMUM is taken: an item is similar
if EITHER measure says so, because a contamination filter that can be defeated by rewording is
worse than none.
"""
from __future__ import annotations

import re
from collections import Counter

_WORD = re.compile(r"[a-z0-9]+")
#: Words too common in mathematics prose to carry identity.
_STOP = frozenset("the a an of is are be to and or in on for with what find compute "
                  "determine let such that if then all each every number numbers value "
                  "values which how many prove show given".split())


def normalise(text: str) -> str:
    """Lowercase, strip LaTeX punctuation noise and collapse whitespace."""
    t = text.lower()
    t = re.sub(r"\\[a-z]+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return " ".join(t.split())


def token_set(text: str) -> set[str]:
    """Content words of a statement."""
    return {w for w in _WORD.findall(normalise(text)) if w not in _STOP}


def token_jaccard(a: str, b: str) -> float:
    """Jaccard over content words; robust to formatting, weak to paraphrase."""
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def char_ngrams(text: str, n: int = 5) -> Counter:
    """Character n-grams of the normalised statement, whitespace collapsed."""
    s = normalise(text).replace(" ", "")
    return Counter(s[i:i + n] for i in range(max(len(s) - n + 1, 0)))


def char_cosine(a: str, b: str, n: int = 5) -> float:
    """Cosine over character n-gram counts; survives reordering and small edits."""
    ca, cb = char_ngrams(a, n), char_ngrams(b, n)
    if not ca or not cb:
        return 0.0
    common = set(ca) & set(cb)
    num = sum(ca[g] * cb[g] for g in common)
    da = sum(v * v for v in ca.values()) ** 0.5
    db = sum(v * v for v in cb.values()) ** 0.5
    return num / (da * db) if da and db else 0.0


def similarity(a: str, b: str) -> float:
    """The similarity the whole layer uses: the larger of the two measures.

    Taking the maximum rather than an average is deliberate. The two measures fail on
    different things -- Jaccard on paraphrase, n-grams on genuinely distinct problems that
    share boilerplate -- and for a REJECTION filter a false accept is far more expensive
    than a false reject, since a contaminated held-out set cannot be repaired after the fact.
    """
    return max(token_jaccard(a, b), char_cosine(a, b))


def most_similar(text: str, corpus) -> tuple[float, int]:
    """Highest similarity of `text` against a corpus, and the index attaining it."""
    best, at = 0.0, -1
    for i, other in enumerate(corpus):
        s = similarity(text, other)
        if s > best:
            best, at = s, i
    return best, at
