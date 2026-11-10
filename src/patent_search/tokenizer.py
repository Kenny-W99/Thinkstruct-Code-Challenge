"""One tokenizer shared by BM25, the keyword filter and highlighting.

Using the same rules everywhere means a keyword filter of "tires" matches
"tire", and the highlighted words are exactly the words that were scored.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z]+")

# Common English words plus patent boilerplate that appears in almost every claim.
STOPWORDS = frozenset(
    """
    a an and are as at be been being by for from has have in into is it its of on or
    that the their them then there these this those to was were which with
    wherein whereby said such each least plurality claim claims
    comprising comprises comprise including includes include thereof therein
    """.split()
)


def normalize(word: str) -> str:
    """Lowercase + fold simple plurals ("tires" -> "tire", "batteries" -> "battery")."""
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _keep(word_lower: str) -> bool:
    return len(word_lower) >= 2 and word_lower not in STOPWORDS


def tokenize(text: str) -> list[str]:
    return [normalize(m.group()) for m in _WORD.finditer(text) if _keep(m.group().lower())]


def highlight_spans(text: str, terms: set[str]) -> list[tuple[int, int]]:
    """Character spans of words in `text` whose normalized form is in `terms`."""
    if not terms:
        return []
    return [
        (m.start(), m.end())
        for m in _WORD.finditer(text)
        if _keep(m.group().lower()) and normalize(m.group()) in terms
    ]
