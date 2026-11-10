"""Text cleaning and claim parsing. Pure functions, no I/O."""

from __future__ import annotations

import re
from dataclasses import dataclass

_WS = re.compile(r"\s+")
_GLUED_FIG = re.compile(r"(?<=[a-z])(?=FIGS?\.)")  # "inFIG.1"  -> "in FIG.1"
_FIG_THEN_WORD = re.compile(r"(FIGS?\.\s?\d+[A-Z]?)(?=[a-z])")  # "FIG.1is" -> "FIG.1 is"


def is_missing(value: object) -> bool:
    """True for None, empty strings/lists and the dataset's "N/A" placeholder."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().upper() in ("", "N/A")
    if isinstance(value, (list, tuple)):
        return all(is_missing(v) for v in value)
    return False


def clean_text(text: str) -> str:
    """Light cleanup: un-glue "inFIG.1is" and collapse whitespace."""
    if "FIG" in text:  # cheap check first: most text has no figure reference
        text = _GLUED_FIG.sub(" ", text)
        text = _FIG_THEN_WORD.sub(r"\1 ", text)
    return _WS.sub(" ", text).strip()


@dataclass(frozen=True)
class Claim:
    number: int
    text: str  # claim body without the leading "7 ."
    depends_on: tuple[int, ...] = ()

    @property
    def is_independent(self) -> bool:
        return not self.depends_on


# "7 . The tire..." / "1. A method..."  (but not "2.5 mm thick...")
_CLAIM_NUMBER = re.compile(r"^\s*(\d+)\s*\.(?!\d)\s*")
# "1 - 5 . (canceled)" / "3. (Cancelled)"
_CANCELED_RANGE = re.compile(
    r"^\s*(\d+)(?:\s*(?:-|–|to|through)\s*(\d+))?\s*\.?\s*\((?:canceled|cancelled)\)\s*\.?\s*$",
    re.IGNORECASE,
)
_CANCELED_ANY = re.compile(r"\((?:canceled|cancelled)\)", re.IGNORECASE)
_DEPENDS = re.compile(r"\bclaims?\s+(\d+)(?:\s*(?:-|–|to|or|and|through)\s*(\d+))?", re.IGNORECASE)


def parse_claims(raw_claims: list[object]) -> list[Claim]:
    """Number every claim, drop canceled ones, and detect dependencies.

    The small dataset dropped the number (and opening words) of many
    independent claims, so a missing number is inferred as previous + 1.
    """
    claims: list[Claim] = []
    last = 0
    for raw in raw_claims:
        if not isinstance(raw, str) or is_missing(raw):
            continue
        text = clean_text(raw)
        canceled = _CANCELED_RANGE.match(text)
        if canceled:
            last = int(canceled.group(2) or canceled.group(1))
            continue
        if len(text) < 40 and _CANCELED_ANY.search(text):
            last += 1
            continue
        numbered = _CLAIM_NUMBER.match(text)
        if numbered:
            number, body = int(numbered.group(1)), text[numbered.end():]
        else:
            number, body = last + 1, text
        if not body:
            continue
        refs = {int(n) for pair in _DEPENDS.findall(body) for n in pair if n}
        claims.append(Claim(number, body, tuple(sorted(r for r in refs if r < number))))
        last = number
    return claims
