"""Parse and match CPC classification codes.

The two datasets write codes differently:

    small data:  "B60B2700FI"  -> subclass B60B, digits "2700", no "/" separator.
                                  "FI" is a USPTO marker ("first, inventive"),
                                  not part of the code.
    large data:  "A61B5/097"   -> subclass A61B, group 5, subgroup 097
                 "A01G009/02"  -> group written with leading zeros
    both:        "N/A"         -> missing

Without the "/", "B60B1110" could be 11/10 or 1/110, so for small-data codes
we only know the subclass for sure. Deeper matching is best effort there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CODE_RE = re.compile(r"^([A-HY]\d{2}[A-Z])\s*(\d+)\s*(?:/\s*(\d+))?\s*(?:[FL][IA])?$")
# A query may be broader than a subclass: "B", "B60", "B60B", "B60B27", "B60B 27/00"
_QUERY_RE = re.compile(r"^([A-HY](?:\d{2}[A-Z]?)?)(\d*)(/?)(\d*)$")


@dataclass(frozen=True)
class CPCCode:
    raw: str
    subclass: str  # e.g. "B60B"
    group: str | None  # e.g. "27"; None when the source had no "/" separator
    subgroup: str | None  # e.g. "00"
    digits: str  # everything after the subclass, digits only

    @property
    def compact(self) -> str:
        return self.subclass + self.digits

    @property
    def display(self) -> str:
        if self.group is not None:
            return f"{self.subclass} {self.group}/{self.subgroup}"
        return f"{self.subclass} {self.digits}"


def parse_cpc(raw: object) -> CPCCode | None:
    """Parse a code from either dataset. Returns None for missing/unparseable codes."""
    if not isinstance(raw, str):
        return None
    text = raw.strip().upper()
    if not text or text == "N/A":
        return None
    match = _CODE_RE.match(text)
    if not match:
        return None
    subclass, first, second = match.groups()
    if second is not None:  # "A61B5/097": the split is known
        group = first.lstrip("0") or "0"
        return CPCCode(raw, subclass, group, second, group + second)
    return CPCCode(raw, subclass, None, None, first)


@dataclass(frozen=True)
class CPCQuery:
    """A user's classification filter, e.g. "B60", "B60B", "B60B 27/00"."""

    prefix: str  # section / class / subclass prefix
    group: str | None = None
    subgroup: str = ""  # prefix of the subgroup ("" = any)
    has_slash: bool = False

    def matches(self, code: CPCCode | None) -> bool:
        if code is None:
            return False
        if not code.subclass.startswith(self.prefix):
            return False
        if self.group is None:
            return True
        if code.group is not None:  # structured code (large data): exact group match
            if code.group != self.group:
                return False
            return code.subgroup.startswith(self.subgroup) if code.subgroup is not None else True
        # unstructured code (small data): best-effort prefix on the digits
        return code.digits.startswith(self.group + self.subgroup)


def parse_cpc_query(text: str) -> CPCQuery:
    """Parse a filter like "b60b", "B60B 27/00" or "B60". Raises ValueError if invalid."""
    cleaned = re.sub(r"\s+", "", text.upper())
    match = _QUERY_RE.match(cleaned)
    if not match or not cleaned:
        raise ValueError(f"Not a valid classification prefix: {text!r}")
    prefix, group, slash, subgroup = match.groups()
    if (group or slash or subgroup) and len(prefix) != 4:
        raise ValueError(f"A group number needs a full subclass first (like B60B): {text!r}")
    if not group:
        if slash or subgroup:
            raise ValueError(f"Missing group number before '/': {text!r}")
        return CPCQuery(prefix=prefix)
    return CPCQuery(
        prefix=prefix,
        group=group.lstrip("0") or "0",
        subgroup=subgroup,
        has_slash=bool(slash),
    )
