"""Metadata filters ("hybrid search" in the challenge PDF).

Filters work at the patent level:
    - classification prefix  (any of several, e.g. B60B or B60C11)
    - keywords that must ALL appear in the title or abstract
    - title contains / title equals
    - publication date range

`MetadataIndex.candidates()` is the PRE-filter: it uses lookup tables
(inverted index, subclass table) to find matching patents BEFORE any scoring,
smallest set first. `MetadataIndex.passes()` checks one patent at a time and
is what a naive POST-filter uses after scoring everything.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

import numpy as np

from .cpc import CPCCode, CPCQuery, parse_cpc_query
from .load import Patent
from .tokenizer import tokenize

_EMPTY = np.empty(0, dtype=np.int32)


@dataclass
class SearchFilters:
    cpc: list[str] = field(default_factory=list)  # any-of, e.g. ["B60B", "B60C 11/03"]
    keywords: list[str] = field(default_factory=list)  # all must appear in title/abstract
    title: str | None = None
    title_exact: bool = False
    date_from: date | None = None
    date_to: date | None = None

    def is_empty(self) -> bool:
        return not (self.cpc or self.keywords or self.title or self.date_from or self.date_to)


@dataclass(frozen=True)
class CompiledFilters:
    """Filters parsed once per query (CPC strings parsed, keywords tokenized)."""

    cpc: tuple[CPCQuery, ...]
    keyword_terms: tuple[str, ...]
    title: str | None
    title_exact: bool
    date_from: int | None  # date.toordinal()
    date_to: int | None


def _norm_title(title: str) -> str:
    return " ".join(title.lower().split())


def compile_filters(f: SearchFilters) -> CompiledFilters:
    """Raises ValueError for a bad classification code or a backwards date range."""
    if f.date_from and f.date_to and f.date_from > f.date_to:
        raise ValueError(f"date_from ({f.date_from}) is after date_to ({f.date_to})")
    terms = tuple(dict.fromkeys(t for kw in f.keywords for t in tokenize(kw)))
    return CompiledFilters(
        cpc=tuple(parse_cpc_query(c) for c in f.cpc if c.strip()),
        keyword_terms=terms,
        title=_norm_title(f.title) if f.title and f.title.strip() else None,
        title_exact=f.title_exact,
        date_from=f.date_from.toordinal() if f.date_from else None,
        date_to=f.date_to.toordinal() if f.date_to else None,
    )


@dataclass
class FilterResult:
    patents: np.ndarray  # sorted patent indices that pass
    excluded_no_cpc: int = 0  # patents with no classification (can never match a CPC filter)


class MetadataIndex:
    def __init__(self, patents: Sequence[Patent]):
        self.n = len(patents)
        self.codes: list[CPCCode | None] = [p.cpc for p in patents]
        self.no_cpc = sum(c is None for c in self.codes)

        by_subclass: dict[str, list[int]] = defaultdict(list)
        postings: dict[str, list[int]] = defaultdict(list)
        for i, p in enumerate(patents):
            if p.cpc is not None:
                by_subclass[p.cpc.subclass].append(i)
            for term in set(tokenize(f"{p.title or ''} {p.abstract or ''}")):
                postings[term].append(i)
        self.by_subclass = {k: np.array(v, dtype=np.int32) for k, v in by_subclass.items()}
        self.postings = {k: np.array(v, dtype=np.int32) for k, v in postings.items()}  # sorted by construction
        self.titles = [_norm_title(p.title or "") for p in patents]
        # "Title contains X" without a Python loop over every title: all titles in
        # one string, searched with str.find (C speed). Titles never contain "\n",
        # so a match can't span two titles.
        self._title_blob = "\n".join(self.titles)
        self._title_starts = np.cumsum([0] + [len(t) + 1 for t in self.titles[:-1]]).astype(np.int64)
        by_title: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(self.titles):
            by_title[t].append(i)
        self._title_exact = {k: np.array(v, dtype=np.int32) for k, v in by_title.items()}

        # Large-data patents only know their year: they cover Jan 1 - Dec 31 and
        # pass a date filter if that range overlaps (we'd rather not lose prior art).
        lo, hi = np.full(self.n, -1, dtype=np.int64), np.full(self.n, -1, dtype=np.int64)
        for i, p in enumerate(patents):
            if p.pub_date:
                lo[i] = hi[i] = p.pub_date.toordinal()
            elif p.year:
                lo[i], hi[i] = date(p.year, 1, 1).toordinal(), date(p.year, 12, 31).toordinal()
        self.date_lo, self.date_hi = lo, hi

    # ------------------------------------------------------------ pre-filter

    def _cpc_candidates(self, queries: Sequence[CPCQuery]) -> np.ndarray:
        found: list[np.ndarray] = []
        for q in queries:
            subclasses = [q.prefix] if len(q.prefix) == 4 else [s for s in self.by_subclass if s.startswith(q.prefix)]
            for s in subclasses:
                ids = self.by_subclass.get(s, _EMPTY)
                if q.group is not None:
                    ids = ids[[q.matches(self.codes[i]) for i in ids]] if ids.size else ids
                found.append(ids)
        return np.unique(np.concatenate(found)).astype(np.int32) if found else _EMPTY

    def _title_candidates(self, f: CompiledFilters) -> np.ndarray:
        """Still a linear scan of all title text (~8 ms for 457k titles); at
        millions of patents this would need a trigram index (like Postgres pg_trgm)."""
        if f.title_exact:
            return self._title_exact.get(f.title, _EMPTY)
        hits = np.fromiter((m.start() for m in re.finditer(re.escape(f.title), self._title_blob)), dtype=np.int64)
        if hits.size == 0:
            return _EMPTY
        return np.unique(np.searchsorted(self._title_starts, hits, side="right") - 1).astype(np.int32)

    def candidates(self, f: CompiledFilters) -> FilterResult:
        """Patents passing every filter. Each indexed filter gives a sorted set of
        patent ids; they are intersected smallest-first, so the work shrinks fast.
        The date filter (a cheap array comparison) runs on what's left."""
        sets: list[np.ndarray] = []
        if f.cpc:
            sets.append(self._cpc_candidates(f.cpc))
        for term in f.keyword_terms:
            sets.append(self.postings.get(term, _EMPTY))
        if f.title:
            sets.append(self._title_candidates(f))
        sets.sort(key=len)

        result = sets[0] if sets else np.arange(self.n, dtype=np.int32)
        for s in sets[1:]:
            if result.size == 0:
                break
            result = np.intersect1d(result, s, assume_unique=True)

        if (f.date_from is not None or f.date_to is not None) and result.size:
            result = result[self._date_mask(result, f)]
        return FilterResult(result.astype(np.int32), excluded_no_cpc=self.no_cpc if f.cpc else 0)

    # ------------------------------------------------------------ per-patent check

    def _title_ok(self, i: int, f: CompiledFilters) -> bool:
        return self.titles[i] == f.title if f.title_exact else f.title in self.titles[i]

    def _date_mask(self, ids: np.ndarray, f: CompiledFilters) -> np.ndarray:
        lo, hi = self.date_lo[ids], self.date_hi[ids]
        mask = lo >= 0
        if f.date_from is not None:
            mask &= hi >= f.date_from
        if f.date_to is not None:
            mask &= lo <= f.date_to
        return mask

    def _has_term(self, i: int, term: str) -> bool:
        ids = self.postings.get(term)
        if ids is None:
            return False
        pos = int(np.searchsorted(ids, i))
        return pos < ids.size and ids[pos] == i

    def passes(self, i: int, f: CompiledFilters) -> bool:
        """Does patent i pass all filters? (used by the naive post-filter)"""
        if f.cpc and not any(q.matches(self.codes[i]) for q in f.cpc):
            return False
        if any(not self._has_term(i, t) for t in f.keyword_terms):
            return False
        if f.title and not self._title_ok(i, f):
            return False
        if f.date_from is not None or f.date_to is not None:
            return bool(self._date_mask(np.array([i]), f)[0])
        return True
