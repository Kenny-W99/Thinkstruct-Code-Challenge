"""Pure functions for the Postgres PoC: engine objects -> SQL rows and WHERE
clauses. No database needed, so they are unit-tested directly."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..chunk import ChunkConfig, Passage, chunk_patent, index_text
from ..filters import SearchFilters, compile_filters
from ..load import Patent

# The PoC indexes title + abstract and independent claims only: enough to show
# the pipeline end to end at a fraction of the embedding cost (see DESIGN.md).
POC_CHUNK_CONFIG = ChunkConfig(include_description=False, independent_claims_only=True)

PATENT_COLUMNS = (
    "doc_number",
    "title",
    "abstract",
    "cpc_raw",
    "cpc_subclass",
    "cpc_group",
    "cpc_subgroup",
    "cpc_digits",
    "pub_date",
    "pub_year",
    "has_description",
    "n_claims",
    "source_file",
    "text_size",
)
PASSAGE_COLUMNS = ("doc_number", "kind", "label", "text", "embedding")


def patent_row(p: Patent, source_file: str) -> tuple:
    c = p.cpc
    return (
        p.doc_number,
        p.title,
        p.abstract,
        c.raw if c else None,
        c.subclass if c else None,
        c.group if c else None,
        c.subgroup if c else None,
        c.digits if c else None,
        p.pub_date,
        p.year,
        p.has_description,
        len(p.claims),
        source_file,
        p.text_size,
    )


def poc_passages(p: Patent, config: ChunkConfig = POC_CHUNK_CONFIG) -> list[tuple[Passage, str]]:
    """(passage, text to embed) pairs, same chunking and title context as the main index."""
    return [(ps, index_text(ps, p.title, config.title_context)) for ps in chunk_patent(p, 0, config)]


def vector_literal(vec: Sequence[float] | np.ndarray) -> str:
    """pgvector's text format: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{float(x):.6g}" for x in vec) + "]"


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def where_clause(filters: SearchFilters | None) -> tuple[str, dict]:
    """SQL condition on table alias `p` (patents) with named parameters.
    Same meaning as the in-memory filters in filters.py."""
    if filters is None or filters.is_empty():
        return "TRUE", {}
    f = compile_filters(filters)  # validates codes and the date range
    parts: list[str] = []
    params: dict = {}

    def param(value) -> str:
        key = f"f{len(params)}"
        params[key] = value
        return f"%({key})s"

    if f.cpc:
        ors = []
        for q in f.cpc:
            if q.group is None:
                ors.append(f"p.cpc_subclass LIKE {param(q.prefix + '%')}")
            else:  # structured codes: exact group + subgroup prefix; unstructured: digit prefix
                subclass, group, subgroup = param(q.prefix), param(q.group), param(q.subgroup + "%")
                digits = param(q.group + q.subgroup + "%")
                ors.append(
                    f"(p.cpc_subclass = {subclass} AND ("
                    f"(p.cpc_group IS NOT NULL AND p.cpc_group = {group} AND coalesce(p.cpc_subgroup, '') LIKE {subgroup})"
                    f" OR (p.cpc_group IS NULL AND p.cpc_digits LIKE {digits})))"
                )
        parts.append("(" + " OR ".join(ors) + ")")
    if filters.keywords:
        parts.append(f"p.tsv @@ plainto_tsquery('english', {param(' '.join(filters.keywords))})")
    if f.title:
        if f.title_exact:
            parts.append(f"lower(p.title) = {param(f.title)}")
        else:
            parts.append(f"lower(p.title) LIKE {param('%' + _like_escape(f.title) + '%')} ESCAPE '\\'")
    # Year-only patents cover Jan 1 - Dec 31 of their year (same rule as filters.py).
    if filters.date_from:
        parts.append(f"coalesce(p.pub_date, make_date(p.pub_year, 12, 31)) >= {param(filters.date_from)}")
    if filters.date_to:
        parts.append(f"coalesce(p.pub_date, make_date(p.pub_year, 1, 1)) <= {param(filters.date_to)}")
    return " AND ".join(parts), params
