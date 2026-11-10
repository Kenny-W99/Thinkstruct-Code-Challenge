"""Plain functions between HTTP shapes and engine objects (no FastAPI here, so
they are unit-tested without a server)."""

from __future__ import annotations

from datetime import date

from ..filters import SearchFilters
from ..index import PatentIndex
from ..load import Patent
from ..search import MODES, SearchEngine
from .schemas import ClaimOut, FacetsOut, HealthOut, PatentOut, SubclassCount


def _clean_list(values: list[str] | None) -> list[str]:
    """Repeated query params, minus blanks ("?cpc=&cpc=B60B" -> ["B60B"])."""
    return [v.strip() for v in values or [] if v and v.strip()]


def filters_from_params(
    cpc: list[str] | None = None,
    keyword: list[str] | None = None,
    title: str | None = None,
    title_exact: bool = False,
    date_from: date | None = None,
    date_to: date | None = None,
) -> SearchFilters:
    return SearchFilters(
        cpc=_clean_list(cpc),
        keywords=_clean_list(keyword),
        title=title.strip() if title and title.strip() else None,
        title_exact=title_exact,
        date_from=date_from,
        date_to=date_to,
    )


def patent_out(p: Patent) -> PatentOut:
    return PatentOut(
        doc_number=p.doc_number,
        title=p.title,
        abstract=p.abstract,
        cpc=p.cpc.display if p.cpc else None,
        cpc_subclass=p.cpc.subclass if p.cpc else None,
        pub_date=p.pub_date,
        year=p.year,
        filename=p.filename,
        has_description=p.has_description,
        claims=[
            ClaimOut(number=c.number, text=c.text, depends_on=list(c.depends_on), independent=c.is_independent)
            for c in p.claims
        ],
        description=p.paragraphs,
    )


def facets_out(index: PatentIndex) -> FacetsOut:
    """What the filter panel needs: subclasses with counts and the date range."""
    meta = index.metadata
    subclasses = sorted(((code, int(ids.size)) for code, ids in meta.by_subclass.items()), key=lambda x: (-x[1], x[0]))
    dates = [p.pub_date for p in index.patents if p.pub_date]
    years = [p.year for p in index.patents if p.year]
    return FacetsOut(
        n_patents=len(index.patents),
        no_cpc=meta.no_cpc,
        subclasses=[SubclassCount(code=c, count=n) for c, n in subclasses],
        date_min=min(dates) if dates else None,
        date_max=max(dates) if dates else None,
        year_min=min(years) if years else None,
        year_max=max(years) if years else None,
    )


def health_out(engine: SearchEngine) -> HealthOut:
    m = engine.index.meta
    return HealthOut(
        dataset=m.get("dataset"),
        n_patents=len(engine.index.patents),
        n_passages=len(engine.index.passages),
        passages_by_kind=m["passages_by_kind"],
        embedder=m["embedder"],
        reranker=engine.reranker.name if engine.reranker else None,
        modes=list(MODES) if engine.embedder is not None else ["bm25"],
    )
