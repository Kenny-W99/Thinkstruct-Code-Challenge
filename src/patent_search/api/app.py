"""The REST API (FastAPI).

    GET /api/health                 is the server up, what index and models it has
    GET /api/search?q=...           search, with filters as query parameters
    GET /api/patents/{doc_number}   one full patent
    GET /api/facets                 classification codes and date range, for filter menus

`create_app(engine)` takes a ready engine, so tests pass in a tiny offline one
and the server passes in the real one. Interactive docs are at /docs.
Errors: 400 = the request is valid but can't be run (bad classification
code, unknown patent ID, ...); 404 = unknown patent; 422 = wrong parameter type.
"""

from __future__ import annotations

from datetime import date

from fastapi import FastAPI, HTTPException, Query

from ..config import RERANK_CANDIDATES
from ..search import RERANK_CHOICES, QueryError, SearchEngine
from .convert import facets_out, filters_from_params, health_out, patent_out
from .schemas import FacetsOut, HealthOut, Kind, Mode, PatentOut, RerankChoice, SearchOut, Strategy

MAX_TOP_K = 100
MAX_QUERY_CHARS = 20_000  # a long claim is ~2,000 characters


def create_app(engine: SearchEngine) -> FastAPI:
    app = FastAPI(
        title="Patent search API",
        version="1.0",
        description="Hybrid (keyword + meaning) patent search with filters and optional two-phase reranking.",
    )
    app.state.engine = engine
    facets = facets_out(engine.index)  # fixed for the life of the index

    @app.get("/api/health", response_model=HealthOut, tags=["meta"])
    def health() -> HealthOut:
        return health_out(engine)

    @app.get("/api/facets", response_model=FacetsOut, tags=["meta"])
    def get_facets() -> FacetsOut:
        return facets

    @app.get("/api/search", response_model=SearchOut, tags=["search"])
    def search(
        q: str = Query(..., max_length=MAX_QUERY_CHARS, description='free text, a patent ID, or "<patent ID> claim <N>"'),
        cpc: list[str] | None = Query(None, description="classification prefix, repeatable (any of): B60B, B60C 11/03"),
        keyword: list[str] | None = Query(None, description="word that must be in the title or abstract, repeatable (all of)"),
        title: str | None = Query(None, description="title contains this text"),
        title_exact: bool = Query(False, description="title must equal `title`"),
        date_from: date | None = None,
        date_to: date | None = None,
        mode: Mode = "hybrid",
        strategy: Strategy = Query("pre", description="pre = filter first (fast); post = naive filter after scoring"),
        kinds: list[Kind] | None = Query(None, description="only search these passage kinds"),
        top_k: int = Query(10, ge=1, le=MAX_TOP_K),
        rerank: RerankChoice = Query("auto", description="phase 2 cross-encoder: auto = short queries only"),
        rerank_candidates: int = Query(
            RERANK_CANDIDATES, ge=1, le=MAX_TOP_K, description="how many phase-1 patents the reranker reads"
        ),
    ) -> dict:
        filters = filters_from_params(cpc, keyword, title, title_exact, date_from, date_to)
        try:
            response = engine.search(
                q,
                filters,
                top_k=top_k,
                mode=mode,
                strategy=strategy,
                kinds=kinds or None,
                rerank=RERANK_CHOICES[rerank],
                rerank_candidates=rerank_candidates,
            )
        except (QueryError, ValueError) as err:  # ValueError: bad classification code or date range
            raise HTTPException(status_code=400, detail=str(err)) from err
        return response.to_dict()

    @app.get("/api/patents/{doc_number}", response_model=PatentOut, tags=["patents"])
    def get_patent(doc_number: str) -> PatentOut:
        idx = engine.index.doc_to_idx.get(doc_number.strip())
        if idx is None:
            raise HTTPException(status_code=404, detail=f"Patent {doc_number} is not in this dataset.")
        return patent_out(engine.index.patents[idx])

    return app
