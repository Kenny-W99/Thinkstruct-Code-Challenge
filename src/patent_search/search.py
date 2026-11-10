"""The search engine.

    phase 1:  query -> filters -> BM25 + embeddings -> RRF -> group by patent
    phase 2:  (optional) cross-encoder rerank of the top ~100 patents

Every stage is timed, so the CLI, API, benchmark and evaluation all report the
same numbers.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from .chunk import KINDS
from .config import (
    PASSAGES_PER_PATENT,
    POST_FILTER_OVERFETCH,
    RERANK_AUTO_MAX_WORDS,
    RERANK_CANDIDATES,
    RERANK_PASSAGES_PER_PATENT,
    RETRIEVE_N,
    RRF_K,
)
from .embed import Embedder
from .filters import SearchFilters, compile_filters
from .fusion import rrf, top_k_indices
from .index import PatentIndex, rows_for_patents
from .rerank import Reranker, rerank_hits  # noqa: F401  (Reranker: type of self.reranker)
from .results import PassageHit, PatentHit, SearchResponse
from .tokenizer import highlight_spans, tokenize

__all__ = ["MODES", "STRATEGIES", "RERANK_CHOICES", "QueryError", "SearchEngine", "SearchResponse", "PatentHit", "PassageHit"]

MODES = ("hybrid", "bm25", "dense")
STRATEGIES = ("pre", "post")
# What users type (CLI flag, API query param) -> the `rerank` argument of search()
RERANK_CHOICES: dict[str, bool | str] = {"off": False, "on": True, "auto": "auto"}


class QueryError(ValueError):
    """The query can't be run (empty, unknown patent, unknown claim, bad option...)."""


@dataclass
class ResolvedQuery:
    text: str
    exclude_patent: int | None = None  # don't return the query patent itself
    note: str | None = None


# "20240051338", "US20240051338A1", "20240051338 claim 6"
_PATENT_REF = re.compile(r"^\s*(?:US)?\s*(\d{11})\s*(?:A\d)?\s*(?:,?\s*claim\s*(\d+))?\s*$", re.IGNORECASE)


class _Timer:
    def __init__(self) -> None:
        self.ms: dict[str, float] = {}
        self._start = self._t = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.ms[name] = round((now - self._t) * 1000, 3)
        self._t = now

    def total(self) -> dict[str, float]:
        self.ms["total"] = round((time.perf_counter() - self._start) * 1000, 3)
        return self.ms


class SearchEngine:
    def __init__(self, index: PatentIndex, embedder: Embedder | None, reranker: Reranker | None = None):
        if embedder is not None and embedder.dim != index.vectors.dim:
            raise ValueError(f"embedder dim {embedder.dim} != index dim {index.vectors.dim}")
        self.index = index
        self.embedder = embedder
        self.reranker = reranker

    def warmup(self) -> None:
        """Run the models once. The first call includes one-time setup (~250 ms on a
        GPU); without this, the first timing would not show real speed."""
        if self.embedder is not None:
            self.embedder.encode(["warm up"], is_query=True)
        if self.reranker is not None:
            self.reranker.score("warm up", ["warm up"])

    # ------------------------------------------------------------ query input

    def resolve_query(self, text: str) -> ResolvedQuery:
        """Free text, or a patent ID (optionally with "claim N") from the dataset."""
        text = (text or "").strip()
        if not text:
            raise QueryError("Query is empty.")
        match = _PATENT_REF.match(text)
        if not match:
            return ResolvedQuery(text)
        doc, claim_no = match.group(1), match.group(2)
        idx = self.index.doc_to_idx.get(doc)
        if idx is None:
            raise QueryError(f"Patent {doc} is not in this dataset.")
        patent = self.index.patents[idx]
        if claim_no:
            claim = next((c for c in patent.claims if c.number == int(claim_no)), None)
            if claim is None:
                raise QueryError(f"Patent {doc} has no claim {claim_no}.")
            body = f"{patent.title}: {claim.text}" if patent.title else claim.text
            return ResolvedQuery(body, idx, f"Searching with claim {claim_no} of {doc}")
        first_independent = next((c.text for c in patent.claims if c.is_independent), None)
        body = " ".join(x for x in (patent.title, patent.abstract, first_independent) if x)
        return ResolvedQuery(body, idx, f"Searching with the title, abstract and first claim of {doc}")

    # ------------------------------------------------------------ search

    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        *,
        top_k: int = 10,
        mode: str = "hybrid",
        strategy: str = "pre",
        kinds: Iterable[str] | None = None,
        passages_per_patent: int = PASSAGES_PER_PATENT,
        retrieve_n: int = RETRIEVE_N,
        overfetch: int = POST_FILTER_OVERFETCH,
        rerank: bool | str = False,
        rerank_candidates: int = RERANK_CANDIDATES,
        rerank_passages: int = RERANK_PASSAGES_PER_PATENT,
    ) -> SearchResponse:
        """rerank: False, True, or "auto" (rerank short queries only; see config)."""
        if mode not in MODES:
            raise QueryError(f"mode must be one of {MODES}")
        if strategy not in STRATEGIES:
            raise QueryError(f"strategy must be one of {STRATEGIES}")
        if kinds is not None and any(k not in KINDS for k in kinds):
            raise QueryError(f"kinds must be from {KINDS}")
        if mode != "bm25" and self.embedder is None:
            raise QueryError("This engine has no embedder; use mode='bm25'.")
        if rerank not in (True, False, "auto"):
            raise QueryError('rerank must be true, false or "auto"')
        if rerank is True and self.reranker is None:
            raise QueryError("This engine has no reranker loaded.")
        if top_k < 1:
            raise QueryError("top_k must be at least 1")
        if rerank_candidates < 1 or rerank_passages < 1:
            raise QueryError("rerank_candidates and rerank_passages must be at least 1")

        timer = _Timer()
        resolved = self.resolve_query(query)
        rerank, rerank_note = self._decide_rerank(rerank, resolved.text)
        q_tokens = tokenize(resolved.text)
        q_vec = self.embedder.encode([resolved.text], is_query=True)[0] if mode != "bm25" else None
        timer.lap("embed")

        # 1. which passage rows may be scored
        compiled = compile_filters(filters) if filters and not filters.is_empty() else None
        rows, candidate_patents, excluded_no_cpc = self._candidate_rows(compiled, strategy, kinds, resolved)
        timer.lap("filter")

        # 2. score with each retriever, keep the top N of each
        n_take = overfetch if compiled and strategy == "post" else retrieve_n
        if rerank:  # phase 2 needs ~rerank_candidates patents out of phase 1
            n_take = max(n_take, rerank_candidates * 4)
        rankings, ranks, raw_scores = self._retrieve(q_tokens, q_vec, rows, mode, n_take, timer)

        # 3. merge (a single mode keeps its own scores: BM25 score or cosine)
        if mode == "hybrid":
            fused_rows, fused_scores = rrf(rankings, k=RRF_K)
        else:
            fused_rows = rankings[0]
            fused_scores = np.array([raw_scores[r] for r in fused_rows.tolist()], dtype=np.float32)
        timer.lap("fuse")

        # 4. naive post-filter: drop results that fail the filters (may leave fewer than top_k!)
        if compiled and strategy == "post":
            passes = self.index.metadata.passes
            keep = np.array([passes(int(self.index.passage_patent[r]), compiled) for r in fused_rows.tolist()], dtype=bool)
            fused_rows, fused_scores = fused_rows[keep], fused_scores[keep]
            timer.lap("post_filter")

        # 5. group passages into patents (+ optional phase 2)
        phase1_k = max(top_k, rerank_candidates) if rerank else top_k
        per_patent = max(passages_per_patent, rerank_passages) if rerank else passages_per_patent
        results = self._group(fused_rows, fused_scores, ranks, set(q_tokens), phase1_k, per_patent)
        timer.lap("group")
        if rerank:
            results = self._rerank(resolved.text, results, top_k, rerank_candidates, rerank_passages, passages_per_patent)
            timer.lap("rerank")

        return SearchResponse(
            query=resolved.text,
            note=resolved.note,
            mode=mode,
            strategy=strategy,
            results=results,
            timings_ms=timer.total(),
            candidate_patents=candidate_patents,
            candidate_passages=len(self.index.passages) if rows is None else int(rows.size),
            excluded_no_cpc=excluded_no_cpc,
            reranker=self.reranker.name if rerank else None,
            rerank_note=rerank_note,
        )

    # ------------------------------------------------------------ helpers

    def _candidate_rows(self, compiled, strategy, kinds, resolved) -> tuple[np.ndarray | None, int | None, int]:
        """Passage rows allowed by the filters (None = all rows)."""
        index = self.index
        rows: np.ndarray | None = None
        candidate_patents: int | None = None
        excluded = 0
        if compiled is not None:
            if strategy == "pre":
                found = index.metadata.candidates(compiled)
                rows = rows_for_patents(found.patents, index.starts)
                candidate_patents, excluded = int(found.patents.size), found.excluded_no_cpc
            else:
                excluded = index.metadata.no_cpc if compiled.cpc else 0
        if kinds:
            wanted = [KINDS.index(k) for k in kinds]
            if rows is None:
                rows = np.flatnonzero(np.isin(index.passage_kind, wanted))
            else:
                rows = rows[np.isin(index.passage_kind[rows], wanted)]
        if resolved.exclude_patent is not None:
            lo, hi = index.starts[resolved.exclude_patent], index.starts[resolved.exclude_patent + 1]
            if rows is None:
                rows = np.concatenate((np.arange(lo), np.arange(hi, len(index.passages))))
            else:
                rows = rows[(rows < lo) | (rows >= hi)]
        return rows, candidate_patents, excluded

    def _retrieve(self, q_tokens, q_vec, rows, mode, n_take, timer):
        """Top n_take passages from BM25 and/or the vector index (as global row ids)."""
        rankings: list[np.ndarray] = []
        ranks: dict[str, dict[int, int]] = {"bm25": {}, "dense": {}}
        raw_scores: dict[int, float] = {}
        if mode != "dense":
            s = self.index.bm25.score(q_tokens, rows)
            top = top_k_indices(s, n_take)
            top = top[s[top] > 0]  # passages with none of the query words
            ranked = top if rows is None else rows[top]
            rankings.append(ranked)
            ranks["bm25"] = {r: i for i, r in enumerate(ranked.tolist(), 1)}
            raw_scores.update(zip(ranked.tolist(), s[top].tolist()))
            timer.lap("bm25")
        if mode != "bm25":
            s = self.index.vectors.score(q_vec, rows)
            top = top_k_indices(s, n_take)
            ranked = top if rows is None else rows[top]
            rankings.append(ranked)
            ranks["dense"] = {r: i for i, r in enumerate(ranked.tolist(), 1)}
            raw_scores.update(zip(ranked.tolist(), s[top].tolist()))
            timer.lap("dense")
        return rankings, ranks, raw_scores

    def _decide_rerank(self, rerank: bool | str, query_text: str) -> tuple[bool, str | None]:
        """Resolve "auto" into on/off. Returns (rerank?, note for the user)."""
        if rerank != "auto":
            return bool(rerank), None
        if self.reranker is None:
            return False, "Reranking is not available (no reranker loaded)."
        words = len(query_text.split())
        if words > RERANK_AUTO_MAX_WORDS:
            return False, (
                f"Reranking skipped: a {words}-word, claim-style query. In our evaluation, "
                "reranking helped short plain-English queries but hurt claim-length ones."
            )
        return True, None

    def _rerank(self, query, results, top_k, candidates, rerank_passages, keep_passages) -> list[PatentHit]:
        """Phase 2 on the first `candidates` patents only; the rest keep their
        phase-1 order after them (so asking for more results than were reranked works).

        The cross-encoder reads only each patent's best `rerank_passages` passages
        (the measured setting: cost = candidates x rerank_passages pairs). The other
        phase-1 passages are still shown after the reranked ones."""
        head_in = results[:candidates]
        head = rerank_hits(
            query,
            [replace(h, passages=h.passages[:rerank_passages]) for h in head_in],
            self.reranker,
            top_k=candidates,
            keep_passages=rerank_passages,
            passage_text=self._rerank_text,
        )
        unscored = {h.doc_number: h.passages[rerank_passages:] for h in head_in}
        head = [replace(h, passages=(h.passages + unscored[h.doc_number])[:keep_passages]) for h in head]
        tail = [replace(h, first_stage_rank=h.rank, passages=h.passages[:keep_passages]) for h in results[candidates:]]
        return [replace(h, rank=r) for r, h in enumerate((head + tail)[:top_k], start=1)]

    def _rerank_text(self, hit: PatentHit, passage: PassageHit) -> str:
        """What the cross-encoder reads: same title context as the index used."""
        if self.index.chunk_config.title_context and passage.kind != "abstract" and hit.title:
            return f"{hit.title}: {passage.text}"
        return passage.text

    def _group(self, rows, scores, ranks, q_terms, top_k, per_patent) -> list[PatentHit]:
        """Best passage decides a patent's rank (the 'MaxP' rule); keep its top passages."""
        index = self.index
        hits: dict[int, PatentHit] = {}
        for row, score in zip(rows.tolist(), scores.tolist()):
            p_idx = int(index.passage_patent[row])
            hit = hits.get(p_idx)
            if hit is None:
                if len(hits) >= top_k:
                    continue
                p = index.patents[p_idx]
                hit = hits[p_idx] = PatentHit(
                    rank=len(hits) + 1,
                    doc_number=p.doc_number,
                    title=p.title,
                    cpc=p.cpc.display if p.cpc else None,
                    pub_date=p.pub_date.isoformat() if p.pub_date else None,
                    year=p.year,
                    has_description=p.has_description,
                    score=float(score),
                )
            if len(hit.passages) < per_patent:
                passage = index.passages[row]
                hit.passages.append(
                    PassageHit(
                        row=int(row),
                        kind=passage.kind,
                        label=passage.label,
                        text=passage.text,
                        score=float(score),
                        bm25_rank=ranks["bm25"].get(row),
                        dense_rank=ranks["dense"].get(row),
                        highlights=highlight_spans(passage.text, q_terms),
                    )
                )
        return list(hits.values())
