"""Phase 2 of two-phase search: re-score the top candidates with a cross-encoder.

Phase 1 (hybrid search) is fast because the query and each passage are turned
into vectors *separately*. A cross-encoder reads the query and a passage
*together*, so it can judge the match much more precisely, but it costs a full
model run per passage. So it only runs on the best phase-1 patents (30 by
default, RERANK_CANDIDATES in config.py).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Protocol, Sequence

import numpy as np

from .config import RERANK_MAX_LENGTH, RERANK_MODEL
from .results import PassageHit, PatentHit
from .tokenizer import tokenize


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """One relevance score per passage (higher = better)."""
        ...


class OverlapReranker:
    """Offline stand-in for tests: share of query words found in the passage."""

    name = "overlap"

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        q = set(tokenize(query))
        if not q:
            return np.zeros(len(passages), dtype=np.float32)
        return np.array([len(q & set(tokenize(p))) / len(q) for p in passages], dtype=np.float32)


class CrossEncoderReranker:
    """A Hugging Face cross-encoder, called directly through `transformers`.

    Always loaded in float32. Recent `transformers` versions load a model in the
    precision it was saved in, and mxbai-rerank-base was saved in float16, which
    CPUs run very slowly: 8-20 s for 30 pairs vs ~2 s in float32. CPU speed
    matters because that is all a Docker container on a Mac gets.
    """

    def __init__(
        self,
        model_name: str = RERANK_MODEL,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = RERANK_MAX_LENGTH,
    ):
        import torch  # lazy: importing torch is slow
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.name, self.device, self.batch_size = model_name, device, batch_size
        # max_length caps query + passage tokens. Cost grows faster than linearly
        # with length, so this is the main speed/quality knob.
        self.max_length = max_length
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name, dtype=torch.float32).to(device).eval()
        )

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        out: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, len(passages), self.batch_size):
                batch = list(passages[start : start + self.batch_size])
                features = self.tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation="longest_first",  # long claim queries get cut too, never just the passage to nothing
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**features).logits.float()
                # 1 output = relevance score; 2 outputs = (not relevant, relevant)
                scores = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1] - logits[:, 0]
                out.append(scores.cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def make_reranker(name: str, device: str | None = None) -> Reranker:
    """'overlap' -> OverlapReranker; anything else is a Hugging Face cross-encoder name."""
    return OverlapReranker() if name == "overlap" else CrossEncoderReranker(name, device=device)


PassageText = Callable[[PatentHit, PassageHit], str]


def rerank_hits(
    query: str,
    hits: list[PatentHit],
    reranker: Reranker,
    *,
    top_k: int,
    keep_passages: int,
    passage_text: PassageText = lambda hit, passage: passage.text,
) -> list[PatentHit]:
    """Score every (query, passage) pair in one batch. A patent's new score is its
    best passage's score. Re-sort, keep top_k, remember each patent's phase-1 rank."""
    pairs = [(i, j) for i, hit in enumerate(hits) for j in range(len(hit.passages))]
    if not pairs:
        return []
    scores = reranker.score(query, [passage_text(hits[i], hits[i].passages[j]) for i, j in pairs])

    per_patent: list[list[float]] = [[] for _ in hits]
    for (i, _), s in zip(pairs, scores.tolist()):
        per_patent[i].append(s)

    reranked: list[PatentHit] = []
    for hit, passage_scores in zip(hits, per_patent):
        if not passage_scores:
            continue
        passages = sorted(
            (replace(p, rerank_score=s) for p, s in zip(hit.passages, passage_scores)),
            key=lambda p: -p.rerank_score,
        )
        reranked.append(
            replace(hit, score=max(passage_scores), passages=passages[:keep_passages], first_stage_rank=hit.rank)
        )
    reranked.sort(key=lambda h: (-h.score, h.first_stage_rank))
    return [replace(h, rank=r) for r, h in enumerate(reranked[:top_k], start=1)]
