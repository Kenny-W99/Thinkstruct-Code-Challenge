"""Cut patents into passages: the unit we score and show to the user.

One passage = title + abstract, OR one claim, OR one description paragraph.
Long ones are split into overlapping word windows.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import MAX_PASSAGE_WORDS, PASSAGE_OVERLAP_WORDS
from .load import Patent

KINDS = ("abstract", "claim", "description")  # position = numeric code stored in the index


@dataclass(frozen=True)
class ChunkConfig:
    include_description: bool = True
    include_claims: bool = True
    # Independent claims define what is protected; dependent claims mostly add
    # details. Keeping only independent ones cuts large-data passages ~5x.
    independent_claims_only: bool = False
    title_context: bool = True  # put the title in front of claim/description text when embedding
    max_words: int = MAX_PASSAGE_WORDS
    overlap: int = PASSAGE_OVERLAP_WORDS


@dataclass(frozen=True)
class Passage:
    patent_idx: int
    kind: str  # one of KINDS
    label: str  # shown to the user: "Claim 7", "Description ¶12"
    text: str


def split_words(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    step = max(1, max_words - overlap)
    pieces = []
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):
            break
    return pieces


def _labeled(base: str, pieces: list[str]) -> list[tuple[str, str]]:
    if len(pieces) == 1:
        return [(base, pieces[0])]
    return [(f"{base} (part {i}/{len(pieces)})", piece) for i, piece in enumerate(pieces, 1)]


def chunk_patent(patent: Patent, patent_idx: int, config: ChunkConfig = ChunkConfig()) -> list[Passage]:
    out: list[Passage] = []

    def add(kind: str, base_label: str, text: str) -> None:
        for label, piece in _labeled(base_label, split_words(text, config.max_words, config.overlap)):
            out.append(Passage(patent_idx, kind, label, piece))

    head = ". ".join(x for x in (patent.title, patent.abstract) if x)
    if head:
        add("abstract", "Title + Abstract", head)
    for claim in patent.claims if config.include_claims else ():
        if config.independent_claims_only and not claim.is_independent:
            continue
        add("claim", f"Claim {claim.number}", claim.text)
    if config.include_description:
        for n, paragraph in enumerate(patent.paragraphs, 1):
            add("description", f"Description ¶{n}", paragraph)
    return out


def index_text(passage: Passage, title: str | None, title_context: bool = True) -> str:
    """The text we embed / keyword-index for a passage.

    Many small-data claims lost their opening words ("an axle body, ..."), so
    putting the title in front ("SPOKE: an axle body, ...") gives back context.
    The abstract passage already starts with the title.
    """
    if not title_context or passage.kind == "abstract" or not title:
        return passage.text
    return f"{title}: {passage.text}"
