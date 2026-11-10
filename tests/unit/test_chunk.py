from patent_search.chunk import ChunkConfig, Passage, chunk_patent, index_text, split_words
from patent_search.load import parse_record
from tests.conftest import SMALL_FILES


def _tire():
    return parse_record(SMALL_FILES["patents_ipa240215.json"][0])


def test_one_passage_per_abstract_claim_and_paragraph():
    passages = chunk_patent(_tire(), 7)
    kinds = [p.kind for p in passages]
    assert kinds == ["abstract", "claim", "claim", "claim", "description", "description"]
    assert [p.label for p in passages][:3] == ["Title + Abstract", "Claim 1", "Claim 2"]
    assert passages[-1].label == "Description ¶2"
    assert all(p.patent_idx == 7 for p in passages)
    assert passages[0].text.startswith("PNEUMATIC TIRE. A pneumatic tire")


def test_without_description():
    passages = chunk_patent(_tire(), 0, ChunkConfig(include_description=False))
    assert "description" not in {p.kind for p in passages}


def test_title_and_abstract_only():
    passages = chunk_patent(_tire(), 0, ChunkConfig(include_description=False, include_claims=False))
    assert [p.kind for p in passages] == ["abstract"]


def test_independent_claims_only():
    passages = chunk_patent(_tire(), 0, ChunkConfig(independent_claims_only=True))
    assert [p.label for p in passages if p.kind == "claim"] == ["Claim 1"]  # 2 and 5 depend on claim 1


def test_split_words_windows_overlap():
    text = " ".join(str(i) for i in range(10))
    assert split_words(text, max_words=4, overlap=1) == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]
    assert split_words("short text", max_words=4, overlap=1) == ["short text"]


def test_long_paragraph_gets_part_labels():
    patent = parse_record({"doc_number": "1", "abstract": "x", "detailed_description": [" ".join(["w"] * 30)]})
    passages = chunk_patent(patent, 0, ChunkConfig(max_words=10, overlap=2))
    labels = [p.label for p in passages if p.kind == "description"]
    assert labels[0] == "Description ¶1 (part 1/4)" and len(labels) == 4


def test_index_text_adds_title_context_except_for_abstract():
    claim = Passage(0, "claim", "Claim 1", "an axle body")
    assert index_text(claim, "SPOKE") == "SPOKE: an axle body"
    assert index_text(claim, "SPOKE", title_context=False) == "an axle body"
    assert index_text(claim, None) == "an axle body"
    assert index_text(Passage(0, "abstract", "Title + Abstract", "SPOKE. x"), "SPOKE") == "SPOKE. x"
