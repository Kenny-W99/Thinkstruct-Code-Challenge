"""Building, saving and loading an index gives the same search results."""

import pytest

from patent_search.chunk import ChunkConfig
from patent_search.index import PatentIndex
from patent_search.load import load_patents
from patent_search.search import SearchEngine


def test_save_load_round_trip(tmp_path, small_index, embedder):
    small_index.save(tmp_path / "idx")
    loaded = PatentIndex.load(tmp_path / "idx")
    assert loaded.meta == small_index.meta
    assert [p.label for p in loaded.passages] == [p.label for p in small_index.passages]
    before = SearchEngine(small_index, embedder).search("carbon fiber wheel")
    after = SearchEngine(loaded, embedder).search("carbon fiber wheel")
    assert [h.doc_number for h in after.results] == [h.doc_number for h in before.results]


def test_meta_records_how_it_was_built(small_index):
    m = small_index.meta
    assert m["embedder"] == "hashing-128" and m["n_patents"] == 4
    assert m["n_passages"] == len(small_index.passages)
    assert sum(m["passages_by_kind"].values()) == m["n_passages"]
    assert {"chunk", "bm25", "embed"} <= set(m["build_seconds"])


def test_passages_are_contiguous_per_patent(small_index):
    for p_idx in range(len(small_index.patents)):
        lo, hi = small_index.starts[p_idx], small_index.starts[p_idx + 1]
        assert all(small_index.passages[r].patent_idx == p_idx for r in range(lo, hi))


def test_no_description_build(large_dir, embedder):
    patents, _ = load_patents(large_dir, keep_description=False)
    index = PatentIndex.build(patents, embedder, ChunkConfig(include_description=False))
    assert index.meta["passages_by_kind"]["description"] == 0
    assert all(p.has_description for p in patents if p.doc_number == "20150000001")


def test_keyword_only_index_without_embedder(tmp_path, large_dir):
    patents, _ = load_patents(large_dir, keep_description=False, keep_claims=False)
    index = PatentIndex.build(patents, None, ChunkConfig(include_description=False, include_claims=False))
    assert index.meta["embedder"] is None and index.vectors.dim == 0
    index.save(tmp_path / "kw")
    engine = SearchEngine(PatentIndex.load(tmp_path / "kw"), None)
    assert engine.search("wireless charging coil", mode="bm25").results[0].doc_number == "20180000002"


def test_cli_build_keyword_only(large_dir, tmp_path, capsys):
    from patent_search.cli import main

    out = tmp_path / "kw"
    assert main(["build", "--data-dir", str(large_dir), "--index", str(out), "--embedder", "none", "--no-description", "--no-claims"]) == 0
    assert main(["search", "battery cooling", "--index", str(out), "--mode", "bm25"]) == 0
    assert "20150000001" in capsys.readouterr().out


def test_missing_index_folder_has_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="build"):
        PatentIndex.load(tmp_path)
