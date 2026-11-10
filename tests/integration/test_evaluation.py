"""The evaluation pipeline end to end on the fixture data (hashing embedder, overlap reranker)."""

import json

import pytest

from patent_search.cli import main
from patent_search.evaluate import EVAL_CHUNK_CONFIG, SETUPS, build_eval_queries, load_query_file, run_setup
from patent_search.index import PatentIndex
from patent_search.load import load_patents
from patent_search.rerank import OverlapReranker
from patent_search.search import SearchEngine


@pytest.fixture(scope="module")
def eval_engine(small_dir, embedder):
    patents, _ = load_patents(small_dir)
    index = PatentIndex.build(patents, embedder, EVAL_CHUNK_CONFIG)
    return SearchEngine(index, embedder, OverlapReranker())


def test_eval_index_has_no_titles_in_passages(eval_engine):
    assert not eval_engine.index.chunk_config.title_context
    assert eval_engine._rerank_text(None, eval_engine.index.passages[0]) == eval_engine.index.passages[0].text


@pytest.mark.parametrize("key", list(SETUPS))
def test_every_setup_runs_and_finds_the_obvious_patent(eval_engine, key):
    queries = build_eval_queries(eval_engine.index.patents, min_words=3)
    result = run_setup(eval_engine, queries, SETUPS[key], ks=(1, 3))
    assert result.n_queries == len(queries) == 3
    assert result.recall[3] == 1.0  # only 3 patents have descriptions
    assert 0 < result.mrr <= 1 and result.median_ms >= 0
    assert (result.median_rerank_ms is not None) == SETUPS[key].rerank


def test_cli_evaluate_claims_builds_index_and_writes_report(small_dir, tmp_path, capsys):
    out = tmp_path / "eval.md"
    args = [
        "evaluate", "--task", "claims",
        "--data-dir", str(small_dir),
        "--index", str(tmp_path / "eval_idx"),
        "--embedder", "hashing-64",
        "--rerankers", "overlap",
        "--setups", "bm25", "hybrid", "rerank",
        "--out", str(out),
    ]
    assert main(args) == 0
    text = out.read_text()
    assert text.startswith("# Evaluation: claim -> its own description")
    assert "Recall@1" in text and "Hybrid + rerank `overlap`" in text and "vs hybrid" in text
    assert len(json.loads(out.with_suffix(".json").read_text())) == 3
    assert "building the evaluation index" in capsys.readouterr().out
    assert main(args) == 0  # second run reuses the saved index
    assert "building the evaluation index" not in capsys.readouterr().out


def test_cli_evaluate_plain_uses_query_file(small_index, tmp_path):
    small_index.save(tmp_path / "idx")
    queries = tmp_path / "q.jsonl"
    queries.write_text(
        '{"doc_number": "20240000004", "query": "rim with carbon fiber spokes"}\n'
        '{"doc_number": "20240000003", "query": "ball mount locked with a pin"}\n'
    )
    out = tmp_path / "plain.md"
    args = ["evaluate", "--task", "plain", "--index", str(tmp_path / "idx"), "--queries", str(queries),
            "--setups", "bm25", "rerank", "--rerankers", "overlap", "--out", str(out)]
    assert main(args) == 0
    text = out.read_text()
    assert text.startswith("# Evaluation: plain-English idea -> patent") and "2 queries" in text


def test_query_file_with_unknown_patent_fails_loudly(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"doc_number": "99999999999", "query": "x"}\n')
    with pytest.raises(ValueError, match="not in the index"):
        load_query_file(bad, known_docs={"20240000001"})
