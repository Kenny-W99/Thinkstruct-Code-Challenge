"""The CLI end to end, using the offline hashing embedder."""

import json

import pytest

from patent_search.cli import main


@pytest.fixture(scope="module")
def built_index(small_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("cli") / "idx"
    assert main(["build", "--data-dir", str(small_dir), "--index", str(out), "--embedder", "hashing-64"]) == 0
    return out


def test_build_writes_index_files(built_index):
    names = {p.name for p in built_index.iterdir()}
    assert {"meta.json", "patents.jsonl", "bm25.npz", "bm25.json", "vectors.npy"} <= names


def test_search_json_output(built_index, capsys):
    code = main(["search", "carbon fiber spokes", "--index", str(built_index), "--cpc", "B60B", "--json"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["results"][0]["doc_number"] == "20240000004"
    assert out["candidate_patents"] == 2


def test_search_text_output_with_filters(built_index, capsys):
    code = main(["search", "wheel", "--index", str(built_index), "--keyword", "carbon", "--date-from", "2024-02-01"])
    assert code == 0
    out = capsys.readouterr().out
    assert "20240000004" in out and "Time:" in out and "Filters: 1 patents" in out


def test_search_with_rerank_on_and_auto(built_index, capsys):
    base = ["search", "--index", str(built_index), "--reranker", "overlap"]
    assert main([*base[:1], "carbon fiber spokes", *base[1:], "--rerank", "on", "--rerank-candidates", "100", "--top-k", "3"]) == 0
    assert "Reranked by overlap" in capsys.readouterr().out
    claim_style = "a wheel rim comprising " + " ".join(["carbon fiber spokes connected to an aluminum hub"] * 6)
    assert main([*base[:1], claim_style, *base[1:], "--rerank", "auto"]) == 0
    out = capsys.readouterr().out
    assert "Reranked by" not in out and "Reranking skipped" in out


def test_search_error_exit_code(built_index, capsys):
    assert main(["search", "99999999999", "--index", str(built_index)]) == 2
    assert "not in this dataset" in capsys.readouterr().err


def test_setup_data_unzips_then_skips(tmp_path, capsys):
    import zipfile

    z = tmp_path / "small.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("data/patent_data_small/patents_ipa240215.json", "[]")
    dest = tmp_path / "small"
    assert main(["setup-data", "--zip", str(z), "--dest", str(dest)]) == 0
    assert main(["setup-data", "--zip", str(z), "--dest", str(dest)]) == 0
    out = capsys.readouterr().out
    assert "unzipped 1 files" in out and "already there" in out


def test_data_report(large_dir, capsys):
    assert main(["data-report", "--data-dir", str(large_dir)]) == 0
    out = capsys.readouterr().out
    assert "duplicates dropped 1" in out and "bad file" in out


def test_serve_needs_an_index(tmp_path, capsys):
    assert main(["serve", "--index", str(tmp_path / "none")]) == 2
    assert "no index" in capsys.readouterr().err


def test_serve_builds_if_missing_then_runs_the_app(small_dir, tmp_path, monkeypatch):
    import uvicorn
    from fastapi.testclient import TestClient

    started = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: started.update(app=app, **kw))
    index = tmp_path / "idx"
    args = ["serve", "--build-if-missing", "--data-dir", str(small_dir), "--index", str(index)]
    assert main([*args, "--embedder", "hashing-64", "--reranker", "overlap", "--port", "9999"]) == 0
    assert (index / "meta.json").exists() and started["port"] == 9999
    assert TestClient(started["app"]).get("/api/health").json()["reranker"] == "overlap"
