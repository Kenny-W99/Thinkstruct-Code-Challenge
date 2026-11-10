"""The timing benchmark on the fixture index (hashing embedder)."""

from datetime import date

from patent_search.benchmark import (
    FilterCase,
    format_timing_markdown,
    measure_query_embedding,
    run_benchmark,
)
from patent_search.cli import main
from patent_search.filters import SearchFilters

CASES = (
    FilterCase("no filter", None),
    FilterCase("CPC B60B", SearchFilters(cpc=["B60B"])),
    FilterCase("from 2024-02-20", SearchFilters(date_from=date(2024, 2, 20))),
)


def test_one_row_without_filter_and_two_per_filter(engine):
    rows = run_benchmark(engine, ["wheel spokes", "tire noise"], CASES, repeats=1)
    assert [(r.case, r.strategy) for r in rows] == [
        ("no filter", "-"),
        ("CPC B60B", "pre"),
        ("CPC B60B", "post"),
        ("from 2024-02-20", "pre"),
        ("from 2024-02-20", "post"),
    ]
    no_filter, b60b_pre, b60b_post = rows[:3]
    assert no_filter.patents_passing == 4 and no_filter.passages_scored == len(engine.index.passages)
    assert b60b_pre.patents_passing == b60b_post.patents_passing == 2
    assert b60b_pre.passages_scored < b60b_post.passages_scored  # pre-filter scores fewer passages
    assert all(r.median_ms >= 0 and r.p95_ms >= r.median_ms for r in rows)
    assert all(r.avg_results <= 10 for r in rows)


def test_markdown_has_one_line_per_row_and_ratio(engine):
    rows = run_benchmark(engine, ["wheel"], CASES[:2], repeats=1)
    md = format_timing_markdown(rows)
    assert md.count("\n") == 2 + len(rows)
    assert "| no filter | - |" in md and "1.00×" in md


def test_query_embedding_cost(embedder):
    cost = measure_query_embedding(embedder, ["a", "b"], repeats=2)
    assert cost["median_ms"] >= 0 and cost["p95_ms"] >= cost["median_ms"]


def test_cli_benchmark_writes_report(small_index, tmp_path, capsys):
    small_index.save(tmp_path / "idx")
    out = tmp_path / "timing.md"
    assert main(["benchmark", "--index", str(tmp_path / "idx"), "--repeats", "1", "--out", str(out)]) == 0
    text = out.read_text()
    assert text.startswith("# Search timing: `idx`") and "Query embedding" in text and "| no filter |" in text
    assert out.with_suffix(".json").exists()
