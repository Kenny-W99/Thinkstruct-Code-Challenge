"""Command line interface.

    python cli.py setup-data                      # unzip data/patent_data_small.zip
    python cli.py data-report --dataset large
    python cli.py build --dataset small
    python cli.py search "tire tread that reduces road noise" --cpc B60C
    python cli.py search "20240051338 claim 6" --top-k 5
    python cli.py serve                           # REST API on http://localhost:8000
    python cli.py pg-load --max-files 5           # Part 2 PoC: load into Postgres
    python cli.py pg-search "battery cooling" --cpc H01M
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

from .config import (
    ARTIFACTS_DIR,
    DATA_DIRS,
    EMBEDDING_MODEL,
    PG_DSN,
    PROJECT_ROOT,
    QUERY_DEVICE,
    RERANK_CANDIDATES,
    RERANK_DEVICE,
    RERANK_MAX_LENGTH,
    RERANK_MODEL,
    SMALL_DATA_ZIP,
)
from .chunk import KINDS, ChunkConfig
from .filters import SearchFilters
from .loader import build_index, load_engine
from .search import MODES, RERANK_CHOICES, STRATEGIES, QueryError, SearchEngine, SearchResponse

log = logging.getLogger("patent_search")


def _data_dir(args) -> Path:
    """The data folder to read. For the default small data, unzip the bundled zip
    first if needed, so a fresh clone works with no manual step."""
    if args.data_dir:
        return Path(args.data_dir).expanduser()
    if args.dataset == "small":
        from .setup_data import ensure_small_data

        ensure_small_data()
    return DATA_DIRS[args.dataset]


def _index_dir(args) -> Path:
    if args.index:
        return Path(args.index).expanduser()
    return ARTIFACTS_DIR / args.dataset


# ------------------------------------------------------------------ commands


def cmd_setup_data(args) -> int:
    from .setup_data import ensure_small_data

    zip_path = Path(args.zip).expanduser() if args.zip else SMALL_DATA_ZIP
    dest = Path(args.dest).expanduser() if args.dest else DATA_DIRS["small"]
    dest, n = ensure_small_data(zip_path, dest)
    print(f"unzipped {n} files -> {dest}" if n else f"already there: {dest}")
    return 0


def cmd_data_report(args) -> int:
    from .load import load_patents

    _, report = load_patents(_data_dir(args), max_files=args.max_files, keep_description=False)
    print(report.summary())
    for bad in report.bad_files:
        print("  bad file:", bad)
    return 0


def _chunk_config(args) -> ChunkConfig:
    return ChunkConfig(
        include_description=not args.no_description,
        include_claims=not args.no_claims,
        independent_claims_only=args.independent_claims_only,
        title_context=not args.no_title_context,
    )


def cmd_build(args) -> int:
    out = _index_dir(args)
    index, report = build_index(
        _data_dir(args),
        out,
        config=_chunk_config(args),
        embedder_name=args.embedder,
        max_files=args.max_files,
        dataset=args.dataset,
    )
    m = index.meta
    print("load:", report.summary())
    print(f"built {m['n_passages']:,} passages from {m['n_patents']:,} patents -> {out}")
    print("passages by kind:", m["passages_by_kind"])
    print("build seconds:", m["build_seconds"])
    return 0


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def filters_from_args(args) -> SearchFilters:
    return SearchFilters(
        cpc=args.cpc or [],
        keywords=args.keyword or [],
        title=args.title,
        title_exact=args.title_exact,
        date_from=_parse_date(args.date_from),
        date_to=_parse_date(args.date_to),
    )


def _snippet(text: str, spans: list[tuple[int, int]], width: int = 260, color: bool = False) -> str:
    start = max(0, spans[0][0] - width // 3) if spans else 0
    end = min(len(text), start + width)
    on, off = ("\033[1m", "\033[0m") if color else ("[", "]")
    out, pos = [], start
    for a, b in spans:
        if a < start or b > end:
            continue
        out += [text[pos:a], on, text[a:b], off]
        pos = b
    out.append(text[pos:end])
    return ("…" if start > 0 else "") + "".join(out) + ("…" if end < len(text) else "")


def print_response(resp: SearchResponse, color: bool = False) -> None:
    print(f"Query: {resp.query[:160]}{'…' if len(resp.query) > 160 else ''}")
    if resp.note:
        print(f"       ({resp.note})")
    if resp.candidate_patents is not None:
        extra = f", {resp.excluded_no_cpc:,} had no classification" if resp.excluded_no_cpc else ""
        print(f"Filters: {resp.candidate_patents:,} patents / {resp.candidate_passages:,} passages pass{extra}")
    t = resp.timings_ms
    parts = " · ".join(f"{k} {v:.1f}" for k, v in t.items() if k != "total")
    print(f"Time: {t['total']:.1f} ms  ({parts})   mode={resp.mode} strategy={resp.strategy}")
    if resp.reranker:
        print(f"Reranked by {resp.reranker}")
    if resp.rerank_note:
        print(resp.rerank_note)
    if not resp.results:
        print("\nNo results.")
    for hit in resp.results:
        date_str = hit.pub_date or (str(hit.year) if hit.year else "")
        moved = f"  (was #{hit.first_stage_rank})" if hit.first_stage_rank else ""
        print(f"\n{hit.rank:>2}. {hit.doc_number}  {hit.title or '(no title)'}  [{hit.cpc or 'no code'}]  {date_str}{moved}")
        for p in hit.passages:
            print(f"    · {p.label}: {_snippet(p.text, p.highlights, color=color)}")


def cmd_search(args) -> int:
    rerank = RERANK_CHOICES[args.rerank]
    engine = load_engine(_index_dir(args), reranker_name=args.reranker if rerank else None)
    engine.warmup()  # so the printed timings show steady-state speed, not first-call setup
    try:
        resp = engine.search(
            args.query,
            filters_from_args(args),
            top_k=args.top_k,
            mode=args.mode,
            strategy=args.strategy,
            kinds=args.kinds,
            rerank=rerank,
            rerank_candidates=args.rerank_candidates,
        )
    except (QueryError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(resp.to_dict(), indent=2))
    else:
        print_response(resp, color=sys.stdout.isatty())
    return 0


def cmd_serve(args) -> int:
    """Run the REST API. With --build-if-missing, the first start builds the index
    (this is what the Docker container does)."""
    import uvicorn

    from .api import create_app

    index_dir = _index_dir(args)
    if not (index_dir / "meta.json").exists():
        if not args.build_if_missing:
            print(f"error: no index at {index_dir}. Build it: python cli.py build --dataset {args.dataset}", file=sys.stderr)
            return 2
        print(f"no index at {index_dir}; building it (first start only)...")
        build_index(_data_dir(args), index_dir, embedder_name=args.embedder, dataset=args.dataset)
    reranker = None if args.reranker == "none" else args.reranker
    engine = load_engine(index_dir, reranker_name=reranker)
    engine.warmup()
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="info")
    return 0


def cmd_pg_load(args) -> int:
    """Part 2 PoC: queue the batch files as jobs and load them into Postgres."""
    from .embed import make_embedder
    from .pg.ingest import run_ingest
    from .pg.store import PgStore

    store = PgStore.connect(args.dsn)
    try:
        embedder = make_embedder(args.embedder)
        store.apply_schema(embedder.dim)
        t = time.perf_counter()
        processed = run_ingest(store, _data_dir(args), embedder, max_files=args.max_files)
        print(f"processed {processed} jobs in {time.perf_counter() - t:.1f}s")
        print(json.dumps(store.job_summary(), indent=2, default=str))
    finally:
        store.close()
    return 0


def cmd_pg_status(args) -> int:
    from .pg.store import PgStore

    store = PgStore.connect(args.dsn)
    try:
        print(json.dumps(store.job_summary(), indent=2, default=str))
    finally:
        store.close()
    return 0


def cmd_pg_search(args) -> int:
    """Part 2 PoC: filter + vector search as one SQL query."""
    from .embed import make_embedder
    from .pg.rows import vector_literal, where_clause
    from .pg.store import PgStore

    try:
        where, params = where_clause(filters_from_args(args))
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    embedder = make_embedder(args.embedder, device=QUERY_DEVICE)
    query_vector = vector_literal(embedder.encode([args.query], is_query=True)[0])
    store = PgStore.connect(args.dsn)
    try:
        store.search(query_vector, where, params, top_k=1)  # warm up the connection and caches
        rows, ms, plan = store.search(query_vector, where, params, top_k=args.top_k, exact=args.exact, explain=args.explain)
    finally:
        store.close()
    print(f"{len(rows)} patents in {ms:.1f} ms ({'exact scan' if args.exact else 'HNSW index'})  WHERE {where}")
    for i, r in enumerate(rows, 1):
        print(f"{i:>2}. {r['doc_number']}  {r['title'] or '(no title)'}  [{r['cpc_raw'] or 'no code'}]  {r['published'] or ''}"
              f"  distance {r['distance']:.3f}  via {r['best_passage']}")
    if plan:
        print("\nQuery plan:\n" + "\n".join(plan))
    return 0


def _filter_args(p) -> None:
    p.add_argument("--cpc", action="append", help="classification prefix, repeatable (any of): B60B, 'B60C 11/03'")
    p.add_argument("--keyword", action="append", help="word that must be in title/abstract, repeatable (all of)")
    p.add_argument("--title", help="title contains this text")
    p.add_argument("--title-exact", action="store_true", help="title must equal --title")
    p.add_argument("--date-from", help="YYYY-MM-DD")
    p.add_argument("--date-to", help="YYYY-MM-DD")


def _load_or_build_eval_index(args):
    """The evaluation index (description paragraphs, no titles). Built once, then reused."""
    from .embed import make_embedder
    from .evaluate import EVAL_CHUNK_CONFIG
    from .index import PatentIndex
    from .load import load_patents

    index_dir = Path(args.index).expanduser() if args.index else ARTIFACTS_DIR / "eval_small"
    if (index_dir / "meta.json").exists():
        return PatentIndex.load(index_dir)
    print(f"building the evaluation index -> {index_dir}")
    patents, report = load_patents(_data_dir(args))
    index = PatentIndex.build(
        patents,
        make_embedder(args.embedder),
        EVAL_CHUNK_CONFIG,
        extra_meta={"dataset": args.dataset, "purpose": "evaluation", "load_report": report.to_dict()},
        show_progress=True,
    )
    index.save(index_dir)
    return index


def cmd_benchmark(args) -> int:
    from .benchmark import BENCH_QUERIES, format_timing_markdown, measure_query_embedding, run_benchmark
    from .embed import CachedEmbedder, make_embedder
    from .index import PatentIndex

    index_dir = _index_dir(args)
    index = PatentIndex.load(index_dir)
    if index.meta["embedder"]:
        query_embedder = make_embedder(index.meta["embedder"], device=QUERY_DEVICE)
        embed_cost = measure_query_embedding(query_embedder)
        cached = CachedEmbedder(query_embedder)
        cached.prefill(BENCH_QUERIES)
        engine, mode = SearchEngine(index, cached), args.mode
    else:  # keyword-only index
        embed_cost = {"median_ms": 0.0, "p95_ms": 0.0}
        engine, mode = SearchEngine(index, None), "bm25"
    args.mode = mode
    rows = run_benchmark(engine, repeats=args.repeats, mode=mode)

    m = index.meta
    header = [
        f"# Search timing: `{index_dir.name}`",
        "",
        f"- Index: {m['n_patents']:,} patents, {m['n_passages']:,} passages {m['passages_by_kind']}, "
        f"embedder `{m['embedder']}`, chunking {m['chunking']}",
        f"- Build time (s): {m['build_seconds']}",
        f"- Query embedding ({QUERY_DEVICE}): median {embed_cost['median_ms']:.1f} ms, "
        f"p95 {embed_cost['p95_ms']:.1f} ms per query. It is paid once per search and is NOT in the table below.",
        f"- Table: {len(BENCH_QUERIES)} queries × {args.repeats} runs per row, mode `{args.mode}`, top 10 patents. "
        "`pre` = lookup tables first, then score only matching passages; `post` = score all, keep the top 1000, then filter.",
        "",
    ]
    report = "\n".join(header) + format_timing_markdown(rows)
    print(report)
    if args.out != "":
        out = Path(args.out) if args.out else PROJECT_ROOT / "benchmarks" / f"timing_{index_dir.name}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        out.with_suffix(".json").write_text(
            json.dumps({"meta": m, "query_embedding": embed_cost, "rows": [r.to_dict() for r in rows]}, indent=2)
        )
        print(f"wrote {out}")
    return 0


EVAL_TASKS = {
    "claims": "claim -> its own description",
    "plain": "plain-English idea -> patent",
}


def _eval_inputs(args):
    """(index, queries, passage kinds to search, description of the queries) for a task."""
    from .evaluate import build_eval_queries, load_query_file
    from .index import PatentIndex

    if args.task == "claims":
        index = _load_or_build_eval_index(args)
        queries = build_eval_queries(index.patents, limit=args.limit)
        n_desc = index.meta["passages_by_kind"]["description"]
        return index, queries, ("description",), (
            f"first independent claim of each patent with a description; searched against "
            f"{n_desc:,} description passages indexed without titles"
        )
    index = PatentIndex.load(Path(args.index).expanduser() if args.index else ARTIFACTS_DIR / args.dataset)
    queries = load_query_file(args.queries, known_docs=set(index.doc_to_idx))[: args.limit]
    return index, queries, None, (
        f"one-sentence plain-English descriptions from `{Path(args.queries).name}`; "
        f"searched against all {len(index.passages):,} passages of the normal index"
    )


def cmd_evaluate(args) -> int:
    from .embed import CachedEmbedder, make_embedder
    from .evaluate import SETUPS, Setup, compare_ranks, format_markdown, run_setup
    from .rerank import make_reranker

    if args.out is None:
        args.out = str(PROJECT_ROOT / "benchmarks" / f"eval_{args.task}.md")
    index, queries, kinds, about = _eval_inputs(args)
    embedder = CachedEmbedder(make_embedder(index.meta["embedder"]))  # default device: GPU if present
    print(f"embedding {len(queries)} queries...")
    embedder.prefill([q.text for q in queries], show_progress=True)

    results, rerank_results = [], []
    plain = SearchEngine(index, embedder)
    for key in args.setups:
        if key != "rerank":
            print(f"running: {SETUPS[key].name}")
            results.append(run_setup(plain, queries, SETUPS[key], kinds=kinds))
            continue
        for model in args.rerankers:
            engine = SearchEngine(index, embedder, make_reranker(model, device=RERANK_DEVICE))
            engine.warmup()
            setup = Setup(f"Hybrid + rerank `{model.split('/')[-1]}`", "hybrid", rerank=True)
            print(f"running: {setup.name}")
            result = run_setup(
                engine,
                queries,
                setup,
                kinds=kinds,
                rerank_passages=args.rerank_passages,
                rerank_candidates=args.rerank_candidates,
            )
            results.append(result)
            rerank_results.append(result)

    notes = [
        f"- {len(queries)} queries: {about}. Embedder `{index.meta['embedder']}`.",
        "- Times are per query, excluding the query embedding (done once, in a batch). "
        f"Rerank: top {args.rerank_candidates} patents × {args.rerank_passages} passage(s) each, "
        f"max {RERANK_MAX_LENGTH} tokens per pair.",
    ]
    hybrid = next((r for r in results if r.name == SETUPS["hybrid"].name), None)
    for rr in rerank_results:
        if hybrid is not None:
            moved = compare_ranks(hybrid, rr)
            notes.append(
                f"- {rr.name} vs hybrid: right patent ranked higher for {moved['improved']} queries, "
                f"lower for {moved['worse']}, same for {moved['unchanged']}. "
                f"Median rerank cost {rr.median_rerank_ms:.0f} ms/query."
            )
    report = format_markdown(results, notes="\n".join(notes))
    print("\n" + report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"# Evaluation: {EVAL_TASKS[args.task]}\n\n" + report)
        out.with_suffix(".json").write_text(json.dumps([r.to_dict() for r in results], indent=2))
        print(f"wrote {out}")
    return 0


# ------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="patent-search", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def data_args(p):
        p.add_argument("--dataset", choices=sorted(DATA_DIRS), default="small")
        p.add_argument("--data-dir", help="override the data folder")
        p.add_argument("--max-files", type=int, help="only read the first N files (large data slice)")

    p = sub.add_parser("setup-data", help="unzip the bundled small dataset (skipped if already there)")
    p.add_argument("--zip", help=f"zip file (default {SMALL_DATA_ZIP.name})")
    p.add_argument("--dest", help="target folder (default data/patent_data_small)")
    p.set_defaults(func=cmd_setup_data)

    p = sub.add_parser("data-report", help="load the data and print the cleaning report")
    data_args(p)
    p.set_defaults(func=cmd_data_report)

    p = sub.add_parser("build", help="build and save a search index")
    data_args(p)
    p.add_argument("--index", help="output folder (default artifacts/<dataset>)")
    p.add_argument("--embedder", default=EMBEDDING_MODEL, help="HF model name, hashing-256 for a quick test, or none (keyword-only)")
    p.add_argument("--no-description", action="store_true", help="index title, abstract and claims only")
    p.add_argument("--no-claims", action="store_true", help="don't index claims (with --no-description: title + abstract only)")
    p.add_argument("--independent-claims-only", action="store_true", help="skip dependent claims (~5x fewer passages)")
    p.add_argument("--no-title-context", action="store_true", help="don't put the title in front of passages")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("search", help="search an index")
    p.add_argument("query", help='free text, a patent ID, or "<patent ID> claim <N>"')
    p.add_argument("--dataset", choices=sorted(DATA_DIRS), default="small")
    p.add_argument("--index", help="index folder (default artifacts/<dataset>)")
    _filter_args(p)
    p.add_argument("--kinds", nargs="+", choices=KINDS, help="only search these passage kinds")
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.add_argument("--strategy", choices=STRATEGIES, default="pre", help="pre-filter (fast) or naive post-filter")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--json", action="store_true", help="print the raw JSON response")
    p.add_argument(
        "--rerank",
        choices=sorted(RERANK_CHOICES),
        default="off",
        help="two-phase search: rerank the top 30 with a cross-encoder (auto = short queries only)",
    )
    p.add_argument("--reranker", default=RERANK_MODEL, help="cross-encoder model (or 'overlap' for a quick test)")
    p.add_argument(
        "--rerank-candidates",
        type=int,
        default=RERANK_CANDIDATES,
        help=f"how many phase-1 patents the reranker reads (default {RERANK_CANDIDATES}; the PDF's example is 100)",
    )
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("serve", help="run the REST API (docs at /docs)")
    p.add_argument("--dataset", choices=sorted(DATA_DIRS), default="small")
    p.add_argument("--data-dir", help="data folder, used only with --build-if-missing")
    p.add_argument("--index", help="index folder (default artifacts/<dataset>)")
    p.add_argument("--build-if-missing", action="store_true", help="build the index first if the folder has none")
    p.add_argument("--embedder", default=EMBEDDING_MODEL, help="used only when the index has to be built")
    p.add_argument("--reranker", default=RERANK_MODEL, help="cross-encoder for rerank=auto/on, or 'none' (or 'overlap' for a quick test)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("pg-load", help="Part 2 PoC: load batch files into Postgres + pgvector (as tracked jobs)")
    data_args(p)
    p.add_argument("--dsn", default=PG_DSN, help="Postgres connection (env PATENT_PG_DSN)")
    p.add_argument("--embedder", default=EMBEDDING_MODEL)
    p.set_defaults(func=cmd_pg_load)

    p = sub.add_parser("pg-status", help="Part 2 PoC: ingest job status")
    p.add_argument("--dsn", default=PG_DSN)
    p.set_defaults(func=cmd_pg_status)

    p = sub.add_parser("pg-search", help="Part 2 PoC: filter + vector search in one SQL query")
    p.add_argument("query")
    _filter_args(p)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--exact", action="store_true", help="exact scan instead of the HNSW index (to compare)")
    p.add_argument("--explain", action="store_true", help="print the query plan")
    p.add_argument("--dsn", default=PG_DSN)
    p.add_argument("--embedder", default=EMBEDDING_MODEL)
    p.set_defaults(func=cmd_pg_search)

    p = sub.add_parser("benchmark", help="Part 1: time searches with and without filters (pre vs post)")
    p.add_argument("--dataset", choices=sorted(DATA_DIRS), default="small")
    p.add_argument("--index", help="index folder (default artifacts/<dataset>)")
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out", help="markdown report path (default benchmarks/timing_<index>.md; '' = don't write)")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("evaluate", help="Part 3: compare search setups (with/without reranking)")
    data_args(p)
    p.add_argument("--task", choices=sorted(EVAL_TASKS), default="claims", help="claims: claim -> own description; plain: plain-English query -> patent")
    p.add_argument("--index", help="index folder (claims: artifacts/eval_small, built if missing; plain: artifacts/<dataset>)")
    p.add_argument("--queries", default=str(PROJECT_ROOT / "benchmarks" / "plain_english_queries.jsonl"), help="query file for --task plain")
    p.add_argument("--embedder", default=EMBEDDING_MODEL, help="used only when the claims index has to be built")
    p.add_argument("--rerankers", nargs="+", default=[RERANK_MODEL], help="cross-encoder model(s) to compare (or 'overlap')")
    p.add_argument("--rerank-passages", type=int, default=1, help="passages per patent sent to the reranker")
    p.add_argument("--rerank-candidates", type=int, default=RERANK_CANDIDATES, help="how many phase-1 patents get reranked")
    p.add_argument("--setups", nargs="+", default=["bm25", "dense", "hybrid", "rerank"], choices=["bm25", "dense", "hybrid", "rerank"])
    p.add_argument("--limit", type=int, help="only the first N queries (quick check)")
    p.add_argument("--out", help="markdown report path (default benchmarks/eval_<task>.md; '' = don't write)")
    p.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # In Docker the models are baked in and HF_HUB_OFFLINE=1; "can't reach the Hub" is expected there.
    logging.getLogger("huggingface_hub.hf_api").setLevel(logging.ERROR)
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
