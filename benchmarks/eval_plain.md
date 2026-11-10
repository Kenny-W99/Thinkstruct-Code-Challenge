# Evaluation: plain-English idea -> patent

| Setup | Recall@1 | Recall@10 | Recall@30 | Recall@100 | MRR | Median ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 only | 0.417 | 0.667 | 0.792 | 0.958 | 0.514 | 3.7 | 4.3 |
| Embeddings only | 0.438 | 0.854 | 0.917 | 0.938 | 0.587 | 4.0 | 4.5 |
| Hybrid (RRF) | 0.542 | 0.792 | 0.917 | 0.979 | 0.620 | 5.0 | 5.4 |
| Hybrid + rerank `ms-marco-MiniLM-L-6-v2` | 0.500 | 0.854 | 0.917 | 0.979 | 0.617 | 266.1 | 316.7 |
| Hybrid + rerank `mxbai-rerank-base-v1` | 0.625 | 0.917 | 0.917 | 0.979 | 0.710 | 2535.8 | 2577.5 |

- 48 queries: one-sentence plain-English descriptions from `plain_english_queries.jsonl`; searched against all 32,929 passages of the normal index. Embedder `BAAI/bge-small-en-v1.5`.
- Times are per query, excluding the query embedding (done once, in a batch). Rerank: top 30 patents × 1 passage(s) each, max 256 tokens per pair.
- Hybrid + rerank `ms-marco-MiniLM-L-6-v2` vs hybrid: right patent ranked higher for 14 queries, lower for 8, same for 26. Median rerank cost 261 ms/query.
- Hybrid + rerank `mxbai-rerank-base-v1` vs hybrid: right patent ranked higher for 15 queries, lower for 6, same for 27. Median rerank cost 2530 ms/query.
