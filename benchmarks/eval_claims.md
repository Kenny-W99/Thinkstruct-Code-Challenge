# Evaluation: claim -> its own description

| Setup | Recall@1 | Recall@10 | Recall@30 | Recall@100 | MRR | Median ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 only | 0.825 | 0.911 | 0.921 | 0.934 | 0.859 | 4.8 | 5.6 |
| Embeddings only | 0.713 | 0.881 | 0.909 | 0.933 | 0.774 | 4.9 | 5.8 |
| Hybrid (RRF) | 0.778 | 0.906 | 0.919 | 0.938 | 0.826 | 6.1 | 7.0 |
| Hybrid + rerank `ms-marco-MiniLM-L-6-v2` | 0.522 | 0.751 | 0.919 | 0.938 | 0.602 | 260.3 | 266.6 |
| Hybrid + rerank `mxbai-rerank-base-v1` | 0.372 | 0.622 | 0.919 | 0.938 | 0.463 | 2513.0 | 2535.8 |

- 519 queries: first independent claim of each patent with a description; searched against 21,721 description passages indexed without titles. Embedder `BAAI/bge-small-en-v1.5`.
- Times are per query, excluding the query embedding (done once, in a batch). Rerank: top 30 patents × 1 passage(s) each, max 256 tokens per pair.
- Hybrid + rerank `ms-marco-MiniLM-L-6-v2` vs hybrid: right patent ranked higher for 18 queries, lower for 188, same for 313. Median rerank cost 254 ms/query.
- Hybrid + rerank `mxbai-rerank-base-v1` vs hybrid: right patent ranked higher for 14 queries, lower for 270, same for 235. Median rerank cost 2506 ms/query.
