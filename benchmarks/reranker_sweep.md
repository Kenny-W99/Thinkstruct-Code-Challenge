# Choosing the reranker (plain-English task, 48 queries)

How the phase-2 settings in `config.py` were chosen. Every row is a run of:

```bash
python cli.py evaluate --task plain --setups hybrid rerank --rerankers <model> \
    --rerank-candidates <N> --rerank-passages <P> --out ""     # PATENT_RERANK_MAX_LENGTH=<L>
```

Times are left out on purpose: most of these runs shared the machine with a background index build, so their timings are not comparable. The final configuration's timing (on an idle machine, CPU) is in `eval_plain.md` and `eval_claims.md`.

## 1. Which model? (top 100 patents, 1 passage each, 512 tokens)

| Reranker | Size | Recall@1 | Recall@10 | MRR | vs hybrid: better / worse |
|---|---:|---:|---:|---:|---|
| none (hybrid only) | - | 0.542 | 0.792 | 0.620 | - |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | 0.500 | 0.833 | 0.612 | 15 / 11 |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 33M | 0.521 | 0.854 | 0.627 | 14 / 10 |
| `mixedbread-ai/mxbai-rerank-xsmall-v1` | 70M | 0.521 | 0.750 | 0.596 | 13 / 13 |
| **`mixedbread-ai/mxbai-rerank-base-v1`** | 184M | **0.625** | **0.896** | **0.705** | **18 / 6** |

The ms-marco models are trained on short web questions and don't beat plain hybrid search on patent text. `mxbai-rerank-base` clearly does.

## 2. How much work per query? (`mxbai-rerank-base-v1`)

| Patents reranked | Passages each | Max tokens | Recall@1 | Recall@10 | MRR |
|---:|---:|---:|---:|---:|---:|
| 100 | 1 | 512 | 0.625 | 0.896 | 0.705 |
| 100 | 1 | 256 | 0.625 | 0.896 | 0.706 |
| 50 | 1 | 256 | 0.625 | 0.938 | 0.708 |
| **30** | **1** | **256** | **0.625** | **0.917** | **0.710** |
| 30 | 2 | 256 | 0.604 | 0.896 | 0.700 |

Cost grows with every column (pairs × tokens), but quality doesn't: **30 patents × 1 passage × 256 tokens** keeps all of the gain for about 1/7 of the work of the first row.

## 3. Claim-length queries (claims task, first 100 queries)

| Setup | Max tokens | Recall@1 | MRR | vs hybrid: better / worse |
|---|---:|---:|---:|---|
| Hybrid | - | 0.810 | 0.852 | - |
| + `ms-marco-MiniLM-L-6-v2` | 512 | 0.550 | 0.645 | 2 / 33 |
| + `mxbai-rerank-base-v1` | 512 | 0.430 | 0.505 | 2 / 46 |

Giving the reranker more room (512 tokens) does not rescue claim queries, so the drop is not a truncation problem. That is why `rerank="auto"` only reranks queries of 40 words or fewer. Plain-English queries are at most 24 words; 93% of claims are longer than 40.

## 4. Device and precision

`mxbai-rerank-base-v1` is saved in float16. Recent `transformers` versions load a model in its saved precision, and CPUs run float16 very slowly. 30 pairs at 256 tokens took:

| Setup | Time for 30 pairs |
|---|---:|
| CPU, float16 (default load, also via the sentence-transformers wrapper) | 8–20 s |
| CPU, float32 (what we use) | 1.6–2.5 s |
| Apple GPU (MPS), float32 | ~2.2 s |
| Apple GPU (MPS), float16 | ~1.9 s |

So we load it in float32 and run it on the CPU, which is also what a Docker container on a Mac gets.
