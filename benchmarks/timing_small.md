# Search timing: `small`

- Index: 640 patents, 32,929 passages {'abstract': 645, 'claim': 10563, 'description': 21721}, embedder `BAAI/bge-small-en-v1.5`, chunking {'include_description': True, 'title_context': True, 'max_words': 250, 'overlap': 50}
- Build time (s): {'chunk': 0.205, 'bm25': 1.07, 'embed': 105.91, 'load': 0.71, 'total': 122.737}
- Query embedding (cpu): median 8.7 ms, p95 10.0 ms per query. It is paid once per search and is NOT in the table below.
- Table: 16 queries × 3 runs per row, mode `hybrid`, top 10 patents. `pre` = lookup tables first, then score only matching passages; `post` = score all, keep the top 1000, then filter.
| Filter | Strategy | Patents passing | Passages scored | Median ms | p95 ms | Filter ms | Scoring ms | Results | vs no filter |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no filter | - | 640 | 32,929 | 1.96 | 2.28 | 0.00 | 0.90 | 10.0 | 1.00× |
| CPC B60B (wheels) | pre | 298 | 14,352 | 1.83 | 2.18 | 0.06 | 0.75 | 10.0 | 0.93× |
| CPC B60B (wheels) | post | 298 | 32,929 | 3.21 | 3.92 | 0.73 | 1.13 | 10.0 | 1.63× |
| CPC G06F (computing) | pre | 0 | 0 | 0.02 | 0.03 | 0.01 | 0.01 | 0.0 | 0.01× |
| CPC G06F (computing) | post | 0 | 32,929 | 2.11 | 2.27 | 0.54 | 1.11 | 0.0 | 1.07× |
| CPC B60C + keyword 'tread' | pre | 152 | 9,400 | 1.60 | 1.82 | 0.06 | 0.63 | 10.0 | 0.81× |
| CPC B60C + keyword 'tread' | post | 152 | 32,929 | 3.58 | 4.98 | 1.41 | 1.13 | 10.0 | 1.83× |
| keyword 'vehicle' | pre | 204 | 9,798 | 2.28 | 2.58 | 0.02 | 1.19 | 10.0 | 1.16× |
| keyword 'vehicle' | post | 204 | 32,929 | 4.45 | 5.02 | 2.00 | 1.13 | 10.0 | 2.26× |
| title contains 'wheel' | pre | 189 | 9,560 | 2.36 | 2.73 | 0.07 | 1.17 | 10.0 | 1.20× |
| title contains 'wheel' | post | 189 | 32,929 | 2.98 | 3.40 | 0.51 | 1.13 | 10.0 | 1.52× |
| CPC B60 + from 2024 | pre | 640 | 32,929 | 2.07 | 2.27 | 0.12 | 0.91 | 10.0 | 1.05× |
| CPC B60 + from 2024 | post | 640 | 32,929 | 6.23 | 6.85 | 3.54 | 1.14 | 10.0 | 3.17× |
