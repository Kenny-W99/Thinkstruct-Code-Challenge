# Search timing: `large_all_kw`

- Index: 457,385 patents, 461,495 passages {'abstract': 461495, 'claim': 0, 'description': 0}, embedder `None`, chunking {'include_description': False, 'include_claims': False, 'independent_claims_only': False, 'title_context': True, 'max_words': 250, 'overlap': 50}
- Build time (s): {'chunk': 2.395, 'bm25': 20.976, 'embed': 0.0, 'load': 75.495, 'total': 118.378}
- Query embedding (cpu): median 0.0 ms, p95 0.0 ms per query. It is paid once per search and is NOT in the table below.
- Table: 16 queries × 3 runs per row, mode `bm25`, top 10 patents. `pre` = lookup tables first, then score only matching passages; `post` = score all, keep the top 1000, then filter.
| Filter | Strategy | Patents passing | Passages scored | Median ms | p95 ms | Filter ms | Scoring ms | Results | vs no filter |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no filter | - | 457,385 | 461,495 | 1.99 | 3.43 | 0.00 | 1.60 | 10.0 | 1.00× |
| CPC B60B (wheels) | pre | 190 | 192 | 0.48 | 0.58 | 0.03 | 0.13 | 7.0 | 0.24× |
| CPC B60B (wheels) | post | 190 | 461,495 | 2.39 | 3.54 | 0.56 | 1.74 | 2.7 | 1.20× |
| CPC G06F (computing) | pre | 24,438 | 24,539 | 2.70 | 2.92 | 2.00 | 0.23 | 10.0 | 1.35× |
| CPC G06F (computing) | post | 24,438 | 461,495 | 2.53 | 4.12 | 0.54 | 1.72 | 6.0 | 1.27× |
| CPC B60C + keyword 'tread' | pre | 204 | 206 | 0.36 | 0.64 | 0.05 | 0.13 | 4.7 | 0.18× |
| CPC B60C + keyword 'tread' | post | 204 | 461,495 | 2.35 | 3.47 | 0.54 | 1.72 | 1.6 | 1.18× |
| keyword 'vehicle' | pre | 20,885 | 21,120 | 0.83 | 0.97 | 0.13 | 0.26 | 10.0 | 0.41× |
| keyword 'vehicle' | post | 20,885 | 461,495 | 5.65 | 6.97 | 3.42 | 1.72 | 9.6 | 2.83× |
| title contains 'wheel' | pre | 1,268 | 1,285 | 9.13 | 9.63 | 8.58 | 0.16 | 9.6 | 4.58× |
| title contains 'wheel' | post | 1,268 | 461,495 | 2.43 | 3.55 | 0.46 | 1.74 | 4.8 | 1.22× |
| CPC B60 + from 2024 | pre | 1,150 | 1,159 | 1.20 | 1.31 | 0.61 | 0.15 | 9.8 | 0.60× |
| CPC B60 + from 2024 | post | 1,150 | 461,495 | 2.84 | 3.86 | 0.71 | 1.73 | 5.9 | 1.42× |
