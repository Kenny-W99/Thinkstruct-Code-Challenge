# Search timing: `large50`

- Index: 49,867 patents, 198,674 passages {'abstract': 50306, 'claim': 148368, 'description': 0}, embedder `BAAI/bge-small-en-v1.5`, chunking {'include_description': False, 'independent_claims_only': True, 'title_context': True, 'max_words': 250, 'overlap': 50}
- Build time (s): {'chunk': 1.593, 'bm25': 11.727, 'embed': 1558.64, 'load': 22.327, 'total': 1606.749}
- Query embedding (cpu): median 8.6 ms, p95 9.3 ms per query. It is paid once per search and is NOT in the table below.
- Table: 16 queries × 3 runs per row, mode `hybrid`, top 10 patents. `pre` = lookup tables first, then score only matching passages; `post` = score all, keep the top 1000, then filter.
| Filter | Strategy | Patents passing | Passages scored | Median ms | p95 ms | Filter ms | Scoring ms | Results | vs no filter |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no filter | - | 49,867 | 198,674 | 6.02 | 6.52 | 0.00 | 4.97 | 10.0 | 1.00× |
| CPC B60B (wheels) | pre | 20 | 73 | 1.22 | 1.34 | 0.02 | 0.12 | 10.0 | 0.20× |
| CPC B60B (wheels) | post | 20 | 198,674 | 6.67 | 7.50 | 0.84 | 5.26 | 1.8 | 1.11× |
| CPC G06F (computing) | pre | 2,601 | 10,743 | 2.95 | 3.44 | 0.21 | 1.50 | 10.0 | 0.49× |
| CPC G06F (computing) | post | 2,601 | 198,674 | 7.11 | 7.82 | 0.80 | 5.25 | 6.6 | 1.18× |
| CPC B60C + keyword 'tread' | pre | 21 | 59 | 1.25 | 1.45 | 0.02 | 0.11 | 10.0 | 0.21× |
| CPC B60C + keyword 'tread' | post | 21 | 198,674 | 6.37 | 7.63 | 0.80 | 5.15 | 1.4 | 1.06× |
| keyword 'vehicle' | pre | 2,374 | 8,457 | 2.25 | 2.61 | 0.04 | 1.07 | 10.0 | 0.37× |
| keyword 'vehicle' | post | 2,374 | 198,674 | 9.50 | 9.92 | 2.91 | 5.15 | 9.6 | 1.58× |
| title contains 'wheel' | pre | 133 | 442 | 2.35 | 2.57 | 0.91 | 0.18 | 10.0 | 0.39× |
| title contains 'wheel' | post | 133 | 198,674 | 6.49 | 7.42 | 0.68 | 5.15 | 3.8 | 1.08× |
| CPC B60 + from 2024 | pre | 114 | 407 | 1.44 | 1.58 | 0.10 | 0.17 | 10.0 | 0.24× |
| CPC B60 + from 2024 | post | 114 | 198,674 | 7.00 | 8.09 | 0.92 | 5.15 | 4.8 | 1.16× |
