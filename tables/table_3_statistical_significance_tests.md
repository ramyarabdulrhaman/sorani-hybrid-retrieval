| System vs. BM25 (word) | Delta MRR@10 (pts) | 95% bootstrap CI | p (two-sided) | Holm-adjusted p | CI excludes 0 |
|---|---|---|---|---|---|
| BM25 (morpheme stems) | +0.16 | [-0.32, +0.67] | 0.516 | 0.516 | no |
| BM25 (char 4-gram) | +1.72 | [+0.61, +2.95] | 0.003 | 0.009 | yes |
| multilingual-e5-base | -27.70 | [-31.47, -24.06] | < 0.001 | 0.004 | yes |
| bge-m3 (dense) | -6.92 | [-9.47, -4.62] | < 0.001 | 0.004 | yes |
| Hybrid RRF (BM25 stems + e5-base) | -10.61 | [-13.08, -8.05] | < 0.001 | 0.004 | yes |
| Hybrid RRF (BM25 stems + bge-m3) | -1.68 | [-3.14, -0.26] | 0.025 | 0.050 | yes |
| Hybrid RRF (BM25 stems + e5-base + bge-m3) | -3.23 | [-5.14, -1.43] | 0.001 | 0.004 | yes |
