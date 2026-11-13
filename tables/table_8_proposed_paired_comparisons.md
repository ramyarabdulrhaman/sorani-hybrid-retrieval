| Fused retriever minus (system B) | Delta MRR@10 (pts) | 95% bootstrap CI | p (two-sided) | Holm-adjusted p | CI excludes 0 |
|---|---|---|---|---|---|
| BM25 (word) | +1.55 | [+0.60, +2.62] | < 0.001 | 0.004 | yes |
| BM25 (morpheme stems) | +1.57 | [+0.69, +2.58] | < 0.001 | 0.004 | yes |
| BM25 (char 4-gram) | +0.02 | [-0.34, +0.42] | 0.862 | 0.862 | no |
| multilingual-e5-base zero-shot | +15.75 | [+13.13, +18.39] | < 0.001 | 0.004 | yes |
| bge-m3 zero-shot | +4.92 | [+3.42, +6.60] | < 0.001 | 0.004 | yes |
| Hybrid RRF (BM25 char 4-gram + bge-m3) | +1.77 | [+1.06, +2.58] | < 0.001 | 0.004 | yes |
| Fusion of zero-shot components (tuned weights) | +0.20 | [-0.08, +0.48] | 0.180 | 0.360 | no |
| e5-base fine-tuned (selected config) | +3.78 | [+2.68, +4.85] | < 0.001 | 0.004 | yes |
