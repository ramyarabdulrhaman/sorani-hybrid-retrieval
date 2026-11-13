| System (Recall@1 %) | 3-7 words (n=220) | 8-9 words (n=122) | 10-17 words (n=150) |
|---|---|---|---|
| BM25 (word) | 89.55 | 95.90 | 98.67 |
| BM25 (morpheme stems) | 90.45 | 95.08 | 98.67 |
| BM25 (char 4-gram) | 93.64 | 95.90 | 98.00 |
| multilingual-e5-base | 59.09 | 59.02 | 63.33 |
| bge-m3 (dense) | 79.55 | 87.70 | 88.00 |
| Hybrid RRF (BM25 stems + e5-base) | 78.18 | 76.23 | 79.33 |
| Hybrid RRF (BM25 stems + bge-m3) | 86.36 | 92.62 | 95.33 |
| Hybrid RRF (BM25 stems + e5-base + bge-m3) | 88.18 | 85.25 | 91.33 |
