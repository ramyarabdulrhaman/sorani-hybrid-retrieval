| System                              |   Scoring time (ms/query) |
|:------------------------------------|--------------------------:|
| BM25 (word)                         |                     0.307 |
| BM25 (morpheme stems)               |                     0.309 |
| BM25 (char 4-gram)                  |                     2.516 |
| multilingual-e5-base                |                     0.002 |
| bge-m3 (dense)                      |                     0.002 |
| RRF (BM25 stems + e5-base)          |                     0.027 |
| RRF (BM25 stems + bge-m3)           |                     0.03  |
| RRF (BM25 stems + e5-base + bge-m3) |                     0.041 |
