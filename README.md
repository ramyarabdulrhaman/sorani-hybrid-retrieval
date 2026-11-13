# Lexical, Dense and Hybrid Retrieval for Kurdish Sorani

Code, results, tables and figures for a retrieval study of Kurdish Sorani on four collections.

**Question.** Character 4-gram BM25 is a strong Sorani baseline. When does adding dense encoders help?
**Answer.** It depends on how much of the query already appears in the relevant passage. Fusion ties BM25 on high-overlap headline retrieval. On lower-overlap question-to-passage retrieval and on ad hoc search, it gains 3.6–8.1 points.

Only first-stage **retrieval** is evaluated. No reranker, generator, answer quality, factuality or abstention results exist; the repository name reflects the project's original RAG goal.

## Retrievers

| Component | Details |
|---|---|
| Lexical | In-house BM25 (k1 = 1.5, b = 0.75) on words, rule-based clitic stems, KLPT/Jedar stems (Pewan), word-boundary character 4-grams |
| Dense | `intfloat/multilingual-e5-base` (zero-shot or fine-tuned with InfoNCE, τ = 0.05, in-batch + one BM25 hard negative) and `BAAI/bge-m3` (dense mode, zero-shot) |
| Fusion | Per-query z-score fusion `w1·z(BM25-c4) + w2·z(E5) + w3·z(bge-m3)`, weights validation-tuned on a 0.1 grid or fixed at 1/3; reciprocal-rank fusion (k = 60) as a baseline |
| Protocol | News: 3-fold cross-fitting × 2 partitions (seeds 42, 2025) × 5 training seeds = **30 saved models**; validation ranked against 164 candidates like test. Paired (cluster) bootstrap, Holm correction, post-hoc TOST and power analysis |

## Collections and headline results

| Collection | Task | Size | Median overlap* | Fused − char-4gram BM25 (95% CI) |
|---|---|---|---|---|
| News headlines | headline → article | 492 pairs | 0.80 | **+0.02** MRR@10 [−0.34, +0.42] (tie, 30 runs) |
| KNQAD | question → paragraph | 3,061 test Q / 7,271 paragraphs | 0.38 | **+6.52** news models [+5.77, +7.28]; **+8.08** KNQAD-trained [+7.23, +8.95] |
| Belebele (ckb_Arab) | question → passage | 447 test Q / 488 passages | 0.18 | **+3.64** MRR@10 [+1.93, +5.45], better in 30/30 runs |
| Pewan | ad hoc search | 22 topics / 115,340 docs | – | **+3.84** nDCG@10 [+0.48, +7.38], no Pewan tuning |

\*Fraction of query content-word types (length ≥ 3) found in the relevant passage.

**Other findings**
- Fixed equal-weight fusion matches validation-tuned fusion on every collection.
- Fine-tuning on 328 news pairs adds nothing significant inside fusion. Fine-tuning on 4,362 KNQAD pairs adds **+1.35** MRR@10 over zero-shot fusion (Holm p < 0.005).
- Within Belebele, the gain is +6.92 in the lowest-overlap tercile and +0.47 in the highest.
- Stemming (rule-based, KLPT, Jedar) does not beat character 4-grams.

### News headlines (cross-fitted; `results/proposed_results.json`)

| System | R@1 | MRR@10 [95% CI] |
|---|---|---|
| BM25 word | 96.34 | 97.42 [96.17, 98.52] |
| BM25 char-4gram | 98.17 | 98.95 [98.26, 99.50] |
| bge-m3 zero-shot | 91.67 | 94.06 [92.15, 95.71] |
| RRF char-4gram + bge-m3 | 95.12 | 97.21 [96.14, 98.16] |
| Zero-shot fusion (tuned) | 97.83 | 98.78 [98.15, 99.34] |
| E5 fine-tuned | 92.34 | 95.20 [93.92, 96.44] |
| **Validation-tuned fused retriever** | 98.19 | 98.98 [98.39, 99.44] |
| Zero-shot equal-weight fusion | 98.37 | 99.10 [98.52, 99.59] |
| Fine-tuned equal-weight fusion | 98.58 | 99.21 [98.64, 99.68] |

The fused retriever is significantly better than word/stem BM25, all dense encoders, RRF and fine-tuned E5 (Holm p ≤ 0.004). It is not significantly better than char-4gram BM25 (+0.02) or zero-shot fusion (+0.20). TOST bounds these differences within ±1 point; the margins are post hoc.

### KNQAD (`results/knqad/knqad_results.json`)

| System | MRR@10 |
|---|---|
| BM25 word / stems / char-4gram | 24.99 / 25.62 / 33.27 |
| E5 / bge-m3 zero-shot | 16.18 / 24.37 |
| RRF char-4gram + bge-m3 | 32.39 |
| Zero-shot fusion (tuned / equal) | 40.01 / 39.95 |
| News-trained fused retriever (30 models) | 39.80 |
| **KNQAD-trained fused retriever (5 seeds)** | **41.36** |

### Belebele (`results/belebele/belebele_30model_results.json`)

The 30 news models are applied unchanged, with fusion weights selected on 453 validation questions. The fused retriever reaches **82.75** MRR@10 (run SD 0.34) vs 79.10 for char-4gram BM25. Compared with zero-shot equal-weight fusion it gains +1.85 (Holm p = 0.046); compared with tuned zero-shot fusion (+1.33) and fine-tuned equal-weight fusion (+0.72), the differences are not significant.

### Pewan (`results/pewan_dense_fusion.json`, `results/pewan_results.json`)

Five news models are used, with news-selected weights and no Pewan tuning. nDCG@10 is **61.55** for the fused retriever vs 57.71 for char-4gram BM25 (Holm p = 0.048). Zero-shot and fine-tuned equal-weight fusion reach 63.60 and 63.01. MAP differences are not significant.

## Repository layout

```
code/                                        experiments, analyses, table and figure generators
results/                                     JSON/CSV outputs (sources of truth for every table)
tables/                                      generated CSV/Markdown tables
figures/                                     generated figures
```

Included: Belebele (`data/raw/belebele_ckb_Arab.jsonl`, CC BY-SA 4.0) and KNQAD (`data/external/knqad/knqad.xlsx`, CC BY 4.0). Not included: model checkpoints (30 × ~1.05 GB, available from the authors on request), embedding caches, the news source file and Pewan, which must be obtained from their sources (see `data/README.md`).

## Data sources

| Data | Source | Licence / status |
|---|---|---|
| News headlines | [abdulhade/KurdishTextCorpus](https://huggingface.co/datasets/abdulhade/KurdishTextCorpus), revision `2675e164dda260088c729fded823e34b169f7424`, first 1,250,000 bytes (SHA-256 `7e21d6a4…55aef3`; check with `code/verify_news_source.py`) | Card says AFL-3.0; publisher rights unconfirmed, so it is **not redistributed** |
| Belebele | [facebook/belebele](https://huggingface.co/datasets/facebook/belebele), `ckb_Arab` (pinned revision in `results/belebele/README.md`) | CC BY-SA 4.0, **included** |
| KNQAD | [Mendeley Data, doi:10.17632/tc28knsfsn.1](https://data.mendeley.com/datasets/tc28knsfsn/1), `data/external/knqad/knqad.xlsx` (SHA-256 `076F3940…E88730`) | CC BY 4.0, **included** |
| Pewan | [github.com/klpp/pewan](https://github.com/klpp/pewan), extract to `data/external/pewan/Pewan` | No licence declared, so it is **not redistributed** |
| Stemmers | [KLPT 0.1.7](https://github.com/sinaahmadi/klpt); [Jedar](https://github.com/klpp/codes/tree/master/stemming) | |
| Encoders | [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base), [bge-m3](https://huggingface.co/BAAI/bge-m3) (revisions pinned in the code) | |

## Quick start on the included data

```npip install -r requirements.txt
python code/knqad_experiment.py          # KNQAD; its news-transfer step needs the 30 news checkpoints
python code/belebele_only_experiment.py   # Belebele-only training and evaluation
```n
Zero-shot baselines and fusion on KNQAD and Belebele run on the included data. Results for the news-trained models need the news file (to retrain them) or the checkpoints.

## Licence

Code: MIT (see `LICENSE`). Data files keep their original licences (`data/README.md`).

## Reproduce

Requires Python 3.12 and `pip install -r requirements.txt`. Training needs a CUDA GPU (8 GB was enough).

```
# News headline task (needs the verified news file at data/raw/kurdish_news_corpus_source.txt)
python code/real_experiments.py ; python code/real_analysis.py          # full-collection baselines
python code/proposed_experiment.py                                       # 30 cross-fitted runs, saves checkpoints (~1 day)
python code/equal_weight_fusion.py ; python code/zero_shot_equal_weight.py
python code/equivalence_power_analysis.py ; python code/proposed_analysis.py
python code/task_dependence_analysis.py ; python code/bge_m3_input_fairness.py ; python code/bm25_validation_grid.py

# Transfer
python code/belebele_experiment.py ; python code/belebele_analysis.py    # primary Belebele adaptation
python code/belebele_30model.py                                          # 30 saved news models on Belebele
python code/belebele_robustness.py ; python code/belebele_only_experiment.py   # exploratory
python code/pewan_evaluation.py ; python code/pewan_dense_fusion.py      # Pewan lexical and dense/fusion
python code/knqad_experiment.py                                          # KNQAD (also fine-tunes 5 E5 seeds)

# Figures
python code/make_fig1_summary.py ; python code/make_fig_overview.py
```

Every reported number is read from a file in `results/`; none is typed by hand. Bootstrap seed 42. BM25 is deterministic. GPU training uses fp16 and is not bit-wise deterministic.

