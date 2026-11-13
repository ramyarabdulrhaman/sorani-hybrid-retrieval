# Limitations and Reproducibility

## Limitations
1. Only first-stage retrieval is evaluated. No reranker, generator, answer quality, factuality, hallucination or abstention results exist.
2. News headline labels follow article structure (headline → its own body). They are not human judgments, and the task has very high query–passage overlap (median 0.80), so char-4gram BM25 is near saturation. The news file is a prefix, not a random sample. Its dialect mix is unknown (some passages look Northern Kurdish), and publisher redistribution rights are unconfirmed.
3. News cross-fitting uses 328 training pairs per fold and 164-passage test folds. Validation reaches 100% MRR@10 in 15 of 30 runs, so selected fusion weights are unstable (fine-tuned weight 0.0–0.7). The sign of the tuned-fusion − BM25 difference differs between the two partitions.
4. Belebele is a reading-comprehension test adapted to retrieval: only the linked passage is relevant, and validation is drawn from the public test split. The 30-model transfer conditions on one passage split. The exploratory multi-seed and Belebele-only analyses reuse inspected test data.
5. KNQAD questions were written by annotators reading the paragraph. Only the source paragraph is judged relevant, so other paragraphs about the same event count as misses. One paragraph-grouped partition (seed 42) is used, and the news models' fusion weights are re-selected on KNQAD validation.
6. Pewan has only 22 topics and incomplete judgments. Dense runs use five news models and 512-piece document truncation. The Jedar suffix file is taken from Pewan because the original is missing upstream. Pewan declares no licence and is not redistributed.
7. The overlap analyses are post hoc and correlational. Overlap is confounded with question type, passage length and candidate-set size, and it does not order the gain within KNQAD.
8. TOST equivalence margins were chosen after inspection and are descriptive.
9. bge-m3 is used in dense mode only; sparse/multi-vector modes and larger encoders are not evaluated.

## Reproducibility
- Every table and figure is generated from files in `results/` by scripts in `code/`; no value is typed by hand.
- Sources of truth: `results/proposed_results.json` and `proposed_selection_log.json` (news, 30 runs); `results/belebele/belebele_30model_results.json`; `results/pewan_results.json` and `pewan_dense_fusion.json`; `results/knqad/knqad_results.json`; `results/task_dependence.json`; `results/equivalence_power.json`.
- Seeds: news partitions 42 and 2025, training seeds 1–5. Bootstrap seed 42 (2,000 query/topic resamples; 5,000 cluster resamples for Belebele and KNQAD). BM25 is deterministic.
- Environment: Windows 11, Python 3.12, transformers 5.x, torch 2.11 + CUDA 12.8, one 8 GB RTX 2080 (fp16; not bit-wise deterministic).
- The selection log contains duplicate FINAL entries for runs that were resumed after interruptions. Analyses keep the last entry per partition/fold/seed.
- The 30 news checkpoints (~31.5 GB) and the 5 KNQAD checkpoints are not committed; they are available from the authors on request.
