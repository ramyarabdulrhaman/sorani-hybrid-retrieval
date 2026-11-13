# Belebele Sorani retrieval experiment

This is a research adaptation of the Belebele `ckb_Arab` reading-comprehension
test to question-to-passage retrieval. It does not evaluate multiple-choice
answer accuracy. The given passage is treated as the only relevant passage, but
other passages were not judged for relevance. This can underestimate retrieval
quality. It is not a new human-annotated retrieval benchmark.

The 488 unique passages are split into 244 validation and 244 test groups using
NumPy's seeded generator (seed 42). Questions associated with a passage stay
in the same group. Every query is ranked against all 488 passages, including
those associated with the other partition. The validation partition is
created from the public Belebele test set, not an official development set.
Fusion weights are selected by validation MRR@10 and then fixed for the test
partition. Test uncertainty uses 5,000 paired passage-cluster bootstrap
resamples; passages may have multiple questions. Gold-score ties receive the
pessimistic rank, and RRF uses the original project's tie ordering.

The proposed **cross-domain variant** uses an E5 model retrained on all 492
news headline/article pairs using the most frequently selected configuration
from the earlier news study (raw text, learning rate 1e-5, two epochs, seed 1).
It is not the original experiment's cross-fitted checkpoint, which was not
saved. Only this retrained component is evaluated; the original checkpoint
cannot be reconstructed from results alone. Fusion requires its weight to be
at least 0.1. The zero-shot fusion control has no such constraint. Neither
training on news nor validation-tuned fusion is zero-shot with respect to the
entire retrieval pipeline. BM25 and pretrained encoders are zero-shot.

The Belebele and news passages and queries have no exact matches after the
project's word normalization, but this does not exclude near-duplicates or
pretraining exposure. The news excerpt matches a pinned prefix of
`abdulhade/KurdishTextCorpus`, whose card states AFL-3.0; the original
publishers' rights and permission remain unconfirmed. The corpus is labeled
Central Kurdish, but some news passages appear to be Northern Kurdish and were
not independently classified. Using Belebele does not resolve those issues.

## Exploratory sensitivity follow-up

After inspecting the first test results, we retrained on the same news data with
seeds 1, 2 and 3. A different, passage-grouped three-fold partition (seed 2026)
assigns every question to one test fold; the other two folds' questions tune
fusion weights separately for each model and held-out fold. The full 488-passage
collection remains the candidate pool for all queries. We aggregate each
question's score over the three models and report all 900 questions. The
bootstrap resamples passage groups, treating the fitted models, selected
weights and fold assignment as fixed; it does not measure variance across
partitions or independently sampled training runs.

This is not an independent test: the public Belebele test set and the first
results were inspected before choosing this follow-up, and questions used for
testing in one fold are used for tuning in other folds. No fold tunes on its
own test questions. No relevance judgments have been added. Results are saved in
`robustness_results.json` and `robustness_ranks.csv`.

## Exploratory Belebele-only follow-up

After inspecting the earlier analyses, seed 77 splits the 488 passage groups
into 244 training (452 questions), 122 validation (229) and 122 test (219).
The first question for each training passage forms 244 training pairs. The
E5 model is trained for two epochs with learning rate 1e-5, seeds 1, 2 and 3,
using word-BM25 hard negatives drawn from training passages. The same full
488-passage pool is ranked for validation and test. Both variants select fusion
weights on validation only using a nonnegative 0.1 grid; the trained fusion
requires its E5 component to have at least 0.1 weight. Test metrics are
averaged across seeds. The bootstrap uses 5,000 passage-cluster resamples,
seed 42, after averaging reciprocal ranks across seeds; it conditions on the
models, weights and split, and does not capture partition or model uncertainty.

This experiment does not use news data but reuses previously inspected
Belebele questions, so it is exploratory, not independent confirmation.
There are no exact cross-group question matches; near-duplicates and encoder
pretraining exposure are not excluded. Only linked passages are labeled
relevant; alternative passages were not judged.
`belebele_only_results.json` and `belebele_only_ranks.csv` contain results and
per-query ranks. `python code/belebele_only_experiment.py` reproduces the run;
`python code/belebele_only_analysis.py` verifies ranks and generates Table 12.
It requires the pinned Belebele file but not the news corpus.

## Data and reproduction

Download `data/ckb_Arab.jsonl` from the pinned revision
`7899cdfa4e1e0d733fd77c848e2c273cb1d32be2` at
<https://huggingface.co/datasets/facebook/belebele> into
`data/raw/belebele_ckb_Arab.jsonl`. The script checks its SHA-256:
`b7aa4cb5c9ee985c1293615d91a9c25419e735ac83b98068a31830413a8cb053`.
The pinned download URL is
<https://huggingface.co/datasets/facebook/belebele/resolve/7899cdfa4e1e0d733fd77c848e2c273cb1d32be2/data/ckb_Arab.jsonl>.
The pinned dataset file is included at `data/raw/belebele_ckb_Arab.jsonl`.
Its SHA-256 is
`b7aa4cb5c9ee985c1293615d91a9c25419e735ac83b98068a31830413a8cb053`.
The original dataset is licensed CC BY-SA 4.0; see the
[dataset card](https://huggingface.co/datasets/facebook/belebele) and
[license](https://github.com/facebookresearch/belebele/blob/main/LICENSE_CC-BY-SA4.0).
Please comply with attribution, license-notice and share-alike requirements
when redistributing or adapting the data.

From the project root, install the project's existing requirements and run:

```
python code/belebele_experiment.py
python code/belebele_analysis.py
python code/belebele_robustness.py
python code/belebele_robustness_analysis.py
```

The experiment pins the two pretrained model revisions and writes `results.json`
and `ranks.csv` in this directory. The analysis verifies MRR@10 against the saved
ranks, produces the proposed-versus-control comparison and generates Table 10.
Fine-tuning requires substantial resources;
a CUDA GPU is recommended. Floating-point results may vary across hardware.
The source news file is required for the proposed variant.

Dataset citation: Bandarkar et al., "The Belebele Benchmark: a Parallel
Reading Comprehension Dataset in 122 Language Variants," ACL 2024,
DOI [10.18653/v1/2024.acl-long.44](https://doi.org/10.18653/v1/2024.acl-long.44).
