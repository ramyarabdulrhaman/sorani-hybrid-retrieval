# Data

| Path | Included | Source | Licence |
|---|---|---|---|
| `raw/belebele_ckb_Arab.jsonl` | yes | Belebele, Central Kurdish (`ckb_Arab`) split, [facebook/belebele](https://huggingface.co/datasets/facebook/belebele); pinned revision in `results/belebele/README.md` | CC BY-SA 4.0 (redistributed unchanged under the same licence) |
| `external/knqad/knqad.xlsx` | yes | KNQAD, Saeed (2024), Mendeley Data v1, [doi:10.17632/tc28knsfsn.1](https://data.mendeley.com/datasets/tc28knsfsn/1); SHA-256 `076F394097C7ED2E400E15A55D0ED4078BA8480FB15110B98A75654EE9E88730` | CC BY 4.0 (redistributed unchanged) |
| `raw/kurdish_news_corpus_source.txt` | **no** | First 1,250,000 bytes (to the last complete line) of [abdulhade/KurdishTextCorpus](https://huggingface.co/datasets/abdulhade/KurdishTextCorpus) at revision `2675e164dda260088c729fded823e34b169f7424`; check with `python code/verify_news_source.py` | Card says AFL-3.0, but the original publishers' rights are unconfirmed |
| `external/pewan/Pewan/` | **no** | [github.com/klpp/pewan](https://github.com/klpp/pewan): extract `Pewan.zip`, then extract `Sorani.zip` into `Pewan/Corpora/Sorani/docs` (115,340 files) | No licence declared |

## How the collections are used

- **News headlines → article:** 492 pairs after cleaning (`code/real_experiments.py`). Each headline is the query and its body is the only relevant passage.
- **KNQAD question → paragraph:** 14,985 questions over 7,271 unique raw paragraphs (`code/knqad_experiment.py`). The leading `پرسیار:` label is removed. Splits are paragraph-grouped 60/20/20 with seed 42.
- **Belebele question → passage:** 900 questions over 488 passages. Each question's linked passage is the only relevant one; answer options are not used.
- **Pewan ad hoc search:** 22 Sorani topic titles over 115,340 documents, with 925 relevance judgments (`code/pewan_evaluation.py`).

If you use KNQAD or Belebele, please cite their original authors.
