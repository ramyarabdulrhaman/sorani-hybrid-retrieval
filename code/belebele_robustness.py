"""Exploratory passage-grouped three-fold sensitivity analysis for Belebele."""
import csv
import json
import platform
from pathlib import Path

import numpy as np
import torch
import transformers

import belebele_experiment as B

TRAIN_SEEDS = (1, 2, 3)
SPLIT_SEED = 2026
FOLDS = 3
BASELINE = "BM25 char-4gram"
CONTROL = "Validation-tuned zero-shot fusion"
PROPOSED = "Proposed cross-domain variant"


def group_folds(gold, n_passages):
    perm = np.random.default_rng(SPLIT_SEED).permutation(n_passages)
    passage_folds = np.empty(n_passages, dtype=np.int32)
    for fold, groups in enumerate(np.array_split(perm, FOLDS)):
        passage_folds[groups] = fold
    return passage_folds[gold]


def evaluate(c4, e5, bge, fine_tuned, gold, fold_ids):
    n = len(gold)
    seed_ranks = {name: np.empty((len(TRAIN_SEEDS), n), dtype=np.int32)
                  for name in (BASELINE, CONTROL, PROPOSED)}
    selections = []
    for seed_index, seed in enumerate(TRAIN_SEEDS):
        for fold in range(FOLDS):
            valid, test = fold_ids != fold, fold_ids == fold
            zero = [c4, e5, bge]
            trained = [c4, fine_tuned[seed], bge]
            zw, zv = B.select_weights([s[valid] for s in zero], gold[valid])
            pw, pv = B.select_weights([s[valid] for s in trained], gold[valid], 1)
            for name, scores in [
                (BASELINE, c4[test]),
                (CONTROL, B.weighted([s[test] for s in zero], zw)),
                (PROPOSED, B.weighted([s[test] for s in trained], pw)),
            ]:
                seed_ranks[name][seed_index, test] = B.ranks(scores, gold[test])
            selections.append({"training_seed": seed, "test_fold": fold,
                               "validation_questions": int(valid.sum()), "test_questions": int(test.sum()),
                               "zero_shot_weights": zw, "zero_shot_validation_mrr10": zv,
                               "proposed_weights": pw, "proposed_validation_mrr10": pv})
    return seed_ranks, selections


def summarize(seed_ranks, gold):
    rr = {name: np.where(ranks <= 10, 1.0 / ranks, 0)
          for name, ranks in seed_ranks.items()}
    outcomes = {}
    for name, ranks in seed_ranks.items():
        per_seed = rr[name].mean(axis=1) * 100
        outcomes[name] = {
            "MRR@10": float(per_seed.mean()),
            "MRR@10_per_seed": per_seed.tolist(),
            "Recall@1": float((ranks == 1).mean() * 100),
            "Recall@5": float((ranks <= 5).mean() * 100),
        }
    comparisons = {}
    for baseline in (BASELINE, CONTROL):
        a = rr[PROPOSED].mean(axis=0)
        b = rr[baseline].mean(axis=0)
        comparisons[baseline] = {
            "mean_MRR@10_difference": float((a - b).mean() * 100),
            "CI95_passage_cluster_bootstrap": B.cluster_ci(a, b, gold),
            "difference_per_training_seed": ((rr[PROPOSED] - rr[baseline]).mean(axis=1) * 100).tolist(),
        }
    return outcomes, comparisons


def main():
    rows, passages, gold = B.load_dataset()
    queries = [r["question"] for r in rows]
    print("Computing lexical and zero-shot components", flush=True)
    index = B.R.BM25([B.R.char_ngrams(p) for p in passages])
    c4 = np.stack([index.scores(B.R.char_ngrams(q)) for q in queries])
    e5 = B.dense("e5", queries, passages)
    bge = B.dense("bge", queries, passages)
    fine_tuned, news_stats = {}, None
    for seed in TRAIN_SEEDS:
        fine_tuned[seed], news_stats = B.train_news_and_encode(queries, passages, seed)
    fold_ids = group_folds(gold, len(passages))
    seed_ranks, selections = evaluate(c4, e5, bge, fine_tuned, gold, fold_ids)
    outcomes, comparisons = summarize(seed_ranks, gold)
    result = {
        "protocol": {
            "purpose": "Exploratory robustness check after observing the earlier test results; not an untouched independent test.",
            "dataset_revision": B.DATA_REV, "dataset_sha256": B.DATA_SHA,
            "news_training_source": news_stats, "training_seeds": TRAIN_SEEDS,
            "training_config": {"lr": 1e-5, "epochs": 2, "text": "raw"},
            "split_seed": SPLIT_SEED, "folds": FOLDS, "passages": len(passages),
            "questions": len(queries), "candidate_passages_per_query": len(passages),
            "selection": "Per fold and training seed, tune fusion weights on all questions in the other two passage groups; held-out group only for scoring.",
            "grid": "nonnegative weights on 0.1 grid; proposed has fine-tuned component >= 0.1; control unconstrained",
            "uncertainty": "Paired 5000-resample passage-cluster bootstrap on seed-averaged per-query reciprocal rank; conditional on folds, selected weights, and three trained models.",
            "caveats": "One grouped three-fold partition; public test data previously inspected; incomplete relevance labels; news licence unconfirmed; pretraining exposure not ruled out.",
            "model_revisions": B.MODELS, "device": B.P.DEV,
            "torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
        },
        "fold_selection": selections, "systems": outcomes, "comparisons": comparisons,
    }
    out = Path(B.OUT)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "robustness_ranks.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "passage_index", "test_fold", "training_seed", *seed_ranks])
        for j, seed in enumerate(TRAIN_SEEDS):
            for i in range(len(rows)):
                writer.writerow([i, int(gold[i]), int(fold_ids[i]), seed,
                                 *[int(ranks[j, i]) for ranks in seed_ranks.values()]])
    (out / "robustness_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"systems": outcomes, "comparisons": comparisons}, indent=2), flush=True)


if __name__ == "__main__":
    main()
