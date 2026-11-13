"""Tune character BM25 on inner validation sets and evaluate on outer folds."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_experiments as R


K1_VALUES = (0.9, 1.2, 1.5, 2.0)
B_VALUES = (0.4, 0.75, 0.9)
N_VALUES = (3, 4, 5, 6)
TRAIN_SEEDS = (1, 2, 3)
N_FOLDS = 3


def rr10(ranks):
    return np.where(ranks <= 10, 1 / ranks, 0)


def main():
    docs, _ = R.load_data()
    queries = [d["headline"] for d in docs]
    passages = [d["body"] for d in docs]
    n = len(docs)
    permutation = np.random.default_rng(42).permutation(n)
    folds = [np.sort(permutation[i::N_FOLDS]) for i in range(N_FOLDS)]
    default_rr = np.zeros((len(TRAIN_SEEDS), n))
    tuned_rr = np.zeros_like(default_rr)
    selections = []

    for seed_i, seed in enumerate(TRAIN_SEEDS):
        for fold_i, test in enumerate(folds):
            train = np.sort(np.concatenate([folds[j] for j in range(N_FOLDS) if j != fold_i]))
            split = np.random.default_rng(seed).permutation(len(train))
            n_val = len(train) // 4
            val = np.sort(train[split[:n_val]])
            best_score, best_params = -1.0, None

            val_tokens = {ngram: [R.char_ngrams(passages[i], ngram) for i in val]
                          for ngram in N_VALUES}
            val_queries = {ngram: [R.char_ngrams(queries[i], ngram) for i in val]
                           for ngram in N_VALUES}
            for k1 in K1_VALUES:
                for b in B_VALUES:
                    for ngram in N_VALUES:
                        bm25 = R.BM25(val_tokens[ngram], k1=k1, b=b)
                        scores = np.stack([bm25.scores(q) for q in val_queries[ngram]])
                        score = R.metrics_from_ranks(R.rank_of_gold(scores))["MRR@10"]
                        if score > best_score + 1e-9:
                            best_score = score
                            best_params = {"k1": k1, "b": b, "n": ngram}

            test_tokens = [R.char_ngrams(passages[i], best_params["n"]) for i in test]
            test_queries = [R.char_ngrams(queries[i], best_params["n"]) for i in test]
            tuned_bm25 = R.BM25(test_tokens, best_params["k1"], best_params["b"])
            tuned_scores = np.stack([tuned_bm25.scores(q) for q in test_queries])
            tuned_rr[seed_i, test] = rr10(R.rank_of_gold(tuned_scores))

            default_tokens = [R.char_ngrams(passages[i], 4) for i in test]
            default_queries = [R.char_ngrams(queries[i], 4) for i in test]
            default_bm25 = R.BM25(default_tokens, 1.5, 0.75)
            default_scores = np.stack([default_bm25.scores(q) for q in default_queries])
            default_rr[seed_i, test] = rr10(R.rank_of_gold(default_scores))
            selections.append({
                "training_seed": seed,
                "outer_fold": fold_i,
                "validation_mrr10": best_score,
                **best_params,
                "default_test_mrr10": float(default_rr[seed_i, test].mean() * 100),
                "tuned_test_mrr10": float(tuned_rr[seed_i, test].mean() * 100),
            })

    default_mean = default_rr.mean(axis=0)
    tuned_mean = tuned_rr.mean(axis=0)
    diff = tuned_mean - default_mean
    rng = np.random.default_rng(42)
    boot = np.asarray([
        diff[rng.integers(0, n, n)].mean() * 100
        for _ in range(R.N_BOOT)
    ])
    results = {
        "protocol": {
            "outer_folds": N_FOLDS,
            "outer_seed": 42,
            "inner_seeds": list(TRAIN_SEEDS),
            "validation_fraction": 0.25,
            "grid": {"k1": list(K1_VALUES), "b": list(B_VALUES), "n": list(N_VALUES)},
            "tie_break": "first configuration in grid order",
        },
        "default": {"k1": 1.5, "b": 0.75, "n": 4,
                    "MRR@10": float(default_mean.mean() * 100)},
        "validation_tuned": {
            "MRR@10": float(tuned_mean.mean() * 100),
            "difference_from_default": float(diff.mean() * 100),
            "paired_bootstrap_CI95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        },
        "selections": selections,
        "n_selection_counts": {
            str(v): sum(item["n"] == v for item in selections) for v in N_VALUES
        },
    }
    out = Path(R.OUT) / "bm25_validation_grid_results.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    per_query = Path(R.OUT) / "bm25_validation_grid_per_query.csv"
    with per_query.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_index", "default_rr10", "validation_tuned_rr10"])
        writer.writerows((i, f"{default_mean[i]:.6f}", f"{tuned_mean[i]:.6f}") for i in range(n))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
