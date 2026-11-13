"""Evaluate one-third score fusion on the primary Belebele test split."""
import csv
import json

import numpy as np

import belebele_experiment as B


def rr10(ranks):
    return np.where(ranks <= 10, 1 / ranks, 0)


def main():
    rows, passages, gold = B.load_dataset()
    queries = [row["question"] for row in rows]
    permutation = np.random.default_rng(B.SEED).permutation(len(passages))
    val_groups = np.sort(permutation[:len(permutation) // 2])
    test = ~np.isin(gold, val_groups)

    bm25 = B.R.BM25([B.R.char_ngrams(text) for text in passages])
    char_scores = np.stack([bm25.scores(B.R.char_ngrams(text)) for text in queries])
    bge_scores = B.dense("bge", queries, passages)
    e5_scores, _ = B.train_news_and_encode(queries, passages, seed=1)
    equal_scores = B.weighted([char_scores, e5_scores, bge_scores], (1 / 3, 1 / 3, 1 / 3))

    equal_ranks = B.ranks(equal_scores, gold)
    char_ranks = B.ranks(char_scores, gold)
    equal_rr = rr10(equal_ranks[test])
    char_rr = rr10(char_ranks[test])
    groups = gold[test]

    with (B.OUT / "ranks.csv").open(encoding="utf-8", newline="") as f:
        saved = [row for row in csv.DictReader(f) if row["split"] == "test"]
    selected_rr = np.asarray([
        1 / int(row["Proposed cross-domain variant"])
        if int(row["Proposed cross-domain variant"]) <= 10 else 0
        for row in saved
    ])
    if len(saved) != int(test.sum()):
        raise ValueError("Saved test ranks do not match the primary test split")

    result = {
        "protocol": {
            "test_queries": int(test.sum()),
            "candidate_passages": len(passages),
            "weights_char4_finetuned_e5_bge": [1 / 3, 1 / 3, 1 / 3],
            "model": "news-trained E5, raw text, seed 1, learning rate 1e-5, two epochs",
            "source": "Pewan is not used in this experiment",
        },
        "equal_weight_fusion": {
            "MRR@10": float(equal_rr.mean() * 100),
            "Recall@1": float((equal_ranks[test] == 1).mean() * 100),
        },
        "BM25_char4": {"MRR@10": float(char_rr.mean() * 100)},
        "difference_from_char4": {
            "MRR@10_points": float((equal_rr - char_rr).mean() * 100),
            "passage_cluster_CI95": B.cluster_ci(equal_rr, char_rr, groups),
        },
        "difference_from_validation_selected_fusion": {
            "MRR@10_points": float((equal_rr - selected_rr).mean() * 100),
            "passage_cluster_CI95": B.cluster_ci(equal_rr, selected_rr, groups),
        },
    }
    (B.OUT / "equal_weight_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
