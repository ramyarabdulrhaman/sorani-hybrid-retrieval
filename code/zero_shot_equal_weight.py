"""Evaluate fixed equal-weight fusion of zero-shot cross-fitted components."""
import csv
import json
from pathlib import Path

import numpy as np

import proposed_experiment as E
import real_experiments as R


def paired_summary(differences, rng):
    n = len(differences)
    samples = np.asarray([
        differences[rng.integers(0, n, n)].mean() * 100
        for _ in range(R.N_BOOT)
    ])
    p = 2 * min((samples <= 0).mean(), (samples >= 0).mean())
    return {
        "MRR@10_difference_points": float(differences.mean() * 100),
        "paired_bootstrap_CI95": [
            float(x) for x in np.percentile(samples, [2.5, 97.5])
        ],
        "paired_bootstrap_p_two_sided": max(float(p), 1 / R.N_BOOT),
    }


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def main():
    out_dir = Path(R.OUT)
    docs, _ = R.load_data()
    queries = [R.TOK.normalize_orthography(doc["headline"]) for doc in docs]
    passages = [R.TOK.normalize_orthography(doc["body"]) for doc in docs]
    n = len(docs)

    e5 = np.load(out_dir / "emb_intfloat_multilingual-e5-base_normalized.npz")
    bge = np.load(out_dir / "emb_BAAI_bge-m3_normalized.npz")
    bm25_rr = np.zeros((len(E.OUTER_PARTITION_SEEDS), n))
    fusion_rr = np.zeros_like(bm25_rr)
    fusion_r1 = np.zeros_like(bm25_rr)

    for partition_i, partition_seed in enumerate(E.OUTER_PARTITION_SEEDS):
        perm = np.random.default_rng(partition_seed).permutation(n)
        folds = [np.sort(perm[i::E.N_FOLDS]) for i in range(E.N_FOLDS)]
        for test in folds:
            candidates = np.broadcast_to(test, (len(test), len(test)))
            lexical = E.candidate_matrix(
                R.char_ngrams, queries, passages, test, candidates)
            e5_scores = E.score_candidates(e5["q"][test], e5["p"], candidates)
            bge_scores = E.score_candidates(bge["q"][test], bge["p"], candidates)
            fused = E.fuse((1 / 3, 1 / 3, 1 / 3), [lexical, e5_scores, bge_scores])

            bm25_ranks = R.rank_of_gold(lexical)
            fused_ranks = R.rank_of_gold(fused)
            bm25_rr[partition_i, test] = np.where(
                bm25_ranks <= 10, 1 / bm25_ranks, 0)
            fusion_rr[partition_i, test] = np.where(
                fused_ranks <= 10, 1 / fused_ranks, 0)
            fusion_r1[partition_i, test] = fused_ranks == 1

    bm25_mean_rr = bm25_rr.mean(axis=0)
    fusion_mean_rr = fusion_rr.mean(axis=0)
    fusion_mean_r1 = fusion_r1.mean(axis=0)

    rng = np.random.default_rng(R.SEED)
    mrr_boot = np.asarray([
        fusion_mean_rr[rng.integers(0, n, n)].mean() * 100
        for _ in range(R.N_BOOT)
    ])
    output = {
        "protocol": {
            "description": "Fixed one-third weights for cross-fitted char-4gram BM25, zero-shot E5, and zero-shot BGE-M3; scores are z-scored per query before fusion.",
            "outer_folds": E.N_FOLDS,
            "outer_partition_seeds": E.OUTER_PARTITION_SEEDS,
            "queries": n,
            "candidate_passages_per_query": n // E.N_FOLDS,
        },
        "zero_shot_equal_weight_fusion": {
            "MRR@10": float(fusion_mean_rr.mean() * 100),
            "MRR@10_CI95": [
                float(x) for x in np.percentile(mrr_boot, [2.5, 97.5])
            ],
            "Recall@1": float(fusion_mean_r1.mean() * 100),
        },
        "difference_from_char4_bm25": paired_summary(fusion_mean_rr - bm25_mean_rr, rng),
    }
    ranks_path = out_dir / "equal_weight_zero_shot_crossfit_ranks.csv"
    with ranks_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "query_index", "BM25_char4_MRR10",
            "zero_shot_equal_weight_MRR10", "zero_shot_equal_weight_Recall@1",
        ])
        writer.writerows(zip(range(n), bm25_mean_rr, fusion_mean_rr, fusion_mean_r1))

    trained_path = out_dir / "equal_weight_fusion_crossfit_ranks.csv"
    trained_result_path = out_dir / "equal_weight_fusion_crossfit.json"
    if trained_path.exists():
        with trained_path.open(encoding="utf-8", newline="") as file:
            trained_rows = list(csv.DictReader(file))
        if len(trained_rows) != n or any(
            int(row["query_index"]) != i for i, row in enumerate(trained_rows)
        ):
            raise ValueError(f"Unexpected per-query ranks in {trained_path}")
        trained_rr = np.asarray([
            float(row["equal_weight_MRR10"]) for row in trained_rows
        ])
        trained_char_rr = np.asarray([
            float(row["BM25_char4_MRR10"]) for row in trained_rows
        ])
        output["fine_tuned_equal_weight_minus_zero_shot_equal_weight"] = (
            paired_summary(trained_rr - fusion_mean_rr, rng)
        )
        trained_result = json.loads(
            trained_result_path.read_text(encoding="utf-8")
        )
        if trained_result["protocol"].get("outer_partition_seeds") != E.OUTER_PARTITION_SEEDS:
            raise ValueError(
                f"Stale trained equal-weight results in {trained_result_path}; "
                "run equal_weight_fusion.py for the current partitions first."
            )
        trained_result["difference_from_char4_bm25"].update(
            paired_summary(trained_rr - trained_char_rr, rng)
        )
        trained_result_path.write_text(
            json.dumps(trained_result, indent=2) + "\n", encoding="utf-8"
        )
        comparisons = {
            "zero_shot_equal_weight_minus_char4": output["difference_from_char4_bm25"],
            "fine_tuned_equal_weight_minus_char4": trained_result["difference_from_char4_bm25"],
            "fine_tuned_equal_weight_minus_zero_shot_equal_weight": output[
                "fine_tuned_equal_weight_minus_zero_shot_equal_weight"
            ],
        }
        raw_p = np.asarray([
            comparison["paired_bootstrap_p_two_sided"]
            for comparison in comparisons.values()
        ])
        for comparison, adjusted_p in zip(
            comparisons.values(), holm_adjust(raw_p)
        ):
            comparison["holm_adjusted_p_two_sided"] = float(adjusted_p)
        output["equal_weight_paired_comparisons"] = comparisons

    (out_dir / "equal_weight_zero_shot_crossfit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
