"""Evaluate fixed equal-weight fusion from the saved cross-fitted checkpoints."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import proposed_experiment as E
import real_experiments as R


def main():
    out_dir = Path(R.OUT)
    docs, _ = R.load_data()
    queries = [R.TOK.normalize_orthography(doc["headline"]) for doc in docs]
    passages = [R.TOK.normalize_orthography(doc["body"]) for doc in docs]
    split_queries = [E.split_text(text) for text in queries]
    split_passages = [E.split_text(text) for text in passages]
    n = len(docs)

    with (out_dir / "proposed_selection_log.json").open(encoding="utf-8") as f:
        log = json.load(f)
    selected = {
        (item["partition_seed"], item["seed"], item["fold"]): item["FINAL_selected"]
        for item in log if "FINAL_selected" in item
    }
    encodings = {
        name: np.load(out_dir / f"emb_{mid.replace('/', '_')}_normalized.npz")
        for name, (mid, *_rest) in R.DENSE_MODELS.items()
    }
    bge = encodings["bge-m3 (dense)"]
    equal_rr = np.zeros((len(E.OUTER_PARTITION_SEEDS), len(E.TRAIN_SEEDS), n))
    equal_r1 = np.zeros_like(equal_rr)
    char_rr = np.zeros((len(E.OUTER_PARTITION_SEEDS), n))
    settings = []

    for partition_i, partition_seed in enumerate(E.OUTER_PARTITION_SEEDS):
        perm = np.random.default_rng(partition_seed).permutation(n)
        folds = [np.sort(perm[i::E.N_FOLDS]) for i in range(E.N_FOLDS)]
        for seed_i, seed in enumerate(E.TRAIN_SEEDS):
            for fold_i, test in enumerate(folds):
                config = selected[(partition_seed, seed, fold_i)]
                mode = config["mode"]
                qsrc, psrc = (queries, passages) if mode == "raw" else (split_queries, split_passages)
                test_candidates = np.broadcast_to(test, (len(test), len(test)))

                c4 = E.candidate_matrix(
                    R.char_ngrams, queries, passages, test, test_candidates)
                char_ranks = R.rank_of_gold(c4)
                char_rr[partition_i, test] = np.where(
                    char_ranks <= 10, 1 / char_ranks, 0)
                bge_scores = E.score_candidates(
                    bge["q"][test], bge["p"], test_candidates)

                checkpoint = (
                    out_dir / "checkpoints" / f"partition_{partition_seed}"
                    / f"fold_{fold_i}" / f"seed_{seed}"
                )
                if not checkpoint.is_dir():
                    raise FileNotFoundError(
                        f"Missing selected model checkpoint {checkpoint}; "
                        "run proposed_experiment.py first."
                    )
                tok = AutoTokenizer.from_pretrained(checkpoint)
                model = AutoModel.from_pretrained(checkpoint).to(E.DEV).eval()
                dense = E.score_candidate_texts(
                    model, tok, qsrc, psrc, test, test_candidates)
                del model, tok
                if E.DEV == "cuda":
                    torch.cuda.empty_cache()

                fused = E.fuse((1 / 3, 1 / 3, 1 / 3), [c4, dense, bge_scores])
                ranks = R.rank_of_gold(fused)
                equal_rr[partition_i, seed_i, test] = np.where(
                    ranks <= 10, 1 / ranks, 0)
                equal_r1[partition_i, seed_i, test] = ranks == 1
                settings.append({
                    "partition_seed": partition_seed,
                    "training_seed": seed,
                    "outer_fold": fold_i,
                    "weights": [1 / 3, 1 / 3, 1 / 3],
                    "selected_encoder_mode": mode,
                    "learning_rate": config["lr"],
                    "epochs": config["epochs"],
                    "test_mrr10": float(np.where(ranks <= 10, 1 / ranks, 0).mean() * 100),
                })
                print(f"partition {partition_seed}, seed {seed}, fold {fold_i}: "
                      f"equal-weight MRR@10 {settings[-1]['test_mrr10']:.2f}", flush=True)

    mean_rr = equal_rr.mean(axis=(0, 1))
    mean_r1 = equal_r1.mean(axis=(0, 1))
    char_mean_rr = char_rr.mean(axis=0)
    with (out_dir / "proposed_per_query_rr.csv").open(encoding="utf-8", newline="") as f:
        saved = list(csv.DictReader(f))
    selected_rr = np.asarray([
        float(row["Proposed: KurdRAG Stage 1 (fine-tuned dense + lexical fusion)"])
        for row in saved
    ])
    diff = mean_rr - selected_rr
    rng = np.random.default_rng(R.SEED)
    boot = np.asarray([
        diff[rng.integers(0, n, n)].mean() * 100 for _ in range(R.N_BOOT)
    ])
    char_diff = mean_rr - char_mean_rr
    char_boot = np.asarray([
        char_diff[rng.integers(0, n, n)].mean() * 100 for _ in range(R.N_BOOT)
    ])
    char_p = 2 * min((char_boot <= 0).mean(), (char_boot >= 0).mean())
    seed_means = equal_rr.mean(axis=(0, 2)) * 100
    result = {
        "protocol": {
            "description": "Fixed one-third weights for char-4gram BM25, selected fine-tuned E5, and zero-shot bge-m3.",
            "outer_folds": E.N_FOLDS,
            "outer_partition_seeds": E.OUTER_PARTITION_SEEDS,
            "training_seeds": E.TRAIN_SEEDS,
            "selected_models": "Loaded from saved selected fold/seed checkpoints.",
        },
        "equal_weight_fusion": {
            "MRR@10": float(mean_rr.mean() * 100),
            "MRR@10_CI95": [
                float(x) for x in np.percentile(
                    np.asarray([
                        mean_rr[rng.integers(0, n, n)].mean() * 100
                        for _ in range(R.N_BOOT)
                    ]),
                    [2.5, 97.5],
                )
            ],
            "Recall@1": float(mean_r1.mean() * 100),
            "seed_mean_mrr10": [float(x) for x in seed_means],
        },
        "difference_from_validation_selected_fusion": {
            "MRR@10_points": float(diff.mean() * 100),
            "paired_bootstrap_CI95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
            "comparison_reference": "saved per-query reciprocal ranks from the same corrected run",
        },
        "difference_from_char4_bm25": {
            "MRR@10_points": float(char_diff.mean() * 100),
            "paired_bootstrap_CI95": [float(x) for x in np.percentile(char_boot, [2.5, 97.5])],
            "paired_bootstrap_p_two_sided": max(float(char_p), 1 / R.N_BOOT),
            "comparison_reference": "same held-out queries averaged across outer partitions",
        },
        "settings": settings,
    }
    (out_dir / "equal_weight_fusion_crossfit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "equal_weight_fusion_crossfit_ranks.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_index", "BM25_char4_MRR10", "equal_weight_MRR10",
            "equal_weight_Recall@1",
        ])
        writer.writerows(zip(range(n), char_mean_rr, mean_rr, mean_r1))
    print(json.dumps(result["equal_weight_fusion"], indent=2))
    print(json.dumps(result["difference_from_validation_selected_fusion"], indent=2))
    print(json.dumps(result["difference_from_char4_bm25"], indent=2))


if __name__ == "__main__":
    main()
