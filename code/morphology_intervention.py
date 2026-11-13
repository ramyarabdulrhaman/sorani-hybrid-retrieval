"""Controlled clitic-splitting intervention for the news retrieval task.

Raw and split conditions use the same training pairs, folds, initialization
seeds, optimizer settings, and held-out candidate pools. Results are exploratory
because the historical corpus and retrieval outcomes have already been inspected.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import proposed_experiment as E
import real_experiments as R

SEEDS = (1, 2, 3)
PARTITION_SEED = 42
BOOTSTRAP_RESAMPLES = 5000
LR = 1e-5
EPOCHS = 2
OUT = Path(R.OUT)
RAW, SPLIT = "raw", "clitic_split"


def paired_query_bootstrap(differences, seed=42):
    differences = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.asarray([
        differences[rng.integers(0, len(differences), len(differences))].mean() * 100
        for _ in range(BOOTSTRAP_RESAMPLES)
    ])
    probability = 2 * min((samples <= 0).mean(), (samples >= 0).mean())
    return {
        "difference_points": float(differences.mean() * 100),
        "CI95": [float(x) for x in np.percentile(samples, [2.5, 97.5])],
        "paired_bootstrap_p_two_sided": max(float(probability), 1 / BOOTSTRAP_RESAMPLES),
    }


def tokenize_stats(tokenizer, texts):
    pieces_per_word = []
    piece_counts = []
    word_counts = []
    for text in texts:
        words = R.words(text)
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        piece_counts.append(len(tokens))
        word_counts.append(len(words))
        if words:
            pieces_per_word.extend(
                len(tokenizer(word, add_special_tokens=False)["input_ids"])
                for word in words
            )
    ratios = np.asarray(pieces_per_word, dtype=float)
    return {
        "mean_pieces_per_word": float(ratios.mean()) if len(ratios) else 0.0,
        "median_pieces_per_word": float(np.median(ratios)) if len(ratios) else 0.0,
        "p95_pieces_per_word": float(np.percentile(ratios, 95)) if len(ratios) else 0.0,
        "mean_pieces_per_text": float(np.mean(piece_counts)),
        "mean_words_per_text": float(np.mean(word_counts)),
        "texts": len(texts),
    }


def run():
    docs, source_stats = R.load_data()
    queries = [R.TOK.normalize_orthography(doc["headline"]) for doc in docs]
    passages = [R.TOK.normalize_orthography(doc["body"]) for doc in docs]
    split_queries = [E.split_text(text) for text in queries]
    split_passages = [E.split_text(text) for text in passages]
    tokenizer = AutoTokenizer.from_pretrained(E.MODEL_ID)
    tokenization = {
        RAW: {
            "queries": tokenize_stats(tokenizer, queries),
            "passages": tokenize_stats(tokenizer, passages),
        },
        SPLIT: {
            "queries": tokenize_stats(tokenizer, split_queries),
            "passages": tokenize_stats(tokenizer, split_passages),
        },
    }
    del tokenizer

    n = len(docs)
    permutation = np.random.default_rng(PARTITION_SEED).permutation(n)
    folds = [np.sort(permutation[index::E.N_FOLDS]) for index in range(E.N_FOLDS)]
    reciprocal_ranks = {
        mode: np.zeros((len(SEEDS), n), dtype=float) for mode in (RAW, SPLIT)
    }
    fold_log = []

    for seed_index, seed in enumerate(SEEDS):
        for fold_index, test in enumerate(folds):
            train = np.sort(np.concatenate([
                folds[index] for index in range(E.N_FOLDS) if index != fold_index
            ]))
            train_queries = [queries[index] for index in train]
            train_passages = [passages[index] for index in train]
            lexical = R.BM25([R.words(text) for text in train_passages])
            hard_negatives = []
            for local_index, query in enumerate(train_queries):
                scores = lexical.scores(R.words(query))
                scores[local_index] = -1e9
                hard_negatives.append(int(np.argmax(scores)))

            test_candidates = np.broadcast_to(test, (len(test), len(test)))
            for mode, query_texts, passage_texts in (
                (RAW, queries, passages),
                (SPLIT, split_queries, split_passages),
            ):
                print(f"seed {seed}, fold {fold_index}, mode {mode}: training", flush=True)

                def evaluate(model, tok):
                    return E.score_candidate_texts(
                        model, tok, query_texts, passage_texts, test, test_candidates)

                scores = E.train_checkpoints(
                    [query_texts[index] for index in train],
                    [passage_texts[index] for index in train],
                    hard_negatives,
                    LR,
                    seed,
                    [EPOCHS],
                    evaluate,
                    lambda message: print(message, flush=True),
                )[EPOCHS]
                ranks = R.rank_of_gold(scores)
                reciprocal_ranks[mode][seed_index, test] = np.where(
                    ranks <= 10, 1.0 / ranks, 0.0)
                fold_log.append({
                    "seed": seed,
                    "fold": fold_index,
                    "mode": mode,
                    "train_pairs": int(len(train)),
                    "test_pairs": int(len(test)),
                    "candidate_passages": int(len(test)),
                    "MRR@10": float(reciprocal_ranks[mode][seed_index, test].mean() * 100),
                })

    raw_mean = reciprocal_ranks[RAW].mean(axis=0)
    split_mean = reciprocal_ranks[SPLIT].mean(axis=0)
    difference = split_mean - raw_mean
    result = {
        "protocol": {
            "status": "exploratory controlled intervention; historical results and corpus have been inspected",
            "source_sha256": source_stats["source_sha256"],
            "outer_folds": E.N_FOLDS,
            "outer_partition_seed": PARTITION_SEED,
            "training_seeds": list(SEEDS),
            "learning_rate": LR,
            "epochs": EPOCHS,
            "batch_size": E.BS,
            "query_max_pieces": E.MAX_Q,
            "passage_max_pieces": E.MAX_P,
            "negative_sampling": "one BM25 hard negative from each outer training fold",
            "intervention": "detach suffixes identified by the existing rule-based splitter, in queries and passages",
            "held_out_candidates_per_query": int(n // E.N_FOLDS),
            "uncertainty": "paired query bootstrap over mean reciprocal rank across the three training seeds",
            "seed_variance": "reported separately; the bootstrap conditions on these seeds and one outer partition",
        },
        "tokenization": tokenization,
        "systems": {
            RAW: {
                "MRR@10": float(raw_mean.mean() * 100),
                "per_seed_MRR@10": [
                    float(values.mean() * 100) for values in reciprocal_ranks[RAW]
                ],
                "seed_sd": float(np.std(
                    [values.mean() * 100 for values in reciprocal_ranks[RAW]], ddof=1)),
            },
            SPLIT: {
                "MRR@10": float(split_mean.mean() * 100),
                "per_seed_MRR@10": [
                    float(values.mean() * 100) for values in reciprocal_ranks[SPLIT]
                ],
                "seed_sd": float(np.std(
                    [values.mean() * 100 for values in reciprocal_ranks[SPLIT]], ddof=1)),
            },
        },
        "clitic_split_minus_raw": paired_query_bootstrap(difference),
        "folds": fold_log,
        "per_query": {
            "raw_rr10": raw_mean.tolist(),
            "clitic_split_rr10": split_mean.tolist(),
            "clitic_split_minus_raw_rr10": difference.tolist(),
        },
    }
    path = OUT / "morphology_intervention.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ranks_path = OUT / "morphology_intervention_ranks.csv"
    with ranks_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["query_index", "raw_rr10", "clitic_split_rr10", "difference_rr10"])
        writer.writerows(zip(range(n), raw_mean, split_mean, difference))
    print(json.dumps({
        "raw_MRR@10": result["systems"][RAW]["MRR@10"],
        "split_MRR@10": result["systems"][SPLIT]["MRR@10"],
        "difference": result["clitic_split_minus_raw"],
        "tokenization": tokenization,
        "results_path": str(path),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run()
