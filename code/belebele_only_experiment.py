"""Exploratory Belebele-only retriever training and held-out passage-group test.

This reuses an already inspected public test dataset, so it is not an
independent confirmatory benchmark. No news corpus is used.
"""
import csv
import json
import platform

import numpy as np
import torch
import transformers
from huggingface_hub import snapshot_download

import belebele_experiment as B

SEED = 77
TRAIN_SEEDS = (1, 2, 3)
SYSTEMS = ("BM25 char-4gram", "E5 zero-shot", "BGE-M3 zero-shot",
           "Zero-shot fusion", "E5 Belebele-trained", "Proposed Belebele-trained fusion")


def partition(gold, n_passages):
    perm = np.random.default_rng(SEED).permutation(n_passages)
    if n_passages != 488:
        raise ValueError("Unexpected corpus size")
    group = np.empty(n_passages, dtype=np.int8)
    group[perm[:244]], group[perm[244:366]], group[perm[366:]] = 0, 1, 2
    return group[gold]


def train(queries, passages, gold, query_groups, seed):
    ids = np.flatnonzero(query_groups == 0)
    first = {}
    for i in ids:
        first.setdefault(int(gold[i]), int(i))
    selected = list(first.values())
    train_q = [queries[i] for i in selected]
    train_p = [passages[gold[i]] for i in selected]
    index = B.R.BM25([B.R.words(p) for p in train_p])
    negatives = []
    for i, q in enumerate(train_q):
        scores = index.scores(B.R.words(q))
        scores[i] = -1e9
        negatives.append(int(np.argmax(scores)))
    B.P.MODEL_ID = snapshot_download(
        repo_id=B.MODELS["e5"][0], revision=B.MODELS["e5"][1],
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                        "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"])

    def evaluate(model, tokenizer):
        q = B.P.encode(model, tokenizer, [B.P.QP + x for x in queries], B.P.MAX_Q, bs=16)
        p = B.P.encode(model, tokenizer, [B.P.PP + x for x in passages], B.P.MAX_P, bs=4)
        return q @ p.T

    print(f"Training seed {seed} on {len(selected)} distinct training passages", flush=True)
    results = B.P.train_checkpoints(train_q, train_p, negatives, 1e-5, seed, [2],
                                    evaluate, lambda message: print(message, flush=True))
    return results[2]


def main():
    rows, passages, gold = B.load_dataset()
    queries = [r["question"] for r in rows]
    groups = partition(gold, len(passages))
    valid, test = groups == 1, groups == 2
    print("Partition question counts:", np.bincount(groups).tolist(), flush=True)
    index = B.R.BM25([B.R.char_ngrams(p) for p in passages])
    c4 = np.stack([index.scores(B.R.char_ngrams(q)) for q in queries])
    e5, bge = B.dense("e5", queries, passages), B.dense("bge", queries, passages)
    matrices = [c4, e5, bge]
    zw, zv = B.select_weights([s[valid] for s in matrices], gold[valid])
    zero = B.weighted(matrices, zw)
    ranks = {name: [] for name in SYSTEMS}
    selections = []
    for seed in TRAIN_SEEDS:
        ft = train(queries, passages, gold, groups, seed)
        proposed = [c4, ft, bge]
        weights, val_score = B.select_weights([s[valid] for s in proposed], gold[valid], 1)
        mats = [c4, e5, bge, zero, ft, B.weighted(proposed, weights)]
        for name, matrix in zip(SYSTEMS, mats):
            ranks[name].append(B.ranks(matrix[test], gold[test]))
        selections.append({"seed": seed, "weights": weights, "validation_MRR@10": val_score})
    r = {name: np.stack(value) for name, value in ranks.items()}
    rr = {name: np.where(value <= 10, 1 / value, 0) for name, value in r.items()}
    systems = {}
    for name in SYSTEMS:
        systems[name] = {
            "Recall@1": float((r[name] == 1).mean() * 100),
            "Recall@5": float((r[name] <= 5).mean() * 100),
            "MRR@10": float(rr[name].mean() * 100),
            "MRR@10_per_seed": (rr[name].mean(axis=1) * 100).tolist(),
        }
    comparisons = {}
    for name in (SYSTEMS[0], SYSTEMS[3]):
        a, b = rr[SYSTEMS[-1]].mean(axis=0), rr[name].mean(axis=0)
        comparisons[name] = {"MRR@10_difference": float((a - b).mean() * 100),
                             "CI95_passage_cluster": B.cluster_ci(a, b, gold[test])}
    output = {"protocol": {
        "dataset_revision": B.DATA_REV, "dataset_sha256": B.DATA_SHA, "license": "CC BY-SA 4.0",
        "warning": "Exploratory split of already inspected public test data; not an independent benchmark.",
        "passage_split_seed": SEED, "train_val_test_passages": [244, 122, 122],
        "train_val_test_questions": [int((groups == i).sum()) for i in range(3)],
        "candidates": len(passages), "training": "One first question per training passage; no news data",
        "training_seeds": TRAIN_SEEDS, "epochs": 2, "learning_rate": 1e-5,
        "fusion": "Validation-only grid 0.1; proposed requires fine-tuned weight >= 0.1",
        "bootstrap": "5000 paired passage-cluster resamples, seed 42, of seed-averaged per-query RR; models and weights fixed",
        "model_revisions": B.MODELS, "device": B.P.DEV, "torch": torch.__version__,
        "transformers": transformers.__version__, "python": platform.python_version(),
    }, "zero_shot_selection": {"weights": zw, "validation_MRR@10": zv},
        "selection": selections, "systems": systems, "comparisons": comparisons}
    B.OUT.mkdir(parents=True, exist_ok=True)
    test_ids = np.flatnonzero(test)
    with (B.OUT / "belebele_only_ranks.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "passage_index", "training_seed", *SYSTEMS])
        for j, seed in enumerate(TRAIN_SEEDS):
            for k, row_id in enumerate(test_ids):
                writer.writerow([int(row_id), int(gold[row_id]), seed, *[int(r[n][j, k]) for n in SYSTEMS]])
    (B.OUT / "belebele_only_results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"systems": systems, "comparisons": comparisons}, indent=2), flush=True)


if __name__ == "__main__":
    main()
