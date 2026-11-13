"""Exploratory query-only adaptive lexical/dense fusion on existing tasks.

The public Belebele test set and historical headline labels have already been
inspected. This script is an engineering prototype, not confirmatory evidence
for SoraniIR or the preregistered adaptive-fusion hypotheses.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import belebele_experiment as B
import proposed_experiment as E
import real_experiments as R
from task_dependence_analysis import word_overlap

OUT = Path(R.OUT)
SEED = 20261111
FOLDS = 3
BOOTSTRAP_RESAMPLES = 5000
RIDGE_ALPHA = 1.0
INTERCEPTS = tuple(np.round(np.arange(0.1, 0.91, 0.1), 1))
SLOPES = (0.0, 0.25, 0.5, 1.0, 1.5)
WEIGHT_GRID = np.round(np.arange(0.0, 1.01, 0.1), 1)


def query_features(query):
    normalized = R.TOK.normalize_orthography(query)
    words = R.words(normalized)
    content = [word for word in words if len(word) >= 3]
    chars = [char for char in normalized if not char.isspace()]
    char_counts = np.asarray(list({char: chars.count(char) for char in set(chars)}.values()), dtype=float)
    entropy = 0.0
    if char_counts.size:
        probabilities = char_counts / char_counts.sum()
        entropy = float(-(probabilities * np.log(probabilities)).sum())
    return np.asarray([
        np.log1p(len(normalized)),
        np.log1p(len(words)),
        float(np.mean([len(word) for word in words])) if words else 0.0,
        len(set(words)) / max(1, len(words)),
        len(content) / max(1, len(words)),
        np.log1p(len(set(R.char_ngrams(normalized)))),
        entropy,
        sum(char.isdigit() for char in normalized) / max(1, len(normalized)),
    ], dtype=float)


def rr10(ranks):
    ranks = np.asarray(ranks)
    return np.where(ranks <= 10, 1.0 / ranks, 0.0)


def group_folds(groups, n_folds=FOLDS, seed=SEED):
    unique = np.unique(groups)
    shuffled = np.random.default_rng(seed).permutation(unique)
    assignment = {group: index % n_folds for index, group in enumerate(shuffled)}
    return np.asarray([assignment[group] for group in groups], dtype=int)


def candidate_rank(scores, gold):
    gold_scores = scores[np.arange(len(gold)), gold][:, None]
    return 1 + (scores > gold_scores).sum(axis=1) + (scores == gold_scores).sum(axis=1) - 1


def blend_scores(matrices, bm25_weights):
    lexical, e5, bge = (E.zscore(matrix) for matrix in matrices)
    dense = (e5 + bge) / 2.0
    weights = np.asarray(bm25_weights, dtype=float)
    if weights.shape != (lexical.shape[0],):
        raise ValueError("One BM25 weight is required for every query")
    return weights[:, None] * lexical + (1 - weights[:, None]) * dense


def fit_overlap_predictor(features, overlaps, indices):
    if len(indices) < 2:
        raise ValueError("At least two independent training queries are required")
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(features[indices], overlaps[indices])
    return model


def select_policy(matrices, gold, predictions):
    centered = np.clip(predictions, 0.0, 1.0) - 0.5
    best = None
    for intercept in INTERCEPTS:
        for slope in SLOPES:
            weights = np.clip(intercept + slope * centered, 0.0, 1.0)
            ranks = candidate_rank(blend_scores(matrices, weights), gold)
            score = float(rr10(ranks).mean() * 100)
            candidate = (score, -abs(intercept - 1 / 3), -slope, intercept, slope)
            if best is None or candidate > best:
                best = candidate
    return {"validation_mrr10": best[0], "intercept": best[3], "slope": best[4]}


def grouped_bootstrap(differences, groups, seed=SEED):
    differences = np.asarray(differences, dtype=float)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAP_RESAMPLES)
    for index in range(BOOTSTRAP_RESAMPLES):
        selected = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[group] for group in selected])
        samples[index] = differences[rows].mean() * 100
    probability = 2 * min((samples <= 0).mean(), (samples >= 0).mean())
    return {
        "difference_points": float(differences.mean() * 100),
        "CI95": [float(x) for x in np.percentile(samples, [2.5, 97.5])],
        "paired_bootstrap_p_two_sided": max(float(probability), 1 / BOOTSTRAP_RESAMPLES),
    }


def load_news():
    docs, stats = R.load_data()
    queries = [R.TOK.normalize_orthography(doc["headline"]) for doc in docs]
    passages = [R.TOK.normalize_orthography(doc["body"]) for doc in docs]
    gold = np.arange(len(docs), dtype=int)
    groups = gold.copy()
    cache_scores = {}
    for model_name in ("e5", "bge"):
        cache = OUT / f"adaptive_fusion_{model_name}_{stats['source_sha256'][:12]}.npz"
        if cache.exists():
            with np.load(cache) as saved:
                scores = saved["scores"]
        else:
            scores = B.dense(model_name, queries, passages)
            np.savez_compressed(cache, scores=scores)
        if scores.shape != (len(queries), len(passages)):
            raise ValueError(f"Unexpected {model_name} score matrix shape: {scores.shape}")
        cache_scores[model_name] = scores
    bm25 = R.BM25([R.char_ngrams(passage) for passage in passages])
    lexical = np.stack([bm25.scores(R.char_ngrams(query)) for query in queries])
    return {
        "queries": queries,
        "passages": passages,
        "gold": gold,
        "groups": groups,
        "features": np.stack([query_features(query) for query in queries]),
        "overlaps": np.asarray(
            [word_overlap(query, passages[index]) for query, index in zip(queries, gold)]),
        "matrices": [lexical, cache_scores["e5"], cache_scores["bge"]],
        "data": stats,
    }


def load_belebele():
    rows, passages, gold = B.load_dataset()
    queries = [row["question"] for row in rows]
    lexical_index = R.BM25([R.char_ngrams(passage) for passage in passages])
    lexical = np.stack([
        lexical_index.scores(R.char_ngrams(query)) for query in queries])
    e5_scores = B.dense("e5", queries, passages)
    bge_scores = B.dense("bge", queries, passages)
    groups = np.asarray(gold, dtype=int)
    return {
        "queries": queries,
        "passages": passages,
        "gold": np.asarray(gold, dtype=int),
        "groups": groups,
        "features": np.stack([query_features(query) for query in queries]),
        "overlaps": np.asarray([
            word_overlap(query, passages[index]) for query, index in zip(queries, gold)]),
        "matrices": [lexical, e5_scores, bge_scores],
        "data": {"dataset_revision": B.DATA_REV, "dataset_sha256": B.DATA_SHA},
    }


def evaluate_task(name, dataset, seed_offset=0, source_policy=None):
    gold, groups = dataset["gold"], dataset["groups"]
    features, overlaps = dataset["features"], dataset["overlaps"]
    matrices = dataset["matrices"]
    assignments = group_folds(groups, seed=SEED + seed_offset)
    fixed_rr = np.zeros(len(gold))
    adaptive_rr = np.zeros(len(gold))
    oracle_rr = np.zeros(len(gold))
    predicted_overlap = np.zeros(len(gold))
    fold_records = []

    for fold in range(FOLDS):
        test = np.flatnonzero(assignments == fold)
        outer_train = np.flatnonzero(assignments != fold)
        unique_groups = np.unique(groups[outer_train])
        shuffled = np.random.default_rng(SEED + seed_offset + fold).permutation(unique_groups)
        validation_group_count = max(1, len(shuffled) // 4)
        validation_groups = set(shuffled[:validation_group_count])
        validation_mask = np.asarray([groups[index] in validation_groups for index in outer_train])
        fit = outer_train[~validation_mask]
        validation = outer_train[validation_mask]

        if source_policy is None:
            predictor = fit_overlap_predictor(features, overlaps, fit)
            val_prediction = np.clip(predictor.predict(features[validation]), 0, 1)
            policy = select_policy(
                [matrix[validation] for matrix in matrices], gold[validation], val_prediction)
            predictor = fit_overlap_predictor(features, overlaps, outer_train)
        else:
            predictor, policy = source_policy
        test_prediction = np.clip(predictor.predict(features[test]), 0, 1)
        predicted_overlap[test] = test_prediction
        fixed_weights = np.full(len(test), 1 / 3)
        adaptive_weights = np.clip(
            policy["intercept"] + policy["slope"] * (test_prediction - 0.5), 0, 1)
        fixed_ranks = candidate_rank(
            blend_scores([matrix[test] for matrix in matrices], fixed_weights), gold[test])
        adaptive_ranks = candidate_rank(
            blend_scores([matrix[test] for matrix in matrices], adaptive_weights), gold[test])

        # Per-query oracle is an explicitly non-deployable upper bound, not a test-time method.
        oracle_best = np.zeros(len(test))
        test_matrices = [matrix[test] for matrix in matrices]
        for weight in WEIGHT_GRID:
            candidate = candidate_rank(
                blend_scores(test_matrices, np.full(len(test), weight)), gold[test])
            oracle_best = np.maximum(oracle_best, rr10(candidate))
        fixed_rr[test] = rr10(fixed_ranks)
        adaptive_rr[test] = rr10(adaptive_ranks)
        oracle_rr[test] = oracle_best
        fold_records.append({
            "fold": fold,
            "fit_queries": int(len(fit)),
            "validation_queries": int(len(validation)),
            "test_queries": int(len(test)),
            "test_passage_groups": int(len(np.unique(groups[test]))),
            "selected_policy": policy,
            "policy_source": "target validation" if source_policy is None else "news training only",
            "mean_test_predicted_overlap": float(test_prediction.mean()),
        })

    difference = adaptive_rr - fixed_rr
    result = {
        "queries": len(gold),
        "passages": len(dataset["passages"]),
        "passage_groups": int(len(np.unique(groups))),
        "mean_true_overlap_descriptive": float(overlaps.mean()),
        "median_true_overlap_descriptive": float(np.median(overlaps)),
        "fixed_equal_weight_MRR10": float(fixed_rr.mean() * 100),
        "adaptive_MRR10": float(adaptive_rr.mean() * 100),
        "oracle_MRR10_non_deployable": float(oracle_rr.mean() * 100),
        "adaptive_minus_fixed": grouped_bootstrap(difference, groups),
        "low_vs_high_predicted_overlap": {},
        "folds": fold_records,
        "per_query": {
            "fixed_rr": fixed_rr.tolist(),
            "adaptive_rr": adaptive_rr.tolist(),
            "predicted_overlap": predicted_overlap.tolist(),
            "descriptive_true_overlap": overlaps.tolist(),
        },
    }
    quantiles = np.quantile(predicted_overlap, [1 / 3, 2 / 3])
    strata = np.digitize(predicted_overlap, quantiles, right=True)
    for label, mask in (
        ("low", strata == 0),
        ("mid", strata == 1),
        ("high", strata == 2),
    ):
        result["low_vs_high_predicted_overlap"][label] = {
            "queries": int(mask.sum()),
            "mean_predicted_overlap": float(predicted_overlap[mask].mean()),
            "adaptive_MRR10": float(adaptive_rr[mask].mean() * 100),
            "fixed_equal_weight_MRR10": float(fixed_rr[mask].mean() * 100),
            "difference_points": float(difference[mask].mean() * 100),
        }
    return result


def fit_source_policy(dataset):
    features, overlaps, groups = (
        dataset["features"], dataset["overlaps"], dataset["groups"])
    assignments = group_folds(groups, seed=SEED)
    policies = []
    for fold in range(FOLDS):
        test = np.flatnonzero(assignments == fold)
        outer_train = np.flatnonzero(assignments != fold)
        unique_groups = np.unique(groups[outer_train])
        shuffled = np.random.default_rng(SEED + fold).permutation(unique_groups)
        validation_group_count = max(1, len(shuffled) // 4)
        validation_groups = set(shuffled[:validation_group_count])
        validation_mask = np.asarray([groups[index] in validation_groups for index in outer_train])
        fit = outer_train[~validation_mask]
        validation = outer_train[validation_mask]
        predictor = fit_overlap_predictor(features, overlaps, fit)
        predictions = np.clip(predictor.predict(features[validation]), 0, 1)
        policy = select_policy(
            [matrix[validation] for matrix in dataset["matrices"]],
            dataset["gold"][validation],
            predictions,
        )
        policies.append(policy)
    predictor = fit_overlap_predictor(features, overlaps, np.arange(len(groups)))
    aggregate = {
        "intercept": float(np.mean([policy["intercept"] for policy in policies])),
        "slope": float(np.mean([policy["slope"] for policy in policies])),
        "selection_source": "mean policy parameters selected on the three news validation folds",
    }
    return predictor, aggregate, policies


def main():
    print("Loading headline components", flush=True)
    news = load_news()
    print(f"Evaluating headline: {len(news['queries'])} queries", flush=True)
    headline = evaluate_task("headline", news)
    headline["data"] = news["data"]
    source_predictor, source_policy, policy_folds = fit_source_policy(news)

    print("Loading Belebele components", flush=True)
    belebele = load_belebele()
    print(f"Evaluating Belebele: {len(belebele['queries'])} queries", flush=True)
    belebele_in_domain = evaluate_task("belebele_in_domain_exploratory", belebele, 100)
    belebele_transfer = evaluate_task(
        "belebele_news_transfer", belebele, 200,
        source_policy=(source_predictor, source_policy),
    )
    belebele_in_domain["data"] = belebele["data"]
    belebele_transfer["data"] = belebele["data"]
    tasks = {
        "headline": headline,
        "belebele_in_domain_exploratory": belebele_in_domain,
        "belebele_news_transfer": belebele_transfer,
    }
    for name, task in tasks.items():
        print(json.dumps({
            "task": name,
            "fixed_MRR10": task["fixed_equal_weight_MRR10"],
            "adaptive_MRR10": task["adaptive_MRR10"],
            "difference": task["adaptive_minus_fixed"],
            "oracle_MRR10_non_deployable": task["oracle_MRR10_non_deployable"],
        }, indent=2), flush=True)

    output = {
        "protocol": {
            "status": "exploratory; not confirmatory and not SoraniIR",
            "reason": "The historical headline results and public Belebele test set were inspected before this analysis.",
            "primary_metric": "MRR@10",
            "features": "query-only surface statistics; no relevant passage or query-passage overlap is used at inference",
            "overlap_predictor": f"StandardScaler + Ridge(alpha={RIDGE_ALPHA}) trained on training-fold overlap labels",
            "policy": "BM25 weight = clip(intercept + slope * (predicted overlap - 0.5), 0, 1); dense weight is shared equally by E5 and BGE-M3",
            "policy_selection": "validation-fold MRR@10 grid search; test folds are passage-grouped for Belebele",
            "transfer": "Belebele transfer uses an overlap predictor and mean fusion policy selected only from the headline corpus; no Belebele tuning",
            "headline_selected_transfer_policy": source_policy,
            "headline_validation_policies": policy_folds,
            "intercepts": INTERCEPTS,
            "slopes": SLOPES,
            "oracle": "per-query test-optimal BM25 weight over the 0.0..1.0 grid; non-deployable upper bound only",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_unit": "source passage group",
            "seed": SEED,
        },
        "tasks": tasks,
    }
    result_path = OUT / "adaptive_fusion_exploratory.json"
    result_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    ranks_path = OUT / "adaptive_fusion_exploratory_ranks.csv"
    with ranks_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "task", "query_index", "fixed_equal_weight_rr10", "adaptive_rr10",
            "predicted_overlap", "descriptive_true_overlap",
        ])
        for name, task in tasks.items():
            per_query = task["per_query"]
            writer.writerows(
                (name, index, fixed, adaptive, predicted, true_overlap)
                for index, (fixed, adaptive, predicted, true_overlap) in enumerate(zip(
                    per_query["fixed_rr"], per_query["adaptive_rr"],
                    per_query["predicted_overlap"], per_query["descriptive_true_overlap"]))
            )
    print(f"Wrote {result_path} and {ranks_path}", flush=True)


if __name__ == "__main__":
    main()
