"""Test BGE-M3 sensitivity to Unicode normalization and passage truncation."""
import json
from pathlib import Path

import numpy as np

import real_experiments as R


def mean_rr(ranks):
    return np.where(ranks <= 10, 1 / ranks, 0)


def summarize(ranks, rng):
    n = len(ranks)
    mrr = mean_rr(ranks)
    samples = np.asarray([
        mrr[rng.integers(0, n, n)].mean() * 100
        for _ in range(R.N_BOOT)
    ])
    return {
        "Recall@1": float((ranks <= 1).mean() * 100),
        "Recall@5": float((ranks <= 5).mean() * 100),
        "MRR@10": float(mrr.mean() * 100),
        "MRR@10_CI95": [float(x) for x in np.percentile(samples, [2.5, 97.5])],
    }


def paired_summary(ranks, baseline_rr, rng):
    difference = mean_rr(ranks) - baseline_rr
    n = len(difference)
    samples = np.asarray([
        difference[rng.integers(0, n, n)].mean() * 100
        for _ in range(R.N_BOOT)
    ])
    p = 2 * min((samples <= 0).mean(), (samples >= 0).mean())
    return {
        "MRR@10_difference_points": float(difference.mean() * 100),
        "paired_bootstrap_CI95": [
            float(x) for x in np.percentile(samples, [2.5, 97.5])
        ],
        "paired_bootstrap_p_two_sided": max(float(p), 1 / R.N_BOOT),
    }


def adjust_holm(results):
    comparisons = list(results.values())
    order = np.argsort([
        item["paired_bootstrap_p_two_sided"] for item in comparisons
    ])
    adjusted = np.empty(len(comparisons))
    running_max = 0.0
    for rank, index in enumerate(order):
        p = comparisons[index]["paired_bootstrap_p_two_sided"]
        running_max = max(running_max, (len(comparisons) - rank) * p)
        adjusted[index] = min(1.0, running_max)
    for item, p_adjusted in zip(comparisons, adjusted):
        item["holm_adjusted_p_two_sided"] = float(p_adjusted)


def encode(model, tokenizer, texts, device, max_length, batch_size):
    import torch

    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            vectors.append(
                torch.nn.functional.normalize(
                    model(**encoded).last_hidden_state[:, 0].float(),
                    dim=-1,
                ).cpu().numpy()
            )
    return np.concatenate(vectors)


def main():
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    docs, _ = R.load_data()
    queries = [doc["headline"] for doc in docs]
    passages = [doc["body"] for doc in docs]
    normalized_queries = [R.TOK.normalize_orthography(text) for text in queries]
    normalized_passages = [R.TOK.normalize_orthography(text) for text in passages]
    n = len(docs)
    model_id = "BAAI/bge-m3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    model_max = min(
        int(getattr(config, "max_position_embeddings", 8192)),
        int(tokenizer.model_max_length),
    )
    raw_lengths = [
        len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        for text in passages
    ]
    normalized_lengths = [
        len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        for text in normalized_passages
    ]
    full_length = max(raw_lengths + normalized_lengths)
    if full_length > model_max:
        raise ValueError(
            f"Full passage length {full_length} exceeds model limit {model_max}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_id).eval().to(device)

    bm = R.BM25([R.char_ngrams(text) for text in passages])
    lexical_scores = np.stack([
        bm.scores(R.char_ngrams(text)) for text in queries
    ])
    lexical_ranks = R.rank_of_gold(lexical_scores)
    lexical_rr = mean_rr(lexical_ranks)

    cached_path = Path(R.OUT) / "emb_BAAI_bge-m3.npz"
    cached = np.load(cached_path)
    cached_ranks = R.rank_of_gold(cached["q"] @ cached["p"].T)
    results = {
        "protocol": {
            "model": model_id,
            "pooling": "CLS with L2 normalization",
            "unicode_normalization": "MorphoBPETokenizer.normalize_orthography, the same orthographic normalizer used before BM25 tokenization",
            "queries": n,
            "candidate_passages": n,
            "full_passage_max_tokens": full_length,
            "full_length_input_limit": model_max,
            "device": str(device),
            "raw_passage_lengths": {
                "mean": float(np.mean(raw_lengths)),
                "p95": float(np.percentile(raw_lengths, 95)),
                "max": max(raw_lengths),
                "over_512": int(np.count_nonzero(np.asarray(raw_lengths) > 512)),
            },
            "normalized_passage_lengths": {
                "mean": float(np.mean(normalized_lengths)),
                "p95": float(np.percentile(normalized_lengths, 95)),
                "max": max(normalized_lengths),
                "over_512": int(np.count_nonzero(np.asarray(normalized_lengths) > 512)),
            },
        },
        "BM25_char4": summarize(lexical_ranks, np.random.default_rng(R.SEED)),
        "BGE-M3_raw_512_cached": summarize(
            cached_ranks, np.random.default_rng(R.SEED)
        ),
        "BGE-M3_raw_512_cached_vs_BM25_char4": paired_summary(
            cached_ranks, lexical_rr, np.random.default_rng(R.SEED)
        ),
    }
    variant_ranks = {"BGE-M3_raw_512": cached_ranks}

    variants = [
        ("BGE-M3_normalized_512", normalized_queries, normalized_passages, 512, 8),
        ("BGE-M3_raw_full", queries, passages, full_length, 1),
        ("BGE-M3_normalized_full", normalized_queries, normalized_passages, full_length, 1),
    ]
    for name, qs, ps, passage_limit, passage_batch_size in variants:
        qv = encode(model, tokenizer, qs, device, 64, 16)
        pv = encode(model, tokenizer, ps, device, passage_limit, passage_batch_size)
        ranks = R.rank_of_gold(qv @ pv.T)
        variant_ranks[name] = ranks
        results[name] = summarize(ranks, np.random.default_rng(R.SEED))
        results[f"{name}_vs_BM25_char4"] = paired_summary(
            ranks, lexical_rr, np.random.default_rng(R.SEED)
        )
        print(f"{name}: {results[name]}", flush=True)
        del qv, pv
        if device.type == "cuda":
            torch.cuda.empty_cache()

    sensitivity_pairs = [
        ("BGE-M3_normalized_512", "BGE-M3_raw_512"),
        ("BGE-M3_raw_full", "BGE-M3_raw_512"),
        ("BGE-M3_normalized_full", "BGE-M3_normalized_512"),
        ("BGE-M3_normalized_full", "BGE-M3_raw_full"),
    ]
    sensitivity = {}
    for system, baseline in sensitivity_pairs:
        sensitivity[f"{system}_minus_{baseline}"] = paired_summary(
            variant_ranks[system],
            mean_rr(variant_ranks[baseline]),
            np.random.default_rng(R.SEED),
        )
    adjust_holm(sensitivity)
    results["paired_input_sensitivity"] = sensitivity

    output = Path(R.OUT) / "bge_m3_input_fairness.json"
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
