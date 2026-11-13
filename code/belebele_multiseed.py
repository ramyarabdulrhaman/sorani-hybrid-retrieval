"""Multi-seed version of the primary Belebele cross-domain experiment.

Uses the same passage-grouped validation/test split and the same training
configuration as belebele_experiment.py, but trains the news-adapted E5
component with seeds 1, 2 and 3. Fusion weights are selected on the validation
questions separately for each seed. Test metrics are reported per seed and as
a mean with standard deviation.

Outputs
-------
results/belebele/multiseed_results.json
results/belebele/multiseed_ranks.csv
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_experiments as R
import belebele_experiment as B

SEEDS = [1, 2, 3]
OUT = Path(R.BASE) / "results" / "belebele"
BOOT = 5000
SEED = 42


def main():
    rows, passages, gold = B.load_dataset()
    queries = [r["question"] for r in rows]
    perm = np.random.default_rng(42).permutation(len(passages))
    val_groups = np.sort(perm[: len(perm) // 2])
    val = np.isin(gold, val_groups)
    test = ~val
    print(f"Validation: {val.sum()} questions; test: {test.sum()}; corpus: {len(passages)}", flush=True)

    bm = R.BM25([R.char_ngrams(p) for p in passages])
    c4 = np.stack([bm.scores(R.char_ngrams(q)) for q in queries])
    print("encoding zero-shot E5", flush=True)
    e5m = B.dense("e5", queries, passages)
    print("encoding zero-shot BGE-M3", flush=True)
    bgem = B.dense("bge", queries, passages)

    per_seed = []
    rank_rows = []
    for seed in SEEDS:
        ft, _ = B.train_news_and_encode(queries, passages, seed=seed)
        w, val_mrr = B.select_weights([c4[val], ft[val], bgem[val]], gold[val], required_index=1)
        fused = B.weighted([c4, ft, bgem], w)
        r = B.ranks(fused, gold)
        r_test = r[test]
        m = B.metrics(r_test)
        per_seed.append({
            "seed": seed, "weights_char4_ft_bge": list(w),
            "validation_MRR@10": float(val_mrr), "test_MRR@10": float(m["MRR@10"]),
            "test_Recall@1": float(m["Recall@1"]),
            "test_nDCG@10": float(m["nDCG@10"]),
        })
        rank_rows.append(r_test)
        print(f"seed {seed}: weights {w} val {val_mrr:.2f} test {m['MRR@10']:.2f}", flush=True)

    rr = np.stack([np.where(r <= 10, 1.0 / r, 0) for r in rank_rows])
    mean_rr = rr.mean(0)
    rng = np.random.default_rng(SEED)
    groups = gold[test]
    unique = np.unique(groups)
    per_group = [np.flatnonzero(groups == g) for g in unique]
    base_r = B.ranks(c4, gold)[test]
    base_rr = np.where(base_r <= 10, 1.0 / base_r, 0)
    samples = []
    for _ in range(BOOT):
        ids = np.concatenate([per_group[i] for i in rng.integers(0, len(unique), len(unique))])
        samples.append(float(np.mean(mean_rr[ids] - base_rr[ids]) * 100))
    result = {
        "protocol": {
            "seeds": SEEDS,
            "split": "same passage-grouped 50/50 validation/test split as belebele_experiment.py",
            "training": "news-adapted E5, lr=1e-5, 2 epochs, raw text; one model per seed",
            "fusion": "validation-selected nonnegative 0.1-grid weights; fine-tuned E5 weight >= 0.1",
            "bootstrap": "5000 paired passage-cluster resamples after averaging reciprocal ranks over seeds",
            "note": "conditions on the split, models and weights; does not capture partition uncertainty",
        },
        "per_seed": per_seed,
        "mean_test_MRR@10": float(mean_rr.mean() * 100),
        "seed_std_test_MRR@10": float(np.std([x["test_MRR@10"] for x in per_seed], ddof=1)),
        "mean_minus_char4_MRR@10": float((mean_rr.mean() - base_rr.mean()) * 100),
        "mean_minus_char4_CI95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "char4_test_MRR@10": float(base_rr.mean() * 100),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "multiseed_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "multiseed_ranks.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "seed", "test_rank_fused"])
        for si, seed in enumerate(SEEDS):
            for i, r in enumerate(rank_rows[si]):
                w.writerow([int(np.flatnonzero(test)[i]), seed, int(r)])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
