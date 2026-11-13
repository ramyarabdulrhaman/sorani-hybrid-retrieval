"""Full-corpus zero-shot equal-weight fusion on the 492-passage headline task.

The cross-fitted equal-weight analysis ranks each query among 164 passages.
This script evaluates the same fixed one-third fusion (char-4gram BM25,
zero-shot E5, zero-shot BGE-M3) on the full 492-passage collection, so the
headline and Belebele tasks can be compared under one candidate-set protocol.
"""
import os, sys, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'code'))
import real_experiments as R

OUT = os.path.join(BASE, 'results')
SEED = 42
N_BOOT = 5000


def zscore(s):
    mu = s.mean(1, keepdims=True)
    sd = s.std(1, keepdims=True)
    sd[sd == 0] = 1.0
    return (s - mu) / sd


def rank_of_gold(s):
    n = s.shape[0]
    gold = s[np.arange(n), np.arange(n)][:, None]
    return 1 + (s > gold).sum(1) + ((s == gold).sum(1) - 1)


def rr10(ranks):
    return np.where(ranks <= 10, 1.0 / ranks, 0.0)


def metrics(ranks):
    return {
        'Recall@1': float((ranks <= 1).mean() * 100),
        'Recall@5': float((ranks <= 5).mean() * 100),
        'MRR@10': float(rr10(ranks).mean() * 100),
        'nDCG@10': float(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0).mean() * 100),
    }


def main():
    docs, stats = R.load_data()
    queries = [R.TOK.normalize_orthography(d['headline']) for d in docs]
    passages = [R.TOK.normalize_orthography(d['body']) for d in docs]
    n = len(docs)

    bm25 = R.BM25([R.char_ngrams(p) for p in passages])
    char_scores = np.stack([bm25.scores(R.char_ngrams(q)) for q in queries])

    e5 = np.load(os.path.join(OUT, 'emb_intfloat_multilingual-e5-base_normalized.npz'))
    bge = np.load(os.path.join(OUT, 'emb_BAAI_bge-m3_normalized.npz'))
    e5_scores = e5['q'] @ e5['p'].T
    bge_scores = bge['q'] @ bge['p'].T

    fused_scores = zscore(char_scores) + zscore(e5_scores) + zscore(bge_scores)
    char_ranks = rank_of_gold(char_scores)
    fused_ranks = rank_of_gold(fused_scores)

    char_rr = rr10(char_ranks)
    fused_rr = rr10(fused_ranks)
    diff = (fused_rr - char_rr) * 100
    rng = np.random.default_rng(SEED)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())

    result = {
        'protocol': {
            'task': 'headline-to-article retrieval',
            'queries': n,
            'candidate_passages': n,
            'fusion': 'fixed one-third weights; per-query z-score of char-4gram BM25, zero-shot E5, and zero-shot BGE-M3',
            'source': 'recomputed from results/real_results.json inputs and cached embeddings',
            'bootstrap_resamples': N_BOOT,
            'seed': SEED,
        },
        'data': stats,
        'systems': {
            'BM25 char-4gram': metrics(char_ranks),
            'Zero-shot equal-weight fusion (full corpus)': metrics(fused_ranks),
        },
        'difference_from_char4': {
            'MRR@10_points': float(diff.mean()),
            'paired_bootstrap_CI95': [float(lo), float(hi)],
            'paired_bootstrap_p_two_sided': float(max(p, 1.0 / N_BOOT)),
            'wins': int((diff > 0).sum()),
            'losses': int((diff < 0).sum()),
            'ties': int((diff == 0).sum()),
        },
    }
    with open(os.path.join(OUT, 'full_corpus_equal_weight.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    pd.DataFrame({
        'query_index': np.arange(n),
        'BM25_char4_MRR10': char_rr,
        'zero_shot_equal_weight_MRR10': fused_rr,
        'BM25_char4_rank': char_ranks,
        'zero_shot_equal_weight_rank': fused_ranks,
    }).to_csv(os.path.join(OUT, 'full_corpus_equal_weight_ranks.csv'), index=False)
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
