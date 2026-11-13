"""Cross-encoder reranking on top of the first-stage retrievers.

First stage: character 4-gram BM25 and the zero-shot equal-weight fusion
(headline) or the validation-tuned / fine-tuned fusion (Belebele).
Second stage: BAAI/bge-reranker-base rescores the top-k first-stage candidates.
Non-candidates keep their first-stage order below the reranked candidates.

Outputs
-------
results/reranker.json
results/reranker_ranks.csv
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_experiments as R
import proposed_experiment as P

MODEL_ID = "BAAI/bge-reranker-base"
MODEL_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 50
MAX_LEN = 512
BATCH = 32
BOOT = 5000
SEED = 42


def zscore(m):
    return (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-9)


def rank_of_gold(scores, gold):
    correct = scores[np.arange(len(gold)), gold][:, None]
    return 1 + (scores > correct).sum(1) + (scores == correct).sum(1) - 1


def metrics(r):
    rr = np.where(r <= 10, 1.0 / r, 0)
    return {
        "Recall@1": float((r <= 1).mean() * 100),
        "Recall@5": float((r <= 5).mean() * 100),
        "Recall@10": float((r <= 10).mean() * 100),
        "MRR@10": float(rr.mean() * 100),
        "nDCG@10": float(np.where(r <= 10, 1.0 / np.log2(r + 1), 0).mean() * 100),
    }


def paired(a_ranks, b_ranks, rng):
    a = np.where(a_ranks <= 10, 1.0 / a_ranks, 0)
    b = np.where(b_ranks <= 10, 1.0 / b_ranks, 0)
    d = a - b
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(BOOT)])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {
        "mean_MRR@10_diff": float(d.mean() * 100),
        "CI95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "p_two_sided": min(1.0, max(float(p), 1.0 / BOOT)),
        "discordant": int(np.count_nonzero(np.abs(a - b) > 1e-12)),
    }


def load_reranker():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16
    ).to(DEV).eval()
    return tok, model


def rerank_scores(tok, model, queries, passages, first_stage, gold, top_k=TOP_K):
    """Return a new score matrix with the top-k candidates reordered by the reranker."""
    n = len(queries)
    order = np.argsort(-first_stage, axis=1)
    cand = order[:, :top_k]
    final = first_stage.copy()
    offset = float(np.abs(first_stage).max()) + 1e6
    oracle = 0
    pairs_q, pairs_p, pair_pos = [], [], []
    for i in range(n):
        if gold[i] in cand[i]:
            oracle += 1
        for j, pidx in enumerate(cand[i]):
            pairs_q.append(queries[i])
            pairs_p.append(passages[pidx])
            pair_pos.append((i, j, pidx))
    print(f"  reranking {len(pairs_q)} pairs (oracle recall@{top_k} = {oracle / n * 100:.2f}%)", flush=True)
    scores = []
    with torch.inference_mode():
        for start in range(0, len(pairs_q), BATCH):
            enc = tok(
                [R.TOK.normalize_orthography(text) for text in pairs_q[start:start + BATCH]],
                [R.TOK.normalize_orthography(text) for text in pairs_p[start:start + BATCH]],
                padding=True, truncation=True, max_length=MAX_LEN,
                return_tensors="pt",
            ).to(DEV)
            logits = model(**enc).logits.squeeze(-1).float().cpu().numpy()
            scores.extend(logits.tolist())
    scores = np.asarray(scores, dtype=np.float32)
    # normalize reranker scores per query so the offset keeps them above non-candidates
    by_query = {}
    for (i, j, pidx), s in zip(pair_pos, scores):
        by_query.setdefault(i, []).append((j, pidx, float(s)))
    for i, items in by_query.items():
        items.sort(key=lambda x: -x[2])
        for rank, (j, pidx, s) in enumerate(items):
            final[i, pidx] = offset - rank
    return final, oracle / n * 100


def headline_data():
    import belebele_experiment as B

    docs, _ = R.load_data()
    q = [d["headline"] for d in docs]
    p = [d["body"] for d in docs]
    gold = np.arange(len(docs))
    # char-4gram BM25
    bm = R.BM25([R.char_ngrams(x) for x in p])
    c4 = np.stack([bm.scores(R.char_ngrams(x)) for x in q])
    # zero-shot equal-weight fusion
    e5_path = os.path.join(R.OUT, "emb_intfloat_multilingual-e5-base_normalized.npz")
    bge_path = os.path.join(R.OUT, "emb_BAAI_bge-m3_normalized.npz")
    if os.path.exists(e5_path) and os.path.exists(bge_path):
        with np.load(e5_path) as e5, np.load(bge_path) as bge:
            e5m = e5["q"] @ e5["p"].T
            bgem = bge["q"] @ bge["p"].T
    else:
        e5m = B.dense("e5", q, p)
        bgem = B.dense("bge", q, p)
    fusion = (zscore(c4) + zscore(e5m) + zscore(bgem)) / 3.0
    return q, p, gold, {"BM25 char-4gram": c4, "Zero-shot equal-weight fusion": fusion}


def belebele_data():
    import belebele_experiment as B
    rows, passages, gold = B.load_dataset()
    queries = [r["question"] for r in rows]
    perm = np.random.default_rng(42).permutation(len(passages))
    val_groups = np.sort(perm[: len(perm) // 2])
    val = np.isin(gold, val_groups)
    test = ~val
    bm = R.BM25([R.char_ngrams(x) for x in passages])
    c4 = np.stack([bm.scores(R.char_ngrams(x)) for x in queries])
    e5m = B.dense("e5", queries, passages)
    bgem = B.dense("bge", queries, passages)
    result = json.load(open(os.path.join(R.OUT, "belebele", "results.json"), encoding="utf-8"))
    zs_w = result["validation_selection"]["zero_shot_weights_char4_e5_bge"]
    prop_w = result["validation_selection"]["proposed_weights_char4_finetuned_e5_bge"]
    zs_fusion = B.weighted([c4, e5m, bgem], zs_w)
    ft, _ = B.train_news_and_encode(queries, passages)
    prop_fusion = B.weighted([c4, ft, bgem], prop_w)
    return queries, passages, gold, test, {
        "BM25 char-4gram": c4,
        "Zero-shot fusion": zs_fusion,
        "Fine-tuned fusion": prop_fusion,
    }


def run(name, queries, passages, gold, first_stages, test_mask=None):
    tok, model = load_reranker()
    rng = np.random.default_rng(SEED)
    out = {"systems": {}}
    ranks = {}
    for sysname, mat in first_stages.items():
        r0 = rank_of_gold(mat, gold)
        final, oracle = rerank_scores(tok, model, queries, passages, mat, gold)
        r1 = rank_of_gold(final, gold)
        ranks[sysname] = r0
        ranks[sysname + " + reranker"] = r1
        if test_mask is None:
            out["systems"][sysname] = metrics(r0)
            out["systems"][sysname + " + reranker"] = metrics(r1)
            out["systems"][sysname + " + reranker"]["oracle_recall_at_k"] = oracle
            out["systems"][sysname + " + reranker"]["vs_first_stage"] = paired(r1, r0, rng)
        else:
            out["systems"][sysname] = metrics(r0[test_mask])
            out["systems"][sysname + " + reranker"] = metrics(r1[test_mask])
            out["systems"][sysname + " + reranker"]["oracle_recall_at_k"] = oracle
            out["systems"][sysname + " + reranker"]["vs_first_stage"] = paired(
                r1[test_mask], r0[test_mask], rng)
        print(f"[{name}] {sysname} -> {out['systems'][sysname]['MRR@10']:.2f} | "
              f"+reranker {out['systems'][sysname + ' + reranker']['MRR@10']:.2f}", flush=True)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return out, ranks


def main():
    print("=== headline ===", flush=True)
    hq, hp, hgold, hfirst = headline_data()
    hout, hranks = run("headline", hq, hp, hgold, hfirst)

    print("=== belebele ===", flush=True)
    bq, bp, bgold, test, bfirst = belebele_data()
    bout, branks = run("belebele", bq, bp, bgold, bfirst, test_mask=test)

    result = {
        "protocol": {
            "reranker": MODEL_ID, "reranker_revision": MODEL_REVISION,
            "top_k": TOP_K, "max_pieces": MAX_LEN,
            "device": DEV, "bootstrap_resamples": BOOT, "seed": SEED,
            "first_stage_headline": "char-4gram BM25 and fixed one-third zero-shot fusion (char-4gram + E5 + BGE-M3 dense)",
            "first_stage_belebele": "char-4gram BM25, validation-tuned zero-shot fusion, and fine-tuned fusion (validation-selected weights)",
            "belebele_test_questions": int(test.sum()),
            "note": "Only the top-k first-stage candidates are reranked; remaining passages keep first-stage order. Oracle recall@k is the fraction of queries whose gold passage is in the first-stage top-k.",
        },
        "headline": hout,
        "belebele_test": bout,
    }
    outdir = Path(R.OUT)
    (outdir / "reranker.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (outdir / "reranker_ranks.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        cols = list(hranks.keys())
        w.writerow(["task", "query_index", *cols])
        for i in range(len(hq)):
            w.writerow(["headline", i, *[int(hranks[k][i]) for k in cols]])
        cols = list(branks.keys())
        w.writerow(["task", "query_index", *cols])
        for i in range(len(bq)):
            w.writerow(["belebele", i, *[int(branks[k][i]) for k in cols]])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
