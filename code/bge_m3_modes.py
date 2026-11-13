"""BGE-M3 retrieval in dense, sparse (lexical), and multi-vector (ColBERT) modes.

The cached BAAI/bge-m3 checkpoint used elsewhere in the project stores the
dense encoder only. The same pinned revision also publishes `sparse_linear.pt`
and `colbert_linear.pt`, which this script loads to reproduce the three
retrieval modes of BGE-M3. All modes use the same 512-piece passage limit as
the rest of the benchmark.

Outputs
-------
results/bge_m3_modes.json          metrics, CIs, paired tests
results/bge_m3_modes_ranks.csv     per-query gold ranks (headline + Belebele)
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_experiments as R

REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
MODEL_ID = "BAAI/bge-m3"
MAX_LEN = 512
QUERY_LEN = 128
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BOOT = 5000
SEED = 42


def get_model_dir():
    return snapshot_download(
        MODEL_ID,
        revision=REVISION,
        allow_patterns=[
            "config.json", "pytorch_model.bin", "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json",
            "sentencepiece.bpe.model", "sparse_linear.pt", "colbert_linear.pt",
        ],
    )


def load_model(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16).to(DEV).eval()
    sparse = torch.load(os.path.join(model_dir, "sparse_linear.pt"), map_location=DEV)
    colbert = torch.load(os.path.join(model_dir, "colbert_linear.pt"), map_location=DEV)
    sparse_w = sparse["weight"].to(DEV).float()
    sparse_b = sparse["bias"].to(DEV).float()
    colbert_w = colbert["weight"].to(DEV).float()
    colbert_b = colbert["bias"].to(DEV).float()
    return tok, model, (sparse_w, sparse_b), (colbert_w, colbert_b)


def encode_all(tok, model, texts, max_len, batch_size=8):
    """Return CLS dense vectors, sparse token weights, and ColBERT token vectors."""
    dense, sparse_rows, colbert_list, lengths = [], [], [], []
    special = set(tok.all_special_ids)
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = [
                R.TOK.normalize_orthography(text)
                for text in texts[start:start + batch_size]
            ]
            enc = tok(batch, padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(DEV)
            hidden = model(**enc).last_hidden_state.float()
            mask = enc["attention_mask"].bool()
            # dense: CLS, L2 normalized (matches the project's BGE-M3 baseline)
            d = torch.nn.functional.normalize(hidden[:, 0], dim=-1)
            dense.append(d.cpu().numpy())
            # sparse and colbert heads
            ids = enc["input_ids"]
            for b in range(len(batch)):
                valid = mask[b] & ~torch.isin(ids[b], torch.tensor(list(special), device=DEV))
                h = hidden[b][valid]
                # sparse: ReLU linear token weights
                sw = torch.relu(h @ model.sparse_w.T + model.sparse_b).squeeze(-1)
                # colbert: projected, L2-normalized token vectors
                cv = torch.nn.functional.normalize(h @ model.colbert_w.T + model.colbert_b, dim=-1)
                sparse_rows.append((ids[b][valid].cpu().numpy(), sw.cpu().numpy()))
                colbert_list.append(cv.cpu().numpy().astype(np.float16))
                lengths.append(cv.shape[0])
    return np.concatenate(dense), sparse_rows, colbert_list, np.asarray(lengths)


def build_sparse(rows, vocab_size):
    import scipy.sparse as sp
    indptr, indices, data = [0], [], []
    for ids, weights in rows:
        # aggregate max weight per token id
        best = {}
        for t, w in zip(ids, weights):
            t = int(t)
            if w > best.get(t, -1.0):
                best[t] = float(w)
        for t in sorted(best):
            indices.append(t)
            data.append(best[t])
        indptr.append(len(indices))
    return sp.csr_matrix((data, indices, indptr), shape=(len(rows), vocab_size))


def build_colbert_tensor(vectors, max_len):
    n = len(vectors)
    dim = vectors[0].shape[1]
    out = np.zeros((n, max_len, dim), dtype=np.float16)
    mask = np.zeros((n, max_len), dtype=bool)
    for i, v in enumerate(vectors):
        L = min(len(v), max_len)
        out[i, :L] = v[:L]
        mask[i, :L] = True
    return torch.from_numpy(out).to(DEV), torch.from_numpy(mask).to(DEV)


def colbert_scores(qvecs, qlens, pvecs, pmask):
    """Late-interaction scores: sum_q max_p dot(q_t, p_t)."""
    n_p, lp, dim = pvecs.shape
    flat = pvecs.reshape(-1, dim)
    scores = np.zeros((len(qvecs), n_p), dtype=np.float32)
    with torch.inference_mode():
        for i, (q, L) in enumerate(zip(qvecs, qlens)):
            qt = torch.from_numpy(q[:L]).to(DEV).half()
            sim = qt @ flat.T  # (L, n_p*lp), fp16
            sim = sim.reshape(L, n_p, lp)
            sim = sim.masked_fill(~pmask.unsqueeze(0), float("-inf"))
            scores[i] = sim.max(dim=2).values.sum(dim=0).float().cpu().numpy()
    return scores


def sparse_scores(qrows, prows, vocab_size):
    Q = build_sparse(qrows, vocab_size)
    P = build_sparse(prows, vocab_size)
    return np.asarray((Q @ P.T).todense(), dtype=np.float32)


def rank_of_gold(scores, gold):
    correct = scores[np.arange(len(gold)), gold][:, None]
    return 1 + (scores > correct).sum(1) + (scores == correct).sum(1) - 1


def summarize(ranks, gold, rng):
    r = ranks
    rr = np.where(r <= 10, 1.0 / r, 0)
    boots = np.array([rr[rng.integers(0, len(r), len(r))].mean() * 100 for _ in range(BOOT)])
    return {
        "Recall@1": float((r <= 1).mean() * 100),
        "Recall@5": float((r <= 5).mean() * 100),
        "Recall@10": float((r <= 10).mean() * 100),
        "MRR@10": float(rr.mean() * 100),
        "MRR@10_CI95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "nDCG@10": float(np.where(r <= 10, 1.0 / np.log2(r + 1), 0).mean() * 100),
    }


def paired(ranks, base_ranks, rng):
    a = np.where(ranks <= 10, 1.0 / ranks, 0)
    b = np.where(base_ranks <= 10, 1.0 / base_ranks, 0)
    d = a - b
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(BOOT)])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {
        "mean_MRR@10_diff": float(d.mean() * 100),
        "CI95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "p_two_sided": min(1.0, max(float(p), 1.0 / BOOT)),
    }


def zscore(m):
    return (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-9)


def run_corpus(name, queries, passages, gold, test_mask=None):
    model_dir = get_model_dir()
    tok, model, (sparse_w, sparse_b), (colbert_w, colbert_b) = load_model(model_dir)
    # attach heads for encode_all
    model.sparse_w, model.sparse_b = sparse_w, sparse_b
    model.colbert_w, model.colbert_b = colbert_w, colbert_b
    vocab_size = len(tok)

    print(f"[{name}] encoding {len(passages)} passages", flush=True)
    pd, ps, pc, pl = encode_all(tok, model, passages, MAX_LEN, batch_size=4)
    print(f"[{name}] encoding {len(queries)} queries", flush=True)
    qd, qs, qc, ql = encode_all(tok, model, queries, QUERY_LEN, batch_size=16)

    dense = qd @ pd.T
    sparse = sparse_scores(qs, ps, vocab_size)
    pvecs, pmask = build_colbert_tensor(pc, MAX_LEN)
    multi = colbert_scores(qc, ql, pvecs, pmask)
    combined = zscore(dense) + zscore(sparse) + zscore(multi)

    mats = {
        "BGE-M3 dense": dense,
        "BGE-M3 sparse": sparse,
        "BGE-M3 multi-vector": multi,
        "BGE-M3 dense+sparse+multi-vector": combined,
    }
    all_ranks = {k: rank_of_gold(v, gold) for k, v in mats.items()}
    rng = np.random.default_rng(SEED)
    out = {"n_queries": len(queries), "n_passages": len(passages), "systems": {}}
    for k, r in all_ranks.items():
        if test_mask is None:
            out["systems"][k] = summarize(r, gold, rng)
        else:
            out["systems"][k] = summarize(r[test_mask], gold[test_mask], rng)
    # paired vs dense
    for k in ["BGE-M3 sparse", "BGE-M3 multi-vector", "BGE-M3 dense+sparse+multi-vector"]:
        if test_mask is None:
            out["systems"][k]["vs_dense"] = paired(all_ranks[k], all_ranks["BGE-M3 dense"], rng)
        else:
            out["systems"][k]["vs_dense"] = paired(all_ranks[k][test_mask], all_ranks["BGE-M3 dense"][test_mask], rng)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return out, all_ranks


def main():
    docs, stats = R.load_data()
    hq = [d["headline"] for d in docs]
    hp = [d["body"] for d in docs]
    hgold = np.arange(len(docs))
    print("=== headline ===", flush=True)
    head, hranks = run_corpus("headline", hq, hp, hgold)

    # Belebele: reuse the exact split from belebele_experiment.py
    rows = [json.loads(line) for line in open("data/raw/belebele_ckb_Arab.jsonl", encoding="utf-8")]
    passages, indices, gold = [], {}, []
    for row in rows:
        key = (row["link"], row["ds"])
        if key not in indices:
            indices[key] = len(passages)
            passages.append(row["flores_passage"])
        gold.append(indices[key])
    gold = np.asarray(gold)
    bq = [r["question"] for r in rows]
    perm = np.random.default_rng(42).permutation(len(passages))
    val_groups = np.sort(perm[: len(perm) // 2])
    val = np.isin(gold, val_groups)
    test = ~val
    print(f"=== belebele: {test.sum()} test questions, {len(passages)} passages ===", flush=True)
    bel, branks = run_corpus("belebele", bq, passages, gold, test_mask=test)

    result = {
        "protocol": {
            "model": MODEL_ID, "revision": REVISION,
            "dense": "CLS pooling, L2 normalized",
            "sparse": "ReLU(sparse_linear(hidden)); max weight per input token; sparse dot product",
            "multi_vector": "L2-normalized colbert_linear(hidden); late interaction (sum of max dot products)",
            "combined": "equal-weight sum of per-query z-scored dense, sparse, and multi-vector scores",
            "passage_max_pieces": MAX_LEN, "query_max_pieces": QUERY_LEN,
            "device": DEV, "bootstrap_resamples": BOOT, "seed": SEED,
            "headline": "492 queries over 492 passages",
            "belebele": "447 test questions over 488 passages, same split as belebele_experiment.py",
            "note": "sparse_linear.pt and colbert_linear.pt from the pinned BGE-M3 revision; the dense encoder is the same checkpoint used elsewhere.",
        },
        "headline": head,
        "belebele_test": bel,
    }
    outdir = Path(R.OUT)
    (outdir / "bge_m3_modes.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (outdir / "bge_m3_modes_ranks.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "query_index", *hranks.keys()])
        for i in range(len(hq)):
            w.writerow(["headline", i, *[int(hranks[k][i]) for k in hranks]])
        for i in range(len(bq)):
            w.writerow(["belebele", i, *[int(branks[k][i]) for k in branks]])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
