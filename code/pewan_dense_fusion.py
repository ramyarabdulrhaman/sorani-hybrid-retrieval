"""Dense and fused retrieval on Pewan Sorani (22 topics, 115k documents).

No Pewan data is used for any choice: each fine-tuned E5 model keeps the fusion
weights it selected on its own news validation split, and equal-weight fusion
uses fixed one-third weights. Models: partition 42, fold 0, training seeds 1-5.
Topic-level paired bootstrap (2,000 resamples, seed 42) with Holm correction.
"""
import argparse, json, math, os, sys
from array import array
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_experiments as R
import proposed_experiment as P
from pewan_evaluation import load_topics, score_query, query_metrics

ROOT = Path(R.BASE)
CACHE = ROOT / "results" / "pewan_cache"
MODELS = [(42, 0, s) for s in (1, 2, 3, 4, 5)]
METRICS = ("MRR@10", "nDCG@10", "P@10", "MAP", "Recall@100")
B = 2000


def encode(model, tok, texts, max_len, bs, pool):
    order = np.argsort([len(t) for t in texts])[::-1]
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(texts), bs):
            ids = order[i:i + bs]
            batch = [R.TOK.normalize_orthography(texts[j]) for j in ids]
            enc = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(P.DEV)
            h = model(**enc).last_hidden_state
            v = h[:, 0] if pool == "cls" else P.mean_pool(h, enc["attention_mask"])
            out[ids] = torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy().astype(np.float16)
            if (i // bs) % 500 == 0:
                print(f"  encoded {i:,}/{len(texts):,}", flush=True)
    return out


def dense_scores(name, path, queries, docs, qprefix, pprefix, pool, mode="raw"):
    f = CACHE / f"{name}.npy"
    if f.exists():
        return np.load(f)
    print("Encoding", name, flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).to(P.DEV).eval()
    if mode == "split":
        queries = [P.split_text(x) for x in queries]; docs = [P.split_text(x) for x in docs]
    q = encode(model, tok, [qprefix + x for x in queries], 64, 16, pool).astype(np.float32)
    d = encode(model, tok, [pprefix + x for x in docs], 512, 32, pool).astype(np.float32)
    del model; torch.cuda.empty_cache()
    s = q @ d.T
    np.save(f, s)
    return s


def zs(m):
    return (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-9)


def boot(diff, rng):
    n = len(diff); s = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(B)])
    p = max(min(1.0, 2 * min((s <= 0).mean(), (s >= 0).mean())), 1 / B)
    return [float(x) for x in np.percentile(s, [2.5, 97.5])], p


def holm(ps):
    ps = np.asarray(ps); o = ps.argsort(); a = np.empty_like(ps); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); a[i] = run
    return a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pewan-root", type=Path, required=True)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    topics, qrels = load_topics(args.pewan_root)
    paths = sorted((args.pewan_root / "Corpora" / "Sorani" / "docs").glob("*.txt"), key=lambda p: int(p.stem))
    idx = {p.stem: i for i, p in enumerate(paths)}
    print(f"Reading {len(paths):,} documents", flush=True)
    docs = [p.read_text(encoding="utf-8") for p in paths]
    qids = sorted(topics); queries = [topics[q] for q in qids]
    rel = [np.asarray([idx[d] for d in qrels[q]]) for q in qids]

    c4f = CACHE / "bm25_c4.npy"
    if c4f.exists():
        c4 = np.load(c4f)
    else:
        qterms = [R.char_ngrams(q) for q in queries]; vocab = set(t for ts in qterms for t in ts)
        post = {t: (array("I"), array("H")) for t in vocab}; dl = np.zeros(len(docs), np.int32)
        for i, d in enumerate(docs):
            toks = R.char_ngrams(d); dl[i] = len(toks)
            for t, tf in Counter(x for x in toks if x in post).items():
                post[t][0].append(i); post[t][1].append(min(tf, 65535))
        index = {"postings": post, "doc_len": dl, "avgdl": float(dl.mean())}
        c4 = np.stack([score_query(index, t) for t in qterms]); np.save(c4f, c4)

    e5 = dense_scores("e5_zs", "intfloat/multilingual-e5-base", queries, docs, "query: ", "passage: ", "mean")
    bge = dense_scores("bge_zs", "BAAI/bge-m3", queries, docs, "", "", "cls")

    log = json.loads((ROOT / "results/proposed_selection_log.json").read_text(encoding="utf-8-sig"))
    finals = {(e["partition_seed"], e["fold"], e["seed"]): e["FINAL_selected"] for e in log if "FINAL_selected" in e}
    zc4, ze5, zbge = zs(c4), zs(e5), zs(bge)
    sys_scores = {"BM25 char-4gram": [c4], "E5 zero-shot": [e5], "BGE-M3 zero-shot": [bge],
                  "Zero-shot equal-weight fusion": [(zc4 + ze5 + zbge) / 3],
                  "E5 news-fine-tuned": [], "Fine-tuned equal-weight fusion": [], "Proposed fused retriever": []}
    used = []
    for ps, fd, sd in MODELS:
        cfg = finals[(ps, fd, sd)]
        ft = dense_scores(f"ft_p{ps}_f{fd}_s{sd}", ROOT / f"results/checkpoints/partition_{ps}/fold_{fd}/seed_{sd}",
                          queries, docs, "query: ", "passage: ", "mean", cfg["mode"])
        zft = zs(ft); w = cfg["w"]
        sys_scores["E5 news-fine-tuned"].append(ft)
        sys_scores["Fine-tuned equal-weight fusion"].append((zc4 + zft + zbge) / 3)
        sys_scores["Proposed fused retriever"].append(w[0] * zc4 + w[1] * zft + w[2] * zbge)
        used.append({"partition": ps, "fold": fd, "seed": sd, "news_selected_weights": w, "mode": cfg["mode"]})

    per = {}
    for name, mats in sys_scores.items():
        vals = np.array([[[query_metrics(m[i], rel[i])[k] for k in METRICS] for i in range(len(qids))] for m in mats])
        per[name] = vals.mean(0)  # topics x metrics, averaged over models
    rng = np.random.default_rng(42)
    systems = {n: {k: float(v[:, j].mean()) for j, k in enumerate(METRICS)} for n, v in per.items()}
    PROP = "Proposed fused retriever"; comps = {}
    for n in per:
        if n == PROP: continue
        comps[n] = {}
        for j, k in enumerate(METRICS):
            d = per[PROP][:, j] - per[n][:, j]; ci, p = boot(d, rng)
            comps[n][k] = {"diff": float(d.mean()), "CI95": ci, "p": p,
                           "topics_better_worse": [int((d > 1e-9).sum()), int((d < -1e-9).sum())]}
    for k in METRICS:
        adj = holm([comps[n][k]["p"] for n in comps])
        for n, a in zip(comps, adj): comps[n][k]["p_holm"] = float(a)
    res = {"protocol": {"documents": len(docs), "topics": len(qids), "models": used,
                        "weights": "news-validation-selected per model; no Pewan tuning",
                        "dense_truncation": "queries 64, documents 512 pieces; fp16 inference",
                        "bootstrap": f"{B} topic resamples, seed 42; Holm over {len(comps)} comparisons per metric"},
           "systems": systems, "comparisons_proposed_minus": comps}
    (ROOT / "results/pewan_dense_fusion.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
