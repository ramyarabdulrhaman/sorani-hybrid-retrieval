"""KNQAD question-to-paragraph retrieval: a fourth Sorani collection.

KNQAD (Saeed 2024, Mendeley Data doi:10.17632/tc28knsfsn.1, CC BY 4.0): 15,002
manually written questions over Sorani news paragraphs. Queries are questions
(prefix "پرسیار:" removed); the relevant item is the raw source paragraph; all
unique paragraphs are candidates. Paragraph-grouped split (seed 42): 60% train,
20% validation, 20% test. Every choice is made on validation; test is scored once.

Systems: BM25 word / stems / char-4gram (same formula and k1=1.5, b=0.75 as
real_experiments.BM25, computed as a sparse product), zero-shot E5 and bge-m3,
RRF, zero-shot fusion (tuned and equal weights), transfer of the 30 saved news
models (weights tuned on KNQAD validation), and E5 fine-tuned on KNQAD training
pairs (5 seeds, one question per training paragraph, word-BM25 hard negatives).
"""
import hashlib, json, math, os, sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_experiments as R
import proposed_experiment as P
import belebele_experiment as B

ROOT = Path(R.BASE)
DATA = ROOT / "data/external/knqad/knqad.xlsx"
DATA_SHA = "076F394097C7ED2E400E15A55D0ED4078BA8480FB15110B98A75654EE9E88730"
OUT = ROOT / "results/knqad"
SEED, NBOOT = 42, 5000
TRAIN_SEEDS = [1, 2, 3, 4, 5]


def load():
    if hashlib.sha256(DATA.read_bytes()).hexdigest().upper() != DATA_SHA:
        raise ValueError("unexpected KNQAD checksum")
    d = pd.read_excel(DATA)
    paras, pidx, queries, gold = [], {}, [], []
    seen = set()
    for raw, q in zip(d["Raw Paragraph"], d["Question"]):
        if not isinstance(raw, str) or not isinstance(q, str):
            continue
        p = " ".join(raw.split()); q = " ".join(q.split())
        for pre in ("پرسیار:", "پرسیار :"):
            if q.startswith(pre):
                q = q[len(pre):].strip()
        if len(R.words(p)) < 3 or len(R.words(q)) < 1:
            continue
        if p not in pidx:
            pidx[p] = len(paras); paras.append(p)
        key = (q, pidx[p])
        if key in seen:
            continue
        seen.add(key); queries.append(q); gold.append(pidx[p])
    return paras, queries, np.asarray(gold)


class SparseBM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.vocab = {}
        rows, cols, vals = [], [], []
        dl = np.array([len(d) for d in docs], float); avg = dl.mean()
        df = Counter()
        for i, d in enumerate(docs):
            c = Counter(d); df.update(c.keys())
            for t, f in c.items():
                j = self.vocab.setdefault(t, len(self.vocab))
                rows.append(i); cols.append(j)
                vals.append(f * (k1 + 1) / (f + k1 * (1 - b + b * dl[i] / avg)))
        n = len(docs)
        idf = np.zeros(len(self.vocab))
        for t, j in self.vocab.items():
            idf[j] = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
        W = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(self.vocab)))
        self.W = (W @ sp.diags(idf)).T.tocsr()  # terms x docs

    def scores(self, qs):
        rows, cols = [], []
        for i, q in enumerate(qs):
            for t in set(q):
                j = self.vocab.get(t)
                if j is not None:
                    rows.append(i); cols.append(j)
        Q = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(qs), len(self.vocab)))
        return np.asarray((Q @ self.W).todense(), dtype=np.float32)


def rr_of(s, g):
    r = B.ranks(s, g)
    return np.where(r <= 10, 1.0 / r, 0.0)


def zs(m):
    return (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-9)


def select(zmats, g, required=None):
    best, bw = -1, None
    for w in P.GRID:
        if required is not None and w[required] < 0.1:
            continue
        s = rr_of(sum(a * m for a, m in zip(w, zmats)), g).mean()
        if s > best + 1e-12:
            best, bw = s, w
    return bw, float(best * 100)


def cluster_boot(diff, groups):
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=diff); cnts = np.bincount(inv)
    rng = np.random.default_rng(SEED); s = np.empty(NBOOT)
    for k in range(NBOOT):
        c = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        s[k] = (c * sums).sum() / (c * cnts).sum() * 100
    p = max(min(1.0, 2 * min((s <= 0).mean(), (s >= 0).mean())), 1 / NBOOT)
    return float(diff.mean() * 100), [float(x) for x in np.percentile(s, [2.5, 97.5])], p


def holm(ps):
    ps = np.asarray(ps); o = ps.argsort(); a = np.empty_like(ps); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); a[i] = run
    return a


def overlap(q, p):
    qs = {w for w in R.words(q) if len(w) >= 3}
    return len(qs & set(R.words(p))) / len(qs) if qs else 0.0


def encode_with(path, texts_q, texts_p, mode="raw"):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).to(P.DEV).eval()
    if mode == "split":
        texts_q = [P.split_text(x) for x in texts_q]; texts_p = [P.split_text(x) for x in texts_p]
    with torch.autocast("cuda", dtype=torch.float16):
        q = P.encode(model, tok, [P.QP + x for x in texts_q], P.MAX_Q, bs=64)
        p = P.encode(model, tok, [P.PP + x for x in texts_p], P.MAX_P, bs=32)
    del model; torch.cuda.empty_cache()
    return q, p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paras, queries, gold = load()
    perm = np.random.default_rng(SEED).permutation(len(paras))
    n_tr, n_va = int(0.6 * len(paras)), int(0.2 * len(paras))
    part = np.empty(len(paras), int); part[perm[:n_tr]] = 0; part[perm[n_tr:n_tr + n_va]] = 1; part[perm[n_tr + n_va:]] = 2
    qpart = part[gold]
    tr, va, te = (np.flatnonzero(qpart == k) for k in range(3))
    ev = np.concatenate([va, te]); gv, gt = gold[va], gold[te]
    print(f"{len(queries)} questions, {len(paras)} paragraphs; train {len(tr)} / val {len(va)} / test {len(te)}", flush=True)
    evq = [queries[i] for i in ev]; nv = len(va)

    def split(m):
        return m[:nv], m[nv:]

    lex = {}
    for name, tk in (("BM25 word", R.words), ("BM25 stems", R.morph), ("BM25 char-4gram", R.char_ngrams)):
        print("scoring", name, flush=True)
        lex[name] = SparseBM25([tk(p) for p in paras]).scores([tk(q) for q in evq])
    cache = OUT / "cache"; cache.mkdir(exist_ok=True)

    def dense(name):
        f = cache / f"{name}.npy"
        if f.exists():
            return np.load(f)
        mid, rev, pool = B.MODELS[name]
        s = B.dense(name, evq, paras).astype(np.float32); np.save(f, s); return s
    e5, bge = dense("e5"), dense("bge")
    c4 = lex["BM25 char-4gram"]
    zc4, ze5, zbge = zs(c4), zs(e5), zs(bge)

    rr = {}  # system -> test rr (averaged over models where applicable)
    for name, m in lex.items():
        rr[name] = rr_of(split(m)[1], gt)
    rr["E5 zero-shot"] = rr_of(split(e5)[1], gt)
    rr["BGE-M3 zero-shot"] = rr_of(split(bge)[1], gt)
    rr["RRF char-4gram + BGE-M3"] = rr_of(split(R.rrf([c4, bge]))[1], gt)
    zw, _ = select([split(x)[0] for x in (zc4, ze5, zbge)], gv)
    rr["Zero-shot fusion (tuned weights)"] = rr_of(split(zw[0] * zc4 + zw[1] * ze5 + zw[2] * zbge)[1], gt)
    rr["Zero-shot equal-weight fusion"] = rr_of(split((zc4 + ze5 + zbge) / 3)[1], gt)

    def fused_systems(ft, label):
        zft = zs(ft)
        w, v = select([split(x)[0] for x in (zc4, zft, zbge)], gv, required=1)
        return (rr_of(split(ft)[1], gt), rr_of(split(w[0] * zc4 + w[1] * zft + w[2] * zbge)[1], gt),
                rr_of(split((zc4 + zft + zbge) / 3)[1], gt), w, v)

    # Transfer of the 30 saved news models
    log = json.loads((ROOT / "results/proposed_selection_log.json").read_text(encoding="utf-8-sig"))
    finals = {(e["partition_seed"], e["fold"], e["seed"]): e["FINAL_selected"] for e in log if "FINAL_selected" in e}
    news = {"ft": [], "tuned": [], "equal": [], "runs": []}
    for (ps, fd, sd), cfg in sorted(finals.items()):
        path = ROOT / f"results/checkpoints/partition_{ps}/fold_{fd}/seed_{sd}"
        q, p = encode_with(path, evq, paras, cfg["mode"])
        a, b_, c, w, v = fused_systems(q @ p.T, "news")
        news["ft"].append(a); news["tuned"].append(b_); news["equal"].append(c)
        news["runs"].append({"partition": ps, "fold": fd, "seed": sd, "weights": list(w), "val": v, "test": float(b_.mean() * 100)})
        print(f"news p{ps} f{fd} s{sd}: w={w} test={b_.mean()*100:.2f}", flush=True)
    R1 = {"E5 news-fine-tuned (30)": news["ft"], "News-trained fused retriever (30)": news["tuned"], "News fine-tuned equal-weight fusion (30)": news["equal"]}
    rr["E5 news-fine-tuned (30)"] = np.mean(news["ft"], 0)
    rr["News-trained fused retriever (30)"] = np.mean(news["tuned"], 0)
    rr["News fine-tuned equal-weight fusion (30)"] = np.mean(news["equal"], 0)

    # E5 fine-tuned on KNQAD training pairs
    first = {}
    for i in tr:
        first.setdefault(gold[i], i)
    tq = [queries[i] for i in first.values()]; tp_idx = list(first.keys())
    tr_paras = [paras[j] for j in tp_idx]
    wb = SparseBM25([R.words(p) for p in tr_paras]); hs = wb.scores([R.words(q) for q in tq])
    hs[np.arange(len(tq)), np.arange(len(tq))] = -1e9
    hard = hs.argmax(1).tolist(); del hs
    P.MODEL_ID = snapshot_download(repo_id=B.MODELS["e5"][0], revision=B.MODELS["e5"][1],
                                   allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                                                   "special_tokens_map.json", "sentencepiece.bpe.model"])
    kq = {"ft": [], "tuned": [], "equal": [], "runs": []}
    for sd in TRAIN_SEEDS:
        ck = OUT / "checkpoints" / f"seed_{sd}"
        if not (ck / "model.safetensors").exists():
            print(f"training KNQAD E5 seed {sd} on {len(tq)} pairs", flush=True)
            P.train_checkpoints(tq, tr_paras, hard, 1e-5, sd, [2], lambda m, t: None,
                                lambda msg: print(msg, flush=True), checkpoint_dir=str(ck))
        q, p = encode_with(ck, evq, paras)
        a, b_, c, w, v = fused_systems(q @ p.T, "knqad")
        kq["ft"].append(a); kq["tuned"].append(b_); kq["equal"].append(c)
        kq["runs"].append({"seed": sd, "weights": list(w), "val": v, "test": float(b_.mean() * 100)})
        print(f"knqad seed {sd}: w={w} test={b_.mean()*100:.2f}", flush=True)
    R1.update({"E5 KNQAD-fine-tuned (5)": kq["ft"], "KNQAD-trained fused retriever (5)": kq["tuned"], "KNQAD fine-tuned equal-weight fusion (5)": kq["equal"]})
    rr["E5 KNQAD-fine-tuned (5)"] = np.mean(kq["ft"], 0)
    rr["KNQAD-trained fused retriever (5)"] = np.mean(kq["tuned"], 0)
    rr["KNQAD fine-tuned equal-weight fusion (5)"] = np.mean(kq["equal"], 0)

    systems = {k: {"MRR@10": float(v.mean() * 100), "Recall@1": float(np.mean([(x == 1).mean() for x in R1.get(k, [v])]) * 100)} for k, v in rr.items()}
    comps = {}
    for prop in ("News-trained fused retriever (30)", "KNQAD-trained fused retriever (5)"):
        cmp = {}
        for k, v in rr.items():
            if k == prop: continue
            d, ci, pval = cluster_boot(rr[prop] - v, gt)
            cmp[k] = {"diff": d, "CI95": ci, "p": pval}
        for k, a in zip(cmp, holm([c["p"] for c in cmp.values()])):
            cmp[k]["p_holm"] = float(a)
        comps[prop] = cmp
    ov_all = np.array([overlap(queries[i], paras[gold[i]]) for i in range(len(queries))])
    ov = ov_all[te]; cuts = np.quantile(ov, [1/3, 2/3]); terc = np.digitize(ov, cuts)
    terciles = {}
    for prop in ("News-trained fused retriever (30)", "KNQAD-trained fused retriever (5)"):
        terciles[prop] = {}
        for t, nm in enumerate(("low", "mid", "high")):
            m = terc == t
            d, ci, _ = cluster_boot(rr[prop][m] - rr["BM25 char-4gram"][m], gt[m])
            terciles[prop][nm] = {"n": int(m.sum()), "diff_vs_char4": d, "CI95": ci,
                                  "overlap_range": [float(ov[m].min()), float(ov[m].max())]}
    res = {"protocol": {"source": "https://data.mendeley.com/datasets/tc28knsfsn/1", "doi": "10.17632/tc28knsfsn.1",
                        "license": "CC BY 4.0", "sha256": DATA_SHA, "questions": len(queries), "paragraphs": len(paras),
                        "split": "paragraph-grouped 60/20/20, seed 42", "train_pairs": len(tq),
                        "train_questions": int(len(tr)), "validation_questions": int(len(va)), "test_questions": int(len(te)),
                        "candidates": "all unique raw paragraphs",
                        "bootstrap": f"{NBOOT} paired paragraph-cluster resamples, seed {SEED}; Holm within each family",
                        "knqad_training": "multilingual-e5-base, lr 1e-5, 2 epochs, batch 8, InfoNCE tau 0.05, one word-BM25 hard negative"},
           "overlap": {"median_all": float(np.median(ov_all)), "mean_all": float(ov_all.mean()),
                       "median_test": float(np.median(ov)), "share_ge_0.5": float((ov_all >= 0.5).mean())},
           "zero_shot_weights": list(zw), "systems": systems, "comparisons": comps, "terciles_vs_char4": terciles,
           "news_runs": news["runs"], "knqad_runs": kq["runs"]}
    (OUT / "knqad_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("overlap", "systems", "terciles_vs_char4")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
