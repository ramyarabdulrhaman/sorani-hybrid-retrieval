"""Belebele transfer evaluation of all 30 saved news cross-fitted checkpoints.

Same passage-grouped validation/test split as belebele_experiment.py (seed 42).
For every saved checkpoint (2 partitions x 3 folds x 5 seeds), fusion weights for
char-4gram BM25 + that fine-tuned E5 + zero-shot bge-m3 are selected on the
validation questions only (fine-tuned weight >= 0.1, as in the primary analysis);
test reciprocal ranks are averaged over the 30 models. Paired passage-cluster
bootstrap CIs, Holm correction over the comparison family.
"""
import json, os, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_experiments as R
import proposed_experiment as P
import belebele_experiment as B

ROOT = Path(R.BASE)
OUT = ROOT / "results" / "belebele"
CACHE = OUT / "belebele_30model_cache"
CKPT = ROOT / "results" / "checkpoints"


def rr_of(scores, gold):
    r = B.ranks(scores, gold)
    return np.where(r <= 10, 1.0 / r, 0.0), r


def holm(ps):
    ps = np.asarray(ps, float); order = ps.argsort(); adj = np.empty_like(ps); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


def cluster_boot(diff, groups, n=B.BOOTSTRAP_RESAMPLES):
    uniq = np.unique(groups); per = [np.flatnonzero(groups == g) for g in uniq]
    rng = np.random.default_rng(B.SEED); s = np.empty(n)
    for k in range(n):
        ids = np.concatenate([per[i] for i in rng.integers(0, len(uniq), len(uniq))])
        s[k] = diff[ids].mean() * 100
    obs = diff.mean() * 100
    p = min(1.0, 2 * min((s <= 0).mean(), (s >= 0).mean()))
    return obs, [float(x) for x in np.percentile(s, [2.5, 97.5])], max(p, 1 / n)


def overlap(q, p):
    qs = {w for w in R.words(q) if len(w) >= 3}
    ps = set(R.words(p))
    return len(qs & ps) / len(qs) if qs else 0.0


def main():
    rows, passages, gold = B.load_dataset()
    queries = [r["question"] for r in rows]
    perm = np.random.default_rng(B.SEED).permutation(len(passages))
    val = np.isin(gold, np.sort(perm[:len(perm) // 2])); test = ~val
    CACHE.mkdir(parents=True, exist_ok=True)

    def cached(name, fn):
        f = CACHE / f"{name}.npy"
        if f.exists():
            return np.load(f)
        m = fn(); np.save(f, m); return m

    idx = R.BM25([R.char_ngrams(p) for p in passages])
    c4 = cached("bm25_c4", lambda: np.stack([idx.scores(R.char_ngrams(q)) for q in queries]))
    e5 = cached("e5_zs", lambda: B.dense("e5", queries, passages))
    bge = cached("bge_zs", lambda: B.dense("bge", queries, passages))

    log = json.loads((ROOT / "results/proposed_selection_log.json").read_text(encoding="utf-8-sig"))
    finals = {}
    for e in log:
        if "FINAL_selected" in e:
            finals[(e["partition_seed"], e["seed"], e["fold"])] = e["FINAL_selected"]
    assert len(finals) == 30

    split_q = split_p = None
    tuned_rr, equal_rr, ft_rr, runs = [], [], [], []
    for (ps, sd, fd), cfg in sorted(finals.items()):
        def enc():
            nonlocal split_q, split_p
            path = CKPT / f"partition_{ps}" / f"fold_{fd}" / f"seed_{sd}"
            tok = AutoTokenizer.from_pretrained(path)
            model = AutoModel.from_pretrained(path).to(P.DEV).eval()
            qs, pss = queries, passages
            if cfg["mode"] == "split":
                if split_q is None:
                    split_q = [P.split_text(x) for x in queries]; split_p = [P.split_text(x) for x in passages]
                qs, pss = split_q, split_p
            q = P.encode(model, tok, [P.QP + x for x in qs], P.MAX_Q, bs=16)
            p = P.encode(model, tok, [P.PP + x for x in pss], P.MAX_P, bs=4)
            del model; torch.cuda.empty_cache()
            return q @ p.T
        ft = cached(f"ft_p{ps}_f{fd}_s{sd}", enc)
        comps = [c4, ft, bge]
        w, vscore = B.select_weights([m[val] for m in comps], gold[val], required_index=1)
        tuned_rr.append(rr_of(B.weighted(comps, w), gold)[0])
        equal_rr.append(rr_of(B.weighted(comps, (1/3, 1/3, 1/3)), gold)[0])
        ft_rr.append(rr_of(ft, gold)[0])
        runs.append({"partition_seed": ps, "seed": sd, "fold": fd, "mode": cfg["mode"], "weights": list(w),
                     "val_MRR@10": vscore,
                     "test_MRR@10": float(tuned_rr[-1][test].mean() * 100)})
        print(f"p{ps} f{fd} s{sd}: w={w} test={runs[-1]['test_MRR@10']:.2f}", flush=True)

    zw, _ = B.select_weights([m[val] for m in (c4, e5, bge)], gold[val])
    sysrr = {
        "BM25 char-4gram": rr_of(c4, gold)[0],
        "E5 zero-shot": rr_of(e5, gold)[0],
        "BGE-M3 zero-shot": rr_of(bge, gold)[0],
        "RRF char-4gram + BGE-M3": rr_of(R.rrf([c4, bge]), gold)[0],
        "Zero-shot fusion (tuned weights)": rr_of(B.weighted([c4, e5, bge], zw), gold)[0],
        "Zero-shot equal-weight fusion": rr_of(B.weighted([c4, e5, bge], (1/3, 1/3, 1/3)), gold)[0],
        "E5 news-fine-tuned (30 models)": np.mean(ft_rr, 0),
        "Fine-tuned equal-weight fusion (30 models)": np.mean(equal_rr, 0),
    }
    PROP = "Proposed fused retriever (30 models)"
    sysrr[PROP] = np.mean(tuned_rr, 0)
    run_mrr = np.array([r["test_MRR@10"] for r in runs])

    g = gold[test]
    systems = {k: {"MRR@10": float(v[test].mean() * 100)} for k, v in sysrr.items()}
    systems[PROP]["MRR@10_run_SD"] = float(run_mrr.std(ddof=1))
    systems[PROP]["MRR@10_run_range"] = [float(run_mrr.min()), float(run_mrr.max())]
    comps = {}
    for k, v in sysrr.items():
        if k == PROP: continue
        d, ci, p = cluster_boot(sysrr[PROP][test] - v[test], g)
        comps[k] = {"diff": d, "CI95": ci, "p": p,
                    "runs_above": int((run_mrr > v[test].mean() * 100 + 1e-9).sum())}
    for k, a in zip(comps, holm([c["p"] for c in comps.values()])):
        comps[k]["p_holm"] = float(a)

    ov = np.array([overlap(queries[i], passages[gold[i]]) for i in range(len(queries))])[test]
    cuts = np.quantile(ov, [1/3, 2/3]); terc = np.digitize(ov, cuts)
    tercile = {}
    for t, name in enumerate(["low", "mid", "high"]):
        m = terc == t
        d, ci, p = cluster_boot(sysrr[PROP][test][m] - sysrr["BM25 char-4gram"][test][m], g[m])
        tercile[name] = {"n": int(m.sum()), "diff_vs_char4": d, "CI95": ci}

    res = {"protocol": {"split": "belebele_experiment.py seed 42; 453 validation / 447 test questions; 488 candidates",
                        "models": "30 saved news cross-fitted checkpoints, weights selected per model on validation",
                        "bootstrap": f"{B.BOOTSTRAP_RESAMPLES} paired passage-cluster resamples; p two-sided, floored at 1/{B.BOOTSTRAP_RESAMPLES}; Holm over {len(comps)} comparisons"},
           "zero_shot_weights": list(zw), "systems": systems, "comparisons": comps,
           "overlap_terciles_vs_char4": tercile, "runs": runs}
    (OUT / "belebele_30model_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"systems": systems, "comparisons": comps, "terciles": tercile}, indent=2))


if __name__ == "__main__":
    main()
