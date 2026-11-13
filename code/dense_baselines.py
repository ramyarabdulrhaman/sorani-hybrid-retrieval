"""Additional dense retrieval baselines: LaBSE and KuBERT.

These are compared with the existing E5 and BGE-M3 dense baselines on the
headline task (492 queries over 492 passages) and the Belebele test split
(447 questions over 488 passages). Pooling follows each model's convention:
CLS for LaBSE, mean pooling for KuBERT.

Outputs
-------
results/dense_baselines.json
results/dense_baselines_ranks.csv
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_experiments as R
import bge_m3_modes as M

MODELS = {
    "LaBSE": ("sentence-transformers/LaBSE", "cls"),
    "KuBERT": ("asosoft/KuBERT-Central-Kurdish-BERT-Model", "mean"),
}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAX_P, MAX_Q = 512, 64
BOOT = 5000
SEED = 42


def encode(model_id, pooling, texts, max_len, batch_size=16):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(DEV).eval()
    vecs = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            normalized = [
                R.TOK.normalize_orthography(text)
                for text in texts[start:start + batch_size]
            ]
            enc = tok(normalized, padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(DEV)
            h = model(**enc).last_hidden_state.float()
            if pooling == "cls":
                v = h[:, 0]
            else:
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1)
            vecs.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(vecs)


def main():
    docs, _ = R.load_data()
    hq = [d["headline"] for d in docs]
    hp = [d["body"] for d in docs]
    hgold = np.arange(len(docs))

    rows = [json.loads(line) for line in open("data/raw/belebele_ckb_Arab.jsonl", encoding="utf-8")]
    passages, indices, bgold = [], {}, []
    for row in rows:
        key = (row["link"], row["ds"])
        if key not in indices:
            indices[key] = len(passages)
            passages.append(row["flores_passage"])
        bgold.append(indices[key])
    bgold = np.asarray(bgold)
    bq = [r["question"] for r in rows]
    perm = np.random.default_rng(42).permutation(len(passages))
    val_groups = np.sort(perm[: len(perm) // 2])
    test = ~np.isin(bgold, val_groups)

    result = {"protocol": {
        "models": MODELS, "passage_max_pieces": MAX_P, "query_max_pieces": MAX_Q,
        "device": DEV, "bootstrap_resamples": BOOT, "seed": SEED,
        "headline": "492 queries over 492 passages",
        "belebele": "447 test questions over 488 passages",
        "note": "LaBSE uses CLS pooling; KuBERT uses mean pooling; both L2 normalized.",
    }, "headline": {}, "belebele_test": {}}
    hranks, branks = {}, {}
    rng = np.random.default_rng(SEED)
    for name, (mid, pool) in MODELS.items():
        print(f"encoding {name} ({mid})", flush=True)
        qh = encode(mid, pool, hq, MAX_Q, 16)
        ph = encode(mid, pool, hp, MAX_P, 8)
        hm = qh @ ph.T
        hr = M.rank_of_gold(hm, hgold)
        hranks[name] = hr
        result["headline"][name] = M.summarize(hr, hgold, rng)
        qb = encode(mid, pool, bq, MAX_Q, 16)
        pb = encode(mid, pool, passages, MAX_P, 8)
        bm = qb @ pb.T
        br = M.rank_of_gold(bm, bgold)
        branks[name] = br
        result["belebele_test"][name] = M.summarize(br[test], bgold[test], rng)
        print(f"  headline {result['headline'][name]['MRR@10']:.2f} | "
              f"belebele {result['belebele_test'][name]['MRR@10']:.2f}", flush=True)

    outdir = Path(R.OUT)
    (outdir / "dense_baselines.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (outdir / "dense_baselines_ranks.csv").open("w", encoding="utf-8", newline="") as f:
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
