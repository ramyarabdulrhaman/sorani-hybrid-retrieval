"""Passage-grouped retrieval adaptation of Belebele Sorani.

The original dataset is a reading-comprehension test, not a retrieval set.
This script creates its own validation/test partition by source passage.
All queries rank the same full passage collection. Fusion weights are chosen
only on the validation questions. The trained E5 variant is trained on the
separate headline/news corpus, not on Belebele.
"""
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_experiments as R
import proposed_experiment as P

ROOT = Path(R.BASE)
DATA = ROOT / "data" / "raw" / "belebele_ckb_Arab.jsonl"
OUT = ROOT / "results" / "belebele"
DATA_SHA = "b7aa4cb5c9ee985c1293615d91a9c25419e735ac83b98068a31830413a8cb053"
DATA_REV = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
MODELS = {
    "e5": ("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a", "mean"),
    "bge": ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181", "cls"),
}
SEED = 42
BOOTSTRAP_RESAMPLES = 5000


def load_dataset():
    raw = DATA.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA:
        raise ValueError(f"Unexpected dataset hash: {DATA}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    passages, indices, texts = [], {}, set()
    gold = []
    for row in rows:
        key = (row["link"], row["ds"])
        if row["dialect"] != "ckb_Arab":
            raise ValueError("Unexpected dialect")
        if key not in indices:
            if row["flores_passage"] in texts:
                raise ValueError("Identical passages have different identifiers")
            indices[key] = len(passages)
            passages.append(row["flores_passage"])
            texts.add(row["flores_passage"])
        idx = indices[key]
        if passages[idx] != row["flores_passage"]:
            raise ValueError("Inconsistent passage text for an identifier")
        gold.append(idx)
    if len(rows) != 900 or len(passages) != 488:
        raise ValueError("Unexpected dataset dimensions")
    return rows, passages, np.asarray(gold)


def ranks(scores, gold):
    correct = scores[np.arange(len(gold)), gold][:, None]
    return 1 + (scores > correct).sum(axis=1) + (scores == correct).sum(axis=1) - 1


def metrics(r):
    return R.metrics_from_ranks(r)


def weighted(matrices, weights):
    return sum(weight * P.zscore(matrix) for weight, matrix in zip(weights, matrices))


def select_weights(matrices, gold, required_index=None):
    best_score, best_weights = -1.0, None
    for weights in P.GRID:
        if required_index is not None and weights[required_index] < 0.1:
            continue
        rr = ranks(weighted(matrices, weights), gold)
        score = metrics(rr)["MRR@10"]
        if score > best_score + 1e-9:
            best_score, best_weights = score, weights
    return best_weights, best_score


def dense(model_name, queries, passages):
    model_id, revision, pool = MODELS[model_name]
    print(f"Encoding {model_id} @ {revision}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(model_id, revision=revision).to(P.DEV).eval()

    def encode(texts, max_length, batch_size):
        vectors = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                normalized = [
                    R.TOK.normalize_orthography(text)
                    for text in texts[start:start + batch_size]
                ]
                batch = tokenizer(normalized, padding=True, truncation=True,
                                  max_length=max_length, return_tensors="pt").to(P.DEV)
                hidden = model(**batch).last_hidden_state
                v = hidden[:, 0] if pool == "cls" else P.mean_pool(hidden, batch["attention_mask"])
                vectors.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
        return np.concatenate(vectors)

    qp, pp = ("query: ", "passage: ") if model_name == "e5" else ("", "")
    q = encode([qp + x for x in queries], 64, 16)
    p = encode([pp + x for x in passages], 512, 4)
    del model, tokenizer
    if P.DEV == "cuda":
        torch.cuda.empty_cache()
    return q @ p.T


def train_news_and_encode(queries, passages, seed=1):
    news, stats = R.load_data()
    news_q = [d["headline"] for d in news]
    news_p = [d["body"] for d in news]
    index = R.BM25([R.words(p) for p in news_p])
    hard = []
    for i, q in enumerate(news_q):
        scores = index.scores(R.words(q))
        scores[i] = -1e9
        hard.append(int(np.argmax(scores)))
    # This configuration was most frequently selected in the earlier news study.
    # The full-news model is a new cross-domain variant, not a saved fold model.
    P.MODEL_ID = snapshot_download(
        repo_id=MODELS["e5"][0], revision=MODELS["e5"][1],
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                        "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"],
    )

    def evaluate(model, tokenizer):
        q = P.encode(model, tokenizer, [P.QP + x for x in queries], P.MAX_Q, bs=16)
        p = P.encode(model, tokenizer, [P.PP + x for x in passages], P.MAX_P, bs=4)
        return q @ p.T

    print(f"Training E5 on the separate news benchmark: seed={seed}, lr=1e-5, epochs=2", flush=True)
    output = P.train_checkpoints(news_q, news_p, hard, 1e-5, seed, [2], evaluate,
                                 lambda message: print(message, flush=True))
    return output[2], stats


def cluster_ci(a, b, groups):
    unique = np.unique(groups)
    rng = np.random.default_rng(SEED)
    samples = []
    per_group = [np.flatnonzero(groups == group) for group in unique]
    for _ in range(BOOTSTRAP_RESAMPLES):
        ids = np.concatenate([per_group[i] for i in rng.integers(0, len(unique), len(unique))])
        samples.append(float(np.mean(a[ids] - b[ids]) * 100))
    return [float(x) for x in np.percentile(samples, [2.5, 97.5])]


def main():
    if P.DEV != "cuda":
        print("CUDA unavailable; training will be slow.", flush=True)
    rows, passages, gold = load_dataset()
    queries = [r["question"] for r in rows]
    perm = np.random.default_rng(SEED).permutation(len(passages))
    val_groups = np.sort(perm[:len(perm) // 2])
    val = np.isin(gold, val_groups)
    test = ~val
    print(f"Validation: {val.sum()} questions; test: {test.sum()} questions; corpus: {len(passages)} passages", flush=True)
    systems = {}
    for label, tokenizer in [("BM25 word", R.words), ("BM25 stems", R.morph),
                             ("BM25 char-4gram", R.char_ngrams)]:
        print("Scoring", label, flush=True)
        index = R.BM25([tokenizer(p) for p in passages])
        systems[label] = np.stack([index.scores(tokenizer(q)) for q in queries])
    systems["E5 zero-shot"] = dense("e5", queries, passages)
    systems["BGE-M3 zero-shot"] = dense("bge", queries, passages)
    for name, components in [
        ("RRF stems + E5", ["BM25 stems", "E5 zero-shot"]),
        ("RRF stems + BGE-M3", ["BM25 stems", "BGE-M3 zero-shot"]),
        ("RRF stems + E5 + BGE-M3", ["BM25 stems", "E5 zero-shot", "BGE-M3 zero-shot"]),
        ("RRF char-4gram + BGE-M3", ["BM25 char-4gram", "BGE-M3 zero-shot"]),
    ]:
        systems[name] = R.rrf([systems[c] for c in components])
    systems["E5 news-fine-tuned"], news_stats = train_news_and_encode(queries, passages)
    components = ["BM25 char-4gram", "E5 zero-shot", "BGE-M3 zero-shot"]
    control_weights, control_val = select_weights([systems[x][val] for x in components], gold[val])
    systems["Zero-shot fusion"] = weighted([systems[x] for x in components], control_weights)
    components = ["BM25 char-4gram", "E5 news-fine-tuned", "BGE-M3 zero-shot"]
    proposed_weights, proposed_val = select_weights([systems[x][val] for x in components], gold[val], 1)
    systems["Proposed cross-domain variant"] = weighted([systems[x] for x in components], proposed_weights)

    all_ranks = {name: ranks(matrix, gold) for name, matrix in systems.items()}
    base = all_ranks["BM25 char-4gram"][test]
    base_rr = np.where(base <= 10, 1 / base, 0)
    outputs = {}
    for name, ranked in all_ranks.items():
        r = ranked[test]
        rr = np.where(r <= 10, 1 / r, 0)
        outputs[name] = metrics(r)
        outputs[name]["MRR@10_difference_vs_char4"] = float((rr - base_rr).mean() * 100)
        outputs[name]["difference_CI95_passage_cluster_bootstrap"] = cluster_ci(rr, base_rr, gold[test])
    result = {
        "protocol": {
            "dataset": "facebook/belebele ckb_Arab", "dataset_revision": DATA_REV,
            "dataset_sha256": DATA_SHA, "dataset_license": "CC BY-SA 4.0",
            "task": "question to linked passage retrieval; unjudged passages treated as non-relevant",
            "limitation": "original multiple-choice test adapted for retrieval; linked passage is not an exhaustive relevance judgment",
            "seed": SEED, "passage_grouped_validation_fraction": 0.5,
            "validation_questions": int(val.sum()), "test_questions": int(test.sum()),
            "validation_passages": len(val_groups), "test_passages": len(passages) - len(val_groups),
            "candidate_passages_for_all_queries": len(passages),
            "tie_policy": "pessimistic gold rank; RRF uses existing implementation",
            "bootstrap": "5000 paired resamples of test passage groups",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "models": MODELS, "device": P.DEV, "torch": torch.__version__,
            "transformers": transformers.__version__, "python": platform.python_version(),
            "news_training_source": news_stats, "training_seed": 1, "training_lr": 1e-5,
            "training_epochs": 2, "training_mode": "raw",
            "training_note": "New full-news model with configuration most frequently selected in prior study; not the original cross-fitted checkpoints.",
            "validation_note": "Validation partition is carved out of the public Belebele test set; it is not an official development set.",
        },
        "validation_selection": {
            "zero_shot_weights_char4_e5_bge": control_weights, "zero_shot_MRR@10": control_val,
            "proposed_weights_char4_finetuned_e5_bge": proposed_weights, "proposed_MRR@10": proposed_val,
            "grid": "nonnegative weights on 0.1 grid, sum 1; proposed requires fine-tuned E5 weight >= 0.1",
        },
        "test_systems": outputs,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with (OUT / "ranks.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "passage_index", "split", *all_ranks])
        for i in range(len(rows)):
            writer.writerow([i, int(gold[i]), "validation" if val[i] else "test",
                             *[int(r[i]) for r in all_ranks.values()]])
    print(json.dumps({"validation_selection": result["validation_selection"], "test_systems": outputs}, indent=2), flush=True)


if __name__ == "__main__":
    main()
