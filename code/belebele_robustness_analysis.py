"""Verify robustness results from per-query ranks and generate Table 11."""
import csv
import json

import numpy as np

import belebele_experiment as B
import belebele_robustness as S


def main():
    result_path = B.OUT / "robustness_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows, passages, gold = B.load_dataset()
    folds = S.group_folds(gold, len(passages))
    with (B.OUT / "robustness_ranks.csv").open(encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    ranks = {name: np.zeros((len(S.TRAIN_SEEDS), len(rows)), dtype=np.int32)
             for name in (S.BASELINE, S.CONTROL, S.PROPOSED)}
    seen = set()
    for r in records:
        i, seed = int(r["row"]), int(r["training_seed"])
        if (seed, i) in seen or int(r["passage_index"]) != gold[i] or int(r["test_fold"]) != folds[i]:
            raise ValueError(f"Duplicate or inconsistent record for seed {seed}, row {i}")
        seen.add((seed, i))
        j = S.TRAIN_SEEDS.index(seed)
        for name in ranks:
            rank = int(r[name])
            if not 1 <= rank <= len(passages):
                raise ValueError(f"Invalid rank for {name}: {rank}")
            ranks[name][j, i] = rank
    if len(seen) != len(rows) * len(S.TRAIN_SEEDS):
        raise ValueError("Incomplete robustness ranks")
    systems, comparisons = S.summarize(ranks, gold)
    if systems != result["systems"] or comparisons != result["comparisons"]:
        raise ValueError("Saved robustness summary disagrees with per-query ranks")
    with (B.ROOT / "tables" / "table_11_belebele_robustness.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["System", "Recall@1", "Recall@5", "MRR@10"])
        for name, metrics in systems.items():
            display_name = "Fused retriever (news-trained)" if name == "Proposed cross-domain variant" else name
            writer.writerow([display_name, *[f"{metrics[m]:.2f}" for m in ("Recall@1", "Recall@5", "MRR@10")]])
    print("Validated robustness ranks, metrics, comparisons and bootstrap intervals; generated Table 11.")


if __name__ == "__main__":
    main()
