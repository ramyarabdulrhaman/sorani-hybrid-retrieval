"""Verify the Belebele-only per-query ranks and generate Table 12."""
import csv
import json

import numpy as np

import belebele_experiment as B
import belebele_only_experiment as E


def main():
    result = json.loads((B.OUT / "belebele_only_results.json").read_text(encoding="utf-8"))
    rows, passages, gold = B.load_dataset()
    groups = E.partition(gold, len(passages))
    expected = np.flatnonzero(groups == 2).tolist()
    with (B.OUT / "belebele_only_ranks.csv").open(encoding="utf-8", newline="") as f:
        ranks = list(csv.DictReader(f))
    table, rr = [], {}
    for seed in E.TRAIN_SEEDS:
        seed_rows = [r for r in ranks if int(r["training_seed"]) == seed]
        if [int(r["row"]) for r in seed_rows] != expected:
            raise ValueError(f"Missing, duplicate, or misplaced rows for seed {seed}")
        if any(int(r["passage_index"]) != gold[int(r["row"])] for r in seed_rows):
            raise ValueError("Incorrect passage index")
    for system in E.SYSTEMS:
        values = np.asarray([[int(r[system]) for r in ranks if int(r["training_seed"]) == seed]
                             for seed in E.TRAIN_SEEDS])
        if np.any((values < 1) | (values > len(passages))):
            raise ValueError(f"Invalid rank for {system}")
        actual = [100 * (values == 1).mean(), 100 * (values <= 5).mean(),
                  100 * np.where(values <= 10, 1 / values, 0).mean()]
        saved = result["systems"][system]
        for key, value in zip(("Recall@1", "Recall@5", "MRR@10"), actual):
            if not np.isclose(value, saved[key], atol=1e-10):
                raise ValueError(f"{system} {key} mismatch")
        rr[system] = np.where(values <= 10, 1 / values, 0).mean(axis=0)
        display_name = "Fused retriever (Belebele-trained)" if system == "Proposed Belebele-trained fusion" else system
        table.append([display_name, *[f"{x:.2f}" for x in actual]])
    for baseline, saved in result["comparisons"].items():
        a, b = rr[E.SYSTEMS[-1]], rr[baseline]
        if not np.isclose(100 * (a - b).mean(), saved["MRR@10_difference"], atol=1e-10):
            raise ValueError(f"{baseline} difference mismatch")
        if not np.allclose(B.cluster_ci(a, b, gold[groups == 2]), saved["CI95_passage_cluster"], atol=1e-10):
            raise ValueError(f"{baseline} bootstrap mismatch")
    out = B.ROOT / "tables" / "table_12_belebele_only.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["System", "R@1", "R@5", "MRR@10"])
        writer.writerows(table)
    print(f"Verified {len(ranks)} ranks; wrote {out}")


if __name__ == "__main__":
    main()
