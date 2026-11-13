"""Generate the manuscript table and control comparison from saved ranks."""
import csv
import json
from pathlib import Path

import numpy as np

import belebele_experiment as experiment


def main():
    root = Path(experiment.ROOT)
    directory = root / "results" / "belebele"
    result = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    with (directory / "ranks.csv").open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    if len(rows) != result["protocol"]["test_questions"]:
        raise ValueError("Rank file does not match test question count")
    group = np.array([int(r["passage_index"]) for r in rows])

    def rr(name):
        rank = np.array([int(r[name]) for r in rows])
        return np.where(rank <= 10, 1.0 / rank, 0)

    proposed = rr("Proposed cross-domain variant")
    control = rr("Zero-shot fusion")
    comparison = {
        "proposed_minus_zero_shot_fusion_MRR@10": float((proposed - control).mean() * 100),
        "CI95_passage_cluster_bootstrap": experiment.cluster_ci(proposed, control, group),
        "bootstrap_resamples": experiment.BOOTSTRAP_RESAMPLES,
    }
    (directory / "control_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    table = root / "tables" / "table_10_belebele_retrieval.csv"
    with table.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["System", "Recall@1", "Recall@5", "MRR@10"])
        for name, value in result["test_systems"].items():
            measured = float(rr(name).mean() * 100)
            if not np.isclose(measured, value["MRR@10"], atol=1e-10):
                raise ValueError(f"Ranks and result disagree for {name}")
            display_name = "Fused retriever (news-trained)" if name == "Proposed cross-domain variant" else name
            writer.writerow([display_name, f'{value["Recall@1"]:.2f}', f'{value["Recall@5"]:.2f}',
                             f'{value["MRR@10"]:.2f}'])
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
