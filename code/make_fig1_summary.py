"""Figure 1: fused retriever minus char-4gram BM25 across collections and overlap terciles."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
res = json.loads((ROOT / "results/proposed_results.json").read_text(encoding="utf-8"))
bel = json.loads((ROOT / "results/belebele/belebele_30model_results.json").read_text(encoding="utf-8"))
kn = json.loads((ROOT / "results/knqad/knqad_results.json").read_text(encoding="utf-8"))["comparisons"]
pew = json.loads((ROOT / "results/pewan_dense_fusion.json").read_text(encoding="utf-8"))

head = res["paired_comparisons"]["Proposed vs BM25 (char 4-gram)"]
bc = bel["comparisons"]["BM25 char-4gram"]
pc = pew["comparisons_proposed_minus"]["BM25 char-4gram"]["nDCG@10"]
rows = [
    ("Headlines (30 runs, MRR@10)\nmedian overlap 0.80", head["mean_MRR@10_diff"], head["CI95"]),
    ("KNQAD, news-trained (30, MRR@10)\nmedian overlap 0.38", kn["News-trained fused retriever (30)"]["BM25 char-4gram"]["diff"],
     kn["News-trained fused retriever (30)"]["BM25 char-4gram"]["CI95"]),
    ("KNQAD, KNQAD-trained (5, MRR@10)\nmedian overlap 0.38", kn["KNQAD-trained fused retriever (5)"]["BM25 char-4gram"]["diff"],
     kn["KNQAD-trained fused retriever (5)"]["BM25 char-4gram"]["CI95"]),
    ("Belebele (30 models, MRR@10)\nmedian overlap 0.18", bc["diff"], bc["CI95"]),
    ("Pewan (5 models, nDCG@10)\n22 topics", pc["diff"], pc["CI95"]),
]
terc = [(f"Belebele {k} overlap", v["diff_vs_char4"], v["CI95"]) for k, v in bel["overlap_terciles_vs_char4"].items()]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
for ax, data, title in ((axes[0], rows, "(a) By collection"), (axes[1], terc, "(b) Belebele by overlap tercile")):
    for i, (lab, d, (lo, hi)) in enumerate(data):
        y = len(data) - 1 - i
        col = "#1b7f5b" if lo > 0 else "#7a7a7a"
        ax.plot([lo, hi], [y, y], color=col, lw=2)
        ax.plot(d, y, "o", color=col, ms=6)
        ax.text(hi + 0.3, y, f"{d:+.2f}", va="center", fontsize=8)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([x[0] for x in data][::-1], fontsize=7.5)
    ax.set_xlabel("Fused $-$ char-4gram BM25 (points)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
axes[1].set_xlim(-2.5, 13)
fig.tight_layout()
fig.savefig(ROOT / "figures/fig1_overlap_summary.pdf")
fig.savefig(ROOT / "figures/fig1_overlap_summary.png", dpi=300)
print("ok")
