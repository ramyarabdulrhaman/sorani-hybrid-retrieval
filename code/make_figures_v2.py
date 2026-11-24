"""Generate the figures (v2) from saved result files.

Outputs (PDF + PNG) in figures/:
  fig_overview      study design diagram
  fig_main_gains    fused retriever minus char-4gram BM25 per collection + tercile gains
  fig_overlap       overlap distributions and per-query gain vs overlap
  fig_adaptive      fixed vs query-adaptive vs oracle fusion
  fig_recall        Recall@k curves on the headline collection
Every plotted number is read from results/ except a few confidence intervals copied
from the result files of the 30-model Belebele, Pewan, and cross-fitted analyses.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"

# Reference palette (validated categorical slots 1-3, light mode) and chrome ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": BASE,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load(path):
    return json.load(open(RES / path, encoding="utf-8-sig"))


# ---------------------------------------------------------------- overview
def overview():
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    def box(x, y, w, h, title, lines, edge, fill="#f9f9f7"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fill, ec=edge, lw=1.2))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.2,
                weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.2, "\n".join(lines), ha="center", va="top",
                fontsize=6.9, color=INK2, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1))

    # Collections with overlap bars
    ax.text(11, 45.5, "Sorani collections", ha="center", fontsize=8.5, weight="bold", color=INK)
    cols = [("News headlines", "492 q / 492 p", 0.80), ("KNQAD", "3,061 q / 7,271 p", 0.38),
            ("Belebele", "447 q / 488 p", 0.18), ("Pewan (judged)", "22 topics / 115k docs", None)]
    for i, (name, size, ov) in enumerate(cols):
        y = 34 - i * 10.3
        ax.add_patch(FancyBboxPatch((0.5, y), 21, 8, boxstyle="round,pad=0.3,rounding_size=1",
                                    fc="#f9f9f7", ec=BASE, lw=0.9))
        ax.text(1.8, y + 6.2, name, fontsize=7.5, weight="bold", color=INK, va="center")
        ax.text(1.8, y + 3.9, size, fontsize=6.4, color=INK2, va="center")
        if ov is not None:
            ax.add_patch(plt.Rectangle((1.8, y + 0.9), 15 * ov, 1.4, color=BLUE, lw=0))
            ax.add_patch(plt.Rectangle((1.8, y + 0.9), 15, 1.4, fill=False, ec=BASE, lw=0.5))
            ax.text(17.6, y + 1.6, f"{ov:.2f}", fontsize=6.2, color=INK2, va="center")
        else:
            ax.text(1.8, y + 1.6, "human relevance judgments", fontsize=6.1,
                    color=MUTED, va="center", style="italic")
    ax.text(11, 0.4, "bar = median query-passage overlap", ha="center", fontsize=6, color=MUTED)

    box(27, 26, 20, 17, "Lexical", ["BM25: word, stem,", "char 4-gram", "KLPT, Jedar (Pewan)",
                                       "$k_1$=1.5, $b$=0.75"], BLUE)
    box(27, 4, 20, 19, "Dense", ["multilingual-e5-base", "(zero-shot / fine-tuned", "news 328 or KNQAD",
                                     "4,362 pairs, InfoNCE)", "bge-m3 (zero-shot)"], ORANGE)
    box(52, 13, 20, 21, "Fusion", ["per-query z-scores", "equal weights (1/3)", "or validation-tuned",
                                       "or query-adaptive", "(vs. RRF, $k$=60)"], AQUA)
    box(77, 25, 22.5, 18, "Protocol", ["3-fold cross-fitting", "2 partitions x 5 seeds", "= 30 saved models",
                                          "cluster bootstrap, Holm"], BASE)
    box(77, 2, 22.5, 20, "Analyses", ["RQ1-2 headline + tuning", "RQ3 transfer (no retrain)",
                                          "RQ4 stemming", "RQ5 overlap dependence", "RQ6 adaptive + oracle"], BASE)
    for y in (38.0, 27.7, 17.4, 7.1):
        arrow(22.3, y, 26.4, 34 if y > 25 else 13)
    arrow(47.6, 34, 51.6, 27); arrow(47.6, 13, 51.6, 20)
    arrow(72.6, 26, 76.4, 33); arrow(72.6, 21, 76.4, 13)
    save(fig, "fig_overview")


# ---------------------------------------------------------------- main gains
def main_gains():
    kn = load("knqad/knqad_results.json")
    c = kn["comparisons"]
    news = c["News-trained fused retriever (30)"]["BM25 char-4gram"]
    knq = c["KNQAD-trained fused retriever (5)"]["BM25 char-4gram"]
    rows = [  # label, overlap, diff, lo, hi, metric
        ("Headlines (cross-fitted)", "0.80", 0.02, -0.34, 0.42, "MRR@10"),
        ("KNQAD, news-trained", "0.38", news["diff"], *news["CI95"], "MRR@10"),
        ("KNQAD, KNQAD-trained", "0.38", knq["diff"], *knq["CI95"], "MRR@10"),
        ("Belebele, news-trained", "0.18", 3.64, 1.93, 5.45, "MRR@10"),
        ("Pewan, news-trained", "n/a", 3.84, 0.48, 7.38, "nDCG@10"),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.35, 1]})
    y = np.arange(len(rows))[::-1]
    for yi, (lab, ov, d, lo, hi, met) in zip(y, rows):
        col = AQUA if lo > 0 else MUTED
        a.plot([lo, hi], [yi, yi], color=col, lw=2, solid_capstyle="round")
        a.plot(d, yi, "o", ms=6, color=col, mec="white", mew=1.2)
        a.text(hi + 0.35, yi, f"{d:+.2f}", va="center", fontsize=7, color=INK)
    a.axvline(0, color=INK2, lw=0.8)
    a.set_yticks(y, [f"{r[0]}\noverlap {r[1]}, {r[5]}" for r in rows], fontsize=6.8)
    a.set_xlabel("Fused retriever minus char-4gram BM25 (points, 95% CI)")
    a.set_xlim(-1.5, 11.5); a.grid(axis="y", visible=False)
    a.set_title("(a) Gain over the strongest lexical baseline", fontsize=8.5, loc="left", color=INK)

    terc = kn["terciles_vs_char4"]["News-trained fused retriever (30)"]
    bel = [(6.92, 2.92, 11.04), (4.64, 1.74, 7.76), (0.47, -1.15, 2.25)]
    knr = [(terc[k]["diff_vs_char4"], *terc[k]["CI95"]) for k in ("low", "mid", "high")]
    x = np.arange(3)
    for off, data, col, lab in ((-0.12, bel, BLUE, "Belebele"), (0.12, knr, ORANGE, "KNQAD")):
        for xi, (d, lo, hi) in zip(x + off, data):
            b.plot([xi, xi], [lo, hi], color=col, lw=2, solid_capstyle="round")
        b.plot(x + off, [d[0] for d in data], "-o", color=col, lw=1.4, ms=6, mec="white", mew=1.2,
               label=lab)
    b.axhline(0, color=INK2, lw=0.8)
    b.set_xticks(x, ["low", "mid", "high"]); b.set_xlabel("Within-collection overlap tercile")
    b.set_ylabel("Gain (MRR@10 points)"); b.legend(loc="upper right", fontsize=7)
    b.set_title("(b) Gain by overlap tercile (news models)", fontsize=8.5, loc="left", color=INK)
    fig.tight_layout(w_pad=2)
    save(fig, "fig_main_gains")


# ---------------------------------------------------------------- overlap
def overlap():
    t = pd.read_csv(RES / "task_dependence_per_query.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    bins = np.linspace(0, 1, 21)
    for task, col, lab in (("headline", BLUE, "Headlines"), ("belebele", ORANGE, "Belebele")):
        g = t[t.task == task]
        a.hist(g.overlap_words, bins=bins, color=col, alpha=0.75, label=f"{lab} (median {g.overlap_words.median():.2f})",
               edgecolor="white", linewidth=0.8, weights=np.ones(len(g)) / len(g) * 100)
    a.set_xlabel("Query content-word overlap with relevant passage"); a.set_ylabel("% of queries")
    a.legend(fontsize=7); a.set_title("(a) Overlap distributions", fontsize=8.5, loc="left", color=INK)

    g = t[t.task == "belebele"].copy()
    g["gain"] = (g.Fusion_rr - g.BM25_rr) * 100
    edges = [0, 0.1, 0.2, 0.3, 0.4, 1.01]
    g["bin"] = pd.cut(g.overlap_words, edges, right=False)
    rng = np.random.default_rng(42)
    centers, means, los, his, ns = [], [], [], [], []
    for iv, s in g.groupby("bin", observed=True):
        v = s.gain.values
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
        centers.append((iv.left + min(iv.right, 0.6)) / 2); means.append(v.mean())
        los.append(np.percentile(bs, 2.5)); his.append(np.percentile(bs, 97.5)); ns.append(len(v))
    for c_, m, lo, hi, n in zip(centers, means, los, his, ns):
        b.plot([c_, c_], [lo, hi], color=ORANGE, lw=2, solid_capstyle="round")
        b.text(c_ + 0.012, lo - 0.3, f"n={n}", ha="left", va="top", fontsize=6.2, color=MUTED)
    b.plot(centers, means, "-o", color=ORANGE, ms=6, mec="white", mew=1.2)
    slope, icpt = np.polyfit(g.overlap_words, g.gain, 1)
    xs = np.linspace(0, 0.6, 50)
    b.plot(xs, icpt + slope * xs, "--", color=INK2, lw=1,
           label=f"OLS slope {slope / 10:+.2f} pts per +0.1")
    b.axhline(0, color=INK2, lw=0.8)
    b.set_xlabel("Belebele word overlap (binned)"); b.set_ylabel("Fusion minus BM25 (MRR@10 pts)")
    b.legend(fontsize=7, loc="upper right"); b.set_ylim(-8, 11)
    b.set_title("(b) Belebele gain falls as overlap rises", fontsize=8.5, loc="left", color=INK)
    fig.tight_layout(w_pad=2)
    save(fig, "fig_overlap")


# ---------------------------------------------------------------- adaptive
def adaptive():
    d = load("adaptive_fusion_exploratory.json")["tasks"]
    keys = [("headline", "Headlines"), ("belebele_in_domain_exploratory", "Belebele\nin-domain policy"),
            ("belebele_news_transfer", "Belebele\nnews-transferred")]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = np.arange(len(keys)); w = 0.24
    series = [("fixed_equal_weight_MRR10", "Fixed equal weights", MUTED),
              ("adaptive_MRR10", "Query-adaptive", AQUA),
              ("oracle_MRR10_non_deployable", "Oracle (upper bound)", BLUE)]
    for i, (k, lab, col) in enumerate(series):
        vals = [d[t][k] for t, _ in keys]
        ax.bar(x + (i - 1) * w, vals, w - 0.03, color=col, label=lab)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + 0.25, f"{v:.1f}", ha="center", fontsize=5.8, color=INK2)
    ax.set_ylim(78, 101); ax.set_xticks(x, [k[1] for k in keys], fontsize=7)
    ax.set_ylabel("MRR@10 (%)"); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=6.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13), handlelength=1)
    save(fig, "fig_adaptive")


# ---------------------------------------------------------------- recall
def recall():
    r = pd.read_csv(RES / "per_query_gold_ranks.csv").drop(columns="query_index")
    ks = np.arange(1, 21)
    sel = [("BM25 (char 4-gram)", "Char-4gram BM25", BLUE, "-"),
           ("BM25 (word)", "Word BM25", BLUE, ":"),
           ("bge-m3 (dense)", "bge-m3", ORANGE, "-"),
           ("multilingual-e5-base", "multilingual-e5-base", ORANGE, ":"),
           ("Hybrid RRF (BM25 stems + bge-m3)", "RRF stems + bge-m3", AQUA, "-")]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for col, lab, c, ls in sel:
        ax.plot(ks, [(r[col] <= k).mean() * 100 for k in ks], ls, color=c, lw=1.8, label=lab)
    ax.set_xlabel("Cut-off $k$"); ax.set_ylabel("Recall@$k$ (%)")
    ax.set_xticks([1, 5, 10, 15, 20]); ax.set_ylim(55, 101)
    ax.legend(fontsize=6.5, loc="lower right")
    save(fig, "fig_recall")


if __name__ == "__main__":
    overview(); main_gains(); overlap(); adaptive(); recall()
    print("figures written to", FIG)
