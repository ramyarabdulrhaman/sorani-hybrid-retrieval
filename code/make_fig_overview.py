"""Study-design overview figure: collections -> retrievers -> fusion -> analyses."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED = "#1f2933", "#7b8794"
C = {"data": "#e3f2fd", "lex": "#fff4e0", "dense": "#ede7f6", "fuse": "#e0f2ec", "ana": "#fdecea", "train": "#f2f4f7"}
E = {"data": "#1e6fb8", "lex": "#c27c0e", "dense": "#6a4c9c", "fuse": "#1b7f5b", "ana": "#b23a2e", "train": "#52606d"}

kn = ROOT / "results/knqad/knqad_results.json"
if kn.exists():
    k = json.loads(kn.read_text(encoding="utf-8"))
    kq = f"{k['protocol']['questions']:,} Q, {k['protocol']['paragraphs']:,} paragraphs"
    kov = f"median overlap {k['overlap']['median_all']:.2f}"
    ktrain = f"KNQAD: {k['protocol']['train_pairs']:,} training pairs"
else:
    kq, kov, ktrain = "14,985 Q, 7,271 paragraphs", "median overlap 0.38", "KNQAD training split"

fig, ax = plt.subplots(figsize=(7.4, 4.0))
ax.set_xlim(0, 104); ax.set_ylim(0, 57); ax.axis("off")


def box(x, y, w, h, kind, title, lines, fs=5.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0",
                                fc=C[kind], ec=E[kind], lw=1.0))
    ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=6.0, weight="bold", color=E[kind])
    ax.text(x + w / 2, y + h - 4.3, "\n".join(lines), ha="center", va="top", fontsize=fs, color=INK, linespacing=1.3)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=MUTED))


for x, t in ((13, "1  Collections"), (40, "2  Retrievers"), (66, "3  Fusion and protocol"), (91.5, "4  Analyses")):
    ax.text(x, 55.5, t, ha="center", fontsize=7.6, weight="bold", color=INK)

# collections
cols = [("Headlines (news)", ["492 headline–article pairs", "median overlap 0.80"], 43),
        ("KNQAD", [kq, kov], 32.5),
        ("Belebele (ckb)", ["900 Q, 488 passages", "median overlap 0.18"], 22),
        ("Pewan", ["22 judged topics", "115,340 documents"], 5)]
for t, l, y in cols:
    box(4, y, 22, 8.5, "data", t, l)
ax.annotate("", xy=(1.6, 22), xytext=(1.6, 51.5), arrowprops=dict(arrowstyle="-|>", color=E["data"], lw=1.1))
ax.text(0.2, 37, "lower query–passage overlap", rotation=90, ha="center", va="center", fontsize=5.6, color=E["data"])
ax.text(15, 15.6, "judged ad hoc search (complementary)", ha="center", fontsize=5.4, color=E["data"], style="italic")

# retrievers
box(30, 39, 20, 12.5, "lex", "Lexical: BM25", ["word tokens", "stems (rule-based, KLPT, Jedar)", "character 4-grams"])
box(30, 22, 20, 13.5, "dense", "Dense encoders", ["multilingual-e5-base", "zero-shot or fine-tuned", "bge-m3 (zero-shot)"])
box(30, 5, 20, 13, "train", "E5 fine-tuning data", ["news: 30 cross-fitted runs", ktrain, "Pewan: no training/tuning"])
for _, _, y in cols:
    arrow(26.4, y + 4.25, 29.6, 36.5)
arrow(40, 18.3, 40, 21.6)

# fusion and protocol
box(55, 30, 22, 21.5, "fuse", "Per-query z-score fusion",
    ["$w_1 z(S_{c4}) + w_2 z(S_{E5}) + w_3 z(S_{bge})$", "", "validation-tuned weights", "fixed equal weights (1/3)",
     "reciprocal-rank fusion (baseline)"])
box(55, 5, 22, 20.5, "fuse", "Evaluation protocol",
    ["passage-grouped splits", "2 partitions × 3 folds × 5 seeds", "all checkpoints saved", "paired (cluster) bootstrap",
     "Holm correction"])
arrow(50.4, 45, 54.6, 42); arrow(50.4, 29, 54.6, 36)

# analyses
box(81, 37, 21.5, 14.5, "ana", "Overlap analysis", ["fusion minus char-4gram BM25", "across collections and", "within-task overlap terciles"])
box(81, 20.5, 21.5, 13, "ana", "Transfer", ["news-trained models applied", "unchanged to KNQAD,", "Belebele and Pewan"])
box(81, 5, 21.5, 12, "ana", "Controls", ["equal vs tuned weights", "fine-tuned vs zero-shot", "post-hoc TOST and power"])
arrow(77.4, 44, 80.6, 44); arrow(77.4, 36, 80.6, 27); arrow(77.4, 12, 80.6, 11)

fig.savefig(ROOT / "figures/fig0_study_overview.pdf", bbox_inches="tight")
fig.savefig(ROOT / "figures/fig0_study_overview.png", dpi=300, bbox_inches="tight")
print("ok")
