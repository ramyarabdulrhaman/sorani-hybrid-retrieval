"""Task-dependence analysis for the Kurdish Sorani retrieval paper.

Quantifies how much of the query is lexically present in the relevant passage
for (a) headline-to-article retrieval and (b) Belebele question-to-passage
retrieval, then stratifies the fusion-minus-BM25 difference by overlap tercile.

The analysis is descriptive and post hoc. It uses saved per-query ranks and the
public Belebele file; no model is retrained.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'code'))
import real_experiments as R

RESULTS = os.path.join(BASE, 'results')
TABLES = os.path.join(BASE, 'tables')
FIGURES = os.path.join(BASE, 'figures')
SEED = 42
N_BOOT = 5000
MIN_CONTENT_LEN = 3


def content_words(text):
    """Normalized tokens of length >= 3; a transparent content-word heuristic."""
    return set(t for t in R.words(text) if len(t) >= MIN_CONTENT_LEN)


def word_overlap(query, passage):
    q = content_words(query)
    p = content_words(passage)
    return len(q & p) / len(q) if q else float('nan')


def char4_overlap(query, passage):
    q = set(R.char_ngrams(query))
    p = set(R.char_ngrams(passage))
    return len(q & p) / len(q) if q else float('nan')


def rr10(ranks):
    return np.where(ranks <= 10, 1.0 / ranks, 0.0)


def mrr10(ranks):
    return float(rr10(ranks).mean() * 100)


def bootstrap_diff(a, b, groups=None, nboot=N_BOOT, seed=SEED):
    """Paired bootstrap of (a - b) in MRR@10 points.

    If groups is given, resample groups (passages) rather than queries.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = (a - b) * 100
    if groups is None:
        n = len(d)
        boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    else:
        groups = np.asarray(groups)
        keys = np.unique(groups)
        parts = {k: np.where(groups == k)[0] for k in keys}
        boots = np.empty(nboot)
        for i in range(nboot):
            sampled = rng.choice(keys, size=len(keys), replace=True)
            idx = np.concatenate([parts[k] for k in sampled])
            boots[i] = d[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {
        'difference_points': float(d.mean()),
        'CI95': [float(lo), float(hi)],
        'p_two_sided': float(max(p, 1.0 / nboot)),
        'wins': int((d > 0).sum()),
        'losses': int((d < 0).sum()),
        'ties': int((d == 0).sum()),
    }


def headline_frame():
    docs, _ = R.load_data()
    full = pd.read_csv(os.path.join(RESULTS, 'full_corpus_equal_weight_ranks.csv'))
    rows = []
    for i, d in enumerate(docs):
        rows.append({
            'query_index': i,
            'overlap_words': word_overlap(d['headline'], d['body']),
            'overlap_char4': char4_overlap(d['headline'], d['body']),
        })
    ov = pd.DataFrame(rows)
    df = ov.merge(full, on='query_index')
    df['BM25_rr'] = df['BM25_char4_MRR10']
    df['Fusion_rr'] = df['zero_shot_equal_weight_MRR10']
    return df


def belebele_frame():
    items = [json.loads(l) for l in open(
        os.path.join(BASE, 'data', 'raw', 'belebele_ckb_Arab.jsonl'), encoding='utf-8')]
    ranks = pd.read_csv(os.path.join(RESULTS, 'belebele', 'ranks.csv'))
    rows = []
    for i, o in enumerate(items):
        rows.append({
            'row': i,
            'overlap_words': word_overlap(o['question'], o['flores_passage']),
            'overlap_char4': char4_overlap(o['question'], o['flores_passage']),
        })
    df = ranks.merge(pd.DataFrame(rows), on='row')
    df = df[df['split'] == 'test'].copy()
    df['BM25_rr'] = rr10(df['BM25 char-4gram'].values)
    df['Fusion_rr'] = rr10(df['Zero-shot fusion'].values)
    df['Fine-tuned_rr'] = rr10(df['Proposed cross-domain variant'].values)
    return df


def tercile_summary(df, task, fusion_col='Fusion_rr', group_col=None):
    df = df.copy()
    df['tercile'] = pd.qcut(df['overlap_words'], 3, labels=['Low', 'Mid', 'High'])
    out = []
    for label, sub in [('All', df)] + [(t, df[df.tercile == t]) for t in ['Low', 'Mid', 'High']]:
        groups = sub[group_col].values if group_col else None
        diff = bootstrap_diff(sub[fusion_col].values, sub['BM25_rr'].values, groups=groups)
        out.append({
            'task': task,
            'tercile': label,
            'queries': int(len(sub)),
            'passages': int(sub[group_col].nunique()) if group_col else int(len(sub)),
            'mean_word_overlap': float(sub['overlap_words'].mean()),
            'median_word_overlap': float(sub['overlap_words'].median()),
            'mean_char4_overlap': float(sub['overlap_char4'].mean()),
            'BM25_MRR10': float(sub['BM25_rr'].mean() * 100),
            'Fusion_MRR10': float(sub[fusion_col].mean() * 100),
            'difference': diff,
        })
    return out


def main():
    headline = headline_frame()
    belebele = belebele_frame()

    headline_summary = tercile_summary(headline, 'Headline-to-article')
    belebele_summary = tercile_summary(belebele, 'Belebele question-to-passage', group_col='passage_index')
    belebele_finetuned = tercile_summary(belebele, 'Belebele (fine-tuned fusion)', fusion_col='Fine-tuned_rr', group_col='passage_index')

    # Random-pairing baseline for the headline task.
    docs, _ = R.load_data()
    rng = np.random.default_rng(SEED)
    random_overlap = []
    for _ in range(1000):
        perm = rng.permutation(len(docs))
        random_overlap.append(np.mean([
            word_overlap(docs[i]['headline'], docs[perm[i]]['body']) for i in range(len(docs))
        ]))

    result = {
        'protocol': {
            'content_word_definition': 'normalized tokens of length >= 3',
            'overlap_definition': 'fraction of query content-word types present in the passage',
            'char4_definition': 'fraction of query character-4-gram types present in the passage',
            'terciles': 'within-task terciles of word overlap; Belebele restricted to the 447-question test split',
            'headline_fusion': 'fixed one-third zero-shot fusion on the full 492-passage collection',
            'belebele_fusion': 'validation-selected zero-shot fusion; fine-tuned variant reported separately',
            'bootstrap_resamples': N_BOOT,
            'seed': SEED,
            'status': 'descriptive post-hoc analysis of saved ranks',
        },
        'headline': {
            'queries': int(len(headline)),
            'mean_word_overlap': float(headline['overlap_words'].mean()),
            'median_word_overlap': float(headline['overlap_words'].median()),
            'mean_char4_overlap': float(headline['overlap_char4'].mean()),
            'median_char4_overlap': float(headline['overlap_char4'].median()),
            'fraction_word_overlap_ge_0.5': float((headline['overlap_words'] >= 0.5).mean()),
            'fraction_word_overlap_ge_0.8': float((headline['overlap_words'] >= 0.8).mean()),
            'fraction_char4_overlap_ge_0.8': float((headline['overlap_char4'] >= 0.8).mean()),
            'random_pairing_mean_word_overlap': float(np.mean(random_overlap)),
            'random_pairing_CI95': [float(x) for x in np.percentile(random_overlap, [2.5, 97.5])],
            'terciles': headline_summary,
        },
        'belebele': {
            'queries': int(len(belebele)),
            'passages': int(belebele['passage_index'].nunique()),
            'mean_word_overlap': float(belebele['overlap_words'].mean()),
            'median_word_overlap': float(belebele['overlap_words'].median()),
            'mean_char4_overlap': float(belebele['overlap_char4'].mean()),
            'median_char4_overlap': float(belebele['overlap_char4'].median()),
            'fraction_word_overlap_ge_0.5': float((belebele['overlap_words'] >= 0.5).mean()),
            'fraction_word_overlap_ge_0.8': float((belebele['overlap_words'] >= 0.8).mean()),
            'fraction_char4_overlap_ge_0.8': float((belebele['overlap_char4'] >= 0.8).mean()),
            'terciles': belebele_summary,
            'fine_tuned_terciles': belebele_finetuned,
        },
    }

    with open(os.path.join(RESULTS, 'task_dependence.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=1, ensure_ascii=False)

    per_query = pd.concat([
        headline.assign(task='headline')[['task', 'query_index', 'overlap_words', 'overlap_char4', 'BM25_rr', 'Fusion_rr']],
        belebele.assign(task='belebele')[['task', 'row', 'passage_index', 'overlap_words', 'overlap_char4', 'BM25_rr', 'Fusion_rr', 'Fine-tuned_rr']],
    ], ignore_index=True)
    per_query.to_csv(os.path.join(RESULTS, 'task_dependence_per_query.csv'), index=False)

    # Compact table for the manuscript.
    rows = []
    for r in [headline_summary[0], belebele_summary[0]]:
        d = r['difference']
        rows.append({
            'Task': r['task'],
            'Queries': r['queries'],
            'Mean word overlap': f"{r['mean_word_overlap']:.3f}",
            'Median word overlap': f"{r['median_word_overlap']:.3f}",
            'Mean char-4gram overlap': f"{r['mean_char4_overlap']:.3f}",
            'BM25 char-4gram MRR@10': f"{r['BM25_MRR10']:.2f}",
            'Zero-shot fusion MRR@10': f"{r['Fusion_MRR10']:.2f}",
            'Difference (95% CI)': f"{d['difference_points']:+.2f} [{d['CI95'][0]:+.2f}, {d['CI95'][1]:+.2f}]",
        })
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(TABLES, 'table_task_dependence.csv'), index=False)
    with open(os.path.join(TABLES, 'table_task_dependence.md'), 'w', encoding='utf-8') as f:
        f.write(table.to_markdown(index=False) + '\n')

    make_figure(headline_summary, belebele_summary, belebele_finetuned, headline, belebele)
    print(json.dumps(result, indent=1, ensure_ascii=False))
    print(table.to_string(index=False))


def make_figure(headline_summary, belebele_summary, belebele_finetuned, headline, belebele):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={'width_ratios': [1.0, 1.45]})

    ax = axes[0]
    data = [headline['overlap_words'].dropna().values, belebele['overlap_words'].dropna().values]
    bp = ax.boxplot(data, tick_labels=['Headline', 'Belebele'], widths=0.55, patch_artist=True,
                    medianprops={'color': 'black', 'linewidth': 1.2},
                    flierprops={'markersize': 2, 'markerfacecolor': 'gray', 'markeredgecolor': 'gray'})
    for patch, color in zip(bp['boxes'], ['#4C72B0', '#DD8452']):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel('Query content words in passage')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title('(a) Lexical overlap', loc='left', fontsize=9)
    ax.grid(axis='y', alpha=0.25)

    ax = axes[1]
    labels = ['Low', 'Mid', 'High']
    x = np.arange(len(labels))
    series = [
        ('Headline', headline_summary, '#4C72B0', 'o', '-'),
        ('Belebele, zero-shot', belebele_summary, '#DD8452', 's', '-'),
        ('Belebele, fine-tuned', belebele_finetuned, '#55A868', '^', '--'),
    ]
    for name, summary, color, marker, ls in series:
        vals = [summary[i + 1]['difference']['difference_points'] for i in range(3)]
        cis = [summary[i + 1]['difference']['CI95'] for i in range(3)]
        lo = [v - c[0] for v, c in zip(vals, cis)]
        hi = [c[1] - v for v, c in zip(vals, cis)]
        ax.errorbar(x, vals, yerr=[lo, hi], label=name, color=color, marker=marker,
                    linestyle=ls, capsize=2.5, linewidth=1.3, markersize=4)
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Word-overlap tercile')
    ax.set_ylabel('Fusion $-$ BM25 char-4gram (MRR@10 points)')
    ax.set_title('(b) Fusion gain by overlap', loc='left', fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(frameon=False, fontsize=7.5, loc='upper right')

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIGURES, f'fig12_task_dependence.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
