"""Post-hoc equivalence (TOST) and power analysis for headline cross-fitted comparisons.

The tests use saved per-query reciprocal ranks. The equivalence margins are
chosen for interpretation after the main analysis and are not pre-registered.
Because most queries have identical reciprocal ranks, the number of discordant
queries is reported alongside every test.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, 'results')
TABLES = os.path.join(BASE, 'tables')
FIGURES = os.path.join(BASE, 'figures')
SEED = 42
N_BOOT = 20000


def load(name):
    return pd.read_csv(os.path.join(RESULTS, name)).set_index('query_index').sort_index()


def summarize(diff, margin):
    diff = np.asarray(diff, float)
    n = len(diff)
    mean = float(diff.mean() * 100)
    sd = float(diff.std(ddof=1) * 100)
    se = sd / math.sqrt(n)
    rng = np.random.default_rng(SEED)
    boots = np.array([diff[rng.integers(0, n, n)].mean() * 100 for _ in range(N_BOOT)])
    lo95, hi95 = np.percentile(boots, [2.5, 97.5])
    lo90, hi90 = np.percentile(boots, [5, 95])
    p_low = ((boots <= -margin).sum() + 1) / (N_BOOT + 1)
    p_high = ((boots >= margin).sum() + 1) / (N_BOOT + 1)
    tost_p = max(p_low, p_high)
    mde80 = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * se
    return {
        'n': n,
        'mean_difference_points': mean,
        'sd_points': sd,
        'se_points': se,
        'CI95': [float(lo95), float(hi95)],
        'CI90': [float(lo90), float(hi90)],
        'equivalence_margin_points': margin,
        'tost_p': float(tost_p),
        'tost_p_lower': float(p_low),
        'tost_p_upper': float(p_high),
        'mde80_points': float(mde80),
        'wins': int((diff > 0).sum()),
        'losses': int((diff < 0).sum()),
        'ties': int((diff == 0).sum()),
        'discordant': int((diff != 0).sum()),
    }


def main():
    zs = load('equal_weight_zero_shot_crossfit_ranks.csv')
    ft = load('equal_weight_fusion_crossfit_ranks.csv')
    prop = load('proposed_per_query_rr.csv')

    comparisons = {
        'Zero-shot equal-weight fusion - char-4gram BM25': {
            'diff': zs['zero_shot_equal_weight_MRR10'].values - zs['BM25_char4_MRR10'].values,
            'margin': 1.0,
        },
        'Fine-tuned equal-weight fusion - char-4gram BM25': {
            'diff': ft['equal_weight_MRR10'].values - ft['BM25_char4_MRR10'].values,
            'margin': 1.0,
        },
        'Fine-tuned equal-weight fusion - zero-shot equal-weight fusion': {
            'diff': ft['equal_weight_MRR10'].values - zs['zero_shot_equal_weight_MRR10'].values,
            'margin': 0.5,
        },
        'Validation-tuned fusion - char-4gram BM25': {
            'diff': prop['Proposed: KurdRAG Stage 1 (fine-tuned dense + lexical fusion)'].values
                    - prop['BM25 (char 4-gram)'].values,
            'margin': 1.0,
        },
        'Validation-tuned fusion - zero-shot tuned fusion': {
            'diff': prop['Proposed: KurdRAG Stage 1 (fine-tuned dense + lexical fusion)'].values
                    - prop['Fusion of zero-shot components (tuned weights)'].values,
            'margin': 1.0,
        },
    }

    out = {}
    rows = []
    for name, spec in comparisons.items():
        s = summarize(spec['diff'], spec['margin'])
        out[name] = s
        rows.append({
            'Comparison': name,
            'Mean difference': f"{s['mean_difference_points']:+.2f}",
            '95% CI': f"[{s['CI95'][0]:+.2f}, {s['CI95'][1]:+.2f}]",
            'Discordant (W/L)': f"{s['discordant']} ({s['wins']}/{s['losses']})",
            '90% CI': f"[{s['CI90'][0]:+.2f}, {s['CI90'][1]:+.2f}]",
            'Margin': f"{s['equivalence_margin_points']:.1f}",
            'TOST p': f"{s['tost_p']:.3f}",
            'MDE (80% power)': f"{s['mde80_points']:.2f}",
        })

    result = {
        'protocol': {
            'status': 'post-hoc equivalence and power analysis of saved per-query reciprocal ranks',
            'equivalence_margins': 'chosen for interpretation after the main analysis; not pre-registered',
            'tost': 'two one-sided tests; 90% bootstrap interval is the TOST interval at alpha=0.05',
            'mde': 'two-sided superiority design, alpha=0.05, power=0.80, observed paired standard deviation',
            'bootstrap_resamples': N_BOOT,
            'seed': SEED,
            'note': 'most queries are ties; discordant counts show the effective evidence',
        },
        'comparisons': out,
    }
    with open(os.path.join(RESULTS, 'equivalence_power.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=1, ensure_ascii=False)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(TABLES, 'table_equivalence_power.csv'), index=False)
    with open(os.path.join(TABLES, 'table_equivalence_power.md'), 'w', encoding='utf-8') as f:
        f.write(table.to_markdown(index=False) + '\n')

    make_figure(out)
    print(json.dumps(result, indent=1, ensure_ascii=False))
    print(table.to_string(index=False))


def make_figure(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False})
    names = list(results.keys())
    short = [
        'Zero-shot EW - BM25',
        'Fine-tuned EW - BM25',
        'Fine-tuned EW - zero-shot EW',
        'Tuned fusion - BM25',
        'Tuned fusion - zero-shot tuned',
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    y = np.arange(len(names))[::-1]
    ax.axvspan(-1, 1, color='#DDDDDD', alpha=0.45, linewidth=0)
    ax.axvline(0, color='black', linewidth=0.9)
    for margin, style in [(1.0, '--'), (0.5, ':')]:
        ax.axvline(-margin, color='#666666', linestyle=style, linewidth=0.8)
        ax.axvline(margin, color='#666666', linestyle=style, linewidth=0.8)
    for yi, name in zip(y, names):
        s = results[name]
        mean = s['mean_difference_points']
        lo, hi = s['CI95']
        ax.errorbar(mean, yi, xerr=[[mean - lo], [hi - mean]], fmt='o', color='#1B365D',
                    capsize=3, markersize=5, linewidth=1.2)
        ax.text(hi + 0.08, yi, f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]", va='center', fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(short)
    ax.set_xlabel('MRR@10 difference (percentage points)')
    ax.set_xlim(-2.0, 2.6)
    ax.set_title('Paired differences with 95% bootstrap intervals', loc='left', fontsize=9)
    ax.grid(axis='x', alpha=0.2)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(os.path.join(FIGURES, f'fig13_equivalence_power.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
