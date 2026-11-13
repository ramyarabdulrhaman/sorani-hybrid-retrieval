"""
Builds every table (tables/*.csv|md) and figure (figures/fig2..fig9) of the paper
directly from results/real_results.json and results/per_query_gold_ranks.csv,
plus the real tokenization statistics used in Table 5 / Figure 8 / Figure 10.
No number is typed by hand. Run after real_experiments.py.
"""
import os, sys, json, collections, platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_experiments as R

RES = json.load(open(os.path.join(BASE, 'results', 'real_results.json'), encoding='utf-8'))
RANKS = pd.read_csv(os.path.join(BASE, 'results', 'per_query_gold_ranks.csv'))
TAB = os.path.join(BASE, 'tables'); FIG = os.path.join(BASE, 'figures')
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
SYS = RES['systems']
NAMES = list(SYS)
SHORT = {'BM25 (word)': 'BM25 word', 'BM25 (morpheme stems)': 'BM25 stems', 'BM25 (char 4-gram)': 'BM25 char-4g',
         'multilingual-e5-base': 'mE5-base', 'bge-m3 (dense)': 'bge-m3',
         'Hybrid RRF (BM25 stems + e5-base)': 'RRF stems+mE5',
         'Hybrid RRF (BM25 stems + bge-m3)': 'RRF stems+bge-m3',
         'Hybrid RRF (BM25 stems + e5-base + bge-m3)': 'RRF stems+mE5+bge-m3'}
COL = {'BM25 (word)': '#64748b', 'BM25 (morpheme stems)': '#2563eb', 'BM25 (char 4-gram)': '#0ea5e9',
       'multilingual-e5-base': '#f59e0b', 'bge-m3 (dense)': '#dc2626',
       'Hybrid RRF (BM25 stems + e5-base)': '#a78bfa',
       'Hybrid RRF (BM25 stems + bge-m3)': '#059669',
       'Hybrid RRF (BM25 stems + e5-base + bge-m3)': '#be185d'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': .25, 'figure.dpi': 120})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + '.png'), dpi=400)
    fig.savefig(os.path.join(FIG, name + '.pdf'))
    plt.close(fig)

def write_table(name, df):
    df.to_csv(os.path.join(TAB, name + '.csv'), index=False, encoding='utf-8')
    cols = list(df.columns)
    md = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join('---' for _ in cols) + '|']
    for _, r in df.iterrows():
        md.append('| ' + ' | '.join(str(v) for v in r.values) + ' |')
    open(os.path.join(TAB, name + '.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')

def holm_adjusted_pvalues(reported_p_values):
    p = np.asarray(reported_p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted

rk = {n: RANKS[n].values for n in NAMES}
N = len(RANKS)

# ---------------------------------------------------------------- tokenization statistics
from transformers import AutoTokenizer
xlmr = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
docs, dstats = R.load_data()
ws = [w for d in docs for w in R.words(d['body'])]
cnt = collections.Counter(ws)
det_tokens = sum(n for w, n in cnt.items() if len(R.TOK.segment_word(w)) == 2)
det_types = sum(1 for w in cnt if len(R.TOK.segment_word(w)) == 2)
fert_raw = sum(len(xlmr.tokenize(w)) for w in ws) / len(ws)
fert_det = 0
for w in ws:
    for p in R.TOK.segment_word(w):
        fert_det += len(xlmr.tokenize(p.replace('##', '')))
fert_det /= len(ws)
tokstats = {'word_tokens': len(ws), 'word_types': len(cnt),
            'detached_token_share_pct': det_tokens / len(ws) * 100, 'detached_type_share_pct': det_types / len(cnt) * 100,
            'fertility_raw': fert_raw, 'fertility_detached': fert_det}
json.dump(tokstats, open(os.path.join(BASE, 'results', 'tokenization_stats.json'), 'w', encoding='utf-8'), indent=2)

# Case-study words for Fig. 10: most frequent words (>= 15 occurrences) that the heuristic splits
cands = [(w, n) for w, n in cnt.most_common() if n >= 15 and len(R.TOK.segment_word(w)) == 2]
case = [{'word': w, 'count': n, 'xlmr': xlmr.tokenize(w), 'split': R.TOK.segment_word(w),
         'stem_xlmr': xlmr.tokenize(R.TOK.segment_word(w)[0])} for w, n in cands]
json.dump(case, open(os.path.join(BASE, 'results', 'tokenization_case_words.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- tables
D = RES['data']
write_table('table_1_corpus_and_benchmark_profile', pd.DataFrame([
    ['Source file lines (kurdish_news_corpus_source.txt)', f"{D['source_lines']:,}"],
    ['Exact duplicate lines removed', f"{D['duplicate_lines_removed']:,}"],
    ['Lines excluded (no headline/timestamp pattern or body < 20 words)', f"{D['lines_without_headline_or_body']:,}"],
    ['Articles kept (= passages = queries)', f"{D['articles']:,}"],
    ['Mean body length (words)', f"{D['mean_body_words']:.1f}"],
    ['Mean headline length (words)', f"{D['mean_headline_words']:.1f}"],
    ['Word tokens in passages', f"{tokstats['word_tokens']:,}"],
    ['Distinct word types in passages', f"{tokstats['word_types']:,}"],
    ['Source SHA-256 (first 16 hex)', D['source_sha256'][:16]],
], columns=['Property', 'Value']))

rows = []
for n in NAMES:
    m = SYS[n]
    rows.append([n, f"{m['Recall@1']:.2f}", f"{m['Recall@5']:.2f} [{m['Recall@5_CI95'][0]:.1f}, {m['Recall@5_CI95'][1]:.1f}]",
                 f"{m['Recall@10']:.2f}", f"{m['Recall@20']:.2f}",
                 f"{m['MRR@10']:.2f} [{m['MRR@10_CI95'][0]:.1f}, {m['MRR@10_CI95'][1]:.1f}]", f"{m['nDCG@10']:.2f}"])
write_table('table_2_main_benchmark_results', pd.DataFrame(rows, columns=[
    'System', 'Recall@1 (%)', 'Recall@5 (%) [95% CI]', 'Recall@10 (%)', 'Recall@20 (%)', 'MRR@10 (%) [95% CI]', 'nDCG@10 (%)']))

sig_items = list(RES['paired_bootstrap_vs_BM25_word'].items())
reported_ps = [1 / 2000 if v['p_two_sided'] < 0.001 else float(f"{v['p_two_sided']:.3f}") for _, v in sig_items]
holm_ps = holm_adjusted_pvalues(reported_ps)
rows = []
for (n, v), p_holm in zip(sig_items, holm_ps):
    p = v['p_two_sided']
    rows.append([n, f"{v['mean_MRR@10_diff']:+.2f}", f"[{v['CI95'][0]:+.2f}, {v['CI95'][1]:+.2f}]",
                 ('< 0.001' if p < 0.001 else f'{p:.3f}'), f'{p_holm:.3f}',
                 'yes' if (v['CI95'][0] > 0 or v['CI95'][1] < 0) else 'no'])
write_table('table_3_statistical_significance_tests', pd.DataFrame(rows, columns=[
    'System vs. BM25 (word)', 'Delta MRR@10 (pts)', '95% bootstrap CI', 'p (two-sided)', 'Holm-adjusted p', 'CI excludes 0']))

lens = np.array([len(d['headline'].split()) for d in docs])
edges = np.quantile(lens, [1 / 3, 2 / 3])
bucket = np.digitize(lens, edges, right=True)
blabel = []
for b in range(3):
    lo = lens[bucket == b].min(); hi = lens[bucket == b].max()
    blabel.append(f'{lo}-{hi} words (n={int((bucket == b).sum())})')
rows = []
for n in NAMES:
    rows.append([n] + [f"{(rk[n][bucket == b] <= 1).mean() * 100:.2f}" for b in range(3)])
write_table('table_4_headline_length_breakdown', pd.DataFrame(rows, columns=['System (Recall@1 %)'] + blabel))

write_table('table_5_tokenization_statistics', pd.DataFrame([
    ['Word tokens in passages', f"{tokstats['word_tokens']:,}"],
    ['Word types in passages', f"{tokstats['word_types']:,}"],
    ['Word tokens split by the clitic heuristic (%)', f"{tokstats['detached_token_share_pct']:.2f}"],
    ['Word types split by the clitic heuristic (%)', f"{tokstats['detached_type_share_pct']:.2f}"],
    ['XLM-R subword fertility, raw words (pieces/word)', f"{fert_raw:.3f}"],
    ['XLM-R subword fertility, after clitic split (pieces/word)', f"{fert_det:.3f}"],
    ['BM25 stems vs. BM25 word: Delta MRR@10 (pts) [95% CI]',
     f"{RES['paired_bootstrap_vs_BM25_word']['BM25 (morpheme stems)']['mean_MRR@10_diff']:+.2f} "
     f"[{RES['paired_bootstrap_vs_BM25_word']['BM25 (morpheme stems)']['CI95'][0]:+.2f}, "
     f"{RES['paired_bootstrap_vs_BM25_word']['BM25 (morpheme stems)']['CI95'][1]:+.2f}]"],
], columns=['Tokenization / morphology measurement', 'Value']))

import torch
rows = []
for n in NAMES:
    m = SYS[n]
    rows.append([n, f"{m['retrieval_ms_per_query']:.2f}" if 'retrieval_ms_per_query' in m else 'n/a (sum of components)'])
write_table('table_6_computational_efficiency', pd.DataFrame(rows, columns=[
    'System', 'Scoring time (ms/query, 492 queries x 492 passages, CPU)']))

# ---------------------------------------------------------------- figures
ks = np.arange(1, 21)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for n in NAMES:
    ax.plot(ks, [(rk[n] <= k).mean() * 100 for k in ks], label=SHORT[n], color=COL[n], lw=1.8)
ax.set_xlabel('Retrieval depth k'); ax.set_ylabel('Recall@k (%)'); ax.set_ylim(55, 100.5); ax.set_xticks([1, 5, 10, 15, 20])
ax.legend(fontsize=7, loc='lower right', ncol=2)
save(fig, 'fig2_recall_at_k_curves')

fig, ax = plt.subplots(figsize=(6.4, 3.8))
x = np.arange(len(NAMES)); w = 0.27
for i, (key, c) in enumerate([('Recall@1', '#2563eb'), ('MRR@10', '#059669'), ('nDCG@10', '#f59e0b')]):
    ax.bar(x + (i - 1) * w, [SYS[n][key] for n in NAMES], w, label=key, color=c)
ax.set_xticks(x); ax.set_xticklabels([SHORT[n] for n in NAMES], rotation=30, ha='right'); ax.set_ylabel('Score (%)')
ax.set_ylim(50, 101); ax.legend(ncol=3, fontsize=8)
save(fig, 'fig3_mrr_ndcg_comparison')

fig, ax = plt.subplots(figsize=(6.4, 3.6))
sig = RES['paired_bootstrap_vs_BM25_word']
names = list(sig)
y = np.arange(len(names))[::-1]
for yi, n in zip(y, names):
    v = sig[n]; lo, hi = v['CI95']
    ax.plot([lo, hi], [yi, yi], color=COL[n], lw=2.2); ax.plot(v['mean_MRR@10_diff'], yi, 'o', color=COL[n])
ax.axvline(0, color='k', lw=.8)
ax.set_yticks(y); ax.set_yticklabels([SHORT[n] for n in names]); ax.set_xlabel('Delta MRR@10 vs. BM25 word (points), 95% paired bootstrap CI')
save(fig, 'fig5_paired_bootstrap_forest')

fig, ax = plt.subplots(figsize=(6.4, 3.8))
for n in NAMES:
    ax.plot(range(3), [(rk[n][bucket == b] <= 1).mean() * 100 for b in range(3)], 'o-', color=COL[n], label=SHORT[n])
ax.set_xticks(range(3)); ax.set_xticklabels(blabel, fontsize=7.5); ax.set_ylabel('Recall@1 (%)'); ax.legend(fontsize=7, ncol=2)
save(fig, 'fig7_headline_length_breakdown')

bins = [(1, 1, 'rank 1'), (2, 5, 'rank 2-5'), (6, 10, 'rank 6-10'), (11, 20, 'rank 11-20'), (21, 10 ** 9, 'rank > 20')]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
left = np.zeros(len(NAMES))
for (lo, hi, lab), c in zip(bins, ['#059669', '#84cc16', '#facc15', '#fb923c', '#dc2626']):
    v = np.array([((rk[n] >= lo) & (rk[n] <= hi)).mean() * 100 for n in NAMES])
    ax.barh(range(len(NAMES)), v, left=left, color=c, label=lab); left += v
ax.set_yticks(range(len(NAMES))); ax.set_yticklabels([SHORT[n] for n in NAMES]); ax.invert_yaxis()
ax.set_xlabel('Share of queries (%)'); ax.legend(ncol=5, fontsize=7, loc='upper center', bbox_to_anchor=(.5, 1.14))
save(fig, 'fig4_gold_rank_distribution')

a, b = 'BM25 (morpheme stems)', 'bge-m3 (dense)'
both = ((rk[a] == 1) & (rk[b] == 1)).mean() * 100; la = ((rk[a] == 1) & (rk[b] != 1)).mean() * 100
db = ((rk[a] != 1) & (rk[b] == 1)).mean() * 100; nei = ((rk[a] != 1) & (rk[b] != 1)).mean() * 100
comp = {'both': both, 'lexical_only': la, 'dense_only': db, 'neither': nei}
json.dump(comp, open(os.path.join(BASE, 'results', 'complementarity_stems_vs_bge.json'), 'w', encoding='utf-8'), indent=2)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
vals = [both, la, db, nei]
bars = ax.bar(['Both\nrank gold 1st', 'BM25 stems\nonly', 'bge-m3\nonly', 'Neither'], vals, color=['#059669', '#2563eb', '#dc2626', '#94a3b8'])
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f'{v:.1f}%', ha='center', fontsize=8)
ax.set_ylabel('Share of queries (%)'); ax.set_ylim(0, 100)
save(fig, 'fig8_lexical_dense_complementarity')

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.0))
a1.bar(['raw words', 'after clitic split'], [fert_raw, fert_det], color=['#f59e0b', '#2563eb'])
for i, v in enumerate([fert_raw, fert_det]):
    a1.text(i, v + .05, f'{v:.3f}', ha='center', fontsize=8)
a1.set_ylim(0, 4.2); a1.set_ylabel('XLM-R pieces per word')
a2.bar(['word tokens', 'word types'], [tokstats['detached_token_share_pct'], tokstats['detached_type_share_pct']], color=['#059669', '#a78bfa'])
for i, v in enumerate([tokstats['detached_token_share_pct'], tokstats['detached_type_share_pct']]):
    a2.text(i, v + .3, f'{v:.2f}%', ha='center', fontsize=8)
a2.set_ylabel('Share split by heuristic (%)'); a2.set_ylim(0, 17)
save(fig, 'fig9_tokenizer_fertility')

fig, ax = plt.subplots(figsize=(6.4, 3.6))
for yi, n in zip(np.arange(len(NAMES))[::-1], NAMES):
    lo, hi = SYS[n]['Recall@5_CI95']
    ax.plot([lo, hi], [yi, yi], color=COL[n], lw=2.2); ax.plot(SYS[n]['Recall@5'], yi, 'o', color=COL[n])
ax.set_yticks(np.arange(len(NAMES))[::-1]); ax.set_yticklabels([SHORT[n] for n in NAMES])
ax.set_xlabel('Recall@5 (%) with 95% bootstrap CI')
save(fig, 'fig6_recall5_confidence_intervals')

print(json.dumps({'tokstats': tokstats, 'complementarity': comp, 'terciles': blabel}, indent=1))
