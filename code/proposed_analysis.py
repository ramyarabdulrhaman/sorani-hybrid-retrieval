"""Tables 7-9 and Figure 11 from results/proposed_results.json and proposed_selection_log.json."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = json.load(open(os.path.join(BASE, 'results', 'proposed_results.json'), encoding='utf-8'))
LOG = json.load(open(os.path.join(BASE, 'results', 'proposed_selection_log.json'), encoding='utf-8'))
SYS, CMP, CFG = RES['systems'], RES['paired_comparisons'], RES['config']
TAB, FIG = os.path.join(BASE, 'tables'), os.path.join(BASE, 'figures')
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25})
PROP = 'Proposed: KurdRAG Stage 1 (fine-tuned dense + lexical fusion)'
PROP_LABEL = 'Validation-tuned fusion (fine-tuned E5 + char-4gram BM25 + bge-m3)'
ORDER = list(SYS)


def write_table(name, df):
    df.to_csv(os.path.join(TAB, name + '.csv'), index=False, encoding='utf-8')
    cols = list(df.columns)
    md = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join('---' for _ in cols) + '|']
    md += ['| ' + ' | '.join(str(v) for v in r.values) + ' |' for _, r in df.iterrows()]
    open(os.path.join(TAB, name + '.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')

def holm_adjusted_pvalues(reported_p_values):
    p = pd.Series(reported_p_values, dtype=float).to_numpy()
    order = p.argsort()
    adjusted = p.copy()
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


rows = []
plot_extra = []
seed_count = len(CFG['training_seeds'])
partition_count = len(CFG.get('outer_partition_seeds', [CFG.get('outer_seed')]))
for n in ORDER:
    m = SYS[n]
    rows.append([PROP_LABEL if n == PROP else n, f"{m['Recall@1']:.2f}", f"{m['MRR@10']:.2f}", f"[{m['MRR@10_CI95'][0]:.2f}, {m['MRR@10_CI95'][1]:.2f}]",
                 f"{m['MRR@10_seed_std']:.2f}"])

equal_path = os.path.join(BASE, 'results', 'equal_weight_fusion_crossfit.json')
zero_path = os.path.join(BASE, 'results', 'equal_weight_zero_shot_crossfit.json')
if os.path.exists(equal_path) and os.path.exists(zero_path):
    equal_result = json.load(open(equal_path, encoding='utf-8'))
    zero_result = json.load(open(zero_path, encoding='utf-8'))
    expected_partitions = CFG.get('outer_partition_seeds', [CFG.get('outer_seed')])
    for label, result in [('equal-weight', equal_result), ('zero-shot', zero_result)]:
        actual_partitions = result['protocol'].get(
            'outer_partition_seeds', [result['protocol'].get('outer_seed')])
        if actual_partitions != expected_partitions:
            raise ValueError(
                f"Stale {label} fusion outputs; rerun the fusion scripts for "
                "the current proposed_results.json protocol before analysis."
            )
    equal = equal_result['equal_weight_fusion']
    zero = zero_result['zero_shot_equal_weight_fusion']
    rows.extend([
        ['Zero-shot equal-weight fusion', f"{zero['Recall@1']:.2f}",
         f"{zero['MRR@10']:.2f}",
         f"[{zero['MRR@10_CI95'][0]:.2f}, {zero['MRR@10_CI95'][1]:.2f}]",
         '0.00'],
        ['Fine-tuned equal-weight fusion', f"{equal['Recall@1']:.2f}",
         f"{equal['MRR@10']:.2f}",
         f"[{equal['MRR@10_CI95'][0]:.2f}, {equal['MRR@10_CI95'][1]:.2f}]",
         f"{np.std(equal['seed_mean_mrr10'], ddof=1):.2f}"],
    ])
    plot_extra = [
        ("Zero-shot equal-weight fusion", zero["MRR@10"],
         zero["MRR@10_CI95"], "zero-shot\nequal weights", "#0ea5e9"),
        ("Fine-tuned equal-weight fusion", equal["MRR@10"],
         equal["MRR@10_CI95"], "fine-tuned\nequal weights", "#f59e0b"),
    ]
write_table('table_7_proposed_method_results', pd.DataFrame(
    rows, columns=['System', 'Recall@1', 'MRR@10', 'MRR@10 95% CI',
                   f'SD over {seed_count} training seeds ({partition_count} partitions)']))

cmp_items = list(CMP.items())
reported_ps = [1 / 2000 if v['p_two_sided'] < 0.001 else float(f"{v['p_two_sided']:.3f}") for _, v in cmp_items]
holm_ps = holm_adjusted_pvalues(reported_ps)
rows = []
for (k, v), p_holm in zip(cmp_items, holm_ps):
    p = v['p_two_sided']
    rows.append([k.replace('Proposed vs ', ''), f"{v['mean_MRR@10_diff']:+.2f}", f"[{v['CI95'][0]:+.2f}, {v['CI95'][1]:+.2f}]",
                 '< 0.001' if p < 0.001 else f'{p:.3f}', f'{p_holm:.3f}',
                 'yes' if v['CI95'][0] > 0 or v['CI95'][1] < 0 else 'no'])
write_table('table_8_proposed_paired_comparisons', pd.DataFrame(
    rows, columns=['Fused retriever minus (system B)', 'Delta MRR@10 (pts)', '95% bootstrap CI', 'p (two-sided)', 'Holm-adjusted p', 'CI excludes 0']))

rows = []
final = {}  # resumed runs log a FINAL entry again; keep the last one per partition/fold/seed
for e in LOG:
    if 'FINAL_selected' in e:
        final[(e.get('partition_seed', CFG.get('outer_seed')), e['seed'], e['fold'])] = e
for _, e in sorted(final.items()):
        s = e['FINAL_selected']
        rows.append([e.get('partition_seed', CFG.get('outer_seed')), e['seed'], e['fold'],
                     s['mode'], f"{s['lr']:g}", s['epochs'],
                     '/'.join(f'{x:.1f}' for x in s['w']), f"{s['val']:.2f}"])
write_table('table_9_selected_configurations', pd.DataFrame(
    rows, columns=['Partition seed', 'Training seed', 'Fold', 'Text mode', 'Learning rate',
                   'Epochs', 'Weights BM25-c4 / fine-tuned / bge-m3', 'Validation MRR@10']))

short = {'BM25 (word)': 'BM25\nword', 'BM25 (morpheme stems)': 'BM25\nstems', 'BM25 (char 4-gram)': 'BM25\nchar-4g',
         'multilingual-e5-base zero-shot': 'e5\nzero-shot', 'bge-m3 zero-shot': 'bge-m3\nzero-shot',
         'Hybrid RRF (BM25 char 4-gram + bge-m3)': 'RRF\nc4+bge', 'Fusion of zero-shot components (tuned weights)': 'zero-shot\nfusion',
         'e5-base fine-tuned (selected config)': 'e5\nfine-tuned', PROP: 'Tuned\nfusion'}
fig, ax = plt.subplots(figsize=(7.2, 3.6))
vals = [SYS[n]['MRR@10'] for n in ORDER] + [item[1] for item in plot_extra]
lo = [SYS[n]['MRR@10'] - SYS[n]['MRR@10_CI95'][0] for n in ORDER]
hi = [SYS[n]['MRR@10_CI95'][1] - SYS[n]['MRR@10'] for n in ORDER]
lo += [item[1] - item[2][0] for item in plot_extra]
hi += [item[2][1] - item[1] for item in plot_extra]
cols = ['#16a34a' if n == PROP else '#94a3b8' for n in ORDER]
cols += [item[4] for item in plot_extra]
ax.bar(range(len(vals)), vals, yerr=[lo, hi], color=cols, capsize=2)
for i, v in enumerate(vals):
    ax.text(i, 61, f'{v:.2f}', ha='center', va='bottom', color='white', fontsize=7, fontweight='bold')
labels = [short[n] for n in ORDER] + [item[3] for item in plot_extra]
ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel('MRR@10 (%)'); ax.set_ylim(60, 101)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig11_finetuning_results.png'), dpi=400)
fig.savefig(os.path.join(FIG, 'fig11_finetuning_results.pdf'))
print('ok')
