"""Builds code/figure_html/fig10.html from the real tokenization measurements
(results/tokenization_case_words.json, tokenization_stats.json, real_results.json).
Render afterwards with render_html_figures.py."""
import os, json, html

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, 'results')
case = {c['word']: c for c in json.load(open(os.path.join(RES, 'tokenization_case_words.json'), encoding='utf-8'))}
ts = json.load(open(os.path.join(RES, 'tokenization_stats.json')))
rr = json.load(open(os.path.join(RES, 'real_results.json'), encoding='utf-8'))
st = rr['paired_bootstrap_vs_BM25_word']['BM25 (morpheme stems)']

# Words are the most frequent split words; validity was judged by the authors (see paper text).
VALID = ['کوردستاندا', 'کاتێکدا', 'تایبەتدا']
INVALID = ['کوردستان', 'بەغدا', 'پەرلەمان']

def pieces(items, cls):
    out = []
    for p in items:
        p = p.replace('\u2581', '')
        out.append(f'<div class="tok {cls}"><span class="kur k">{html.escape(p) if p else "&#9601;"}</span></div>')
    return ''.join(out)

def block(word, ok):
    c = case[word]
    stem, clitic = c['split'][0], c['split'][1].replace('##', '')
    badge = ('<span class="badge ok">valid split</span>' if ok else '<span class="badge no">over-segmentation</span>')
    cls2 = 'g' if ok else 'r'
    return f'''<div class="blk"><div class="bh"><span class="kur w">{word}</span><span class="cnt">{c["count"]} occurrences in the corpus</span>{badge}</div>
<div class="line"><div class="nm">XLM-R pieces ({len(c["xlmr"])})</div><div class="toks">{pieces(c["xlmr"], "t0")}</div></div>
<div class="line"><div class="nm">Clitic heuristic</div><div class="toks">{pieces([stem], "t" + cls2)}<div class="tok tc"><span class="k plus">+</span></div>{pieces([clitic], "tc")}</div></div></div>'''

base_css = '''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<style>
@page { size: 1180px 640px; margin: 0; }
*{box-sizing:border-box;margin:0;padding:0}
body{width:1180px;height:640px;overflow:hidden;background:#fff;font-family:"Segoe UI",Arial,sans-serif;color:#1e293b;padding:14px}
.kur{font-family:"Segoe UI",Tahoma,Arial,sans-serif;direction:rtl;unicode-bidi:isolate}
.hdr{background:#0f172a;border-radius:12px;color:#fff;text-align:center;padding:13px 10px 11px}
.hdr h1{font-size:19px;font-weight:700}
.hdr p{font-size:10.5px;color:#cbd5e1;margin-top:4px}
'''
extra = '''
.cols{display:flex;gap:12px;margin-top:10px}
.panel{flex:1;border:1.5px solid #e2e8f0;border-radius:12px;padding:10px 12px}
.ph2{font-size:12.5px;font-weight:700;margin-bottom:6px}
.blk{border:1.2px solid #e2e8f0;border-radius:9px;padding:7px 9px;margin-top:8px;background:#f8fafc}
.bh{display:flex;align-items:center;gap:10px;margin-bottom:5px}
.w{font-size:21px;font-weight:700;color:#1e293b}
.cnt{font-size:9px;color:#64748b;flex:1}
.badge{font-size:9px;font-weight:700;border-radius:10px;padding:2px 9px}
.badge.ok{background:#dcfce7;color:#166534;border:1px solid #86efac}
.badge.no{background:#fee2e2;color:#991b1b;border:1px solid #fca5a5}
.line{display:flex;align-items:center;gap:8px;margin-top:4px}
.nm{width:96px;font-size:9px;font-weight:700;color:#475569;flex:none}
.toks{display:flex;gap:5px;flex-wrap:wrap}
.tok{border-radius:6px;padding:2px 9px;border:1.5px solid;background:#fff;min-width:34px;text-align:center}
.k{font-size:17px;font-weight:700}
.plus{font-size:13px}
.t0{border-color:#fbbf24;background:#fffbeb;color:#92400e}
.tg{border-color:#34d399;background:#ecfdf5;color:#065f46}
.tr{border-color:#f87171;background:#fef2f2;color:#991b1b}
.tc{border-color:#818cf8;background:#eef2ff;color:#3730a3}
.foot{background:#0f172a;border-radius:12px;margin-top:10px;padding:10px 12px;color:#e2e8f0}
.foot h4{color:#38bdf8;font-size:10px;margin-bottom:7px}
.stats{display:flex;gap:8px}
.st{flex:1;background:#1e293b;border:1px solid #334155;border-radius:7px;padding:6px 8px;text-align:center;font-size:9.5px}
.st div{color:#94a3b8;margin-bottom:3px}
.st span{font-weight:700;color:#34d399;font-size:12px}
.note{font-size:9px;color:#64748b;margin-top:8px;line-height:1.4}
'''
body = f'''</style></head><body>
<div class="hdr"><h1>Clitic Splitting: Plausible and Over-Segmented Examples</h1>
<p>Real XLM-R SentencePiece pieces (the tokenizer of multilingual-e5 and bge-m3) vs. the single-suffix clitic heuristic, on the most frequent words it splits</p></div>
<div class="cols">
<div class="panel"><div class="ph2" style="color:#166534">A. Splits that match a genuine Sorani suffix</div>{''.join(block(w, True) for w in VALID)}</div>
<div class="panel"><div class="ph2" style="color:#991b1b">B. Over-segmentation: the word ends in a clitic-like string but is not inflected</div>{''.join(block(w, False) for w in INVALID)}</div>
</div>
<div class="foot"><h4>MEASURED ON {ts["word_tokens"]:,} WORD TOKENS ({ts["word_types"]:,} TYPES) OF THE {rr["data"]["articles"]} PASSAGES</h4><div class="stats">
<div class="st"><div>Word tokens split by heuristic</div><span>{ts["detached_token_share_pct"]:.2f}%</span></div>
<div class="st"><div>Word types split</div><span>{ts["detached_type_share_pct"]:.2f}%</span></div>
<div class="st"><div>XLM-R pieces/word: raw &rarr; split</div><span>{ts["fertility_raw"]:.3f} &rarr; {ts["fertility_detached"]:.3f}</span></div>
<div class="st"><div>BM25 stems vs. word, &Delta;MRR@10 (95% CI)</div><span>{st["mean_MRR@10_diff"]:+.2f} [{st["CI95"][0]:+.2f}, {st["CI95"][1]:+.2f}]</span></div>
</div><div class="note">Words are the most frequent ones the heuristic splits; the valid/over-segmentation labels are the authors' judgement and have not been validated by an independent annotator. The splitter has no lexicon, so it cannot tell inflected words from words that merely end in the same letters.</div></div>
</body></html>'''
open(os.path.join(BASE, 'code', 'figure_html', 'fig10.html'), 'w', encoding='utf-8').write(base_css + extra + body)
print('fig10.html written')
