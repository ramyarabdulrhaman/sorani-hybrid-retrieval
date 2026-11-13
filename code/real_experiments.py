"""
Real, reproducible retrieval benchmark for Kurdish Sorani (headline -> article retrieval).

Every number reported in the paper's evaluation is computed by this script.
No value is hard-coded. Nothing is randomly generated except the bootstrap
resampling, which is seeded.

Task : the headline of each news article is the query; the article body (headline
       removed) is the only relevant passage among all passages in the corpus.
Data : article_9_kurdish_efficient_llm/data/raw/kurdish_cpt_corpus.txt
       (the real Kurdish Sorani news lines; duplicates and lines without a
       headline/timestamp are removed and counted).
Usage: python code/real_experiments.py
"""
import os, re, json, time, hashlib, math, sys
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(os.path.dirname(BASE), 'article_9_kurdish_efficient_llm', 'data', 'raw', 'kurdish_cpt_corpus.txt')
LOCAL_SRC = os.path.join(BASE, 'data', 'raw', 'kurdish_news_corpus_source.txt')
OUT = os.path.join(BASE, 'results')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_and_algorithms import MorphoBPETokenizer

SEED = 42
N_BOOT = 2000
DENSE_MODELS = {
    'multilingual-e5-base': ('intfloat/multilingual-e5-base', 'query: ', 'passage: ', 'mean'),
    'bge-m3 (dense)': ('BAAI/bge-m3', '', '', 'cls'),
}

# ------------------------------------------------------------------ data
def load_data():
    path = LOCAL_SRC if os.path.exists(LOCAL_SRC) else SRC
    raw = [l.strip() for l in open(path, encoding='utf-8') if l.strip()]
    sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    pat = re.compile(r'^(.{5,300}?)\s*\[\s*\d+\s*:\s*\d+\s*\]\s*(.*)$')
    date_prefix = re.compile(r'^\d{4}\s*/\s*/\s*\d+\s*[-\u2013]?\s*')
    seen, docs = set(), []
    n_dup = n_nohead = 0
    for line in raw:
        if line in seen:
            n_dup += 1
            continue
        seen.add(line)
        m = pat.match(line)
        if not m:
            n_nohead += 1
            continue
        body = date_prefix.sub('', m.group(2)).strip()
        if len(body.split()) < 20:
            n_nohead += 1
            continue
        docs.append({'headline': m.group(1).strip(), 'body': body})
    stats = {'source_lines': len(raw), 'duplicate_lines_removed': n_dup,
             'lines_without_headline_or_body': n_nohead, 'articles': len(docs),
             'source_sha256': sha,
             'mean_body_words': float(np.mean([len(d['body'].split()) for d in docs])),
             'mean_headline_words': float(np.mean([len(d['headline'].split()) for d in docs]))}
    return docs, stats

# ------------------------------------------------------------------ lexical
TOK = MorphoBPETokenizer()
PUNCT = re.compile(r'[\[\]\(\)\{\}<>"\'\u060c\u061b\u061f,.;:!?/\\|\-\u2013\u2014\u00ab\u00bb*]+')

def words(text):
    return [w for w in PUNCT.sub(' ', TOK.normalize_orthography(text)).split() if w]

def morph(text):
    out = []
    for w in words(text):
        out.append(TOK.segment_word(w)[0])
    return out

def char_ngrams(text, n=4):
    out = []
    for w in words(text):
        w = f'#{w}#'
        out.extend(w[i:i + n] for i in range(max(1, len(w) - n + 1)))
    return out

class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.vocab, rows = {}, []
        self.dl = np.array([len(d) for d in docs_tokens], dtype=float)
        self.avg = self.dl.mean()
        df = {}
        self.tf = []
        for d in docs_tokens:
            c = {}
            for t in d:
                c[t] = c.get(t, 0) + 1
            self.tf.append(c)
            for t in c:
                df[t] = df.get(t, 0) + 1
        n = len(docs_tokens)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.inv = {}
        for i, c in enumerate(self.tf):
            for t, f in c.items():
                self.inv.setdefault(t, []).append((i, f))
        self.n = n

    def scores(self, q_tokens):
        s = np.zeros(self.n)
        for t in set(q_tokens):
            if t not in self.inv:
                continue
            for i, f in self.inv[t]:
                s[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avg))
        return s

# ------------------------------------------------------------------ dense
def embed(model_id, texts, pool, bs=8, max_len=512):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            normalized = [TOK.normalize_orthography(text) for text in texts[i:i + bs]]
            enc = tok(normalized, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
            h = model(**enc).last_hidden_state
            if pool == 'cls':
                v = h[:, 0]
            else:
                m = enc['attention_mask'].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1)
            vecs.append(torch.nn.functional.normalize(v, dim=-1).numpy())
    return np.concatenate(vecs)

# ------------------------------------------------------------------ metrics
def rank_of_gold(score_matrix):
    """1-based rank of the gold passage (index i for query i); ties resolved pessimistically."""
    n = score_matrix.shape[0]
    gold = score_matrix[np.arange(n), np.arange(n)][:, None]
    return 1 + (score_matrix > gold).sum(1) + ((score_matrix == gold).sum(1) - 1)

def metrics_from_ranks(r):
    return {'Recall@1': float((r <= 1).mean() * 100), 'Recall@5': float((r <= 5).mean() * 100),
            'Recall@10': float((r <= 10).mean() * 100), 'Recall@20': float((r <= 20).mean() * 100),
            'MRR@10': float(np.where(r <= 10, 1.0 / r, 0).mean() * 100),
            'nDCG@10': float(np.where(r <= 10, 1.0 / np.log2(r + 1), 0).mean() * 100)}

def bootstrap_ci(r, fn, rng):
    n = len(r)
    vals = [fn(r[rng.integers(0, n, n)]) for _ in range(N_BOOT)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

def rrf(score_mats, k=60):
    out = 0
    for s in score_mats:
        order = np.argsort(-s, axis=1)
        ranks = np.empty_like(order)
        rows = np.arange(s.shape[0])[:, None]
        ranks[rows, order] = np.arange(1, s.shape[1] + 1)[None, :]
        out = out + 1.0 / (k + ranks)
    return out

def main():
    docs, stats = load_data()
    n = len(docs)
    print('Data:', json.dumps(stats, indent=1))
    queries = [d['headline'] for d in docs]
    passages = [d['body'] for d in docs]
    systems, latency = {}, {}

    def timed(name, fn):
        t0 = time.perf_counter()
        mat = fn()
        latency[name] = (time.perf_counter() - t0) / n * 1000
        systems[name] = mat

    def bm25_system(tokfn):
        def run():
            bm = BM25([tokfn(p) for p in passages])
            return np.stack([bm.scores(tokfn(q)) for q in queries])
        return run

    timed('BM25 (word)', bm25_system(words))
    timed('BM25 (morpheme stems)', bm25_system(morph))
    timed('BM25 (char 4-gram)', bm25_system(char_ngrams))

    for name, (mid, qp, pp, pool) in DENSE_MODELS.items():
        cache = os.path.join(OUT, f"emb_{mid.replace('/', '_')}_normalized.npz")
        t0 = time.perf_counter()
        if os.path.exists(cache):
            z = np.load(cache)
            qe, pe = z['q'], z['p']
        else:
            print('Encoding with', mid, flush=True)
            qe = embed(mid, [qp + q for q in queries], pool, bs=32, max_len=64)
            pe = embed(mid, [pp + p for p in passages], pool, bs=8, max_len=512)
            np.savez_compressed(cache, q=qe, p=pe)
        t1 = time.perf_counter()
        systems[name] = qe @ pe.T
        latency[name] = (time.perf_counter() - t1) / n * 1000

    dense_names = list(DENSE_MODELS)
    systems['Hybrid RRF (BM25 stems + e5-base)'] = rrf([systems['BM25 (morpheme stems)'], systems[dense_names[0]]])
    systems['Hybrid RRF (BM25 stems + bge-m3)'] = rrf([systems['BM25 (morpheme stems)'], systems[dense_names[1]]])
    systems['Hybrid RRF (BM25 stems + e5-base + bge-m3)'] = rrf(
        [systems['BM25 (morpheme stems)'], systems[dense_names[0]], systems[dense_names[1]]])

    rng = np.random.default_rng(SEED)
    results, ranks = {}, {}
    for name, mat in systems.items():
        r = rank_of_gold(mat)
        ranks[name] = r
        m = metrics_from_ranks(r)
        m['Recall@5_CI95'] = bootstrap_ci(r, lambda x: (x <= 5).mean() * 100, rng)
        m['MRR@10_CI95'] = bootstrap_ci(r, lambda x: np.where(x <= 10, 1.0 / x, 0).mean() * 100, rng)
        if name in latency:
            m['retrieval_ms_per_query'] = latency[name]
        results[name] = m

    # paired bootstrap vs. BM25 (word) on reciprocal rank
    base = np.where(ranks['BM25 (word)'] <= 10, 1.0 / ranks['BM25 (word)'], 0)
    sig = {}
    for name, r in ranks.items():
        if name == 'BM25 (word)':
            continue
        rr = np.where(r <= 10, 1.0 / r, 0)
        d = rr - base
        boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
        p = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
        sig[name] = {'mean_MRR@10_diff': float(d.mean() * 100),
                     'CI95': [float(np.percentile(boots, 2.5) * 100), float(np.percentile(boots, 97.5) * 100)],
                     'p_two_sided': max(p, 1.0 / N_BOOT)}

    out = {'data': stats, 'seed': SEED, 'bootstrap_resamples': N_BOOT, 'systems': results,
           'paired_bootstrap_vs_BM25_word': sig}
    with open(os.path.join(OUT, 'real_results.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, 'per_query_gold_ranks.csv'), 'w', encoding='utf-8') as f:
        names = list(ranks)
        f.write('query_index,' + ','.join('"%s"' % x for x in names) + '\n')
        for i in range(n):
            f.write(f'{i},' + ','.join(str(int(ranks[x][i])) for x in names) + '\n')
    print(json.dumps({k: {m: (round(v, 2) if isinstance(v, float) else v) for m, v in d.items()}
                      for k, d in results.items()}, indent=1))

if __name__ == '__main__':
    main()
