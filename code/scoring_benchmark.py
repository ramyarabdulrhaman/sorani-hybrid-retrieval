"""Consistent scoring-stage timing for the headline systems.

The earlier timing table mixed BM25 index construction with dense matrix
multiplication. This script times only the scoring/ranking stage given
precomputed query representations:

* BM25: index is built before timing; query tokens are precomputed; only
  ``BM25.scores`` is timed.
* Dense: embeddings are loaded before timing; only the query-passage dot
  product is timed.
* RRF: component score matrices are precomputed; only rank fusion is timed.

Query encoding, model loading, and index construction are excluded and are
reported separately where available. This is not end-to-end latency.
"""
import json
import os
import platform
import statistics
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'code'))
import real_experiments as R

RESULTS = os.path.join(BASE, 'results')
TABLES = os.path.join(BASE, 'tables')
REPEATS = 5


def time_loop(fn, repeats=REPEATS):
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        vals.append((time.perf_counter() - t0) * 1000)
    return vals


def main():
    docs, stats = R.load_data()
    queries = [d['headline'] for d in docs]
    passages = [d['body'] for d in docs]
    n = len(docs)

    specs = {
        'BM25 (word)': R.words,
        'BM25 (morpheme stems)': R.morph,
        'BM25 (char 4-gram)': R.char_ngrams,
    }
    timings = {}
    matrices = {}
    build_seconds = {}
    for name, tokfn in specs.items():
        t0 = time.perf_counter()
        bm = R.BM25([tokfn(p) for p in passages])
        build_seconds[name] = time.perf_counter() - t0
        q_tokens = [tokfn(q) for q in queries]

        def run(bm=bm, q_tokens=q_tokens):
            for q in q_tokens:
                bm.scores(q)

        vals = time_loop(run)
        timings[name] = statistics.median(vals) / n
        matrices[name] = np.stack([bm.scores(q) for q in q_tokens])

    emb = {
        'multilingual-e5-base': 'emb_intfloat_multilingual-e5-base_normalized.npz',
        'bge-m3 (dense)': 'emb_BAAI_bge-m3_normalized.npz',
    }
    for name, fname in emb.items():
        z = np.load(os.path.join(RESULTS, fname))
        qe, pe = z['q'], z['p']
        vals = time_loop(lambda qe=qe, pe=pe: qe @ pe.T)
        timings[name] = statistics.median(vals) / n
        matrices[name] = qe @ pe.T

    fusion_specs = {
        'RRF (BM25 stems + e5-base)': [matrices['BM25 (morpheme stems)'], matrices['multilingual-e5-base']],
        'RRF (BM25 stems + bge-m3)': [matrices['BM25 (morpheme stems)'], matrices['bge-m3 (dense)']],
        'RRF (BM25 stems + e5-base + bge-m3)': [
            matrices['BM25 (morpheme stems)'], matrices['multilingual-e5-base'], matrices['bge-m3 (dense)']
        ],
    }
    for name, mats in fusion_specs.items():
        vals = time_loop(lambda mats=mats: R.rrf(mats))
        timings[name] = statistics.median(vals) / n

    result = {
        'protocol': {
            'stage': 'scoring/ranking given precomputed query representations',
            'excludes': 'query encoding, model loading, tokenization, and index construction',
            'not_end_to_end_latency': True,
            'queries': n,
            'candidate_passages': n,
            'repeats': REPEATS,
            'aggregation': 'median of repeats divided by the number of queries',
            'machine': {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'python': platform.python_version(),
                'numpy': np.__version__,
                'cpu_count': os.cpu_count(),
            },
            'index_build_seconds': build_seconds,
        },
        'scoring_ms_per_query': {k: float(v) for k, v in timings.items()},
    }
    with open(os.path.join(RESULTS, 'scoring_benchmark.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=1, ensure_ascii=False)

    rows = [{'System': k, 'Scoring time (ms/query)': f"{v:.3f}"} for k, v in timings.items()]
    import pandas as pd
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(TABLES, 'table_6_computational_efficiency.csv'), index=False)
    with open(os.path.join(TABLES, 'table_6_computational_efficiency.md'), 'w', encoding='utf-8') as f:
        f.write(table.to_markdown(index=False) + '\n')
    print(json.dumps(result, indent=1, ensure_ascii=False))
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
