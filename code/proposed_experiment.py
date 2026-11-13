"""
KurdRAG Stage 1 (proposed method): contrastively fine-tuned dense retriever + lexical BM25 + zero-shot dense,
combined by z-score weighted score fusion. All hyper-parameters are chosen on a validation split that is carved
out of the TRAINING part of each outer fold; test articles are never used for any choice.

Protocol (per training seed and outer partition):
  outer split  : 3 folds over the 492 articles, using two independently seeded partitions. For fold f, test = fold f,
                 train = the other two folds. Queries of the test fold are ranked among the 164 passages of that fold.
  inner split  : train is split 75/25 into inner-train / validation (split seed = training seed).
  validation   : each validation query ranks among its 82 validation passages plus its 82 highest-BM25 passages from
                 inner-train, matching the 164-passage test candidate-set size.
  selection    : for each (learning rate, text mode in {raw, clitic-split}) fine-tune on inner-train and evaluate on
                 validation after 2/4/6/8 epochs; for each checkpoint grid-search the fusion weights on validation.
                 The configuration with the highest validation MRR@10 is selected.
  final model  : retrain with the selected (lr, mode, epochs) on the full training part, apply the selected fusion
                 weights, save the selected checkpoint, and evaluate on the test fold.
  control      : the same weight search on the zero-shot components (no fine-tuning), so the gain from fine-tuning can
                 be separated from the gain from fusion.

Outputs: results/proposed_results.json, results/proposed_per_query_rr.csv, results/proposed_selection_log.json
"""
import argparse, os, sys, json, time, math, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import real_experiments as R
import torch
from transformers import AutoTokenizer, AutoModel

BASE, OUT = R.BASE, R.OUT
MODEL_ID = 'intfloat/multilingual-e5-base'
QP, PP = 'query: ', 'passage: '
OUTER_PARTITION_SEEDS, N_FOLDS = [42, 2025], 3
TRAIN_SEEDS = [1, 2, 3, 4, 5]
LRS = [1e-5, 2e-5, 4e-5]
MODES = ['raw', 'split']
CKPT_EPOCHS = [2, 4, 6, 8]
BS, TAU = 8, 0.05
MAX_Q, MAX_P = 64, 512
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
GRID = [(a / 10, b / 10, round(1 - a / 10 - b / 10, 1)) for a in range(11) for b in range(11 - a)]  # sums to 1


def split_text(text):
    out = []
    for w in R.words(text):
        out.extend(t.lstrip('#') for t in R.TOK.segment_word(w))
    return ' '.join(out)


def mean_pool(h, mask):
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(1) / m.sum(1)


def encode(model, tok, texts, max_len, bs=16):
    model.eval()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            normalized = [R.TOK.normalize_orthography(text) for text in texts[i:i + bs]]
            enc = tok(normalized, padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(DEV)
            v = mean_pool(model(**enc).last_hidden_state, enc['attention_mask'])
            vecs.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
    return np.concatenate(vecs)


def zscore(m):
    return (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-9)


def rank_of_gold(mat, gold_positions):
    correct = mat[np.arange(len(gold_positions)), gold_positions][:, None]
    return 1 + (mat > correct).sum(1) + (mat == correct).sum(1) - 1


def mrr10(mat):
    r = rank_of_gold(mat, np.arange(mat.shape[0]))
    return float(np.where(r <= 10, 1.0 / r, 0).mean() * 100)


def mrr10_for_candidates(mat, gold_positions):
    r = rank_of_gold(mat, gold_positions)
    return float(np.where(r <= 10, 1.0 / r, 0).mean() * 100)


def validation_candidate_indices(queries, passages, validation_indices, hard_pool_indices,
                                 distractor_count=None):
    validation_indices = np.asarray(validation_indices, dtype=int)
    hard_pool_indices = np.asarray(hard_pool_indices, dtype=int)
    if distractor_count is None:
        distractor_count = len(validation_indices)
    if distractor_count < 1 or distractor_count > len(hard_pool_indices):
        raise ValueError("The hard-negative pool must contain the requested distractor count")
    if np.intersect1d(validation_indices, hard_pool_indices).size:
        raise ValueError("Validation passages cannot also be hard distractors")

    bm25 = R.BM25([R.words(passages[index]) for index in hard_pool_indices])
    candidates = np.empty((len(validation_indices), len(validation_indices) + distractor_count),
                          dtype=int)
    candidates[:, :len(validation_indices)] = validation_indices
    for row, query_index in enumerate(validation_indices):
        scores = bm25.scores(R.words(queries[query_index]))
        hard = np.argsort(-scores, kind="stable")[:distractor_count]
        candidates[row, len(validation_indices):] = hard_pool_indices[hard]
    return candidates


def score_candidates(query_embeddings, passage_embeddings, candidate_indices):
    candidate_vectors = passage_embeddings[np.asarray(candidate_indices, dtype=int)]
    return np.einsum("qd,qcd->qc", query_embeddings, candidate_vectors)


def score_candidate_texts(model, tok, queries, passages, query_indices, candidate_indices):
    query_vectors = encode(
        model, tok, [QP + queries[index] for index in query_indices], MAX_Q, bs=32)
    unique_passages, inverse = np.unique(candidate_indices, return_inverse=True)
    passage_vectors = encode(
        model, tok, [PP + passages[index] for index in unique_passages], MAX_P, bs=16)
    passage_vectors = passage_vectors[inverse.reshape(candidate_indices.shape)]
    return np.einsum("qd,qcd->qc", query_vectors, passage_vectors)


def candidate_matrix(tokfn, queries, passages, query_indices, candidate_indices):
    indexes = {}
    rows = []
    for query_index, passage_indices in zip(query_indices, candidate_indices):
        key = tuple(int(index) for index in passage_indices)
        if key not in indexes:
            indexes[key] = R.BM25([tokfn(passages[index]) for index in passage_indices])
        rows.append(indexes[key].scores(tokfn(queries[query_index])))
    return np.stack(rows)


def best_weights(mats, gold_positions):
    """Grid search of fusion weights (non-negative, sum 1) over z-scored score matrices; returns (weights, val MRR@10)."""
    zs = [zscore(m) for m in mats]
    best, bw = -1, None
    for w in GRID:
        v = mrr10_for_candidates(sum(wi * z for wi, z in zip(w, zs)), gold_positions)
        if v > best + 1e-9:
            best, bw = v, w
    return bw, best


def fuse(w, mats):
    return sum(wi * zscore(m) for wi, m in zip(w, mats))


def load_resume_log(path):
    with open(path, encoding='utf-8') as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Resume log must contain a list: {path}")

    completed = {}
    normalized_records = []
    for record in records:
        if 'FINAL_selected' not in record:
            normalized_records.append(record)
            continue
        try:
            key = (int(record['partition_seed']), int(record['seed']), int(record['fold']))
            selected = record['FINAL_selected']
            if not all(name in selected for name in ('val', 'mode', 'lr', 'epochs', 'w')):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Invalid completed run entry in resume log: {path}") from None
        if key in completed:
            if completed[key] != record:
                raise ValueError(f"Conflicting duplicate completed run in resume log: {key}")
            continue
        completed[key] = record
        normalized_records.append(record)
    return normalized_records, completed


def save_json_atomic(path, data):
    temp_path = path + '.tmp'
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)
    os.replace(temp_path, path)


def train_checkpoints(queries, passages, hard_idx, lr, seed, epochs_to_eval, eval_fn, log,
                      checkpoint_dir=None):
    """Fine-tune with InfoNCE (Eq. 4); constant LR after 10% warm-up of 8 epochs so checkpoints are comparable."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(DEV)
    model.gradient_checkpointing_enable()
    n = len(queries)
    per_epoch = math.ceil(n / BS)
    total_warm = max(1, int(0.1 * 8 * per_epoch))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler()
    step, outs = 0, {}
    for ep in range(1, max(epochs_to_eval) + 1):
        model.train()
        order = rng.permutation(n)
        for i in range(0, n, BS):
            idx = order[i:i + BS]
            for g in opt.param_groups:
                g['lr'] = lr * min(1.0, (step + 1) / total_warm)
            qtxt = [R.TOK.normalize_orthography(QP + queries[j]) for j in idx]
            q = tok(qtxt, padding=True, truncation=True, max_length=MAX_Q, return_tensors='pt').to(DEV)
            ptxt = [PP + passages[j] for j in idx] + [PP + passages[hard_idx[j]] for j in idx]
            ptxt = [R.TOK.normalize_orthography(text) for text in ptxt]
            p = tok(ptxt, padding=True, truncation=True, max_length=MAX_P, return_tensors='pt').to(DEV)
            with torch.autocast('cuda', dtype=torch.float16, enabled=DEV == 'cuda'):
                qv = mean_pool(model(**q).last_hidden_state, q['attention_mask'])
                pv = mean_pool(model(**p).last_hidden_state, p['attention_mask'])
            qv = torch.nn.functional.normalize(qv.float(), dim=-1)
            pv = torch.nn.functional.normalize(pv.float(), dim=-1)
            loss = torch.nn.functional.cross_entropy(qv @ pv.T / TAU, torch.arange(len(idx), device=DEV))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            step += 1
        if ep in epochs_to_eval:
            outs[ep] = eval_fn(model, tok)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir, safe_serialization=True)
                tok.save_pretrained(checkpoint_dir)
            log(f'    lr {lr:g} epoch {ep} loss {loss.item():.4f}')
    del model, opt
    torch.cuda.empty_cache()
    return outs


def main(resume=False):
    log = lambda s: print(s, flush=True)
    docs, stats = R.load_data()
    n = len(docs)
    queries = [R.TOK.normalize_orthography(d['headline']) for d in docs]
    passages = [R.TOK.normalize_orthography(d['body']) for d in docs]
    qs_split = [split_text(x) for x in queries]
    ps_split = [split_text(x) for x in passages]
    zs = {}
    for name, (mid, qp, pp, pool) in R.DENSE_MODELS.items():
        cache = os.path.join(OUT, f"emb_{mid.replace('/', '_')}_normalized.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            qe, pe = z['q'], z['p']
        else:
            log(f"Encoding normalized inputs with {mid}")
            qe = R.embed(mid, [qp + q for q in queries], pool, bs=32, max_len=MAX_Q)
            pe = R.embed(mid, [pp + p for p in passages], pool, bs=8, max_len=MAX_P)
            np.savez_compressed(cache, q=qe, p=pe)
        zs[name] = (qe, pe)
    e5, bge = 'multilingual-e5-base', 'bge-m3 (dense)'

    def hard_negs(tr):
        bm = R.BM25([R.words(passages[i]) for i in tr])
        out = []
        for j, i in enumerate(tr):
            s = bm.scores(R.words(queries[i])); s[j] = -1e9
            out.append(int(np.argmax(s)))
        return out

    sysnames = ['BM25 (word)', 'BM25 (morpheme stems)', 'BM25 (char 4-gram)', 'multilingual-e5-base zero-shot', 'bge-m3 zero-shot',
                'Hybrid RRF (BM25 char 4-gram + bge-m3)', 'Fusion of zero-shot components (tuned weights)',
                'e5-base fine-tuned (selected config)', 'Proposed: KurdRAG Stage 1 (fine-tuned dense + lexical fusion)']
    rr = {s: np.zeros((len(OUTER_PARTITION_SEEDS), len(TRAIN_SEEDS), n)) for s in sysnames}
    selection_log_path = os.path.join(OUT, 'proposed_selection_log.json')
    if resume:
        if not os.path.exists(selection_log_path):
            raise FileNotFoundError(
                f"Cannot resume: selection log does not exist: {selection_log_path}")
        sel_log, completed_runs = load_resume_log(selection_log_path)
        expected_runs = {
            (outer_seed, seed, fold)
            for outer_seed in OUTER_PARTITION_SEEDS
            for seed in TRAIN_SEEDS
            for fold in range(N_FOLDS)
        }
        unexpected = set(completed_runs) - expected_runs
        if unexpected:
            raise ValueError(f"Resume log contains runs outside the current protocol: {sorted(unexpected)}")
        save_json_atomic(selection_log_path, sel_log)
        log(f"Resuming from {len(completed_runs)} of {len(expected_runs)} completed fold runs")
    else:
        sel_log, completed_runs = [], {}

    def rr_of(mat):
        r = rank_of_gold(mat, np.arange(mat.shape[0]))
        return np.where(r <= 10, 1.0 / r, 0)

    for pi, outer_seed in enumerate(OUTER_PARTITION_SEEDS):
        perm = np.random.default_rng(outer_seed).permutation(n)
        folds = [np.sort(perm[f::N_FOLDS]) for f in range(N_FOLDS)]
        for si, seed in enumerate(TRAIN_SEEDS):
            for fi, test in enumerate(folds):
                t0 = time.time()
                train = np.sort(np.concatenate([folds[j] for j in range(N_FOLDS) if j != fi]))
                r = np.random.default_rng(seed).permutation(len(train))
                nval = len(train) // 4
                val, itr = np.sort(train[r[:nval]]), np.sort(train[r[nval:]])
                val_candidates = validation_candidate_indices(queries, passages, val, itr)
                val_gold = np.arange(len(val))
                test_candidates = np.broadcast_to(test, (len(test), len(test)))
                log(f'=== partition {outer_seed} seed {seed} fold {fi}: inner-train {len(itr)} / '
                    f'val {len(val)} / validation candidates {val_candidates.shape[1]} / '
                    f'outer-train {len(train)} / test {len(test)}')

                comp = {}
                for tag, ix, candidates in (('val', val, val_candidates),
                                            ('test', test, test_candidates)):
                    comp[tag] = {
                        'c4': candidate_matrix(R.char_ngrams, queries, passages, ix, candidates),
                        'word': candidate_matrix(R.words, queries, passages, ix, candidates),
                        'stem': candidate_matrix(R.morph, queries, passages, ix, candidates),
                        'e5zs': score_candidates(zs[e5][0][ix], zs[e5][1], candidates),
                        'bgezs': score_candidates(zs[bge][0][ix], zs[bge][1], candidates),
                    }

                w_zs, v_zs = best_weights(
                    [comp['val']['c4'], comp['val']['e5zs'], comp['val']['bgezs']], val_gold)
                ft_zs = fuse(w_zs, [comp['test']['c4'], comp['test']['e5zs'], comp['test']['bgezs']])

                run_key = (outer_seed, seed, fi)
                previous = completed_runs.get(run_key)
                if previous:
                    best = previous['FINAL_selected']
                    log(f"  reusing validation selection from completed fold "
                        f"(mode={best['mode']} lr={best['lr']:g} epochs={best['epochs']})")
                else:
                    best = {'val': -1}
                    for mode in MODES:
                        qsrc, psrc = (queries, passages) if mode == 'raw' else (qs_split, ps_split)
                        hard = hard_negs(itr)

                        def eval_val(model, tok, qsrc=qsrc, psrc=psrc):
                            return score_candidate_texts(
                                model, tok, qsrc, psrc, val, val_candidates)

                        for lr in LRS:
                            outs = train_checkpoints(
                                [qsrc[i] for i in itr], [psrc[i] for i in itr],
                                hard, lr, seed, CKPT_EPOCHS, eval_val, log)
                            for ep, dv in outs.items():
                                w, v = best_weights([comp['val']['c4'], dv, comp['val']['bgezs']], val_gold)
                                sel_log.append({
                                    'partition_seed': outer_seed, 'seed': seed, 'fold': fi,
                                    'mode': mode, 'lr': lr, 'epochs': ep,
                                    'val_dense_MRR@10': mrr10_for_candidates(dv, val_gold),
                                    'val_fused_MRR@10': v, 'weights_c4_ft_bge': w})
                                if v > best['val'] + 1e-9:
                                    best = {'val': v, 'mode': mode, 'lr': lr, 'epochs': ep, 'w': w}

                log(f"  selected: mode={best['mode']} lr={best['lr']:g} epochs={best['epochs']} "
                    f"weights(c4,ft,bge)={best['w']} val MRR@10={best['val']:.2f}")
                mode = best['mode']
                qsrc, psrc = (queries, passages) if mode == 'raw' else (qs_split, ps_split)

                def eval_test(model, tok):
                    return score_candidate_texts(
                        model, tok, qsrc, psrc, test, test_candidates)

                checkpoint_dir = os.path.join(
                    OUT, "checkpoints", f"partition_{outer_seed}", f"fold_{fi}", f"seed_{seed}")
                if previous:
                    with open(os.path.join(checkpoint_dir, "run_metadata.json"), encoding="utf-8") as f:
                        metadata = json.load(f)
                    expected_metadata = {
                        "partition_seed": outer_seed, "training_seed": seed, "fold": fi,
                        "selected_mode": mode, "learning_rate": best["lr"],
                        "epochs": best["epochs"], "fusion_weights": best["w"],
                    }
                    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
                        raise ValueError(f"Checkpoint metadata does not match resume log: {checkpoint_dir}")
                    if not os.path.exists(os.path.join(checkpoint_dir, "config.json")):
                        raise FileNotFoundError(f"Missing model checkpoint: {checkpoint_dir}")
                    tok = AutoTokenizer.from_pretrained(checkpoint_dir)
                    model = AutoModel.from_pretrained(checkpoint_dir).to(DEV)
                    dt = eval_test(model, tok)
                    del model
                else:
                    outs = train_checkpoints(
                        [qsrc[i] for i in train], [psrc[i] for i in train],
                        hard_negs(train), best['lr'], seed, [best['epochs']],
                        eval_test, lambda message: None, checkpoint_dir=checkpoint_dir)
                    dt = outs[best['epochs']]
                    with open(os.path.join(checkpoint_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
                        json.dump({
                            "partition_seed": outer_seed, "training_seed": seed, "fold": fi,
                            "selected_mode": mode, "learning_rate": best["lr"],
                            "epochs": best["epochs"], "fusion_weights": best["w"],
                            "validation_candidate_count": int(val_candidates.shape[1]),
                            "test_candidate_count": int(len(test)),
                        }, f, indent=2)
                prop = fuse(best['w'], [comp['test']['c4'], dt, comp['test']['bgezs']])

                mats = {
                    'BM25 (word)': comp['test']['word'],
                    'BM25 (morpheme stems)': comp['test']['stem'],
                    'BM25 (char 4-gram)': comp['test']['c4'],
                    'multilingual-e5-base zero-shot': comp['test']['e5zs'],
                    'bge-m3 zero-shot': comp['test']['bgezs'],
                    'Hybrid RRF (BM25 char 4-gram + bge-m3)': R.rrf(
                        [comp['test']['c4'], comp['test']['bgezs']]),
                    'Fusion of zero-shot components (tuned weights)': ft_zs,
                    'e5-base fine-tuned (selected config)': dt,
                    'Proposed: KurdRAG Stage 1 (fine-tuned dense + lexical fusion)': prop,
                }
                for system, matrix in mats.items():
                    rr[system][pi, si, test] = rr_of(matrix)
                test_rr = {system: float(rr[system][pi, si, test].mean() * 100)
                           for system in sysnames}
                sel_log.append({
                    'partition_seed': outer_seed, 'seed': seed, 'fold': fi,
                    'FINAL_selected': {
                        key: (list(value) if isinstance(value, tuple) else value)
                        for key, value in best.items()},
                    'zero_shot_weights_c4_e5_bge': w_zs,
                    'zero_shot_val_MRR@10': v_zs,
                    'test_MRR@10': test_rr,
                    'checkpoint_dir': os.path.relpath(checkpoint_dir, BASE),
                })
                log(f"  test MRR@10 proposed {test_rr[sysnames[-1]]:.2f} | "
                    f"BM25 c4 {test_rr['BM25 (char 4-gram)']:.2f} | "
                    f"zero-shot fusion {test_rr['Fusion of zero-shot components (tuned weights)']:.2f} "
                    f"({time.time() - t0:.0f}s)")
                save_json_atomic(selection_log_path, sel_log)

    # ------------------------------------------------------------ aggregate
    rng = np.random.default_rng(R.SEED)
    res = {}
    mean_rr = {s: rr[s].mean((0, 1)) for s in sysnames}
    for s in sysnames:
        per_seed = rr[s].mean((0, 2)) * 100
        boots = [mean_rr[s][rng.integers(0, n, n)].mean() * 100 for _ in range(R.N_BOOT)]
        res[s] = {'MRR@10': float(mean_rr[s].mean() * 100),
                  'MRR@10_CI95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                  'MRR@10_per_training_seed': [float(x) for x in per_seed],
                  'MRR@10_seed_std': float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0,
                  'Recall@1': float((rr[s] == 1.0).mean() * 100)}

    def paired(a, b):
        d = mean_rr[a] - mean_rr[b]
        boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(R.N_BOOT)])
        p = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
        return {'mean_MRR@10_diff': float(d.mean() * 100),
                'CI95': [float(np.percentile(boots, 2.5) * 100), float(np.percentile(boots, 97.5) * 100)],
                'p_two_sided': max(p, 1.0 / R.N_BOOT)}
    P = sysnames[-1]
    comps = {f'Proposed vs {s}': paired(P, s) for s in sysnames[:-1]}
    cfg = {'model': MODEL_ID, 'outer_folds': N_FOLDS,
           'outer_partition_seeds': OUTER_PARTITION_SEEDS,
           'training_seeds': TRAIN_SEEDS, 'lr_grid': LRS,
           'modes': MODES, 'epoch_checkpoints': CKPT_EPOCHS, 'batch_size': BS, 'temperature': TAU, 'max_len_query': MAX_Q,
           'max_len_passage': MAX_P, 'fusion': 'z-score weighted sum, weights on a 0.1 grid summing to 1',
           'validation_candidates': 'validation passages plus BM25 hard distractors from inner-train',
           'validation_candidate_count': 164,
           'dense_normalization': 'MorphoBPETokenizer.normalize_orthography',
           'device': DEV, 'torch': torch.__version__,
           'test_passages_per_fold': [n // N_FOLDS] * (N_FOLDS - 1) + [n - (n // N_FOLDS) * (N_FOLDS - 1)]}
    json.dump({'config': cfg, 'systems': res, 'paired_comparisons': comps},
              open(os.path.join(OUT, 'proposed_results.json'), 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, 'proposed_per_query_rr.csv'), 'w', encoding='utf-8') as f:
        f.write('query_index,' + ','.join('"%s"' % s for s in sysnames) + '\n')
        for i in range(n):
            f.write(f'{i},' + ','.join(f'{mean_rr[s][i]:.6f}' for s in sysnames) + '\n')
    for s in sysnames:
        log(f"{s:70s} MRR@10 {res[s]['MRR@10']:.2f} (seed sd {res[s]['MRR@10_seed_std']:.2f})")
    log(json.dumps(comps, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the proposed retriever experiment.")
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse completed folds in results/proposed_selection_log.json and their checkpoints",
    )
    main(resume=parser.parse_args().resume)
