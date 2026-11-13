"""Evaluate lexical retrieval and stemmers on the Pewan Sorani test collection."""
import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from array import array
from collections import Counter
from pathlib import Path

import numpy as np

import real_experiments as R


SYSTEMS = ("BM25 word", "BM25 char-4gram", "BM25 splitter", "BM25 Jedar", "BM25 KLPT")
BOOTSTRAPS = 2000


def load_topics(pewan_root):
    query_text = (pewan_root / "Queries" / "in-sorani.txt").read_text(encoding="utf-8-sig")
    topics = {
        int(qid): title.strip()
        for qid, title in re.findall(
            r"<qid>\s*(\d+)\s*</qid>\s*<title>\s*(.*?)\s*</title>",
            query_text,
            flags=re.DOTALL,
        )
    }
    qrels = {}
    for line in (pewan_root / "Qrels" / "sorani-qrel.txt").read_text(encoding="utf-8-sig").splitlines():
        fields = [field.strip() for field in line.split(",") if field.strip()]
        if len(fields) < 2:
            raise ValueError(f"Malformed Pewan qrel line: {line!r}")
        qrels[int(fields[0])] = set(fields[1:])
    if not topics or topics.keys() != qrels.keys():
        raise ValueError("Pewan topics and qrels have different query IDs")
    return topics, qrels


def build_jedar_stems(words, source_path, suffix_path):
    with tempfile.TemporaryDirectory(prefix="jedar_pewan_") as folder:
        work = Path(folder)
        shutil.copy2(source_path, work / "Jedar.java")
        shutil.copy2(Path(__file__).with_name("JedarBatchRunner.java"), work / "JedarBatchRunner.java")
        shutil.copy2(suffix_path, work / "suffixes.txt")
        subprocess.run(
            ["javac", "-encoding", "UTF-8", "Jedar.java", "JedarBatchRunner.java"],
            cwd=work,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        input_text = "\n".join(words) + "\n"
        result = subprocess.run(
            ["java", "-cp", str(work), "JedarBatchRunner"],
            cwd=work,
            input=input_text,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    rows = result.stdout.splitlines()
    if len(rows) != len(words):
        raise ValueError("Jedar returned an incomplete stem list")
    stems = {}
    empty_count = 0
    for word, row in zip(words, rows):
        status, separator, stem = row.partition("\t")
        if not separator or status not in {"0", "1"} or not stem:
            raise ValueError("Jedar returned a malformed stem record")
        if status == "1":
            empty_count += 1
        stems[word] = stem
    return stems, empty_count


def make_index(doc_paths, query_terms, tokenizers):
    n_docs = len(doc_paths)
    lengths = {name: np.zeros(n_docs, dtype=np.int32) for name in tokenizers}
    postings = {
        name: {term: (array("I"), array("H")) for term in query_terms[name]}
        for name in tokenizers
    }
    for doc_i, path in enumerate(doc_paths):
        text = path.read_text(encoding="utf-8")
        words = R.words(text)
        token_lists = {
            "BM25 word": words,
            "BM25 char-4gram": R.char_ngrams(text),
            "BM25 splitter": [R.TOK.segment_word(word)[0] for word in words],
            "BM25 Jedar": [tokenizers["BM25 Jedar"](word) for word in words],
            "BM25 KLPT": [tokenizers["BM25 KLPT"](word) for word in words],
        }
        for name, tokens in token_lists.items():
            lengths[name][doc_i] = len(tokens)
            counts = Counter(token for token in tokens if token in postings[name])
            for term, tf in counts.items():
                doc_ids, frequencies = postings[name][term]
                doc_ids.append(doc_i)
                frequencies.append(tf)
        if (doc_i + 1) % 20000 == 0:
            print(f"Indexed {doc_i + 1:,}/{n_docs:,} Pewan documents", flush=True)

    indices = {}
    for name in tokenizers:
        doc_len = lengths[name]
        indices[name] = {
            "postings": postings[name],
            "doc_len": doc_len,
            "avgdl": float(doc_len.mean()),
        }
    return indices


def score_query(index, terms, k1=1.5, b=0.75):
    n_docs = len(index["doc_len"])
    scores = np.zeros(n_docs, dtype=np.float64)
    for term in set(terms):
        posting = index["postings"].get(term)
        if posting is None:
            continue
        doc_ids = np.frombuffer(posting[0], dtype=np.uint32)
        tf = np.frombuffer(posting[1], dtype=np.uint16)
        df = len(doc_ids)
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        norm = k1 * (1 - b + b * index["doc_len"][doc_ids] / index["avgdl"])
        scores[doc_ids] += idf * tf * (k1 + 1) / (tf + norm)
    return scores


def query_metrics(scores, relevant_indices):
    order = np.argsort(-scores, kind="stable")
    rank_by_document = np.empty(len(scores), dtype=np.int32)
    rank_by_document[order] = np.arange(1, len(scores) + 1)
    ranks = np.sort(rank_by_document[relevant_indices])
    n_rel = len(relevant_indices)
    first = int(ranks.min())
    values = {}
    for k in (1, 5, 10, 20):
        values[f"Recall@{k}"] = float(np.count_nonzero(ranks <= k) / n_rel * 100)
    values["P@10"] = float(np.count_nonzero(ranks <= 10) / 10 * 100)
    values["Recall@100"] = float(np.count_nonzero(ranks <= 100) / n_rel * 100)
    values["MRR@10"] = 100 / first if first <= 10 else 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank in ranks if rank <= 10)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, n_rel) + 1))
    values["nDCG@10"] = dcg / idcg * 100
    values["MAP"] = float(
        np.sum(np.arange(1, n_rel + 1) / ranks) / n_rel * 100
    )
    return values


def bootstrap_ci(values, rng):
    n = len(values)
    samples = np.asarray([
        values[rng.integers(0, n, n)].mean()
        for _ in range(BOOTSTRAPS)
    ])
    return [float(x) for x in np.percentile(samples, [2.5, 97.5])]


def write_tex_table(metrics, output_path):
    labels = {
        "BM25 word": "BM25 word",
        "BM25 char-4gram": "BM25 char-4gram",
        "BM25 splitter": "BM25 splitter",
        "BM25 Jedar": "BM25 Jedar*",
        "BM25 KLPT": "BM25 KLPT",
    }
    rows = []
    for system in SYSTEMS:
        for metric in ("P@10", "nDCG@10", "MAP", "Recall@100"):
            ci = metrics[system][f"{metric}_CI95"]
            rows.append(
                f"{labels[system]} & {metric} & "
                f"{metrics[system][metric]:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] \\\\"
            )
    table = "\n".join([
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"System & Metric & Value (95\% CI) \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table, encoding="utf-8")


def write_paired_tex_table(metric_comparisons, output_path):
    rows = []
    for system, label in (
        ("BM25 word", "Char-4gram - word BM25"),
        ("BM25 KLPT", "Char-4gram - KLPT"),
    ):
        comparison = metric_comparisons[f"{system} minus BM25 char-4gram"]
        for metric in ("P@10", "nDCG@10", "MAP", "Recall@100"):
            result = comparison[metric]
            low, high = result["paired_bootstrap_CI95"]
            difference = -result["difference_points"]
            rows.append(
                f"{label} & {metric} & "
                f"{difference:+.2f} [{-high:+.2f}, {-low:+.2f}] \\\\"
            )
    table = "\n".join([
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"Comparison & Metric & Difference (95\% paired CI) \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pewan-root", type=Path, required=True,
                        help="Extracted Pewan directory containing Corpora, Queries, and Qrels")
    parser.add_argument("--jedar-source", type=Path, required=True,
                        help="Unmodified upstream Jedar.java")
    parser.add_argument("--jedar-suffixes", type=Path, required=True,
                        help="Pewan bundle's Others/suffixes.txt; upstream Jedar repository omits this file")
    parser.add_argument("--output", type=Path, default=Path(R.OUT) / "pewan_results.json")
    parser.add_argument("--table-output", type=Path,
                        default=Path(R.BASE) / "tables" / "tex_table_pewan.tex")
    parser.add_argument("--paired-table-output", type=Path,
                        default=Path(R.BASE) / "tables" / "tex_table_pewan_paired.tex")
    args = parser.parse_args()

    from klpt.stem import Stem

    if not args.jedar_source.is_file():
        raise FileNotFoundError(f"Jedar source not found: {args.jedar_source}")
    if not args.jedar_suffixes.is_file():
        raise FileNotFoundError(f"Jedar suffix list not found: {args.jedar_suffixes}")
    if shutil.which("javac") is None:
        raise FileNotFoundError("javac not found; install a JDK to run the Jedar baseline")
    klpt = Stem("Sorani", "Arabic")

    topics, qrels = load_topics(args.pewan_root)
    doc_dir = args.pewan_root / "Corpora" / "Sorani" / "docs"
    if not doc_dir.is_dir():
        raise FileNotFoundError(f"Pewan Sorani document directory not found: {doc_dir}")
    doc_paths = sorted(doc_dir.glob("*.txt"), key=lambda path: int(path.stem))
    if not doc_paths:
        raise ValueError(f"No Pewan Sorani documents found in {doc_dir}")
    doc_ids = [path.stem for path in doc_paths]
    doc_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    missing = {
        doc_id for relevant in qrels.values() for doc_id in relevant
        if doc_id not in doc_index
    }
    if missing:
        raise ValueError(f"{len(missing)} qrel document IDs are missing from the corpus")

    print(f"Collecting vocabulary from {len(doc_paths):,} Sorani documents", flush=True)
    vocab = set()
    for i, path in enumerate(doc_paths):
        vocab.update(R.words(path.read_text(encoding="utf-8")))
        if (i + 1) % 20000 == 0:
            print(f"Read {i + 1:,}/{len(doc_paths):,} Pewan documents", flush=True)
    for topic in topics.values():
        vocab.update(R.words(topic))

    sorted_vocab = sorted(vocab)
    klpt_map = {}
    for i, word in enumerate(sorted_vocab):
        candidates = klpt.stem(word)
        klpt_map[word] = candidates[0] if candidates else word
        if (i + 1) % 50000 == 0:
            print(f"KLPT stemmed {i + 1:,}/{len(sorted_vocab):,} word types", flush=True)
    jedar_map, jedar_empty_count = build_jedar_stems(
        sorted_vocab, args.jedar_source, args.jedar_suffixes
    )
    print(
        f"Jedar returned empty stems for {jedar_empty_count:,} word types; "
        "the original token is used for those cases",
        flush=True,
    )

    tokenizers = {
        "BM25 word": lambda word: word,
        "BM25 char-4gram": lambda word: word,
        "BM25 splitter": lambda word: R.TOK.segment_word(word)[0],
        "BM25 Jedar": jedar_map.__getitem__,
        "BM25 KLPT": klpt_map.__getitem__,
    }
    query_tokens = {}
    for name in tokenizers:
        if name == "BM25 char-4gram":
            query_tokens[name] = {
                qid: R.char_ngrams(text) for qid, text in topics.items()
            }
        elif name == "BM25 Jedar":
            query_tokens[name] = {
                qid: [jedar_map[word] for word in R.words(text)]
                for qid, text in topics.items()
            }
        elif name == "BM25 KLPT":
            query_tokens[name] = {
                qid: [klpt_map[word] for word in R.words(text)]
                for qid, text in topics.items()
            }
        elif name == "BM25 splitter":
            query_tokens[name] = {
                qid: [R.TOK.segment_word(word)[0] for word in R.words(text)]
                for qid, text in topics.items()
            }
        else:
            query_tokens[name] = {
                qid: R.words(text) for qid, text in topics.items()
            }
    term_sets = {
        name: set(term for terms in queries.values() for term in terms)
        for name, queries in query_tokens.items()
    }

    indices = make_index(doc_paths, term_sets, tokenizers)
    per_query = {name: [] for name in tokenizers}
    for name in tokenizers:
        for qid in topics:
            rel = np.asarray([doc_index[doc_id] for doc_id in qrels[qid]], dtype=np.int32)
            scores = score_query(indices[name], query_tokens[name][qid])
            per_query[name].append({"qid": qid, **query_metrics(scores, rel)})

    rng = np.random.default_rng(42)
    metrics = {}
    for name, rows in per_query.items():
        metrics[name] = {}
        for metric in (
            "Recall@1", "Recall@5", "Recall@10", "Recall@20",
            "P@10", "Recall@100", "MRR@10", "nDCG@10", "MAP",
        ):
            values = np.asarray([row[metric] for row in rows])
            metrics[name][metric] = float(values.mean())
            metrics[name][f"{metric}_CI95"] = bootstrap_ci(values, rng)

    comparisons = {}
    metric_comparisons = {}
    for name in ("BM25 word", "BM25 splitter", "BM25 Jedar", "BM25 KLPT"):
        key = f"{name} minus BM25 char-4gram"
        base_mrr = np.asarray([
            row["MRR@10"] for row in per_query["BM25 char-4gram"]
        ])
        other_mrr = np.asarray([row["MRR@10"] for row in per_query[name]])
        difference = other_mrr - base_mrr
        comparisons[key] = {
            "MRR@10_difference_points": float(difference.mean()),
            "paired_bootstrap_CI95": bootstrap_ci(difference, rng),
        }
        metric_comparisons[key] = {}
        for metric in ("P@10", "nDCG@10", "MAP", "Recall@100"):
            base = np.asarray([
                row[metric] for row in per_query["BM25 char-4gram"]
            ])
            other = np.asarray([row[metric] for row in per_query[name]])
            metric_difference = other - base
            metric_comparisons[key][metric] = {
                "difference_points": float(metric_difference.mean()),
                "paired_bootstrap_CI95": bootstrap_ci(metric_difference, rng),
            }
    splitter = np.asarray([row["MRR@10"] for row in per_query["BM25 splitter"]])
    stemmer_comparisons = {}
    stemmer_rng = np.random.default_rng(42)
    for name in ("BM25 Jedar", "BM25 KLPT"):
        other = np.asarray([row["MRR@10"] for row in per_query[name]])
        difference = other - splitter
        stemmer_comparisons[f"{name} minus BM25 splitter"] = {
            "MRR@10_difference_points": float(difference.mean()),
            "paired_bootstrap_CI95": bootstrap_ci(difference, stemmer_rng),
        }

    suffix_sha = hashlib.sha256(args.jedar_suffixes.read_bytes()).hexdigest()
    jedar_sha = hashlib.sha256(args.jedar_source.read_bytes()).hexdigest()
    output = {
        "provenance": {
            "collection": "Pewan Sorani",
            "source": "https://github.com/klpp/pewan",
            "documents": len(doc_paths),
            "queries": len(topics),
            "positive_qrel_judgments": sum(len(ids) for ids in qrels.values()),
            "unique_relevant_documents": len({doc_id for ids in qrels.values() for doc_id in ids}),
            "queries_use": "Sorani topic titles only",
            "qrels_use": "binary relevance IDs supplied by Pewan; unjudged documents are treated as nonrelevant",
            "license": "No license declared in the public GitHub repository metadata; do not redistribute the collection.",
            "bootstrap": {"unit": "query", "resamples": BOOTSTRAPS, "seed": 42},
        },
        "settings": {
            "bm25_k1": 1.5,
            "bm25_b": 0.75,
            "pessimistic_tie_handling": True,
            "Jedar_source": "upstream Jedar.java with Pewan archive Others/suffixes.txt",
            "Jedar_repository": "https://github.com/klpp/codes/tree/master/stemming",
            "Jedar_source_sha256": jedar_sha,
            "Jedar_suffix_sha256": suffix_sha,
            "Jedar_caveat": "The upstream Jedar repository does not include its expected suffixes.txt; exact equivalence to the missing author file is unverified.",
            "Jedar_empty_stem_fallbacks": jedar_empty_count,
            "KLPT_version": "0.1.7",
            "KLPT_multi_stem_policy": "first stem returned by KLPT; original token if no stem returned",
        },
        "systems": metrics,
        "paired_comparisons_vs_char4": comparisons,
        "paired_metric_comparisons_vs_char4": metric_comparisons,
        "paired_stemmer_comparisons_vs_splitter": stemmer_comparisons,
        "per_query_metrics": per_query,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_tex_table(metrics, args.table_output)
    write_paired_tex_table(metric_comparisons, args.paired_table_output)
    print(json.dumps({
        "systems": metrics,
        "comparisons_vs_char4": comparisons,
        "stemmer_comparisons_vs_splitter": stemmer_comparisons,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
