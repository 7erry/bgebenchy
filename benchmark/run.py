"""Run the BGE-large fp16 HNSW recall sweep and open-loop RPS latency test."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from usearch.index import Index

from benchmark.embed import build_corpus
from benchmark.schema import (
    FILTER_FIELDS,
    FULLTEXT_FIELDS,
    MODELS,
    VECTOR_DIMENSIONS,
    atlas_fulltext_index,
    atlas_vector_index,
)

K = 10
EF_SWEEP = (16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 400)
CONNECTIVITY = 16
EXPANSION_ADD = 128


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "mean_ms": float(array.mean()),
        "max_ms": float(array.max()),
    }


def exact_topk(doc_vectors: np.ndarray, query_vectors: np.ndarray, k: int) -> np.ndarray:
    """Brute-force cosine top-k. Vectors are assumed L2-normalized."""
    docs = doc_vectors.astype(np.float32, copy=False)
    queries = query_vectors.astype(np.float32, copy=False)
    scores = queries @ docs.T
    return np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]


def build_index(doc_vectors: np.ndarray) -> Index:
    index = Index(
        ndim=VECTOR_DIMENSIONS,
        metric="cos",
        dtype="f16",
        connectivity=CONNECTIVITY,
        expansion_add=EXPANSION_ADD,
        expansion_search=64,
    )
    keys = np.arange(doc_vectors.shape[0], dtype=np.uint64)
    index.add(keys, doc_vectors)
    return index


def search_topk(index: Index, query_vectors: np.ndarray, k: int, ef: int) -> np.ndarray:
    index.expansion_search = ef
    matches = index.search(query_vectors.astype(np.float16, copy=False), count=k, threads=1)
    result = np.empty((query_vectors.shape[0], k), dtype=np.int64)
    for row in range(query_vectors.shape[0]):
        result[row] = np.asarray(matches[row].keys, dtype=np.int64)
    return result


def recall_at_k(truth: np.ndarray, predicted: np.ndarray) -> float:
    hits = 0
    for truth_row, predicted_row in zip(truth, predicted):
        hits += len(set(truth_row.tolist()) & set(predicted_row.tolist()))
    return hits / float(truth.size)


def sweep_recall(index: Index, query_vectors: np.ndarray, truth: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for ef in EF_SWEEP:
        predicted = search_topk(index, query_vectors, K, ef)
        recall = recall_at_k(truth, predicted)
        rows.append({"ef_search": ef, "num_candidates": ef, "recall_at_10": recall})
        if recall >= 0.95 and ef > K:
            break
    return rows


def measure_rps(
    index: Index,
    query_vectors: np.ndarray,
    ef: int,
    rps: int,
    duration_s: float,
    warmup_s: float,
) -> dict:
    """Open-loop load: fire one search every 1/rps seconds on a single thread."""
    index.expansion_search = ef
    query_count = query_vectors.shape[0]
    interval = 1.0 / rps

    def once(offset: int) -> float:
        query = query_vectors[offset % query_count]
        started = time.perf_counter()
        index.search(query, count=K, threads=1)
        return (time.perf_counter() - started) * 1000.0

    warmup_until = time.perf_counter() + warmup_s
    warmup_i = 0
    while time.perf_counter() < warmup_until:
        once(warmup_i)
        warmup_i += 1

    latencies: list[float] = []
    late = 0
    planned = time.perf_counter()
    end = planned + duration_s
    request_i = 0
    while planned < end:
        now = time.perf_counter()
        delay = planned - now
        if delay > 0:
            time.sleep(delay)
        elif -delay > interval:
            late += 1
        latencies.append(once(request_i))
        request_i += 1
        planned += interval

    achieved = request_i / duration_s
    return {
        "target_rps": rps,
        "achieved_rps": achieved,
        "requests": request_i,
        "late_starts": late,
        "late_start_ratio": late / request_i if request_i else 0.0,
        **_percentiles(latencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--rps", default="50,75,100")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--model", default="bge-large-en-v1.5", choices=sorted(MODELS))
    args = parser.parse_args()

    model = MODELS[args.model]
    paths = build_corpus(
        args.data_dir,
        args.docs,
        args.queries,
        args.batch_size,
        args.seed,
        provider=model["provider"],
        model_id=model["id"],
    )
    doc_vectors = np.load(paths["docs"], mmap_mode="r")
    query_vectors = np.load(paths["queries"])

    print(f"docs={doc_vectors.shape} queries={query_vectors.shape} dtype={doc_vectors.dtype}")
    print("computing exact top-10 ground truth")
    truth = exact_topk(np.asarray(doc_vectors), query_vectors, K)

    print("building fp16 HNSW index")
    built = time.perf_counter()
    index = build_index(np.asarray(doc_vectors))
    build_s = time.perf_counter() - built
    print(f"index size={index.size} memory_bytes={index.memory_usage} build_s={build_s:.2f}")

    recall_rows = sweep_recall(index, query_vectors, truth)
    chosen = next(
        (row for row in recall_rows if row["recall_at_10"] >= args.target_recall),
        recall_rows[-1],
    )
    print(f"recall sweep: {recall_rows}")
    print(f"operating point: {chosen}")

    rps_values = [int(item) for item in args.rps.split(",") if item.strip()]
    latency_rows = [
        measure_rps(index, query_vectors, chosen["ef_search"], rps, args.duration, args.warmup)
        for rps in rps_values
    ]

    args.results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model["id"],
        "model_url": model["url"],
        "provider": model["provider"],
        "documents": args.docs,
        "queries": int(query_vectors.shape[0]),
        "dimensions": VECTOR_DIMENSIONS,
        "vector_dtype": "fp16",
        "metric": "cosine",
        "k": K,
        "target_recall": args.target_recall,
        "indexed_filter_fields": len(FILTER_FIELDS),
        "fulltext_fields": len(FULLTEXT_FIELDS),
        "filter_field_paths": list(FILTER_FIELDS),
        "fulltext_field_paths": list(FULLTEXT_FIELDS),
        "hnsw": {
            "connectivity_m": CONNECTIVITY,
            "ef_construction": EXPANSION_ADD,
            "ef_search": chosen["ef_search"],
            "implementation": "usearch",
            "dtype": "f16",
        },
        "index_build_seconds": build_s,
        "index_memory_bytes": int(index.memory_usage),
        "recall_sweep": recall_rows,
        "operating_point": chosen,
        "latency": latency_rows,
        "atlas_indexes": {
            "vector": atlas_vector_index(f"motion_{model['slug']}_vector"),
            "fulltext": atlas_fulltext_index(f"motion_{model['slug']}_fulltext"),
        },
        "hardware": "Apple M4 Pro, in-process usearch HNSW (not Atlas Search Nodes)",
        "note": (
            "Latency is local in-process ANN search on fp16 vectors. "
            "Atlas M0 cannot host 100k x 1024-d documents; use these numbers as "
            "a model/index baseline, not as Atlas p99."
        ),
    }
    result_path = args.results_dir / f"{model['slug']}_benchmark.json"
    result_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
