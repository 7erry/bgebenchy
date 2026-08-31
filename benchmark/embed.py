"""Embed catalog text and persist fp16 vectors for local or Voyage models."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from benchmark.generate import ProductRecord, generate_products
from benchmark.schema import QUERY_PREFIX, VECTOR_DIMENSIONS


def _load_dotenv() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _fastembed_model(model_id: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_id)


def _voyage_client():
    _load_dotenv()
    if not os.environ.get("VOYAGE_API_KEY"):
        raise RuntimeError("VOYAGE_API_KEY is not set")
    import voyageai

    return voyageai.Client()


def embed_fastembed_to_path(
    texts: list[str],
    output_path: Path,
    batch_size: int,
    model,
) -> None:
    """Stream local embeddings into a float16 .npy memmap."""
    done_path = output_path.with_suffix(output_path.suffix + ".done")
    if output_path.exists() and done_path.exists():
        existing = np.load(output_path, mmap_mode="r")
        if existing.shape == (len(texts), VECTOR_DIMENSIONS):
            return
        output_path.unlink()
        done_path.unlink()
    elif output_path.exists():
        output_path.unlink()

    memmap = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(texts), VECTOR_DIMENSIONS),
    )
    offset = 0
    for vector in tqdm(
        model.embed(texts, batch_size=batch_size, parallel=1),
        total=len(texts),
        desc=output_path.name,
        unit="vec",
    ):
        array = np.asarray(vector, dtype=np.float32)
        if array.shape[0] != VECTOR_DIMENSIONS:
            raise RuntimeError(
                f"Expected {VECTOR_DIMENSIONS} dimensions, got {array.shape[0]}"
            )
        memmap[offset] = array.astype(np.float16, copy=False)
        offset += 1
    memmap.flush()
    if offset != len(texts):
        raise RuntimeError(f"Embedded {offset} vectors, expected {len(texts)}")
    done_path.write_text(str(offset))


def embed_voyage_to_path(
    texts: list[str],
    output_path: Path,
    batch_size: int,
    model_id: str,
    input_type: str,
) -> None:
    """Call Voyage embeddings in batches and write resumable fp16 vectors."""
    done_path = output_path.with_suffix(output_path.suffix + ".done")
    progress_path = output_path.with_suffix(output_path.suffix + ".progress")
    if output_path.exists() and done_path.exists():
        existing = np.load(output_path, mmap_mode="r")
        if existing.shape == (len(texts), VECTOR_DIMENSIONS):
            return
        output_path.unlink()
        done_path.unlink()
        if progress_path.exists():
            progress_path.unlink()

    start = 0
    if output_path.exists() and progress_path.exists():
        start = int(progress_path.read_text().strip() or "0")
        memmap = np.lib.format.open_memmap(output_path, mode="r+")
    else:
        if output_path.exists():
            output_path.unlink()
        memmap = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(texts), VECTOR_DIMENSIONS),
        )

    client = _voyage_client()
    import voyageai.error

    min_interval_s = float(os.environ.get("VOYAGE_MIN_INTERVAL_S", "21"))
    last_request = 0.0
    for offset in tqdm(
        range(start, len(texts), batch_size),
        desc=f"{output_path.name} ({input_type})",
        unit="batch",
    ):
        batch = texts[offset : offset + batch_size]
        result = None
        for attempt in range(12):
            wait = last_request + min_interval_s - time.time()
            if wait > 0:
                time.sleep(wait)
            try:
                result = client.embed(
                    batch,
                    model=model_id,
                    input_type=input_type,
                    output_dimension=VECTOR_DIMENSIONS,
                    truncation=True,
                )
                last_request = time.time()
                break
            except voyageai.error.RateLimitError:
                last_request = time.time()
                time.sleep(min_interval_s * (1.5 if attempt else 1.0))
        if result is None:
            raise RuntimeError("Voyage embeddings failed after repeated rate limits")
        for index, vector in enumerate(result.embeddings):
            array = np.asarray(vector, dtype=np.float32)
            if array.shape[0] != VECTOR_DIMENSIONS:
                raise RuntimeError(
                    f"Expected {VECTOR_DIMENSIONS} dimensions, got {array.shape[0]}"
                )
            memmap[offset + index] = array.astype(np.float16, copy=False)
        written = min(offset + len(batch), len(texts))
        progress_path.write_text(str(written))
        memmap.flush()
    done_path.write_text(str(len(texts)))
    if progress_path.exists():
        progress_path.unlink()


def build_corpus(
    output_dir: Path,
    doc_count: int,
    query_count: int,
    batch_size: int,
    seed: int,
    provider: str,
    model_id: str,
) -> dict[str, Path]:
    """Generate documents, embed them, and write npz artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    products: list[ProductRecord] = list(generate_products(doc_count, seed=seed))
    doc_texts = [product.embed_text for product in products]
    query_indexes = list(range(0, doc_count, max(1, doc_count // query_count)))[:query_count]
    raw_queries = [products[index].query_text for index in query_indexes]

    doc_path = output_dir / "doc_vectors.fp16.npy"
    query_path = output_dir / "query_vectors.fp16.npy"
    meta_path = output_dir / "meta.npz"

    if provider == "voyage":
        embed_voyage_to_path(doc_texts, doc_path, batch_size, model_id, "document")
        embed_voyage_to_path(raw_queries, query_path, batch_size, model_id, "query")
    elif provider == "fastembed":
        need_model = not (
            doc_path.exists()
            and query_path.exists()
            and doc_path.with_suffix(doc_path.suffix + ".done").exists()
            and query_path.with_suffix(query_path.suffix + ".done").exists()
        )
        if need_model:
            model = _fastembed_model(model_id)
            embed_fastembed_to_path(doc_texts, doc_path, batch_size, model)
            prefixed = [QUERY_PREFIX + text for text in raw_queries]
            embed_fastembed_to_path(prefixed, query_path, batch_size, model)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

    np.savez_compressed(
        meta_path,
        doc_ids=np.array([product.doc_id for product in products]),
        query_indexes=np.array(query_indexes, dtype=np.int32),
        query_texts=np.array(raw_queries),
        vendor=np.array([product.filters["data.vendorName"] for product in products]),
        product_type=np.array(
            [product.filters["data.attributes.product_type_LOV"] for product in products]
        ),
        source=np.array([product.filters["meta.source"] for product in products]),
        web_description=np.array([product.web_description for product in products]),
    )
    return {"docs": doc_path, "queries": query_path, "meta": meta_path}
