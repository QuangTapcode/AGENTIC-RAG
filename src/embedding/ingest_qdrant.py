"""Embed chunk datasets and ingest dense + BM25 sparse vectors into Qdrant.

The dense side uses ``intfloat/multilingual-e5-small`` with the E5 convention:
documents are prefixed with ``passage:`` and future queries with ``query:``.
The sparse side is a self-contained BM25-compatible representation. Document
vectors store IDF-weighted BM25 term weights, so Qdrant's sparse dot product
can be used as the keyword-search score in the next retrieval step.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
BM25_K1 = 1.2
BM25_B = 0.75
TOKEN_RE = re.compile(r"(?u)\b[\w]+(?:[-'][\w]+)*\b")


def read_chunks(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not chunk.get("chunk_id") or not chunk.get("text"):
                raise ValueError(f"Missing chunk_id or text at {path}:{line_number}")
            chunks.append(chunk)
    if not chunks:
        raise ValueError(f"No chunks found in {path}")
    return chunks


def tokenize(text: str) -> list[str]:
    """Tokenize Unicode medical text consistently for BM25 indexing/querying."""

    return [match.group(0).casefold() for match in TOKEN_RE.finditer(text)]


def build_bm25_vectors(
    chunks: list[dict[str, Any]],
) -> tuple[list[models.SparseVector], dict[str, Any]]:
    """Build sparse vectors with IDF-weighted BM25 document term weights."""

    term_counts: list[Counter[str]] = []
    document_frequency: Counter[str] = Counter()
    for chunk in chunks:
        counts = Counter(tokenize(chunk["text"]))
        term_counts.append(counts)
        document_frequency.update(counts.keys())

    document_count = len(chunks)
    vocabulary_terms = sorted(document_frequency)
    vocabulary = {term: index for index, term in enumerate(vocabulary_terms)}
    idf = {
        term: math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in document_frequency.items()
    }
    document_lengths = [sum(counts.values()) for counts in term_counts]
    average_document_length = sum(document_lengths) / max(document_count, 1)

    vectors: list[models.SparseVector] = []
    for counts, document_length in zip(term_counts, document_lengths):
        indices: list[int] = []
        values: list[float] = []
        length_ratio = document_length / max(average_document_length, 1e-12)
        for term in sorted(counts):
            frequency = counts[term]
            denominator = frequency + BM25_K1 * (1.0 - BM25_B + BM25_B * length_ratio)
            bm25_tf = frequency * (BM25_K1 + 1.0) / denominator
            indices.append(vocabulary[term])
            values.append(float(idf[term] * bm25_tf))
        vectors.append(models.SparseVector(indices=indices, values=values))

    artifact = {
        "algorithm": "bm25",
        "tokenizer": "unicode_word_regex_casefold_v1",
        "k1": BM25_K1,
        "b": BM25_B,
        "document_count": document_count,
        "average_document_length": average_document_length,
        "vocabulary_size": len(vocabulary),
        "vocabulary": {
            term: {"index": vocabulary[term], "idf": idf[term]}
            for term in vocabulary_terms
        },
    }
    return vectors, artifact


def query_sparse_vector(query: str, artifact: dict[str, Any]) -> models.SparseVector:
    """Encode a future BM25 query using a saved corpus artifact."""

    vocabulary = artifact["vocabulary"]
    indices: list[int] = []
    values: list[float] = []
    for term in sorted(set(tokenize(query))):
        entry = vocabulary.get(term)
        if entry is not None:
            indices.append(int(entry["index"]))
            values.append(1.0)
    return models.SparseVector(indices=indices, values=values)


def load_encoder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_passages(
    encoder: SentenceTransformer,
    chunks: list[dict[str, Any]],
    *,
    batch_size: int,
) -> np.ndarray:
    texts = [f"passage: {chunk['text']}" for chunk in chunks]
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
    return embeddings.astype(np.float32, copy=False)


def embed_query(encoder: SentenceTransformer, query: str) -> np.ndarray:
    """Encode the original user query with the E5 ``query:`` prefix."""

    if not query.strip():
        raise ValueError("Query must not be empty")
    embedding = encoder.encode(
        [f"query: {query}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    embedding_dimension = (
        encoder.get_embedding_dimension()
        if hasattr(encoder, "get_embedding_dimension")
        else encoder.get_sentence_embedding_dimension()
    )
    if embedding.ndim != 2 or embedding.shape != (1, embedding_dimension):
        raise ValueError(f"Unexpected query embedding shape: {embedding.shape}")
    return embedding[0].astype(np.float32, copy=False)


def recreate_collection(
    client: QdrantClient,
    collection_name: str,
    *,
    dense_dimension: int,
    recreate: bool,
) -> None:
    exists = client.collection_exists(collection_name)
    if exists and recreate:
        client.delete_collection(collection_name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=dense_dimension,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )
    payload_indexes = {
        "document_id": models.PayloadSchemaType.KEYWORD,
        "topic": models.PayloadSchemaType.KEYWORD,
        "source_language": models.PayloadSchemaType.KEYWORD,
        "section": models.PayloadSchemaType.KEYWORD,
        "chunk_size": models.PayloadSchemaType.INTEGER,
    }
    for field_name, field_schema in payload_indexes.items():
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception as exc:
            # An existing index is harmless on an idempotent re-run.
            if "already exists" not in str(exc).lower():
                raise


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[dict[str, Any]],
    dense_vectors: np.ndarray,
    sparse_vectors: list[models.SparseVector],
    *,
    batch_size: int,
) -> None:
    points: list[models.PointStruct] = []
    for point_id, (chunk, dense, sparse) in enumerate(
        zip(chunks, dense_vectors, sparse_vectors), start=1
    ):
        points.append(
            models.PointStruct(
                id=point_id,
                vector={
                    DENSE_VECTOR_NAME: dense.tolist(),
                    SPARSE_VECTOR_NAME: sparse,
                },
                payload=chunk,
            )
        )
        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points, wait=True)
            points = []
    if points:
        client.upsert(collection_name=collection_name, points=points, wait=True)


def verify_collection(
    client: QdrantClient,
    collection_name: str,
    expected_count: int,
    *,
    dense_dimension: int,
) -> dict[str, Any]:
    count = client.count(collection_name=collection_name, exact=True).count
    info = client.get_collection(collection_name=collection_name)
    sample, _ = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=True,
        with_vectors=True,
    )
    if count != expected_count:
        raise RuntimeError(f"{collection_name}: expected {expected_count} points, found {count}")
    if not sample:
        raise RuntimeError(f"{collection_name}: no sample point returned")
    payload = sample[0].payload or {}
    missing_payload = [key for key in ("chunk_id", "document_id", "source_url", "text") if key not in payload]
    if missing_payload:
        raise RuntimeError(f"{collection_name}: missing payload fields {missing_payload}")
    vectors = sample[0].vector or {}
    dense = vectors.get(DENSE_VECTOR_NAME)
    sparse = vectors.get(SPARSE_VECTOR_NAME)
    if dense is None or len(dense) != dense_dimension or sparse is None:
        raise RuntimeError(f"{collection_name}: dense/sparse vector verification failed")
    return {
        "collection_name": collection_name,
        "points_count": count,
        "expected_points": expected_count,
        "payload_fields_verified": ["chunk_id", "document_id", "source_url", "text"],
        "dense_dimension": dense_dimension,
        "dense_vector_name": DENSE_VECTOR_NAME,
        "sparse_vector_name": SPARSE_VECTOR_NAME,
        "status": "ok",
        "collection_status": str(info.status),
    }


def ingest_configuration(
    client: QdrantClient,
    encoder: SentenceTransformer,
    *,
    input_path: Path,
    output_root: Path,
    model_name: str,
    batch_size: int,
    recreate: bool,
) -> dict[str, Any]:
    chunks = read_chunks(input_path)
    sparse_vectors, sparse_artifact = build_bm25_vectors(chunks)
    dense_vectors = embed_passages(encoder, chunks, batch_size=batch_size)
    dense_dimension = int(dense_vectors.shape[1])
    collection_name = f"medical_chunks_{input_path.parent.name.removeprefix('chunks_')}"
    recreate_collection(
        client,
        collection_name,
        dense_dimension=dense_dimension,
        recreate=recreate,
    )
    upsert_chunks(
        client,
        collection_name,
        chunks,
        dense_vectors,
        sparse_vectors,
        batch_size=batch_size,
    )
    verification = verify_collection(
        client,
        collection_name,
        len(chunks),
        dense_dimension=dense_dimension,
    )

    artifact_dir = output_root / input_path.parent.name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "bm25_index.json").write_text(
        json.dumps(sparse_artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "input": str(input_path).replace("\\", "/"),
        "collection": collection_name,
        "chunk_count": len(chunks),
        "model": model_name,
        "dense_dimension": dense_dimension,
        "dense_prefix": "passage:",
        "sparse": {
            "algorithm": "bm25",
            "vocabulary_size": sparse_artifact["vocabulary_size"],
            "average_document_length": sparse_artifact["average_document_length"],
            "artifact": str((artifact_dir / "bm25_index.json")).replace("\\", "/"),
        },
        "verification": verification,
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--data-root", type=Path, default=project_root / "data")
    parser.add_argument("--output-root", type=Path, default=project_root / "data" / "vector_store")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", MODEL_NAME))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--recreate", action="store_true", help="Recreate collections before ingestion")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    client = QdrantClient(url=args.qdrant_url)
    encoder = load_encoder(args.model)
    reports = []
    for size in (300, 800):
        reports.append(
            ingest_configuration(
                client,
                encoder,
                input_path=args.data_root / f"chunks_{size}" / "chunks.jsonl",
                output_root=args.output_root,
                model_name=args.model,
                batch_size=args.batch_size,
                recreate=args.recreate,
            )
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "qdrant_url": args.qdrant_url,
        "model": args.model,
        "dense_query_prefix": "query:",
        "dense_passage_prefix": "passage:",
        "reports": reports,
    }
    (args.output_root / "ingestion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
