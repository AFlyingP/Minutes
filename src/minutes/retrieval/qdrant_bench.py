"""Compare pgvector and Qdrant on the same item-chunk embeddings."""

from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter, sleep
from typing import Protocol, cast

import numpy as np
import psycopg
from pgvector import Vector
from psycopg.rows import TupleRow
from qdrant_client import QdrantClient, models

from minutes.evals.metrics import percentile

__all__ = [
    "QdrantClient",
    "QdrantIndex",
    "exact_top_k",
    "load_vectors",
    "pgvector_search",
    "run_benchmark",
]

VECTOR_ROWS = """
SELECT id, embedding
FROM chunks
WHERE chunker = 'item' AND embedding IS NOT NULL
ORDER BY id
"""

PGVECTOR_QUERY = """
SELECT c.id
FROM chunks c
WHERE c.embedding IS NOT NULL AND c.chunker = 'item'
ORDER BY c.embedding <=> %(qvec)s::vector ASC, c.id ASC
LIMIT %(limit)s
"""


class Embedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...


def load_vectors(conn: psycopg.Connection[TupleRow]) -> tuple[np.ndarray, np.ndarray]:
    """Load item chunk IDs and vectors in stable ID order."""
    rows = conn.execute(VECTOR_ROWS).fetchall()
    ids = np.asarray([row[0] for row in rows], dtype=np.int64)
    vectors = np.asarray([cast(Vector, row[1]).to_numpy() for row in rows], dtype=np.float32)
    if not rows:
        vectors = np.empty((0, 768), dtype=np.float32)
    return ids, vectors


def exact_top_k(vectors: np.ndarray, ids: np.ndarray, query: np.ndarray, k: int = 10) -> list[int]:
    """Return the cosine nearest neighbors, breaking equal scores by chunk ID."""
    if k <= 0 or not len(ids):
        return []
    matrix = np.asarray(vectors, dtype=np.float32)
    vector = np.asarray(query, dtype=np.float32)
    denominators = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    scores = np.divide(
        matrix @ vector,
        denominators,
        out=np.full(len(ids), -np.inf, dtype=np.float32),
        where=denominators != 0,
    )
    order = np.lexsort((np.asarray(ids, dtype=np.int64), -scores))[:k]
    return [int(ids[index]) for index in order]


def pgvector_search(
    conn: psycopg.Connection[TupleRow], query: np.ndarray, k: int = 10
) -> list[int]:
    """Search item chunks in pgvector with the production HNSW search setting."""
    conn.execute("SET LOCAL hnsw.ef_search = 100")
    conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")
    rows = conn.execute(PGVECTOR_QUERY, {"qvec": query, "limit": k}).fetchall()
    return [int(row[0]) for row in rows]


class QdrantIndex:
    def __init__(self, client: QdrantClient, collection: str = "minutes_bench") -> None:
        self.client = client
        self.collection = collection

    def build(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=64),
        )
        for start in range(0, len(ids), 1000):
            end = min(start + 1000, len(ids))
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(id=int(ids[index]), vector=vectors[index].tolist())
                    for index in range(start, end)
                ],
                wait=True,
            )
        deadline = perf_counter() + 300
        while True:
            status = self.client.get_collection(self.collection).status
            if status == models.CollectionStatus.GREEN:
                return
            if perf_counter() >= deadline:
                raise TimeoutError(f"Qdrant collection {self.collection} did not become green")
            sleep(0.5)

    def search(self, query: np.ndarray, k: int = 10) -> list[int]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=np.asarray(query, dtype=np.float32).tolist(),
            # twice k, so that equal scores at rank k can take exact search's ID tie break
            limit=2 * k,
            search_params=models.SearchParams(hnsw_ef=100),
            with_payload=False,
        )
        ranked = sorted(response.points, key=lambda point: (-round(point.score, 6), int(point.id)))
        return [int(point.id) for point in ranked[:k]]


def _ann_recall(ann: list[int], exact: list[int]) -> float:
    return len(set(ann) & set(exact)) / 10


def _engine_metrics(
    search: Callable[[np.ndarray, int], list[int]],
    query_vectors: np.ndarray,
    vectors: np.ndarray,
    ids: np.ndarray,
    repeats: int,
    warmup: int,
) -> dict[str, float | None]:
    for query in query_vectors[:warmup]:
        search(query, 10)
    recalls: list[float] = []
    latencies: list[float] = []
    for query in query_vectors:
        exact = exact_top_k(vectors, ids, query, 10)
        for _ in range(repeats):
            started = perf_counter()
            found = search(query, 10)
            latencies.append((perf_counter() - started) * 1000)
            recalls.append(_ann_recall(found, exact))
    return {
        "ann_recall_at_10": float(np.mean(recalls)) if recalls else 0.0,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
    }


def run_benchmark(
    conn: psycopg.Connection[TupleRow],
    client: QdrantClient,
    questions: list[str],
    embedder: Embedder,
    repeats: int = 5,
    warmup: int = 10,
) -> dict[str, object]:
    """Measure both indexes using the same corpus vectors and query vectors."""
    ids, vectors = load_vectors(conn)
    if len(questions) != 60:
        raise ValueError(f"expected 60 benchmark questions, got {len(questions)}")
    query_vectors = np.stack([embedder.embed_query(question) for question in questions]).astype(
        np.float32, copy=False
    )
    index = QdrantIndex(client)
    index.build(ids, vectors)
    pgvector_metrics = _engine_metrics(
        lambda query, k: pgvector_search(conn, query, k),
        query_vectors,
        vectors,
        ids,
        repeats,
        warmup,
    )
    qdrant_metrics = _engine_metrics(index.search, query_vectors, vectors, ids, repeats, warmup)
    return {
        "vectors": len(ids),
        "queries": len(questions),
        "pgvector": pgvector_metrics,
        "qdrant": qdrant_metrics,
        "generated_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
