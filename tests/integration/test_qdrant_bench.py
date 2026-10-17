from types import SimpleNamespace
from typing import cast

import numpy as np
import psycopg
import pytest
from pgvector import Vector
from psycopg.rows import TupleRow
from qdrant_client import QdrantClient
from typer.testing import CliRunner

from minutes import cli
from minutes.evals import runner as eval_runner
from minutes.labels import store
from minutes.models.stubs import StubEmbedder
from minutes.retrieval import qdrant_bench

pytestmark = pytest.mark.integration


def test_exact_top_k_orders_by_cosine_then_id() -> None:
    ids = np.array([30, 10, 20, 40], dtype=np.int64)
    vectors = np.array([[1, 0], [1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    assert qdrant_bench.exact_top_k(vectors, ids, np.array([1, 0]), k=4) == [10, 30, 20, 40]


def test_load_vectors_returns_only_item_chunks_with_embeddings(
    fixture_corpus: str, conn: psycopg.Connection[TupleRow]
) -> None:
    ids, vectors = qdrant_bench.load_vectors(conn)
    rows = conn.execute(
        "SELECT id, embedding FROM chunks WHERE chunker = 'item' AND embedding IS NOT NULL "
        "ORDER BY id"
    ).fetchall()
    assert ids.dtype == np.int64
    assert vectors.dtype == np.float32
    assert ids.tolist() == [row[0] for row in rows]
    expected = np.asarray([cast(Vector, row[1]).to_numpy() for row in rows], dtype=np.float32)
    assert np.array_equal(vectors, expected)


def test_pgvector_search_returns_ids_of_item_chunks(
    fixture_corpus: str, conn: psycopg.Connection[TupleRow]
) -> None:
    ids, vectors = qdrant_bench.load_vectors(conn)
    found = qdrant_bench.pgvector_search(conn, vectors[0])
    assert found
    assert set(found) <= set(ids.tolist())
    assert found[0] == int(ids[0])


def test_qdrant_index_build_and_search_round_trip() -> None:
    ids = np.array([7, 3, 9], dtype=np.int64)
    vectors = np.zeros((3, 768), dtype=np.float32)
    vectors[0, 0] = 1
    vectors[1, 1] = 1
    vectors[2, 2] = 1
    client = QdrantClient(":memory:")
    try:
        index = qdrant_bench.QdrantIndex(client)
        index.build(ids, vectors)
        assert index.search(vectors[0])[0] == 7
    finally:
        client.close()


def test_benchmark_reports_both_engines_with_required_keys(
    fixture_corpus: str, conn: psycopg.Connection[TupleRow]
) -> None:
    client = QdrantClient(":memory:")
    try:
        result = qdrant_bench.run_benchmark(
            conn,
            client,
            [f"fixture question {index}" for index in range(60)],
            StubEmbedder(),
            repeats=1,
            warmup=1,
        )
    finally:
        client.close()
    assert cast(int, result["queries"]) == 60
    assert cast(int, result["vectors"]) < 100
    for engine in ("pgvector", "qdrant"):
        metrics = cast(dict[str, float], result[engine])
        assert set(metrics) == {"ann_recall_at_10", "p50_ms", "p95_ms"}
        assert 0 <= metrics["ann_recall_at_10"] <= 1
        assert metrics["p50_ms"] <= metrics["p95_ms"]
        assert metrics["ann_recall_at_10"] == 1.0


def test_benchmark_uses_same_queries_for_both_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = np.arange(12, dtype=np.int64)
    vectors = np.eye(12, 768, dtype=np.float32)
    monkeypatch.setattr(qdrant_bench, "load_vectors", lambda conn: (ids, vectors))
    calls: list[np.ndarray] = []
    qdrant_calls: list[np.ndarray] = []

    def pg_search(conn: object, query: np.ndarray, k: int = 10) -> list[int]:
        calls.append(query.copy())
        return list(range(k))

    monkeypatch.setattr(qdrant_bench, "pgvector_search", pg_search)
    original_search = qdrant_bench.QdrantIndex.search

    def qdrant_search(self: qdrant_bench.QdrantIndex, query: np.ndarray, k: int = 10) -> list[int]:
        qdrant_calls.append(query.copy())
        return original_search(self, query, k)

    monkeypatch.setattr(qdrant_bench.QdrantIndex, "search", qdrant_search)
    client = QdrantClient(":memory:")
    try:
        qdrant_bench.run_benchmark(
            cast(psycopg.Connection[TupleRow], object()),
            client,
            [f"question {index}" for index in range(60)],
            StubEmbedder(),
            repeats=1,
            warmup=0,
        )
    finally:
        client.close()
    assert len(calls) == 60
    assert len(qdrant_calls) == 60
    assert all(np.array_equal(left, right) for left, right in zip(calls, qdrant_calls, strict=True))


def test_cli_reports_unreachable_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "use_corpus", lambda corpus: "full")
    monkeypatch.setattr(store, "require_frozen", lambda labels_dir: None)
    monkeypatch.setattr(
        eval_runner,
        "load_labels",
        lambda labels_dir: [
            SimpleNamespace(type="question", payload={"question": f"question {i}"})
            for i in range(60)
        ],
    )

    def unavailable(*args: object, **kwargs: object) -> QdrantClient:
        raise ConnectionError("connection refused")

    monkeypatch.setattr(qdrant_bench, "QdrantClient", unavailable)
    result = CliRunner().invoke(cli.app, ["eval", "bench-vector"])
    assert result.exit_code == 1
    assert "error: qdrant is not reachable on 127.0.0.1:6333" in result.output
