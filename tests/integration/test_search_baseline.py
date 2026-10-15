from itertools import pairwise

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import TupleRow

from minutes.api.app import create_app
from minutes.retrieval.search import Filters, search

pytestmark = pytest.mark.integration

Connection = psycopg.Connection[TupleRow]


def test_vector_search_returns_k_hits_ordered_by_score(
    fixture_corpus: str, conn: Connection
) -> None:
    hits = search(conn, "sidewalk repair contract", Filters(), mode="vector", chunker="fixed", k=5)
    assert len(hits) == 5
    assert [hit.rank for hit in hits] == [1, 2, 3, 4, 5]
    assert all(left.score >= right.score for left, right in pairwise(hits))


def test_vector_search_respects_city_filter(fixture_corpus: str, conn: Connection) -> None:
    hits = search(
        conn,
        "sidewalk repair contract",
        Filters(city="birch"),
        mode="vector",
        chunker="fixed",
        k=5,
    )
    assert hits
    assert all(hit.city_id == "birch" for hit in hits)


def test_vector_and_keyword_modes_available_over_http(fixture_corpus: str) -> None:
    client = TestClient(create_app())
    vector = client.get(
        "/api/search",
        params={"q": "sidewalk repair contract", "mode": "vector", "chunker": "fixed"},
    )
    hybrid = client.get(
        "/api/search",
        params={"q": "sidewalk repair contract", "mode": "hybrid", "chunker": "fixed"},
    )
    keyword = client.get(
        "/api/search",
        params={"q": "sidewalk repair contract", "mode": "keyword", "chunker": "fixed"},
    )
    assert vector.status_code == 200
    assert vector.json()["hits"]
    assert keyword.status_code == 200
    assert keyword.json()["hits"]
    assert hybrid.status_code == 422


def test_vector_search_finds_sidewalk_chunk(fixture_corpus: str, conn: Connection) -> None:
    hits = search(
        conn,
        "sidewalk repair contract",
        Filters(city="birch"),
        mode="vector",
        chunker="fixed",
        k=3,
    )
    assert any(hit.document_id == "birch-minutes-201" for hit in hits)
