from datetime import date
from typing import Literal

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import TupleRow

from minutes.api.app import create_app
from minutes.errors import ValidationError
from minutes.models.stubs import StubReranker
from minutes.retrieval.search import Filters, hit_heading, rrf, search, span_label

pytestmark = pytest.mark.integration
Connection = psycopg.Connection[TupleRow]
Mode = Literal["keyword", "vector", "hybrid", "hybrid_rerank"]
Chunker = Literal["fixed", "item"]
MODES: tuple[Mode, ...] = ("keyword", "vector", "hybrid", "hybrid_rerank")
QUERY = "sidewalk repair contract"


@pytest.mark.parametrize("chunker", ["fixed", "item"])
def test_keyword_query_is_or_of_terms(
    fixture_corpus: str, conn: Connection, chunker: Chunker
) -> None:
    expected = search(conn, "sidewalk", Filters(), mode="keyword", chunker=chunker, k=50)
    actual = search(conn, "sidewalk zzzz", Filters(), mode="keyword", chunker=chunker, k=50)
    assert expected
    assert [hit.chunk_id for hit in actual] == [hit.chunk_id for hit in expected]
    assert [hit.score for hit in actual] == pytest.approx([hit.score for hit in expected])


@pytest.mark.parametrize("chunker", ["fixed", "item"])
def test_stopword_only_query_returns_no_keyword_hits(
    fixture_corpus: str, conn: Connection, chunker: Chunker
) -> None:
    assert search(conn, "the and of", Filters(), mode="keyword", chunker=chunker, k=10) == []


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("chunker", ["fixed", "item"])
def test_filters_apply_in_every_mode(
    fixture_corpus: str, conn: Connection, mode: Mode, chunker: Chunker
) -> None:
    for filters in (
        Filters(city="birch"),
        Filters(body="Planning Commission"),
        Filters(kind="minutes"),
        Filters(date_from=date(2024, 3, 1), date_to=date(2024, 3, 31)),
    ):
        hits = search(conn, "meeting", filters, mode=mode, chunker=chunker, k=50)
        assert hits
        for hit in hits:
            if filters.city:
                assert hit.city_id == filters.city
            if filters.body:
                assert hit.body == filters.body
            if filters.kind:
                assert hit.doc_kind == filters.kind
            if filters.date_from:
                assert hit.meeting_date >= filters.date_from
            if filters.date_to:
                assert hit.meeting_date <= filters.date_to


def test_hybrid_contains_union_of_keyword_and_vector_top_results(
    fixture_corpus: str, conn: Connection
) -> None:
    lists = []
    for mode in ("keyword", "vector"):
        hits = search(conn, QUERY, Filters(city="birch"), mode=mode, chunker="fixed", k=50)
        lists.append([hit.chunk_id for hit in hits])
    expected = rrf(lists)
    assert len(expected) <= 50
    hits = search(conn, QUERY, Filters(city="birch"), mode="hybrid", chunker="fixed", k=50)
    assert {hit.chunk_id for hit in hits} == set(lists[0]) | set(lists[1])
    assert [hit.chunk_id for hit in hits] == [chunk_id for chunk_id, _ in expected]
    assert [hit.score for hit in hits] == pytest.approx([score for _, score in expected])


@pytest.mark.parametrize("chunker", ["fixed", "item"])
def test_rerank_orders_by_reranker_score_then_chunk_id(
    fixture_corpus: str, conn: Connection, chunker: Chunker
) -> None:
    hits = search(conn, QUERY, Filters(city="birch"), mode="hybrid_rerank", chunker=chunker, k=50)
    assert len(hits) > 1
    scores = StubReranker().score(QUERY, [hit.spans[0].text for hit in hits])
    assert [hit.score for hit in hits] == pytest.approx(scores)
    assert len(set(scores)) > 1
    assert len(set(scores)) < len(scores)
    order = [(-hit.score, hit.chunk_id) for hit in hits]
    assert order == sorted(order)


@pytest.mark.parametrize("mode", MODES)
def test_transcript_hit_is_one_segment_with_start_ms(
    fixture_corpus: str, conn: Connection, mode: Mode
) -> None:
    hits = search(
        conn, QUERY, Filters(city="birch", kind="transcript"), mode=mode, chunker="item", k=50
    )
    assert hits
    assert len({hit.chunk_id for hit in hits}) == len(hits)
    for hit in hits:
        assert hit.item_id is None
        assert hit.start_ms is not None
        assert len(hit.spans) == 1
        span = hit.spans[0]
        assert span.unit_kind == "segment"
        row = conn.execute(
            "SELECT text, start_ms FROM units WHERE document_id = %s AND unit_index = %s",
            (span.document_id, span.unit_index),
        ).fetchone()
        assert row is not None
        assert span.text == row[0]
        assert span.start_offset == 0
        assert span.end_offset == len(row[0])
        assert hit.start_ms == row[1]


def test_demo_search_first_hit(fixture_corpus: str, conn: Connection) -> None:
    hits = search(conn, QUERY, Filters(city="birch"), mode="hybrid_rerank", chunker="item", k=10)
    hit = hits[0]
    assert hit.rank == 1
    assert hit_heading("Birch", hit) == "Birch City Council, 2024-03-12, minutes, item 9.B"
    assert "Granite Works Inc." in hit.spans[0].text.replace("\n", " ")[:240]
    assert span_label(hit.spans[0].unit_kind, hit.spans[0].unit_index, hit.start_ms) == "p. 1"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("chunker", ["fixed", "item"])
def test_k_bounds_and_empty_query_rejected(
    fixture_corpus: str, conn: Connection, mode: Mode, chunker: Chunker
) -> None:
    for query, k in (("", 10), (" \n ", 10), ("x" * 501, 10), (QUERY, 0), (QUERY, 51)):
        with pytest.raises(ValidationError):
            search(conn, query, Filters(), mode=mode, chunker=chunker, k=k)
    assert len(search(conn, QUERY, Filters(), mode=mode, chunker=chunker, k=1)) == 1
    client = TestClient(create_app())
    response = client.get(
        "/api/search", params={"q": QUERY, "k": 1, "mode": mode, "chunker": chunker}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["mode"] == mode
    assert result["chunker"] == chunker
    assert len(result["hits"]) == 1
    for query, k in ((" ", 10), ("x" * 501, 10), (QUERY, 0), (QUERY, 51)):
        response = client.get(
            "/api/search", params={"q": query, "k": k, "mode": mode, "chunker": chunker}
        )
        assert response.status_code == 422
        assert response.json()["error"] == "validation_error"
