from datetime import date
from typing import Literal

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import TupleRow

from minutes.api.app import create_app
from minutes.errors import ValidationError
from minutes.models.stubs import StubReranker
from minutes.retrieval.search import SPAN_BUDGET, Filters, hit_heading, rrf, search, span_label

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


def test_item_hit_includes_other_page_and_linked_agenda_item(
    fixture_corpus: str, conn: Connection
) -> None:
    hits = search(conn, QUERY, Filters(city="birch"), mode="hybrid_rerank", chunker="item", k=10)
    hit = next(
        hit
        for hit in hits
        if hit.document_id == "birch-minutes-201" and hit.item_identifier == "9.B"
    )
    assert hit.spans[0].chunk_id == hit.chunk_id
    assert hit.spans[0].unit_index == 1
    assert {(span.document_id, span.unit_index) for span in hit.spans} >= {
        ("birch-minutes-201", 1),
        ("birch-minutes-201", 2),
        ("birch-agenda-201", 2),
    }
    assert any("Ayes: Gray, Hale, Ivers, Jones" in span.text for span in hit.spans)
    rows = conn.execute(
        "SELECT id FROM chunks WHERE chunker = 'item' AND item_id IN "
        "(SELECT id FROM items WHERE id = %s OR id = "
        "(SELECT linked_item_id FROM items WHERE id = %s)) "
        "ORDER BY (item_id = %s) DESC, ordinal, id",
        (hit.item_id, hit.item_id, hit.item_id),
    ).fetchall()
    assert [span.chunk_id for span in hit.spans] == [hit.chunk_id] + [
        row[0] for row in rows if row[0] != hit.chunk_id
    ]


@pytest.mark.parametrize("mode", MODES)
def test_item_hits_are_deduplicated_by_item(
    fixture_corpus: str, conn: Connection, mode: Mode
) -> None:
    hits = search(conn, "meeting sidewalk vote", Filters(), mode=mode, chunker="item", k=50)
    item_ids = [hit.item_id for hit in hits if hit.item_id is not None]
    assert item_ids
    assert len(item_ids) == len(set(item_ids))
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))


@pytest.mark.parametrize(
    ("lengths", "matched", "included"),
    [([1800] * 6, 2, [2, 0, 1, 3, 4]), ([1200, 8000, 1], 0, [0]), ([9001, 1], 0, [0])],
)
def test_span_budget_caps_at_9000_characters_and_keeps_matched_chunk(
    fixture_corpus: str,
    conn: Connection,
    lengths: list[int],
    matched: int,
    included: list[int],
) -> None:
    template = conn.execute(
        "SELECT c.item_id, c.city_id, c.body, c.meeting_date, c.doc_kind "
        "FROM chunks c JOIN items i ON i.id = c.item_id "
        "WHERE c.document_id = 'birch-minutes-201' AND c.chunker = 'item' "
        "AND i.identifier = '9.B' ORDER BY c.ordinal LIMIT 1"
    ).fetchone()
    assert template is not None
    item_id = template[0]
    conn.execute("UPDATE items SET linked_item_id = NULL WHERE id = %s", (item_id,))
    conn.execute("DELETE FROM chunks WHERE item_id = %s", (item_id,))
    texts = [
        ("budgetneedle " if index == matched else "x").ljust(length, "x")
        for index, length in enumerate(lengths)
    ]
    assert sum(len(text) for text in texts) > SPAN_BUDGET
    conn.execute(
        "UPDATE units SET text = %s WHERE document_id = 'birch-minutes-201' AND unit_index = 1",
        ("".join(texts),),
    )
    ids = []
    offset = 0
    for index, text in enumerate(texts):
        row = conn.execute(
            "INSERT INTO chunks (document_id, unit_index, item_id, chunker, ordinal, "
            "start_offset, end_offset, text, embed_text, city_id, body, meeting_date, doc_kind) "
            "VALUES ('birch-minutes-201', 1, %s, 'item', %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "RETURNING id",
            (item_id, 10000 + index, offset, offset + len(text), text, text, *template[1:]),
        ).fetchone()
        assert row is not None
        ids.append(row[0])
        offset += len(text)
    hits = search(conn, "budgetneedle", Filters(), mode="keyword", chunker="item", k=10)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.chunk_id == ids[matched]
    assert [span.chunk_id for span in hit.spans] == [ids[index] for index in included]
    assert sum(len(span.text) for span in hit.spans) == sum(lengths[index] for index in included)
    assert sum(len(span.text) for span in hit.spans) <= max(SPAN_BUDGET, lengths[matched])


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
