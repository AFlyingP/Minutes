import pytest
from psycopg import Connection
from psycopg.rows import TupleRow

from minutes.ingest.segment import run

pytestmark = pytest.mark.integration


def test_fixture_business_item_counts(fixture_corpus: str, conn: Connection[TupleRow]) -> None:
    rows = conn.execute(
        "SELECT document_id, count(*) FROM items WHERE document_id = ANY(%s) "
        "AND kind = 'business' GROUP BY document_id ORDER BY document_id",
        (["alder-agenda-101", "alder-agenda-102", "birch-agenda-201", "birch-agenda-202"],),
    ).fetchall()

    assert dict(rows) == {
        "alder-agenda-101": 4,
        "alder-agenda-102": 2,
        "birch-agenda-201": 5,
        "birch-agenda-202": 2,
    }


def test_minutes_item_spans_two_pages(fixture_corpus: str, conn: Connection[TupleRow]) -> None:
    rows = conn.execute(
        "SELECT document_id, identifier, start_unit, end_unit FROM items "
        "WHERE (document_id, identifier) IN ((%s, %s), (%s, %s))",
        ("birch-minutes-201", "9.B", "alder-minutes-101", "CB 3001"),
    ).fetchall()

    assert {(document, identifier): (start, end) for document, identifier, start, end in rows} == {
        ("birch-minutes-201", "9.B"): (1, 2),
        ("alder-minutes-101", "CB 3001"): (2, 3),
    }


def test_legistar_links_by_identifier(fixture_corpus: str, conn: Connection[TupleRow]) -> None:
    rows = conn.execute(
        "SELECT i.document_id, i.identifier, linked.document_id, linked.identifier "
        "FROM items i JOIN items linked ON linked.id = i.linked_item_id "
        "WHERE i.document_id IN ('alder-agenda-101', 'alder-minutes-101') "
        "AND i.identifier = 'CB 3001'",
    ).fetchall()

    assert {
        (doc, identifier, linked_doc, linked_identifier)
        for doc, identifier, linked_doc, linked_identifier in rows
    } == {
        ("alder-agenda-101", "CB 3001", "alder-minutes-101", "CB 3001"),
        ("alder-minutes-101", "CB 3001", "alder-agenda-101", "CB 3001"),
    }


def test_granicus_links_by_title(fixture_corpus: str, conn: Connection[TupleRow]) -> None:
    rows = conn.execute(
        "SELECT i.identifier, linked.identifier FROM items i "
        "JOIN items linked ON linked.id = i.linked_item_id "
        "WHERE i.document_id = %s AND i.identifier IN ('9.B', '11.A')",
        ("birch-minutes-201",),
    ).fetchall()

    assert set(rows) == {("9.B", "9.B"), ("11.A", "11.A")}


def test_transcripts_have_no_items(fixture_corpus: str, conn: Connection[TupleRow]) -> None:
    row = conn.execute(
        "SELECT count(*) FROM items i JOIN documents d ON d.id = i.document_id "
        "WHERE d.kind = 'transcript'"
    ).fetchone()
    assert row is not None and row[0] == 0


def test_rerun_replaces_items_without_duplicates(
    fixture_corpus: str, conn: Connection[TupleRow]
) -> None:
    before_row = conn.execute(
        "SELECT count(*) FROM items WHERE document_id = %s", ("alder-agenda-101",)
    ).fetchone()
    assert before_row is not None
    before = before_row[0]

    detail = run(conn, "alder-agenda-101")
    after_row = conn.execute(
        "SELECT count(*) FROM items WHERE document_id = %s", ("alder-agenda-101",)
    ).fetchone()
    assert after_row is not None
    after = after_row[0]

    assert detail == {"items": before, "business": 4}
    assert after == before
