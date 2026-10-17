from typing import cast

import pytest
from psycopg import Connection
from psycopg.rows import TupleRow

from minutes.errors import NotFoundError, ValidationError
from minutes.ingest import chunk, segment
from minutes.ingest.chunk import header_for

pytestmark = pytest.mark.integration


def test_item_chunk_never_crosses_unit_or_item(
    fixture_corpus: str, conn: Connection[TupleRow]
) -> None:
    rows = conn.execute(
        "SELECT c.unit_index, c.start_offset, c.end_offset, u.text, i.start_unit, "
        "i.start_offset, i.end_unit, i.end_offset "
        "FROM chunks c JOIN items i ON i.id = c.item_id "
        "JOIN units u ON u.document_id = c.document_id AND u.unit_index = c.unit_index "
        "WHERE c.chunker = 'item' ORDER BY c.document_id, c.ordinal"
    ).fetchall()

    assert rows
    for (
        unit_index,
        start,
        end,
        text,
        item_start,
        item_start_offset,
        item_end,
        item_end_offset,
    ) in rows:
        lower = item_start_offset if unit_index == item_start else 0
        upper = item_end_offset if unit_index == item_end else len(text)
        assert item_start <= unit_index <= item_end
        assert lower <= start < end <= upper


def test_multi_page_item_has_chunks_on_both_pages_with_same_parent(
    fixture_corpus: str, conn: Connection[TupleRow]
) -> None:
    rows = conn.execute(
        "SELECT c.unit_index, c.item_id FROM chunks c JOIN items i ON i.id = c.item_id "
        "WHERE c.document_id = 'birch-minutes-201' AND c.chunker = 'item' "
        "AND i.identifier = '9.B' ORDER BY c.unit_index"
    ).fetchall()

    assert {unit_index for unit_index, _ in rows} == {1, 2}
    assert len({item_id for _, item_id in rows}) == 1


def test_chunk_text_equals_unit_slice(fixture_corpus: str, conn: Connection[TupleRow]) -> None:
    rows = conn.execute(
        "SELECT c.text, u.text, c.start_offset, c.end_offset FROM chunks c "
        "JOIN units u ON u.document_id = c.document_id AND u.unit_index = c.unit_index "
        "WHERE c.chunker IN ('fixed', 'item') ORDER BY c.document_id, c.chunker, c.ordinal"
    ).fetchall()

    assert rows
    for chunk_text, unit_text, start, end in rows:
        assert chunk_text == unit_text[start:end]


def test_embed_text_starts_with_header_for_item_chunks_only(
    fixture_corpus: str, conn: Connection[TupleRow]
) -> None:
    rows = conn.execute(
        "SELECT c.chunker, c.text, c.embed_text, c.city_id, c.body, c.meeting_date, "
        "c.doc_kind, i.identifier, i.title, city.name, u.start_ms FROM chunks c "
        "LEFT JOIN items i ON i.id = c.item_id JOIN cities city ON city.id = c.city_id "
        "JOIN units u ON u.document_id = c.document_id AND u.unit_index = c.unit_index "
        "ORDER BY c.document_id, c.chunker, c.ordinal"
    ).fetchall()

    assert rows
    for (
        chunker,
        text,
        embed_text,
        _city_id,
        body,
        meeting_date,
        kind,
        identifier,
        title,
        city,
        start_ms,
    ) in rows:
        if chunker == "item":
            assert embed_text.endswith("\n" + text)
            assert embed_text.split("\n", 1)[0] == header_for(
                city,
                body,
                meeting_date,
                kind,
                identifier or "",
                title or chunk._time_title(start_ms),
            )
        else:
            assert embed_text == text


def test_chunk_time_title_accepts_missing_time() -> None:
    assert chunk._time_title(None) == ""


def test_chunk_and_segment_reject_unknown_documents(conn: Connection[TupleRow]) -> None:
    with pytest.raises(NotFoundError, match="unknown document missing-document"):
        chunk.run(conn, "missing-document")
    with pytest.raises(NotFoundError, match="unknown document missing-document"):
        segment.run(conn, "missing-document")


def test_segment_rejects_unknown_city_item_format() -> None:
    class Cursor:
        def fetchone(self) -> tuple[str, str, str]:
            return "agenda", "meeting", "unknown"

    class FakeConnection:
        def execute(self, *_args: object) -> Cursor:
            return Cursor()

    with pytest.raises(ValidationError, match="unsupported item format unknown"):
        segment.run(cast(Connection[TupleRow], FakeConnection()), "doc")


def test_transcript_segment_is_one_item_chunk_without_parent(
    fixture_corpus: str, conn: Connection[TupleRow]
) -> None:
    rows = conn.execute(
        "SELECT unit_index, item_id FROM chunks WHERE document_id = 'birch-transcript-201' "
        "AND chunker = 'item' ORDER BY ordinal"
    ).fetchall()

    assert len(rows) == 2
    assert [unit_index for unit_index, _ in rows] == [1, 2]
    assert all(item_id is None for _, item_id in rows)


def test_both_chunkers_coexist_and_all_have_embeddings(
    fixture_corpus: str, conn: Connection[TupleRow]
) -> None:
    rows = conn.execute(
        "SELECT chunker, count(*), count(embedding) FROM chunks GROUP BY chunker ORDER BY chunker"
    ).fetchall()

    assert {chunker for chunker, _, _ in rows} == {"fixed", "item"}
    assert all(count > 0 and count == embedded for _, count, embedded in rows)


def test_no_chunk_exceeds_1200_characters(fixture_corpus: str, conn: Connection[TupleRow]) -> None:
    row = conn.execute("SELECT max(length(text)) FROM chunks").fetchone()

    assert row is not None and row[0] <= 1200
