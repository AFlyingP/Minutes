import psycopg
import pytest
from psycopg.rows import TupleRow

from minutes.ingest import parse

pytestmark = pytest.mark.integration
Connection = psycopg.Connection[TupleRow]


def test_fixture_transcripts_have_expected_segments(fixture_corpus: str, conn: Connection) -> None:
    expected = {
        "alder-transcript-101": 2,
        "alder-transcript-102": 1,
        "birch-transcript-201": 2,
        "birch-transcript-202": 1,
    }
    rows = conn.execute(
        "SELECT document_id, count(*) FROM units WHERE unit_kind = 'segment' "
        "GROUP BY document_id ORDER BY document_id"
    ).fetchall()
    assert dict(rows) == expected
    assert sum(count for _, count in rows) == 6
    assert conn.execute("SELECT count(*) FROM units WHERE unit_kind = 'page'").fetchone() == (17,)
    for document_id, count in expected.items():
        assert conn.execute(
            "SELECT unit_count FROM documents WHERE id = %s", (document_id,)
        ).fetchone() == (count,)
    with conn.transaction(force_rollback=True):
        for document_id, count in expected.items():
            assert parse.run(conn, document_id) == {"units": count, "needs_ocr": 0}
        assert (
            conn.execute(
                "SELECT document_id, count(*) FROM units WHERE unit_kind = 'segment' "
                "GROUP BY document_id ORDER BY document_id"
            ).fetchall()
            == rows
        )


def test_segment_units_have_times_and_no_boxes(fixture_corpus: str, conn: Connection) -> None:
    row = conn.execute(
        "SELECT start_ms, end_ms, speaker, boxes FROM units "
        "WHERE document_id = %s AND unit_index = 1",
        ("birch-transcript-201",),
    ).fetchone()
    assert row == (0, 24000, "SPEAKER 1 (COUNCIL)", None)
    rows = conn.execute(
        "SELECT start_ms, end_ms, boxes, text_source, needs_ocr FROM units "
        "WHERE unit_kind = 'segment' ORDER BY document_id, unit_index"
    ).fetchall()
    assert len(rows) == 6
    for start_ms, end_ms, boxes, source, needs_ocr in rows:
        assert 0 <= start_ms < end_ms
        assert end_ms - start_ms <= 60000
        assert boxes is None
        assert source == "caption"
        assert needs_ocr is False


def test_segment_text_never_exceeds_1200_characters(fixture_corpus: str, conn: Connection) -> None:
    rows = conn.execute(
        "SELECT text FROM units WHERE unit_kind = 'segment' ORDER BY document_id, unit_index"
    ).fetchall()
    assert len(rows) == 6
    assert all(0 < len(text) <= 1200 for (text,) in rows)
