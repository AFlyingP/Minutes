import psycopg
import pytest
from psycopg.rows import TupleRow

from minutes.ingest import ocr, parse, pipeline
from minutes.ingest.quality import parse_quality_report

pytestmark = pytest.mark.integration
Connection = psycopg.Connection[TupleRow]


def test_scanned_fixture_page_is_flagged_then_read_by_stub(
    fixture_corpus: str, conn: Connection
) -> None:
    document_id = "birch-minutes-201"
    with conn.transaction(force_rollback=True):
        assert parse.run(conn, document_id) == {"units": 3, "needs_ocr": 1}
        row = conn.execute(
            "SELECT needs_ocr, text FROM units WHERE document_id = %s AND unit_index = 3",
            (document_id,),
        ).fetchone()
        assert row == (True, "")
        assert ocr.run(conn, document_id) == {"ocr_pages": 1}
        row = conn.execute(
            "SELECT text, text_source, ocr_confidence, needs_ocr, boxes FROM units "
            "WHERE document_id = %s AND unit_index = 3",
            (document_id,),
        ).fetchone()
        assert row is not None
        text, source, confidence, needs_ocr, boxes = row
        assert "11.A Introduce ORDINANCE 1042" in text
        assert source == "ocr"
        assert confidence == pytest.approx(0.99, abs=0.001)
        assert needs_ocr is False
        assert boxes
        for start, end, *coordinates in boxes:
            assert 0 <= start < end <= len(text)
            assert all(0 <= coordinate <= 1 for coordinate in coordinates)
        assert ocr.run(conn, document_id) == {"ocr_pages": 0}


def test_ocr_stage_touches_only_flagged_units(fixture_corpus: str, conn: Connection) -> None:
    with conn.transaction(force_rollback=True):
        documents = pipeline.documents_for(conn, "fixture", None)
        for document_id in documents:
            parse.run(conn, document_id)
        before = conn.execute(
            "SELECT id, text, boxes, text_source, ocr_confidence, needs_ocr FROM units "
            "WHERE unit_kind = 'page' AND NOT needs_ocr ORDER BY id"
        ).fetchall()
        assert len(before) == 16
        assert all(row[3] == "pdf" for row in before)
        counts = [ocr.run(conn, document_id)["ocr_pages"] for document_id in documents]
        assert counts.count(1) == 1
        assert counts.count(0) == len(documents) - 1
        after = conn.execute(
            "SELECT id, text, boxes, text_source, ocr_confidence, needs_ocr FROM units "
            "WHERE id = ANY(%s) ORDER BY id",
            ([row[0] for row in before],),
        ).fetchall()
        assert after == before


def test_parse_quality_report_has_one_row_per_city(fixture_corpus: str, conn: Connection) -> None:
    rows = parse_quality_report(conn)
    assert [row.city_id for row in rows] == ["alder", "birch"]
    alder, birch = rows
    assert alder.ocr_pages == 0
    assert alder.mean_ocr_confidence is None
    assert birch.ocr_pages == 1
    assert birch.mean_ocr_confidence == pytest.approx(0.99, abs=0.001)
    assert alder.pages + birch.pages == 17
    assert alder.empty_pages == birch.empty_pages == 0
    assert alder.parse_quality is not None and alder.parse_quality >= 0.90
    assert birch.parse_quality is not None and birch.parse_quality >= 0.90
