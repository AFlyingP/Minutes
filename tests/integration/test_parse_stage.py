import json
import random
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import TupleRow

from minutes.errors import PermanentStageError
from minutes.ingest import parse

pytestmark = pytest.mark.integration
Connection = psycopg.Connection[TupleRow]
CORPUS = Path(__file__).resolve().parents[1] / "fixtures" / "corpus"


def test_two_column_fixture_page_reads_left_then_right(
    fixture_corpus: str, conn: Connection
) -> None:
    content = json.loads((CORPUS / "content.json").read_text(encoding="utf-8"))
    for document_id in ("alder-agenda-102", "birch-agenda-202"):
        row = conn.execute(
            "SELECT text FROM units WHERE document_id = %s AND unit_index = 1", (document_id,)
        ).fetchone()
        assert row is not None
        document = next(doc for doc in content["documents"] if doc["id"] == document_id)
        page = document["pages"][0]
        assert row[0] == "\n".join(page["left"] + page["right"])
        if document_id == "alder-agenda-102":
            assert row[0].startswith("Land Use Committee Agenda\nMarch 5, 2024 at 9:30 a.m.")
            assert row[0].splitlines()[8] == "Committee members are"


def test_table_fixture_page_has_vote_rows(fixture_corpus: str, conn: Connection) -> None:
    row = conn.execute(
        "SELECT text FROM units WHERE document_id = %s AND unit_index = 2", ("birch-minutes-201",)
    ).fetchone()
    assert row is not None
    assert row[0].splitlines()[-6:] == [
        "Member | Vote",
        "Gray | Aye",
        "Hale | Aye",
        "Ivers | Aye",
        "Jones | Aye",
        "King | Aye",
    ]


def test_single_column_fixture_page_matches_content_lines(
    fixture_corpus: str, conn: Connection
) -> None:
    content = json.loads((CORPUS / "content.json").read_text(encoding="utf-8"))
    document = next(doc for doc in content["documents"] if doc["id"] == "alder-agenda-101")
    row = conn.execute(
        "SELECT text FROM units WHERE document_id = %s AND unit_index = 2", (document["id"],)
    ).fetchone()
    assert row == ("\n".join(document["pages"][1]["lines"]),)


def test_boxes_stored_for_every_pdf_page_with_text(fixture_corpus: str, conn: Connection) -> None:
    rows = conn.execute(
        "SELECT document_id, unit_index, text, boxes, needs_ocr FROM units "
        "WHERE unit_kind = 'page' ORDER BY document_id, unit_index"
    ).fetchall()
    assert len(rows) == 17
    for document_id, unit_index, text, boxes, needs_ocr in rows:
        assert needs_ocr is False
        if not text:
            assert boxes == []
            continue
        assert isinstance(boxes, list) and boxes, (document_id, unit_index)
        boxed_text = []
        for box in boxes:
            assert isinstance(box, list) and len(box) == 6
            start, end, *coordinates = box
            assert 0 <= start < end <= len(text)
            assert all(0 <= coordinate <= 1 for coordinate in coordinates)
            boxed_text.append(text[start:end])
        assert " ".join(boxed_text) == " ".join(text.replace(" | ", " ").split())


def test_unreadable_pdf_is_permanent_error(
    fixture_corpus: str, conn: Connection, tmp_path: Path
) -> None:
    file = tmp_path / "unreadable.pdf"
    file.write_bytes(random.Random(0).randbytes(256))
    document_id = "alder-unreadable-pdf"
    meeting_id = "alder-unreadable-meeting"
    with conn.transaction(force_rollback=True):
        conn.execute(
            "INSERT INTO meetings (id, city_id, body, meeting_date, title, source_key) "
            "SELECT %s, city_id, body, meeting_date, title, %s FROM meetings WHERE id = %s",
            (meeting_id, meeting_id, "alder-101"),
        )
        conn.execute(
            "INSERT INTO documents (id, meeting_id, city_id, kind, source_url, media_type, "
            "unit_kind, file_path) SELECT %s, %s, city_id, kind, source_url, "
            "media_type, unit_kind, %s FROM documents WHERE id = %s",
            (document_id, meeting_id, str(file), "alder-agenda-101"),
        )
        with pytest.raises(PermanentStageError, match=r"^unreadable pdf$"):
            parse.run(conn, document_id)
    assert conn.execute("SELECT id FROM documents WHERE id = %s", (document_id,)).fetchone() is None
    assert conn.execute("SELECT id FROM meetings WHERE id = %s", (meeting_id,)).fetchone() is None
