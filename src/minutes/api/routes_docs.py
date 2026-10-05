from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Query, Response
from psycopg.rows import dict_row

from minutes.api.routes_core import Conn
from minutes.api.schemas import (
    DocKind,
    DocumentOut,
    DocumentsOut,
    DocumentSummaryOut,
    MeetingDocumentOut,
    MeetingOut,
    MeetingSummaryOut,
    UnitOut,
)
from minutes.errors import NotFoundError, ValidationError
from minutes.ingest.parse import render_page_png
from minutes.retrieval.search import span_label

router = APIRouter(prefix="/api")


@router.get("/documents")
def documents(
    conn: Conn,
    city: str | None = None,
    kind: DocKind | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentsOut:
    if (
        city is not None
        and not conn.execute("SELECT 1 FROM cities WHERE id = %s", (city,)).fetchone()
    ):
        raise ValidationError(f"city: unknown city {city}")
    where = "d.status = 'downloaded'"
    params: list[str | int] = []
    if city is not None:
        where += " AND d.city_id = %s"
        params.append(city)
    if kind is not None:
        where += " AND d.kind = %s"
        params.append(kind)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SELECT count(*) AS total FROM documents d WHERE {where}", params)
        count = cur.fetchone()
        assert count is not None
        cur.execute(
            "SELECT d.id, d.meeting_id, d.city_id, d.kind, m.body, m.meeting_date, "
            "d.unit_kind, d.unit_count FROM documents d JOIN meetings m ON m.id = d.meeting_id "
            f'WHERE {where} ORDER BY m.meeting_date ASC, d.id COLLATE "C" ASC LIMIT %s OFFSET %s',
            [*params, limit, offset],
        )
        return DocumentsOut(
            total=count["total"],
            documents=[DocumentSummaryOut(**row) for row in cur.fetchall()],
        )


@router.get("/documents/{id}")
def document(conn: Conn, id: str) -> DocumentOut:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, city_id, kind, unit_kind, unit_count, source_url, meeting_id "
            "FROM documents WHERE id = %s AND status = 'downloaded'",
            (id,),
        )
        row = cur.fetchone()
        if row is None:
            raise NotFoundError(f"unknown document {id}")
        cur.execute(
            "SELECT id, body, meeting_date, title, recording_url FROM meetings WHERE id = %s",
            (row.pop("meeting_id"),),
        )
        meeting = cur.fetchone()
        assert meeting is not None
        return DocumentOut(**row, meeting=MeetingSummaryOut(**meeting))


@router.get("/documents/{id}/units/{index}")
def unit(conn: Conn, id: str, index: int) -> UnitOut:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT u.document_id, u.unit_index, u.unit_kind, u.text, u.boxes, u.text_source, "
            "u.start_ms, u.end_ms, u.speaker, u.width_pt, u.height_pt, "
            "m.recording_url, m.recording_seek FROM units u "
            "JOIN documents d ON d.id = u.document_id JOIN meetings m ON m.id = d.meeting_id "
            "WHERE d.id = %s AND d.status = 'downloaded' AND u.unit_index = %s",
            (id, index),
        )
        row = cur.fetchone()
    if row is None:
        raise NotFoundError(f"unknown unit {id}/{index}")
    recording_url = row.pop("recording_url")
    recording_seek = row.pop("recording_seek")
    recording_link = None
    if row["unit_kind"] == "segment":
        recording_link = recording_url
        if recording_url is not None and recording_seek == "granicus_entrytime":
            recording_link = f"{recording_url}&entrytime={row['start_ms'] // 1000}"
    return UnitOut(
        **row,
        label=span_label(row["unit_kind"], index, row["start_ms"]),
        recording_link=recording_link,
    )


@router.get("/documents/{id}/units/{index}/image")
def page_image(
    conn: Conn, id: str, index: int, dpi: Annotated[int, Query(ge=72, le=200)] = 110
) -> Response:
    row = conn.execute(
        "SELECT u.unit_kind, d.file_path FROM units u JOIN documents d ON d.id = u.document_id "
        "WHERE d.id = %s AND d.status = 'downloaded' AND u.unit_index = %s",
        (id, index),
    ).fetchone()
    if row is None or row[0] != "page":
        raise NotFoundError(f"unknown page {id}/{index}")
    file_path = row[1]
    if file_path is None or not Path(file_path).is_file():
        raise NotFoundError(f"missing source file for {id}")
    return Response(
        render_page_png(file_path, index - 1, dpi),
        media_type="image/png",
        headers={"Cache-Control": "max-age=3600"},
    )


@router.get("/meetings/{id}")
def meeting(conn: Conn, id: str) -> MeetingOut:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, city_id, body, meeting_date, title, recording_url "
            "FROM meetings WHERE id = %s",
            (id,),
        )
        row = cur.fetchone()
        if row is None:
            raise NotFoundError(f"unknown meeting {id}")
        cur.execute(
            "SELECT id, kind, status, unit_count FROM documents WHERE meeting_id = %s "
            'ORDER BY id COLLATE "C" ASC',
            (id,),
        )
        return MeetingOut(**row, documents=[MeetingDocumentOut(**doc) for doc in cur.fetchall()])
